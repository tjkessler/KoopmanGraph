"""Decoded-space mass, positivity, and linear-conservation heads."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch
from torch import nn

import koopman_graph
from koopman_graph.nn import (
    LinearConservingDecoder,
    MassConservingDecoder,
    PositivityDecoder,
)
from koopman_graph.nn.constraint_decoders import DEFAULT_CONSERVATION_ATOL
from koopman_graph.operators import KoopmanOperator

_ATOL = DEFAULT_CONSERVATION_ATOL
_N_NODES = 6
_LATENT_DIM = 4
_STEPS = 8
_MASS = 1.25
_SEED = 0


class _FirstChannelDecoder(nn.Module):
    """Map latents to a one-channel table (decoded-space teaching stub)."""

    def forward(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del edge_index, edge_weight
        return z[:, :1]


class _TwoChannelDecoder(nn.Module):
    """Copy the first two latent columns as decoded features."""

    def forward(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del edge_index, edge_weight
        return z[:, :2]


def _path_edges(num_nodes: int) -> torch.Tensor:
    """Oriented path used only to satisfy the decoder signature."""
    tails = torch.arange(num_nodes - 1, dtype=torch.long)
    return torch.stack((tails, tails + 1), dim=0)


def test_constraint_decoders_export_off_root() -> None:
    """Constraint heads live on ``nn.__all__``, not the root façade."""
    assert "MassConservingDecoder" in koopman_graph.nn.__all__
    assert "PositivityDecoder" in koopman_graph.nn.__all__
    assert "LinearConservingDecoder" in koopman_graph.nn.__all__
    assert "MassConservingDecoder" not in koopman_graph.__all__
    assert "PositivityDecoder" not in koopman_graph.__all__
    assert "LinearConservingDecoder" not in koopman_graph.__all__


def test_constraint_decoders_module_does_not_import_model() -> None:
    """L3 decoder heads must not import L4 model."""
    source = Path(__file__).resolve().parents[2] / (
        "src/koopman_graph/nn/constraint_decoders.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)
    offenders = [
        name
        for name in imported
        if name == "koopman_graph.model" or name.startswith("koopman_graph.model.")
    ]
    assert not offenders


def test_affine_mass_bound_over_rollout() -> None:
    """``||1^T x_t - M||_inf`` stays below ``1e-6`` for ``T`` decoded steps."""
    torch.manual_seed(_SEED)
    edge_index = _path_edges(_N_NODES)
    operator = KoopmanOperator(_LATENT_DIM).double()
    decoder = MassConservingDecoder(
        _FirstChannelDecoder(),
        channels=(0,),
        mass=_MASS,
        method="affine",
    ).double()
    latents = torch.randn(_N_NODES, _LATENT_DIM, dtype=torch.float64)
    residuals: list[float] = []
    for _ in range(_STEPS):
        decoded = decoder(latents, edge_index)
        residual = abs(float(decoded[:, 0].sum().item()) - _MASS)
        residuals.append(residual)
        latents = operator.advance(latents)
    assert max(residuals) < _ATOL
    assert decoded.shape == (_N_NODES, 1)


def test_linear_conservation_bound_over_rollout() -> None:
    """``||C x_t - c_0||_inf`` is bounded on a two-partition toy."""
    torch.manual_seed(_SEED)
    constraint = torch.tensor(
        [[1.0, 1.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]],
        dtype=torch.float64,
    )
    target = torch.tensor([0.5, 0.75], dtype=torch.float64)
    edge_index = _path_edges(_N_NODES)
    operator = KoopmanOperator(_LATENT_DIM).double()
    decoder = LinearConservingDecoder(
        _FirstChannelDecoder(),
        channels=(0,),
        constraint=constraint,
        target=target,
    ).double()
    latents = torch.randn(_N_NODES, _LATENT_DIM, dtype=torch.float64)
    for _ in range(_STEPS):
        decoded = decoder(latents, edge_index)
        residual = constraint @ decoded[:, 0] - target
        assert float(residual.abs().max().item()) < _ATOL
        latents = operator.advance(latents)


def test_softmax_mass_is_positive_simplex() -> None:
    """Softmax mass is non-negative and sums to ``M``."""
    decoder = MassConservingDecoder(
        _FirstChannelDecoder(),
        channels=(0,),
        mass=_MASS,
        method="softmax",
    )
    decoded = decoder(torch.randn(_N_NODES, _LATENT_DIM), _path_edges(_N_NODES))
    assert float(decoded[:, 0].sum().item()) == pytest.approx(_MASS, abs=_ATOL)
    assert bool((decoded[:, 0] > 0).all().item())


def test_positivity_maps_named_channel() -> None:
    """``exp`` forces a strictly positive named channel."""
    decoder = PositivityDecoder(
        _TwoChannelDecoder(),
        channels=(0,),
        method="exp",
    )
    raw = torch.tensor(
        [[-2.0, 0.3], [0.5, -1.0], [1.0, 0.0], [0.0, 2.0], [-0.1, 0.2], [3.0, -4.0]]
    )
    decoded = decoder.project(raw)
    assert bool((decoded[:, 0] > 0).all().item())
    assert torch.allclose(decoded[:, 1], raw[:, 1])


def test_symplectic_k_alone_does_not_conserve_decoded_mass() -> None:
    """A symplectic latent map does not keep ``1^T x`` without a head.

    ``Greydanus2019HNN``: structure on the latent vector field is not a
    decoded-space conservation certificate. Teaching floor is
    ``DEFAULT_CONSERVATION_ATOL`` (``1e-6``).
    """
    torch.manual_seed(_SEED)
    edge_index = _path_edges(_N_NODES)
    operator = KoopmanOperator(
        _LATENT_DIM,
        parameterization="symplectic",
    ).double()
    bare = _FirstChannelDecoder()
    headed = MassConservingDecoder(
        _FirstChannelDecoder(),
        channels=(0,),
        mass=_MASS,
        method="affine",
    ).double()
    latents = torch.randn(_N_NODES, _LATENT_DIM, dtype=torch.float64)
    bare_masses: list[float] = []
    headed_masses: list[float] = []
    state_bare = latents.clone()
    state_headed = latents.clone()
    for _ in range(_STEPS):
        bare_masses.append(float(bare(state_bare, edge_index)[:, 0].sum().item()))
        headed_masses.append(float(headed(state_headed, edge_index)[:, 0].sum().item()))
        state_bare = operator.advance(state_bare)
        state_headed = operator.advance(state_headed)
    assert max(abs(mass - bare_masses[0]) for mass in bare_masses) > _ATOL
    assert max(abs(mass - _MASS) for mass in headed_masses) < _ATOL


def test_constraint_heads_reject_invalid_inputs() -> None:
    """Channel, mass, and node-count guards raise ``ValueError``."""
    inner = _FirstChannelDecoder()
    with pytest.raises(ValueError, match="softmax mass"):
        MassConservingDecoder(inner, channels=(0,), mass=0.0, method="softmax")
    with pytest.raises(ValueError, match="method"):
        MassConservingDecoder(inner, channels=(0,), mass=1.0, method="clip")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="method"):
        PositivityDecoder(inner, channels=(0,), method="relu")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_eq"):
        LinearConservingDecoder(
            inner,
            channels=(0,),
            constraint=torch.zeros(0, _N_NODES),
            target=torch.zeros(0),
        )
    headed = MassConservingDecoder(inner, channels=(5,), mass=1.0)
    with pytest.raises(ValueError, match="outside"):
        headed(torch.randn(_N_NODES, _LATENT_DIM), _path_edges(_N_NODES))
    decoder = LinearConservingDecoder(
        inner,
        channels=(0,),
        constraint=torch.ones(1, _N_NODES),
        target=torch.ones(1),
    )
    with pytest.raises(ValueError, match="node count"):
        decoder.project(torch.randn(_N_NODES - 1, 1))
