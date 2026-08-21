"""Cochain state, k<=1 operator, and boundary nilpotency tests."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

import koopman_graph
from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
from koopman_graph.model.factory import parse_koopman_arg
from koopman_graph.nn.cell_complex import boundary_incidence_b2
from koopman_graph.observables import boundary_incidence_b1
from koopman_graph.operators import (
    CochainKoopmanOperator,
    CochainState,
    HodgeKoopmanOperator,
    KoopmanOperator,
    boundary_nilpotency,
)
from koopman_graph.operators.cochain import DEFAULT_NILPOTENCY_ATOL

# float32 dense B1 B2 on the teaching triangle; same floor as cell-complex tests.
_ATOL = DEFAULT_NILPOTENCY_ATOL
_ORACLE_REL = 1e-6
_ORACLE_ABS = 1e-8
_LATENT_DIM = 2


def _filled_triangle_edges() -> tuple[torch.Tensor, torch.Tensor]:
    """Oriented 1-skeleton and one face of the filled triangle."""
    edge_index = torch.tensor([[0, 1, 0], [1, 2, 2]], dtype=torch.long)
    face_index = torch.tensor([[0], [1], [2]], dtype=torch.long)
    return edge_index, face_index


def test_package_exports_cochain_off_root() -> None:
    """Cochain helpers live on ``operators.__all__``, not the root façade."""
    assert "CochainState" in koopman_graph.operators.__all__
    assert "CochainKoopmanOperator" in koopman_graph.operators.__all__
    assert "boundary_nilpotency" in koopman_graph.operators.__all__
    assert "CochainState" not in koopman_graph.__all__
    assert "CochainKoopmanOperator" not in koopman_graph.__all__


def test_cochain_module_does_not_import_model_data_or_nn() -> None:
    """L2 operator must not import L4 model, L1 data, or L3 nn."""
    source = Path(__file__).resolve().parents[2] / (
        "src/koopman_graph/operators/cochain.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)
    forbidden = (
        "koopman_graph.model",
        "koopman_graph.data",
        "koopman_graph.nn",
    )
    offenders = [
        name
        for name in imported
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in forbidden)
    ]
    assert not offenders


def test_boundary_nilpotency_on_filled_triangle() -> None:
    """``B_1 B_2 = 0`` on the oriented teaching triangle to ``1e-6``."""
    edge_index, face_index = _filled_triangle_edges()
    b1 = boundary_incidence_b1(edge_index, num_nodes=3)
    b2 = boundary_incidence_b2(edge_index, face_index, num_nodes=3)
    report = boundary_nilpotency(b1, b2, atol=_ATOL)
    assert report.product.shape == (3, 1)
    assert report.nilpotent is True
    assert report.max_abs <= _ATOL
    torch.testing.assert_close(
        report.product,
        torch.zeros_like(report.product),
        rtol=0.0,
        atol=_ATOL,
    )


def test_default_factory_stays_node_centered() -> None:
    """``koopman=None`` remains per-node LTI; ``cochain`` is not a kind."""
    kind, injected = parse_koopman_arg(None)
    assert kind == "pernode"
    assert injected is None
    with pytest.raises(ValueError, match="cochain"):
        parse_koopman_arg("cochain")
    model = GraphKoopmanModel(
        encoder=GNNEncoder(2, 4, 4, num_layers=1),
        decoder=GNNDecoder(4, 4, 2, num_layers=1),
        latent_dim=4,
        time_step=0.1,
    )
    assert model.koopman_kind == "pernode"
    assert isinstance(model.koopman, KoopmanOperator)
    assert not isinstance(model.koopman, CochainKoopmanOperator)
    assert not isinstance(model.koopman, HodgeKoopmanOperator)


def test_commuting_residual_vanishes_when_maps_match() -> None:
    """Feature-axis intertwining is exact when ``K_0 = K_1``."""
    edge_index, _ = _filled_triangle_edges()
    operator = CochainKoopmanOperator(
        _LATENT_DIM,
        edge_index,
        num_nodes=3,
    ).double()
    shared = torch.diag(torch.tensor([0.4, -0.2], dtype=torch.float64))
    with torch.no_grad():
        operator.k_node.copy_(shared)
        operator.k_edge.copy_(shared)
    state = CochainState(
        node=torch.tensor(
            [[1.0, 0.0], [0.5, -0.25], [0.0, 0.75]],
            dtype=torch.float64,
        ),
        edge=torch.zeros(3, _LATENT_DIM, dtype=torch.float64),
    )
    residual = operator.commuting_residual(state)
    torch.testing.assert_close(
        residual,
        torch.zeros_like(residual),
        rtol=_ORACLE_REL,
        atol=_ORACLE_ABS,
    )
    assert float(operator.commuting_loss(state).item()) == pytest.approx(
        0.0, abs=_ORACLE_ABS
    )


def test_commuting_residual_flags_mismatched_maps() -> None:
    """Distinct feature maps leave a nonzero coboundary residual."""
    edge_index, _ = _filled_triangle_edges()
    operator = CochainKoopmanOperator(_LATENT_DIM, edge_index, num_nodes=3)
    with torch.no_grad():
        operator.k_node.copy_(0.2 * torch.eye(_LATENT_DIM))
        operator.k_edge.copy_(0.8 * torch.eye(_LATENT_DIM))
    state = CochainState(
        node=torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.5]]),
        edge=torch.zeros(3, _LATENT_DIM),
    )
    residual = operator.commuting_residual(state)
    assert float(residual.abs().max().item()) > 0.1
    assert float(operator.commuting_loss(state).item()) > 0.01


def test_cross_degree_matches_independent_incidence_step() -> None:
    """Cross terms are ``B_1 z_1`` / ``B_1ᵀ z_0`` times the C maps."""
    edge_index, _ = _filled_triangle_edges()
    operator = CochainKoopmanOperator(
        _LATENT_DIM,
        edge_index,
        num_nodes=3,
        use_cross_degree=True,
    ).double()
    k0 = torch.diag(torch.tensor([0.5, 0.25], dtype=torch.float64))
    k1 = torch.diag(torch.tensor([0.3, -0.1], dtype=torch.float64))
    c01 = torch.diag(torch.tensor([0.2, 0.4], dtype=torch.float64))
    c10 = torch.diag(torch.tensor([-0.15, 0.05], dtype=torch.float64))
    with torch.no_grad():
        operator.k_node.copy_(k0)
        operator.k_edge.copy_(k1)
        operator.c_node_from_edge.copy_(c01)
        operator.c_edge_from_node.copy_(c10)
    state = CochainState(
        node=torch.arange(6, dtype=torch.float64).reshape(3, 2) / 10.0,
        edge=torch.arange(6, 12, dtype=torch.float64).reshape(3, 2) / 10.0,
    )
    incidence = operator.incidence.to(dtype=torch.float64)
    expected_node = state.node @ k0.T + (incidence @ state.edge) @ c01.T
    expected_edge = state.edge @ k1.T + (incidence.T @ state.node) @ c10.T
    got = operator.advance(state)
    torch.testing.assert_close(
        got.node, expected_node, rtol=_ORACLE_REL, atol=_ORACLE_ABS
    )
    torch.testing.assert_close(
        got.edge, expected_edge, rtol=_ORACLE_REL, atol=_ORACLE_ABS
    )
    assert got.face is None


def test_face_latents_are_copied_not_evolved() -> None:
    """Optional ``k=2`` tables pass through unchanged."""
    edge_index, _ = _filled_triangle_edges()
    operator = CochainKoopmanOperator(_LATENT_DIM, edge_index, num_nodes=3)
    face = torch.tensor([[0.3, -0.2]])
    state = CochainState(
        node=torch.ones(3, _LATENT_DIM),
        edge=torch.zeros(3, _LATENT_DIM),
        face=face,
    )
    got = operator.advance(state)
    torch.testing.assert_close(got.face, face)
    assert got.node.shape == (3, _LATENT_DIM)


def test_cochain_rejects_invalid_inputs() -> None:
    """Boundary validation names the broken constraint."""
    edge_index, _ = _filled_triangle_edges()
    with pytest.raises(ValueError, match="latent width"):
        CochainState(node=torch.ones(3, 2), edge=torch.ones(3, 3))
    with pytest.raises(ValueError, match="finite"):
        CochainState(node=torch.tensor([[float("nan"), 0.0]]), edge=torch.zeros(1, 2))
    with pytest.raises(ValueError, match="num_nodes"):
        CochainKoopmanOperator(2, edge_index, num_nodes=0)
    operator = CochainKoopmanOperator(2, edge_index, num_nodes=3)
    with pytest.raises(ValueError, match="num_nodes"):
        operator.advance(CochainState(node=torch.ones(2, 2), edge=torch.ones(3, 2)))
    with pytest.raises(ValueError, match="b1 columns"):
        boundary_nilpotency(torch.ones(3, 2), torch.ones(3, 1))
