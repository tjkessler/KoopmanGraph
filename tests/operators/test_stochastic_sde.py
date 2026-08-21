"""Tests for the opt-in drift–diffusion stepper."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from koopman_graph.model import GraphKoopmanModel
from koopman_graph.nn import GNNDecoder, GNNEncoder
from koopman_graph.operators import DriftDiffusionKoopman
from koopman_graph.operators.stochastic_sde import DEFAULT_DIFFUSION_RANK


def test_euler_maruyama_mean_matches_affine_drift() -> None:
    """EM conditional mean is z + Δt z Lᵀ."""
    operator = DriftDiffusionKoopman(2, step_kind="euler_maruyama")
    drift = torch.tensor([[-0.4, 0.1], [0.0, -0.5]])
    operator.set_drift(drift)
    z = torch.tensor([[1.0, -2.0], [0.5, 0.25]])
    delta_t = 0.2
    expected = z + delta_t * (z @ drift.transpose(-1, -2))
    torch.testing.assert_close(operator.mean_advance(z, delta_t), expected)
    torch.testing.assert_close(operator(z, delta_t), expected)


def test_yosida_mean_matches_implicit_resolvent() -> None:
    """Yosida mean solves (I − Δt L) z_nextᵀ = zᵀ."""
    operator = DriftDiffusionKoopman(2, step_kind="yosida")
    drift = -torch.eye(2)
    operator.set_drift(drift)
    z = torch.ones(3, 2)
    delta_t = 0.1
    left = torch.eye(2) - delta_t * drift
    expected = torch.linalg.solve(left, z.transpose(-1, -2)).transpose(-1, -2)
    torch.testing.assert_close(
        operator.mean_advance(z, delta_t),
        expected,
        atol=1e-6,
        rtol=1e-6,
    )


def test_sampled_paths_match_mean_and_diffusion_scale() -> None:
    """Seeded EM samples recover the mean and Δt FFᵀ covariance."""
    operator = DriftDiffusionKoopman(2, diffusion_rank=2)
    operator.set_drift(-torch.eye(2))
    factor = torch.diag(torch.tensor([0.4, 0.2]))
    operator.set_diffusion(factor)
    z = torch.zeros(1, 2)
    delta_t = 0.25
    mean = operator.mean_advance(z, delta_t)
    draws: list[torch.Tensor] = []
    for seed in range(4000):
        generator = torch.Generator().manual_seed(seed)
        draws.append(operator.advance(z, delta_t, generator=generator))
    samples = torch.stack(draws, dim=0)
    torch.testing.assert_close(samples.mean(dim=0), mean, atol=0.03, rtol=0.0)
    expected_cov = delta_t * (factor @ factor.T)
    centered = samples[:, 0, :] - mean[0]
    empirical = centered.T @ centered / float(centered.shape[0] - 1)
    torch.testing.assert_close(empirical, expected_cov, atol=0.03, rtol=0.15)


def test_default_stochastic_factory_is_still_diagonal_noise() -> None:
    """Factory dynamics_mode='stochastic' does not build the SDE stepper."""
    model = GraphKoopmanModel(
        GNNEncoder(2, 4, 2, num_layers=1),
        GNNDecoder(2, 4, 2, num_layers=1),
        latent_dim=2,
        time_step=0.1,
        dynamics_mode="stochastic",
    )
    assert not isinstance(model.koopman, DriftDiffusionKoopman)
    assert getattr(model.koopman, "stochastic", False)
    assert hasattr(model.koopman, "process_log_std")


def test_drift_diffusion_guards() -> None:
    """Constructor and setters refuse invalid ranks, kinds, and controls."""
    with pytest.raises(ValueError, match="latent_dim"):
        DriftDiffusionKoopman(0)
    with pytest.raises(ValueError, match="diffusion_rank"):
        DriftDiffusionKoopman(2, diffusion_rank=3)
    with pytest.raises(ValueError, match="step_kind"):
        DriftDiffusionKoopman(2, step_kind="milstein")  # type: ignore[arg-type]
    operator = DriftDiffusionKoopman(2, diffusion_rank=DEFAULT_DIFFUSION_RANK)
    with pytest.raises(ValueError, match="drift"):
        operator.set_drift(torch.zeros(2, 3))
    with pytest.raises(ValueError, match="diffusion"):
        operator.set_diffusion(torch.zeros(2, 2))
    z = torch.ones(2, 2)
    with pytest.raises(ValueError, match="delta_t"):
        operator.advance(z)
    with pytest.raises(ValueError, match="uncontrolled"):
        operator.advance(z, 0.1, control=torch.ones(1))
    with pytest.raises(ValueError, match="trailing"):
        operator.mean_advance(torch.ones(2, 3), 0.1)
    torch.testing.assert_close(operator.mean_advance(z, 0.0), z)


def test_stochastic_sde_module_does_not_import_model_data_or_uq() -> None:
    """L2 stepper must not import L4 model, L1 data, or L5 uq."""
    source = Path(__file__).resolve().parents[2] / (
        "src/koopman_graph/operators/stochastic_sde.py"
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
        "koopman_graph.uq",
    )
    offenders = [
        name
        for name in imported
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in forbidden)
    ]
    assert not offenders
