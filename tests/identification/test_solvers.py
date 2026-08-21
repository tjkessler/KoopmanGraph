"""Closed-form identification solvers (ridge, TLS, constrained LS)."""

from __future__ import annotations

import pytest
import torch

from koopman_graph.baselines.base import fit_row_operator, fit_tls_row_operator
from koopman_graph.identification import (
    ClosedFormBackend,
    IdentificationBackend,
    IdentificationConfig,
    LatentPairs,
    OperatorSnapshot,
    identify_operator,
)
from koopman_graph.identification.solvers import apply_operator_snapshot
from koopman_graph.operators import KoopmanOperator


def _eigvals_match(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> bool:
    """Greedy multiset match of complex eigenvalues.

    Parameters
    ----------
    left, right : Tensor
        Eigenvalue vectors.
    rtol, atol : float
        Relative and absolute tolerances from construction.

    Returns
    -------
    bool
        Whether every left value matches an unused right value.
    """
    if left.shape != right.shape:
        return False
    remaining = right.detach().clone()
    for value in left.detach():
        diffs = (remaining - value).abs()
        index = int(torch.argmin(diffs))
        if not torch.isclose(value, remaining[index], rtol=rtol, atol=atol):
            return False
        remaining[index] = complex(float("inf"), float("inf"))
    return True


def _linear_pairs(
    true_k: torch.Tensor,
    *,
    n_samples: int = 32,
    seed: int = 0,
    noise: float = 0.0,
) -> LatentPairs:
    """Build consecutive encodings from a known linear map.

    Parameters
    ----------
    true_k : Tensor
        Ground-truth ``K`` with shape ``(d, d)``.
    n_samples : int, optional
        Number of pairs.
    seed : int, optional
        Generator seed.
    noise : float, optional
        Gaussian standard deviation added to ``z_next``.

    Returns
    -------
    LatentPairs
        ``z_t`` / ``z_next`` on the same device/dtype as ``true_k``.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    dim = true_k.shape[0]
    z_t = torch.randn(n_samples, dim, dtype=true_k.dtype, generator=generator)
    z_next = z_t @ true_k.T
    if noise > 0.0:
        z_next = z_next + noise * torch.randn(
            n_samples, dim, dtype=true_k.dtype, generator=generator
        )
    return LatentPairs(z_t=z_t, z_next=z_next)


def test_closed_form_backend_satisfies_protocol() -> None:
    """Default backend is runtime-checkable."""
    backend = ClosedFormBackend()
    assert isinstance(backend, IdentificationBackend)


def test_ridge_zero_matches_lstsq_row_operator() -> None:
    """Ridge with ``ridge=0`` matches unregularized row-convention LS."""
    true_k = torch.tensor([[0.9, 0.1], [0.0, 0.5]], dtype=torch.float64)
    pairs = _linear_pairs(true_k)
    snapshot = identify_operator(pairs, IdentificationConfig(solver="ridge", ridge=0.0))
    baseline = fit_row_operator(pairs.z_t, pairs.z_next, rank=None)
    assert snapshot.matrix is not None
    torch.testing.assert_close(snapshot.matrix, baseline, rtol=1e-10, atol=1e-12)


def test_tls_matches_baseline_helper() -> None:
    """TLS identification matches ``fit_tls_row_operator``."""
    true_k = torch.tensor([[0.8, 0.2], [-0.1, 0.7]], dtype=torch.float64)
    pairs = _linear_pairs(true_k, n_samples=48)
    snapshot = identify_operator(pairs, IdentificationConfig(solver="tls"))
    baseline = fit_tls_row_operator(pairs.z_t, pairs.z_next, rank=None)
    assert snapshot.matrix is not None
    torch.testing.assert_close(snapshot.matrix, baseline, rtol=1e-10, atol=1e-12)


def test_alternating_aliases_ridge() -> None:
    """``solver='alternating'`` uses the ridge formula."""
    true_k = torch.diag(torch.tensor([0.9, 0.4], dtype=torch.float64))
    pairs = _linear_pairs(true_k)
    ridge = identify_operator(pairs, IdentificationConfig(solver="ridge", ridge=1e-4))
    alt = identify_operator(
        pairs, IdentificationConfig(solver="alternating", ridge=1e-4)
    )
    assert ridge.matrix is not None and alt.matrix is not None
    torch.testing.assert_close(alt.matrix, ridge.matrix, rtol=1e-12, atol=1e-14)


def test_varpro_raises_not_implemented() -> None:
    """Variable projection is refused with a pointer to OptDMD."""
    true_k = torch.eye(2, dtype=torch.float64)
    pairs = _linear_pairs(true_k, n_samples=8)
    with pytest.raises(NotImplementedError, match="varpro"):
        identify_operator(pairs, IdentificationConfig(solver="varpro"))


def test_constrained_ls_projects_spectral_radius() -> None:
    """Constrained LS scales ``ρ(K)`` onto the unit disk.

    Synthetic ground truth: an expanding map with ``ρ=2``; after
    projection ``ρ≤1`` within construction tolerance.
    """
    true_k = torch.tensor([[2.0, 0.0], [0.0, 1.5]], dtype=torch.float64)
    pairs = _linear_pairs(true_k, n_samples=40)
    unconstrained = identify_operator(
        pairs, IdentificationConfig(solver="ridge", ridge=0.0)
    )
    constrained = identify_operator(
        pairs, IdentificationConfig(solver="constrained_ls", ridge=0.0)
    )
    assert unconstrained.matrix is not None and constrained.matrix is not None
    rho_open = float(torch.linalg.eigvals(unconstrained.matrix).abs().max().real)
    rho_proj = float(torch.linalg.eigvals(constrained.matrix).abs().max().real)
    assert rho_open > 1.0
    assert rho_proj == pytest.approx(1.0, rel=1e-6, abs=1e-8)


def test_noiseless_oracle_recovers_eigenvalues() -> None:
    """Ridge on a noiseless linear map recovers eigenvalues.

    Synthetic ground truth in float64; ``rtol=1e-5``, ``atol=1e-8`` from
    construction (independent oracle, not a literature table).
    """
    true_k = torch.tensor([[0.9, 0.2], [-0.1, 0.6]], dtype=torch.float64)
    pairs = _linear_pairs(true_k, n_samples=64)
    snapshot = identify_operator(pairs, IdentificationConfig(solver="ridge", ridge=0.0))
    assert snapshot.matrix is not None
    assert _eigvals_match(
        torch.linalg.eigvals(snapshot.matrix),
        torch.linalg.eigvals(true_k),
        rtol=1e-5,
        atol=1e-8,
    )


def test_noisy_oracle_tls_recovers_eigenvalues() -> None:
    """TLS on a linear Gaussian latent recovers eigenvalues within slack.

    Process noise ``σ=1e-3`` on ``z_next``; ``rtol=5e-2``, ``atol=1e-3``
    from this construction (seeded; not a bootstrap interval).
    """
    true_k = torch.diag(torch.tensor([0.85, 0.4], dtype=torch.float64))
    pairs = _linear_pairs(true_k, n_samples=128, seed=7, noise=1e-3)
    snapshot = identify_operator(pairs, IdentificationConfig(solver="tls"))
    assert snapshot.matrix is not None
    assert _eigvals_match(
        torch.linalg.eigvals(snapshot.matrix),
        torch.linalg.eigvals(true_k),
        rtol=5e-2,
        atol=1e-3,
    )


def test_identification_report_fills_one_step_and_radius() -> None:
    """Report helper fills one-step MSE and spectral radius; invariance stays empty."""
    from koopman_graph.identification import build_identification_report

    true_k = torch.diag(torch.tensor([0.7, 0.4], dtype=torch.float64))
    pairs = _linear_pairs(true_k, n_samples=20)
    snapshot = identify_operator(pairs, IdentificationConfig(solver="ridge", ridge=0.0))
    report = build_identification_report(pairs, snapshot, rollout_horizon=3)
    assert report.one_step.mse == pytest.approx(0.0, abs=1e-12)
    assert report.one_step.n_samples == 20 * 2
    assert report.rollout.mse is not None
    assert report.stability.spectral_radius == pytest.approx(0.7, rel=1e-6, abs=1e-8)
    assert report.invariance.leakage is None
    assert report.spectral.polluted is None
    assert report.reconstruction.mse is None


def test_apply_operator_snapshot_writes_dense_k() -> None:
    """Identified matrix is copied onto a dense ``KoopmanOperator``."""
    operator = KoopmanOperator(latent_dim=2, parameterization="dense")
    matrix = torch.tensor([[0.5, 0.1], [0.0, 0.3]])
    apply_operator_snapshot(operator, OperatorSnapshot(matrix=matrix))
    torch.testing.assert_close(operator.K.detach(), matrix, rtol=1e-6, atol=1e-8)


def test_apply_rejects_graph_and_controlled_operators() -> None:
    """Non-per-node and controlled operators raise."""
    from koopman_graph.operators import GraphKoopmanOperator

    matrix = torch.eye(2)
    snapshot = OperatorSnapshot(matrix=matrix)
    graph = GraphKoopmanOperator(latent_dim=2)
    with pytest.raises(ValueError, match="per-node"):
        apply_operator_snapshot(graph, snapshot)
    controlled = KoopmanOperator(latent_dim=2, control_dim=1)
    with pytest.raises(ValueError, match="controlled"):
        apply_operator_snapshot(controlled, snapshot)
