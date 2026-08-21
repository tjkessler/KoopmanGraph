"""Residual-aware dictionary selection on a synthetic polluted lift."""

from __future__ import annotations

import pytest
import torch

from koopman_graph.analysis.resdmd import resdmd
from koopman_graph.identification import (
    DEFAULT_RESDMD_GATE_TOLERANCE,
    IdentificationConfig,
    LatentPairs,
    ResDMDGateCandidate,
    build_identification_report,
    identify_operator,
    select_resdmd_gated,
)


def _noisy_linear_trajectory(
    true_k: torch.Tensor,
    *,
    n_times: int,
    noise: float,
    seed: int,
) -> torch.Tensor:
    """Simulate ``z_{t+1} = z_t K^\\top + \\varepsilon`` in :math:`\\mathbb{R}^{2}`.

    Parameters
    ----------
    true_k : Tensor
        Shared linear map with shape ``(2, 2)``.
    n_times : int
        Trajectory length.
    noise : float
        i.i.d. Gaussian standard deviation on the increment.
    seed : int
        Generator seed.

    Returns
    -------
    Tensor
        Time-major states with shape ``(n_times, 2)``.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    state = torch.tensor([0.4, -0.2], dtype=true_k.dtype)
    rows = []
    for _ in range(n_times):
        rows.append(state.clone())
        state = state @ true_k.T + noise * torch.randn(
            2, dtype=true_k.dtype, generator=generator
        )
    return torch.stack(rows, dim=0)


def _one_step_mse(psi0: torch.Tensor, psi1: torch.Tensor) -> float:
    """Train one-step mean squared error of unconstrained least squares.

    Parameters
    ----------
    psi0, psi1 : Tensor
        Consecutive dictionary rows with the same shape.

    Returns
    -------
    float
        Mean squared residual of ``psi0 @ K ≈ psi1``.
    """
    solution = torch.linalg.lstsq(psi0, psi1).solution
    return float((psi0 @ solution - psi1).square().mean().item())


def _residual_max(psi0: torch.Tensor, psi1: torch.Tensor) -> float:
    """Maximum finite-dictionary ResDMD residual.

    Parameters
    ----------
    psi0, psi1 : Tensor
        Consecutive dictionary rows.

    Returns
    -------
    float
        ``max`` of :func:`~koopman_graph.analysis.resdmd` residuals.
    """
    report = resdmd(psi0, psi1, tolerance=DEFAULT_RESDMD_GATE_TOLERANCE)
    return float(report.residuals.max().real.item())


def _polluted_dictionaries() -> tuple[ResDMDGateCandidate, ResDMDGateCandidate]:
    """Identity dictionary versus identity plus a leaked future coordinate.

    Returns
    -------
    tuple of ResDMDGateCandidate
        ``identity`` then ``identity_plus_junk``.
    """
    true_k = torch.tensor([[0.9, 0.15], [-0.05, 0.55]], dtype=torch.float64)
    traj = _noisy_linear_trajectory(true_k, n_times=48, noise=1e-4, seed=0)
    psi0 = traj[:-1]
    psi1 = traj[1:]
    junk0 = traj[1:, :1]
    junk1 = torch.cat([traj[2:, :1], traj[-1:, :1]], dim=0)
    pol0 = torch.cat([psi0, junk0], dim=1)
    pol1 = torch.cat([psi1, junk1], dim=1)
    clean = ResDMDGateCandidate(
        name="identity",
        mse=_one_step_mse(psi0, psi1),
        residual_max=_residual_max(psi0, psi1),
    )
    polluted = ResDMDGateCandidate(
        name="identity_plus_junk",
        mse=_one_step_mse(pol0, pol1),
        residual_max=_residual_max(pol0, pol1),
    )
    return clean, polluted


def test_rmse_only_picks_polluted_dictionary() -> None:
    """Min train MSE prefers the junk lift; the residual gate keeps identity."""
    clean, polluted = _polluted_dictionaries()
    assert polluted.mse < clean.mse
    assert clean.residual_max <= DEFAULT_RESDMD_GATE_TOLERANCE
    assert polluted.residual_max > DEFAULT_RESDMD_GATE_TOLERANCE
    rmse_only = select_resdmd_gated(
        (clean, polluted),
        gate_resdmd=False,
    )
    assert rmse_only.selected == "identity_plus_junk"
    assert rmse_only.rejected_alternatives == ()
    assert rmse_only.gated is False
    gated = select_resdmd_gated((clean, polluted), gate_resdmd=True)
    assert gated.selected == "identity"
    assert gated.rejected_alternatives == ("identity_plus_junk",)
    assert gated.gated is True
    assert gated.residual_tolerance == pytest.approx(DEFAULT_RESDMD_GATE_TOLERANCE)


def test_all_polluted_candidates_raise() -> None:
    """Gating raises when every residual exceeds the tolerance."""
    candidates = (
        ResDMDGateCandidate(name="a", mse=0.1, residual_max=0.5),
        ResDMDGateCandidate(name="b", mse=0.2, residual_max=0.4),
    )
    with pytest.raises(ValueError, match="rejected every candidate"):
        select_resdmd_gated(candidates, gate_resdmd=True)


def test_empty_and_duplicate_names_raise() -> None:
    """Empty pools and colliding labels are refused."""
    with pytest.raises(ValueError, match="at least one candidate"):
        select_resdmd_gated(())
    twin = ResDMDGateCandidate(name="same", mse=0.1, residual_max=0.0)
    with pytest.raises(ValueError, match="unique"):
        select_resdmd_gated((twin, twin))
    with pytest.raises(ValueError, match="non-empty str"):
        ResDMDGateCandidate(name="", mse=0.1, residual_max=0.0)
    with pytest.raises(TypeError, match="gate_resdmd must be a bool"):
        select_resdmd_gated(
            (ResDMDGateCandidate(name="ok", mse=0.1, residual_max=0.0),),
            gate_resdmd=1,  # type: ignore[arg-type]
        )


def test_build_report_fills_spectral_only_when_gated() -> None:
    """Default reports leave ``spectral`` unset; ``gate_resdmd`` fills it."""
    true_k = torch.diag(torch.tensor([0.7, 0.4], dtype=torch.float64))
    generator = torch.Generator()
    generator.manual_seed(0)
    z_t = torch.randn(20, 2, dtype=torch.float64, generator=generator)
    pairs = LatentPairs(z_t=z_t, z_next=z_t @ true_k.T)
    snapshot = identify_operator(pairs, IdentificationConfig(solver="ridge", ridge=0.0))
    default = build_identification_report(pairs, snapshot)
    assert default.spectral.polluted is None
    assert default.spectral.residual_max is None
    gated = build_identification_report(pairs, snapshot, gate_resdmd=True)
    assert gated.spectral.residual_max is not None
    assert gated.spectral.residual_max == pytest.approx(0.0, abs=1e-8)
    assert gated.spectral.polluted is False
