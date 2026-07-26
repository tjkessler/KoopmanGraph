"""Oracle tests for long-horizon statistics (TASK-1406)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from koopman_graph import statistics as statistics_module
from koopman_graph.statistics import (
    LongHorizonReport,
    compute_long_horizon_report,
    invariant_measure_distance,
    largest_lyapunov_exponent,
    power_spectral_density,
    spectral_distance,
)

# Classical Lorenz-63 largest LE (σ=10, ρ=28, β=8/3), commonly tabulated as
# ≈0.906 (see Sprott, Chaos and Time-Series Analysis, 2003; numerical
# consensus ~0.9). Absolute tolerance documented with the oracle hyperparameters.
_LORENZ63_LLE_REF = 0.906
_LORENZ63_LLE_TOL = 0.15


def _rk4_lorenz63(state: torch.Tensor, dt: float) -> torch.Tensor:
    """One RK4 step of classical Lorenz-63."""
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0

    def rhs(s: torch.Tensor) -> torch.Tensor:
        x, y, z = s.unbind(-1)
        return torch.stack(
            [sigma * (y - x), x * (rho - z) - y, x * y - beta * z],
            dim=-1,
        )

    k1 = rhs(state)
    k2 = rhs(state + 0.5 * dt * k1)
    k3 = rhs(state + 0.5 * dt * k2)
    k4 = rhs(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def _lorenz63_x(
    *,
    num_steps: int = 10_000,
    dt: float = 0.01,
    burn_in: int = 5_000,
    seed: int = 0,
) -> torch.Tensor:
    """Seeded Lorenz-63 ``x(t)`` after burn-in (float64)."""
    generator = torch.Generator().manual_seed(seed)
    state = torch.tensor([0.0, 1.0, 1.05], dtype=torch.float64)
    state = state + 1e-3 * torch.randn(3, dtype=torch.float64, generator=generator)
    samples: list[torch.Tensor] = []
    for step in range(burn_in + num_steps):
        state = _rk4_lorenz63(state, dt)
        if step >= burn_in:
            samples.append(state[0].clone())
    return torch.stack(samples)


def test_statistics_module_imports_torch_only() -> None:
    """``statistics.py`` must stay a torch-only leaf."""
    source = Path(statistics_module.__file__).read_text(encoding="utf-8")
    forbidden = (
        "koopman_graph.analysis",
        "koopman_graph.data",
        "koopman_graph.model",
        "koopman_graph.uq",
        "koopman_graph.nn",
        "koopman_graph.operators",
        "koopman_graph.metrics",
        "koopman_graph.protocols",
        "scipy",
        "nolds",
        "numpy",
    )
    for name in forbidden:
        assert f"import {name}" not in source
        assert f"from {name}" not in source


def test_not_on_root_all() -> None:
    """Power-user module stays off root ``__all__``."""
    import koopman_graph

    assert "statistics" not in koopman_graph.__all__
    assert "power_spectral_density" not in koopman_graph.__all__
    assert "largest_lyapunov_exponent" not in koopman_graph.__all__


def test_psd_peak_at_analytic_frequency() -> None:
    """Pure sinusoid places the Welch PSD peak within one frequency bin.

    Construction: segment length ``L=64``, tone at bin ``k=4`` so
    ``f = k/L = 1/16`` cycles/sample. Peak index must equal ``k``.
    """
    segment_length = 64
    bin_index = 4
    freq = bin_index / segment_length
    time = torch.arange(512, dtype=torch.float64)
    series = torch.sin(2.0 * torch.pi * freq * time)
    psd = power_spectral_density(
        series,
        segment_length=segment_length,
        overlap=0.5,
    )
    assert psd.shape == (segment_length // 2 + 1, 1)
    peak = int(torch.argmax(psd[:, 0]).item())
    assert peak == bin_index


def test_identical_series_distances_are_zero() -> None:
    r"""Identical series yield zero :math:`d_{\mathrm{PSD}}` and :math:`W_1`."""
    torch.manual_seed(0)
    series = torch.randn(200, 3, 2)
    spec = spectral_distance(series, series.clone())
    w1 = invariant_measure_distance(series, series.clone())
    assert spec.item() == pytest.approx(0.0, abs=1e-12)
    assert w1.item() == pytest.approx(0.0, abs=1e-12)


def test_invariant_measure_distance_increases_with_location_shift() -> None:
    """Wasserstein-1 grows monotonically under a seeded location shift."""
    torch.manual_seed(1)
    base = torch.randn(300, 4)
    shifts = [0.0, 0.5, 1.0, 2.0]
    distances = [
        float(invariant_measure_distance(base, base + shift).item()) for shift in shifts
    ]
    assert distances[0] == pytest.approx(0.0, abs=1e-12)
    for left, right in zip(distances, distances[1:], strict=False):
        assert right > left


def test_invariant_measure_matches_brute_force_wasserstein() -> None:
    """Sorted-sample :math:`W_1` matches a brute-force 1-D reference.

    Tolerance: ``abs=1e-12`` (exact formula for equal-length 1-D samples).
    """
    torch.manual_seed(2)
    prediction = torch.randn(32, 2)
    target = torch.randn(32, 2)
    got = invariant_measure_distance(prediction, target, per_node=True)

    expected = []
    for channel in range(2):
        p_sorted = torch.sort(prediction[:, channel]).values
        t_sorted = torch.sort(target[:, channel]).values
        expected.append(torch.mean(torch.abs(p_sorted - t_sorted)))
    expected_t = torch.stack(expected)
    assert torch.allclose(got, expected_t, atol=1e-12, rtol=0.0)
    pooled = invariant_measure_distance(prediction, target, per_node=False)
    assert pooled.item() == pytest.approx(float(expected_t.mean().item()), abs=1e-12)


def test_lorenz63_largest_lyapunov_near_literature() -> None:
    """Rosenstein LLE on seeded Lorenz-63 recovers the literature value.

    Oracle setup (documented hyperparameters, not bare defaults):
    - classical parameters σ=10, ρ=28, β=8/3
    - RK4 with ``dt=0.01``, burn-in 5000, then 10000 samples of ``x(t)``
    - ``embedding_dim=6``, ``trajectory_len=40``, auto lag / Theiler
    - reference ``λ₁ ≈ 0.906``; absolute tolerance ``0.15``

    The tolerance is wider than PSD/W1 oracles because Rosenstein estimates
    remain sensitive to finite-sample embedding choices; the point is a
    construction check against a cited attractor value, not a high-precision
    spectral computation.
    """
    dt = 0.01
    series = _lorenz63_x(num_steps=10_000, dt=dt, burn_in=5_000, seed=0)
    estimate = largest_lyapunov_exponent(
        series,
        embedding_dim=6,
        trajectory_len=40,
        dt=dt,
    )
    assert float(estimate.item()) == pytest.approx(
        _LORENZ63_LLE_REF,
        abs=_LORENZ63_LLE_TOL,
    )


def test_largest_lyapunov_rejects_bad_inputs() -> None:
    """Lyapunov helper validates finiteness and basic parameter contracts."""
    bad = torch.tensor([1.0, float("nan"), 0.0])
    with pytest.raises(ValueError, match="finite"):
        largest_lyapunov_exponent(bad)
    series = torch.randn(64)
    with pytest.raises(ValueError, match="dt"):
        largest_lyapunov_exponent(series, dt=0.0)
    with pytest.raises(ValueError, match="embedding_dim"):
        largest_lyapunov_exponent(series, embedding_dim=1)


def test_long_horizon_report_smoke() -> None:
    """Report aggregates spectral / Wasserstein distances; Lyapunov optional."""
    torch.manual_seed(3)
    target = torch.randn(128, 2)
    prediction = target + 0.1 * torch.randn_like(target)
    report = compute_long_horizon_report(prediction, target)
    assert isinstance(report, LongHorizonReport)
    assert report.num_steps == 128
    assert report.spectral_distance > 0.0
    assert report.invariant_measure_distance > 0.0
    assert report.largest_lyapunov_exponent is None

    report_l = compute_long_horizon_report(
        prediction,
        target,
        lyapunov=True,
        lyapunov_kwargs={
            "embedding_dim": 3,
            "trajectory_len": 10,
            "delay": 1,
            "theiler": 5,
        },
    )
    assert report_l.largest_lyapunov_exponent is not None


def test_rejects_nonfinite_and_bad_overlap() -> None:
    """Finite-input and overlap contracts raise clearly."""
    bad = torch.tensor([1.0, float("nan"), 0.0])
    with pytest.raises(ValueError, match="finite"):
        power_spectral_density(bad)
    series = torch.randn(32)
    with pytest.raises(ValueError, match="overlap"):
        power_spectral_density(series, overlap=1.0)


def test_psd_and_distance_validation_branches() -> None:
    """Welch / distance helpers reject invalid shapes and hyperparameters."""
    with pytest.raises(ValueError, match="at least 1 dimension"):
        power_spectral_density(torch.tensor(1.0))
    series = torch.randn(16, 2)
    with pytest.raises(ValueError, match="out of range"):
        power_spectral_density(series, dim=5)
    with pytest.raises(ValueError, match='window must be "hann"'):
        power_spectral_density(series, window="hamming")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="time length must be >= 2"):
        power_spectral_density(torch.tensor([1.0]))
    with pytest.raises(ValueError, match="segment_length must be >= 2"):
        power_spectral_density(series, segment_length=1)
    with pytest.raises(ValueError, match="cannot exceed time length"):
        power_spectral_density(series, segment_length=32)
    # Even segment length exercises the Nyquist half-scale branch.
    psd = power_spectral_density(torch.randn(32), segment_length=8, overlap=0.0)
    assert psd.shape[0] == 5
    with pytest.raises(ValueError, match="same shape"):
        spectral_distance(torch.randn(16), torch.randn(8))
    with pytest.raises(ValueError, match="eps must be positive"):
        spectral_distance(series, series, eps=0.0)
    with pytest.raises(ValueError, match="same shape"):
        invariant_measure_distance(torch.randn(16), torch.randn(8))
    with pytest.raises(ValueError, match="same shape"):
        compute_long_horizon_report(torch.randn(16, 2), torch.randn(8, 2))


def test_lyapunov_validation_and_edge_branches() -> None:
    """Rosenstein estimator covers lag/Theiler/fit failure paths."""
    series = torch.randn(256)
    with pytest.raises(ValueError, match="trajectory_len"):
        largest_lyapunov_exponent(series, trajectory_len=1, delay=1, theiler=1)
    with pytest.raises(ValueError, match="fit_offset"):
        largest_lyapunov_exponent(series, fit_offset=-1, delay=1, theiler=1)
    with pytest.raises(ValueError, match="channel="):
        largest_lyapunov_exponent(torch.randn(64, 2), channel=3, delay=1, theiler=1)
    with pytest.raises(ValueError, match="fit_range"):
        largest_lyapunov_exponent(
            series, fit_range=(-1, 5), delay=1, theiler=1, embedding_dim=3
        )
    with pytest.raises(ValueError, match="delay must be >= 1"):
        largest_lyapunov_exponent(series, delay=0, theiler=1, embedding_dim=3)
    with pytest.raises(ValueError, match="too short for delay embedding"):
        largest_lyapunov_exponent(
            torch.randn(4),
            embedding_dim=3,
            delay=2,
            theiler=1,
            trajectory_len=2,
        )
    with pytest.raises(ValueError, match="too short for Rosenstein"):
        largest_lyapunov_exponent(
            torch.randn(20),
            embedding_dim=3,
            delay=1,
            theiler=5,
            trajectory_len=10,
        )
    # Constant series: auto Theiler falls back when FFT power is zero, then
    # divergence is non-finite so the fit fails.
    with pytest.raises(ValueError, match="finite divergence|Theiler|too short"):
        largest_lyapunov_exponent(
            torch.ones(128),
            embedding_dim=3,
            delay=1,
            theiler=None,
            trajectory_len=5,
            min_neighbors=5,
        )
    # Explicit fit_range path on a long noisy series.
    lle = largest_lyapunov_exponent(
        series,
        embedding_dim=3,
        delay=1,
        theiler=2,
        trajectory_len=8,
        fit_range=(1, 4),
    )
    assert torch.isfinite(lle)
    # Auto delay with aggressive min_neighbors truncates lag search.
    lle_auto = largest_lyapunov_exponent(
        torch.randn(80),
        embedding_dim=4,
        delay=None,
        theiler=2,
        trajectory_len=5,
        min_neighbors=40,
    )
    assert torch.isfinite(lle_auto)
    # Degenerate / insufficient fit: identical samples → zero distances.
    flat = torch.zeros(64)
    with pytest.raises(ValueError, match="finite divergence|degenerate"):
        largest_lyapunov_exponent(
            flat,
            embedding_dim=2,
            delay=1,
            theiler=1,
            trajectory_len=4,
            fit_range=(0, 3),
        )
