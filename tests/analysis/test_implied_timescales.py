"""Tests for discrete-time implied timescale helpers."""

from __future__ import annotations

import math

import pytest
import torch

from koopman_graph.analysis import ImpliedTimescales, implied_timescales


def test_analytic_diagonal_oracle_in_steps() -> None:
    """Known ``|λ|`` values match hand-computed timescales in step units."""
    eigenvalues = torch.tensor([0.9, 0.5, 1.0, 0.0], dtype=torch.float64)
    report = implied_timescales(eigenvalues, lag_steps=1)

    assert isinstance(report, ImpliedTimescales)
    assert report.unit == "steps"
    assert report.tau == 1.0
    assert report.timestep is None
    assert report.valid.tolist() == [True, True, False, False]
    assert report.aliasing_warning.tolist() == [False, False, False, False]

    expected_0 = -1.0 / math.log(0.9)
    expected_1 = -1.0 / math.log(0.5)
    assert report.timescales[0].item() == pytest.approx(expected_0)
    assert report.timescales[1].item() == pytest.approx(expected_1)
    assert math.isinf(report.timescales[2].item())
    assert math.isinf(report.timescales[3].item())
    assert not any(math.isnan(float(v)) for v in report.timescales.tolist())


def test_physical_timestep_scales_tau_and_records_unit() -> None:
    """Optional timestep sets ``τ = lag_steps * timestep`` and unit name."""
    eigenvalues = torch.tensor([0.8], dtype=torch.float64)
    report = implied_timescales(
        eigenvalues,
        lag_steps=2,
        timestep=0.1,
        timestep_unit="ps",
    )
    assert report.unit == "ps"
    assert report.tau == pytest.approx(0.2)
    assert report.timestep == pytest.approx(0.1)
    expected = -0.2 / math.log(0.8)
    assert report.timescales[0].item() == pytest.approx(expected)


def test_complex_eigenvalues_use_magnitude() -> None:
    """Complex eigenvalues enter through ``|λ|`` only."""
    eigenvalues = torch.tensor([0.6 + 0.0j, 0.3 + 0.4j], dtype=torch.complex128)
    report = implied_timescales(eigenvalues, lag_steps=1)
    assert report.magnitudes[0].item() == pytest.approx(0.6)
    assert report.magnitudes[1].item() == pytest.approx(0.5)
    assert report.valid.tolist() == [True, True]
    assert report.timescales[1].item() == pytest.approx(-1.0 / math.log(0.5))


def test_near_unit_and_near_zero_marked_invalid() -> None:
    """Values inside the ``ε`` bands are invalid with ``+∞``, not ``NaN``."""
    eps = 1e-6
    eigenvalues = torch.tensor(
        [1.0 - 0.5 * eps, eps * 0.5, 1.0 + eps, -0.7],
        dtype=torch.float64,
    )
    report = implied_timescales(eigenvalues, lag_steps=3, magnitude_atol=eps)
    assert report.valid.tolist() == [False, False, False, True]
    assert report.aliasing_warning.tolist() == [False, False, False, True]
    assert all(math.isinf(float(v)) for v in report.timescales[:3].tolist())
    assert report.timescales[3].item() == pytest.approx(-3.0 / math.log(0.7))


def test_negative_and_positive_share_timescale_only_negative_flags() -> None:
    """λ=-0.9 flags aliasing; λ=+0.9 does not; both remain valid.

    Independent oracle: t = -1 / ln(0.9) for both magnitudes.
    """
    report = implied_timescales(
        torch.tensor([-0.9, 0.9], dtype=torch.float64),
        lag_steps=1,
    )
    assert report.aliasing_warning.tolist() == [True, False]
    assert report.valid.tolist() == [True, True]
    expected = -1.0 / math.log(0.9)
    assert report.timescales[0].item() == pytest.approx(expected)
    assert report.timescales[1].item() == pytest.approx(expected)


def test_argument_validation() -> None:
    """Inconsistent lag / timestep / unit arguments raise clearly."""
    eigenvalues = torch.tensor([0.5], dtype=torch.float64)
    with pytest.raises(ValueError, match="lag_steps must be >= 1"):
        implied_timescales(eigenvalues, lag_steps=0)
    with pytest.raises(ValueError, match="timestep_unit is required"):
        implied_timescales(eigenvalues, lag_steps=1, timestep=0.1)
    with pytest.raises(ValueError, match="timestep_unit must be None"):
        implied_timescales(
            eigenvalues,
            lag_steps=1,
            timestep_unit="ps",
        )
    with pytest.raises(ValueError, match="timestep must be a finite value > 0"):
        implied_timescales(
            eigenvalues,
            lag_steps=1,
            timestep=0.0,
            timestep_unit="ps",
        )


def test_exported_from_analysis_not_root() -> None:
    """Implied timescale helpers stay on the analysis surface."""
    import koopman_graph as kg
    import koopman_graph.analysis as analysis

    assert "implied_timescales" in analysis.__all__
    assert "ImpliedTimescales" in analysis.__all__
    assert "implied_timescales" not in kg.__all__
