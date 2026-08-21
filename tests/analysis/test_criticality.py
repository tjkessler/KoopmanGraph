"""Tests for the sliding-window spectral-gap criticality monitor."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

import koopman_graph
import koopman_graph.analysis as analysis
from koopman_graph.analysis.criticality import (
    CriticalityReport,
    monitor_critical_transition,
)
from koopman_graph.spectrum_types import KoopmanSpectrum, compute_spectrum

# Independent construction: diag(1, 1+δ_t) has gap δ_t. Float64 eig residual.
_ORACLE_REL = 1e-10
_ORACLE_ABS = 1e-12
_N_TIMES = 11
_WINDOW = 5
_GAP0 = 1.0
# Unit-spaced δ_t = 1 - t/10 ⇒ window-5 closure rate is 0.4/4 = 0.1.
_UNIT_CLOSURE_RATE = 0.1


def _diagonal_spectrum(primary: float, secondary: float) -> KoopmanSpectrum:
    matrix = torch.diag(
        torch.tensor([primary, secondary], dtype=torch.float64),
    )
    return compute_spectrum(matrix, time_step=1.0)


def _closing_spectra() -> list[KoopmanSpectrum]:
    spectra = []
    for step in range(_N_TIMES):
        gap = _GAP0 * (1.0 - step / (_N_TIMES - 1))
        spectra.append(_diagonal_spectrum(1.0, 1.0 + gap))
    return spectra


def _stable_spectra() -> list[KoopmanSpectrum]:
    return [_diagonal_spectrum(1.0, 1.0 + _GAP0) for _ in range(_N_TIMES)]


def test_criticality_not_on_root_facade() -> None:
    """Monitor types stay off the thin root façade."""
    assert "CriticalityReport" not in koopman_graph.__all__
    assert "monitor_critical_transition" not in koopman_graph.__all__
    assert "CriticalityReport" in analysis.__all__
    assert "monitor_critical_transition" in analysis.__all__
    assert analysis.CriticalityReport is CriticalityReport
    assert analysis.monitor_critical_transition is monitor_critical_transition


def test_gap_closure_raises_score_versus_stable_control() -> None:
    """Linearly closing δ_t raises the score relative to a constant gap.

    Independent oracle: K_t = diag(1, 1+δ_t) with δ_t = 1 - t/10 on
    unit sample times. Window 5 ⇒ (γ_{t-4} - γ_t)/4 = 0.1. The control
    keeps δ ≡ 1 so the rate is 0.
    """
    closing = monitor_critical_transition(_closing_spectra(), window=_WINDOW)
    control = monitor_critical_transition(_stable_spectra(), window=_WINDOW)
    expected_gaps = [_GAP0 * (1.0 - step / (_N_TIMES - 1)) for step in range(_N_TIMES)]
    assert closing.spectral_gap.tolist() == pytest.approx(
        expected_gaps, rel=_ORACLE_REL, abs=_ORACLE_ABS
    )
    assert closing.gap_closure_rate.tolist() == pytest.approx(
        [_UNIT_CLOSURE_RATE] * (_N_TIMES - _WINDOW + 1),
        rel=_ORACLE_REL,
        abs=_ORACLE_ABS,
    )
    assert closing.max_gap_closure_rate == pytest.approx(
        _UNIT_CLOSURE_RATE, rel=_ORACLE_REL, abs=_ORACLE_ABS
    )
    assert control.max_gap_closure_rate == pytest.approx(0.0, abs=_ORACLE_ABS)
    assert closing.max_gap_closure_rate > control.max_gap_closure_rate
    assert not bool(closing.near_defective.any().item())
    assert not bool(control.near_defective.any().item())


def test_sample_times_scale_the_closure_rate() -> None:
    """Doubling sample spacing halves the unit-step closure rate."""
    times = [2.0 * step for step in range(_N_TIMES)]
    report = monitor_critical_transition(
        _closing_spectra(),
        window=_WINDOW,
        sample_times=times,
    )
    assert report.max_gap_closure_rate == pytest.approx(
        _UNIT_CLOSURE_RATE / 2.0, rel=_ORACLE_REL, abs=_ORACLE_ABS
    )


def test_pairwise_gap_is_not_magnitude_consecutive() -> None:
    """Closest pair need not be adjacent after |λ|-sort (1, -1, 1.05)."""
    matrix = torch.diag(torch.tensor([1.0, -1.0, 1.05], dtype=torch.float64))
    with pytest.warns(UserWarning, match="Nyquist"):
        spectrum = compute_spectrum(matrix, time_step=1.0)
    report = monitor_critical_transition([spectrum, spectrum], window=2)
    assert report.spectral_gap[0].item() == pytest.approx(
        0.05, rel=_ORACLE_REL, abs=_ORACLE_ABS
    )


def test_near_defective_flags_singular_eigenbasis_without_raising() -> None:
    """A zero column sets the flag; the monitor does not raise."""
    eigenvalues = torch.tensor([1.0 + 0.0j, 0.5 + 0.0j])
    vectors = torch.tensor(
        [[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.0 + 0.0j]],
    )
    spectrum = KoopmanSpectrum(
        eigenvalues=eigenvalues,
        eigenvectors=vectors,
        magnitudes=eigenvalues.abs(),
        growth_rates=torch.log(eigenvalues.abs()),
        frequencies=torch.zeros(2),
        time_step=1.0,
    )
    assert spectrum.diagnostics is None
    report = monitor_critical_transition([spectrum, spectrum], window=2)
    assert report.near_defective.tolist() == [True, True]


def test_near_defective_uses_stored_diagnostics() -> None:
    """Stored κ(V)=+∞ flags even when the live basis is well conditioned."""
    well = _diagonal_spectrum(1.0, 2.0)
    assert well.diagnostics is not None
    inf_diag = replace(well.diagnostics, eigenvector_condition=float("inf"))
    defective = replace(well, diagnostics=inf_diag)
    report = monitor_critical_transition([defective, well], window=2)
    assert report.near_defective.tolist() == [True, False]


def test_monitor_rejects_invalid_inputs() -> None:
    """Boundary validation mentions the broken constraint."""
    pair = _stable_spectra()[:2]
    with pytest.raises(ValueError, match="nonempty sequence"):
        monitor_critical_transition([], window=2)
    with pytest.raises(ValueError, match="window must be an int >= 2"):
        monitor_critical_transition(pair, window=1)
    with pytest.raises(ValueError, match="window must be an int >= 2"):
        monitor_critical_transition(pair, window=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="window must be <= len"):
        monitor_critical_transition(pair, window=3)
    scalar = compute_spectrum(torch.ones(1, 1, dtype=torch.float64), time_step=1.0)
    with pytest.raises(ValueError, match="at least two eigenvalues"):
        monitor_critical_transition([scalar, scalar], window=2)
    wide = _diagonal_spectrum(1.0, 2.0)
    tall = compute_spectrum(torch.eye(3, dtype=torch.float64), time_step=1.0)
    with pytest.raises(ValueError, match="share latent_dim"):
        monitor_critical_transition([wide, tall], window=2)
    with pytest.raises(ValueError, match="strictly increasing"):
        monitor_critical_transition(pair, window=2, sample_times=[0.0, 0.0])
    with pytest.raises(ValueError, match="sample_times must have length"):
        monitor_critical_transition(pair, window=2, sample_times=[0.0])
    with pytest.raises(ValueError, match="sample_times must be finite"):
        monitor_critical_transition(pair, window=2, sample_times=[0.0, float("nan")])
