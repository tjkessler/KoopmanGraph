"""Tests for SpectralDiagnostics attached to compute_spectrum."""

from __future__ import annotations

import math
import warnings
from dataclasses import replace

import pytest
import torch

import koopman_graph
import koopman_graph.analysis as analysis
from koopman_graph.spectrum_types import (
    ALIASING_ARG_ATOL,
    CONDITION_WARN,
    DefectiveSpectrumError,
    KoopmanSpectrum,
    SpectralDiagnostics,
    compute_generator_spectrum,
    compute_spectral_diagnostics,
    compute_spectrum,
)

# Closed-form 2×2 triangular toy: eigenvalues 1 and 2; both κ_i = sqrt(1+a²).
_NONNORMAL_SHEAR = 10.0
_NONNORMAL_KAPPA = math.sqrt(1.0 + _NONNORMAL_SHEAR**2)
_NONNORMAL_DEPARTURE = _NONNORMAL_SHEAR * math.sqrt(2.0 * (_NONNORMAL_SHEAR**2 + 1.0))
_NONNORMAL_RELATIVE = _NONNORMAL_DEPARTURE / (_NONNORMAL_SHEAR**2 + 5.0)
# Independent algebraic identities on float64 eigendecomposition.
_ORACLE_REL = 1e-10
_ORACLE_ABS = 1e-12
_NORMAL_ABS = 1e-8


def _nonnormal_operator() -> torch.Tensor:
    return torch.tensor(
        [[1.0, _NONNORMAL_SHEAR], [0.0, 2.0]],
        dtype=torch.float64,
    )


def test_spectral_diagnostics_not_on_root_facade() -> None:
    """Conditioning types stay off the thin root façade."""
    assert "SpectralDiagnostics" not in koopman_graph.__all__
    assert "compute_spectral_diagnostics" not in koopman_graph.__all__
    assert "DefectiveSpectrumError" not in koopman_graph.__all__
    assert "CONDITION_WARN" not in koopman_graph.__all__
    assert "SpectralDiagnostics" in analysis.__all__
    assert "compute_spectral_diagnostics" in analysis.__all__
    assert "DefectiveSpectrumError" in analysis.__all__
    assert "CONDITION_WARN" in analysis.__all__
    assert analysis.SpectralDiagnostics is SpectralDiagnostics
    assert analysis.compute_spectral_diagnostics is compute_spectral_diagnostics
    assert analysis.DefectiveSpectrumError is DefectiveSpectrumError
    assert analysis.CONDITION_WARN == CONDITION_WARN == 1e6


def test_compute_spectrum_attaches_diagnostics() -> None:
    """Discrete assembly always fills optional diagnostics."""
    spectrum = compute_spectrum(torch.eye(2, dtype=torch.float64), time_step=1.0)
    assert spectrum.diagnostics is not None
    assert isinstance(spectrum.diagnostics, SpectralDiagnostics)


def test_identity_is_normal_with_unit_condition() -> None:
    """The identity is normal: κ(V)≈1, κ_i≈1, both departures ≈0."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        spectrum = compute_spectrum(torch.eye(3, dtype=torch.float64), time_step=1.0)
    diag = spectrum.diagnostics
    assert diag is not None
    assert diag.eigenvector_condition == pytest.approx(1.0, rel=1e-6, abs=_NORMAL_ABS)
    assert diag.eigenvalue_sensitivities.tolist() == pytest.approx(
        [1.0, 1.0, 1.0],
        rel=1e-6,
        abs=_NORMAL_ABS,
    )
    assert diag.departure_from_normality == pytest.approx(0.0, abs=_NORMAL_ABS)
    assert diag.departure_from_normality_relative == pytest.approx(0.0, abs=_NORMAL_ABS)
    assert diag.nyquist_frequency == pytest.approx(
        0.5, rel=_ORACLE_REL, abs=_ORACLE_ABS
    )
    assert diag.aliasing_warning.tolist() == [False, False, False]
    assert diag.sign_of_eigenvalue.tolist() == pytest.approx([1.0, 1.0, 1.0])


def test_hermitian_matrix_is_normal() -> None:
    """A real symmetric map is unitarily diagonalizable (departure ≈0)."""
    operator = torch.tensor([[2.0, 1.0], [1.0, 2.0]], dtype=torch.float64)
    spectrum = compute_spectrum(operator, time_step=0.5)
    diag = spectrum.diagnostics
    assert diag is not None
    assert diag.eigenvector_condition == pytest.approx(1.0, rel=1e-5, abs=_NORMAL_ABS)
    assert diag.eigenvalue_sensitivities.tolist() == pytest.approx(
        [1.0, 1.0],
        rel=1e-5,
        abs=_NORMAL_ABS,
    )
    assert diag.departure_from_normality == pytest.approx(0.0, abs=_NORMAL_ABS)
    assert diag.departure_from_normality_relative == pytest.approx(0.0, abs=_NORMAL_ABS)
    assert diag.nyquist_frequency == pytest.approx(
        1.0, rel=_ORACLE_REL, abs=_ORACLE_ABS
    )
    assert not bool(diag.aliasing_warning.any().item())


def test_nonnormal_triangular_matches_closed_form() -> None:
    """Shear [[1, a], [0, 2]] has κ_i=sqrt(1+a²) and known Frobenius departure.

    Algebraic identities for this 2×2 (not a literature table). Tolerances
    are independent-oracle float64 residuals after ``torch.linalg.eig``.
    """
    operator = _nonnormal_operator()
    spectrum = compute_spectrum(operator, time_step=1.0)
    diag = spectrum.diagnostics
    assert diag is not None
    expected = [_NONNORMAL_KAPPA, _NONNORMAL_KAPPA]
    assert diag.eigenvalue_sensitivities.tolist() == pytest.approx(
        expected, rel=_ORACLE_REL, abs=_ORACLE_ABS
    )
    assert diag.departure_from_normality == pytest.approx(
        _NONNORMAL_DEPARTURE, rel=_ORACLE_REL, abs=_ORACLE_ABS
    )
    assert diag.departure_from_normality_relative == pytest.approx(
        _NONNORMAL_RELATIVE, rel=_ORACLE_REL, abs=_ORACLE_ABS
    )
    assert diag.eigenvector_condition > 1.0
    vectors = spectrum.eigenvectors
    col_norms = torch.linalg.vector_norm(vectors, dim=0)
    unit = vectors / col_norms.to(dtype=vectors.dtype)
    oracle_cond = float(torch.linalg.cond(unit).real.item())
    assert diag.eigenvector_condition == pytest.approx(
        oracle_cond, rel=_ORACLE_REL, abs=_ORACLE_ABS
    )


def test_zero_matrix_relative_departure_is_zero() -> None:
    """The zero map is normal; relative departure is defined as 0."""
    spectrum = compute_spectrum(torch.zeros(2, 2, dtype=torch.float64), time_step=1.0)
    diag = spectrum.diagnostics
    assert diag is not None
    assert diag.departure_from_normality == pytest.approx(0.0, abs=_NORMAL_ABS)
    assert diag.departure_from_normality_relative == pytest.approx(0.0, abs=_NORMAL_ABS)
    assert diag.sign_of_eigenvalue.tolist() == pytest.approx([0.0, 0.0])
    assert not bool(diag.aliasing_warning.any().item())


def test_generator_spectrum_populates_diagnostics() -> None:
    """Continuous assembly uses the same diagnostic helper."""
    generator = torch.tensor([[-1.0, 0.2], [0.2, -2.0]], dtype=torch.float64)
    spectrum = compute_generator_spectrum(generator)
    assert spectrum.diagnostics is not None
    assert spectrum.diagnostics.departure_from_normality == pytest.approx(
        0.0, abs=_NORMAL_ABS
    )


def test_koopman_spectrum_without_diagnostics_remains_valid() -> None:
    """Manual constructors stay backward compatible (diagnostics optional)."""
    eigenvalues = torch.tensor([0.5 + 0.0j, 0.1 + 0.0j])
    spectrum = KoopmanSpectrum(
        eigenvalues=eigenvalues,
        eigenvectors=torch.eye(2, dtype=torch.complex128),
        magnitudes=eigenvalues.abs(),
        growth_rates=torch.log(eigenvalues.abs()),
        frequencies=torch.zeros(2),
        time_step=1.0,
    )
    assert spectrum.diagnostics is None


def test_singular_eigenvectors_yield_infinite_condition() -> None:
    """A zero column in V cannot be normalized; departure is still finite."""
    matrix = torch.eye(2, dtype=torch.float64)
    vectors = torch.tensor(
        [[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.0 + 0.0j]],
    )
    eigenvalues = torch.tensor([1.0 + 0.0j, 0.5 + 0.0j])
    diag = compute_spectral_diagnostics(
        matrix,
        vectors,
        eigenvalues=eigenvalues,
        time_step=1.0,
    )
    assert math.isinf(diag.eigenvector_condition)
    assert torch.isinf(diag.eigenvalue_sensitivities).all()
    assert math.isfinite(diag.departure_from_normality)
    assert diag.nyquist_frequency == pytest.approx(
        0.5, rel=_ORACLE_REL, abs=_ORACLE_ABS
    )
    assert diag.aliasing_warning.tolist() == [False, False]
    assert diag.sign_of_eigenvalue.tolist() == pytest.approx([1.0, 1.0])


def test_negative_real_eigenvalue_flags_nyquist_aliasing() -> None:
    """Diagonal λ=-1 is at Nyquist; discrete assembly warns once.

    Independent oracle: frequency = arg(λ)/(2π Δt) = 1/(2 Δt).
    """
    time_step = 0.25
    expected_nyquist = 1.0 / (2.0 * time_step)
    operator = torch.diag(torch.tensor([-1.0, 0.5], dtype=torch.float64))
    with pytest.warns(UserWarning, match="Nyquist"):
        spectrum = compute_spectrum(operator, time_step=time_step)
    diag = spectrum.diagnostics
    assert diag is not None
    assert diag.nyquist_frequency == pytest.approx(
        expected_nyquist, rel=_ORACLE_REL, abs=_ORACLE_ABS
    )
    # |λ| sort: 1 then 0.5.
    assert diag.aliasing_warning.tolist() == [True, False]
    assert diag.sign_of_eigenvalue.tolist() == pytest.approx([-1.0, 1.0])
    assert spectrum.frequencies[0].item() == pytest.approx(
        expected_nyquist, rel=_ORACLE_REL, abs=_ORACLE_ABS
    )
    assert spectrum.frequencies[1].item() == pytest.approx(0.0, abs=_ORACLE_ABS)


def test_opposite_real_signs_share_log_magnitude_growth() -> None:
    """λ=+0.9 and λ=-0.9 share log|λ| rates; only the negative-real flags."""
    operator = torch.diag(torch.tensor([0.9, -0.9], dtype=torch.float64))
    with pytest.warns(UserWarning, match="Nyquist"):
        spectrum = compute_spectrum(operator, time_step=1.0)
    diag = spectrum.diagnostics
    assert diag is not None
    expected_rate = math.log(0.9)
    assert spectrum.growth_rates.tolist() == pytest.approx(
        [expected_rate, expected_rate],
        rel=_ORACLE_REL,
        abs=_ORACLE_ABS,
    )
    real = spectrum.eigenvalues.real
    positive = real > 0
    negative = real < 0
    assert int(positive.sum().item()) == 1
    assert int(negative.sum().item()) == 1
    assert not bool(diag.aliasing_warning[positive].item())
    assert bool(diag.aliasing_warning[negative].item())
    assert diag.sign_of_eigenvalue[positive].item() == pytest.approx(1.0)
    assert diag.sign_of_eigenvalue[negative].item() == pytest.approx(-1.0)
    assert diag.nyquist_frequency == pytest.approx(
        0.5, rel=_ORACLE_REL, abs=_ORACLE_ABS
    )


def test_generator_spectrum_has_no_nyquist() -> None:
    """Generator placeholder Δt is not a sampling interval."""
    generator = torch.diag(torch.tensor([-1.0, -2.0], dtype=torch.float64))
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        spectrum = compute_generator_spectrum(generator)
    diag = spectrum.diagnostics
    assert diag is not None
    assert diag.nyquist_frequency is None
    assert diag.aliasing_warning.tolist() == [False, False]
    assert diag.sign_of_eigenvalue.tolist() == pytest.approx([-1.0, -1.0])


def test_frequency_units_and_related_sampling_citation() -> None:
    """Nyquist docs use cycles per unit time; Zeng is related literature."""
    doc = SpectralDiagnostics.__doc__
    assert doc is not None
    assert "cycles per unit time" in doc
    assert "Zeng2022Sampling" in doc
    assert "does not implement that identification theorem" in doc
    assert ALIASING_ARG_ATOL == 1e-3


def test_compute_spectral_diagnostics_validates_eigenvalues_and_time_step() -> None:
    """Helper rejects mismatched eigenvalues and non-positive Δt."""
    matrix = torch.eye(2, dtype=torch.float64)
    vectors = torch.eye(2, dtype=torch.complex128)
    with pytest.raises(ValueError, match="eigenvalues must have shape"):
        compute_spectral_diagnostics(
            matrix,
            vectors,
            eigenvalues=torch.ones(3, dtype=torch.float64),
            time_step=1.0,
        )
    with pytest.raises(ValueError, match="time_step must be a finite value > 0"):
        compute_spectral_diagnostics(
            matrix,
            vectors,
            eigenvalues=torch.ones(2, dtype=torch.float64),
            time_step=0.0,
        )


def _spectrum_from_vectors(vectors: torch.Tensor) -> KoopmanSpectrum:
    """Build a spectrum with prescribed eigenvectors and no diagnostics."""
    n = int(vectors.shape[0])
    eigenvalues = torch.ones(n, dtype=torch.complex128)
    return KoopmanSpectrum(
        eigenvalues=eigenvalues,
        eigenvectors=vectors.to(dtype=torch.complex128),
        magnitudes=eigenvalues.abs(),
        growth_rates=torch.zeros(n, dtype=torch.float64),
        frequencies=torch.zeros(n, dtype=torch.float64),
        time_step=1.0,
    )


def test_well_conditioned_amplitudes_match_linalg_solve() -> None:
    """Va=z stays torch.linalg.solve on a well-conditioned identity basis.

    Independent oracle: the same solve on the stored V. Float64 residuals.
    """
    spectrum = compute_spectrum(torch.eye(3, dtype=torch.float64), time_step=1.0)
    states = torch.tensor(
        [[1.0, -0.5, 0.25], [0.0, 2.0, -1.0]],
        dtype=torch.float64,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        amplitudes = spectrum.mode_amplitudes(states)
    vectors = spectrum.eigenvectors
    oracle = torch.linalg.solve(
        vectors,
        states.to(dtype=vectors.dtype).reshape(-1, 3).T,
    ).T.reshape(states.shape)
    assert torch.allclose(amplitudes, oracle, rtol=_ORACLE_REL, atol=_ORACLE_ABS)


def test_moderate_nonnormal_shear_does_not_warn_on_amplitudes() -> None:
    """The 2306 a=10 shear has κ(V) well below CONDITION_WARN."""
    spectrum = compute_spectrum(_nonnormal_operator(), time_step=1.0)
    states = torch.ones(2, dtype=torch.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        spectrum.mode_amplitudes(states)


def test_ill_conditioned_eigenvectors_warn_then_solve() -> None:
    """Nearly parallel columns exceed CONDITION_WARN but remain solvable."""
    scale = 1e-8
    vectors = torch.tensor(
        [[1.0 + 0.0j, 1.0 + 0.0j], [0.0 + 0.0j, scale]],
        dtype=torch.complex128,
    )
    spectrum = _spectrum_from_vectors(vectors)
    states = torch.tensor([1.0, 0.5], dtype=torch.float64)
    with pytest.warns(UserWarning, match="CONDITION_WARN"):
        amplitudes = spectrum.mode_amplitudes(states)
    oracle = torch.linalg.solve(
        vectors,
        states.to(dtype=vectors.dtype).reshape(-1, 2).T,
    ).T.reshape(states.shape)
    assert torch.allclose(amplitudes, oracle, rtol=_ORACLE_REL, atol=_ORACLE_ABS)


def test_singular_eigenvectors_raise_defective_spectrum_error() -> None:
    """A zero column cannot be inverted; the error hints at a Schur subspace."""
    vectors = torch.tensor(
        [[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.0 + 0.0j]],
    )
    spectrum = _spectrum_from_vectors(vectors)
    with pytest.raises(DefectiveSpectrumError, match="Schur") as exc_info:
        spectrum.mode_amplitudes(torch.ones(2, dtype=torch.float64))
    assert isinstance(exc_info.value, torch.linalg.LinAlgError)
    assert isinstance(exc_info.value, RuntimeError)


def test_infinite_stored_condition_raises_before_solve() -> None:
    """Stored κ(V)=+∞ raises even if diagnostics were attached separately."""
    vectors = torch.tensor(
        [[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.0 + 0.0j]],
    )
    diag = compute_spectral_diagnostics(
        torch.eye(2, dtype=torch.float64),
        vectors,
        eigenvalues=torch.ones(2, dtype=torch.complex128),
        time_step=1.0,
    )
    assert math.isinf(diag.eigenvector_condition)
    spectrum = replace(_spectrum_from_vectors(vectors), diagnostics=diag)
    with pytest.raises(DefectiveSpectrumError, match="Schur"):
        spectrum.mode_amplitudes(torch.ones(2, dtype=torch.float64))
