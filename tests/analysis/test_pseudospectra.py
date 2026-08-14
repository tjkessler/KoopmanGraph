"""Finite-matrix resolvent grids and empirical spectral measures (TASK-1835)."""

from __future__ import annotations

import pytest
import torch
from tests.helpers import REPO_ROOT

import koopman_graph as kg
import koopman_graph.analysis as analysis
from koopman_graph.analysis.pseudospectra import (
    EmpiricalSpectralMeasure,
    ResolventNormGrid,
    empirical_spectral_measure,
    resolvent_norm_grid,
)

_ATOL = 1e-5
_MODULE_PATH = REPO_ROOT / "src" / "koopman_graph" / "analysis" / "pseudospectra.py"


def test_resolvent_norm_grid_shapes_and_near_eigenvalue_peak() -> None:
    """Grid shape is (n_imag, n_real); norm peaks near spectrum."""
    operator = torch.diag(torch.tensor([0.5, -0.25], dtype=torch.float64))
    real_grid = torch.linspace(-1.0, 1.0, 9, dtype=torch.float64)
    imag_grid = torch.linspace(-0.5, 0.5, 5, dtype=torch.float64)
    grid = resolvent_norm_grid(operator, real_grid, imag_grid)

    assert isinstance(grid, ResolventNormGrid)
    assert grid.real_grid.shape == (9,)
    assert grid.imag_grid.shape == (5,)
    assert grid.resolvent_norms.shape == (5, 9)
    assert torch.all(grid.resolvent_norms > 0)

    # Index of real≈0.5, imag≈0 (near eigenvalue 0.5).
    j_near = int(torch.argmin((real_grid - 0.5).abs()))
    i_zero = int(torch.argmin(imag_grid.abs()))
    j_far = int(torch.argmin((real_grid - (-1.0)).abs()))
    near = float(grid.resolvent_norms[i_zero, j_near])
    far = float(grid.resolvent_norms[i_zero, j_far])
    assert near > far * 5.0


def test_empirical_spectral_measure_from_operator() -> None:
    """Uniform weights sum to 1; eigenvalues match eigvals."""
    operator = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float64)
    measure = empirical_spectral_measure(operator)
    assert isinstance(measure, EmpiricalSpectralMeasure)
    assert measure.eigenvalues.shape == (2,)
    assert measure.weights.shape == (2,)
    assert float(measure.weights.sum()) == pytest.approx(1.0, abs=_ATOL)
    expected = torch.linalg.eigvals(operator.to(torch.complex128))
    # Unordered match via sorting by real then imag.
    got = measure.eigenvalues[torch.argsort(measure.eigenvalues.real)]
    exp = expected[torch.argsort(expected.real)]
    assert torch.allclose(got, exp, atol=_ATOL)


def test_empirical_spectral_measure_from_eigenvalues() -> None:
    """Caller-supplied eigenvalues get uniform weights."""
    eigs = torch.tensor([1.0 + 0.0j, 0.5 + 0.1j, -0.2 + 0.0j])
    measure = empirical_spectral_measure(eigenvalues=eigs)
    assert torch.equal(measure.eigenvalues, eigs.to(torch.complex128))
    assert torch.allclose(
        measure.weights, torch.full((3,), 1.0 / 3.0, dtype=torch.float64)
    )


def test_resolvent_norm_grid_validation() -> None:
    """Nonsquare operators and empty grids raise."""
    with pytest.raises(ValueError, match="square 2-D"):
        resolvent_norm_grid(torch.zeros(2, 3), torch.zeros(2), torch.zeros(2))
    with pytest.raises(ValueError, match="real_grid"):
        resolvent_norm_grid(torch.eye(2), torch.zeros(0), torch.zeros(2))
    with pytest.raises(ValueError, match="imag_grid"):
        resolvent_norm_grid(torch.eye(2), torch.zeros(2), torch.zeros(2, 2))


def test_empirical_spectral_measure_validation() -> None:
    """Exactly one of operator / eigenvalues required."""
    with pytest.raises(ValueError, match="exactly one"):
        empirical_spectral_measure()
    with pytest.raises(ValueError, match="exactly one"):
        empirical_spectral_measure(torch.eye(2), eigenvalues=torch.ones(2))
    with pytest.raises(ValueError, match="nonempty 1-D"):
        empirical_spectral_measure(eigenvalues=torch.ones(2, 2))


def test_pseudospectra_public_export_not_root() -> None:
    """Analysis-public symbols; not root ``__all__``."""
    names = {
        "ResolventNormGrid",
        "EmpiricalSpectralMeasure",
        "resolvent_norm_grid",
        "empirical_spectral_measure",
    }
    assert names.issubset(set(analysis.__all__))
    assert names.isdisjoint(set(kg.__all__))
    assert analysis.resolvent_norm_grid is resolvent_norm_grid


def test_pseudospectra_honesty_finite_matrix_scope() -> None:
    """Docs bound scope to finite matrix / deny infinite-dimensional claims."""
    source = _MODULE_PATH.read_text(encoding="utf-8")
    combined = source + (resolvent_norm_grid.__doc__ or "")
    assert "finite" in combined.lower()
    assert "infinite-dimensional" in combined.lower() or "infinite" in combined.lower()
    assert "not" in combined.lower()
