"""Private Galerkin assemblies and ResDMD report (TASK-1833 / TASK-1834 / TASK-1836).

TASK-1833: fixed-dictionary Gram / ``𝒦*𝒦`` helpers in private
``analysis._galerkin``. TASK-1834: public ``ResDMDReport`` / ``resdmd``.
TASK-1836: dictionary-identity residual oracle vs analytic reference.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import torch

import koopman_graph as kg
import koopman_graph.analysis as analysis
from koopman_graph.analysis._galerkin import (
    GalerkinGramMatrices,
    assemble_edmd_matrix,
    assemble_galerkin_grams,
    assemble_k_star_k_galerkin,
    transition_observables,
)
from koopman_graph.analysis.resdmd import (
    ResDMDReport,
    _eigenpair_residual,
    resdmd,
)
from koopman_graph.analysis.residuals import spectral_residuals

_ATOL = 1e-5
# Float64 exact Grams (G00 = I); analytic ‖(A − λI)g‖/‖g‖ match.
_ORACLE_ATOL = 1e-10
_GALERKIN_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "koopman_graph"
    / "analysis"
    / "_galerkin.py"
)


def _linear_trajectory(matrix: torch.Tensor, *, length: int, seed: int) -> torch.Tensor:
    """Identity-dictionary trajectory with ``x_{t+1} = x_t @ matrix``."""
    torch.manual_seed(seed)
    dim = matrix.shape[0]
    x0 = torch.randn(dim, dtype=matrix.dtype)
    rows = [x0]
    for _ in range(length - 1):
        rows.append(rows[-1] @ matrix)
    return torch.stack(rows)


def test_transition_observables_splits_trajectory() -> None:
    """Consecutive pairs are ``psi[:-1]`` / ``psi[1:]``."""
    psi = torch.arange(12, dtype=torch.float64).reshape(4, 3)
    psi0, psi1 = transition_observables(psi)
    assert torch.equal(psi0, psi[:3])
    assert torch.equal(psi1, psi[1:])


def test_assemble_galerkin_grams_matches_hand_built() -> None:
    """Sample Grams match explicit ``Ψᴴ Ψ`` products."""
    a = torch.tensor([[0.8, 0.1], [-0.2, 0.7]], dtype=torch.float64)
    psi = _linear_trajectory(a, length=8, seed=1)
    psi0, psi1 = transition_observables(psi)
    grams = assemble_galerkin_grams(psi0, psi1)
    assert grams.num_pairs == 7
    assert torch.allclose(grams.g00, psi0.mH @ psi0, atol=_ATOL)
    assert torch.allclose(grams.g01, psi0.mH @ psi1, atol=_ATOL)
    assert torch.allclose(grams.g11, psi1.mH @ psi1, atol=_ATOL)


def test_edmd_and_k_star_k_on_identity_dictionary() -> None:
    """Exact linear row map: EDMD recovers ``A``; ``𝒦*𝒦`` matches ``Aᴴ G00 A``."""
    a = torch.tensor([[0.6, 0.2], [0.0, 0.5]], dtype=torch.float64)
    psi = _linear_trajectory(a, length=10, seed=2)
    psi0, psi1 = transition_observables(psi)
    # Row convention: x_{t+1} = x_t @ A  ⇒  Williams EDMD matrix is A.
    assert torch.allclose(psi1, psi0 @ a, atol=_ATOL)

    grams = assemble_galerkin_grams(psi0, psi1)
    edmd = assemble_edmd_matrix(grams)
    k_star_k = assemble_k_star_k_galerkin(grams)

    assert torch.allclose(edmd, a, atol=_ATOL)
    expected = edmd.mH @ grams.g00 @ edmd
    assert torch.allclose(k_star_k, expected, atol=_ATOL)


def test_complex_rotation_edmd_smoke() -> None:
    """Complex eigen-rotation dictionary recovers the row map."""
    theta = 0.3
    a = torch.tensor(
        [
            [torch.cos(torch.tensor(theta)), -torch.sin(torch.tensor(theta))],
            [torch.sin(torch.tensor(theta)), torch.cos(torch.tensor(theta))],
        ],
        dtype=torch.complex128,
    )
    # Scale slightly inside the unit disk.
    a = (0.9 + 0.0j) * a
    torch.manual_seed(3)
    x0 = torch.randn(2, dtype=torch.complex128)
    rows = [x0]
    for _ in range(7):
        rows.append(rows[-1] @ a)
    psi = torch.stack(rows)
    psi0, psi1 = transition_observables(psi)
    grams = assemble_galerkin_grams(psi0, psi1)
    edmd = assemble_edmd_matrix(grams)
    assert torch.allclose(edmd, a, atol=1e-8)


def test_transition_observables_rejects_short_or_bad_rank() -> None:
    """Validation for trajectory splits."""
    with pytest.raises(ValueError, match="2-D"):
        transition_observables(torch.zeros(3))
    with pytest.raises(ValueError, match="at least two timesteps"):
        transition_observables(torch.zeros(1, 2))


def test_assemble_grams_rejects_mismatched_or_empty() -> None:
    """Pair shape / emptiness guards."""
    with pytest.raises(ValueError, match="share shape"):
        assemble_galerkin_grams(torch.zeros(3, 2), torch.zeros(3, 1))
    with pytest.raises(ValueError, match="at least one transition"):
        assemble_galerkin_grams(torch.zeros(0, 2), torch.zeros(0, 2))
    with pytest.raises(ValueError, match="regularization"):
        assemble_galerkin_grams(
            torch.zeros(2, 2), torch.zeros(2, 2), regularization=-1.0
        )
    with pytest.raises(ValueError, match="2-D"):
        assemble_galerkin_grams(torch.zeros(3), torch.zeros(3))
    with pytest.raises(ValueError, match="dictionary dimension"):
        assemble_galerkin_grams(torch.zeros(2, 0), torch.zeros(2, 0))


def test_assemble_galerkin_grams_applies_regularization() -> None:
    """Positive regularization adds Tikhonov mass to G00."""
    psi0 = torch.eye(2, dtype=torch.float64)
    psi1 = torch.diag(torch.tensor([0.5, 0.2], dtype=torch.float64))
    grams = assemble_galerkin_grams(psi0, psi1, regularization=1e-2)
    expected = psi0.mH @ psi0 + 1e-2 * torch.eye(2, dtype=torch.float64)
    assert torch.allclose(grams.g00, expected, atol=_ATOL)


def test_assemble_edmd_matrix_handles_singular_g00_and_extra_reg() -> None:
    """Rank-deficient G00 uses pinv; extra regularization is applied."""
    psi0 = torch.ones(5, 2, dtype=torch.float64)
    psi1 = 0.5 * psi0
    grams = assemble_galerkin_grams(psi0, psi1)
    edmd = assemble_edmd_matrix(grams)
    assert edmd.shape == (2, 2)
    edmd_reg = assemble_edmd_matrix(grams, regularization=1e-2)
    assert edmd_reg.shape == (2, 2)
    assert not torch.allclose(edmd, edmd_reg, atol=1e-8)


def test_galerkin_module_has_no_residuals_coupling() -> None:
    """Private Galerkin helpers must not import learned-residual APIs."""
    source = _GALERKIN_PATH.read_text(encoding="utf-8")
    assert "from koopman_graph.analysis.residuals" not in source
    assert "import koopman_graph.analysis.residuals" not in source
    assert "ModeShapeModel" not in source
    # Disallow callable coupling; docstring may mention spectral_residuals
    # for honesty disambiguation only.
    assert "spectral_residuals(" not in source


def test_galerkin_symbols_not_in_public_all() -> None:
    """Leading-underscore module is not a public analysis / root export."""
    private_names = {
        "GalerkinGramMatrices",
        "assemble_galerkin_grams",
        "assemble_edmd_matrix",
        "assemble_k_star_k_galerkin",
        "transition_observables",
    }
    assert private_names.isdisjoint(set(analysis.__all__))
    assert private_names.isdisjoint(set(kg.__all__))
    assert not hasattr(analysis, "assemble_galerkin_grams")


def test_galerkin_gram_matrices_is_frozen() -> None:
    """Result type is a frozen dataclass (package convention)."""
    grams = assemble_galerkin_grams(torch.eye(2), torch.eye(2))
    assert isinstance(grams, GalerkinGramMatrices)
    with pytest.raises(FrozenInstanceError):
        grams.num_pairs = 0  # type: ignore[misc]


def test_dictionary_identity_residual_matches_analytic_reference() -> None:
    """Dictionary-identity oracle: Colbrook residual matches closed form.

    Fixture (TASK-1836 / DESIGN §2.3)
    ---------------------------------
    * Dictionary: identity ``ψ(x) = x``.
    * Hand-built pairs ``psi0 = I₂``, ``psi1 = A`` with
      ``A = diag(0.5, 0.2)`` (row map ``x ↦ x @ A``).
    * Then ``G00 = I``, ``G01 = A``, ``G11 = Aᵀ A``, so

          res(λ, g) = ‖(A − λ I) g‖₂ / ‖g‖₂

    Reference cases
    ---------------
    * ``λ = 0``, ``g = [1, 0]ᵀ`` → analytic residual ``0.5``.
    * ``λ = 0.5``, ``g = [1, 0]ᵀ`` (true eigenpair) → analytic residual ``0``.

    Tolerance ``1e-10`` is justified by float64 exact Grams with ``G00 = I``.
    """
    a = torch.diag(torch.tensor([0.5, 0.2], dtype=torch.float64))
    psi0 = torch.eye(2, dtype=torch.float64)
    psi1 = a.clone()
    grams = assemble_galerkin_grams(psi0, psi1)
    eye = torch.eye(2, dtype=torch.float64)
    assert torch.allclose(grams.g00, eye, atol=_ORACLE_ATOL)
    assert torch.allclose(grams.g01, a, atol=_ORACLE_ATOL)

    g = torch.tensor([1.0, 0.0], dtype=torch.complex128)
    residual_wrong = _eigenpair_residual(grams, torch.tensor(0.0 + 0.0j), g)
    residual_true = _eigenpair_residual(grams, torch.tensor(0.5 + 0.0j), g)

    assert float(residual_wrong) == pytest.approx(0.5, abs=_ORACLE_ATOL)
    assert float(residual_true) == pytest.approx(0.0, abs=_ORACLE_ATOL)

    # Same fixture: EDMD eigenpairs of the identity dictionary are exact.
    report = resdmd(psi0, psi1, tolerance=1e-8)
    assert torch.all(report.residuals < _ORACLE_ATOL)


def test_resdmd_exact_linear_map_near_zero_residuals() -> None:
    """Identity dictionary on exact linear dynamics: ResDMD residuals ≈ 0."""
    a = torch.tensor([[0.7, 0.1], [0.0, 0.55]], dtype=torch.float64)
    psi = _linear_trajectory(a, length=12, seed=4)
    psi0, psi1 = transition_observables(psi)
    report = resdmd(psi0, psi1, tolerance=1e-3)
    assert isinstance(report, ResDMDReport)
    assert report.num_pairs == 11
    assert report.residuals.shape == (2,)
    assert torch.all(report.residuals < 1e-5)
    assert torch.all(report.trustworthy_mask())

    true_eigs = torch.linalg.eigvals(a.to(torch.complex128))
    true_sorted = true_eigs[torch.argsort(true_eigs.abs(), descending=True)]
    assert torch.allclose(report.eigenvalues, true_sorted, atol=_ATOL)


def test_resdmd_wrong_eigenpair_has_large_residual() -> None:
    """Wrong λ / random coefficient is not certified (vacuity contrast)."""
    a = torch.tensor([[0.65, 0.15], [-0.1, 0.5]], dtype=torch.float64)
    psi = _linear_trajectory(a, length=12, seed=5)
    psi0, psi1 = transition_observables(psi)
    grams = assemble_galerkin_grams(psi0, psi1)
    report = resdmd(psi0, psi1)
    assert float(report.residuals.max()) < 1e-5

    bad = _eigenpair_residual(
        grams,
        torch.tensor(2.0 + 0.0j),
        torch.tensor([1.0, 0.0], dtype=torch.complex128),
    )
    assert float(bad) > 0.5


def test_resdmd_num_modes_and_tolerance() -> None:
    """``num_modes`` truncates; ``trustworthy_mask`` uses ``tolerance``."""
    a = torch.tensor([[0.9, 0.0], [0.0, 0.2]], dtype=torch.float64)
    psi = _linear_trajectory(a, length=10, seed=6)
    psi0, psi1 = transition_observables(psi)
    report = resdmd(psi0, psi1, num_modes=1, tolerance=1e-8)
    assert report.eigenvalues.shape == (1,)
    assert report.eigenvectors.shape == (2, 1)
    assert report.residuals.shape == (1,)
    # Exact linear map → residual below tight tolerance.
    assert torch.all(report.trustworthy_mask())

    with pytest.raises(ValueError, match="num_modes"):
        resdmd(psi0, psi1, num_modes=0)
    with pytest.raises(ValueError, match="tolerance"):
        resdmd(psi0, psi1, tolerance=-0.1)


def test_resdmd_public_export_not_root() -> None:
    """ResDMD is analysis-public; not a root ``__all__`` symbol."""
    assert "ResDMDReport" in analysis.__all__
    assert "resdmd" in analysis.__all__
    assert analysis.resdmd is resdmd
    assert "ResDMDReport" not in kg.__all__
    assert "resdmd" not in kg.__all__


def test_resdmd_honesty_distinct_from_spectral_residuals() -> None:
    """Docs keep ResDMD and spectral_residuals as separate APIs."""
    doc = (resdmd.__doc__ or "") + (analysis.resdmd.__doc__ or "")
    module_doc = Path(
        Path(__file__).resolve().parents[1]
        / "src"
        / "koopman_graph"
        / "analysis"
        / "resdmd.py"
    ).read_text(encoding="utf-8")
    combined = doc + module_doc
    assert "ResDMD" in combined or "resdmd" in combined.lower()
    assert "spectral_residuals" in combined
    assert "not" in combined.lower()
    # Learned diagnostic remains importable and separate.
    assert callable(spectral_residuals)
    assert "spectral_residuals" in analysis.__all__
    assert "from koopman_graph.analysis.residuals" not in module_doc
