"""Private fixed-dictionary Galerkin assemblies for ResDMD / pseudospectra.

Private module (leading underscore). Not part of
:mod:`koopman_graph.analysis` public ``__all__``. Callers supply pre-lifted
observables on a **fixed** dictionary (identity, EDMD polynomial, or frozen
encodings). This is operator-math scaffolding for Colbrook–Townsend ResDMD
and finite-dictionary pseudospectra — **not**
:func:`~koopman_graph.analysis.spectral_residuals` (learned-observable
diagnostic) and **not** trainer DDP / ``[distributed]`` extras.

Sample convention
-----------------
Row-vector pairs ``psi0``, ``psi1`` with shape ``(num_pairs, m)``, matching
:func:`~koopman_graph.baselines.base.fit_row_operator`. Sample Grams use the
conjugate transpose::

    G00 = Ψ₀ᴴ Ψ₀,   G01 = Ψ₀ᴴ Ψ₁,   G11 = Ψ₁ᴴ Ψ₁

Unnormalized (scale cancels in ResDMD residual ratios). The EDMD column map
is ``A = (G00 + εI)⁺ G01``. The Galerkin matrix of ``𝒦*𝒦`` is
``G01ᴴ (G00 + εI)⁺ G01`` (= ``Aᴴ G00_reg A`` when ``A`` uses the same
regularized inverse).

Notes
-----
When ``G00 + εI`` is well-conditioned, solves use
:func:`torch.linalg.solve`. Rank-deficient dictionaries fall back to
:func:`torch.linalg.pinv` so tiny / collinear dictionaries remain usable
for tests and MVP assemblies.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = [
    "GalerkinGramMatrices",
    "assemble_edmd_matrix",
    "assemble_galerkin_grams",
    "assemble_k_star_k_galerkin",
    "transition_observables",
]


@dataclass(frozen=True)
class GalerkinGramMatrices:
    """Sample Galerkin Grams on a fixed dictionary.

    Attributes
    ----------
    g00 : Tensor
        ``Ψ₀ᴴ Ψ₀`` with shape ``(m, m)``.
    g01 : Tensor
        ``Ψ₀ᴴ Ψ₁`` with shape ``(m, m)``.
    g11 : Tensor
        ``Ψ₁ᴴ Ψ₁`` with shape ``(m, m)``.
    num_pairs : int
        Number of consecutive observable pairs used to form the Grams.
    """

    g00: Tensor
    g01: Tensor
    g11: Tensor
    num_pairs: int


def transition_observables(psi: Tensor) -> tuple[Tensor, Tensor]:
    """Split a trajectory lift into consecutive observable pairs.

    Parameters
    ----------
    psi : Tensor
        Fixed-dictionary observables with shape ``(T, m)``, ``T ≥ 2``.

    Returns
    -------
    psi0, psi1 : Tensor
        ``psi[:-1]`` and ``psi[1:]``, each with shape ``(T - 1, m)``.

    Raises
    ------
    ValueError
        If ``psi`` is not 2-D or has fewer than two rows.
    """
    if psi.ndim != 2:
        msg = f"psi must be 2-D with shape (T, m), got ndim={psi.ndim}"
        raise ValueError(msg)
    if psi.shape[0] < 2:
        msg = (
            "psi must contain at least two timesteps for transition pairs; "
            f"got T={psi.shape[0]}"
        )
        raise ValueError(msg)
    return psi[:-1], psi[1:]


def assemble_galerkin_grams(
    psi0: Tensor,
    psi1: Tensor,
    *,
    regularization: float = 0.0,
) -> GalerkinGramMatrices:
    """Assemble sample Galerkin Grams from consecutive observables.

    Parameters
    ----------
    psi0 : Tensor
        Observables at time ``t`` with shape ``(num_pairs, m)``.
    psi1 : Tensor
        Observables at time ``t + 1`` with the same shape as ``psi0``.
    regularization : float, optional
        Non-negative Tikhonov strength added to the returned ``g00`` only
        (``g00 ← g00 + ε I``). Default ``0.0``. Use ``0.0`` when Grams
        feed residual formulas that also involve ``g01`` / ``g11``; prefer
        passing ``regularization`` to :func:`assemble_edmd_matrix` /
        :func:`assemble_k_star_k_galerkin` for solve-time Tikhonov.

    Returns
    -------
    GalerkinGramMatrices
        Unnormalized sample Grams ``g00``, ``g01``, ``g11``.

    Raises
    ------
    ValueError
        If shapes mismatch, ``num_pairs < 1``, or ``regularization < 0``.
    """
    _validate_pair_shapes(psi0, psi1)
    _validate_regularization(regularization)
    g00 = psi0.mH @ psi0
    g01 = psi0.mH @ psi1
    g11 = psi1.mH @ psi1
    if regularization > 0.0:
        eye = torch.eye(g00.shape[0], dtype=g00.dtype, device=g00.device)
        g00 = g00 + regularization * eye
    return GalerkinGramMatrices(
        g00=g00,
        g01=g01,
        g11=g11,
        num_pairs=int(psi0.shape[0]),
    )


def assemble_edmd_matrix(
    grams: GalerkinGramMatrices,
    *,
    regularization: float = 0.0,
) -> Tensor:
    """EDMD matrix ``A = (G00 + εI)⁺ G01`` with ``Ψ₁ ≈ Ψ₀ A``.

    For row-sample observables this is the Williams / Colbrook dictionary
    map (not the column-acting ``K`` returned by
    :func:`~koopman_graph.baselines.base.fit_row_operator`, which equals
    ``A.mT`` when the fit is exact).

    Parameters
    ----------
    grams : GalerkinGramMatrices
        Sample Grams from :func:`assemble_galerkin_grams`.
    regularization : float, optional
        Additional Tikhonov on ``g00`` (on top of any already baked into
        ``grams.g00``). Default ``0.0``.

    Returns
    -------
    Tensor
        Square EDMD matrix ``A`` with shape ``(m, m)``.

    Raises
    ------
    ValueError
        If ``regularization < 0``.
    """
    g00_reg = _regularized_g00(grams.g00, regularization)
    return _apply_g00_inverse(g00_reg, grams.g01)


def assemble_k_star_k_galerkin(
    grams: GalerkinGramMatrices,
    *,
    regularization: float = 0.0,
) -> Tensor:
    """Galerkin matrix of ``𝒦*𝒦``: ``G01ᴴ (G00 + εI)⁺ G01``.

    Equals ``Aᴴ G00_reg A`` when ``A = assemble_edmd_matrix(...)`` uses the
    same regularized inverse.

    Parameters
    ----------
    grams : GalerkinGramMatrices
        Sample Grams from :func:`assemble_galerkin_grams`.
    regularization : float, optional
        Additional Tikhonov on ``g00``. Default ``0.0``.

    Returns
    -------
    Tensor
        Hermitian ``(m, m)`` Galerkin ``𝒦*𝒦`` matrix.

    Raises
    ------
    ValueError
        If ``regularization < 0``.
    """
    g00_reg = _regularized_g00(grams.g00, regularization)
    # Solve G00 X = G01  ⇒  X = G00⁺ G01; then G01ᴴ X.
    g00_inv_g01 = _apply_g00_inverse(g00_reg, grams.g01)
    return grams.g01.mH @ g00_inv_g01


def _validate_pair_shapes(psi0: Tensor, psi1: Tensor) -> None:
    """Internal helper: validate pair shapes.

    Parameters
    ----------
    psi0
        Value for ``psi0``.
    psi1
        Value for ``psi1``.
    """
    if psi0.ndim != 2 or psi1.ndim != 2:
        msg = (
            "psi0 and psi1 must be 2-D with shape (num_pairs, m); "
            f"got ndim {psi0.ndim} and {psi1.ndim}"
        )
        raise ValueError(msg)
    if psi0.shape != psi1.shape:
        msg = (
            "psi0 and psi1 must share shape (num_pairs, m); "
            f"got {tuple(psi0.shape)} and {tuple(psi1.shape)}"
        )
        raise ValueError(msg)
    if psi0.shape[0] < 1:
        msg = (
            "psi0/psi1 must contain at least one transition pair; "
            f"got num_pairs={psi0.shape[0]}"
        )
        raise ValueError(msg)
    if psi0.shape[1] < 1:
        msg = f"dictionary dimension m must be ≥ 1; got m={psi0.shape[1]}"
        raise ValueError(msg)


def _validate_regularization(regularization: float) -> None:
    """Internal helper: validate regularization.

    Parameters
    ----------
    regularization
        Value for ``regularization``.
    """
    if regularization < 0.0:
        msg = f"regularization must be ≥ 0, got {regularization}"
        raise ValueError(msg)


def _regularized_g00(g00: Tensor, regularization: float) -> Tensor:
    """Internal helper: regularized g00.

    Parameters
    ----------
    g00
        Value for ``g00``.
    regularization
        Value for ``regularization``.

    Returns
    -------
    object
        Function result.
    """
    _validate_regularization(regularization)
    if regularization <= 0.0:
        return g00
    eye = torch.eye(g00.shape[0], dtype=g00.dtype, device=g00.device)
    return g00 + regularization * eye


def _apply_g00_inverse(g00: Tensor, rhs: Tensor) -> Tensor:
    """Compute ``G00⁺ rhs``, preferring ``solve`` when well-conditioned.

    Parameters
    ----------
    g00
        Value for ``g00``.
    rhs
        Value for ``rhs``.

    Returns
    -------
    object
        Function result.
    """
    try:
        return torch.linalg.solve(g00, rhs)
    except RuntimeError:
        return torch.linalg.pinv(g00) @ rhs
