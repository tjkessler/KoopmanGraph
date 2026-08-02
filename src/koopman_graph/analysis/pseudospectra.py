"""Finite-matrix resolvent-norm grids and empirical spectral measures.

Honesty contract
----------------
Helpers here evaluate the **2-norm resolvent** of a square assembled
operator (EDMD dictionary matrix, dense ``K``, or ``K_eff``) on a complex
plane grid, and form a discrete empirical spectral measure from eigenvalues.
This is a **finite-matrix / finite-dictionary MVP**. It does **not** certify
infinite-dimensional ResDMD pseudospectra or continuous spectral measures
(Colbrook–Townsend). Cost scales as ``O(n_imag · n_real · m³)`` via SVD of
``zI − A`` at each grid point.

References
----------
Colbrook, M. J. and Townsend, A. (2023/2024). Rigorous data-driven computation
of spectral properties of Koopman operators for dynamical systems.
*Communications on Pure and Applied Mathematics*, 77(1), 221–283.
https://doi.org/10.1002/cpa.22125 (``ColbrookTownsend2023ResDMD``) —
context for ResDMD / pseudospectra; this module is the finite-matrix grid
only.

Colbrook, M. J., Ayton, L. J., and Szőke, M. (2023). Residual dynamic mode
decomposition: robust and verified Koopmanism. *Journal of Fluid Mechanics*,
955, A21. https://doi.org/10.1017/jfm.2022.1052
(``Colbrook2023ResidualDMD``)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

_EPS = 1e-12

__all__ = [
    "EmpiricalSpectralMeasure",
    "ResolventNormGrid",
    "empirical_spectral_measure",
    "resolvent_norm_grid",
]


@dataclass(frozen=True)
class ResolventNormGrid:
    """Resolvent 2-norms on a rectangular complex-plane grid.

    Attributes
    ----------
    real_grid : Tensor
        Real parts with shape ``(n_real,)``.
    imag_grid : Tensor
        Imaginary parts with shape ``(n_imag,)``.
    resolvent_norms : Tensor
        ``\\| (z I - A)^{-1} \\|_2`` with shape ``(n_imag, n_real)``, where
        ``z = real_grid[j] + 1j * imag_grid[i]``. Non-negative ``float64``.
    """

    real_grid: Tensor
    imag_grid: Tensor
    resolvent_norms: Tensor


@dataclass(frozen=True)
class EmpiricalSpectralMeasure:
    """Discrete uniform spectral measure on eigenvalues of a finite matrix.

    Attributes
    ----------
    eigenvalues : Tensor
        Complex eigenvalues with shape ``(k,)``.
    weights : Tensor
        Non-negative weights with shape ``(k,)`` summing to ``1``
        (uniform ``1/k``). Not a certified continuous spectral measure.
    """

    eigenvalues: Tensor
    weights: Tensor


def resolvent_norm_grid(
    operator: Tensor,
    real_grid: Tensor,
    imag_grid: Tensor,
) -> ResolventNormGrid:
    """Evaluate ``\\| (z I - A)^{-1} \\|_2`` on a complex-plane grid.

    Honesty contract
    ----------------
    Finite-matrix numerical resolvent norms only. Near eigenvalues the norm
    is large (``σ_min → 0``); values are capped via a floor on
    ``σ_min``. This is **not** an infinite-dimensional ResDMD
    pseudospectrum certificate.

    Parameters
    ----------
    operator : Tensor
        Square assembled operator ``A`` with shape ``(m, m)``.
    real_grid : Tensor
        1-D real parts with shape ``(n_real,)``, ``n_real ≥ 1``.
    imag_grid : Tensor
        1-D imaginary parts with shape ``(n_imag,)``, ``n_imag ≥ 1``.

    Returns
    -------
    ResolventNormGrid
        Grids and resolvent norms with shape ``(n_imag, n_real)``.

    Raises
    ------
    ValueError
        If ``operator`` is not square 2-D or grids are not nonempty 1-D.
    """
    matrix = _require_square_operator(operator)
    real = _require_1d_grid(real_grid, name="real_grid")
    imag = _require_1d_grid(imag_grid, name="imag_grid")

    n_imag = int(imag.shape[0])
    n_real = int(real.shape[0])
    dim = int(matrix.shape[0])
    eye = torch.eye(dim, dtype=torch.complex128, device=matrix.device)
    norms = torch.empty(n_imag, n_real, dtype=torch.float64, device=matrix.device)

    for i in range(n_imag):
        for j in range(n_real):
            z = complex(float(real[j]), float(imag[i]))
            shifted = z * eye - matrix
            sigma_min = torch.linalg.svdvals(shifted).min().real
            sigma_min = torch.clamp(sigma_min, min=_EPS)
            norms[i, j] = 1.0 / sigma_min

    return ResolventNormGrid(
        real_grid=real.to(dtype=torch.float64),
        imag_grid=imag.to(dtype=torch.float64),
        resolvent_norms=norms,
    )


def empirical_spectral_measure(
    operator: Tensor | None = None,
    *,
    eigenvalues: Tensor | None = None,
) -> EmpiricalSpectralMeasure:
    """Build a discrete uniform measure on finite-matrix eigenvalues.

    Honesty contract
    ----------------
    Returns equal weights ``1/k`` on the ``k`` eigenvalues of a square
    assembled operator (or on caller-supplied eigenvalues). This is an
    empirical point measure for the **finite** matrix — **not** a certified
    continuous spectral measure of an infinite-dimensional Koopman operator.

    Parameters
    ----------
    operator : Tensor or None, optional
        Square assembled operator. Mutually exclusive with
        ``eigenvalues``.
    eigenvalues : Tensor or None, optional
        Precomputed complex (or real) eigenvalues with shape ``(k,)``.
        Mutually exclusive with ``operator``.

    Returns
    -------
    EmpiricalSpectralMeasure
        Eigenvalues and uniform weights summing to ``1``.

    Raises
    ------
    ValueError
        If neither or both of ``operator`` / ``eigenvalues`` are given, or
        inputs have invalid shape.
    """
    if (operator is None) == (eigenvalues is None):
        msg = "provide exactly one of operator or eigenvalues"
        raise ValueError(msg)

    if eigenvalues is None:
        assert operator is not None
        matrix = _require_square_operator(operator)
        eigs = torch.linalg.eigvals(matrix)
    else:
        if eigenvalues.ndim != 1 or eigenvalues.numel() < 1:
            msg = (
                "eigenvalues must be a nonempty 1-D tensor; "
                f"got shape {tuple(eigenvalues.shape)}"
            )
            raise ValueError(msg)
        eigs = eigenvalues.to(torch.complex128)

    num = int(eigs.shape[0])
    weights = torch.full((num,), 1.0 / num, dtype=torch.float64, device=eigs.device)
    return EmpiricalSpectralMeasure(eigenvalues=eigs, weights=weights)


def _require_square_operator(operator: Tensor) -> Tensor:
    """Internal helper: require square operator.

    Parameters
    ----------
    operator
        Value for ``operator``.

    Returns
    -------
    object
        Function result.
    """
    if operator.ndim != 2 or operator.shape[0] != operator.shape[1]:
        msg = f"operator must be a square 2-D tensor; got shape {tuple(operator.shape)}"
        raise ValueError(msg)
    if operator.shape[0] < 1:
        msg = "operator dimension m must be ≥ 1"
        raise ValueError(msg)
    return operator.to(torch.complex128)


def _require_1d_grid(grid: Tensor, *, name: str) -> Tensor:
    """Internal helper: require 1d grid.

    Parameters
    ----------
    grid
        Value for ``grid``.
    name
        Value for ``name``.

    Returns
    -------
    object
        Function result.
    """
    if grid.ndim != 1 or grid.numel() < 1:
        msg = f"{name} must be a nonempty 1-D tensor; got shape {tuple(grid.shape)}"
        raise ValueError(msg)
    return grid.to(dtype=torch.float64)
