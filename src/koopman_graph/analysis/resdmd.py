"""ResDMD-faithful residual bounds on a fixed dictionary.

Honesty contract
----------------
This module implements Colbrook–Townsend residual dynamic mode decomposition
(ResDMD) on a **fixed** finite dictionary supplied by the caller (identity,
EDMD polynomial / RBF lift, or frozen encodings). Residuals use Galerkin
assemblies for :math:`\\mathcal{K}` and :math:`\\mathcal{K}^*\\mathcal{K}`
and, in the large-data limit, approximate the infinite-dimensional Koopman
residual. This MVP does **not** claim certified infinite-dimensional
pseudospectra or spectral measures.

This API is **not**
:func:`~koopman_graph.analysis.spectral_residuals`, which is a held-out
diagnostic in a **learned** observable norm. Do not rename or treat
``spectral_residuals`` as ResDMD.

References
----------
Colbrook, M. J. and Townsend, A. (2023/2024). Rigorous data-driven computation
of spectral properties of Koopman operators for dynamical systems.
*Communications on Pure and Applied Mathematics*, 77(1), 221–283.
https://doi.org/10.1002/cpa.22125 (``ColbrookTownsend2023ResDMD``)

Colbrook, M. J., Ayton, L. J., and Szőke, M. (2023). Residual dynamic mode
decomposition: robust and verified Koopmanism. *Journal of Fluid Mechanics*,
955, A21. https://doi.org/10.1017/jfm.2022.1052
(``Colbrook2023ResidualDMD``)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from koopman_graph.analysis._galerkin import (
    GalerkinGramMatrices,
    assemble_edmd_matrix,
    assemble_galerkin_grams,
)

_EPS = 1e-12

__all__ = [
    "ResDMDReport",
    "resdmd",
]


@dataclass(frozen=True)
class ResDMDReport:
    """Per-mode ResDMD residual bounds on a fixed dictionary.

    Attributes
    ----------
    eigenvalues : Tensor
        Complex eigenvalues with shape ``(k,)``, sorted by descending
        magnitude.
    residuals : Tensor
        Non-negative real Colbrook–Townsend residuals with shape ``(k,)``.
    eigenvectors : Tensor
        Dictionary coefficient eigenvectors with shape ``(m, k)`` (complex),
        columns aligned with ``eigenvalues``.
    num_pairs : int
        Number of consecutive transition pairs used to form the Grams.
    tolerance : float
        Trustworthiness threshold applied by :meth:`trustworthy_mask`.
    """

    eigenvalues: Tensor
    residuals: Tensor
    eigenvectors: Tensor
    num_pairs: int
    tolerance: float

    def trustworthy_mask(self) -> Tensor:
        """Return a boolean mask of modes with residual at most ``tolerance``.

        Returns
        -------
        Tensor
            Boolean tensor with shape ``(k,)``.
        """
        return self.residuals <= self.tolerance


def resdmd(
    psi0: Tensor,
    psi1: Tensor,
    *,
    tolerance: float = 1e-2,
    regularization: float = 0.0,
    num_modes: int | None = None,
) -> ResDMDReport:
    """Compute ResDMD residuals for EDMD eigenpairs on a fixed dictionary.

    Honesty contract
    ----------------
    Callers supply pre-lifted observables ``psi0`` / ``psi1`` on a **fixed**
    dictionary. The residual for eigenpair :math:`(\\lambda, g)` is the
    Colbrook–Townsend finite-dictionary Galerkin residual

    .. math::

        \\mathrm{res}(\\lambda, g)^2 =
        \\frac{g^{H}
        (G_{11} - \\lambda G_{01}^{H} - \\bar{\\lambda} G_{01}
        + |\\lambda|^{2} G_{00}) g}
        {g^{H} G_{00} g}

    with unnormalized sample Grams from
    :mod:`koopman_graph.analysis._galerkin`. This is **not**
    :func:`~koopman_graph.analysis.spectral_residuals`.

    Parameters
    ----------
    psi0 : Tensor
        Observables at time ``t`` with shape ``(num_pairs, m)``.
    psi1 : Tensor
        Observables at time ``t + 1`` with the same shape as ``psi0``.
    tolerance : float, optional
        Threshold for :meth:`ResDMDReport.trustworthy_mask`. Default
        ``1e-2``.
    regularization : float, optional
        Non-negative Tikhonov strength used only when forming the EDMD
        matrix ``A = (G00 + εI)⁺ G01``. Residual Grams remain
        unregularized. Default ``0.0``.
    num_modes : int or None, optional
        If set, keep the ``num_modes`` largest-magnitude eigenpairs after
        sorting. ``None`` keeps all ``m`` modes. Default ``None``.

    Returns
    -------
    ResDMDReport
        Magnitude-sorted eigenvalues, residuals, and dictionary
        eigenvectors.

    Raises
    ------
    ValueError
        If pair shapes are invalid, ``tolerance < 0``,
        ``regularization < 0``, or ``num_modes`` is outside ``1..m``.
    """
    if tolerance < 0.0:
        msg = f"tolerance must be ≥ 0, got {tolerance}"
        raise ValueError(msg)

    grams = assemble_galerkin_grams(psi0, psi1, regularization=0.0)
    edmd = assemble_edmd_matrix(grams, regularization=regularization)
    eigenvalues, eigenvectors = torch.linalg.eig(edmd.to(torch.complex128))

    order = torch.argsort(eigenvalues.abs(), descending=True)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    dict_dim = int(eigenvalues.shape[0])
    if num_modes is not None:
        if num_modes < 1 or num_modes > dict_dim:
            msg = f"num_modes must lie in 1..{dict_dim} when set; got {num_modes}"
            raise ValueError(msg)
        eigenvalues = eigenvalues[:num_modes]
        eigenvectors = eigenvectors[:, :num_modes]

    residuals = torch.stack(
        [
            _eigenpair_residual(grams, lam, eigenvectors[:, idx])
            for idx, lam in enumerate(eigenvalues)
        ]
    )

    return ResDMDReport(
        eigenvalues=eigenvalues,
        residuals=residuals,
        eigenvectors=eigenvectors,
        num_pairs=grams.num_pairs,
        tolerance=float(tolerance),
    )


def _eigenpair_residual(
    grams: GalerkinGramMatrices,
    eigenvalue: Tensor,
    coefficient: Tensor,
) -> Tensor:
    """Colbrook–Townsend residual for one ``(λ, g)`` pair (scalar tensor).

    Parameters
    ----------
    grams
        Value for ``grams``.
    eigenvalue
        Value for ``eigenvalue``.
    coefficient
        Value for ``coefficient``.

    Returns
    -------
    object
        Function result.
    """
    g = coefficient.reshape(-1).to(torch.complex128)
    lam = eigenvalue.to(torch.complex128)
    g00 = grams.g00.to(torch.complex128)
    g01 = grams.g01.to(torch.complex128)
    g11 = grams.g11.to(torch.complex128)

    # M = G11 − λ G01ᴴ − λ̄ G01 + |λ|² G00
    matrix = g11 - lam * g01.mH - torch.conj(lam) * g01 + (lam.abs() ** 2) * g00
    numerator = torch.real(torch.conj(g) @ (matrix @ g))
    denominator = torch.real(torch.conj(g) @ (g00 @ g))
    if float(denominator) <= _EPS:
        msg = "eigenvector has vanishing G00-norm; cannot form ResDMD residual"
        raise ValueError(msg)
    residual_sq = numerator / denominator
    # Numerical noise can yield tiny negatives.
    residual_sq = torch.clamp(residual_sq, min=0.0)
    return torch.sqrt(residual_sq).to(dtype=torch.float64)
