"""Measure-preserving EDMD (mpEDMD) via a Gram-weighted Procrustes polar factor.

Colbrook (2023) replaces the unconstrained EDMD map with the orthogonal
Procrustes solution in the dictionary Gram inner product, so the fitted
matrix is similar to a unitary matrix. Convergence claims apply to
measure-preserving systems in that Gram geometry. This baseline does
**not** certify Euclidean spectral conditioning of a general directed
:math:`K_{\\mathrm{eff}}`.

References
----------
Colbrook, M. J. (2023). The mpEDMD algorithm for data-driven computations
of measure-preserving dynamical systems. *SIAM Journal on Numerical
Analysis*, 61(3), 1585–1608. https://doi.org/10.1137/22M1521407
(``Colbrook2023mpEDMD``)
"""

from __future__ import annotations

import torch
from torch import Tensor

from koopman_graph.analysis._galerkin import assemble_galerkin_grams
from koopman_graph.baselines.edmd import EDMDBaseline

__all__ = [
    "MpEDMDBaseline",
    "fit_mpedmd_row_operator",
]

_GRAM_EIG_RTOL = 1e-10


def _hermitian_sqrt_factors(
    gram: Tensor,
    *,
    rank: int | None,
) -> tuple[Tensor, Tensor]:
    """Return truncated Hermitian square-root and pseudoinverse square-root.

    Parameters
    ----------
    gram : Tensor
        Hermitian Gram ``G`` with shape ``(m, m)``.
    rank : int or None
        If set, keep the largest ``rank`` eigenpairs that pass the relative
        cutoff. ``None`` keeps every eigenpair above the cutoff.

    Returns
    -------
    sqrt, inv_sqrt : Tensor
        ``G^{1/2}`` and ``G^{+/2}`` on the retained subspace.

    Raises
    ------
    ValueError
        If no eigenvalue exceeds the relative cutoff.
    """
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    eigenvalues = eigenvalues.real.clamp(min=0.0)
    peak = float(eigenvalues.max().item())
    cutoff = _GRAM_EIG_RTOL * max(peak, 1.0)
    mask = eigenvalues > cutoff
    if rank is not None:
        keep = torch.zeros_like(mask)
        n_keep = min(int(rank), int(eigenvalues.numel()))
        if n_keep > 0:
            keep[-n_keep:] = True
        mask = mask & keep
    if not bool(mask.any()):
        msg = (
            f"mpEDMD Gram square-root is empty (rank={rank!r}, max eigenvalue={peak!r})"
        )
        raise ValueError(msg)
    sqrt_vals = torch.zeros_like(eigenvalues)
    inv_sqrt_vals = torch.zeros_like(eigenvalues)
    sqrt_vals[mask] = eigenvalues[mask].sqrt()
    inv_sqrt_vals[mask] = eigenvalues[mask].rsqrt()
    basis = eigenvectors.to(dtype=gram.dtype)
    diag_dtype = basis.dtype
    sqrt = basis @ torch.diag(sqrt_vals.to(dtype=diag_dtype)) @ basis.mH
    inv_sqrt = basis @ torch.diag(inv_sqrt_vals.to(dtype=diag_dtype)) @ basis.mH
    return sqrt, inv_sqrt


def fit_mpedmd_row_operator(
    left: Tensor,
    right: Tensor,
    rank: int | None,
) -> Tensor:
    """Fit a Gram-weighted mpEDMD operator in the package row convention.

    Forms Galerkin Grams ``G=Ψ₀ᴴΨ₀`` and ``A=Ψ₀ᴴΨ₁`` (Colbrook
    Algorithm 4.1 with ``W = I``), takes the polar factor of
    ``G^{+/2} A G^{+/2}`` (equivalent to the paper's SVD of
    ``G^{-1/2} A^{\\ast} G^{-1/2}``), and returns ``K`` so that
    ``psi_next = psi @ K.T``. Eigenvalues of ``K`` lie on the unit
    circle when ``G`` is full rank (``K`` is similar to a unitary
    matrix). This is **not** a Euclidean certificate on a general
    directed :math:`K_{\\mathrm{eff}}`.

    Parameters
    ----------
    left : Tensor
        Dictionary snapshots at time ``t`` with shape ``(num_pairs, m)``.
    right : Tensor
        Dictionary snapshots at time ``t + 1`` with the same shape.
    rank : int or None
        Optional Gram truncation (largest eigenpairs of ``G``). ``None``
        keeps every eigenpair above a relative cutoff of ``1e-10``.

    Returns
    -------
    Tensor
        Row-convention mpEDMD matrix with shape ``(m, m)``.

    Raises
    ------
    ValueError
        If pair shapes are invalid or the truncated Gram is empty.

    References
    ----------
    Colbrook, M. J. (2023). The mpEDMD algorithm for data-driven
    computations of measure-preserving dynamical systems. *SIAM Journal
    on Numerical Analysis*, 61(3), 1585–1608.
    https://doi.org/10.1137/22M1521407 (``Colbrook2023mpEDMD``)
    """
    grams = assemble_galerkin_grams(left, right)
    sqrt, inv_sqrt = _hermitian_sqrt_factors(grams.g00, rank=rank)
    whitened = inv_sqrt @ grams.g01 @ inv_sqrt
    left_factors, _, right_factors = torch.linalg.svd(whitened, full_matrices=False)
    polar = left_factors @ right_factors
    column_map = inv_sqrt @ polar @ sqrt
    return column_map.mT.real.to(dtype=left.dtype)


class MpEDMDBaseline(EDMDBaseline):
    """Measure-preserving EDMD baseline (Gram-weighted Procrustes polar factor).

    Same topology-blind dictionaries and Data-only ``predict`` surface as
    :class:`~koopman_graph.baselines.EDMDBaseline`. After the lift, the
    observable map is the Colbrook mpEDMD polar factor in the dictionary
    Gram inner product rather than unconstrained least squares. Fitted
    eigenvalues lie on the unit circle when the Gram is full rank.
    Convergence claims apply to measure-preserving systems in that Gram
    geometry — **not** a Euclidean spectral-conditioning certificate for
    a general directed graph :math:`K_{\\mathrm{eff}}`.

    Notes
    -----
    Constructor arguments match
    :class:`~koopman_graph.baselines.EDMDBaseline` (``time_step``,
    ``rank``, and the same dictionary knobs). Here ``rank`` truncates
    dictionary Gram eigenpairs (Colbrook Algorithm 4.1), not the
    snapshot SVD used by unconstrained EDMD least squares.
    """

    def _fit_observable_operator(self, left: Tensor, right: Tensor) -> Tensor:
        """Fit the mpEDMD polar factor on consecutive dictionary snapshots.

        Parameters
        ----------
        left : Tensor
            Observables at time ``t`` with shape ``(num_pairs, m)``.
        right : Tensor
            Observables at time ``t + 1`` with the same shape as ``left``.

        Returns
        -------
        Tensor
            Operator ``K`` with shape ``(m, m)`` for ``psi_next = psi @ K.T``.
        """
        return fit_mpedmd_row_operator(left, right, rank=self.selected_rank)
