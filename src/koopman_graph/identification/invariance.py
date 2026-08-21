"""Finite-sample subspace invariance proximity (projection leakage).

Let :math:`\\psi_t` be encoded snapshots and :math:`P` the orthogonal
projector onto a numerical basis of those encodings (truncated SVD).
The leakage

.. math::

    \\eta
    =
    \\frac{\\mathbb{E}\\,\\|(I-P)K\\psi\\|_{2}}
          {\\mathbb{E}\\,\\|K\\psi\\|_{2}}

is a dimensionless finite-sample proximity of the encoded cloud to a
:math:`K`-invariant subspace of :math:`\\mathbb{R}^{d}`. It is **not**
the Haseli–Cortés invariance-proximity certificate (principal angles /
worst-case relative error on a function subspace), **not**
:class:`~koopman_graph.losses.ForwardConsistencyLoss`, and **not**
:func:`~koopman_graph.analysis.spectral_residuals`.

This module must not import :mod:`koopman_graph.training` or
:mod:`koopman_graph.model`.

References
----------
Haseli, M. & Cortés, J. (2023). Invariance proximity: closed-form
error bounds for finite-dimensional Koopman-based models.
arXiv:2311.13033.
(``HaseliCortes2023``)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = [
    "SINGULAR_VALUE_REL_CUTOFF",
    "SubspaceInvarianceReport",
    "subspace_invariance_report",
]

SINGULAR_VALUE_REL_CUTOFF = 1e-8
_MIN_HELDOUT_TIMES = 4


@dataclass(frozen=True)
class SubspaceInvarianceReport:
    """Finite-sample projection leakage for a per-node latent map.

    ``leakage`` is dimensionless (a ratio of Euclidean means). It is
    **not** a Haseli–Cortés certificate.

    Attributes
    ----------
    leakage : float
        :math:`\\eta` on the selected time window.
    n_samples : int
        Number of node×time rows in that window.
    rank : int
        Numerical rank of the encoding basis (truncated SVD).
    held_out : bool
        Whether the last half of the time axis was used.
    """

    leakage: float
    n_samples: int
    rank: int
    held_out: bool

    def __post_init__(self) -> None:
        """Validate leakage, counts, and rank.

        Raises
        ------
        ValueError
            If ``leakage`` is non-finite or negative, or if counts are
            invalid.
        """
        if isinstance(self.leakage, bool) or not isinstance(self.leakage, (int, float)):
            msg = f"leakage must be a finite float, got {type(self.leakage).__name__}"
            raise ValueError(msg)
        if not math.isfinite(float(self.leakage)) or self.leakage < 0.0:
            msg = f"leakage must be a finite non-negative float, got {self.leakage!r}"
            raise ValueError(msg)
        if isinstance(self.n_samples, bool) or not isinstance(self.n_samples, int):
            msg = (
                f"n_samples must be a positive int, got {type(self.n_samples).__name__}"
            )
            raise ValueError(msg)
        if self.n_samples < 1:
            msg = f"n_samples must be >= 1, got {self.n_samples}"
            raise ValueError(msg)
        if isinstance(self.rank, bool) or not isinstance(self.rank, int):
            msg = f"rank must be a positive int, got {type(self.rank).__name__}"
            raise ValueError(msg)
        if self.rank < 1:
            msg = f"rank must be >= 1, got {self.rank}"
            raise ValueError(msg)
        if type(self.held_out) is not bool:
            msg = f"held_out must be a bool, got {type(self.held_out).__name__}"
            raise ValueError(msg)


def _as_time_major(encodings: Tensor) -> Tensor:
    """Require time-major encodings with trailing latent width.

    Parameters
    ----------
    encodings : Tensor
        ``(T, d)`` or ``(T, N, d)``.

    Returns
    -------
    Tensor
        The input tensor if the layout is valid.

    Raises
    ------
    ValueError
        If the tensor is not 2-D or 3-D.
    """
    if encodings.ndim not in {2, 3}:
        msg = (
            "encodings must have shape (T, d) or (T, N, d), "
            f"got {tuple(encodings.shape)}"
        )
        raise ValueError(msg)
    return encodings


def _select_window(encodings: Tensor, held_out: bool) -> Tensor:
    """Optionally keep the last half of the time axis.

    Parameters
    ----------
    encodings : Tensor
        Time-major encodings.
    held_out : bool
        When ``True``, keep ``encodings[T // 2 :]``.

    Returns
    -------
    Tensor
        Selected time window.

    Raises
    ------
    ValueError
        If ``held_out`` is True and ``T < 4``.
    """
    n_times = encodings.shape[0]
    if not held_out:
        if n_times < 1:
            msg = "encodings must contain at least one snapshot"
            raise ValueError(msg)
        return encodings
    if n_times < _MIN_HELDOUT_TIMES:
        msg = (
            "held_out=True requires at least "
            f"{_MIN_HELDOUT_TIMES} snapshots, got {n_times}"
        )
        raise ValueError(msg)
    return encodings[n_times // 2 :]


def _orthonormal_basis(rows: Tensor) -> Tensor:
    """Orthonormal columns spanning the row space of ``rows``.

    Parameters
    ----------
    rows : Tensor
        Samples with shape ``(n_samples, d)``.

    Returns
    -------
    Tensor
        Basis ``Q`` with shape ``(d, rank)``.

    Raises
    ------
    ValueError
        If every singular value is below the relative cutoff.
    """
    _, singular_values, vh = torch.linalg.svd(rows, full_matrices=False)
    peak = float(singular_values[0].item()) if singular_values.numel() else 0.0
    if not math.isfinite(peak) or peak <= 0.0:
        msg = "encoding basis is degenerate (no positive singular value)"
        raise ValueError(msg)
    cutoff = SINGULAR_VALUE_REL_CUTOFF * peak
    keep = singular_values > cutoff
    rank = int(keep.sum().item())
    if rank < 1:
        msg = (
            "encoding basis rank is 0 after relative SVD cutoff "
            f"{SINGULAR_VALUE_REL_CUTOFF:g}"
        )
        raise ValueError(msg)
    return vh[:rank].T


def subspace_invariance_report(
    encodings: Tensor,
    matrix: Tensor,
    *,
    held_out: bool = True,
) -> SubspaceInvarianceReport:
    """Compute projection leakage :math:`\\eta` for a dense per-node map.

    Encodings are samples in :math:`\\mathbb{R}^{d}` (node×time rows).
    ``matrix`` is the package row-convention operator ``K`` so
    ``K ψ = ψ @ K.T``. ``P`` is the orthogonal projector onto the
    truncated-SVD basis of the selected encodings. ``held_out=True``
    (default) builds ``P`` and the expectation on the last half of the
    time axis (same window: is that cloud approximately ``K``-invariant?).

    Parameters
    ----------
    encodings : Tensor
        Time-major latents ``(T, d)`` or ``(T, N, d)``.
    matrix : Tensor
        Dense ``K`` with shape ``(d, d)``.
    held_out : bool, optional
        Use the last half of the time axis. Default is ``True``.
        Requires ``T >= 4``.

    Returns
    -------
    SubspaceInvarianceReport
        Leakage, sample count, numerical rank, and ``held_out`` flag.

    Raises
    ------
    ValueError
        If layouts disagree, the hold-out window is too short, the
        basis is degenerate, or :math:`\\mathbb{E}\\|K\\psi\\|_{2}` is
        zero.
    TypeError
        If ``held_out`` is not a ``bool``.
    """
    if type(held_out) is not bool:
        msg = f"held_out must be a bool, got {type(held_out).__name__}"
        raise TypeError(msg)
    encodings = _as_time_major(encodings)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        msg = f"matrix must be square 2-D, got shape {tuple(matrix.shape)}"
        raise ValueError(msg)
    latent_dim = encodings.shape[-1]
    if matrix.shape[0] != latent_dim:
        msg = (
            "matrix trailing width must match encodings, "
            f"got K {tuple(matrix.shape)} vs latent_dim {latent_dim}"
        )
        raise ValueError(msg)
    window = _select_window(encodings, held_out)
    rows = window.reshape(-1, latent_dim)
    if rows.shape[0] < 1:
        msg = "invariance report requires at least one encoding row"
        raise ValueError(msg)
    basis = _orthonormal_basis(rows)
    mapped = rows @ matrix.T.to(dtype=rows.dtype, device=rows.device)
    projected = (mapped @ basis) @ basis.T
    residual_mean = (mapped - projected).norm(dim=-1).mean()
    mapped_mean = mapped.norm(dim=-1).mean()
    denom = float(mapped_mean.item())
    if not math.isfinite(denom) or denom <= 0.0:
        msg = "E||K ψ||_2 must be positive and finite to form η"
        raise ValueError(msg)
    leakage = float((residual_mean / mapped_mean).item())
    return SubspaceInvarianceReport(
        leakage=leakage,
        n_samples=int(rows.shape[0]),
        rank=int(basis.shape[1]),
        held_out=held_out,
    )
