"""Topology-blind VAMP-2 score precursor (Wu & Noé / VAMPnets).

Computes the empirical mean-free VAMP-2 score

.. math::

    \\hat{R}_2 = \\bigl\\|
        C_{00}^{-1/2} C_{01} C_{11}^{-1/2}
    \\bigr\\|_F^2

on time-lagged feature matrices. This is a **topology-blind** precursor for
use as an optional training term on flattened encoder latents. It is **not**
GraphVAMPnets, a molecular-dynamics / Markov-state-model toolchain, or a
deeptime interop layer — those belong to the 0.11 roadmap.

References
----------
Wu, H. & Noé, F. Variational approach for learning Markov processes from
time series data. *J. Nonlinear Sci.* 30, 23–66 (2020).
https://doi.org/10.1007/s00332-019-09567-y (``Wu2020VAMP``)
Mardt, A., Pasquali, L., Wu, H. & Noé, F. VAMPnets for deep learning of
molecular kinetics. *Nat. Commun.* 9, 5 (2018).
https://doi.org/10.1038/s41467-017-02388-1 (``Mardt2018VAMPnets``)
"""

from __future__ import annotations

import torch
from torch import Tensor


def _symmetric_inverse_sqrt(matrix: Tensor, *, epsilon: float) -> Tensor:
    """Return a symmetric inverse square-root with ridge regularization.

    Parameters
    ----------
    matrix : Tensor
        Symmetric PSD candidate ``(f, f)``.
    epsilon : float
        Ridge added to the diagonal before eigendecomposition.

    Returns
    -------
    Tensor
        ``matrix^{-1/2}`` with shape ``(f, f)``.
    """
    dim = matrix.shape[0]
    ridge = matrix + epsilon * torch.eye(dim, dtype=matrix.dtype, device=matrix.device)
    eigenvalues, eigenvectors = torch.linalg.eigh(ridge)
    inv_sqrt = eigenvalues.clamp_min(epsilon).rsqrt()
    return (eigenvectors * inv_sqrt.unsqueeze(0)) @ eigenvectors.mT


def vamp2_score(
    x: Tensor,
    y: Tensor,
    *,
    epsilon: float = 1e-6,
) -> Tensor:
    """Compute the mean-free empirical VAMP-2 score on lag pairs.

    Feature matrices are column-mean-centered. Covariances use the ``1/n``
    normalization common in VAMPnet implementations:

    ``C00 = (X^T X) / n``, ``C01 = (X^T Y) / n``, ``C11 = (Y^T Y) / n``.

    Parameters
    ----------
    x : Tensor
        Instantaneous features with shape ``(n_samples, n_features)``.
    y : Tensor
        Time-lagged features with the same shape as ``x``.
    epsilon : float, optional
        Ridge on ``C00`` / ``C11`` and eigenvalue floor for the inverse
        square-root. Default ``1e-6``.

    Returns
    -------
    Tensor
        Scalar VAMP-2 score (standard Frobenius norm squared).

    Raises
    ------
    ValueError
        If shapes are incompatible, ``n_samples < 2``, or ``epsilon <= 0``.
    """
    if epsilon <= 0.0:
        msg = f"epsilon must be positive, got {epsilon}"
        raise ValueError(msg)
    if x.ndim != 2 or y.ndim != 2:
        msg = (
            f"x and y must be 2D (n_samples, n_features), "
            f"got shapes {tuple(x.shape)} and {tuple(y.shape)}"
        )
        raise ValueError(msg)
    if x.shape != y.shape:
        msg = f"x and y must share shape, got {tuple(x.shape)} and {tuple(y.shape)}"
        raise ValueError(msg)
    n_samples, n_features = x.shape
    if n_samples < 2:
        msg = f"need at least 2 lag samples for VAMP-2, got {n_samples}"
        raise ValueError(msg)
    if n_features < 1:
        msg = "n_features must be positive"
        raise ValueError(msg)

    x_centered = x - x.mean(dim=0, keepdim=True)
    y_centered = y - y.mean(dim=0, keepdim=True)
    scale = 1.0 / float(n_samples)
    c00 = scale * (x_centered.mT @ x_centered)
    c01 = scale * (x_centered.mT @ y_centered)
    c11 = scale * (y_centered.mT @ y_centered)
    c00_inv_sqrt = _symmetric_inverse_sqrt(c00, epsilon=epsilon)
    c11_inv_sqrt = _symmetric_inverse_sqrt(c11, epsilon=epsilon)
    koopman = c00_inv_sqrt @ c01 @ c11_inv_sqrt
    return torch.sum(koopman * koopman)


def vamp2_loss(
    x: Tensor,
    y: Tensor,
    *,
    epsilon: float = 1e-6,
) -> Tensor:
    """Return ``-vamp2_score`` so gradient descent maximizes VAMP-2.

    Parameters
    ----------
    x : Tensor
        Instantaneous features ``(n_samples, n_features)``.
    y : Tensor
        Time-lagged features with the same shape as ``x``.
    epsilon : float, optional
        Ridge forwarded to :func:`vamp2_score`. Default ``1e-6``.

    Returns
    -------
    Tensor
        Scalar training loss ``-R̂₂``.
    """
    return -vamp2_score(x, y, epsilon=epsilon)
