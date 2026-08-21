"""Monomial polynomial helpers for discrete graph Koopman operators.

The degree-``P`` networked map (row-state convention
``Z ← Z K^T``) is the Kronecker polynomial

.. math::

    K_{\\mathrm{eff}}
    = \\sum_{k=0}^{P} \\widehat{A}^{k} \\otimes K_{k},
    \\qquad
    K_{0} \\equiv K_{\\mathrm{self}},
    \\quad
    K_{1} \\equiv K_{\\mathrm{nbr}}.

This module assembles that sum and applies the equivalent stacked matvec
``vec(Z_{t+1}) = K_eff vec(Z_t)``. It is **not** a second public operator
class; :class:`~koopman_graph.operators.graph.GraphKoopmanOperator` owns
``filter_degree`` (default ``1``). Dual random-walk is an extra
**direction** (a second shift), not an extra hop: callers pass each
normalized adjacency separately.

Chebyshev bases and the exact Kronecker spectrum reduction
``eig(sum_k λ^k K_k)`` are out of scope here.

References
----------
Guo, Y., Schaller, M., Worthmann, K. & Streif, S. (2025). Modularized
data-driven approximation of the Koopman operator and generator.
*Physica D: Nonlinear Phenomena*, 476, 134651.
https://doi.org/10.1016/j.physd.2025.134651
(``Guo2025ModularEDMD``)

The monomial hop expansion here is **not** Guo et al. modularized EDMD
(subsystem dictionaries and generator-side coupling). That paper is
related networked-EDMD literature only.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import Tensor

__all__ = [
    "apply_monomial_powers",
    "dense_polynomial_kronecker",
    "validate_filter_degree",
]


def validate_filter_degree(filter_degree: object) -> int:
    """Return a non-negative integer hop degree, or raise ``ValueError``.

    Parameters
    ----------
    filter_degree : object
        Requested monomial degree ``P``. Must be a non-boolean ``int``
        with ``P >= 0``.

    Returns
    -------
    int
        Validated degree.

    Raises
    ------
    ValueError
        If ``filter_degree`` is a ``bool``, not an ``int``, or is negative.
    """
    if isinstance(filter_degree, bool) or not isinstance(filter_degree, int):
        msg = (
            "filter_degree must be a non-negative int, "
            f"got {filter_degree!r} ({type(filter_degree).__name__})"
        )
        raise ValueError(msg)
    if filter_degree < 0:
        msg = f"filter_degree must be >= 0, got {filter_degree}"
        raise ValueError(msg)
    return filter_degree


def dense_polynomial_kronecker(
    adjacency: Tensor,
    hop_matrices: Sequence[Tensor],
    *,
    start_power: int = 0,
) -> Tensor:
    """Assemble ``sum_k A^{start_power+k} ⊗ K_k`` for dense factors.

    Parameters
    ----------
    adjacency : Tensor
        Dense shift ``(N, N)`` (normalized adjacency ``Â`` or a dual
        counterpart).
    hop_matrices : sequence of Tensor
        Factors ``K_{start_power}, …`` each with shape ``(d, d)``. Must be
        non-empty.
    start_power : int, optional
        Exponent on ``adjacency`` for ``hop_matrices[0]``. Default ``0``
        (identity Kronecker term). Must be ``>= 0``.

    Returns
    -------
    Tensor
        Dense ``(N·d, N·d)`` Kronecker polynomial. Empty ``hop_matrices``
        is rejected because the latent width ``d`` would be undefined.

    Raises
    ------
    ValueError
        If ``adjacency`` is not square 2-D, ``start_power < 0``,
        ``hop_matrices`` is empty, or a factor is not square ``(d, d)``.
    """
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        msg = (
            f"adjacency must be a square 2-D tensor, got shape {tuple(adjacency.shape)}"
        )
        raise ValueError(msg)
    if start_power < 0:
        msg = f"start_power must be >= 0, got {start_power}"
        raise ValueError(msg)
    if not hop_matrices:
        msg = "hop_matrices must be non-empty"
        raise ValueError(msg)

    latent_dim = hop_matrices[0].shape[0]
    expected = (latent_dim, latent_dim)
    for index, matrix in enumerate(hop_matrices):
        if matrix.shape != expected:
            msg = (
                f"hop_matrices[{index}] must have shape {expected}, "
                f"got {tuple(matrix.shape)}"
            )
            raise ValueError(msg)

    power = torch.linalg.matrix_power(adjacency, start_power)
    total = torch.kron(power, hop_matrices[0])
    for matrix in hop_matrices[1:]:
        power = power @ adjacency
        total = total + torch.kron(power, matrix)
    return total


def apply_monomial_powers(
    states: Tensor,
    hop_matrices: Sequence[Tensor],
    matvec: Callable[[Tensor], Tensor],
    *,
    min_power: int = 1,
) -> Tensor:
    """Apply ``sum_k (A^k Z) K_k^T`` via repeated adjacency matvecs.

    ``hop_matrices[0]`` is the factor at hop ``min_power``. The self term
    (``k = 0``) is **not** applied here.

    Parameters
    ----------
    states : Tensor
        Node latents ``Z`` with shape ``(N, d)``.
    hop_matrices : sequence of Tensor
        Factors ``K_{min_power}, …`` each ``(d, d)``. An empty sequence
        returns zeros like ``states``.
    matvec : callable
        One-hop map ``Z → Â Z`` (or a dual backward shift).
    min_power : int, optional
        Hop count for ``hop_matrices[0]``. Default ``1``. Must be
        ``>= 1``.

    Returns
    -------
    Tensor
        Neighbor-style contribution with the same shape as ``states``.

    Raises
    ------
    ValueError
        If ``min_power < 1`` or a factor shape does not match ``d``.
    """
    if min_power < 1:
        msg = f"min_power must be >= 1 (self term is not a matvec), got {min_power}"
        raise ValueError(msg)
    if not hop_matrices:
        return torch.zeros_like(states)

    latent_dim = states.shape[-1]
    expected = (latent_dim, latent_dim)
    for index, matrix in enumerate(hop_matrices):
        if matrix.shape != expected:
            msg = (
                f"hop_matrices[{index}] must have shape {expected}, "
                f"got {tuple(matrix.shape)}"
            )
            raise ValueError(msg)

    term = torch.zeros_like(states)
    powered = states
    n_factors = len(hop_matrices)
    max_power = min_power + n_factors - 1
    for power in range(1, max_power + 1):
        powered = matvec(powered)
        index = power - min_power
        if 0 <= index < n_factors:
            term = term + powered @ hop_matrices[index].T
    return term
