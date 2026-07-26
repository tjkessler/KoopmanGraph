"""Block-diagonal approximate inverse helpers for networked operators.

Used by :class:`~koopman_graph.operators.GraphKoopmanOperator` and
:class:`~koopman_graph.operators.HypergraphKoopmanOperator` when
``sparsity="block_diagonal"``. Forward advance stays the sparse (or
hyperedge) message-passing matvec; this module only approximates
``inverse_advance`` with per-node ``d×d`` solves (one Jacobi step). Exact
whole-network inversion remains the ``sparsity="dense"`` path.
"""

from __future__ import annotations

import torch
from torch import Tensor

from koopman_graph.graph_utils.topology import (
    hyperedge_normalized_adjacency_matvec,
    symmetric_normalized_adjacency_matvec,
)


def _invert_square(matrix: Tensor) -> Tensor:
    """Invert a square matrix, falling back to a pseudoinverse if singular.

    Parameters
    ----------

    matrix : Tensor
        See the function signature / summary for ``matrix``.

    Returns
    -------

    Tensor
        See summary line."""
    try:
        return torch.linalg.inv(matrix)
    except RuntimeError:
        return torch.linalg.pinv(matrix)


def apply_self_inverse(
    rhs: Tensor,
    *,
    k_self: Tensor | None = None,
    k_self_blocks: Tensor | None = None,
) -> Tensor:
    """Apply per-node self-term inverses to row-stacked latents.

    Solves ``Z @ K.T = R`` for each node (shared ``K`` or per-node blocks).

    Parameters
    ----------
    rhs : Tensor
        Right-hand side with shape ``(num_nodes, latent_dim)``.
    k_self : Tensor or None, optional
        Shared self matrix with shape ``(latent_dim, latent_dim)``.
    k_self_blocks : Tensor or None, optional
        Per-node self matrices with shape
        ``(num_nodes, latent_dim, latent_dim)``.

    Returns
    -------
    Tensor
        Recovered latents with the same shape as ``rhs``.

    Raises
    ------
    ValueError
        If neither or both of ``k_self`` / ``k_self_blocks`` are provided, or
        shapes are inconsistent.
    """
    if (k_self is None) == (k_self_blocks is None):
        msg = "provide exactly one of k_self or k_self_blocks"
        raise ValueError(msg)
    if rhs.ndim != 2:
        msg = f"rhs must have shape (num_nodes, latent_dim), got {tuple(rhs.shape)}"
        raise ValueError(msg)

    if k_self is not None:
        if k_self.shape != (rhs.shape[-1], rhs.shape[-1]):
            msg = (
                f"k_self must have shape ({rhs.shape[-1]}, {rhs.shape[-1]}), "
                f"got {tuple(k_self.shape)}"
            )
            raise ValueError(msg)
        return rhs @ _invert_square(k_self).T

    assert k_self_blocks is not None
    num_nodes, latent_dim = rhs.shape
    if k_self_blocks.shape != (num_nodes, latent_dim, latent_dim):
        msg = (
            "k_self_blocks must have shape "
            f"({num_nodes}, {latent_dim}, {latent_dim}), "
            f"got {tuple(k_self_blocks.shape)}"
        )
        raise ValueError(msg)
    # K_i @ z_i.T = r_i.T  ⇒  batched solve on column right-hand sides.
    try:
        return torch.linalg.solve(k_self_blocks, rhs.unsqueeze(-1)).squeeze(-1)
    except RuntimeError:
        return torch.stack(
            [
                rhs[index] @ _invert_square(k_self_blocks[index]).T
                for index in range(num_nodes)
            ]
        )


def block_diagonal_graph_inverse_advance(
    z_adjusted: Tensor,
    *,
    k_self: Tensor,
    k_nbr: Tensor,
    edge_index: Tensor,
    edge_weight: Tensor | None = None,
    k_self_blocks: Tensor | None = None,
) -> Tensor:
    """Approximate networked inverse via one Jacobi / block-diagonal step.

    Forward advance is ``Z_next = Z K_self.T + (Â Z) K_nbr.T`` (plus control,
    already removed into ``z_adjusted``). This helper forms::

        R = Z_adj - (Â Z_adj) K_nbr.T
        Z = R K_self^{-T}

    (or per-node ``K_self`` blocks when ``k_self_blocks`` is set). The step is
    **exact** when ``K_nbr = 0`` or the graph has no edges, and otherwise an
    approximation of the dense ``N·d`` inverse.

    Parameters
    ----------
    z_adjusted : Tensor
        Control-adjusted latents at ``t+1`` with shape
        ``(num_nodes, latent_dim)``.
    k_self : Tensor
        Shared self factor (ignored when ``k_self_blocks`` is provided).
    k_nbr : Tensor
        Neighbor factor with shape ``(latent_dim, latent_dim)``.
    edge_index : Tensor
        Graph topology ``(2, num_edges)``.
    edge_weight : Tensor or None, optional
        Optional edge weights.
    k_self_blocks : Tensor or None, optional
        Optional per-node bilinear self blocks
        ``(num_nodes, latent_dim, latent_dim)``.

    Returns
    -------
    Tensor
        Approximate latents at ``t`` with the same shape as ``z_adjusted``.
    """
    neighbor = symmetric_normalized_adjacency_matvec(
        edge_index,
        z_adjusted,
        edge_weight=edge_weight,
        num_nodes=z_adjusted.shape[0],
    )
    rhs = z_adjusted - neighbor @ k_nbr.T
    if k_self_blocks is None:
        return apply_self_inverse(rhs, k_self=k_self)
    return apply_self_inverse(rhs, k_self_blocks=k_self_blocks)


def block_diagonal_hypergraph_inverse_advance(
    z_adjusted: Tensor,
    *,
    k_self: Tensor,
    k_hedge: Tensor,
    hyperedge_index: Tensor,
    hyperedge_weight: Tensor | None = None,
    k_self_blocks: Tensor | None = None,
) -> Tensor:
    """Approximate hypergraph inverse via one Jacobi / block-diagonal step.

    Forward advance is ``Z_next = Z K_self.T + (Ĥ Z) K_hedge.T`` (plus control,
    already removed into ``z_adjusted``). This helper forms::

        R = Z_adj - (Ĥ Z_adj) K_hedge.T
        Z = R K_self^{-T}

    (or per-node ``K_self`` blocks when ``k_self_blocks`` is set). The step is
    **exact** when ``K_hedge = 0`` or there are no hyperedges, and otherwise an
    approximation of the dense ``N·d`` inverse.

    Parameters
    ----------
    z_adjusted : Tensor
        Control-adjusted latents at ``t+1`` with shape
        ``(num_nodes, latent_dim)``.
    k_self : Tensor
        Shared self factor (ignored when ``k_self_blocks`` is provided).
    k_hedge : Tensor
        Hyperedge coupling factor with shape ``(latent_dim, latent_dim)``.
    hyperedge_index : Tensor
        Bipartite incidence ``(2, nnz)``.
    hyperedge_weight : Tensor or None, optional
        Optional hyperedge weights.
    k_self_blocks : Tensor or None, optional
        Optional per-node bilinear / orbit-tied self blocks
        ``(num_nodes, latent_dim, latent_dim)``.

    Returns
    -------
    Tensor
        Approximate latents at ``t`` with the same shape as ``z_adjusted``.
    """
    coupled = hyperedge_normalized_adjacency_matvec(
        hyperedge_index,
        z_adjusted,
        hyperedge_weight=hyperedge_weight,
        num_nodes=z_adjusted.shape[0],
    )
    rhs = z_adjusted - coupled @ k_hedge.T
    if k_self_blocks is None:
        return apply_self_inverse(rhs, k_self=k_self)
    return apply_self_inverse(rhs, k_self_blocks=k_self_blocks)
