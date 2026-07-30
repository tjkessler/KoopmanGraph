"""Topology and Laplacian helpers for graph inputs (power-user).

Shared degree / adjacency / Laplacian mathematics and graph-input resolution.
Import from :mod:`koopman_graph.graph_utils` (package re-exports) unless you
need the peer module explicitly.

Includes symmetric normalization (``D^{-1/2} A D^{-1/2}``), directed
random-walk normalization (``D_{\\mathrm{out}}^{-1} A`` /
``D_{\\mathrm{in}}^{-1} A^{\\top}``), and per-relation R-GCN-style
in-degree normalization for multiplex / heterogeneous graphs.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor
from torch_geometric.data import Data, HeteroData

RandomWalkDirection = Literal["forward", "backward"]
RelationNormalization = Literal["rgcn_in_degree", "random_walk"]
RELATION_NORMALIZATION_MODES: frozenset[str] = frozenset(
    {"rgcn_in_degree", "random_walk"}
)

# Match DiffConv / DCRNN-style row normalization (nn.gnn legacy floor).
_RANDOM_WALK_DEGREE_FLOOR = 1e-6

# Ephemeral Zhou Ĥ reuse for static incidence (pointer-keyed; not in checkpoints).
_HyperedgeHatCacheKey = tuple[int, int, int | None, torch.dtype, torch.device]
_hat_cache_key: _HyperedgeHatCacheKey | None = None
_hat_cache: Tensor | None = None


def clear_hyperedge_cache() -> None:
    """Drop the cached dense Zhou hypergraph adjacency ``Ĥ``.

    Call after in-place edits to ``hyperedge_index`` / ``hyperedge_weight``
    storage that do not change tensor ``data_ptr`` values. Ordinary incidence
    swaps (new tensors) invalidate automatically on the next assembly.

    Notes
    -----
    The cache is module-scoped and ephemeral; it is never written to
    ``state_dict``. Caching does **not** remove the dense
    :math:`O(N^2)` representation of ``Ĥ``.
    """
    global _hat_cache_key, _hat_cache
    _hat_cache_key = None
    _hat_cache = None


def _hyperedge_hat_cache_key(
    hyperedge_index: Tensor,
    hyperedge_weight: Tensor | None,
    num_nodes: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> _HyperedgeHatCacheKey:
    """Build a cheap incidence fingerprint for dense ``Ĥ`` reuse.

    Uses tensor storage pointers. Callers that mutate incidence / weight
    storage in place without changing pointers must call
    :func:`clear_hyperedge_cache`.

    Parameters
    ----------
    hyperedge_index : Tensor
        Bipartite incidence ``(2, nnz)``.
    hyperedge_weight : Tensor or None
        Optional hyperedge weights.
    num_nodes : int
        Node count ``N``.
    dtype : torch.dtype
        Floating dtype of ``Ĥ``.
    device : torch.device
        Device of ``Ĥ``.

    Returns
    -------
    tuple
        Cache key ``(N, index_ptr, weight_ptr, dtype, device)``.
    """
    weight_ptr = None if hyperedge_weight is None else hyperedge_weight.data_ptr()
    return (
        num_nodes,
        hyperedge_index.data_ptr(),
        weight_ptr,
        dtype,
        device,
    )


def snapshot_edge_weight(snapshot: Data) -> Tensor | None:
    """Return optional scalar edge weights attached to a snapshot.

    Parameters
    ----------
    snapshot : Data
        Graph snapshot that may carry ``edge_weight``.

    Returns
    -------
    Tensor or None
        Edge weights with shape ``(num_edges,)``, or ``None`` when absent.
    """
    edge_weight = getattr(snapshot, "edge_weight", None)
    if edge_weight is None:
        return None
    return edge_weight


def snapshot_hyperedge_index(snapshot: Data) -> Tensor | None:
    """Return optional hyperedge incidence index attached to a snapshot.

    Parameters
    ----------
    snapshot : Data
        Graph snapshot that may carry ``hyperedge_index``.

    Returns
    -------
    Tensor or None
        Bipartite incidence with shape ``(2, nnz)`` (row 0 = node indices,
        row 1 = hyperedge indices), or ``None`` when absent.
    """
    hyperedge_index = getattr(snapshot, "hyperedge_index", None)
    if hyperedge_index is None:
        return None
    return hyperedge_index


def snapshot_hyperedge_weight(snapshot: Data) -> Tensor | None:
    """Return optional hyperedge weights attached to a snapshot.

    Parameters
    ----------
    snapshot : Data
        Graph snapshot that may carry ``hyperedge_weight``.

    Returns
    -------
    Tensor or None
        Weights with shape ``(num_hyperedges,)``, or ``None`` when absent.
    """
    hyperedge_weight = getattr(snapshot, "hyperedge_weight", None)
    if hyperedge_weight is None:
        return None
    return hyperedge_weight


def dense_hyperedge_normalized_adjacency(
    hyperedge_index: Tensor,
    *,
    num_nodes: int,
    hyperedge_weight: Tensor | None = None,
    dtype: torch.dtype,
    device: torch.device | None = None,
) -> Tensor:
    """Assemble dense Zhou incidence-normalized hypergraph adjacency ``Ĥ``.

    For incidence matrix ``B ∈ {0,1}^{N×M}`` (nodes × hyperedges), hyperedge
    weights ``W_e = diag(w)``, node degrees ``D_v = diag(B w)``, and hyperedge
    degrees ``D_e = diag(Bᵀ 1)`` (hyperedge sizes),

    .. math::

        \\hat{H}
        = D_v^{-1/2} B W_e D_e^{-1} B^{\\top} D_v^{-1/2}.

    Isolated nodes (never appearing in ``hyperedge_index``) receive a zero
    row/column via the same ``d ↦ d^{-1/2}`` with non-finite → ``0`` convention
    used for ``Â``. Singleton hyperedges are retained (degree 1).

    Repeated calls with the same incidence storage pointers, ``num_nodes``,
    dtype, and device reuse an ephemeral dense ``Ĥ`` (see
    :func:`clear_hyperedge_cache`). Caching does not change the dense
    :math:`O(N^2)` representation.

    Parameters
    ----------
    hyperedge_index : Tensor
        Bipartite incidence with shape ``(2, nnz)``: row 0 = node indices,
        row 1 = hyperedge indices (PyG ``HypergraphConv`` convention).
    num_nodes : int
        Number of graph nodes ``N``.
    hyperedge_weight : Tensor or None, optional
        Non-negative hyperedge weights with shape ``(num_hyperedges,)``.
        Defaults to ones.
    dtype : torch.dtype
        Floating dtype for the dense matrix.
    device : torch.device or None, optional
        Device for the computation. Defaults to ``hyperedge_index.device``.

    Returns
    -------
    Tensor
        Dense adjacency with shape ``(num_nodes, num_nodes)``.

    Raises
    ------
    ValueError
        If ``hyperedge_index`` has invalid shape or ``num_nodes`` is invalid.
    """
    global _hat_cache_key, _hat_cache

    if num_nodes < 0:
        msg = f"num_nodes must be >= 0, got {num_nodes}"
        raise ValueError(msg)
    if hyperedge_index.ndim != 2 or hyperedge_index.shape[0] != 2:
        msg = (
            "hyperedge_index must have shape (2, nnz), "
            f"got {tuple(hyperedge_index.shape)}"
        )
        raise ValueError(msg)

    resolved_device = device if device is not None else hyperedge_index.device
    key = _hyperedge_hat_cache_key(
        hyperedge_index,
        hyperedge_weight,
        num_nodes,
        dtype=dtype,
        device=resolved_device,
    )
    if _hat_cache is not None and _hat_cache_key == key:
        return _hat_cache

    if hyperedge_index.numel() == 0 or num_nodes == 0:
        result = torch.zeros(
            (num_nodes, num_nodes),
            dtype=dtype,
            device=resolved_device,
        )
        _hat_cache = result
        _hat_cache_key = key
        return result

    node_idx = hyperedge_index[0].to(device=resolved_device)
    hedge_idx = hyperedge_index[1].to(device=resolved_device)
    num_hyperedges = int(hedge_idx.max().item()) + 1

    incidence = torch.zeros(
        (num_nodes, num_hyperedges),
        dtype=dtype,
        device=resolved_device,
    )
    ones = torch.ones(node_idx.size(0), dtype=dtype, device=resolved_device)
    incidence.index_put_((node_idx, hedge_idx), ones, accumulate=True)
    # Cap multi-entries at 1 for a Boolean incidence (duplicate (node, hedge)
    # pairs should not inflate degree).
    incidence = incidence.clamp(max=1)

    if hyperedge_weight is None:
        weights = torch.ones(num_hyperedges, dtype=dtype, device=resolved_device)
    else:
        weights = hyperedge_weight.to(dtype=dtype, device=resolved_device)
        if weights.ndim != 1 or weights.shape[0] != num_hyperedges:
            msg = (
                "hyperedge_weight must have shape "
                f"(num_hyperedges={num_hyperedges},), "
                f"got {tuple(weights.shape)}"
            )
            raise ValueError(msg)

    node_degree = incidence @ weights
    hyperedge_degree = incidence.sum(dim=0)

    deg_v_inv_sqrt = node_degree.pow(-0.5)
    deg_v_inv_sqrt = torch.where(
        torch.isfinite(deg_v_inv_sqrt),
        deg_v_inv_sqrt,
        torch.zeros_like(deg_v_inv_sqrt),
    )
    deg_e_inv = hyperedge_degree.pow(-1.0)
    deg_e_inv = torch.where(
        torch.isfinite(deg_e_inv),
        deg_e_inv,
        torch.zeros_like(deg_e_inv),
    )

    # Ĥ = D_v^{-1/2} B W_e D_e^{-1} Bᵀ D_v^{-1/2}
    scaled = incidence * weights.unsqueeze(0) * deg_e_inv.unsqueeze(0)
    mid = scaled @ incidence.transpose(0, 1)
    result = deg_v_inv_sqrt.unsqueeze(1) * mid * deg_v_inv_sqrt.unsqueeze(0)
    _hat_cache = result
    _hat_cache_key = key
    return result


# Design / blueprint alias for the Zhou incidence-normalized operator.
hyperedge_normalized_incidence_weights = dense_hyperedge_normalized_adjacency


def hyperedge_normalized_adjacency_matvec(
    hyperedge_index: Tensor,
    x: Tensor,
    *,
    hyperedge_weight: Tensor | None = None,
    num_nodes: int | None = None,
) -> Tensor:
    """Apply Zhou ``Ĥ`` to node features.

    Assembles dense ``Ĥ`` via
    :func:`dense_hyperedge_normalized_adjacency` (shared ephemeral cache) and
    returns ``Ĥ @ x``. Suitable for modest ``N`` (same dense-operator ceiling
    as
    :meth:`~koopman_graph.operators.HypergraphKoopmanOperator.effective_matrix`).

    Parameters
    ----------
    hyperedge_index : Tensor
        Bipartite incidence with shape ``(2, nnz)``.
    x : Tensor
        Node features with shape ``(num_nodes, feature_dim)``.
    hyperedge_weight : Tensor or None, optional
        Optional hyperedge weights with shape ``(num_hyperedges,)``.
    num_nodes : int or None, optional
        Number of nodes. Inferred from ``x`` when omitted.

    Returns
    -------
    Tensor
        Transformed features with the same shape as ``x``.

    Raises
    ------
    ValueError
        If ``x`` is not 2D.
    """
    if x.dim() != 2:
        msg = f"x must be 2D (num_nodes, features), got shape {tuple(x.shape)}"
        raise ValueError(msg)
    node_count = num_nodes if num_nodes is not None else x.size(0)
    hat = dense_hyperedge_normalized_adjacency(
        hyperedge_index,
        num_nodes=node_count,
        hyperedge_weight=hyperedge_weight,
        dtype=x.dtype,
        device=x.device,
    )
    return hat @ x


def node_degrees(
    edge_index: Tensor,
    *,
    num_nodes: int,
    edge_weight: Tensor | None = None,
    dtype: torch.dtype,
    device: torch.device | None = None,
) -> Tensor:
    """Return weighted out-degrees ``d_i = sum_j A_{ij}`` for each node.

    For ``symmetric`` / Laplacian helpers this assumes an undirected,
    symmetrically represented adjacency (each undirected edge appears in both
    directions, or weights already encode that symmetry). Directed graphs
    should use the random-walk helpers
    (:func:`dense_random_walk_normalized_adjacency` /
    :func:`random_walk_normalized_adjacency_matvec`) instead. Duplicate edges
    accumulate.

    Parameters
    ----------
    edge_index : Tensor
        Edge index with shape ``(2, num_edges)``.
    num_nodes : int
        Number of graph nodes.
    edge_weight : Tensor or None, optional
        Non-negative edge weights with shape ``(num_edges,)``. Defaults to ones.
    dtype : torch.dtype
        Floating dtype for the returned degrees.
    device : torch.device or None, optional
        Device for the computation. Defaults to ``edge_index.device``.

    Returns
    -------
    Tensor
        Degrees with shape ``(num_nodes,)``.
    """
    row = edge_index[0]
    resolved_device = device if device is not None else edge_index.device
    if edge_weight is None:
        weights = torch.ones(row.size(0), dtype=dtype, device=resolved_device)
    else:
        weights = edge_weight.to(dtype=dtype, device=resolved_device)
    deg = torch.zeros(num_nodes, dtype=dtype, device=resolved_device)
    deg.index_add_(0, row, weights)
    return deg


def degree_support_mask(
    edge_index: Tensor,
    *,
    num_nodes: int,
    edge_weight: Tensor | None = None,
    dtype: torch.dtype,
    device: torch.device | None = None,
) -> Tensor:
    """Return diagonal entries of ``P = diag(1_{d_i > 0})``.

    ``P`` is the projector onto non-isolated nodes used by the
    pseudoinverse-normalized Laplacian
    ``L_sym = P - Â = (D^+)^{1/2} (D - A) (D^+)^{1/2}``.

    Parameters
    ----------
    edge_index : Tensor
        Edge index with shape ``(2, num_edges)``.
    num_nodes : int
        Number of graph nodes.
    edge_weight : Tensor or None, optional
        Non-negative edge weights with shape ``(num_edges,)``. Defaults to ones.
    dtype : torch.dtype
        Floating dtype for the returned mask (``0`` / ``1`` entries).
    device : torch.device or None, optional
        Device for the computation. Defaults to ``edge_index.device``.

    Returns
    -------
    Tensor
        Support mask with shape ``(num_nodes,)``.
    """
    deg = node_degrees(
        edge_index,
        num_nodes=num_nodes,
        edge_weight=edge_weight,
        dtype=dtype,
        device=device,
    )
    return (deg > 0).to(dtype=dtype)


def symmetric_normalized_adjacency_edge_weights(
    edge_index: Tensor,
    *,
    num_nodes: int,
    edge_weight: Tensor | None = None,
    dtype: torch.dtype,
    device: torch.device | None = None,
) -> Tensor:
    """Return per-edge weights for ``Â = D_0^{-1/2} A D_0^{-1/2}``.

    ``D_0^{-1/2}`` is the entrywise map ``d ↦ d^{-1/2}`` with non-finite values
    (degree 0) replaced by ``0``, i.e. ``(D^+)^{1/2}`` on the diagonal degree
    matrix. This is the shared adjacency core of the symmetric normalized
    Laplacian ``L_sym = P - Â`` (equal to ``I - Â`` when every node has positive
    degree). Sparse physics lifting and dense benchmark diffusion both build on
    these weights; they differ only in sparse matvec vs dense assembly
    (duplicate edges accumulate in both paths).

    Intended for **undirected, symmetrically represented** adjacency (the
    ``adjacency="symmetric"`` networked-operator mode). For directed graphs use
    the random-walk helpers instead.

    Parameters
    ----------
    edge_index : Tensor
        Edge index with shape ``(2, num_edges)``.
    num_nodes : int
        Number of graph nodes.
    edge_weight : Tensor or None, optional
        Non-negative edge weights with shape ``(num_edges,)``. Defaults to ones.
    dtype : torch.dtype
        Floating dtype for the returned weights.
    device : torch.device or None, optional
        Device for the computation. Defaults to ``edge_index.device``.

    Returns
    -------
    Tensor
        Normalized edge weights with shape ``(num_edges,)``.
    """
    row, col = edge_index
    resolved_device = device if device is not None else edge_index.device
    deg = node_degrees(
        edge_index,
        num_nodes=num_nodes,
        edge_weight=edge_weight,
        dtype=dtype,
        device=resolved_device,
    )
    if edge_weight is None:
        weights = torch.ones(row.size(0), dtype=dtype, device=resolved_device)
    else:
        weights = edge_weight.to(dtype=dtype, device=resolved_device)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt = torch.where(
        torch.isfinite(deg_inv_sqrt),
        deg_inv_sqrt,
        torch.zeros_like(deg_inv_sqrt),
    )
    return deg_inv_sqrt[row] * weights * deg_inv_sqrt[col]


def dense_symmetric_normalized_adjacency(
    edge_index: Tensor,
    num_nodes: int,
    *,
    edge_weight: Tensor | None = None,
    dtype: torch.dtype,
) -> Tensor:
    """Assemble dense ``Â = D_0^{-1/2} A D_0^{-1/2}``.

    Parameters
    ----------
    edge_index : Tensor
        Edge index with shape ``(2, num_edges)``.
    num_nodes : int
        Number of graph nodes.
    edge_weight : Tensor or None, optional
        Non-negative edge weights with shape ``(num_edges,)``. Defaults to ones.
    dtype : torch.dtype
        Floating dtype for the dense matrix.

    Returns
    -------
    Tensor
        Dense adjacency with shape ``(num_nodes, num_nodes)``. Duplicate edges
        **accumulate** (sum), matching
        :func:`symmetric_normalized_adjacency_matvec` / ``index_add_``.
    """
    row, col = edge_index
    norm = symmetric_normalized_adjacency_edge_weights(
        edge_index,
        num_nodes=num_nodes,
        edge_weight=edge_weight,
        dtype=dtype,
    )
    adj = torch.zeros((num_nodes, num_nodes), dtype=dtype, device=edge_index.device)
    adj.index_put_((row, col), norm, accumulate=True)
    return adj


def dense_symmetric_normalized_laplacian(
    edge_index: Tensor,
    num_nodes: int,
    *,
    edge_weight: Tensor | None = None,
    dtype: torch.dtype,
) -> Tensor:
    """Assemble dense ``L_sym = P - Â = (D^+)^{1/2} (D - A) (D^+)^{1/2}``.

    On graphs with no isolated nodes, ``P = I`` and this reduces to ``I - Â``.
    Isolated nodes have a zero diagonal entry (not ``1``). Intended for
    undirected, symmetrically represented adjacency; directed graphs should use
    random-walk normalization rather than ``L_sym``.

    Parameters
    ----------
    edge_index : Tensor
        Edge index with shape ``(2, num_edges)``.
    num_nodes : int
        Number of graph nodes.
    edge_weight : Tensor or None, optional
        Non-negative edge weights with shape ``(num_edges,)``. Defaults to ones.
    dtype : torch.dtype
        Floating dtype for the dense matrix.

    Returns
    -------
    Tensor
        Dense Laplacian with shape ``(num_nodes, num_nodes)``.
    """
    adj = dense_symmetric_normalized_adjacency(
        edge_index,
        num_nodes,
        edge_weight=edge_weight,
        dtype=dtype,
    )
    support = degree_support_mask(
        edge_index,
        num_nodes=num_nodes,
        edge_weight=edge_weight,
        dtype=dtype,
        device=edge_index.device,
    )
    return torch.diag(support) - adj


def symmetric_normalized_adjacency_matvec(
    edge_index: Tensor,
    x: Tensor,
    *,
    edge_weight: Tensor | None = None,
    num_nodes: int | None = None,
) -> Tensor:
    """Apply ``Â = D_0^{-1/2} A D_0^{-1/2}`` to node features without a dense matrix.

    Parameters
    ----------
    edge_index : Tensor
        Edge index with shape ``(2, num_edges)``.
    x : Tensor
        Node features with shape ``(num_nodes, feature_dim)``.
    edge_weight : Tensor or None, optional
        Non-negative edge weights with shape ``(num_edges,)``. Defaults to ones.
    num_nodes : int or None, optional
        Number of nodes. Inferred from ``x`` when omitted.

    Returns
    -------
    Tensor
        Smoothed node features with the same shape as ``x``.

    Raises
    ------
    ValueError
        If ``x`` is not 2D.
    """
    if x.dim() != 2:
        msg = f"x must be 2D (num_nodes, features), got shape {tuple(x.shape)}"
        raise ValueError(msg)

    node_count = num_nodes if num_nodes is not None else x.size(0)
    row, col = edge_index
    norm_weights = symmetric_normalized_adjacency_edge_weights(
        edge_index,
        num_nodes=node_count,
        edge_weight=edge_weight,
        dtype=x.dtype,
        device=x.device,
    )
    out = torch.zeros_like(x)
    out.index_add_(0, row, x[col] * norm_weights.unsqueeze(-1))
    return out


def random_walk_normalized_adjacency_edge_weights(
    edge_index: Tensor,
    *,
    num_nodes: int,
    edge_weight: Tensor | None = None,
    dtype: torch.dtype,
    device: torch.device | None = None,
    direction: RandomWalkDirection = "forward",
) -> Tensor:
    """Return per-edge weights for row-normalized random-walk adjacency.

    ``direction="forward"`` implements ``Â_f = D_{out}^{-1} A`` (DCRNN forward
    support). ``direction="backward"`` implements
    ``Â_b = D_{in}^{-1} A^{\\top}``.

    Zero out-/in-degree nodes yield a **zero row** (not an identity fill). A
    small positive degree floor (``1e-6``) matches the historical DiffConv
    normalization and avoids divide-by-zero; when the adjacency row is already
    zero, the normalized row remains zero.

    Parameters
    ----------
    edge_index : Tensor
        Edge index with shape ``(2, num_edges)`` (row 0 = source, row 1 =
        target).
    num_nodes : int
        Number of graph nodes.
    edge_weight : Tensor or None, optional
        Non-negative edge weights with shape ``(num_edges,)``. Defaults to ones.
    dtype : torch.dtype
        Floating dtype for the returned weights.
    device : torch.device or None, optional
        Device for the computation. Defaults to ``edge_index.device``.
    direction : {"forward", "backward"}, optional
        Random-walk orientation. Default is ``"forward"``.

    Returns
    -------
    Tensor
        Per-edge normalized weights with shape ``(num_edges,)``, aligned with
        ``edge_index`` columns for the corresponding sparse matvec.

    Raises
    ------
    ValueError
        If ``direction`` is not ``"forward"`` or ``"backward"``.
    """
    if direction not in {"forward", "backward"}:
        msg = f'direction must be "forward" or "backward", got {direction!r}'
        raise ValueError(msg)

    resolved_device = device if device is not None else edge_index.device
    row = edge_index[0]
    col = edge_index[1]
    if edge_weight is None:
        weights = torch.ones(row.size(0), dtype=dtype, device=resolved_device)
    else:
        weights = edge_weight.to(dtype=dtype, device=resolved_device)

    if direction == "forward":
        # (Â_f x)_i = sum_j A_{ij} / d_out_i · x_j  → scale by source degree.
        deg = torch.zeros(num_nodes, dtype=dtype, device=resolved_device)
        deg.index_add_(0, row, weights)
        scale = deg[row].clamp_min(_RANDOM_WALK_DEGREE_FLOOR)
        return weights / scale

    # Backward: (Â_b x)_i = sum_j A_{ji} / d_in_i · x_j  → scale by target degree.
    deg = torch.zeros(num_nodes, dtype=dtype, device=resolved_device)
    deg.index_add_(0, col, weights)
    scale = deg[col].clamp_min(_RANDOM_WALK_DEGREE_FLOOR)
    return weights / scale


def dense_random_walk_normalized_adjacency(
    edge_index: Tensor,
    num_nodes: int,
    *,
    edge_weight: Tensor | None = None,
    dtype: torch.dtype,
    direction: RandomWalkDirection = "forward",
) -> Tensor:
    """Assemble dense row-normalized random-walk adjacency.

    ``direction="forward"`` returns ``D_{out}^{-1} A``.
    ``direction="backward"`` returns ``D_{in}^{-1} A^{\\top}``.

    Sink and isolated nodes produce zero rows (not identity). See
    :func:`random_walk_normalized_adjacency_edge_weights` for the degree-floor
    convention.

    Parameters
    ----------
    edge_index : Tensor
        Edge index with shape ``(2, num_edges)``.
    num_nodes : int
        Number of graph nodes.
    edge_weight : Tensor or None, optional
        Non-negative edge weights with shape ``(num_edges,)``. Defaults to ones.
    dtype : torch.dtype
        Floating dtype for the dense matrix.
    direction : {"forward", "backward"}, optional
        Random-walk orientation. Default is ``"forward"``.

    Returns
    -------
    Tensor
        Dense matrix with shape ``(num_nodes, num_nodes)``. Duplicate edges
        accumulate (sum).
    """
    row, col = edge_index
    device = edge_index.device
    if edge_weight is None:
        weights = torch.ones(row.size(0), dtype=dtype, device=device)
    else:
        weights = edge_weight.to(dtype=dtype, device=device)

    adjacency = torch.zeros((num_nodes, num_nodes), dtype=dtype, device=device)
    adjacency.index_put_((row, col), weights, accumulate=True)
    if direction == "backward":
        adjacency = adjacency.transpose(0, 1).contiguous()
    elif direction != "forward":
        msg = f'direction must be "forward" or "backward", got {direction!r}'
        raise ValueError(msg)

    degree = adjacency.sum(dim=1).clamp_min(_RANDOM_WALK_DEGREE_FLOOR)
    return adjacency / degree.unsqueeze(1)


def random_walk_normalized_adjacency_matvec(
    edge_index: Tensor,
    x: Tensor,
    *,
    edge_weight: Tensor | None = None,
    num_nodes: int | None = None,
    direction: RandomWalkDirection = "forward",
) -> Tensor:
    """Apply row-normalized random-walk adjacency without a dense matrix.

    Parameters
    ----------
    edge_index : Tensor
        Edge index with shape ``(2, num_edges)``.
    x : Tensor
        Node features with shape ``(num_nodes, feature_dim)``.
    edge_weight : Tensor or None, optional
        Non-negative edge weights with shape ``(num_edges,)``. Defaults to ones.
    num_nodes : int or None, optional
        Number of nodes. Inferred from ``x`` when omitted.
    direction : {"forward", "backward"}, optional
        Random-walk orientation. Default is ``"forward"``.

    Returns
    -------
    Tensor
        Smoothed node features with the same shape as ``x``.

    Raises
    ------
    ValueError
        If ``x`` is not 2D or ``direction`` is invalid.
    """
    if x.dim() != 2:
        msg = f"x must be 2D (num_nodes, features), got shape {tuple(x.shape)}"
        raise ValueError(msg)

    node_count = num_nodes if num_nodes is not None else x.size(0)
    row, col = edge_index
    norm_weights = random_walk_normalized_adjacency_edge_weights(
        edge_index,
        num_nodes=node_count,
        edge_weight=edge_weight,
        dtype=x.dtype,
        device=x.device,
        direction=direction,
    )
    out = torch.zeros_like(x)
    if direction == "forward":
        # Messages along A: accumulate at source from target features.
        out.index_add_(0, row, x[col] * norm_weights.unsqueeze(-1))
    elif direction == "backward":
        # Messages along A^T: accumulate at target from source features.
        out.index_add_(0, col, x[row] * norm_weights.unsqueeze(-1))
    else:
        msg = f'direction must be "forward" or "backward", got {direction!r}'
        raise ValueError(msg)
    return out


def _relation_normalization_to_rw_direction(
    normalization: RelationNormalization,
) -> RandomWalkDirection:
    """Map hetero relation normalization to a random-walk direction.

    Parameters
    ----------
    normalization : {"rgcn_in_degree", "random_walk"}
        Relation normalization mode.

    Returns
    -------
    {"forward", "backward"}
        Matching :data:`RandomWalkDirection`.

    Raises
    ------
    ValueError
        If ``normalization`` is not a supported mode.
    """
    if normalization == "rgcn_in_degree":
        # R-GCN in-degree at the destination: D_in^{-1} A^T in library layout.
        return "backward"
    if normalization == "random_walk":
        return "forward"
    msg = (
        "normalization must be one of "
        f"{sorted(RELATION_NORMALIZATION_MODES)}, got {normalization!r}"
    )
    raise ValueError(msg)


def relation_degree_normalize(
    edge_index: Tensor,
    *,
    num_nodes: int,
    edge_weight: Tensor | None = None,
    dtype: torch.dtype,
    device: torch.device | None = None,
    normalization: RelationNormalization = "rgcn_in_degree",
) -> Tensor:
    """Return per-edge weights for one relation under hetero normalization.

    ``normalization="rgcn_in_degree"`` (default) implements R-GCN-style
    destination in-degree scaling: for edge ``j -> i`` the weight is
    ``w / c_{i,r}`` where ``c_{i,r}`` is the weighted in-degree of node ``i``
    under this relation (Schlichtkrull et al., R-GCN; normalization only —
    not a full paper reproduction). ``num_nodes`` is the **destination**
    node-type cardinality (shared ``N`` for multiplex).

    ``normalization="random_walk"`` uses forward out-degree scaling
    ``D_{out}^{-1} A`` (same convention as homogeneous
    ``adjacency="random_walk"``). Reverse relations are **not** synthesized
    here; callers add explicit reverse edge types when needed.

    Isolated destinations produce zero contribution (zero row), not an
    identity fill. Empty relation banks (``num_edges == 0``) return an empty
    weight tensor. The degree floor matches
    :func:`random_walk_normalized_adjacency_edge_weights` (``1e-6``).

    Parameters
    ----------
    edge_index : Tensor
        Relation edge index with shape ``(2, num_edges)`` (row 0 = source,
        row 1 = destination).
    num_nodes : int
        Destination-type node count used for degree vectors.
    edge_weight : Tensor or None, optional
        Non-negative edge weights with shape ``(num_edges,)``. Defaults to ones.
    dtype : torch.dtype
        Floating dtype for the returned weights.
    device : torch.device or None, optional
        Device for the computation. Defaults to ``edge_index.device``.
    normalization : {"rgcn_in_degree", "random_walk"}, optional
        Relation normalization mode. Default is ``"rgcn_in_degree"``.

    Returns
    -------
    Tensor
        Per-edge normalized weights with shape ``(num_edges,)``.

    Raises
    ------
    ValueError
        If ``normalization`` is unsupported.
    """
    direction = _relation_normalization_to_rw_direction(normalization)
    return random_walk_normalized_adjacency_edge_weights(
        edge_index,
        num_nodes=num_nodes,
        edge_weight=edge_weight,
        dtype=dtype,
        device=device,
        direction=direction,
    )


def dense_relation_normalized_adjacency(
    edge_index: Tensor,
    num_nodes: int,
    *,
    edge_weight: Tensor | None = None,
    dtype: torch.dtype,
    normalization: RelationNormalization = "rgcn_in_degree",
) -> Tensor:
    """Assemble dense per-relation normalized adjacency.

    See :func:`relation_degree_normalize` for mode semantics. For
    ``"rgcn_in_degree"``, ``(Â x)_i = sum_j A_{ij} / c_{i,r} · x_j`` with
    messages along ``j -> i``.

    Parameters
    ----------
    edge_index : Tensor
        Relation edge index with shape ``(2, num_edges)``.
    num_nodes : int
        Destination-type node count.
    edge_weight : Tensor or None, optional
        Non-negative edge weights with shape ``(num_edges,)``.
    dtype : torch.dtype
        Floating dtype for the dense matrix.
    normalization : {"rgcn_in_degree", "random_walk"}, optional
        Relation normalization mode. Default is ``"rgcn_in_degree"``.

    Returns
    -------
    Tensor
        Dense matrix with shape ``(num_nodes, num_nodes)``.

    Raises
    ------
    ValueError
        If ``normalization`` is unsupported.
    """
    direction = _relation_normalization_to_rw_direction(normalization)
    return dense_random_walk_normalized_adjacency(
        edge_index,
        num_nodes,
        edge_weight=edge_weight,
        dtype=dtype,
        direction=direction,
    )


def relation_normalized_adjacency_matvec(
    edge_index: Tensor,
    x: Tensor,
    *,
    edge_weight: Tensor | None = None,
    num_nodes: int | None = None,
    normalization: RelationNormalization = "rgcn_in_degree",
) -> Tensor:
    """Apply per-relation normalized adjacency without a dense matrix.

    Parameters
    ----------
    edge_index : Tensor
        Relation edge index with shape ``(2, num_edges)``.
    x : Tensor
        Node features with shape ``(num_nodes, feature_dim)``. For typed
        cross-relations, ``x`` is the **source**-type feature matrix and
        ``num_nodes`` must be the destination cardinality when it differs
        (multiplex: shared ``N``).
    edge_weight : Tensor or None, optional
        Non-negative edge weights with shape ``(num_edges,)``.
    num_nodes : int or None, optional
        Destination-type node count. Inferred from ``x`` when omitted
        (multiplex / same-type relations).
    normalization : {"rgcn_in_degree", "random_walk"}, optional
        Relation normalization mode. Default is ``"rgcn_in_degree"``.

    Returns
    -------
    Tensor
        Aggregated features with shape ``(num_nodes, feature_dim)``.

    Raises
    ------
    ValueError
        If ``x`` is not 2D or ``normalization`` is unsupported.
    """
    direction = _relation_normalization_to_rw_direction(normalization)
    if normalization == "rgcn_in_degree":
        # Destination aggregation; allow num_nodes to differ from x.size(0)
        # only when callers pass it explicitly (typed path).
        node_count = num_nodes if num_nodes is not None else x.size(0)
        if x.dim() != 2:
            msg = f"x must be 2D (num_nodes, features), got shape {tuple(x.shape)}"
            raise ValueError(msg)
        row, col = edge_index
        norm_weights = relation_degree_normalize(
            edge_index,
            num_nodes=node_count,
            edge_weight=edge_weight,
            dtype=x.dtype,
            device=x.device,
            normalization=normalization,
        )
        out = x.new_zeros((node_count, x.size(1)))
        if row.numel() == 0:
            return out
        out.index_add_(0, col, x[row] * norm_weights.unsqueeze(-1))
        return out
    return random_walk_normalized_adjacency_matvec(
        edge_index,
        x,
        edge_weight=edge_weight,
        num_nodes=num_nodes,
        direction=direction,
    )


def symmetric_normalized_laplacian_matvec(
    edge_index: Tensor,
    x: Tensor,
    *,
    edge_weight: Tensor | None = None,
    num_nodes: int | None = None,
) -> Tensor:
    """Apply ``L_sym = P - Â`` to node features without a dense matrix.

    Isolated nodes (``d_i = 0``) map to zeros. On graphs with no isolates this
    matches ``(I - Â) x``. Intended for undirected, symmetrically represented
    adjacency; directed graphs should use random-walk normalization rather than
    ``L_sym``.

    Parameters
    ----------
    edge_index : Tensor
        Edge index with shape ``(2, num_edges)``.
    x : Tensor
        Node features with shape ``(num_nodes, feature_dim)``.
    edge_weight : Tensor or None, optional
        Non-negative edge weights with shape ``(num_edges,)``. Defaults to ones.
    num_nodes : int or None, optional
        Number of nodes. Inferred from ``x`` when omitted.

    Returns
    -------
    Tensor
        Laplacian features with the same shape as ``x``.

    Raises
    ------
    ValueError
        If ``x`` is not 2D.
    """
    if x.dim() != 2:
        msg = f"x must be 2D (num_nodes, features), got shape {tuple(x.shape)}"
        raise ValueError(msg)

    node_count = num_nodes if num_nodes is not None else x.size(0)
    support = degree_support_mask(
        edge_index,
        num_nodes=node_count,
        edge_weight=edge_weight,
        dtype=x.dtype,
        device=x.device,
    )
    adj_x = symmetric_normalized_adjacency_matvec(
        edge_index,
        x,
        edge_weight=edge_weight,
        num_nodes=node_count,
    )
    return support.unsqueeze(-1) * x - adj_x


def resolve_edge_index(
    x_or_data: Tensor | Data,
    edge_index: Tensor | None,
) -> Tensor:
    """Extract or validate ``edge_index`` from input arguments.

    Parameters
    ----------
    x_or_data : Tensor or Data
        Graph input; when a ``Data`` object, its ``edge_index`` is returned.
    edge_index : Tensor or None
        Explicit edge index for tensor input.

    Returns
    -------
    Tensor
        Edge index with shape ``(2, num_edges)``.

    Raises
    ------
    ValueError
        If ``x_or_data`` is a tensor and ``edge_index`` is ``None``.
    """
    if isinstance(x_or_data, Data):
        return x_or_data.edge_index
    if edge_index is None:
        msg = "edge_index is required when x_or_data is a tensor"
        raise ValueError(msg)
    return edge_index


def resolve_edge_weight(
    x_or_data: Tensor | Data,
    edge_weight: Tensor | None,
) -> Tensor | None:
    """Extract or validate optional ``edge_weight`` from input arguments.

    Parameters
    ----------
    x_or_data : Tensor or Data
        Graph input; when a ``Data`` object, its ``edge_weight`` is returned
        when present.
    edge_weight : Tensor or None
        Explicit edge weights for tensor input.

    Returns
    -------
    Tensor or None
        Edge weights with shape ``(num_edges,)``, or ``None`` when unweighted.
    """
    if isinstance(x_or_data, Data):
        return snapshot_edge_weight(x_or_data)
    return edge_weight


def resolve_graph_inputs(
    x_or_data: Tensor | Data,
    edge_index: Tensor | None,
    edge_weight: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Extract node features, edge index, and optional weights from input.

    Parameters
    ----------
    x_or_data : Tensor or Data
        Either a PyG ``Data`` object or node features ``x``.
    edge_index : Tensor or None
        Edge index required when ``x_or_data`` is a tensor.
    edge_weight : Tensor or None, optional
        Edge weights for tensor input. Ignored for ``Data`` input.

    Returns
    -------
    tuple of (Tensor, Tensor, Tensor or None)
        Node features, edge index, and optional edge weights.

    Raises
    ------
    ValueError
        If ``x_or_data`` is a tensor and ``edge_index`` is ``None``.
    """
    if isinstance(x_or_data, Data):
        return (
            x_or_data.x,
            x_or_data.edge_index,
            snapshot_edge_weight(x_or_data),
        )
    resolved_edge_index = resolve_edge_index(x_or_data, edge_index)
    return x_or_data, resolved_edge_index, edge_weight


def snapshot_to_device(
    snapshot: Data | HeteroData,
    device: torch.device,
) -> Data | HeteroData:
    """Move a graph snapshot to a target device, preserving topology fields.

    Homogeneous ``Data`` copies ``x``, ``edge_index``, optional ``edge_weight``,
    and optional hyperedge incidence when present. Multiplex ``HeteroData``
    uses PyG ``HeteroData.to(device)`` so node/edge stores stay intact.

    Parameters
    ----------
    snapshot : Data or HeteroData
        Graph snapshot to transfer.
    device : torch.device
        Destination device.

    Returns
    -------
    Data or HeteroData
        Snapshot with tensors moved to ``device``.
    """
    if isinstance(snapshot, HeteroData):
        return snapshot.to(device)

    fields: dict[str, Tensor] = {
        "x": snapshot.x.to(device),
        "edge_index": snapshot.edge_index.to(device),
    }
    edge_weight = snapshot_edge_weight(snapshot)
    if edge_weight is not None:
        fields["edge_weight"] = edge_weight.to(device)
    hyperedge_index = snapshot_hyperedge_index(snapshot)
    if hyperedge_index is not None:
        fields["hyperedge_index"] = hyperedge_index.to(device)
    hyperedge_weight = snapshot_hyperedge_weight(snapshot)
    if hyperedge_weight is not None:
        fields["hyperedge_weight"] = hyperedge_weight.to(device)
    return Data(**fields)
