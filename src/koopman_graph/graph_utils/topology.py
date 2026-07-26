"""Topology and Laplacian helpers for graph inputs (power-user).

Shared degree / adjacency / Laplacian mathematics and graph-input resolution.
Import from :mod:`koopman_graph.graph_utils` (package re-exports) unless you
need the peer module explicitly.

Includes symmetric normalization (``D^{-1/2} A D^{-1/2}``) and directed
random-walk normalization (``D_{\\mathrm{out}}^{-1} A`` /
``D_{\\mathrm{in}}^{-1} A^{\\top}``).
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor
from torch_geometric.data import Data

RandomWalkDirection = Literal["forward", "backward"]

# Match DiffConv / DCRNN-style row normalization (nn.gnn legacy floor).
_RANDOM_WALK_DEGREE_FLOOR = 1e-6


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
    if hyperedge_index.numel() == 0 or num_nodes == 0:
        return torch.zeros(
            (num_nodes, num_nodes),
            dtype=dtype,
            device=resolved_device,
        )

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
    return deg_v_inv_sqrt.unsqueeze(1) * mid * deg_v_inv_sqrt.unsqueeze(0)


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
    :func:`dense_hyperedge_normalized_adjacency` and returns ``Ĥ @ x``.
    Suitable for modest ``N`` (same dense-operator ceiling as
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


def snapshot_to_device(snapshot: Data, device: torch.device) -> Data:
    """Move a graph snapshot to a target device, preserving topology fields.

    Copies ``x``, ``edge_index``, optional ``edge_weight``, and optional
    hyperedge incidence (``hyperedge_index`` / ``hyperedge_weight``) when
    present.

    Parameters
    ----------
    snapshot : Data
        Graph snapshot to transfer.
    device : torch.device
        Destination device.

    Returns
    -------
    Data
        Snapshot with tensors moved to ``device``.
    """
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
