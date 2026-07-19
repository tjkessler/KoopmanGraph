"""Topology and Laplacian helpers for graph inputs (power-user).

Shared degree / adjacency / Laplacian mathematics and graph-input resolution.
Import from :mod:`koopman_graph.graph_utils` (package re-exports) unless you
need the peer module explicitly.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch_geometric.data import Data


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


def node_degrees(
    edge_index: Tensor,
    *,
    num_nodes: int,
    edge_weight: Tensor | None = None,
    dtype: torch.dtype,
    device: torch.device | None = None,
) -> Tensor:
    """Return weighted out-degrees ``d_i = sum_j A_{ij}`` for each node.

    Assumes an undirected, symmetrically represented adjacency (each undirected
    edge appears in both directions, or weights already encode that symmetry).
    Duplicate edges accumulate.

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

    The contract assumes an **undirected, symmetrically represented** adjacency.

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
    Isolated nodes have a zero diagonal entry (not ``1``). Assumes undirected,
    symmetrically represented adjacency.

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


def symmetric_normalized_laplacian_matvec(
    edge_index: Tensor,
    x: Tensor,
    *,
    edge_weight: Tensor | None = None,
    num_nodes: int | None = None,
) -> Tensor:
    """Apply ``L_sym = P - Â`` to node features without a dense matrix.

    Isolated nodes (``d_i = 0``) map to zeros. On graphs with no isolates this
    matches ``(I - Â) x``. Assumes undirected, symmetrically represented
    adjacency.

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
    """Move a graph snapshot to a target device, preserving edge weights.

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
    return Data(**fields)

