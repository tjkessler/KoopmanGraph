"""Physics-informed observable lifting for hybrid Koopman latents."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch
from torch import Tensor
from torch_geometric.data import Data

from koopman_graph.data import _snapshot_edge_weight

PhysicsLiftingFn = Callable[[Data], Tensor]
PhysicsPosition = Literal["prepend"]
PhysicsPresetName = Literal["graph_laplacian"]

PHYSICS_POSITION: PhysicsPosition = "prepend"

PHYSICS_PRESETS: dict[str, PhysicsLiftingFn] = {}


def _symmetric_normalized_adjacency_matvec(
    edge_index: Tensor,
    x: Tensor,
    *,
    edge_weight: Tensor | None = None,
    num_nodes: int | None = None,
) -> Tensor:
    """Apply ``D^{-1/2} A D^{-1/2}`` to node features without forming a dense matrix.

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
    """
    if x.dim() != 2:
        msg = f"x must be 2D (num_nodes, features), got shape {tuple(x.shape)}"
        raise ValueError(msg)

    node_count = num_nodes if num_nodes is not None else x.size(0)
    row, col = edge_index
    dtype = x.dtype
    device = x.device

    if edge_weight is None:
        weights = torch.ones(row.size(0), dtype=dtype, device=device)
    else:
        weights = edge_weight.to(dtype=dtype, device=device)

    deg = torch.zeros(node_count, dtype=dtype, device=device)
    deg.index_add_(0, row, weights)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt = torch.where(
        torch.isfinite(deg_inv_sqrt),
        deg_inv_sqrt,
        torch.zeros_like(deg_inv_sqrt),
    )

    norm_weights = deg_inv_sqrt[row] * weights * deg_inv_sqrt[col]
    out = torch.zeros_like(x)
    out.index_add_(0, row, x[col] * norm_weights.unsqueeze(-1))
    return out


def graph_laplacian_features(data: Data) -> Tensor:
    """Return normalized Laplacian features ``L_norm @ x`` for each node.

    The symmetric normalized graph Laplacian is

    .. math::

        L_{\\text{norm}} = I - D^{-1/2} A D^{-1/2}

    applied to node features ``x``. The output has shape
    ``(num_nodes, in_channels)``, so ``physics_dim`` must equal ``data.x.size(-1)``
    when this preset is used.

    Parameters
    ----------
    data : Data
        Graph snapshot with ``x`` and ``edge_index``. Optional ``edge_weight`` is
        respected when present.

    Returns
    -------
    Tensor
        Physics-informed node features with shape ``(num_nodes, in_channels)``.
    """
    if data.x is None:
        msg = "data.x is required for graph_laplacian_features"
        raise ValueError(msg)

    x = data.x
    adj_x = _symmetric_normalized_adjacency_matvec(
        data.edge_index,
        x,
        edge_weight=_snapshot_edge_weight(data),
        num_nodes=x.size(0),
    )
    return x - adj_x


PHYSICS_PRESETS["graph_laplacian"] = graph_laplacian_features


def resolve_physics_lifting_fn(
    *,
    physics_preset: str | None = None,
    physics_lifting_fn: PhysicsLiftingFn | None = None,
) -> PhysicsLiftingFn | None:
    """Resolve a physics lifting callable from a preset name and/or custom function.

    Parameters
    ----------
    physics_preset : str or None, optional
        Registered preset name such as ``"graph_laplacian"``.
    physics_lifting_fn : callable or None, optional
        Custom lifting function. When both a preset and a custom function are
        provided, the custom function takes precedence.

    Returns
    -------
    callable or None
        Resolved physics lifting function, or ``None`` when neither input is set.

    Raises
    ------
    ValueError
        If ``physics_preset`` is not registered.
    """
    if physics_lifting_fn is not None:
        return physics_lifting_fn
    if physics_preset is None:
        return None
    if physics_preset not in PHYSICS_PRESETS:
        registered = ", ".join(sorted(PHYSICS_PRESETS))
        msg = (
            f"Unknown physics_preset {physics_preset!r}; "
            f"registered presets: {registered}"
        )
        raise ValueError(msg)
    return PHYSICS_PRESETS[physics_preset]


def concatenate_observables(
    physics_features: Tensor,
    gnn_features: Tensor,
    *,
    position: PhysicsPosition = PHYSICS_POSITION,
) -> Tensor:
    """Concatenate physics and GNN latent features along the feature dimension.

    Parameters
    ----------
    physics_features : Tensor
        Physics-informed features with shape ``(num_nodes, physics_dim)``.
    gnn_features : Tensor
        GNN-encoded features with shape ``(num_nodes, gnn_dim)``.
    position : {"prepend"}, optional
        Concatenation order. ``"prepend"`` yields ``[physics || gnn]``.

    Returns
    -------
    Tensor
        Combined latent features with shape ``(num_nodes, physics_dim + gnn_dim)``.
    """
    if position != "prepend":
        msg = f"Unsupported physics position {position!r}; expected 'prepend'"
        raise ValueError(msg)
    if physics_features.size(0) != gnn_features.size(0):
        msg = (
            "physics and GNN features must share num_nodes, got "
            f"{physics_features.size(0)} and {gnn_features.size(0)}"
        )
        raise ValueError(msg)
    return torch.cat([physics_features, gnn_features], dim=-1)


def validate_physics_output(
    physics_features: Tensor,
    *,
    physics_dim: int,
    num_nodes: int,
) -> None:
    """Validate physics lifting output shape against the declared dimension.

    Parameters
    ----------
    physics_features : Tensor
        Output of a physics lifting function.
    physics_dim : int
        Declared physics feature dimension per node.
    num_nodes : int
        Expected number of graph nodes.

    Raises
    ------
    ValueError
        If the tensor rank or trailing dimensions are incorrect.
    """
    expected_shape = (num_nodes, physics_dim)
    if physics_features.shape != expected_shape:
        msg = (
            f"physics lifting output must have shape {expected_shape}, "
            f"got {tuple(physics_features.shape)}"
        )
        raise ValueError(msg)
