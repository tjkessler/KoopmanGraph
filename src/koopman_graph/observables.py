"""Physics-informed observable lifting for hybrid Koopman latents.

The ``graph_laplacian`` preset applies a **sparse** ``L_sym @ x`` matvec using
the shared normalization in :mod:`koopman_graph.graph_utils`. Benchmark
diffusion in :mod:`koopman_graph.datasets.dynamics` uses the same ``L_sym``
definition but a dense one-step operator for offline rollouts. Both paths
accumulate duplicate edges.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, get_args

import torch
from torch import Tensor
from torch_geometric.data import Data

from koopman_graph.graph_utils import (
    snapshot_edge_weight,
    symmetric_normalized_adjacency_matvec,
)

PhysicsLiftingFn = Callable[[Data], Tensor]
PhysicsPosition = Literal["prepend"]
PhysicsPresetName = Literal["graph_laplacian"]

PHYSICS_POSITION: PhysicsPosition = "prepend"
_ALLOWED_PHYSICS_POSITIONS = get_args(PhysicsPosition)

PHYSICS_PRESETS: dict[str, PhysicsLiftingFn] = {}


def graph_laplacian_features(data: Data) -> Tensor:
    """Return normalized Laplacian features ``L_norm @ x`` for each node.

    The symmetric normalized graph Laplacian is

    .. math::

        L_{\\text{norm}} = I - D^{-1/2} A D^{-1/2}

    applied to node features ``x`` via a sparse matvec (see
    :func:`~koopman_graph.graph_utils.symmetric_normalized_adjacency_matvec`).
    Dense benchmark diffusion operators in
    :mod:`koopman_graph.datasets.dynamics` share the same ``L_sym`` weights.
    The output has shape ``(num_nodes, in_channels)``, so ``physics_dim`` must
    equal ``data.x.size(-1)`` when this preset is used.

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
    adj_x = symmetric_normalized_adjacency_matvec(
        data.edge_index,
        x,
        edge_weight=snapshot_edge_weight(data),
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


def resolve_physics_position(
    position: str | None = None,
) -> PhysicsPosition:
    """Validate and normalize hybrid physics concatenation position.

    Used by model construction and checkpoint reconstruct so save/load share
    one allowed-value set. ``None`` resolves to :data:`PHYSICS_POSITION`
    (``"prepend"``). Only ``"prepend"`` is supported today.

    Parameters
    ----------
    position : str or None, optional
        Checkpoint or constructor value. ``None`` uses the default.

    Returns
    -------
    {"prepend"}
        Validated concatenation position.

    Raises
    ------
    ValueError
        If ``position`` is not in the supported :data:`PhysicsPosition` set.
    """
    resolved = PHYSICS_POSITION if position is None else position
    if resolved not in _ALLOWED_PHYSICS_POSITIONS:
        allowed = ", ".join(repr(value) for value in _ALLOWED_PHYSICS_POSITIONS)
        msg = f"Unsupported physics position {resolved!r}; expected one of: {allowed}"
        raise ValueError(msg)
    return resolved  # type: ignore[return-value]


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
    resolve_physics_position(position)
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
