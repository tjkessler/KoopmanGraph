"""Physics-informed observable lifting for hybrid Koopman latents.

The ``graph_laplacian`` preset applies a **sparse** ``L_sym @ x`` matvec using
the shared pseudoinverse-normalized Laplacian in :mod:`koopman_graph.graph_utils`.
Benchmark diffusion in :mod:`koopman_graph.datasets.dynamics` uses the same
``L_sym`` definition but a dense one-step operator for offline rollouts. Both
paths accumulate duplicate edges.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Literal, get_args

import torch
from torch import Tensor
from torch_geometric.data import Data

from koopman_graph.graph_utils import (
    snapshot_edge_weight,
    symmetric_normalized_adjacency_edge_weights,
    symmetric_normalized_laplacian_matvec,
)

PhysicsLiftingFn = Callable[[Data], Tensor]
PhysicsPosition = Literal["prepend"]
PhysicsPresetName = Literal[
    "graph_curvature",
    "graph_gradient",
    "graph_laplacian",
]

PHYSICS_POSITION: PhysicsPosition = "prepend"
_ALLOWED_PHYSICS_POSITIONS = get_args(PhysicsPosition)

PHYSICS_PRESETS: dict[str, PhysicsLiftingFn] = {}
_POLYNOMIAL_PRESET = re.compile(r"polynomial\(([1-9][0-9]*)\)")


def graph_laplacian_features(data: Data) -> Tensor:
    """Return normalized Laplacian features ``L_sym @ x`` for each node.

    The symmetrically normalized graph Laplacian (Chung / Wikipedia) is

    .. math::

        L_{\\mathrm{sym}}
        = (D^{+})^{1/2} (D - A) (D^{+})^{1/2}
        = P - \\widehat{A},

    where ``Â = D_0^{-1/2} A D_0^{-1/2}``, ``D^+`` is the Moore–Penrose
    pseudoinverse of the diagonal degree matrix, and
    ``P = diag(1_{d_i > 0})`` projects onto non-isolated nodes. When every node
    has positive degree, ``P = I`` and this reduces to ``I - Â``. Isolated nodes
    map to **zeros** (not identity passthrough of ``x_i``).

    Applied to node features ``x`` via a sparse matvec (see
    :func:`~koopman_graph.graph_utils.symmetric_normalized_laplacian_matvec`).
    Dense benchmark diffusion operators in
    :mod:`koopman_graph.datasets.dynamics` share the same ``L_sym`` definition.
    The contract assumes an undirected, symmetrically represented adjacency.
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
    return symmetric_normalized_laplacian_matvec(
        data.edge_index,
        x,
        edge_weight=snapshot_edge_weight(data),
        num_nodes=x.size(0),
    )


def graph_gradient_features(data: Data) -> Tensor:
    r"""Return nodewise normalized graph-gradient magnitudes.

    For each node and feature channel this computes

    .. math::

        g_i(x) = \left(\sum_j \widehat{A}_{ij}
        (x_j - x_i)^2\right)^{1/2},

    using the shared symmetric normalization and duplicate-edge accumulation
    conventions. Isolated nodes map to zero. The graph must be undirected and
    symmetrically represented, and edge weights must be non-negative.

    Parameters
    ----------
    data : Data
        Graph snapshot with two-dimensional ``x`` and ``edge_index``.

    Returns
    -------
    Tensor
        Non-negative node features with the same shape as ``data.x``.
    """
    if data.x is None:
        msg = "data.x is required for graph_gradient_features"
        raise ValueError(msg)
    if data.x.dim() != 2:
        msg = f"data.x must be 2D, got shape {tuple(data.x.shape)}"
        raise ValueError(msg)

    x = data.x
    row, col = data.edge_index
    weights = symmetric_normalized_adjacency_edge_weights(
        data.edge_index,
        num_nodes=x.size(0),
        edge_weight=snapshot_edge_weight(data),
        dtype=x.dtype,
        device=x.device,
    )
    squared_differences = (x[col] - x[row]).square()
    energy = torch.zeros_like(x)
    energy.index_add_(0, row, squared_differences * weights.unsqueeze(-1))
    return energy.clamp_min(0).sqrt()


def graph_curvature_features(data: Data) -> Tensor:
    """Return biharmonic graph-curvature features ``L_sym² @ x``.

    Both sparse matvecs use the shared pseudoinverse-normalized
    ``L_sym = P - Â`` contract, so isolated nodes map to zero. The graph must
    be undirected and symmetrically represented.

    Parameters
    ----------
    data : Data
        Graph snapshot with ``x``, ``edge_index``, and optional
        ``edge_weight``.

    Returns
    -------
    Tensor
        Curvature features with the same shape as ``data.x``.
    """
    laplacian = graph_laplacian_features(data)
    return symmetric_normalized_laplacian_matvec(
        data.edge_index,
        laplacian,
        edge_weight=snapshot_edge_weight(data),
        num_nodes=laplacian.size(0),
    )


def polynomial_features(data: Data, *, degree: int) -> Tensor:
    """Concatenate elementwise powers ``[x, x², ..., x^degree]``.

    Parameters
    ----------
    data : Data
        Graph snapshot with two-dimensional node features ``x``.
    degree : int
        Highest positive integer power.

    Returns
    -------
    Tensor
        Features with shape ``(num_nodes, in_channels * degree)``.

    Raises
    ------
    ValueError
        If ``degree < 1`` or ``data.x`` is absent/not two-dimensional.
    """
    if degree < 1:
        msg = f"degree must be >= 1, got {degree}"
        raise ValueError(msg)
    if data.x is None:
        msg = "data.x is required for polynomial_features"
        raise ValueError(msg)
    if data.x.dim() != 2:
        msg = f"data.x must be 2D, got shape {tuple(data.x.shape)}"
        raise ValueError(msg)
    return torch.cat([data.x.pow(power) for power in range(1, degree + 1)], dim=-1)


def make_polynomial_features(degree: int) -> PhysicsLiftingFn:
    """Return a physics-lifting callable for elementwise polynomial features.

    Parameters
    ----------
    degree : int
        Highest positive integer power.

    Returns
    -------
    callable
        Function mapping a graph snapshot to concatenated polynomial features.
    """
    if degree < 1:
        msg = f"degree must be >= 1, got {degree}"
        raise ValueError(msg)

    def lift(data: Data) -> Tensor:
        """Evaluate the configured polynomial lifting.

        Parameters
        ----------
        data : Data
            Graph snapshot containing node features.

        Returns
        -------
        Tensor
            Concatenated elementwise powers through the configured degree.
        """
        return polynomial_features(data, degree=degree)

    lift.__name__ = f"polynomial_features_degree_{degree}"
    return lift


PHYSICS_PRESETS["graph_gradient"] = graph_gradient_features
PHYSICS_PRESETS["graph_curvature"] = graph_curvature_features
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
        Registered preset name such as ``"graph_laplacian"``,
        ``"graph_gradient"``, or ``"graph_curvature"``. Dynamic polynomial
        presets use ``"polynomial(degree)"``, for example ``"polynomial(3)"``.
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
    polynomial_match = _POLYNOMIAL_PRESET.fullmatch(physics_preset)
    if polynomial_match is not None:
        return make_polynomial_features(int(polynomial_match.group(1)))
    if physics_preset not in PHYSICS_PRESETS:
        registered = ", ".join(sorted(PHYSICS_PRESETS))
        msg = (
            f"Unknown physics_preset {physics_preset!r}; "
            f"registered presets: {registered}, polynomial(degree)"
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
