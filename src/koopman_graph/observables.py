"""Physics-informed observable lifting for hybrid Koopman latents.

The ``graph_laplacian`` preset applies a **sparse** ``L_sym @ x`` matvec using
the shared pseudoinverse-normalized Laplacian in :mod:`koopman_graph.graph_utils`.
Benchmark diffusion in :mod:`koopman_graph.datasets.dynamics` uses the same
``L_sym`` definition but a dense one-step operator for offline rollouts. Both
paths accumulate duplicate edges.

Simplicial-1 / Hodge helpers (``hodge_gradient``, ``simplicial_one_laplacian``)
use the combinatorial 1-Laplacian ``L_1 = B_1 B_1^T`` from a signed edge
incidence ``B_1``. They are **not** sheaf or cell-complex operators and are
distinct from ``L_sym``. Optional ``face_index`` on ``Data`` is validated for
downstream simplicial encoders but does not change ``B_1`` (which comes from
``edge_index``).
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
    "hodge_gradient",
    "simplicial_one_laplacian",
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


def coerce_face_index(face_index: Tensor, *, num_nodes: int) -> Tensor:
    """Validate triangular ``face_index`` with shape ``(3, num_faces)``.

    Each column is one 2-simplex (triangle) whose entries are node ids in
    ``[0, num_nodes)``. Used by simplicial-1 pipelines and by
    :mod:`koopman_graph.nn.cell_complex` 2-cell incidence.

    Parameters
    ----------
    face_index : Tensor
        Candidate face incidence with shape ``(3, num_faces)``.
    num_nodes : int
        Number of nodes in the complex.

    Returns
    -------
    Tensor
        Cloned ``long`` tensor with the same shape.

    Raises
    ------
    ValueError
        If rank/shape is wrong, ``num_nodes`` is invalid, or any node id is
        out of range.
    """
    if num_nodes < 1:
        msg = f"num_nodes must be >= 1, got {num_nodes}"
        raise ValueError(msg)
    if face_index.ndim != 2 or face_index.shape[0] != 3:
        msg = (
            f"face_index must have shape (3, num_faces), got {tuple(face_index.shape)}"
        )
        raise ValueError(msg)
    if face_index.numel() == 0:
        return face_index.to(dtype=torch.long).detach().clone()
    if not face_index.dtype.is_floating_point and face_index.dtype != torch.bool:
        coerced = face_index.to(dtype=torch.long)
    else:
        msg = f"face_index must be an integer tensor, got dtype {face_index.dtype}"
        raise ValueError(msg)
    if bool((coerced < 0).any()) or bool((coerced >= num_nodes).any()):
        msg = (
            f"face_index node ids must lie in [0, {num_nodes}), "
            f"got min={int(coerced.min())} max={int(coerced.max())}"
        )
        raise ValueError(msg)
    return coerced.detach().clone()


def boundary_incidence_b1(edge_index: Tensor, *, num_nodes: int) -> Tensor:
    """Build the signed boundary incidence ``B_1`` with shape ``(N, E)``.

    Column ``e`` corresponding to oriented edge ``(i → j)`` has ``+1`` at the
    tail ``i`` and ``-1`` at the head ``j``. Pass **one oriented column per
    undirected edge** (not a doubled symmetric ``edge_index``) so
    ``L_1 = B_1 B_1^T`` matches the combinatorial graph Laplacian on that
    undirected graph.

    Parameters
    ----------
    edge_index : Tensor
        Oriented edges with shape ``(2, num_edges)``.
    num_nodes : int
        Number of nodes ``N``.

    Returns
    -------
    Tensor
        Dense signed incidence of shape ``(num_nodes, num_edges)``.

    Raises
    ------
    ValueError
        If ``edge_index`` shape or node ids are invalid.
    """
    if num_nodes < 1:
        msg = f"num_nodes must be >= 1, got {num_nodes}"
        raise ValueError(msg)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        msg = (
            f"edge_index must have shape (2, num_edges), got {tuple(edge_index.shape)}"
        )
        raise ValueError(msg)
    edges = edge_index.to(dtype=torch.long)
    if edges.numel() > 0 and (
        bool((edges < 0).any()) or bool((edges >= num_nodes).any())
    ):
        msg = (
            f"edge_index node ids must lie in [0, {num_nodes}), "
            f"got min={int(edges.min())} max={int(edges.max())}"
        )
        raise ValueError(msg)
    num_edges = int(edges.shape[1])
    incidence = torch.zeros(
        num_nodes,
        num_edges,
        dtype=torch.float32,
        device=edges.device,
    )
    if num_edges == 0:
        return incidence
    tails = edges[0]
    heads = edges[1]
    edge_ids = torch.arange(num_edges, device=edges.device)
    incidence[tails, edge_ids] = 1.0
    incidence[heads, edge_ids] = -1.0
    # Self-loops cancel to zero in the signed incidence.
    self_loop = tails == heads
    if bool(self_loop.any()):
        incidence[:, self_loop] = 0.0
    return incidence


def _maybe_validate_face_index(data: Data, *, num_nodes: int) -> None:
    """Validate optional ``data.face_index`` when present.

    Parameters
    ----------
    data
        Value for ``data``.
    num_nodes
        Value for ``num_nodes``.
    """
    face_index = getattr(data, "face_index", None)
    if face_index is None:
        return
    coerce_face_index(face_index, num_nodes=num_nodes)


def diagonal_sheaf_laplacian_matvec(
    edge_index: Tensor,
    x: Tensor,
    source_diag: Tensor,
    target_diag: Tensor,
    *,
    num_nodes: int | None = None,
) -> Tensor:
    """Apply a sheaf Laplacian with **diagonal** restriction maps.

    Uses the same oriented-edge convention as
    :func:`simplicial_one_laplacian_matvec` / ``B_1``: one column per undirected
    edge. With identity diagonals this reduces to combinatorial
    ``L_1 = B_1 B_1^T``. General (non-diagonal) restriction maps are
    available via :func:`general_sheaf_laplacian_matvec` (channel ceiling
    documented on the sheaf encoder).

    Coboundary on edge ``e = (s → t)``::

        (δx)_e = r_t ⊙ x_t - r_s ⊙ x_s

    then ``L_F x = δ* δ x`` with the adjoint that scatters ``r ⊙ (δx)`` back
    to endpoints. For dense linear maps see
    :func:`general_sheaf_laplacian_matvec`.

    Parameters
    ----------
    edge_index : Tensor
        Oriented edges ``(2, E)``.
    x : Tensor
        Node features ``(num_nodes, channels)``.
    source_diag : Tensor
        Diagonal restriction at edge sources, shape ``(channels,)``.
    target_diag : Tensor
        Diagonal restriction at edge targets, shape ``(channels,)``.
    num_nodes : int or None, optional
        Node count. Defaults to ``x.size(0)``.

    Returns
    -------
    Tensor
        Features ``L_F @ x`` with the same shape as ``x``.

    Raises
    ------
    ValueError
        If shapes of ``x`` / diagonals are incompatible.
    """
    if x.ndim != 2:
        msg = f"x must be 2D, got shape {tuple(x.shape)}"
        raise ValueError(msg)
    channels = int(x.size(1))
    if source_diag.shape != (channels,) or target_diag.shape != (channels,):
        msg = (
            f"source_diag/target_diag must have shape ({channels},), "
            f"got {tuple(source_diag.shape)} / {tuple(target_diag.shape)}"
        )
        raise ValueError(msg)
    resolved_nodes = x.size(0) if num_nodes is None else int(num_nodes)
    if resolved_nodes != x.size(0):
        msg = f"num_nodes={resolved_nodes} does not match x.size(0)={x.size(0)}"
        raise ValueError(msg)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        msg = (
            f"edge_index must have shape (2, num_edges), got {tuple(edge_index.shape)}"
        )
        raise ValueError(msg)

    src = edge_index[0].to(dtype=torch.long)
    tgt = edge_index[1].to(dtype=torch.long)
    source_diag = source_diag.to(dtype=x.dtype, device=x.device)
    target_diag = target_diag.to(dtype=x.dtype, device=x.device)
    # δx on edges, then δ* back to nodes (diagonal real maps: R^T = R).
    edge_signal = target_diag * x[tgt] - source_diag * x[src]
    out = torch.zeros_like(x)
    out.index_add_(0, tgt, target_diag * edge_signal)
    out.index_add_(0, src, -source_diag * edge_signal)
    return out


def general_sheaf_laplacian_matvec(
    edge_index: Tensor,
    x: Tensor,
    source_map: Tensor,
    target_map: Tensor,
    *,
    num_nodes: int | None = None,
) -> Tensor:
    """Apply a sheaf Laplacian with general linear restriction maps.

    Uses the same oriented-edge convention as
    :func:`diagonal_sheaf_laplacian_matvec`. Coboundary on edge
    ``e = (s → t)``::

        (δx)_e = R_t x_t - R_s x_s

    then ``L_F x = δ* δ x`` with adjoints ``R^T`` scattering edge signals
    back to endpoints. Identity maps recover combinatorial ``L_1``.

    Parameter cost is ``O(C^2)`` per endpoint map (vs ``O(C)`` for the
    diagonal MVP). Callers should enforce a channel ceiling before allocating
    learned maps.

    Parameters
    ----------
    edge_index : Tensor
        Oriented edges ``(2, E)``.
    x : Tensor
        Node features ``(num_nodes, channels)``.
    source_map : Tensor
        Restriction at edge sources, shape ``(channels, channels)``.
    target_map : Tensor
        Restriction at edge targets, shape ``(channels, channels)``.
    num_nodes : int or None, optional
        Node count. Defaults to ``x.size(0)``.

    Returns
    -------
    Tensor
        Features ``L_F @ x`` with the same shape as ``x``.

    Raises
    ------
    ValueError
        If shapes of ``x`` / maps are incompatible.
    """
    if x.ndim != 2:
        msg = f"x must be 2D, got shape {tuple(x.shape)}"
        raise ValueError(msg)
    channels = int(x.size(1))
    expected = (channels, channels)
    if source_map.shape != expected or target_map.shape != expected:
        msg = (
            f"source_map/target_map must have shape {expected}, "
            f"got {tuple(source_map.shape)} / {tuple(target_map.shape)}"
        )
        raise ValueError(msg)
    resolved_nodes = x.size(0) if num_nodes is None else int(num_nodes)
    if resolved_nodes != x.size(0):
        msg = f"num_nodes={resolved_nodes} does not match x.size(0)={x.size(0)}"
        raise ValueError(msg)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        msg = (
            f"edge_index must have shape (2, num_edges), got {tuple(edge_index.shape)}"
        )
        raise ValueError(msg)

    src = edge_index[0].to(dtype=torch.long)
    tgt = edge_index[1].to(dtype=torch.long)
    source_map = source_map.to(dtype=x.dtype, device=x.device)
    target_map = target_map.to(dtype=x.dtype, device=x.device)
    # Batched R @ x_i  ==  x_batch @ R.T ; adjoint R.T @ s  ==  s @ R.
    edge_signal = x[tgt] @ target_map.transpose(0, 1) - x[src] @ source_map.transpose(
        0, 1
    )
    out = torch.zeros_like(x)
    out.index_add_(0, tgt, edge_signal @ target_map)
    out.index_add_(0, src, -(edge_signal @ source_map))
    return out


def simplicial_one_laplacian_matvec(
    edge_index: Tensor,
    x: Tensor,
    *,
    num_nodes: int | None = None,
) -> Tensor:
    """Apply combinatorial ``L_1 = B_1 B_1^T`` as ``B_1 @ (B_1.T @ x)``.

    This is the simplicial-1 / combinatorial graph Laplacian associated with
    the oriented edges — **not** the symmetrically normalized ``L_sym`` used by
    :func:`graph_laplacian_features`, and not a sheaf Laplacian (see
    :func:`diagonal_sheaf_laplacian_matvec` for the diagonal-restriction MVP).

    Parameters
    ----------
    edge_index : Tensor
        Oriented edges ``(2, E)`` (one column per undirected edge).
    x : Tensor
        Node features with shape ``(num_nodes, in_channels)``.
    num_nodes : int or None, optional
        Node count. Defaults to ``x.size(0)``.

    Returns
    -------
    Tensor
        Features ``L_1 @ x`` with the same shape as ``x``.
    """
    if x.ndim != 2:
        msg = f"x must be 2D, got shape {tuple(x.shape)}"
        raise ValueError(msg)
    resolved_nodes = x.size(0) if num_nodes is None else int(num_nodes)
    if resolved_nodes != x.size(0):
        msg = f"num_nodes={resolved_nodes} does not match x.size(0)={x.size(0)}"
        raise ValueError(msg)
    incidence = boundary_incidence_b1(edge_index, num_nodes=resolved_nodes).to(
        dtype=x.dtype,
        device=x.device,
    )
    edge_signal = incidence.T @ x
    return incidence @ edge_signal


def simplicial_one_laplacian_features(data: Data) -> Tensor:
    """Return combinatorial 1-Laplacian features ``L_1 @ x``.

    Requires ``data.x`` and oriented ``data.edge_index``. Optional
    ``data.face_index`` is validated when present. Output shape is
    ``(num_nodes, in_channels)`` so ``physics_dim`` equals ``in_channels``.

    Parameters
    ----------
    data : Data
        Graph snapshot with node features and oriented edges.

    Returns
    -------
    Tensor
        Simplicial-1 Laplacian features with the same shape as ``data.x``.
    """
    if data.x is None:
        msg = "data.x is required for simplicial_one_laplacian_features"
        raise ValueError(msg)
    if data.x.dim() != 2:
        msg = f"data.x must be 2D, got shape {tuple(data.x.shape)}"
        raise ValueError(msg)
    if data.edge_index is None:
        msg = "data.edge_index is required for simplicial_one_laplacian_features"
        raise ValueError(msg)
    _maybe_validate_face_index(data, num_nodes=int(data.x.size(0)))
    return simplicial_one_laplacian_matvec(
        data.edge_index,
        data.x,
        num_nodes=int(data.x.size(0)),
    )


def hodge_gradient_features(data: Data) -> Tensor:
    r"""Return nodewise Hodge-gradient magnitudes from oriented edges.

    Computes the edge signal ``y = B_1^\top x`` and aggregates the RMS of
    incident edge values onto each node (via ``|B_1|``). Isolated nodes map to
    zero. This is a simplicial-1 Hodge-style gradient energy — not a sheaf
    operator and not :func:`graph_gradient_features`.

    Parameters
    ----------
    data : Data
        Graph snapshot with two-dimensional ``x`` and oriented ``edge_index``.
        Optional ``face_index`` is validated when present.

    Returns
    -------
    Tensor
        Non-negative node features with shape ``(num_nodes, in_channels)``.
    """
    if data.x is None:
        msg = "data.x is required for hodge_gradient_features"
        raise ValueError(msg)
    if data.x.dim() != 2:
        msg = f"data.x must be 2D, got shape {tuple(data.x.shape)}"
        raise ValueError(msg)
    if data.edge_index is None:
        msg = "data.edge_index is required for hodge_gradient_features"
        raise ValueError(msg)
    x = data.x
    num_nodes = int(x.size(0))
    _maybe_validate_face_index(data, num_nodes=num_nodes)
    incidence = boundary_incidence_b1(data.edge_index, num_nodes=num_nodes).to(
        dtype=x.dtype,
        device=x.device,
    )
    edge_signal = incidence.T @ x
    energy = incidence.abs() @ edge_signal.square()
    return energy.clamp_min(0).sqrt()


PHYSICS_PRESETS["graph_gradient"] = graph_gradient_features
PHYSICS_PRESETS["graph_curvature"] = graph_curvature_features
PHYSICS_PRESETS["graph_laplacian"] = graph_laplacian_features
PHYSICS_PRESETS["hodge_gradient"] = hodge_gradient_features
PHYSICS_PRESETS["simplicial_one_laplacian"] = simplicial_one_laplacian_features


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
        ``"graph_gradient"``, ``"graph_curvature"``, ``"hodge_gradient"``,
        or ``"simplicial_one_laplacian"``. Dynamic polynomial presets use
        ``"polynomial(degree)"``, for example ``"polynomial(3)"``.
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
