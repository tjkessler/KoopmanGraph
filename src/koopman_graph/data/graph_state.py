"""Graph-state snapshots and frozen dynamics config.

This module must not import :mod:`koopman_graph.nn`. Topology heads live
in :mod:`koopman_graph.nn.predicted_topology`; this file only records
targets and the config that selects a head kind.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor

from koopman_graph.data.containers import GraphSnapshotSequence
from koopman_graph.graph_utils import snapshot_edge_weight

TopologyHeadKind = Literal["none", "sparse_candidate", "dense_mlp"]

TOPOLOGY_HEAD_KINDS: frozenset[str] = frozenset(
    {"none", "sparse_candidate", "dense_mlp"}
)
DEFAULT_CANDIDATE_K = 8
_CONFIG_KEYS: frozenset[str] = frozenset(
    {
        "topology_head",
        "recursive_training",
        "topology_loss_weight",
        "presence_loss_weight",
        "candidate_k",
    }
)

__all__ = [
    "DEFAULT_CANDIDATE_K",
    "TOPOLOGY_HEAD_KINDS",
    "GraphDynamicsConfig",
    "GraphStateSnapshot",
    "TopologyHeadKind",
    "graph_dynamics_from_mapping",
    "graph_state_at",
]


@dataclass(frozen=True, eq=False)
class GraphStateSnapshot:
    """One homogeneous graph-state target at a single timestep.

    Attributes
    ----------
    x : Tensor
        Node features ``(N, F)``.
    edge_index : Tensor
        COO edges ``(2, E)``.
    edge_weight : Tensor or None
        Optional edge weights ``(E,)``.
    presence : Tensor
        Boolean presence mask ``(N,)``. All-true when the sequence has
        no presence masks.
    entity_ids : tuple of str or None
        Optional fixed-union row names.

    Notes
    -----
    This is a supervision record, not a replacement for
    :class:`~koopman_graph.data.GraphSnapshotSequence`. Equality is
    disabled because the payload holds tensors.
    """

    x: Tensor
    edge_index: Tensor
    edge_weight: Tensor | None
    presence: Tensor
    entity_ids: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        """Validate shapes of the stored tensors.

        Raises
        ------
        ValueError
            If a tensor rank or length is inconsistent.
        """
        if self.x.ndim != 2:
            msg = f"x must have shape (N, F), got {tuple(self.x.shape)}"
            raise ValueError(msg)
        num_nodes = int(self.x.shape[0])
        if self.edge_index.ndim != 2 or int(self.edge_index.shape[0]) != 2:
            msg = (
                f"edge_index must have shape (2, E), got {tuple(self.edge_index.shape)}"
            )
            raise ValueError(msg)
        num_edges = int(self.edge_index.shape[1])
        if self.edge_weight is not None and (
            self.edge_weight.ndim != 1 or int(self.edge_weight.shape[0]) != num_edges
        ):
            msg = (
                f"edge_weight must have shape (E,), got {tuple(self.edge_weight.shape)}"
            )
            raise ValueError(msg)
        if self.presence.ndim != 1 or int(self.presence.shape[0]) != num_nodes:
            msg = f"presence must have shape (N,), got {tuple(self.presence.shape)}"
            raise ValueError(msg)
        if self.entity_ids is not None and len(self.entity_ids) != num_nodes:
            msg = (
                "entity_ids length must equal N, "
                f"got {len(self.entity_ids)} != {num_nodes}"
            )
            raise ValueError(msg)


@dataclass(frozen=True)
class GraphDynamicsConfig:
    """Opt-in graph-state closure options.

    Default ``topology_head="sparse_candidate"`` scores at most
    ``candidate_k`` destinations per node. ``dense_mlp`` keeps
    :class:`~koopman_graph.nn.PredictedTopologyHead` as a power-user
    dense :math:`N\\times N` path with an :math:`N` ceiling.
    ``learn_topology="self_adaptive"`` / :class:`~koopman_graph.nn.AdaptiveAdjacency`
    is a separate static Graph WaveNet formula and is unchanged.

    Attributes
    ----------
    topology_head : {"none", "sparse_candidate", "dense_mlp"}
        Head kind. Default ``"sparse_candidate"``.
    recursive_training : bool
        When ``True``, predicted :math:`\\hat A_{t+1}` feeds the next
        operator step when ``future_topologies`` is omitted and
        ``topology_policy="auto"``.
    topology_loss_weight : float
        Non-negative finite weight for structural BCE on candidate
        pairs. Dimensionless. Default ``1.0``.
    presence_loss_weight : float
        Non-negative finite weight for presence BCE. Dimensionless.
        Default ``1.0``.
    candidate_k : int
        Maximum destinations scored per source under
        ``sparse_candidate``. Default ``8``.
    """

    topology_head: TopologyHeadKind = "sparse_candidate"
    recursive_training: bool = True
    topology_loss_weight: float = 1.0
    presence_loss_weight: float = 1.0
    candidate_k: int = DEFAULT_CANDIDATE_K

    def __post_init__(self) -> None:
        """Validate head kind, flags, weights, and ``candidate_k``.

        Raises
        ------
        ValueError
            If a field is outside the allowed set or range.
        """
        if self.topology_head not in TOPOLOGY_HEAD_KINDS:
            allowed = ", ".join(sorted(TOPOLOGY_HEAD_KINDS))
            msg = (
                "topology_head must be one of "
                f"{{{allowed}}}; got {self.topology_head!r}"
            )
            raise ValueError(msg)
        if type(self.recursive_training) is not bool:
            msg = (
                "recursive_training must be a bool, got "
                f"{type(self.recursive_training).__name__}"
            )
            raise ValueError(msg)
        for name in ("topology_loss_weight", "presence_loss_weight"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                msg = f"{name} must be a real float, got {type(value).__name__}"
                raise ValueError(msg)
            if not math.isfinite(float(value)) or float(value) < 0.0:
                msg = f"{name} must be a finite non-negative float, got {value!r}"
                raise ValueError(msg)
        if type(self.candidate_k) is not int or self.candidate_k < 1:
            msg = f"candidate_k must be a positive int, got {self.candidate_k!r}"
            raise ValueError(msg)

    def to_mapping(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping for format-1 checkpoints.

        Returns
        -------
        dict
            Keys ``topology_head``, ``recursive_training``,
            ``topology_loss_weight``, ``presence_loss_weight``,
            ``candidate_k``.
        """
        return {
            "topology_head": self.topology_head,
            "recursive_training": self.recursive_training,
            "topology_loss_weight": float(self.topology_loss_weight),
            "presence_loss_weight": float(self.presence_loss_weight),
            "candidate_k": int(self.candidate_k),
        }


def graph_dynamics_from_mapping(
    payload: Mapping[str, Any] | None,
) -> GraphDynamicsConfig | None:
    """Build a config from a checkpoint mapping.

    Parameters
    ----------
    payload : mapping or None
        Stored ``graph_dynamics`` block. ``None`` means the 0.14 default
        (no graph-state head).

    Returns
    -------
    GraphDynamicsConfig or None
        Parsed config, or ``None`` when ``payload`` is ``None``.

    Raises
    ------
    TypeError
        If ``payload`` is neither ``None`` nor a mapping.
    ValueError
        If unknown keys are present or field validation fails.
    """
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        msg = f"graph_dynamics must be a mapping or None, got {type(payload).__name__}"
        raise TypeError(msg)
    unknown = set(payload) - _CONFIG_KEYS
    if unknown:
        extra = ", ".join(sorted(str(key) for key in unknown))
        msg = f"unknown graph_dynamics keys: {extra}"
        raise ValueError(msg)
    return GraphDynamicsConfig(
        topology_head=payload.get("topology_head", "sparse_candidate"),
        recursive_training=payload.get("recursive_training", True),
        topology_loss_weight=payload.get("topology_loss_weight", 1.0),
        presence_loss_weight=payload.get("presence_loss_weight", 1.0),
        candidate_k=payload.get("candidate_k", DEFAULT_CANDIDATE_K),
    )


def graph_state_at(sequence: GraphSnapshotSequence, index: int) -> GraphStateSnapshot:
    """Return the graph-state target at ``index``.

    Parameters
    ----------
    sequence : GraphSnapshotSequence
        Homogeneous snapshot sequence.
    index : int
        Timestep. Negative indices follow Python sequence rules.

    Returns
    -------
    GraphStateSnapshot
        Features, edges, presence, and optional entity ids.

    Raises
    ------
    IndexError
        If ``index`` is out of range.
    TypeError
        If ``sequence`` is not a :class:`GraphSnapshotSequence`.
    """
    if not isinstance(sequence, GraphSnapshotSequence):
        msg = (
            "graph_state_at expects a GraphSnapshotSequence, got "
            f"{type(sequence).__name__}"
        )
        raise TypeError(msg)
    n_steps = len(sequence)
    resolved = int(index) if int(index) >= 0 else n_steps + int(index)
    if resolved < 0 or resolved >= n_steps:
        msg = f"graph-state index {index} is out of range for {n_steps} timesteps"
        raise IndexError(msg)
    snapshot = sequence[resolved]
    features = snapshot.x
    num_nodes = int(features.shape[0])
    if sequence.has_presence_masks:
        presence = sequence.presence_mask_at(resolved)
    else:
        presence = torch.ones(num_nodes, dtype=torch.bool, device=features.device)
    entity_ids: tuple[str, ...] | None = None
    if sequence.has_entity_ids:
        raw = sequence.entity_ids
        assert raw is not None
        entity_ids = tuple(str(item) for item in raw)
    return GraphStateSnapshot(
        x=features,
        edge_index=snapshot.edge_index,
        edge_weight=snapshot_edge_weight(snapshot),
        presence=presence,
        entity_ids=entity_ids,
    )
