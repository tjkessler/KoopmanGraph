"""Utilities for spatiotemporal graph snapshot sequences.

Capability layout
-----------------
``containers``
    ``GraphSnapshotSequence``, ``HeteroGraphSnapshotSequence``.
``construction``
    Array / dynamic-array / Hankel-window builders used by container
    classmethods. Power-user submodule; not promoted on this package
    ``__all__``.
``validation``
    Coercion and validation helpers for snapshot construction.
``hetero_layout``
    Stacked typed-node layout contract: ``node_type_slices``,
    ``node_type_offsets``, ``latent_type_slices``,
    ``validate_latent_dims``, ``latent_type_slices_from_dims``,
    ``stacked_latent_numel``, ``stack_typed_features`` /
    ``unstack_typed_features``, ``offset_edge_index``,
    ``global_relation_edge_indices``, ``snapshot_num_nodes_dict``, and
    ``mask_hetero_snapshot_features``.
``trajectories``
    ``MultiTrajectory``, ``as_multi_trajectory``, ``resolve_sequence``,
    ``resolve_hetero_sequence``, ``SnapshotSequence``, and
    ``resolve_pair_delta_t``.
``batching``
    Opt-in PyG ``Batch`` collate for independent homogeneous trajectories
    (``BatchedGraphTrajectory``, ``collate_graph_trajectories``). Package
    export; not on root ``__all__``. Must not import ``nn``.
``graph_state``
    Frozen ``GraphDynamicsConfig`` and ``GraphStateSnapshot`` supervision
    records for opt-in graph-state closure. Package export; not on root
    ``__all__``. Must not import ``nn``.
``conditioning``
    Frozen ``ConditioningContext`` and :func:`conditioning_at` for
    per-snapshot :math:`(\\mu, t, u)` records. Homogeneous
    ``parameter_trajectory`` only. Package export; not on root
    ``__all__``. Must not import ``nn``. Distinct from switched /
    mixture operators.
``calendar``
    :func:`~koopman_graph.data.diurnal_control_features` and
    :func:`~koopman_graph.data.diurnal_phase_index` time-of-day
    recipes for existing control / switched ``phase_index``. Package
    export; not on root ``__all__``. Not a calendar serializer. Must
    not import ``nn``.
``remap``
    :class:`~koopman_graph.data.EntityRemap` and
    :func:`~koopman_graph.data.remap_node_features` for injective
    placement into a finite :math:`N_{\\max}`. Not entity resolution;
    unbounded growth is refused. Package export; not on root ``__all__``.
``delay_windows``
    Hankel-style stack / flatten / observation-mask helpers shared by
    containers, ``nn.delay``, and model encode paths. Power-user submodule;
    not promoted on this package ``__all__``.
``sampling``
    ``WindowSampler`` for fixed-length temporal windows;
    ``NeighborWindowSampler`` for k-hop subgraph windows on large static
    graphs (training approximation; not on root ``__all__``);
    :func:`~koopman_graph.data.resolve_window_sampler` shared local /
    distributed construction entry point;
    :class:`~koopman_graph.data.WindowOrigin` /
    :func:`~koopman_graph.data.build_window_index_list` for pure origin
    planning (also used by distributed samplers).
``splits``
    ``TemporalSplit`` and :func:`temporal_split`.
``rollout``
    Shared rollout-origin resolution for training and forecast evaluation.
"""

from koopman_graph.data.batching import (
    BatchedGraphTrajectory,
    collate_graph_trajectories,
)
from koopman_graph.data.calendar import (
    diurnal_control_features,
    diurnal_phase_index,
)
from koopman_graph.data.conditioning import ConditioningContext, conditioning_at
from koopman_graph.data.containers import (
    GraphSnapshotSequence,
    HeteroGraphSnapshotSequence,
)
from koopman_graph.data.graph_state import (
    DEFAULT_CANDIDATE_K,
    GraphDynamicsConfig,
    GraphStateSnapshot,
    graph_dynamics_from_mapping,
    graph_state_at,
)
from koopman_graph.data.hetero_layout import (
    global_relation_edge_indices,
    latent_type_slices,
    latent_type_slices_from_dims,
    mask_hetero_snapshot_features,
    node_type_offsets,
    node_type_slices,
    offset_edge_index,
    snapshot_num_nodes_dict,
    stack_typed_features,
    stacked_latent_numel,
    unstack_typed_features,
    validate_latent_dims,
)
from koopman_graph.data.partition import (
    cluster_node_partition,
    induced_cluster_subgraph,
    iter_cluster_subgraphs,
)
from koopman_graph.data.remap import EntityRemap, remap_node_features
from koopman_graph.data.rollout import (
    RolloutStartIndices,
    resolve_rollout_start_indices,
)
from koopman_graph.data.sampling import (
    NeighborWindowSampler,
    WindowLikeSampler,
    WindowOrigin,
    WindowSampler,
    build_window_index_list,
    resolve_window_sampler,
)
from koopman_graph.data.splits import TemporalSplit, temporal_split
from koopman_graph.data.trajectories import (
    MultiTrajectory,
    SnapshotSequence,
    as_multi_trajectory,
    resolve_hetero_sequence,
    resolve_pair_delta_t,
    resolve_sequence,
)

__all__ = [
    "BatchedGraphTrajectory",
    "ConditioningContext",
    "DEFAULT_CANDIDATE_K",
    "EntityRemap",
    "GraphDynamicsConfig",
    "GraphSnapshotSequence",
    "GraphStateSnapshot",
    "HeteroGraphSnapshotSequence",
    "MultiTrajectory",
    "NeighborWindowSampler",
    "RolloutStartIndices",
    "SnapshotSequence",
    "TemporalSplit",
    "WindowLikeSampler",
    "WindowOrigin",
    "WindowSampler",
    "as_multi_trajectory",
    "build_window_index_list",
    "collate_graph_trajectories",
    "cluster_node_partition",
    "conditioning_at",
    "diurnal_control_features",
    "diurnal_phase_index",
    "induced_cluster_subgraph",
    "iter_cluster_subgraphs",
    "remap_node_features",
    "global_relation_edge_indices",
    "graph_dynamics_from_mapping",
    "graph_state_at",
    "latent_type_slices",
    "latent_type_slices_from_dims",
    "mask_hetero_snapshot_features",
    "node_type_offsets",
    "node_type_slices",
    "offset_edge_index",
    "resolve_hetero_sequence",
    "resolve_pair_delta_t",
    "resolve_rollout_start_indices",
    "resolve_sequence",
    "resolve_window_sampler",
    "snapshot_num_nodes_dict",
    "stack_typed_features",
    "stacked_latent_numel",
    "temporal_split",
    "unstack_typed_features",
    "validate_latent_dims",
]
