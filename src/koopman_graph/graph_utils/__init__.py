"""Shared graph-input and latent-propagation helpers (power-user layer).

This package is the documented internal utility surface for helpers that were
previously duplicated or imported as leading-underscore symbols across modules.
It is importable as ``koopman_graph.graph_utils`` but is **not** part of the
stable public façade (not listed in ``koopman_graph.__all__``).

Capability layout
-----------------
``topology``
    Degree / adjacency / Laplacian mathematics and graph-input resolution
    (``resolve_graph_inputs``, ``snapshot_to_device``, ``L_sym = P - Â``,
    per-relation ``relation_degree_normalize``,
    ``synthesize_reverse_edge_types``,
    ``materialize_reverse_relation_edges``).
``propagation``
    Latent advance / inverse, decode helpers, autoregressive rollout, and
    decoded-rollout → ``Data`` packing (``propagate_latent``,
    ``autoregressive_latent_rollout``, ``pack_rollout_snapshots``, topology
    schedules, ``delta_t`` policy).
``symmetry``
    Node-orbit partitions for symmetry-adapted operators
    (``node_orbit_partition``, ``validate_orbit_partition``; ``[symmetry]``
    extra).
``representation``
    Exact-automorphism isotypic projectors
    (:func:`~koopman_graph.graph_utils.compute_isotypic_decomposition`;
    requires ``pynauty``; WL methods refused).

Prefer ``from koopman_graph.graph_utils import …``. Peer modules may be imported
directly for power-user work; do not reach into leading-underscore helpers
across module boundaries.
"""

from koopman_graph.graph_utils.propagation import (
    ControlAtFn,
    DecoderFn,
    DeltaTAtFn,
    KoopmanPropagator,
    ParametersAtFn,
    PresenceAtFn,
    RelationDecoderFn,
    RelationTopologyAtFn,
    TopologyAtFn,
    advance_and_decode,
    autoregressive_hetero_latent_rollout,
    autoregressive_latent_rollout,
    call_topology_at,
    hold_last_relation_topology_at,
    hold_last_topology_at,
    inverse_propagate_latent,
    pack_hetero_rollout_snapshots,
    pack_rollout_snapshots,
    propagate_latent,
    resolve_delta_t,
    snapshot_topology_at,
)
from koopman_graph.graph_utils.representation import (
    MAX_ISOTYPIC_NODES,
    IsotypicDecomposition,
    IsotypicMethod,
    compute_isotypic_decomposition,
)
from koopman_graph.graph_utils.symmetry import (
    OrbitMethod,
    OrbitPartition,
    apply_orbit_self,
    assemble_orbit_self_blocks,
    hyperedge_two_section,
    identity_orbit_partition,
    node_orbit_index,
    node_orbit_partition,
    validate_orbit_partition,
)
from koopman_graph.graph_utils.topology import (
    RELATION_NORMALIZATION_MODES,
    RandomWalkDirection,
    RelationNormalization,
    clear_hyperedge_cache,
    degree_support_mask,
    dense_hyperedge_backward_random_walk_adjacency,
    dense_hyperedge_dual_random_walk_adjacency,
    dense_hyperedge_dual_random_walk_factors,
    dense_hyperedge_forward_random_walk_adjacency,
    dense_hyperedge_normalized_adjacency,
    dense_random_walk_normalized_adjacency,
    dense_relation_normalized_adjacency,
    dense_symmetric_normalized_adjacency,
    dense_symmetric_normalized_laplacian,
    hyperedge_dual_random_walk_matvec,
    hyperedge_forward_random_walk_matvec,
    hyperedge_normalized_adjacency_matvec,
    hyperedge_normalized_incidence_weights,
    materialize_reverse_relation_edges,
    node_degrees,
    random_walk_normalized_adjacency_edge_weights,
    random_walk_normalized_adjacency_matvec,
    relation_degree_normalize,
    relation_normalized_adjacency_matvec,
    resolve_edge_index,
    resolve_edge_weight,
    resolve_graph_inputs,
    snapshot_edge_weight,
    snapshot_head_index,
    snapshot_hyperedge_index,
    snapshot_hyperedge_weight,
    snapshot_tail_index,
    snapshot_to_device,
    symmetric_normalized_adjacency_edge_weights,
    symmetric_normalized_adjacency_matvec,
    symmetric_normalized_laplacian_matvec,
    synthesize_reverse_edge_types,
)

__all__ = [
    "ControlAtFn",
    "DecoderFn",
    "DeltaTAtFn",
    "KoopmanPropagator",
    "IsotypicDecomposition",
    "IsotypicMethod",
    "MAX_ISOTYPIC_NODES",
    "OrbitMethod",
    "OrbitPartition",
    "ParametersAtFn",
    "PresenceAtFn",
    "RELATION_NORMALIZATION_MODES",
    "RandomWalkDirection",
    "RelationDecoderFn",
    "RelationNormalization",
    "RelationTopologyAtFn",
    "TopologyAtFn",
    "advance_and_decode",
    "apply_orbit_self",
    "assemble_orbit_self_blocks",
    "autoregressive_hetero_latent_rollout",
    "autoregressive_latent_rollout",
    "call_topology_at",
    "clear_hyperedge_cache",
    "compute_isotypic_decomposition",
    "degree_support_mask",
    "dense_hyperedge_backward_random_walk_adjacency",
    "dense_hyperedge_dual_random_walk_adjacency",
    "dense_hyperedge_dual_random_walk_factors",
    "dense_hyperedge_forward_random_walk_adjacency",
    "dense_hyperedge_normalized_adjacency",
    "dense_random_walk_normalized_adjacency",
    "dense_relation_normalized_adjacency",
    "dense_symmetric_normalized_adjacency",
    "dense_symmetric_normalized_laplacian",
    "hold_last_relation_topology_at",
    "hold_last_topology_at",
    "hyperedge_dual_random_walk_matvec",
    "hyperedge_forward_random_walk_matvec",
    "hyperedge_normalized_adjacency_matvec",
    "hyperedge_normalized_incidence_weights",
    "hyperedge_two_section",
    "identity_orbit_partition",
    "inverse_propagate_latent",
    "materialize_reverse_relation_edges",
    "node_degrees",
    "node_orbit_index",
    "node_orbit_partition",
    "pack_hetero_rollout_snapshots",
    "pack_rollout_snapshots",
    "propagate_latent",
    "random_walk_normalized_adjacency_edge_weights",
    "random_walk_normalized_adjacency_matvec",
    "relation_degree_normalize",
    "relation_normalized_adjacency_matvec",
    "resolve_delta_t",
    "synthesize_reverse_edge_types",
    "resolve_edge_index",
    "resolve_edge_weight",
    "resolve_graph_inputs",
    "snapshot_edge_weight",
    "snapshot_head_index",
    "snapshot_hyperedge_index",
    "snapshot_hyperedge_weight",
    "snapshot_tail_index",
    "snapshot_to_device",
    "snapshot_topology_at",
    "symmetric_normalized_adjacency_edge_weights",
    "symmetric_normalized_adjacency_matvec",
    "symmetric_normalized_laplacian_matvec",
    "validate_orbit_partition",
]
