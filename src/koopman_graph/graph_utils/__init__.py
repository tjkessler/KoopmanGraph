"""Shared graph-input and latent-propagation helpers (power-user layer).

This package is the documented internal utility surface for helpers that were
previously duplicated or imported as leading-underscore symbols across modules.
It is importable as ``koopman_graph.graph_utils`` but is **not** part of the
stable public façade (not listed in ``koopman_graph.__all__``).

Capability layout
-----------------
``topology``
    Degree / adjacency / Laplacian mathematics and graph-input resolution
    (``resolve_graph_inputs``, ``snapshot_to_device``, ``L_sym = P - Â``).
``propagation``
    Latent advance / inverse, decode helpers, autoregressive rollout, and
    decoded-rollout → ``Data`` packing (``propagate_latent``,
    ``autoregressive_latent_rollout``, ``pack_rollout_snapshots``, topology
    schedules, ``delta_t`` policy).
``symmetry``
    Node-orbit partitions for symmetry-adapted operators
    (``node_orbit_partition``, ``validate_orbit_partition``; ``[symmetry]``
    extra).

Prefer ``from koopman_graph.graph_utils import …``. Peer modules may be imported
directly for power-user work; do not reach into leading-underscore helpers
across module boundaries.
"""

from koopman_graph.graph_utils.propagation import (
    ControlAtFn,
    DecoderFn,
    DeltaTAtFn,
    KoopmanPropagator,
    TopologyAtFn,
    advance_and_decode,
    autoregressive_latent_rollout,
    hold_last_topology_at,
    inverse_propagate_latent,
    pack_rollout_snapshots,
    propagate_latent,
    resolve_delta_t,
    snapshot_topology_at,
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
    RandomWalkDirection,
    clear_hyperedge_cache,
    degree_support_mask,
    dense_hyperedge_normalized_adjacency,
    dense_random_walk_normalized_adjacency,
    dense_symmetric_normalized_adjacency,
    dense_symmetric_normalized_laplacian,
    hyperedge_normalized_adjacency_matvec,
    hyperedge_normalized_incidence_weights,
    node_degrees,
    random_walk_normalized_adjacency_edge_weights,
    random_walk_normalized_adjacency_matvec,
    resolve_edge_index,
    resolve_edge_weight,
    resolve_graph_inputs,
    snapshot_edge_weight,
    snapshot_hyperedge_index,
    snapshot_hyperedge_weight,
    snapshot_to_device,
    symmetric_normalized_adjacency_edge_weights,
    symmetric_normalized_adjacency_matvec,
    symmetric_normalized_laplacian_matvec,
)

__all__ = [
    "ControlAtFn",
    "DecoderFn",
    "DeltaTAtFn",
    "KoopmanPropagator",
    "OrbitMethod",
    "OrbitPartition",
    "RandomWalkDirection",
    "TopologyAtFn",
    "advance_and_decode",
    "apply_orbit_self",
    "assemble_orbit_self_blocks",
    "autoregressive_latent_rollout",
    "clear_hyperedge_cache",
    "degree_support_mask",
    "dense_hyperedge_normalized_adjacency",
    "dense_random_walk_normalized_adjacency",
    "dense_symmetric_normalized_adjacency",
    "dense_symmetric_normalized_laplacian",
    "hold_last_topology_at",
    "hyperedge_normalized_adjacency_matvec",
    "hyperedge_normalized_incidence_weights",
    "hyperedge_two_section",
    "identity_orbit_partition",
    "inverse_propagate_latent",
    "node_degrees",
    "node_orbit_index",
    "node_orbit_partition",
    "pack_rollout_snapshots",
    "propagate_latent",
    "random_walk_normalized_adjacency_edge_weights",
    "random_walk_normalized_adjacency_matvec",
    "resolve_delta_t",
    "resolve_edge_index",
    "resolve_edge_weight",
    "resolve_graph_inputs",
    "snapshot_edge_weight",
    "snapshot_hyperedge_index",
    "snapshot_hyperedge_weight",
    "snapshot_to_device",
    "snapshot_topology_at",
    "symmetric_normalized_adjacency_edge_weights",
    "symmetric_normalized_adjacency_matvec",
    "symmetric_normalized_laplacian_matvec",
    "validate_orbit_partition",
]
