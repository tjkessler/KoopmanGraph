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
from koopman_graph.graph_utils.topology import (
    degree_support_mask,
    dense_symmetric_normalized_adjacency,
    dense_symmetric_normalized_laplacian,
    node_degrees,
    resolve_edge_index,
    resolve_edge_weight,
    resolve_graph_inputs,
    snapshot_edge_weight,
    snapshot_to_device,
    symmetric_normalized_adjacency_edge_weights,
    symmetric_normalized_adjacency_matvec,
    symmetric_normalized_laplacian_matvec,
)

from . import propagation, topology

__all__ = [
    "ControlAtFn",
    "DecoderFn",
    "DeltaTAtFn",
    "KoopmanPropagator",
    "TopologyAtFn",
    "advance_and_decode",
    "autoregressive_latent_rollout",
    "degree_support_mask",
    "dense_symmetric_normalized_adjacency",
    "dense_symmetric_normalized_laplacian",
    "hold_last_topology_at",
    "inverse_propagate_latent",
    "node_degrees",
    "pack_rollout_snapshots",
    "propagate_latent",
    "resolve_delta_t",
    "resolve_edge_index",
    "resolve_edge_weight",
    "resolve_graph_inputs",
    "snapshot_edge_weight",
    "snapshot_to_device",
    "snapshot_topology_at",
    "symmetric_normalized_adjacency_edge_weights",
    "symmetric_normalized_adjacency_matvec",
    "symmetric_normalized_laplacian_matvec",
]
