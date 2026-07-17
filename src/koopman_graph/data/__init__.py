"""Utilities for spatiotemporal graph snapshot sequences.

Capability layout
-----------------
``containers``
    ``GraphSnapshotSequence``, ``MultiTrajectory``, and related normalizers.
``sampling``
    ``WindowSampler`` for fixed-length temporal windows.
``splits``
    ``TemporalSplit`` and :func:`temporal_split`.
``rollout``
    Shared rollout-origin resolution for training and forecast evaluation.
"""

from koopman_graph.data.containers import (
    GraphSnapshotSequence,
    MultiTrajectory,
    as_multi_trajectory,
    resolve_pair_delta_t,
    resolve_sequence,
)
from koopman_graph.data.rollout import (
    RolloutStartIndices,
    resolve_rollout_start_indices,
)
from koopman_graph.data.sampling import WindowSampler
from koopman_graph.data.splits import TemporalSplit, temporal_split

__all__ = [
    "GraphSnapshotSequence",
    "MultiTrajectory",
    "RolloutStartIndices",
    "TemporalSplit",
    "WindowSampler",
    "as_multi_trajectory",
    "resolve_pair_delta_t",
    "resolve_rollout_start_indices",
    "resolve_sequence",
    "temporal_split",
]
