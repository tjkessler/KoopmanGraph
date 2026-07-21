"""Utilities for spatiotemporal graph snapshot sequences.

Capability layout
-----------------
``containers``
    ``GraphSnapshotSequence``.
``construction``
    Array / dynamic-array / Hankel-window builders used by container
    classmethods. Power-user submodule; not promoted on this package
    ``__all__``.
``validation``
    Coercion and validation helpers for snapshot construction.
``trajectories``
    ``MultiTrajectory``, ``as_multi_trajectory``, ``resolve_sequence``,
    and ``resolve_pair_delta_t``.
``delay_windows``
    Hankel-style stack / flatten / observation-mask helpers shared by
    containers, ``nn.delay``, and model encode paths. Power-user submodule;
    not promoted on this package ``__all__``.
``sampling``
    ``WindowSampler`` for fixed-length temporal windows.
``splits``
    ``TemporalSplit`` and :func:`temporal_split`.
``rollout``
    Shared rollout-origin resolution for training and forecast evaluation.
"""

from koopman_graph.data.containers import GraphSnapshotSequence
from koopman_graph.data.rollout import (
    RolloutStartIndices,
    resolve_rollout_start_indices,
)
from koopman_graph.data.sampling import WindowSampler
from koopman_graph.data.splits import TemporalSplit, temporal_split
from koopman_graph.data.trajectories import (
    MultiTrajectory,
    as_multi_trajectory,
    resolve_pair_delta_t,
    resolve_sequence,
)

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
