"""Collate independent homogeneous snapshot sequences into one PyG ``Batch``.

Power-user helper for opt-in multi-graph training. Shared
:math:`K_{\\mathrm{self}}` / :math:`K_k` apply to the disconnected union;
per-graph shifts stay block-diagonal because ``Batch.from_data_list`` offsets
``edge_index``. This module lives in :mod:`koopman_graph.data` and must not
import :mod:`koopman_graph.nn`.

Component order is **time-major then trajectory**: snapshot ``(t, g)`` is
component ``t * num_trajectories + g``. :attr:`time_index` and
:attr:`graph_index` have length ``num_trajectories * num_timesteps``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor
from torch_geometric.data import Batch, Data

from koopman_graph.data.containers import GraphSnapshotSequence

__all__ = [
    "BatchedGraphTrajectory",
    "collate_graph_trajectories",
]


@dataclass(frozen=True)
class BatchedGraphTrajectory:
    """Disconnected union of ``(trajectory, time)`` homogeneous snapshots.

    Attributes
    ----------
    batch : Batch
        PyG batch whose graphs are the flattened ``(t, g)`` snapshots.
    time_index : Tensor
        Integer time of each component, shape ``(num_components,)``.
    graph_index : Tensor
        Integer trajectory index of each component, shape
        ``(num_components,)``.
    """

    batch: Batch
    time_index: Tensor
    graph_index: Tensor

    @property
    def num_trajectories(self) -> int:
        """Return the number of collated trajectories.

        Returns
        -------
        int
            ``1 + max(graph_index)`` when any components exist, else ``0``.
        """
        if self.graph_index.numel() == 0:
            return 0
        return int(self.graph_index.max().item()) + 1

    @property
    def num_timesteps(self) -> int:
        """Return the shared number of snapshots per trajectory.

        Returns
        -------
        int
            ``1 + max(time_index)`` when any components exist, else ``0``.
        """
        if self.time_index.numel() == 0:
            return 0
        return int(self.time_index.max().item()) + 1

    @property
    def num_components(self) -> int:
        """Return the number of disjoint ``(t, g)`` graphs in :attr:`batch`.

        Returns
        -------
        int
            Length of :attr:`time_index`.
        """
        return int(self.time_index.numel())

    def graphs_at_time(self, timestep: int) -> Batch:
        """Return the disconnected union of all trajectories at ``timestep``.

        The returned ``Batch.batch`` vector is ``0 … G-1`` in trajectory
        order. Edges do not cross graphs.

        Parameters
        ----------
        timestep : int
            Time index in ``[0, num_timesteps)``.

        Returns
        -------
        Batch
            One component per trajectory at ``timestep``.

        Raises
        ------
        ValueError
            If ``timestep`` is out of range.
        """
        n_times = self.num_timesteps
        if timestep < 0 or timestep >= n_times:
            msg = f"timestep must be in [0, {n_times}), got {timestep}"
            raise ValueError(msg)
        n_traj = self.num_trajectories
        start = timestep * n_traj
        data_list = self.batch.to_data_list()
        return Batch.from_data_list(data_list[start : start + n_traj])


def _validate_collate_sequences(
    sequences: Sequence[GraphSnapshotSequence],
) -> tuple[int, int]:
    """Return ``(num_trajectories, num_timesteps)`` after layout checks.

    Parameters
    ----------
    sequences : sequence of GraphSnapshotSequence
        Candidate trajectories.

    Returns
    -------
    tuple of int
        Trajectory count and shared snapshot count.

    Raises
    ------
    TypeError
        If any element is not a :class:`GraphSnapshotSequence`.
    ValueError
        If the input is empty, lengths or feature widths differ, or a
        sequence carries controls or hyperedges.
    """
    if not sequences:
        msg = "collate_graph_trajectories requires at least one sequence"
        raise ValueError(msg)
    for index, sequence in enumerate(sequences):
        if not isinstance(sequence, GraphSnapshotSequence):
            msg = (
                "collate_graph_trajectories requires GraphSnapshotSequence "
                f"instances; index {index} has type {type(sequence).__name__}"
            )
            raise TypeError(msg)
        if sequence.has_controls:
            msg = (
                "collate_graph_trajectories does not support control_inputs; "
                f"sequence {index} has controls"
            )
            raise ValueError(msg)
        if sequence.has_hyperedges:
            msg = (
                "collate_graph_trajectories does not support hyperedge "
                f"incidence; sequence {index} has hyperedges"
            )
            raise ValueError(msg)
        if sequence.has_timestamps:
            msg = (
                "collate_graph_trajectories does not support timestamps; "
                f"sequence {index} has timestamps"
            )
            raise ValueError(msg)
        if sequence.has_parameter_trajectory:
            msg = (
                "collate_graph_trajectories does not support "
                "parameter_trajectory; "
                f"sequence {index} has parameter_trajectory"
            )
            raise ValueError(msg)
    lengths = [sequence.num_timesteps for sequence in sequences]
    if len(set(lengths)) != 1:
        msg = f"collate_graph_trajectories requires equal num_timesteps; got {lengths}"
        raise ValueError(msg)
    channels = [sequence.in_channels for sequence in sequences]
    if len(set(channels)) != 1:
        msg = f"collate_graph_trajectories requires equal in_channels; got {channels}"
        raise ValueError(msg)
    return len(sequences), lengths[0]


def collate_graph_trajectories(
    sequences: Sequence[GraphSnapshotSequence],
) -> BatchedGraphTrajectory:
    """Collate homogeneous trajectories into one disconnected PyG ``Batch``.

    Snapshots are cloned so the source sequences are not mutated with a
    ``batch`` vector. Component ``t * G + g`` is trajectory ``g`` at time
    ``t``.

    Parameters
    ----------
    sequences : sequence of GraphSnapshotSequence
        Non-empty homogeneous trajectories sharing ``num_timesteps`` and
        ``in_channels``. Node counts may differ across trajectories.

    Returns
    -------
    BatchedGraphTrajectory
        Union batch plus per-component time and graph index tensors.

    Raises
    ------
    TypeError
        If any element is not a :class:`GraphSnapshotSequence`.
    ValueError
        If sequences are empty, lengths differ, feature widths differ, or
        a sequence carries controls, timestamps, or hyperedges.
    """
    num_trajectories, num_timesteps = _validate_collate_sequences(sequences)
    data_list: list[Data] = []
    time_ids: list[int] = []
    graph_ids: list[int] = []
    for timestep in range(num_timesteps):
        for graph_id, sequence in enumerate(sequences):
            snapshot = sequence[timestep]
            data_list.append(snapshot.clone())
            time_ids.append(timestep)
            graph_ids.append(graph_id)
    batch = Batch.from_data_list(data_list)
    device = batch.x.device if batch.x is not None else torch.device("cpu")
    time_index = torch.tensor(time_ids, dtype=torch.long, device=device)
    graph_index = torch.tensor(graph_ids, dtype=torch.long, device=device)
    return BatchedGraphTrajectory(
        batch=batch,
        time_index=time_index,
        graph_index=graph_index,
    )
