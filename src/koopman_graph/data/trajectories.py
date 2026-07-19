"""Multi-trajectory containers and sequence resolution helpers."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from torch_geometric.data import Data

from koopman_graph.data.containers import GraphSnapshotSequence


@dataclass(frozen=True)
class MultiTrajectory:
    """Explicit multi-trajectory container for training and validation input.

    Required for multi-trajectory
    :meth:`~koopman_graph.model.GraphKoopmanModel.fit` so multi-trajectory
    intent cannot be confused with a single trajectory of ``Data`` snapshots.

    Attributes
    ----------
    sequences : tuple of GraphSnapshotSequence
        Non-empty trajectories of the same system.
    """

    sequences: tuple[GraphSnapshotSequence, ...]

    def __post_init__(self) -> None:
        """Validate that ``sequences`` is a non-empty trajectory tuple.

        Raises
        ------
        ValueError
            If ``sequences`` is empty.
        TypeError
            If any element is not a :class:`GraphSnapshotSequence`.
        """
        if not self.sequences:
            msg = "MultiTrajectory requires at least one GraphSnapshotSequence"
            raise ValueError(msg)
        for index, sequence in enumerate(self.sequences):
            if not isinstance(sequence, GraphSnapshotSequence):
                msg = (
                    "MultiTrajectory sequences must be GraphSnapshotSequence "
                    f"instances; index {index} has type {type(sequence).__name__}"
                )
                raise TypeError(msg)

    def __len__(self) -> int:
        """Return the number of trajectories.

        Returns
        -------
        int
            Length of :attr:`sequences`.
        """
        return len(self.sequences)

    def __iter__(self) -> Iterator[GraphSnapshotSequence]:
        """Iterate over trajectories.

        Yields
        ------
        GraphSnapshotSequence
            Each trajectory in order.
        """
        return iter(self.sequences)

    def __getitem__(self, index: int) -> GraphSnapshotSequence:
        """Return the trajectory at ``index``.

        Parameters
        ----------
        index : int
            Trajectory index.

        Returns
        -------
        GraphSnapshotSequence
            Trajectory at the requested index.
        """
        return self.sequences[index]


def as_multi_trajectory(
    *trajectories: GraphSnapshotSequence | Sequence[GraphSnapshotSequence],
) -> MultiTrajectory:
    """Build a :class:`MultiTrajectory` from sequences or a sequence of sequences.

    Parameters
    ----------
    *trajectories
        Either one iterable of :class:`GraphSnapshotSequence`, or individual
        sequences passed as separate arguments.

    Returns
    -------
    MultiTrajectory
        Validated multi-trajectory container.

    Raises
    ------
    TypeError
        If arguments are not snapshot sequences.
    ValueError
        If no trajectories are provided.
    """
    if len(trajectories) == 1 and not isinstance(
        trajectories[0], GraphSnapshotSequence
    ):
        sequence_list = list(trajectories[0])
    else:
        sequence_list = list(trajectories)
    return MultiTrajectory(tuple(sequence_list))


def resolve_sequence(
    sequence: GraphSnapshotSequence | Sequence[Data],
) -> GraphSnapshotSequence:
    """Normalize input into a validated snapshot sequence.

    Wraps a plain sequence of ``Data`` snapshots in
    :class:`GraphSnapshotSequence`; existing sequences are returned unchanged.

    Parameters
    ----------
    sequence : GraphSnapshotSequence or sequence of Data
        Raw snapshot input from a training, baseline, or inference API.

    Returns
    -------
    GraphSnapshotSequence
        Validated sequence container.
    """
    if isinstance(sequence, GraphSnapshotSequence):
        return sequence
    return GraphSnapshotSequence(sequence)


def resolve_pair_delta_t(
    sequence: GraphSnapshotSequence,
    timestep: int,
    *,
    default_time_step: float,
) -> float:
    """Return the integration interval for transition ``timestep -> timestep + 1``.

    Parameters
    ----------
    sequence : GraphSnapshotSequence
        Snapshot sequence that may carry timestamps.
    timestep : int
        Source snapshot index.
    default_time_step : float
        Fallback interval when timestamps are absent.

    Returns
    -------
    float
        Positive integration interval.
    """
    if sequence.has_timestamps:
        return float(sequence.delta_t_at(timestep).item())
    return default_time_step
