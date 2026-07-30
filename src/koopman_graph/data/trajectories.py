"""Multi-trajectory containers and sequence resolution helpers."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from torch_geometric.data import Data, HeteroData

from koopman_graph.data.containers import (
    GraphSnapshotSequence,
    HeteroGraphSnapshotSequence,
)

SnapshotSequence = GraphSnapshotSequence | HeteroGraphSnapshotSequence


@dataclass(frozen=True)
class MultiTrajectory:
    """Explicit multi-trajectory container for training and validation input.

    Required for multi-trajectory
    :meth:`~koopman_graph.model.GraphKoopmanModel.fit` so multi-trajectory
    intent cannot be confused with a single trajectory of ``Data`` /
    ``HeteroData`` snapshots.

    All elements must be homogeneous
    :class:`~koopman_graph.data.GraphSnapshotSequence` **or** all
    :class:`~koopman_graph.data.HeteroGraphSnapshotSequence` (Q10). Mixing
    the two families in one container raises.

    Attributes
    ----------
    sequences : tuple of GraphSnapshotSequence or HeteroGraphSnapshotSequence
        Non-empty trajectories of the same system family.
    """

    sequences: tuple[SnapshotSequence, ...]

    def __post_init__(self) -> None:
        """Validate that ``sequences`` is a non-empty, unmixed trajectory tuple.

        Raises
        ------
        ValueError
            If ``sequences`` is empty or mixes homogeneous and hetero
            sequences.
        TypeError
            If any element is not a supported snapshot sequence type.
        """
        if not self.sequences:
            msg = "MultiTrajectory requires at least one snapshot sequence"
            raise ValueError(msg)

        homo_indices = [
            index
            for index, sequence in enumerate(self.sequences)
            if isinstance(sequence, GraphSnapshotSequence)
        ]
        hetero_indices = [
            index
            for index, sequence in enumerate(self.sequences)
            if isinstance(sequence, HeteroGraphSnapshotSequence)
        ]
        if len(homo_indices) + len(hetero_indices) != len(self.sequences):
            bad_index = next(
                index
                for index, sequence in enumerate(self.sequences)
                if not isinstance(
                    sequence,
                    (GraphSnapshotSequence, HeteroGraphSnapshotSequence),
                )
            )
            msg = (
                "MultiTrajectory sequences must be GraphSnapshotSequence or "
                "HeteroGraphSnapshotSequence instances; "
                f"index {bad_index} has type "
                f"{type(self.sequences[bad_index]).__name__}"
            )
            raise TypeError(msg)
        if homo_indices and hetero_indices:
            msg = (
                "MultiTrajectory cannot mix GraphSnapshotSequence and "
                "HeteroGraphSnapshotSequence; got homogeneous indices "
                f"{homo_indices} and hetero indices {hetero_indices}"
            )
            raise ValueError(msg)

    def __len__(self) -> int:
        """Return the number of trajectories.

        Returns
        -------
        int
            Length of :attr:`sequences`.
        """
        return len(self.sequences)

    def __iter__(self) -> Iterator[SnapshotSequence]:
        """Iterate over trajectories.

        Yields
        ------
        GraphSnapshotSequence or HeteroGraphSnapshotSequence
            Each trajectory in order.
        """
        return iter(self.sequences)

    def __getitem__(self, index: int) -> SnapshotSequence:
        """Return the trajectory at ``index``.

        Parameters
        ----------
        index : int
            Trajectory index.

        Returns
        -------
        GraphSnapshotSequence or HeteroGraphSnapshotSequence
            Trajectory at the requested index.
        """
        return self.sequences[index]


def as_multi_trajectory(
    *trajectories: SnapshotSequence | Sequence[SnapshotSequence],
) -> MultiTrajectory:
    """Build a :class:`MultiTrajectory` from sequences or a sequence of sequences.

    Parameters
    ----------
    *trajectories
        Either one iterable of snapshot sequences, or individual sequences
        passed as separate arguments. All elements must be the same family
        (homogeneous or hetero).

    Returns
    -------
    MultiTrajectory
        Validated multi-trajectory container.

    Raises
    ------
    TypeError
        If arguments are not snapshot sequences.
    ValueError
        If no trajectories are provided or families are mixed.
    """
    if len(trajectories) == 1 and not isinstance(
        trajectories[0],
        (GraphSnapshotSequence, HeteroGraphSnapshotSequence),
    ):
        sequence_list = list(trajectories[0])
    else:
        sequence_list = list(trajectories)  # type: ignore[arg-type]
    return MultiTrajectory(tuple(sequence_list))


def resolve_sequence(
    sequence: GraphSnapshotSequence | Sequence[Data],
) -> GraphSnapshotSequence:
    """Normalize input into a validated homogeneous snapshot sequence.

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


def resolve_hetero_sequence(
    sequence: HeteroGraphSnapshotSequence | Sequence[HeteroData],
) -> HeteroGraphSnapshotSequence:
    """Normalize input into a validated heterogeneous snapshot sequence.

    Wraps a plain sequence of ``HeteroData`` snapshots in
    :class:`HeteroGraphSnapshotSequence`; existing sequences are returned
    unchanged.

    Parameters
    ----------
    sequence : HeteroGraphSnapshotSequence or sequence of HeteroData
        Raw heterogeneous snapshot input.

    Returns
    -------
    HeteroGraphSnapshotSequence
        Validated hetero sequence container.
    """
    if isinstance(sequence, HeteroGraphSnapshotSequence):
        return sequence
    return HeteroGraphSnapshotSequence(sequence)


def resolve_pair_delta_t(
    sequence: SnapshotSequence,
    timestep: int,
    *,
    default_time_step: float,
) -> float:
    """Return the integration interval for transition ``timestep -> timestep + 1``.

    Parameters
    ----------
    sequence : GraphSnapshotSequence or HeteroGraphSnapshotSequence
        Snapshot sequence that may carry timestamps.
    timestep : int
        Source snapshot index.
    default_time_step : float
        Fallback interval when timestamps are absent.

    Returns
    -------
    float
        Positive integration interval.

    Raises
    ------
    ValueError
        If a hetero sequence carries timestamps (``delta_t_at`` is not
        implemented for multiplex containers yet).
    """
    if sequence.has_timestamps:
        delta_t_at = getattr(sequence, "delta_t_at", None)
        if not callable(delta_t_at):
            msg = (
                "timestamped HeteroGraphSnapshotSequence is unsupported for "
                "pair delta_t resolution; omit timestamps or use homogeneous "
                "GraphSnapshotSequence"
            )
            raise ValueError(msg)
        return float(delta_t_at(timestep).item())
    return default_time_step
