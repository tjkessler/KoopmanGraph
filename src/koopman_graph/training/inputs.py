"""Multi-trajectory training / validation input resolution."""

from __future__ import annotations

from collections.abc import Sequence

from torch_geometric.data import Data, HeteroData

from koopman_graph.data import (
    GraphSnapshotSequence,
    HeteroGraphSnapshotSequence,
    MultiTrajectory,
    resolve_hetero_sequence,
    resolve_sequence,
)
from koopman_graph.data.trajectories import SnapshotSequence
from koopman_graph.training.history import TrainingInput, ValidationInput


def _classify_trajectory_items(
    items: Sequence[object],
    *,
    empty_message: str,
) -> list[SnapshotSequence]:
    """Normalize a bare list/tuple into a single-trajectory snapshot sequence.

    Multi-trajectory input must use :class:`~koopman_graph.data.MultiTrajectory`.
    A bare list of snapshot sequences is rejected. A bare list of ``Data`` or
    ``HeteroData`` snapshots is treated as one trajectory.

    Parameters
    ----------
    items : sequence
        Elements from a bare list/tuple passed to ``fit``.
    empty_message : str
        Error message when ``items`` is empty.

    Returns
    -------
    list of GraphSnapshotSequence or HeteroGraphSnapshotSequence
        A single-element list wrapping the snapshot family.

    Raises
    ------
    ValueError
        If ``items`` is empty, mixes sequence and graph elements, mixes
        homogeneous and hetero families, or mixes ``Data`` with ``HeteroData``.
    TypeError
        If any element is unsupported, or if the input is a bare list of
        snapshot sequences.
    """
    if not items:
        raise ValueError(empty_message)

    homo_seq_indices = [
        index
        for index, item in enumerate(items)
        if isinstance(item, GraphSnapshotSequence)
    ]
    hetero_seq_indices = [
        index
        for index, item in enumerate(items)
        if isinstance(item, HeteroGraphSnapshotSequence)
    ]
    data_indices = [index for index, item in enumerate(items) if isinstance(item, Data)]
    hetero_data_indices = [
        index for index, item in enumerate(items) if isinstance(item, HeteroData)
    ]

    n_items = len(items)
    if len(homo_seq_indices) == n_items or len(hetero_seq_indices) == n_items:
        msg = (
            "a bare list of snapshot sequences is not accepted; "
            "wrap multi-trajectory input in MultiTrajectory(...) "
            "(or as_multi_trajectory(...))"
        )
        raise TypeError(msg)
    if len(data_indices) == n_items:
        return [resolve_sequence(items)]  # type: ignore[arg-type]
    if len(hetero_data_indices) == n_items:
        return [resolve_hetero_sequence(items)]  # type: ignore[arg-type]

    if homo_seq_indices and hetero_seq_indices:
        msg = (
            "cannot mix GraphSnapshotSequence and HeteroGraphSnapshotSequence "
            "in the same fit input; use separate MultiTrajectory containers"
        )
        raise ValueError(msg)
    if data_indices and hetero_data_indices:
        msg = (
            "cannot mix Data and HeteroData in the same fit input; "
            "pass a list of Data for one homogeneous trajectory or a list of "
            "HeteroData for one hetero trajectory"
        )
        raise ValueError(msg)
    if (homo_seq_indices or hetero_seq_indices) and (
        data_indices or hetero_data_indices
    ):
        msg = (
            "cannot mix snapshot sequences and raw Data/HeteroData in the same "
            "fit input; use MultiTrajectory([...]) for multiple trajectories, "
            "or a list of Data / HeteroData for one trajectory"
        )
        raise ValueError(msg)

    bad_index = next(
        index
        for index, item in enumerate(items)
        if not isinstance(
            item,
            (
                GraphSnapshotSequence,
                HeteroGraphSnapshotSequence,
                Data,
                HeteroData,
            ),
        )
    )
    msg = (
        "fit trajectory elements must be GraphSnapshotSequence, "
        "HeteroGraphSnapshotSequence, Data, or HeteroData; "
        f"index {bad_index} has type {type(items[bad_index]).__name__}"
    )
    raise TypeError(msg)


def resolve_training_sequences(
    data_sequence: TrainingInput,
) -> list[SnapshotSequence]:
    """Normalize training input into one or more snapshot sequences.

    Multi-trajectory input must be a
    :class:`~koopman_graph.data.MultiTrajectory`. A bare list of ``Data`` or
    ``HeteroData`` snapshots is always a single trajectory.

    Parameters
    ----------
    data_sequence : TrainingInput
        Single sequence, list of ``Data`` / ``HeteroData`` snapshots, or
        ``MultiTrajectory``.

    Returns
    -------
    list of GraphSnapshotSequence or HeteroGraphSnapshotSequence
        One or more validated training trajectories.

    Raises
    ------
    ValueError
        If multi-trajectory input is empty or mixes sequence and graph
        elements / families.
    TypeError
        If a bare sequence contains unsupported element types, including a
        bare list of snapshot sequences.
    """
    if isinstance(data_sequence, MultiTrajectory):
        return list(data_sequence.sequences)
    if isinstance(data_sequence, (GraphSnapshotSequence, HeteroGraphSnapshotSequence)):
        return [data_sequence]
    if not isinstance(data_sequence, Sequence) or isinstance(
        data_sequence, (Data, HeteroData, str, bytes)
    ):
        if isinstance(data_sequence, HeteroData):
            msg = (
                "a single HeteroData is not a trajectory; pass a "
                "HeteroGraphSnapshotSequence or a non-empty list of HeteroData"
            )
            raise TypeError(msg)
        return [resolve_sequence(data_sequence)]  # type: ignore[arg-type]
    return _classify_trajectory_items(
        list(data_sequence),
        empty_message=(
            "data_sequence must be non-empty; pass a GraphSnapshotSequence, "
            "HeteroGraphSnapshotSequence, a non-empty list of Data or "
            "HeteroData, or MultiTrajectory(...)"
        ),
    )


def resolve_validation_sequences(
    validation_sequence: ValidationInput,
    *,
    num_training_sequences: int,
) -> list[SnapshotSequence] | None:
    """Normalize validation input for :meth:`fit`.

    A single validation sequence (or list of ``Data`` / ``HeteroData``) is
    reused for all training trajectories. A
    :class:`~koopman_graph.data.MultiTrajectory` must match the training
    trajectory count.

    Parameters
    ----------
    validation_sequence : ValidationInput
        Optional validation data.
    num_training_sequences : int
        Number of training trajectories supplied to :meth:`fit`.

    Returns
    -------
    list of GraphSnapshotSequence or HeteroGraphSnapshotSequence or None
        Validation trajectories aligned with training input.

    Raises
    ------
    ValueError
        If a multi-trajectory validation length does not match
        ``num_training_sequences``, or if input is empty or mixed.
    TypeError
        If a bare sequence contains unsupported element types, including a
        bare list of snapshot sequences.
    """
    if validation_sequence is None:
        return None
    if isinstance(validation_sequence, MultiTrajectory):
        sequences = list(validation_sequence.sequences)
        if len(sequences) != num_training_sequences:
            msg = (
                "validation_sequence list length must match the number of "
                f"training trajectories ({num_training_sequences}), "
                f"got {len(sequences)}"
            )
            raise ValueError(msg)
        return sequences
    if isinstance(
        validation_sequence, (GraphSnapshotSequence, HeteroGraphSnapshotSequence)
    ):
        return [validation_sequence]
    if not isinstance(validation_sequence, Sequence) or isinstance(
        validation_sequence, (Data, HeteroData, str, bytes)
    ):
        if isinstance(validation_sequence, HeteroData):
            msg = (
                "a single HeteroData is not a trajectory; pass a "
                "HeteroGraphSnapshotSequence or a non-empty list of HeteroData"
            )
            raise TypeError(msg)
        return [resolve_sequence(validation_sequence)]  # type: ignore[arg-type]

    return _classify_trajectory_items(
        list(validation_sequence),
        empty_message=(
            "validation_sequence must be non-empty when provided as a list; "
            "pass None, a GraphSnapshotSequence, HeteroGraphSnapshotSequence, "
            "a non-empty list of Data or HeteroData, or MultiTrajectory(...)"
        ),
    )
