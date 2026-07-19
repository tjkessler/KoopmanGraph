"""Multi-trajectory training / validation input resolution."""

from __future__ import annotations

from collections.abc import Sequence

from torch_geometric.data import Data

from koopman_graph.data import (
    GraphSnapshotSequence,
    MultiTrajectory,
    resolve_sequence,
)
from koopman_graph.training.history import TrainingInput, ValidationInput


def _classify_trajectory_items(
    items: Sequence[object],
    *,
    empty_message: str,
) -> list[GraphSnapshotSequence]:
    """Normalize a bare list/tuple into a single-trajectory snapshot sequence.

    Multi-trajectory input must use :class:`~koopman_graph.data.MultiTrajectory`.
    A bare list of :class:`~koopman_graph.data.GraphSnapshotSequence` is rejected.

    Parameters
    ----------
    items : sequence
        Elements from a bare list/tuple passed to ``fit``.
    empty_message : str
        Error message when ``items`` is empty.

    Returns
    -------
    list of GraphSnapshotSequence
        A single-element list wrapping the ``Data`` snapshots.

    Raises
    ------
    ValueError
        If ``items`` is empty, mixes ``GraphSnapshotSequence`` with ``Data``,
        or is a bare list of :class:`~koopman_graph.data.GraphSnapshotSequence`.
    TypeError
        If any element is neither a snapshot sequence nor a ``Data`` graph.
    """
    if not items:
        raise ValueError(empty_message)

    sequence_indices = [
        index
        for index, item in enumerate(items)
        if isinstance(item, GraphSnapshotSequence)
    ]
    data_indices = [index for index, item in enumerate(items) if isinstance(item, Data)]
    if len(sequence_indices) == len(items):
        msg = (
            "a bare list of GraphSnapshotSequence is not accepted; "
            "wrap multi-trajectory input in MultiTrajectory(...) "
            "(or as_multi_trajectory(...))"
        )
        raise TypeError(msg)
    if len(data_indices) == len(items):
        return [resolve_sequence(items)]  # type: ignore[arg-type]
    if sequence_indices and data_indices:
        msg = (
            "cannot mix GraphSnapshotSequence and Data in the same fit input; "
            "use MultiTrajectory([...]) for multiple trajectories, or a list "
            "of Data for one trajectory"
        )
        raise ValueError(msg)
    bad_index = next(
        index
        for index, item in enumerate(items)
        if not isinstance(item, (GraphSnapshotSequence, Data))
    )
    msg = (
        "fit trajectory elements must be GraphSnapshotSequence or Data; "
        f"index {bad_index} has type {type(items[bad_index]).__name__}"
    )
    raise TypeError(msg)


def resolve_training_sequences(
    data_sequence: TrainingInput,
) -> list[GraphSnapshotSequence]:
    """Normalize training input into one or more snapshot sequences.

    Multi-trajectory input must be a
    :class:`~koopman_graph.data.MultiTrajectory`. A bare list of ``Data``
    snapshots is always a single trajectory.

    Parameters
    ----------
    data_sequence : TrainingInput
        Single sequence, list of ``Data`` snapshots, or ``MultiTrajectory``.

    Returns
    -------
    list of GraphSnapshotSequence
        One or more validated training trajectories.

    Raises
    ------
    ValueError
        If multi-trajectory input is empty or mixes sequence and ``Data``
        elements.
    TypeError
        If a bare sequence contains unsupported element types, including a
        bare list of :class:`~koopman_graph.data.GraphSnapshotSequence`.
    """
    if isinstance(data_sequence, MultiTrajectory):
        return list(data_sequence.sequences)
    if isinstance(data_sequence, GraphSnapshotSequence):
        return [data_sequence]
    if not isinstance(data_sequence, Sequence) or isinstance(
        data_sequence, (Data, str, bytes)
    ):
        return [resolve_sequence(data_sequence)]  # type: ignore[arg-type]
    return _classify_trajectory_items(
        list(data_sequence),
        empty_message=(
            "data_sequence must be non-empty; pass a GraphSnapshotSequence, "
            "a non-empty list of Data, or MultiTrajectory(...)"
        ),
    )


def resolve_validation_sequences(
    validation_sequence: ValidationInput,
    *,
    num_training_sequences: int,
) -> list[GraphSnapshotSequence] | None:
    """Normalize validation input for :meth:`fit`.

    A single validation sequence (or list of ``Data``) is reused for all
    training trajectories. A :class:`~koopman_graph.data.MultiTrajectory` must
    match the training trajectory count.

    Parameters
    ----------
    validation_sequence : ValidationInput
        Optional validation data.
    num_training_sequences : int
        Number of training trajectories supplied to :meth:`fit`.

    Returns
    -------
    list of GraphSnapshotSequence or None
        Validation trajectories aligned with training input.

    Raises
    ------
    ValueError
        If a multi-trajectory validation length does not match
        ``num_training_sequences``, or if input is empty or mixed.
    TypeError
        If a bare sequence contains unsupported element types, including a
        bare list of :class:`~koopman_graph.data.GraphSnapshotSequence`.
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
    if isinstance(validation_sequence, GraphSnapshotSequence):
        return [validation_sequence]
    if not isinstance(validation_sequence, Sequence) or isinstance(
        validation_sequence, (Data, str, bytes)
    ):
        return [resolve_sequence(validation_sequence)]  # type: ignore[arg-type]

    return _classify_trajectory_items(
        list(validation_sequence),
        empty_message=(
            "validation_sequence must be non-empty when provided as a list; "
            "pass None, a GraphSnapshotSequence, a non-empty list of Data, "
            "or MultiTrajectory(...)"
        ),
    )
