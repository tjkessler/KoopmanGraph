"""Contiguous temporal train / validation / test splits."""

from __future__ import annotations

from dataclasses import dataclass

from koopman_graph.data.containers import GraphSnapshotSequence


@dataclass(frozen=True)
class TemporalSplit:
    """Train, validation, and test snapshot sequences from a temporal split.

    Attributes
    ----------
    train : GraphSnapshotSequence
        Earliest contiguous snapshots used for training.
    val : GraphSnapshotSequence
        Middle contiguous snapshots used for validation.
    test : GraphSnapshotSequence
        Latest contiguous snapshots held out for evaluation.
    """

    train: GraphSnapshotSequence
    val: GraphSnapshotSequence
    test: GraphSnapshotSequence


def temporal_split(
    sequence: GraphSnapshotSequence,
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    min_train_timesteps: int = 2,
    min_val_timesteps: int = 2,
    min_test_timesteps: int = 1,
) -> TemporalSplit:
    """Split a snapshot sequence into contiguous train, validation, and test sets.

    Earlier snapshots are assigned to training, later snapshots to validation and
    test. Ratios must sum to ``1.0``.

    Parameters
    ----------
    sequence : GraphSnapshotSequence
        Full time-ordered snapshot sequence to split.
    train_ratio : float, optional
        Fraction of timesteps assigned to training. Default is ``0.7``.
    val_ratio : float, optional
        Fraction assigned to validation. Default is ``0.1``.
    test_ratio : float, optional
        Fraction assigned to test. Default is ``0.2``.
    min_train_timesteps : int, optional
        Minimum training snapshots required. Default is ``2``.
    min_val_timesteps : int, optional
        Minimum validation snapshots required. Default is ``2``.
    min_test_timesteps : int, optional
        Minimum test snapshots required. Default is ``1``.

    Returns
    -------
    TemporalSplit
        Contiguous train, validation, and test sequences sharing topology.

    Raises
    ------
    ValueError
        If ratios do not sum to ``1.0``, any minimum is violated, or the
        sequence is too short for the requested split.
    """
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        msg = f"train_ratio + val_ratio + test_ratio must equal 1.0, got {ratio_sum}"
        raise ValueError(msg)
    if min_train_timesteps < 2:
        msg = f"min_train_timesteps must be >= 2, got {min_train_timesteps}"
        raise ValueError(msg)
    if min_val_timesteps < 1 or min_test_timesteps < 1:
        msg = "min_val_timesteps and min_test_timesteps must be >= 1"
        raise ValueError(msg)

    num_timesteps = sequence.num_timesteps
    min_required = min_train_timesteps + min_val_timesteps + min_test_timesteps
    if num_timesteps < min_required:
        msg = (
            f"sequence has {num_timesteps} timesteps but needs at least "
            f"{min_required} for the requested split"
        )
        raise ValueError(msg)

    train_end = int(num_timesteps * train_ratio)
    val_end = train_end + int(num_timesteps * val_ratio)
    train_end = max(train_end, min_train_timesteps)
    val_end = max(val_end, train_end + min_val_timesteps)
    if num_timesteps - val_end < min_test_timesteps:
        val_end = num_timesteps - min_test_timesteps

    train_snapshots = sequence.snapshots[:train_end]
    val_snapshots = sequence.snapshots[train_end:val_end]
    test_snapshots = sequence.snapshots[val_end:]

    if len(train_snapshots) < min_train_timesteps:  # pragma: no cover - defensive
        msg = (
            f"train split has {len(train_snapshots)} timesteps, "
            f"expected at least {min_train_timesteps}"
        )
        raise ValueError(msg)
    if len(val_snapshots) < min_val_timesteps:
        msg = (
            f"validation split has {len(val_snapshots)} timesteps, "
            f"expected at least {min_val_timesteps}"
        )
        raise ValueError(msg)
    if len(test_snapshots) < min_test_timesteps:  # pragma: no cover - defensive
        msg = (
            f"test split has {len(test_snapshots)} timesteps, "
            f"expected at least {min_test_timesteps}"
        )
        raise ValueError(msg)

    return TemporalSplit(
        train=GraphSnapshotSequence(
            train_snapshots,
            allow_dynamic_topology=sequence.allow_dynamic_topology,
            control_inputs=(
                None
                if sequence.control_inputs is None
                else sequence.control_inputs[:train_end]
            ),
            timestamps=(
                None if sequence.timestamps is None else sequence.timestamps[:train_end]
            ),
            observation_masks=(
                None
                if sequence.observation_masks is None
                else sequence.observation_masks[:train_end]
            ),
        ),
        val=GraphSnapshotSequence(
            val_snapshots,
            allow_dynamic_topology=sequence.allow_dynamic_topology,
            control_inputs=(
                None
                if sequence.control_inputs is None
                else sequence.control_inputs[train_end:val_end]
            ),
            timestamps=(
                None
                if sequence.timestamps is None
                else sequence.timestamps[train_end:val_end]
            ),
            observation_masks=(
                None
                if sequence.observation_masks is None
                else sequence.observation_masks[train_end:val_end]
            ),
        ),
        test=GraphSnapshotSequence(
            test_snapshots,
            allow_dynamic_topology=sequence.allow_dynamic_topology,
            control_inputs=(
                None
                if sequence.control_inputs is None
                else sequence.control_inputs[val_end:]
            ),
            timestamps=(
                None if sequence.timestamps is None else sequence.timestamps[val_end:]
            ),
            observation_masks=(
                None
                if sequence.observation_masks is None
                else sequence.observation_masks[val_end:]
            ),
        ),
    )
