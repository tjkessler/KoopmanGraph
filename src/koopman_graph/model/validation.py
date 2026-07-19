"""Control and sequence validation helpers for GraphKoopmanModel.

Shallow peer of :mod:`koopman_graph.model.estimator`. Prefer calling these through
the model façade; peer imports are for power-user / package-internal use.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from torch import Tensor
from torch_geometric.data import Data

from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.training import (
    EarlyStoppingMonitor,
    TrainingInput,
    ValidationInput,
    resolve_early_stopping_monitor,
    resolve_training_sequences,
    resolve_validation_sequences,
)


@dataclass(frozen=True, slots=True)
class PreparedFitInputs:
    """Validated training / validation sequences and early-stopping monitor.

    Attributes
    ----------
    train_sequences : list
        Training trajectory sequences.
    val_sequences : list | None
        Optional validation sequences.
    early_stopping_monitor : Any | None
        Optional early-stopping callback state.
    """

    train_sequences: list[GraphSnapshotSequence]
    val_sequences: list[GraphSnapshotSequence] | None
    early_stopping_monitor: Literal["train", "val"]


def as_data(
    x_or_data: Tensor | Data,
    edge_index: Tensor,
    edge_weight: Tensor | None,
) -> Data:
    """Build a PyG ``Data`` object from tensor or ``Data`` inputs.

    Parameters
    ----------
    x_or_data : Tensor or Data
        Node features or an existing snapshot.
    edge_index : Tensor
        Edge index with shape ``(2, num_edges)``.
    edge_weight : Tensor or None
        Optional edge weights with shape ``(num_edges,)``.

    Returns
    -------
    Data
        Snapshot suitable for physics lifting callables.
    """
    if isinstance(x_or_data, Data):
        return x_or_data
    data = Data(x=x_or_data, edge_index=edge_index)
    if edge_weight is not None:
        data.edge_weight = edge_weight
    return data


def validate_controls(
    *,
    control_dim: int,
    controls: Sequence[Tensor] | None,
    steps: int,
) -> None:
    """Validate rollout controls against model control settings.

    Parameters
    ----------
    control_dim : int
        Model control dimension.
    controls : sequence of Tensor or None
        Control inputs for each rollout step.
    steps : int
        Number of rollout steps.

    Raises
    ------
    ValueError
        If controls are missing, surplus, or provided to an uncontrolled model.
    """
    if control_dim == 0:
        if controls is not None:
            msg = "controls provided to an uncontrolled model"
            raise ValueError(msg)
        return
    if controls is None:
        msg = "controls are required when control_dim > 0"
        raise ValueError(msg)
    if len(controls) != steps:
        msg = f"expected {steps} control inputs for rollout, got {len(controls)}"
        raise ValueError(msg)


def validate_sequence_controls(
    *,
    control_dim: int,
    sequence: GraphSnapshotSequence,
) -> None:
    """Validate sequence controls against a model control dimension.

    Parameters
    ----------
    control_dim : int
        Model control dimension.
    sequence : GraphSnapshotSequence
        Training or validation sequence.

    Raises
    ------
    ValueError
        If controls are missing or dimensions disagree.
    """
    if control_dim == 0:
        if sequence.has_controls:
            msg = "sequence contains control inputs but model control_dim is 0"
            raise ValueError(msg)
        return
    if not sequence.has_controls:
        msg = "controlled model requires sequences with control inputs"
        raise ValueError(msg)
    if sequence.control_dim != control_dim:
        msg = (
            f"sequence control_dim ({sequence.control_dim}) must match "
            f"model control_dim ({control_dim})"
        )
        raise ValueError(msg)


def prepare_fit_inputs(
    *,
    control_dim: int,
    data_sequence: TrainingInput,
    validation_sequence: ValidationInput = None,
    epochs: int = 100,
    early_stopping_patience: int | None = None,
    early_stopping_monitor: EarlyStoppingMonitor = "auto",
) -> PreparedFitInputs:
    """Validate ``fit`` inputs and resolve training / validation sequences.

    Parameters
    ----------
    control_dim : int
        Model control dimension used for sequence control checks.
    data_sequence : TrainingInput
        Training trajectory input accepted by :meth:`GraphKoopmanModel.fit`.
    validation_sequence : ValidationInput, optional
        Optional held-out validation input.
    epochs : int, optional
        Requested training epoch count. Must be ``>= 1``.
    early_stopping_patience : int or None, optional
        When set, must be ``>= 1``.
    early_stopping_monitor : {"auto", "train", "val"}, optional
        Early-stopping monitor mode. Default is ``"auto"``.

    Returns
    -------
    PreparedFitInputs
        Resolved sequences and concrete ``"train"`` / ``"val"`` monitor.

    Raises
    ------
    ValueError
        If epoch / patience bounds fail, controls disagree, or fewer than two
        snapshots are provided for training or validation.
    """
    if epochs < 1:
        msg = f"epochs must be >= 1, got {epochs}"
        raise ValueError(msg)
    if early_stopping_patience is not None and early_stopping_patience < 1:
        msg = (
            f"early_stopping_patience must be >= 1 when set, "
            f"got {early_stopping_patience}"
        )
        raise ValueError(msg)

    train_sequences = resolve_training_sequences(data_sequence)
    for sequence in train_sequences:
        validate_sequence_controls(
            control_dim=control_dim,
            sequence=sequence,
        )
        if sequence.num_timesteps < 2:
            msg = "data_sequence must contain at least 2 snapshots for training"
            raise ValueError(msg)

    val_sequences = resolve_validation_sequences(
        validation_sequence,
        num_training_sequences=len(train_sequences),
    )
    if val_sequences is not None:
        for sequence in val_sequences:
            validate_sequence_controls(
                control_dim=control_dim,
                sequence=sequence,
            )
            if sequence.num_timesteps < 2:
                msg = (
                    "validation_sequence must contain at least 2 snapshots "
                    "for validation"
                )
                raise ValueError(msg)

    monitor = resolve_early_stopping_monitor(
        early_stopping_monitor,
        has_validation=val_sequences is not None,
    )
    return PreparedFitInputs(
        train_sequences=train_sequences,
        val_sequences=val_sequences,
        early_stopping_monitor=monitor,
    )
