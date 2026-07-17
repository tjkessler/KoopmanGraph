"""Training utilities for trainable Koopman graph models.

Functions annotate against
:class:`~koopman_graph.protocols.TrainableKoopmanModel` (satisfied by
:class:`~koopman_graph.model.GraphKoopmanModel`).

Capability layout
-----------------
``history``
    ``LossWeights``, ``FitHistory``, ``TrainingLossBreakdown``, and related
    type aliases.
``schedules``
    Static and ramped loss-weight schedules.
``device``
    Training device resolution and sequence placement.
``objectives``
    Training-side composition of loss primitives from
    :mod:`koopman_graph.losses` (which remains a top-level module).
``loop``
    Epoch helpers, input resolution, and :func:`run_fit_loop`.
"""

from koopman_graph.training.device import resolve_device, sequence_to_device
from koopman_graph.training.history import (
    EarlyStoppingMonitor,
    FitHistory,
    LossWeights,
    LossWeightSchedule,
    LRSchedulerFactory,
    TrainingInput,
    TrainingLossBreakdown,
    ValidationInput,
    mean_training_loss_breakdown,
)
from koopman_graph.training.loop import (
    eval_one_epoch,
    resolve_early_stopping_monitor,
    resolve_lr_scheduler,
    resolve_training_sequences,
    resolve_validation_sequences,
    run_fit_loop,
    should_stop_early,
    train_one_epoch,
    train_windowed_epoch,
)
from koopman_graph.training.objectives import (
    compute_backward_consistency_sequence_loss,
    compute_eigenvalue_regularization_loss,
    compute_forward_consistency_sequence_loss,
    compute_rollout_loss,
    compute_sequence_loss,
    compute_training_loss,
    one_step_loss,
)
from koopman_graph.training.schedules import (
    constant_loss_weights,
    linear_ramp_loss_weights,
    resolve_loss_weights_for_epoch,
)

__all__ = [
    "EarlyStoppingMonitor",
    "FitHistory",
    "LRSchedulerFactory",
    "LossWeightSchedule",
    "LossWeights",
    "TrainingInput",
    "TrainingLossBreakdown",
    "ValidationInput",
    "compute_backward_consistency_sequence_loss",
    "compute_eigenvalue_regularization_loss",
    "compute_forward_consistency_sequence_loss",
    "compute_rollout_loss",
    "compute_sequence_loss",
    "compute_training_loss",
    "constant_loss_weights",
    "eval_one_epoch",
    "linear_ramp_loss_weights",
    "mean_training_loss_breakdown",
    "one_step_loss",
    "resolve_device",
    "resolve_early_stopping_monitor",
    "resolve_loss_weights_for_epoch",
    "resolve_lr_scheduler",
    "resolve_training_sequences",
    "resolve_validation_sequences",
    "run_fit_loop",
    "sequence_to_device",
    "should_stop_early",
    "train_one_epoch",
    "train_windowed_epoch",
]
