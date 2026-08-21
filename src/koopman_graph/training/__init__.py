"""Training utilities for trainable Koopman graph models.

Functions annotate against
:class:`~koopman_graph.protocols.TrainableKoopmanModel` (satisfied by
:class:`~koopman_graph.model.GraphKoopmanModel`).

Capability layout
-----------------
``history``
    ``LossWeights``, ``FitHistory``, and related type aliases (internal
    ``TrainingLossBreakdown`` lives here but is not package-exported).
``schedules``
    Static and ramped loss-weight schedules.
``device``
    Training device resolution and sequence placement.
``pair_objectives``
    One-step / pair reconstruction and forward/backward consistency
    composition over :mod:`koopman_graph.losses`.
``timestep_encode``
    Private ``encode_at_timestep`` helper shared by pair losses and the
    latent cache (not exported from this package ``__all__``).
``latent_cache``
    Private sequence latent cache helpers for shared teacher-forced
    encodings (not exported from this package ``__all__``).
``extra_objectives``
    Lie / PDE / sparsity / worst-case composition helpers.
``objectives``
    Thin ``compute_training_loss`` orchestrator plus eigenvalue / rollout
    composition (re-exports pair helpers for stable deep imports).
``epochs``
    ``train_one_epoch``, ``train_windowed_epoch``, ``eval_one_epoch``.
``batched_objectives``
    Opt-in ``compute_batched_training_loss`` for collated multi-graph
    reconstruction / forward (package import; not on root ``__all__``).
``inputs``
    Multi-trajectory classification and
    ``resolve_training_sequences`` / ``resolve_validation_sequences``.
``loop``
    ``run_fit_loop`` plus early-stopping / LR-scheduler helpers.
``callbacks``
    Observe-only :class:`FitCallback` protocol, :class:`NoOpFitCallback`,
    and :class:`ResDMDFitCallback` (default ``mode="observe"``;
    ``mode="gate"`` may raise at fit end without mutating parameters).
"""

from koopman_graph.training.batched_objectives import (
    compute_batched_training_loss,
    validate_graph_batching_request,
)
from koopman_graph.training.callbacks import (
    FitCallback,
    NoOpFitCallback,
    ResDMDFitCallback,
)
from koopman_graph.training.device import resolve_device, sequence_to_device
from koopman_graph.training.epochs import (
    eval_one_epoch,
    train_one_epoch,
    train_windowed_epoch,
)
from koopman_graph.training.history import (
    EarlyStoppingMonitor,
    ExtraLosses,
    FitHistory,
    LossWeights,
    LossWeightSchedule,
    LRSchedulerFactory,
    TrainingInput,
    ValidationInput,
    mean_training_loss_breakdown,
)
from koopman_graph.training.inputs import (
    resolve_training_sequences,
    resolve_validation_sequences,
)
from koopman_graph.training.loop import (
    resolve_early_stopping_monitor,
    resolve_lr_scheduler,
    run_fit_loop,
    should_stop_early,
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
    "ExtraLosses",
    "FitCallback",
    "FitHistory",
    "LRSchedulerFactory",
    "LossWeightSchedule",
    "LossWeights",
    "NoOpFitCallback",
    "ResDMDFitCallback",
    "TrainingInput",
    "ValidationInput",
    "compute_backward_consistency_sequence_loss",
    "compute_batched_training_loss",
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
    "validate_graph_batching_request",
]
