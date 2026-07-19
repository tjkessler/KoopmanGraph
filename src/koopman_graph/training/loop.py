"""Multi-epoch fit-loop orchestration and early-stop / scheduler helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from koopman_graph.data import (
    GraphSnapshotSequence,
    RolloutStartIndices,
    WindowSampler,
    resolve_rollout_start_indices,
)
from koopman_graph.protocols import TrainableKoopmanModel
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
)
from koopman_graph.training.schedules import resolve_loss_weights_for_epoch

__all__ = [
    "eval_one_epoch",
    "resolve_early_stopping_monitor",
    "resolve_lr_scheduler",
    "run_fit_loop",
    "should_stop_early",
    "train_one_epoch",
    "train_windowed_epoch",
]


def resolve_early_stopping_monitor(
    monitor: EarlyStoppingMonitor,
    *,
    has_validation: bool,
) -> Literal["train", "val"]:
    """Resolve which loss early stopping should monitor.

    Parameters
    ----------
    monitor : {"auto", "train", "val"}
        Requested monitor mode.
    has_validation : bool
        Whether a validation sequence was provided to :meth:`fit`.

    Returns
    -------
    {"train", "val"}
        Resolved monitor target.

    Raises
    ------
    ValueError
        If ``monitor="val"`` but no validation sequence was provided.
    """
    if monitor == "auto":
        return "val" if has_validation else "train"
    if monitor == "val" and not has_validation:
        msg = "early_stopping_monitor='val' requires validation_sequence"
        raise ValueError(msg)
    return monitor


def resolve_lr_scheduler(
    lr_scheduler: LRScheduler | LRSchedulerFactory | None,
    optimizer: Optimizer,
) -> LRScheduler | None:
    """Instantiate an optional learning-rate scheduler.

    Parameters
    ----------
    lr_scheduler : LRScheduler, callable, or None
        Scheduler instance or factory ``optimizer -> scheduler``.
    optimizer : Optimizer
        Optimizer passed to a scheduler factory.

    Returns
    -------
    LRScheduler or None
        Resolved scheduler, if any.
    """
    if lr_scheduler is None:
        return None
    if isinstance(lr_scheduler, LRScheduler):
        return lr_scheduler
    return lr_scheduler(optimizer)


def should_stop_early(
    *,
    epoch_loss: float,
    best_loss: float,
    epochs_without_improvement: int,
    patience: int,
    min_delta: float,
) -> tuple[bool, float, int]:
    """Update early-stopping state after an epoch.

    Parameters
    ----------
    epoch_loss : float
        Training loss for the completed epoch.
    best_loss : float
        Best loss seen so far.
    epochs_without_improvement : int
        Consecutive epochs without sufficient improvement.
    patience : int
        Stop after this many non-improving epochs.
    min_delta : float
        Minimum decrease in loss to count as improvement.

    Returns
    -------
    tuple of (bool, float, int)
        Whether to stop, updated best loss, and updated non-improvement count.
    """
    if epoch_loss < best_loss - min_delta:
        return False, epoch_loss, 0
    updated_count = epochs_without_improvement + 1
    return updated_count >= patience, best_loss, updated_count


def run_fit_loop(
    model: TrainableKoopmanModel,
    train_sequences: Sequence[GraphSnapshotSequence],
    *,
    epochs: int = 100,
    lr: float = 1e-3,
    optimizer: Callable[..., Optimizer] = torch.optim.Adam,
    device: str | torch.device | None = None,
    loss_weights: LossWeights | None = None,
    loss_weight_schedule: LossWeightSchedule | None = None,
    extra_losses: ExtraLosses | None = None,
    rollout_horizon: int | None = None,
    rollout_start_indices: RolloutStartIndices = None,
    rollout_starts_per_epoch: int | None = None,
    rollout_start_seed: int | None = None,
    lr_scheduler: LRScheduler | LRSchedulerFactory | None = None,
    window_length: int | None = None,
    batch_size: int = 8,
    windows_per_epoch: int | None = None,
    window_seed: int | None = None,
    max_grad_norm: float | None = None,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
    early_stopping_monitor: Literal["train", "val"] = "train",
    val_sequences: Sequence[GraphSnapshotSequence] | None = None,
    restore_best_weights: bool = False,
    checkpoint_path: str | Path | None = None,
    **optimizer_kwargs: Any,
) -> FitHistory:
    """Run the multi-epoch training loop for a trainable Koopman model.

    Canonical orchestration for :meth:`~koopman_graph.model.GraphKoopmanModel.fit`.
    Callers are responsible for resolving and validating sequences (controls,
    length, multi-trajectory layout) and for resolving
    ``early_stopping_monitor`` via :func:`resolve_early_stopping_monitor`
    before calling this helper (``"auto"`` is not accepted here).

    Parameters
    ----------
    model : TrainableKoopmanModel
        Model satisfying :class:`~koopman_graph.protocols.TrainableKoopmanModel`
        (also an ``nn.Module`` for device / state-dict operations).
    train_sequences : sequence of GraphSnapshotSequence
        Already-validated training trajectories.
    epochs : int, optional
        Number of training epochs. Default is ``100``.
    lr : float, optional
        Learning rate passed to the optimizer. Default is ``1e-3``.
    optimizer : callable, optional
        Optimizer class. Default is :class:`torch.optim.Adam`.
    device : str, torch.device, or None, optional
        Training device. Defaults to the model's parameter device, or CPU.
    loss_weights : LossWeights or None, optional
        Static loss weights for all epochs.
    loss_weight_schedule : callable or None, optional
        Per-epoch weight schedule; overrides ``loss_weights`` when set.
    extra_losses : ExtraLosses or None, optional
        Fit-time callables for enabled Lie and PDE residual terms.
    rollout_horizon : int or None, optional
        Autoregressive rollout steps when rollout weight is non-zero.
    rollout_start_indices : sequence of int, ``"all"``, or None, optional
        Rollout-loss origin indices.
    rollout_starts_per_epoch : int or None, optional
        Random sample size for rollout origins each epoch.
    rollout_start_seed : int or None, optional
        Base seed for rollout-origin sampling.
    lr_scheduler : LRScheduler or callable, optional
        Scheduler instance or ``optimizer -> scheduler`` factory.
    window_length : int or None, optional
        Fixed window length for mini-batch training; ``None`` uses full
        sequences.
    batch_size : int, optional
        Windows per optimizer step when windowed. Default is ``8``.
    windows_per_epoch : int or None, optional
        Cap on sampled windows per epoch.
    window_seed : int or None, optional
        Base seed for window shuffling.
    max_grad_norm : float or None, optional
        Global gradient-norm clip before each optimizer step.
    early_stopping_patience : int or None, optional
        Stop after this many non-improving epochs. Disabled when ``None``.
    early_stopping_min_delta : float, optional
        Minimum monitored-loss decrease to count as improvement.
    early_stopping_monitor : {"train", "val"}, optional
        Resolved monitor (not ``"auto"``). Default is ``"train"``.
    val_sequences : sequence of GraphSnapshotSequence or None, optional
        Optional held-out trajectories already validated by the caller.
    restore_best_weights : bool, optional
        Reload in-memory weights from the lowest-loss epoch when ``True``.
    checkpoint_path : str, Path, or None, optional
        Write a checkpoint at the lowest-loss epoch when set.
    **optimizer_kwargs
        Extra keyword arguments for the optimizer constructor.

    Returns
    -------
    FitHistory
        Per-epoch losses and early-stop metadata.

    Raises
    ------
    ValueError
        If ``early_stopping_monitor="val"`` without ``val_sequences``.
    """
    if early_stopping_monitor == "val" and val_sequences is None:
        msg = 'early_stopping_monitor="val" requires val_sequences'
        raise ValueError(msg)

    # Lazy import: avoid training → serialization → model edges at module load.
    from koopman_graph.serialization import save_checkpoint, snapshot_state_dict

    module = cast(nn.Module, model)
    train_device = resolve_device(model, device)
    module.to(train_device)
    train_sequences = [
        sequence_to_device(sequence, train_device) for sequence in train_sequences
    ]
    if val_sequences is not None:
        val_sequences = [
            sequence_to_device(sequence, train_device) for sequence in val_sequences
        ]

    optim = optimizer(module.parameters(), lr=lr, **optimizer_kwargs)
    scheduler = resolve_lr_scheduler(lr_scheduler, optim)
    window_sampler = (
        None
        if window_length is None
        else WindowSampler(
            train_sequences,
            window_length=window_length,
            batch_size=batch_size,
            windows_per_epoch=windows_per_epoch,
            seed=window_seed,
        )
    )
    losses: list[float] = []
    reconstruction_losses: list[float] = []
    forward_losses: list[float] = []
    backward_losses: list[float] = []
    rollout_losses: list[float] = []
    eigenvalue_losses: list[float] = []
    lie_losses: list[float] = []
    pde_losses: list[float] = []
    sparsity_losses: list[float] = []
    worst_case_losses: list[float] = []
    val_losses: list[float] | None = [] if val_sequences is not None else None
    val_reconstruction_losses: list[float] | None = (
        [] if val_sequences is not None else None
    )
    val_forward_losses: list[float] | None = [] if val_sequences is not None else None
    val_backward_losses: list[float] | None = [] if val_sequences is not None else None
    val_rollout_losses: list[float] | None = [] if val_sequences is not None else None
    val_eigenvalue_losses: list[float] | None = (
        [] if val_sequences is not None else None
    )
    val_lie_losses: list[float] | None = [] if val_sequences is not None else None
    val_pde_losses: list[float] | None = [] if val_sequences is not None else None
    val_sparsity_losses: list[float] | None = [] if val_sequences is not None else None
    val_worst_case_losses: list[float] | None = (
        [] if val_sequences is not None else None
    )
    best_loss_for_stop = float("inf")
    best_loss: float | None = None
    best_epoch: int | None = None
    best_state_dict: dict[str, Tensor] | None = None
    track_best = restore_best_weights or checkpoint_path is not None
    epochs_without_improvement = 0
    stopped_early = False

    for epoch in range(epochs):
        epoch_weights = resolve_loss_weights_for_epoch(
            epoch,
            loss_weights=loss_weights,
            loss_weight_schedule=loss_weight_schedule,
        )
        epoch_rollout_starts: list[int] | None = None
        if window_sampler is None:
            rollout_horizon_for_epoch = (
                train_sequences[0].num_timesteps - 1
                if rollout_horizon is None
                else rollout_horizon
            )
            epoch_rollout_starts = resolve_rollout_start_indices(
                train_sequences[0],
                horizon=rollout_horizon_for_epoch,
                rollout_start_indices=rollout_start_indices,
                rollout_starts_per_epoch=rollout_starts_per_epoch,
                rollout_start_seed=rollout_start_seed,
                epoch=epoch,
            )
            breakdown = train_one_epoch(
                model,
                train_sequences,
                optim,
                epoch_weights,
                extra_losses=extra_losses,
                max_grad_norm=max_grad_norm,
                rollout_horizon=rollout_horizon,
                rollout_start_indices=epoch_rollout_starts,
            )
        else:
            breakdown = train_windowed_epoch(
                model,
                window_sampler,
                optim,
                epoch_weights,
                extra_losses=extra_losses,
                epoch=epoch,
                max_grad_norm=max_grad_norm,
                rollout_horizon=rollout_horizon,
                rollout_start_indices=rollout_start_indices,
                rollout_starts_per_epoch=rollout_starts_per_epoch,
                rollout_start_seed=rollout_start_seed,
            )
        if scheduler is not None:
            scheduler.step()

        term_values = breakdown.to_floats()
        losses.append(term_values["total"])
        reconstruction_losses.append(term_values["reconstruction"])
        forward_losses.append(term_values["forward"])
        backward_losses.append(term_values["backward"])
        rollout_losses.append(term_values["rollout"])
        eigenvalue_losses.append(term_values["eigenvalue"])
        lie_losses.append(term_values["lie"])
        pde_losses.append(term_values["pde"])
        sparsity_losses.append(term_values["sparsity"])
        worst_case_losses.append(term_values["worst_case"])

        monitored_loss = term_values["total"]
        if val_sequences is not None:
            val_breakdown = eval_one_epoch(
                model,
                val_sequences,
                epoch_weights,
                extra_losses=extra_losses,
                rollout_horizon=rollout_horizon,
                rollout_start_indices=epoch_rollout_starts,
            )
            val_terms = val_breakdown.to_floats()
            assert val_losses is not None
            assert val_reconstruction_losses is not None
            assert val_forward_losses is not None
            assert val_backward_losses is not None
            assert val_rollout_losses is not None
            assert val_eigenvalue_losses is not None
            assert val_lie_losses is not None
            assert val_pde_losses is not None
            assert val_sparsity_losses is not None
            assert val_worst_case_losses is not None
            val_losses.append(val_terms["total"])
            val_reconstruction_losses.append(val_terms["reconstruction"])
            val_forward_losses.append(val_terms["forward"])
            val_backward_losses.append(val_terms["backward"])
            val_rollout_losses.append(val_terms["rollout"])
            val_eigenvalue_losses.append(val_terms["eigenvalue"])
            val_lie_losses.append(val_terms["lie"])
            val_pde_losses.append(val_terms["pde"])
            val_sparsity_losses.append(val_terms["sparsity"])
            val_worst_case_losses.append(val_terms["worst_case"])
            if early_stopping_monitor == "val":
                monitored_loss = val_terms["total"]

        if track_best and (best_loss is None or monitored_loss < best_loss):
            best_loss = monitored_loss
            best_epoch = epoch
            best_state_dict = snapshot_state_dict(module)

        if early_stopping_patience is not None:
            stop, best_loss_for_stop, epochs_without_improvement = should_stop_early(
                epoch_loss=monitored_loss,
                best_loss=best_loss_for_stop,
                epochs_without_improvement=epochs_without_improvement,
                patience=early_stopping_patience,
                min_delta=early_stopping_min_delta,
            )
            if stop:
                stopped_early = True
                break

    if track_best and best_state_dict is not None:
        last_state_dict: dict[str, Tensor] | None = None
        if not restore_best_weights:
            last_state_dict = snapshot_state_dict(module)
        module.load_state_dict(best_state_dict)
        if checkpoint_path is not None:
            save_checkpoint(model, checkpoint_path)  # type: ignore[arg-type]
        if not restore_best_weights and last_state_dict is not None:
            module.load_state_dict(last_state_dict)

    return FitHistory(
        loss=tuple(losses),
        epochs=len(losses),
        reconstruction_loss=tuple(reconstruction_losses),
        forward_loss=tuple(forward_losses),
        backward_loss=tuple(backward_losses),
        rollout_loss=tuple(rollout_losses),
        eigenvalue_loss=tuple(eigenvalue_losses),
        lie_loss=tuple(lie_losses),
        pde_loss=tuple(pde_losses),
        sparsity_loss=tuple(sparsity_losses),
        worst_case_loss=tuple(worst_case_losses),
        val_loss=None if val_losses is None else tuple(val_losses),
        val_reconstruction_loss=(
            None
            if val_reconstruction_losses is None
            else tuple(val_reconstruction_losses)
        ),
        val_forward_loss=(
            None if val_forward_losses is None else tuple(val_forward_losses)
        ),
        val_backward_loss=(
            None if val_backward_losses is None else tuple(val_backward_losses)
        ),
        val_rollout_loss=(
            None if val_rollout_losses is None else tuple(val_rollout_losses)
        ),
        val_eigenvalue_loss=(
            None if val_eigenvalue_losses is None else tuple(val_eigenvalue_losses)
        ),
        val_lie_loss=None if val_lie_losses is None else tuple(val_lie_losses),
        val_pde_loss=None if val_pde_losses is None else tuple(val_pde_losses),
        val_sparsity_loss=(
            None if val_sparsity_losses is None else tuple(val_sparsity_losses)
        ),
        val_worst_case_loss=(
            None if val_worst_case_losses is None else tuple(val_worst_case_losses)
        ),
        stopped_early=stopped_early,
        best_epoch=best_epoch,
        best_loss=best_loss,
    )
