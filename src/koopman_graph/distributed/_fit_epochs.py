"""Shared multi-epoch driver for native DDP and Fabric adapters.

Private module (leading underscore). Not part of ``koopman_graph.distributed``
public ``__all__``. Callers inject metric reduction, main-process gating, and
state broadcast so DDP and Fabric stay framework-specific only at the edges.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal, cast

from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from koopman_graph.data import (
    RolloutStartIndices,
    SnapshotSequence,
    WindowLikeSampler,
    resolve_rollout_start_indices,
    resolve_window_sampler,
)
from koopman_graph.distributed.sampling import DistributedWindowSampler
from koopman_graph.protocols import TrainableKoopmanModel
from koopman_graph.training.epochs import (
    eval_one_epoch,
    train_one_epoch,
    train_windowed_epoch,
)
from koopman_graph.training.history import (
    ExtraLosses,
    FitHistory,
    LossWeights,
    LossWeightSchedule,
)
from koopman_graph.training.loop import should_stop_early
from koopman_graph.training.schedules import resolve_loss_weights_for_epoch

AllReduceFn = Callable[[float], float]
IsMainFn = Callable[[], bool]
BroadcastStateFn = Callable[[nn.Module], None]
UnwrapFn = Callable[[nn.Module], nn.Module]


def resolve_distributed_window_sampler(
    train_sequences: Sequence[SnapshotSequence],
    *,
    window_length: int | None,
    batch_size: int,
    windows_per_epoch: int | None,
    window_seed: int | None,
    sampler: WindowLikeSampler | DistributedWindowSampler | None,
    api_name: str,
    rank: int | None = None,
    world_size: int | None = None,
) -> DistributedWindowSampler | None:
    """Build or accept a rank-aware window sampler for distributed fits.

    Parameters
    ----------
    train_sequences : sequence of SnapshotSequence
        Device-local homogeneous or multiplex training trajectories.
    window_length : int or None
        Window length when constructing a sampler.
    batch_size : int
        Windows per batch.
    windows_per_epoch : int or None
        Global window cap before sharding.
    window_seed : int or None
        Base shuffle seed.
    sampler : WindowLikeSampler, DistributedWindowSampler, or None
        Pre-built sampler. Non-distributed samplers are rejected.
    api_name : str
        Public API name used in error messages.
    rank : int or None, optional
        Rank override when constructing a sampler.
    world_size : int or None, optional
        World-size override when constructing a sampler.

    Returns
    -------
    DistributedWindowSampler or None
        Sampler for windowed epochs, or ``None`` for full-sequence training.

    Raises
    ------
    ValueError
        If both ``sampler`` and ``window_length`` are set, or if ``sampler`` is
        a non-distributed window sampler.
    TypeError
        If ``sampler`` has an unsupported type.
    """
    resolved = resolve_window_sampler(
        train_sequences,
        window_length=window_length,
        batch_size=batch_size,
        windows_per_epoch=windows_per_epoch,
        window_seed=window_seed,
        sampler=sampler,
        distributed=True,
        rank=rank,
        world_size=world_size,
        api_name=api_name,
    )
    if resolved is None:
        return None
    # Shared resolver returns DistributedWindowSampler in distributed mode.
    return cast(DistributedWindowSampler, resolved)


def fit_epochs_distributed(
    trainable: TrainableKoopmanModel,
    train_module: nn.Module,
    *,
    train_shard: Sequence[SnapshotSequence],
    window_sampler: DistributedWindowSampler | None,
    optim: Optimizer,
    scheduler: LRScheduler | None,
    epochs: int,
    loss_weights: LossWeights | None,
    loss_weight_schedule: LossWeightSchedule | None,
    extra_losses: ExtraLosses | None,
    rollout_horizon: int | None,
    rollout_start_indices: RolloutStartIndices,
    rollout_starts_per_epoch: int | None,
    rollout_start_seed: int | None,
    max_grad_norm: float | None,
    amp_kwargs: dict[str, Any],
    early_stopping_patience: int | None,
    early_stopping_min_delta: float,
    early_stopping_monitor: Literal["train", "val"],
    val_sequences: Sequence[SnapshotSequence] | None,
    restore_best_weights: bool,
    checkpoint_path: str | Path | None,
    all_reduce_fn: AllReduceFn,
    is_main_fn: IsMainFn,
    broadcast_state_fn: BroadcastStateFn,
    unwrap_fn: UnwrapFn,
) -> FitHistory:
    """Run the shared distributed multi-epoch fit loop.

    Parameters
    ----------
    trainable : TrainableKoopmanModel
        Model (possibly wrapped) passed to epoch helpers.
    train_module : nn.Module
        Same object as ``trainable`` for unwrap / checkpoint paths.
    train_shard : sequence of SnapshotSequence
        Per-rank full-sequence shard (ignored when ``window_sampler`` is set).
    window_sampler : DistributedWindowSampler or None
        Rank-aware window sampler when windowed.
    optim : Optimizer
        Optimizer already bound to trainable parameters.
    scheduler : LRScheduler or None
        Optional epoch scheduler.
    epochs : int
        Number of epochs.
    loss_weights, loss_weight_schedule, extra_losses
        Scientific loss configuration (same meaning as ``run_fit_loop``).
    rollout_horizon, rollout_start_indices, rollout_starts_per_epoch,
    rollout_start_seed
        Rollout configuration.
    max_grad_norm : float or None
        Gradient clipping threshold.
    amp_kwargs : dict
        Forwarded to epoch helpers (``use_amp``, ``amp_dtype``, ``grad_scaler``).
    early_stopping_patience, early_stopping_min_delta, early_stopping_monitor
        Early-stopping configuration (monitor must already be resolved).
    val_sequences : sequence of SnapshotSequence or None
        Held-out homogeneous or multiplex trajectories.
    restore_best_weights : bool
        Whether to leave best weights loaded at the end.
    checkpoint_path : str, Path, or None
        Main-process checkpoint destination.
    all_reduce_fn : callable
        Scalar mean reduction across ranks.
    is_main_fn : callable
        Whether this rank should write checkpoints.
    broadcast_state_fn : callable
        Broadcast unwrapped module state from rank 0.
    unwrap_fn : callable
        Strip DDP / Fabric wrappers for snapshots and checkpoints.

    Returns
    -------
    FitHistory
        Per-epoch all-reduced losses and early-stop metadata.
    """
    from koopman_graph.serialization import save_checkpoint, snapshot_state_dict

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
    vamp2_losses: list[float] = []
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
    val_vamp2_losses: list[float] | None = [] if val_sequences is not None else None
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
                train_shard[0].num_timesteps - 1
                if rollout_horizon is None
                else rollout_horizon
            )
            epoch_rollout_starts = resolve_rollout_start_indices(
                train_shard[0],
                horizon=rollout_horizon_for_epoch,
                rollout_start_indices=rollout_start_indices,
                rollout_starts_per_epoch=rollout_starts_per_epoch,
                rollout_start_seed=rollout_start_seed,
                epoch=epoch,
            )
            breakdown = train_one_epoch(
                trainable,
                train_shard,
                optim,
                epoch_weights,
                extra_losses=extra_losses,
                max_grad_norm=max_grad_norm,
                rollout_horizon=rollout_horizon,
                rollout_start_indices=epoch_rollout_starts,
                **amp_kwargs,
            )
        else:
            breakdown = train_windowed_epoch(
                trainable,
                cast(Any, window_sampler),
                optim,
                epoch_weights,
                extra_losses=extra_losses,
                epoch=epoch,
                max_grad_norm=max_grad_norm,
                rollout_horizon=rollout_horizon,
                rollout_start_indices=rollout_start_indices,
                rollout_starts_per_epoch=rollout_starts_per_epoch,
                rollout_start_seed=rollout_start_seed,
                **amp_kwargs,
            )
        if scheduler is not None:
            scheduler.step()

        term_values = {
            key: all_reduce_fn(value) for key, value in breakdown.to_floats().items()
        }
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
        vamp2_losses.append(term_values["vamp2"])

        monitored_loss = term_values["total"]
        if val_sequences is not None:
            val_breakdown = eval_one_epoch(
                trainable,
                val_sequences,
                epoch_weights,
                extra_losses=extra_losses,
                rollout_horizon=rollout_horizon,
                rollout_start_indices=epoch_rollout_starts,
            )
            val_terms = {
                key: all_reduce_fn(value)
                for key, value in val_breakdown.to_floats().items()
            }
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
            assert val_vamp2_losses is not None
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
            val_vamp2_losses.append(val_terms["vamp2"])
            if early_stopping_monitor == "val":
                monitored_loss = val_terms["total"]

        if track_best and (best_loss is None or monitored_loss < best_loss):
            best_loss = monitored_loss
            best_epoch = epoch
            best_state_dict = snapshot_state_dict(unwrap_fn(train_module))

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
        core = unwrap_fn(train_module)
        if not restore_best_weights:
            last_state_dict = snapshot_state_dict(core)
        core.load_state_dict(best_state_dict)
        broadcast_state_fn(core)
        if checkpoint_path is not None and is_main_fn():
            save_checkpoint(cast(Any, core), checkpoint_path)
        if not restore_best_weights and last_state_dict is not None:
            core.load_state_dict(last_state_dict)
            broadcast_state_fn(core)

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
        vamp2_loss=tuple(vamp2_losses),
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
        val_vamp2_loss=(None if val_vamp2_losses is None else tuple(val_vamp2_losses)),
        stopped_early=stopped_early,
        best_epoch=best_epoch,
        best_loss=best_loss,
    )
