"""Lightning Fabric adapter for KoopmanGraph fit orchestration.

Lazy-imports ``lightning.fabric.Fabric``. Shares the private epoch driver in
:mod:`koopman_graph.distributed._fit_epochs` with the native DDP path so
scientific loss / early-stop logic is not forked.

Autocast ownership: pass **either** a non-FP32 Fabric ``precision`` **or**
``use_amp=True`` — never both (raises ``ValueError``).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from koopman_graph.data import (
    RolloutStartIndices,
    SnapshotSequence,
    WindowLikeSampler,
)
from koopman_graph.distributed._fit_epochs import (
    fit_epochs_distributed,
    resolve_distributed_window_sampler,
)
from koopman_graph.distributed.ddp import unwrap_model
from koopman_graph.distributed.sampling import (
    DistributedWindowSampler,
    shard_sequences_for_rank,
)
from koopman_graph.protocols import TrainableKoopmanModel
from koopman_graph.training.device import sequence_to_device
from koopman_graph.training.epochs import prepare_training_amp
from koopman_graph.training.history import (
    ExtraLosses,
    FitHistory,
    LossWeights,
    LossWeightSchedule,
    LRSchedulerFactory,
)
from koopman_graph.training.loop import bind_pending_orbit_ties, resolve_lr_scheduler

__all__ = ["fit_with_fabric"]

_LIGHTNING_INSTALL_HINT = 'pip install "koopman-graph[lightning]"'
_FP32_PRECISIONS: frozenset[object] = frozenset({32, "32", "32-true"})


def _import_fabric_class() -> type[Any]:
    """Lazy-import Lightning Fabric with an actionable install hint.

    Returns
    -------
    type
        ``lightning.fabric.Fabric`` class.

    Raises
    ------
    ImportError
        If Lightning is not installed.
    """
    try:
        import importlib

        module = importlib.import_module("lightning.fabric")
    except ImportError as exc:
        msg = (
            "Lightning Fabric is required for fit_with_fabric; "
            f"install with: {_LIGHTNING_INSTALL_HINT}"
        )
        raise ImportError(msg) from exc
    return cast(type[Any], module.Fabric)


def _is_fp32_precision(precision: Any) -> bool:
    """Return whether ``precision`` leaves autocast ownership to the caller.

    Parameters
    ----------
    precision : object
        Fabric precision argument or resolved precision object.

    Returns
    -------
    bool
        ``True`` when ``precision`` is an FP32 sentinel in
        :data:`_FP32_PRECISIONS`.
    """
    return precision in _FP32_PRECISIONS


def _fabric_all_reduce_mean(fabric: Any, value: float) -> float:
    """Mean-reduce a scalar across Fabric ranks (no-op at world size 1).

    Parameters
    ----------
    fabric : lightning.fabric.Fabric
        Active Fabric instance.
    value : float
        Local rank scalar.

    Returns
    -------
    float
        Mean over ranks, or ``value`` when ``world_size <= 1``.
    """
    if int(getattr(fabric, "world_size", 1)) <= 1:
        return float(value)
    tensor = torch.tensor([float(value)], device=fabric.device, dtype=torch.float64)
    reduced = fabric.all_reduce(tensor, reduce_op="mean")
    return float(reduced.item())


def _fabric_broadcast_module_state(fabric: Any, module: nn.Module) -> None:
    """Broadcast parameters and buffers from global rank 0 via Fabric.

    Parameters
    ----------
    fabric : lightning.fabric.Fabric
        Active Fabric instance.
    module : nn.Module
        Module whose tensors are broadcast in-place.
    """
    if int(getattr(fabric, "world_size", 1)) <= 1:
        return
    for parameter in module.parameters():
        fabric.broadcast(parameter.data, src=0)
    for buffer in module.buffers():
        fabric.broadcast(buffer.data, src=0)


def fit_with_fabric(
    model: TrainableKoopmanModel,
    train_sequences: Sequence[SnapshotSequence],
    *,
    fabric: Any | None = None,
    accelerator: str = "auto",
    devices: int | str = "auto",
    precision: str | int = 32,
    strategy: str = "auto",
    epochs: int = 100,
    lr: float = 1e-3,
    optimizer: Callable[..., Optimizer] = torch.optim.Adam,
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
    sampler: WindowLikeSampler | DistributedWindowSampler | None = None,
    max_grad_norm: float | None = None,
    use_amp: bool = False,
    amp_dtype: torch.dtype | None = None,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
    early_stopping_monitor: Literal["train", "val"] = "train",
    val_sequences: Sequence[SnapshotSequence] | None = None,
    restore_best_weights: bool = False,
    checkpoint_path: str | Path | None = None,
    **optimizer_kwargs: Any,
) -> FitHistory:
    """Fit a Koopman model under Lightning Fabric device / precision setup.

    Reuses the same scientific epoch driver as
    :func:`~koopman_graph.distributed.run_ddp_fit_loop`. Fabric owns device
    placement and (when ``precision`` is not FP32) autocast. Do not combine
    non-FP32 ``precision`` with ``use_amp=True``. Homogeneous and multiplex
    (``koopman='hetero_graph'``) sequences share the same device-mover,
    sharding, and orbit-bind contracts as the DDP entry point.

    Parameters
    ----------
    model : TrainableKoopmanModel
        Trainable Koopman façade (composed, never subclassed for Fabric).
    train_sequences : sequence of SnapshotSequence
        Homogeneous or multiplex training trajectories (sharded inside this
        helper).
    fabric : lightning.fabric.Fabric or None, optional
        Pre-built Fabric. When ``None``, constructs one from ``accelerator``,
        ``devices``, ``precision``, and ``strategy``.
    accelerator : str, optional
        Fabric accelerator when ``fabric`` is ``None``. Default is ``"auto"``.
    devices : int or str, optional
        Fabric devices when ``fabric`` is ``None``. Default is ``"auto"``.
    precision : str or int, optional
        Fabric precision when ``fabric`` is ``None``. Default is ``32``.
    strategy : str, optional
        Fabric strategy when ``fabric`` is ``None``. Default is ``"auto"``.
    epochs : int, optional
        Number of training epochs. Default is ``100``.
    lr : float, optional
        Learning rate. Default is ``1e-3``.
    optimizer : callable, optional
        Optimizer class. Default is :class:`torch.optim.Adam`.
    loss_weights : LossWeights or None, optional
        Static loss weights.
    loss_weight_schedule : callable or None, optional
        Per-epoch weight schedule.
    extra_losses : ExtraLosses or None, optional
        Fit-time Lie / PDE callables.
    rollout_horizon : int or None, optional
        Autoregressive rollout steps.
    rollout_start_indices : sequence of int, ``"all"``, or None, optional
        Rollout origins.
    rollout_starts_per_epoch : int or None, optional
        Random rollout-origin sample size.
    rollout_start_seed : int or None, optional
        Base seed for rollout-origin sampling.
    lr_scheduler : LRScheduler or callable, optional
        Scheduler or factory.
    window_length : int or None, optional
        Build a :class:`DistributedWindowSampler` when set.
    batch_size : int, optional
        Windows per optimizer step. Default is ``8``.
    windows_per_epoch : int or None, optional
        Global window cap before sharding.
    window_seed : int or None, optional
        Base seed for window shuffling.
    sampler : DistributedWindowSampler or None, optional
        Pre-built rank-aware sampler. Non-distributed samplers are rejected.
    max_grad_norm : float or None, optional
        Gradient clipping threshold.
    use_amp : bool, optional
        Library AMP path. Allowed only when Fabric ``precision`` is FP32
        (``32`` / ``"32"`` / ``"32-true"``). Default is ``False``.
    amp_dtype : torch.dtype or None, optional
        Autocast dtype when library AMP is enabled.
    early_stopping_patience : int or None, optional
        Early-stopping patience.
    early_stopping_min_delta : float, optional
        Minimum improvement delta.
    early_stopping_monitor : {"train", "val"}, optional
        Resolved monitor (not ``"auto"``).
    val_sequences : sequence of SnapshotSequence or None, optional
        Held-out homogeneous or multiplex trajectories.
    restore_best_weights : bool, optional
        Restore best unwrapped weights at the end.
    checkpoint_path : str, Path, or None, optional
        Rank-0 checkpoint destination.
    optimizer_kwargs
        Extra optimizer constructor kwargs.

    Returns
    -------
    FitHistory
        Per-epoch losses and early-stop metadata.

    Raises
    ------
    ImportError
        If Lightning is not installed.
    ValueError
        If sampler / monitor / autocast configuration is invalid.
    """
    if early_stopping_monitor == "val" and val_sequences is None:
        msg = 'early_stopping_monitor="val" requires val_sequences'
        raise ValueError(msg)

    Fabric = _import_fabric_class()
    if fabric is None:
        fabric = Fabric(
            accelerator=accelerator,
            devices=devices,
            precision=precision,
            strategy=strategy,
        )
    resolved_precision: Any = precision
    if hasattr(fabric, "_precision"):
        resolved_precision = fabric._precision  # noqa: SLF001
    if use_amp and not _is_fp32_precision(resolved_precision):
        msg = (
            "fit_with_fabric refuses stacked autocast: set use_amp=False "
            "when Fabric precision is not FP32, or use precision=32 with "
            "use_amp=True"
        )
        raise ValueError(msg)

    train_device = torch.device(fabric.device)
    unwrapped = cast(nn.Module, model)
    unwrapped.to(train_device)
    train_sequences = [
        sequence_to_device(sequence, train_device) for sequence in train_sequences
    ]
    if val_sequences is not None:
        val_sequences = [
            sequence_to_device(sequence, train_device) for sequence in val_sequences
        ]

    bind_pending_orbit_ties(unwrapped, train_sequences)

    rank = int(fabric.global_rank)
    world_size = int(fabric.world_size)
    window_sampler = resolve_distributed_window_sampler(
        train_sequences,
        window_length=window_length,
        batch_size=batch_size,
        windows_per_epoch=windows_per_epoch,
        window_seed=window_seed,
        sampler=sampler,
        api_name="fit_with_fabric",
        rank=rank,
        world_size=world_size,
    )
    if window_sampler is not None:
        window_sampler.sequences = [
            sequence_to_device(sequence, train_device)
            for sequence in window_sampler.sequences
        ]
        train_shard: Sequence[SnapshotSequence] = train_sequences
    else:
        train_shard = shard_sequences_for_rank(
            train_sequences,
            rank=rank,
            world_size=world_size,
        )

    optim = optimizer(unwrapped.parameters(), lr=lr, **optimizer_kwargs)
    setup_out = fabric.setup(unwrapped, optim)
    if isinstance(setup_out, tuple):
        train_module, optim = setup_out
    else:
        train_module = setup_out
    trainable = cast(TrainableKoopmanModel, train_module)
    scheduler = resolve_lr_scheduler(lr_scheduler, optim)

    # When Fabric owns mixed precision, keep library AMP off.
    amp_use = bool(use_amp) and _is_fp32_precision(resolved_precision)
    amp_enabled, resolved_amp_dtype, grad_scaler = prepare_training_amp(
        amp_use,
        train_device,
        amp_dtype,
    )
    amp_kwargs = {
        "use_amp": amp_enabled,
        "amp_dtype": resolved_amp_dtype,
        "grad_scaler": grad_scaler,
    }

    # Lambdas avoid nested FunctionDef docstring requirements in test_docs.
    all_reduce_fn = lambda value: _fabric_all_reduce_mean(fabric, value)  # noqa: E731
    is_main_fn = lambda: bool(fabric.is_global_zero)  # noqa: E731
    broadcast_state_fn = (  # noqa: E731
        lambda module: _fabric_broadcast_module_state(fabric, module)
    )

    return fit_epochs_distributed(
        trainable,
        cast(nn.Module, train_module),
        train_shard=train_shard,
        window_sampler=window_sampler,
        optim=optim,
        scheduler=scheduler,
        epochs=epochs,
        loss_weights=loss_weights,
        loss_weight_schedule=loss_weight_schedule,
        extra_losses=extra_losses,
        rollout_horizon=rollout_horizon,
        rollout_start_indices=rollout_start_indices,
        rollout_starts_per_epoch=rollout_starts_per_epoch,
        rollout_start_seed=rollout_start_seed,
        max_grad_norm=max_grad_norm,
        amp_kwargs=amp_kwargs,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        early_stopping_monitor=early_stopping_monitor,
        val_sequences=val_sequences,
        restore_best_weights=restore_best_weights,
        checkpoint_path=checkpoint_path,
        all_reduce_fn=all_reduce_fn,
        is_main_fn=is_main_fn,
        broadcast_state_fn=broadcast_state_fn,
        unwrap_fn=unwrap_model,
    )
