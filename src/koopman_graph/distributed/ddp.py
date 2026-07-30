"""Native DistributedDataParallel fit orchestration.

Composes :mod:`koopman_graph.training` epoch helpers with rank-aware sampling.
When no process group is active (``world_size == 1``), wrapping is skipped so
single-process callers match :func:`~koopman_graph.training.run_fit_loop`.

Q7 decision: epoch helpers train through a thin :class:`DistributedDataParallel`
subclass that forwards unknown attributes (``koopman``, ``encode``, …) to
``.module``. Stock PyTorch DDP (including 2.13) does **not** forward arbitrary
attrs, and the training path uses those façades rather than ``forward()``.
Checkpoints and ``state_dict`` snapshots always use the unwrapped module so
format-1 payloads never contain a ``module.`` prefix.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal, cast

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
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
from koopman_graph.distributed.process import (
    get_rank,
    get_world_size,
    init_process_group_from_env,
    is_main_process,
)
from koopman_graph.distributed.sampling import (
    DistributedWindowSampler,
    shard_sequences_for_rank,
)
from koopman_graph.protocols import TrainableKoopmanModel
from koopman_graph.training.device import resolve_device, sequence_to_device
from koopman_graph.training.epochs import prepare_training_amp
from koopman_graph.training.history import (
    ExtraLosses,
    FitHistory,
    LossWeights,
    LossWeightSchedule,
    LRSchedulerFactory,
)
from koopman_graph.training.loop import bind_pending_orbit_ties, resolve_lr_scheduler

__all__ = [
    "all_reduce_mean",
    "prepare_ddp_model",
    "resolve_find_unused_parameters",
    "run_ddp_fit_loop",
    "unwrap_model",
]


class _AttributeForwardDDP(DistributedDataParallel):
    """DDP wrapper that forwards unknown attributes to ``.module``.

    Stock :class:`~torch.nn.parallel.DistributedDataParallel` only resolves
    registered parameters / buffers / submodules. KoopmanGraph training
    accesses façade attributes such as ``.koopman`` and ``.encode`` on the
    trainable module, so this subclass restores the historical forwarding
    pattern used by many application-level DDP helpers.

    Notes
    -----
    Checkpoints and ``state_dict`` snapshots should use
    :func:`unwrap_model` so format-1 payloads never include a ``module.``
    prefix.
    """

    def __getattr__(self, name: str) -> Any:
        """Resolve ``name`` on the wrapper, then on the wrapped module.

        Parameters
        ----------
        name : str
            Attribute name requested on the DDP wrapper.

        Returns
        -------
        object
            Attribute from the wrapper or from ``self.module``.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying module when ``model`` is DDP-wrapped.

    Parameters
    ----------
    model : nn.Module
        Possibly :class:`~torch.nn.parallel.DistributedDataParallel`-wrapped
        module.

    Returns
    -------
    nn.Module
        ``model.module`` when wrapped, otherwise ``model``.
    """
    if isinstance(model, DistributedDataParallel):
        return cast(nn.Module, model.module)
    return model


def all_reduce_mean(value: float) -> float:
    """Average a scalar across ranks, or return it unchanged when inactive.

    Parameters
    ----------
    value : float
        Local rank scalar (typically an epoch loss).

    Returns
    -------
    float
        Mean over the process group, or ``value`` when distributed is
        inactive / ``world_size == 1``.
    """
    if not dist.is_available() or not dist.is_initialized() or get_world_size() == 1:
        return float(value)
    tensor = torch.tensor([float(value)], dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= float(get_world_size())
    return float(tensor.item())


def resolve_find_unused_parameters(
    model: TrainableKoopmanModel,
    find_unused_parameters: bool | None,
) -> bool:
    """Resolve DDP ``find_unused_parameters`` with hetero-aware defaults.

    Parameters
    ----------
    model : TrainableKoopmanModel
        Model whose ``koopman_kind`` selects the default when the argument is
        ``None``.
    find_unused_parameters : bool or None
        Explicit override, or ``None`` for the kind-dependent default.

    Returns
    -------
    bool
        ``True`` when ``koopman_kind == "hetero_graph"`` and the argument is
        ``None``; otherwise ``False`` when ``None``; explicit booleans pass
        through.
    """
    if find_unused_parameters is not None:
        return bool(find_unused_parameters)
    return getattr(model, "koopman_kind", "pernode") == "hetero_graph"


def prepare_ddp_model(
    model: TrainableKoopmanModel,
    *,
    device: torch.device,
    find_unused_parameters: bool | None = None,
) -> nn.Module:
    """Move ``model`` to ``device`` and wrap with DDP when a group is active.

    Parameters
    ----------
    model : TrainableKoopmanModel
        Trainable Koopman façade (also an ``nn.Module``).
    device : torch.device
        Target device for parameters.
    find_unused_parameters : bool or None, optional
        Forwarded to :class:`~torch.nn.parallel.DistributedDataParallel`.
        ``None`` (default) resolves to ``True`` for
        ``koopman_kind="hetero_graph"`` and ``False`` otherwise.

    Returns
    -------
    nn.Module
        Unwrapped module when ``world_size == 1`` and no process group is
        initialized; otherwise a DDP wrapper around the same module.

    Raises
    ------
    RuntimeError
        If ``world_size > 1`` but the process group is not initialized.
    """
    resolved_unused = resolve_find_unused_parameters(model, find_unused_parameters)
    module = cast(nn.Module, model)
    module.to(device)
    if get_world_size() == 1 and not (dist.is_available() and dist.is_initialized()):
        return module
    if not (dist.is_available() and dist.is_initialized()):
        msg = (
            "prepare_ddp_model requires an initialized process group when "
            f"world_size={get_world_size()}; call init_process_group_from_env() "
            "or launch with torchrun"
        )
        raise RuntimeError(msg)
    local_rank = int(os.environ.get("LOCAL_RANK", get_rank()))
    if device.type == "cuda":
        return _AttributeForwardDDP(
            module,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=resolved_unused,
        )
    return _AttributeForwardDDP(
        module,
        find_unused_parameters=resolved_unused,
    )


def _broadcast_module_state(module: nn.Module, *, src: int = 0) -> None:
    """Broadcast parameters and buffers from ``src`` to all ranks.

    Parameters
    ----------
    module : nn.Module
        Unwrapped module whose tensors are broadcast in-place.
    src : int, optional
        Source rank. Default is ``0``.
    """
    if not dist.is_available() or not dist.is_initialized() or get_world_size() == 1:
        return
    for parameter in module.parameters():
        dist.broadcast(parameter.data, src=src)
    for buffer in module.buffers():
        dist.broadcast(buffer.data, src=src)


def _resolve_ddp_device(
    model: TrainableKoopmanModel,
    device: str | torch.device | None,
) -> torch.device:
    """Resolve training device, honoring ``LOCAL_RANK`` on CUDA launches.

    Parameters
    ----------
    model : TrainableKoopmanModel
        Model used as fallback when ``device`` is ``None``.
    device : str, torch.device, or None
        Explicit device override.

    Returns
    -------
    torch.device
        Resolved training device.
    """
    if device is not None:
        return torch.device(device)
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None and local_rank != "" and torch.cuda.is_available():
        return torch.device("cuda", int(local_rank))
    return resolve_device(model, None)


def run_ddp_fit_loop(
    model: TrainableKoopmanModel,
    train_sequences: Sequence[SnapshotSequence],
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
    find_unused_parameters: bool | None = None,
    **optimizer_kwargs: Any,
) -> FitHistory:
    """Run a rank-aware multi-epoch fit using native PyTorch DDP.

    Mirrors :func:`~koopman_graph.training.run_fit_loop` for scientific kwargs.
    Shards training trajectories or windows across ranks; validation runs on
    the full ``val_sequences`` on every rank (small held-out sets stay
    comparable). Metrics are all-reduced before early stopping. Checkpoints
    are written only on the main process from the unwrapped module.

    Parameters
    ----------
    model : TrainableKoopmanModel
        Trainable Koopman façade.
    train_sequences : sequence of SnapshotSequence
        Already-validated training trajectories (all ranks receive the full
        list; sharding happens inside this helper).
    epochs : int, optional
        Number of training epochs. Default is ``100``.
    lr : float, optional
        Learning rate. Default is ``1e-3``.
    optimizer : callable, optional
        Optimizer class. Default is :class:`torch.optim.Adam`.
    device : str, torch.device, or None, optional
        Training device. Honors ``LOCAL_RANK`` when CUDA is available.
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
        CUDA AMP flag. Default is ``False``.
    amp_dtype : torch.dtype or None, optional
        Autocast dtype.
    early_stopping_patience : int or None, optional
        Early-stopping patience.
    early_stopping_min_delta : float, optional
        Minimum improvement delta.
    early_stopping_monitor : {"train", "val"}, optional
        Resolved monitor (not ``"auto"``).
    val_sequences : sequence of SnapshotSequence or None, optional
        Held-out homogeneous or multiplex trajectories (evaluated in full on
        every rank).
    restore_best_weights : bool, optional
        Restore best unwrapped weights at the end.
    checkpoint_path : str, Path, or None, optional
        Rank-0 checkpoint destination.
    find_unused_parameters : bool or None, optional
        DDP ``find_unused_parameters``. ``None`` (default) resolves to
        ``True`` for ``koopman_kind="hetero_graph"`` and ``False`` otherwise.
    **optimizer_kwargs
        Extra optimizer constructor kwargs.

    Returns
    -------
    FitHistory
        Per-epoch losses (all-reduced totals) and early-stop metadata.

    Raises
    ------
    ValueError
        If monitor/sampler configuration is invalid.
    """
    if early_stopping_monitor == "val" and val_sequences is None:
        msg = 'early_stopping_monitor="val" requires val_sequences'
        raise ValueError(msg)

    init_process_group_from_env()

    train_device = _resolve_ddp_device(model, device)
    if train_device.type == "cuda":
        local_rank = int(os.environ.get("LOCAL_RANK", get_rank()))
        torch.cuda.set_device(local_rank)

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

    window_sampler = resolve_distributed_window_sampler(
        train_sequences,
        window_length=window_length,
        batch_size=batch_size,
        windows_per_epoch=windows_per_epoch,
        window_seed=window_seed,
        sampler=sampler,
        api_name="run_ddp_fit_loop",
    )
    if window_sampler is not None:
        window_sampler.sequences = [
            sequence_to_device(sequence, train_device)
            for sequence in window_sampler.sequences
        ]
        train_shard: Sequence[SnapshotSequence] = train_sequences
    else:
        train_shard = shard_sequences_for_rank(train_sequences)

    train_module = prepare_ddp_model(
        cast(TrainableKoopmanModel, unwrapped),
        device=train_device,
        find_unused_parameters=find_unused_parameters,
    )
    trainable = cast(TrainableKoopmanModel, train_module)

    optim = optimizer(train_module.parameters(), lr=lr, **optimizer_kwargs)
    scheduler = resolve_lr_scheduler(lr_scheduler, optim)
    amp_enabled, resolved_amp_dtype, grad_scaler = prepare_training_amp(
        use_amp,
        train_device,
        amp_dtype,
    )
    amp_kwargs = {
        "use_amp": amp_enabled,
        "amp_dtype": resolved_amp_dtype,
        "grad_scaler": grad_scaler,
    }

    return fit_epochs_distributed(
        trainable,
        train_module,
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
        all_reduce_fn=all_reduce_mean,
        is_main_fn=is_main_process,
        broadcast_state_fn=_broadcast_module_state,
        unwrap_fn=unwrap_model,
    )
