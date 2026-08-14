"""Optional Ray Train backend for multi-GPU **model** DDP fits.

Provides :func:`run_ray_train_fit_loop`, a **separate** entry point from
:func:`~koopman_graph.distributed.run_ddp_fit_loop` (Q4) that wraps Ray
Train ``TorchTrainer`` around the shared
:func:`~koopman_graph.distributed._fit_epochs.fit_epochs_distributed`
scientific driver.

Trainer taxonomy (do not conflate)
---------------------------------
* **Recommended default** for multi-GPU *model* training:
  :func:`~koopman_graph.distributed.run_ddp_fit_loop` (native
  ``torchrun``) or :func:`~koopman_graph.distributed.fit_with_fabric`.
* **This module:** Ray Train orchestration of the *same* scientific epoch
  driver for a single model (optional ``[ray]`` / ``ray[train]`` extra).
* **Not this module:** :func:`~koopman_graph.distributed.fit_ensemble_with_ray`
  parallelizes independent ensemble *members*, each with single-process
  ``fit`` — it does not shard one model across GPUs.

Scope and limits
----------------
CI gates world-size-1 parity only. A single-node multi-GPU manual smoke
(for example ``num_workers=2``, ``use_gpu=True`` on a two-GPU host) is
maintainer-run and is **not** a CI gate. This path is **not** a supported
multi-node production trainer, and it does **not** shrink dense
:math:`N\\cdot d` operator ceilings.

Ray is not imported at module load time, so
:mod:`koopman_graph.distributed` stays importable without the ``[ray]``
extra until this submodule is exercised.
"""

from __future__ import annotations

import importlib
import os
import tempfile
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
from koopman_graph.distributed.ddp import (
    _broadcast_module_state,
    all_reduce_mean,
    prepare_ddp_model,
    unwrap_model,
)
from koopman_graph.distributed.process import (
    init_process_group_from_env,
    is_main_process,
)
from koopman_graph.distributed.sampling import (
    DistributedWindowSampler,
    shard_sequences_for_rank,
)
from koopman_graph.distributed.seed import seed_everything
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

__all__ = ["run_ray_train_fit_loop", "multinode_recipe_enabled"]


def multinode_recipe_enabled() -> bool:
    """Return whether opt-in multi-node smoke is requested.

    Default CI stays single-process. Set ``KOOPMAN_GRAPH_MULTINODE=1`` to
    enable the documented ``torchrun`` / Ray Train multi-node recipe.

    Returns
    -------
    bool
        ``True`` when the opt-in flag is set.
    """
    return os.environ.get("KOOPMAN_GRAPH_MULTINODE") == "1"


_RAY_INSTALL_HINT = 'pip install "koopman-graph[ray]"'
_RESULT_FILENAME = "ray_train_result.pt"


def _unwrapped_state_dict(train_module: nn.Module) -> dict[str, Any]:
    """Return a format-1-safe ``state_dict`` with no ``module.`` prefix.

    Reuses :func:`~koopman_graph.distributed.unwrap_model` (the 0.8.0 DDP
    checkpoint lesson). Ray Train result payloads and driver restores must
    use this path — never ``train_module.state_dict()`` on a wrapped module.

    Parameters
    ----------
    train_module : nn.Module
        Possibly DDP-wrapped trainable module.

    Returns
    -------
    dict of str to Tensor
        Unwrapped ``state_dict`` keys matching single-process checkpoints.
    """
    return unwrap_model(train_module).state_dict()


def _prefer_driver_interpreter_for_uv_run() -> None:
    """Disable Ray's ``uv run`` worker rewrite when unset.

    Under ``uv run --no-sync`` after ``uv sync``, Ray's default UV runtime-env
    hook can rewrite workers as ``uv run ... python`` and miss the synced
    project environment. Prefer the driver interpreter instead.

    Notes
    -----
    Must run before ``import ray`` so Ray reads the flag at import time.
    """
    os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")


def _import_ray_train() -> Any:
    """Import ``ray.train`` with an actionable install hint.

    Returns
    -------
    module
        The ``ray.train`` package.

    Raises
    ------
    ImportError
        If Ray / Ray Train is not installed (``[ray]`` extra missing).
    """
    _prefer_driver_interpreter_for_uv_run()
    try:
        importlib.import_module("ray")
        return importlib.import_module("ray.train")
    except ImportError as exc:
        msg = (
            "Ray Train is required for the Ray Train fit backend; "
            f"install with: {_RAY_INSTALL_HINT}"
        )
        raise ImportError(msg) from exc


def _worker_device() -> torch.device:
    """Resolve the local Ray Train torch device for this worker.

    Returns
    -------
        See signature.
    """
    ray_torch = importlib.import_module("ray.train.torch")
    return torch.device(ray_torch.get_device())


def run_ray_train_fit_loop(
    model: TrainableKoopmanModel,
    train_sequences: Sequence[SnapshotSequence],
    *,
    num_workers: int = 1,
    use_gpu: bool = False,
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
    seed: int | None = None,
    find_unused_parameters: bool | None = None,
    **optimizer_kwargs: Any,
) -> FitHistory:
    """Fit a Koopman model under Ray Train ``TorchTrainer`` orchestration.

    Reuses :func:`~koopman_graph.distributed._fit_epochs.fit_epochs_distributed`
    (same scientific epoch driver as
    :func:`~koopman_graph.distributed.run_ddp_fit_loop`). This is a **separate**
    function from the native DDP entry point so the ``[ray]`` lazy-import
    boundary stays explicit (Q4). Prefer native DDP / Fabric for routine
    multi-GPU model training; use this entry when Ray Train orchestration is
    required.

    Workers use the package DDP attribute-forwarding wrapper rather than
    stock ``ray.train.torch.prepare_model``, so façade attributes such as
    ``.koopman`` remain reachable during training. Checkpoints (Train result
    payload and optional ``checkpoint_path`` via the shared epoch driver)
    always persist **unwrapped** state-dict keys through
    :func:`~koopman_graph.distributed.unwrap_model` — no ``module.`` prefix
    and no load-time prefix migration shim.

    Notes
    -----
    Distinct from :func:`~koopman_graph.distributed.fit_ensemble_with_ray`,
    which parallelizes ensemble *members*. World-size-1 CI parity is the
    automated gate; a single-node multi-GPU smoke
    (``num_workers>=2``, ``use_gpu=True``) is manual and not CI-gated. Not a
    multi-node production path; does not reduce dense :math:`N\\cdot d`
    memory ceilings.

    Parameters
    ----------
    model : TrainableKoopmanModel
        Trainable Koopman façade. After a successful fit, parameters are
        updated from the rank-0 worker checkpoint.
    train_sequences : sequence of SnapshotSequence
        Training trajectories (pickled into workers; intended for small /
        single-node smokes — not a multi-node production path).
    num_workers : int, optional
        Ray Train worker count. Default is ``1``.
    use_gpu : bool, optional
        Forwarded to :class:`~ray.train.ScalingConfig`. Default is ``False``.
    epochs, lr, optimizer, loss_weights, loss_weight_schedule, extra_losses,
    rollout_horizon, rollout_start_indices, rollout_starts_per_epoch,
    rollout_start_seed, lr_scheduler, window_length, batch_size,
    windows_per_epoch, window_seed, sampler, max_grad_norm, use_amp,
    amp_dtype, early_stopping_patience, early_stopping_min_delta,
    early_stopping_monitor, val_sequences, restore_best_weights,
    checkpoint_path, find_unused_parameters, optimizer_kwargs
        Scientific / DDP-parallel kwargs with the same meaning as
        :func:`~koopman_graph.distributed.run_ddp_fit_loop`.
    seed : int or None, optional
        When set, seeds each worker with ``seed + world_rank``.

    Returns
    -------
    FitHistory
        Per-epoch losses and early-stop metadata from the rank-0 worker.

    Raises
    ------
    ImportError
        If the ``[ray]`` extra (``ray[train]``) is not installed.
    ValueError
        If ``num_workers < 1`` or monitor/sampler configuration is invalid.
    RuntimeError
        If Ray Train finishes without a recoverable result checkpoint.
    """
    _import_ray_train()
    if num_workers < 1:
        msg = f"num_workers must be >= 1, got {num_workers}"
        raise ValueError(msg)
    if early_stopping_monitor == "val" and val_sequences is None:
        msg = 'early_stopping_monitor="val" requires val_sequences'
        raise ValueError(msg)

    ray_train = importlib.import_module("ray.train")
    TorchTrainer = importlib.import_module("ray.train.torch").TorchTrainer
    ScalingConfig = ray_train.ScalingConfig
    Checkpoint = ray_train.Checkpoint
    report = ray_train.report

    # Capture driver objects for worker pickling (tiny / single-node smoke).
    """Ray Train worker: shared epoch driver + report FitHistory.

Notes
-----
See signature.
    """
    worker_model = cast(nn.Module, model)
    worker_train = list(train_sequences)
    worker_val = None if val_sequences is None else list(val_sequences)

    def train_loop_per_worker() -> None:
        """Ray Train worker: shared epoch driver + report FitHistory.

        Notes
        -----
        See signature."""
        if seed is not None:
            rank = int(ray_train.get_context().get_world_rank())
            seed_everything(seed + rank)

        # TorchTrainer initializes the process group before the loop.
        init_process_group_from_env()
        train_device = _worker_device()
        local_model = worker_model
        local_model.to(train_device)
        local_train = [
            sequence_to_device(sequence, train_device) for sequence in worker_train
        ]
        local_val = None
        if worker_val is not None:
            local_val = [
                sequence_to_device(sequence, train_device) for sequence in worker_val
            ]

        bind_pending_orbit_ties(local_model, local_train)

        window_sampler = resolve_distributed_window_sampler(
            local_train,
            window_length=window_length,
            batch_size=batch_size,
            windows_per_epoch=windows_per_epoch,
            window_seed=window_seed,
            sampler=sampler,
            api_name="run_ray_train_fit_loop",
        )
        if window_sampler is not None:
            window_sampler.sequences = [
                sequence_to_device(sequence, train_device)
                for sequence in window_sampler.sequences
            ]
            train_shard: Sequence[SnapshotSequence] = local_train
        else:
            train_shard = shard_sequences_for_rank(local_train)

        train_module = prepare_ddp_model(
            cast(TrainableKoopmanModel, local_model),
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

        history = fit_epochs_distributed(
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
            val_sequences=local_val,
            restore_best_weights=restore_best_weights,
            checkpoint_path=checkpoint_path,
            all_reduce_fn=all_reduce_mean,
            is_main_fn=is_main_process,
            broadcast_state_fn=_broadcast_module_state,
            unwrap_fn=unwrap_model,
        )

        final_loss = float(history.loss[-1]) if history.loss else float("nan")
        metrics = {"epochs": int(history.epochs), "final_loss": final_loss}
        if is_main_process():
            with tempfile.TemporaryDirectory() as tmp:
                ckpt_dir = Path(tmp)
                torch.save(
                    {
                        "fit_history": history,
                        "state_dict": _unwrapped_state_dict(train_module),
                    },
                    ckpt_dir / _RESULT_FILENAME,
                )
                report(
                    metrics,
                    checkpoint=Checkpoint.from_directory(str(ckpt_dir)),
                )
        else:
            report(metrics)

    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=use_gpu),
    )
    result = trainer.fit()
    if getattr(result, "error", None) is not None:
        raise result.error

    checkpoint = result.checkpoint
    if checkpoint is None:
        msg = "Ray Train fit finished without a result checkpoint"
        raise RuntimeError(msg)

    with checkpoint.as_directory() as directory:
        payload = torch.load(
            Path(directory) / _RESULT_FILENAME,
            map_location="cpu",
            weights_only=False,
        )
    history = cast(FitHistory, payload["fit_history"])
    cast(nn.Module, model).load_state_dict(payload["state_dict"])
    return history
