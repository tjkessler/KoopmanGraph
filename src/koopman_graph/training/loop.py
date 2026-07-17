"""Epoch helpers, input resolution, and the multi-epoch fit loop."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch_geometric.data import Data

from koopman_graph.data import (
    GraphSnapshotSequence,
    MultiTrajectory,
    RolloutStartIndices,
    WindowSampler,
    resolve_rollout_start_indices,
    resolve_sequence,
)
from koopman_graph.protocols import TrainableKoopmanModel
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
from koopman_graph.training.objectives import compute_training_loss
from koopman_graph.training.schedules import resolve_loss_weights_for_epoch


def train_one_epoch(
    model: TrainableKoopmanModel,
    sequences: GraphSnapshotSequence | Sequence[GraphSnapshotSequence],
    optimizer: Optimizer,
    loss_weights: LossWeights,
    *,
    max_grad_norm: float | None = None,
    rollout_horizon: int | None = None,
    rollout_start_indices: Sequence[int] | None = None,
) -> TrainingLossBreakdown:
    """Run one training epoch and return the averaged loss breakdown.

    Parameters
    ----------
    model : TrainableKoopmanModel
        Model satisfying :class:`~koopman_graph.protocols.TrainableKoopmanModel`.
    sequences : GraphSnapshotSequence or sequence of GraphSnapshotSequence
        One or more training trajectories.
    optimizer : Optimizer
        PyTorch optimizer used for the parameter update.
    loss_weights : :class:`~koopman_graph.training.LossWeights`
        Weights for reconstruction and consistency terms this epoch.
    max_grad_norm : float or None, optional
        When set, clip the global gradient norm to this value before
        ``optimizer.step()``.
    rollout_horizon : int or None, optional
        Number of rollout steps when ``loss_weights.rollout`` is non-zero.
    rollout_start_indices : sequence of int or None, optional
        Rollout origin indices for this epoch.

    Returns
    -------
    TrainingLossBreakdown
        Mean loss breakdown across trajectories.
    """
    if isinstance(sequences, GraphSnapshotSequence):
        trajectory_list = [sequences]
    else:
        trajectory_list = list(sequences)

    model.train()
    optimizer.zero_grad()
    breakdowns = [
        compute_training_loss(
            model,
            sequence,
            loss_weights,
            rollout_horizon=rollout_horizon,
            rollout_start_indices=rollout_start_indices,
        )
        for sequence in trajectory_list
    ]
    breakdown = mean_training_loss_breakdown(breakdowns)
    breakdown.total.backward()
    if max_grad_norm is not None:
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    return breakdown


def train_windowed_epoch(
    model: TrainableKoopmanModel,
    sampler: WindowSampler,
    optimizer: Optimizer,
    loss_weights: LossWeights,
    *,
    epoch: int = 0,
    max_grad_norm: float | None = None,
    rollout_horizon: int | None = None,
    rollout_start_indices: RolloutStartIndices = None,
    rollout_starts_per_epoch: int | None = None,
    rollout_start_seed: int | None = None,
) -> TrainingLossBreakdown:
    """Train on mini-batches of fixed-length temporal windows.

    Each batch averages its window losses before one optimizer step. The
    returned breakdown is weighted by the number of windows in each batch, so
    a smaller final batch does not receive disproportionate weight.

    Parameters
    ----------
    model : TrainableKoopmanModel
        Model satisfying :class:`~koopman_graph.protocols.TrainableKoopmanModel`.
    sampler : WindowSampler
        Window sampler defining trajectories, window size, and batch schedule.
    optimizer : Optimizer
        Optimizer updated once per yielded batch.
    loss_weights : LossWeights
        Active loss weights for the epoch.
    epoch : int, optional
        Zero-based epoch index used for sampler shuffling. Default is ``0``.
    max_grad_norm : float or None, optional
        Optional global gradient clipping threshold.
    rollout_horizon : int or None, optional
        Rollout horizon. Defaults to ``window_length - 1``.
    rollout_start_indices : sequence of int, ``"all"``, or None, optional
        Rollout origins relative to each sampled window.
    rollout_starts_per_epoch : int or None, optional
        Number of randomly sampled rollout origins.
    rollout_start_seed : int or None, optional
        Base seed for rollout-origin sampling.

    Returns
    -------
    TrainingLossBreakdown
        Window-weighted mean loss breakdown for the epoch.
    """
    horizon = sampler.window_length - 1 if rollout_horizon is None else rollout_horizon
    reference_window = sampler.sequences[0].slice(0, sampler.window_length)
    starts = None
    if loss_weights.rollout != 0.0:
        starts = resolve_rollout_start_indices(
            reference_window,
            horizon=horizon,
            rollout_start_indices=rollout_start_indices,
            rollout_starts_per_epoch=rollout_starts_per_epoch,
            rollout_start_seed=rollout_start_seed,
            epoch=epoch,
        )

    model.train()
    weighted_terms: dict[str, Tensor] | None = None
    window_count = 0
    for batch in sampler.iter_epoch(epoch):
        optimizer.zero_grad()
        batch_breakdown = mean_training_loss_breakdown(
            [
                compute_training_loss(
                    model,
                    window,
                    loss_weights,
                    rollout_horizon=rollout_horizon,
                    rollout_start_indices=starts,
                )
                for window in batch
            ]
        )
        batch_breakdown.total.backward()
        if max_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        batch_size = len(batch)
        detached = {
            name: getattr(batch_breakdown, name).detach() * batch_size
            for name in (
                "reconstruction",
                "forward",
                "backward",
                "rollout",
                "eigenvalue",
                "total",
            )
        }
        if weighted_terms is None:
            weighted_terms = detached
        else:
            for name, value in detached.items():
                weighted_terms[name] = weighted_terms[name] + value
        window_count += batch_size

    assert weighted_terms is not None
    return TrainingLossBreakdown(
        **{name: value / window_count for name, value in weighted_terms.items()}
    )


def eval_one_epoch(
    model: TrainableKoopmanModel,
    sequences: GraphSnapshotSequence | Sequence[GraphSnapshotSequence],
    loss_weights: LossWeights,
    *,
    rollout_horizon: int | None = None,
    rollout_start_indices: Sequence[int] | None = None,
) -> TrainingLossBreakdown:
    """Compute validation loss for one epoch without parameter updates.

    Parameters
    ----------
    model : TrainableKoopmanModel
        Model to evaluate.
    sequences : GraphSnapshotSequence or sequence of GraphSnapshotSequence
        One or more validation trajectories.
    loss_weights : LossWeights
        Weights for reconstruction and consistency terms.
    rollout_horizon : int or None, optional
        Number of rollout steps when ``loss_weights.rollout`` is non-zero.
    rollout_start_indices : sequence of int or None, optional
        Rollout origin indices for this epoch.

    Returns
    -------
    TrainingLossBreakdown
        Mean loss breakdown across trajectories.
    """
    if isinstance(sequences, GraphSnapshotSequence):
        trajectory_list = [sequences]
    else:
        trajectory_list = list(sequences)

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            breakdowns = [
                compute_training_loss(
                    model,
                    sequence,
                    loss_weights,
                    rollout_horizon=rollout_horizon,
                    rollout_start_indices=rollout_start_indices,
                )
                for sequence in trajectory_list
            ]
    finally:
        model.train(was_training)
    return mean_training_loss_breakdown(breakdowns)


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


def _classify_trajectory_items(
    items: Sequence[object],
    *,
    empty_message: str,
) -> list[GraphSnapshotSequence]:
    """Classify a non-wrapper sequence as multi- or single-trajectory input.

    Parameters
    ----------
    items : sequence
        Elements from a bare list/tuple passed to ``fit``.
    empty_message : str
        Error message when ``items`` is empty.

    Returns
    -------
    list of GraphSnapshotSequence
        One or more validated trajectories.

    Raises
    ------
    ValueError
        If ``items`` is empty or mixes ``GraphSnapshotSequence`` with ``Data``.
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
        return [resolve_sequence(item) for item in items]
    if len(data_indices) == len(items):
        return [resolve_sequence(items)]  # type: ignore[arg-type]
    if sequence_indices and data_indices:
        msg = (
            "cannot mix GraphSnapshotSequence and Data in the same fit input; "
            "use MultiTrajectory([...]) or a list of GraphSnapshotSequence for "
            "multiple trajectories, or a list of Data for one trajectory"
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


def is_sequence_of_sequences(
    data: TrainingInput | ValidationInput,
) -> bool:
    """Return whether ``data`` is multi-trajectory training input.

    Prefer :class:`~koopman_graph.data.MultiTrajectory` for new call sites.
    This helper remains for compatibility and inspects *all* elements of a bare
    sequence (not only the first) so mixed lists are not misclassified.

    Parameters
    ----------
    data : TrainingInput or ValidationInput
        Training or validation input passed to :meth:`fit`.

    Returns
    -------
    bool
        ``True`` for :class:`~koopman_graph.data.MultiTrajectory` or a non-empty
        sequence whose every element is a :class:`GraphSnapshotSequence`.
    """
    if data is None or isinstance(data, GraphSnapshotSequence):
        return False
    if isinstance(data, MultiTrajectory):
        return True
    if not isinstance(data, Sequence) or isinstance(data, (Data, str, bytes)):
        return False
    if len(data) == 0:
        return False
    return all(isinstance(item, GraphSnapshotSequence) for item in data)


def resolve_training_sequences(
    data_sequence: TrainingInput,
) -> list[GraphSnapshotSequence]:
    """Normalize training input into one or more snapshot sequences.

    Preferred multi-trajectory form is
    :class:`~koopman_graph.data.MultiTrajectory`. A bare list of
    :class:`~koopman_graph.data.GraphSnapshotSequence` remains accepted. A bare
    list of ``Data`` snapshots is always a single trajectory.

    Parameters
    ----------
    data_sequence : TrainingInput
        Single sequence, list of ``Data`` snapshots, ``MultiTrajectory``, or
        list of sequences.

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
        If a bare sequence contains unsupported element types.
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
            "a non-empty list of Data, MultiTrajectory(...), or a non-empty "
            "list of GraphSnapshotSequence"
        ),
    )


def resolve_validation_sequences(
    validation_sequence: ValidationInput,
    *,
    num_training_sequences: int,
) -> list[GraphSnapshotSequence] | None:
    """Normalize validation input for :meth:`fit`.

    A single validation sequence (or list of ``Data``) is reused for all
    training trajectories. A :class:`~koopman_graph.data.MultiTrajectory` or
    list of :class:`~koopman_graph.data.GraphSnapshotSequence` must match the
    training trajectory count.

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
        If a bare sequence contains unsupported element types.
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

    items = list(validation_sequence)
    sequences = _classify_trajectory_items(
        items,
        empty_message=(
            "validation_sequence must be non-empty when provided as a list; "
            "pass None, a GraphSnapshotSequence, a non-empty list of Data, "
            "MultiTrajectory(...), or a non-empty list of GraphSnapshotSequence"
        ),
    )
    # Multi-trajectory form (all GraphSnapshotSequence) must match train count;
    # a single-trajectory list of Data yields len == 1 and is reused.
    if is_sequence_of_sequences(items) and len(sequences) != num_training_sequences:
        msg = (
            "validation_sequence list length must match the number of "
            f"training trajectories ({num_training_sequences}), "
            f"got {len(sequences)}"
        )
        raise ValueError(msg)
    return sequences


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

        monitored_loss = term_values["total"]
        if val_sequences is not None:
            val_breakdown = eval_one_epoch(
                model,
                val_sequences,
                epoch_weights,
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
            val_losses.append(val_terms["total"])
            val_reconstruction_losses.append(val_terms["reconstruction"])
            val_forward_losses.append(val_terms["forward"])
            val_backward_losses.append(val_terms["backward"])
            val_rollout_losses.append(val_terms["rollout"])
            val_eigenvalue_losses.append(val_terms["eigenvalue"])
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
        stopped_early=stopped_early,
        best_epoch=best_epoch,
        best_loss=best_loss,
    )
