"""Single-epoch training and evaluation helpers."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from koopman_graph.data import (
    GraphSnapshotSequence,
    HeteroGraphSnapshotSequence,
    RolloutStartIndices,
    SnapshotSequence,
    WindowLikeSampler,
    resolve_rollout_start_indices,
)
from koopman_graph.protocols import TrainableKoopmanModel
from koopman_graph.training.batched_objectives import compute_batched_training_loss
from koopman_graph.training.history import (
    ExtraLosses,
    LossWeights,
    TrainingLossBreakdown,
    mean_training_loss_breakdown,
)
from koopman_graph.training.objectives import compute_training_loss

_AMP_NON_CUDA_WARNED = False


def prepare_training_amp(
    use_amp: bool,
    device: torch.device,
    amp_dtype: torch.dtype | None = None,
    *,
    grad_scaler: torch.amp.GradScaler | None = None,
) -> tuple[bool, torch.dtype | None, torch.amp.GradScaler | None]:
    """Resolve whether CUDA AMP should run for this fit/epoch.

    When ``use_amp`` is requested on a non-CUDA device, emits a one-time
    warning and returns AMP disabled so training continues in FP32 (design
    Q3).

    Parameters
    ----------
    use_amp : bool
        Caller request for automatic mixed precision.
    device : torch.device
        Training device (model parameter device).
    amp_dtype : torch.dtype or None, optional
        Autocast dtype. Defaults to ``torch.float16`` on CUDA when AMP is
        enabled.
    grad_scaler : torch.amp.GradScaler or None, optional
        Existing scaler to reuse across epochs. Created when AMP enables and
        this argument is ``None``.

    Returns
    -------
    tuple of (bool, dtype or None, GradScaler or None)
        ``(amp_enabled, resolved_dtype, scaler)``.
    """
    global _AMP_NON_CUDA_WARNED
    if not use_amp:
        return False, None, None
    if device.type != "cuda":
        if not _AMP_NON_CUDA_WARNED:
            warnings.warn(
                "use_amp=True is only supported on CUDA; continuing in FP32 "
                f"on device type {device.type!r}.",
                UserWarning,
                stacklevel=2,
            )
            _AMP_NON_CUDA_WARNED = True
        return False, None, None
    resolved_dtype = torch.float16 if amp_dtype is None else amp_dtype
    scaler = (
        grad_scaler
        if grad_scaler is not None
        else torch.amp.GradScaler("cuda", enabled=True)
    )
    return True, resolved_dtype, scaler


def _backward_optimizer_step(
    model: TrainableKoopmanModel,
    optimizer: Optimizer,
    total: Tensor,
    *,
    max_grad_norm: float | None,
    amp_enabled: bool,
    grad_scaler: torch.amp.GradScaler | None,
) -> None:
    """Run backward, optional grad clip, and optimizer step (AMP-aware).

    Parameters
    ----------
    model : TrainableKoopmanModel
        Model whose parameters are clipped.
    optimizer : Optimizer
        Optimizer for the step.
    total : Tensor
        Scalar loss to differentiate.
    max_grad_norm : float or None
        Optional global gradient-norm clip.
    amp_enabled : bool
        Whether CUDA GradScaler should wrap the step.
    grad_scaler : torch.amp.GradScaler or None
        Scaler used when ``amp_enabled`` is ``True``.
    """
    if amp_enabled and grad_scaler is not None:
        grad_scaler.scale(total).backward()
        if max_grad_norm is not None:
            grad_scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        grad_scaler.step(optimizer)
        grad_scaler.update()
        return
    total.backward()
    if max_grad_norm is not None:
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()


def train_one_epoch(
    model: TrainableKoopmanModel,
    sequences: SnapshotSequence | Sequence[SnapshotSequence],
    optimizer: Optimizer,
    loss_weights: LossWeights,
    *,
    extra_losses: ExtraLosses | None = None,
    max_grad_norm: float | None = None,
    rollout_horizon: int | None = None,
    rollout_start_indices: Sequence[int] | None = None,
    use_amp: bool = False,
    amp_dtype: torch.dtype | None = None,
    grad_scaler: torch.amp.GradScaler | None = None,
    batch_graphs: bool = False,
) -> TrainingLossBreakdown:
    """Run one training epoch and return the averaged loss breakdown.

    Parameters
    ----------
    model : TrainableKoopmanModel
        Model satisfying :class:`~koopman_graph.protocols.TrainableKoopmanModel`.
    sequences : SnapshotSequence or sequence of SnapshotSequence
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
    extra_losses : ExtraLosses | None
        See the function signature / summary for ``extra_losses``.
    use_amp : bool, optional
        Enable CUDA autocast + GradScaler when the model is on CUDA.
        Non-CUDA devices warn once and stay in FP32. Default is ``False``.
    amp_dtype : torch.dtype or None, optional
        Autocast dtype (default ``float16`` on CUDA when AMP is active).
    grad_scaler : torch.amp.GradScaler or None, optional
        Reused scaler from :func:`~koopman_graph.training.loop.run_fit_loop`.
    batch_graphs : bool, optional
        When ``True``, collate homogeneous trajectories into one PyG
        ``Batch`` and evaluate reconstruction (and forward, when weighted)
        on the disconnected union. Default is ``False`` (per-sequence
        Python loop). Mutually exclusive with windowed sampling at the
        fit-loop layer.

    Returns
    -------
    TrainingLossBreakdown
        Mean loss breakdown across trajectories.
    """
    if isinstance(sequences, (GraphSnapshotSequence, HeteroGraphSnapshotSequence)):
        trajectory_list = [sequences]
    else:
        trajectory_list = list(sequences)

    device = next(model.parameters()).device
    amp_enabled, resolved_dtype, scaler = prepare_training_amp(
        use_amp,
        device,
        amp_dtype,
        grad_scaler=grad_scaler,
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)

    def _epoch_breakdown() -> TrainingLossBreakdown:
        """Return the epoch loss, batched or per-sequence.

        Returns
        -------
        TrainingLossBreakdown
            Mean breakdown for this optimizer step.
        """
        if batch_graphs:
            return compute_batched_training_loss(
                model,
                trajectory_list,
                loss_weights,
                extra_losses=extra_losses,
                rollout_horizon=rollout_horizon,
                rollout_start_indices=rollout_start_indices,
            )
        breakdowns = [
            compute_training_loss(
                model,
                sequence,
                loss_weights,
                extra_losses=extra_losses,
                rollout_horizon=rollout_horizon,
                rollout_start_indices=rollout_start_indices,
            )
            for sequence in trajectory_list
        ]
        return mean_training_loss_breakdown(breakdowns)

    if amp_enabled and resolved_dtype is not None:
        with torch.amp.autocast("cuda", dtype=resolved_dtype):
            breakdown = _epoch_breakdown()
    else:
        breakdown = _epoch_breakdown()
    _backward_optimizer_step(
        model,
        optimizer,
        breakdown.total,
        max_grad_norm=max_grad_norm,
        amp_enabled=amp_enabled,
        grad_scaler=scaler,
    )
    return breakdown


def train_windowed_epoch(
    model: TrainableKoopmanModel,
    sampler: WindowLikeSampler,
    optimizer: Optimizer,
    loss_weights: LossWeights,
    *,
    extra_losses: ExtraLosses | None = None,
    epoch: int = 0,
    max_grad_norm: float | None = None,
    rollout_horizon: int | None = None,
    rollout_start_indices: RolloutStartIndices = None,
    rollout_starts_per_epoch: int | None = None,
    rollout_start_seed: int | None = None,
    use_amp: bool = False,
    amp_dtype: torch.dtype | None = None,
    grad_scaler: torch.amp.GradScaler | None = None,
) -> TrainingLossBreakdown:
    """Train on mini-batches of fixed-length temporal windows.

    Each batch averages its window losses before one optimizer step. The
    returned breakdown is weighted by the number of windows in each batch, so
    a smaller final batch does not receive disproportionate weight.
    :class:`~koopman_graph.data.NeighborWindowSampler` windows carry induced
    subgraph topology; losses (including graph-operator eigenvalue
    regularization) are therefore subgraph approximations.

    Parameters
    ----------
    model : TrainableKoopmanModel
        Model satisfying :class:`~koopman_graph.protocols.TrainableKoopmanModel`.
    sampler : WindowSampler or NeighborWindowSampler
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
    extra_losses : ExtraLosses | None
        See the function signature / summary for ``extra_losses``.
    use_amp : bool, optional
        Enable CUDA autocast + GradScaler when the model is on CUDA.
        Non-CUDA devices warn once and stay in FP32. Default is ``False``.
    amp_dtype : torch.dtype or None, optional
        Autocast dtype (default ``float16`` on CUDA when AMP is active).
    grad_scaler : torch.amp.GradScaler or None, optional
        Reused scaler from :func:`~koopman_graph.training.loop.run_fit_loop`.

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

    device = next(model.parameters()).device
    amp_enabled, resolved_dtype, scaler = prepare_training_amp(
        use_amp,
        device,
        amp_dtype,
        grad_scaler=grad_scaler,
    )

    model.train()
    weighted_terms: dict[str, Tensor] | None = None
    window_count = 0
    for batch in sampler.iter_epoch(epoch):
        optimizer.zero_grad(set_to_none=True)
        if amp_enabled and resolved_dtype is not None:
            with torch.amp.autocast("cuda", dtype=resolved_dtype):
                batch_breakdown = mean_training_loss_breakdown(
                    [
                        compute_training_loss(
                            model,
                            window,
                            loss_weights,
                            extra_losses=extra_losses,
                            rollout_horizon=rollout_horizon,
                            rollout_start_indices=starts,
                        )
                        for window in batch
                    ]
                )
        else:
            batch_breakdown = mean_training_loss_breakdown(
                [
                    compute_training_loss(
                        model,
                        window,
                        loss_weights,
                        extra_losses=extra_losses,
                        rollout_horizon=rollout_horizon,
                        rollout_start_indices=starts,
                    )
                    for window in batch
                ]
            )
        _backward_optimizer_step(
            model,
            optimizer,
            batch_breakdown.total,
            max_grad_norm=max_grad_norm,
            amp_enabled=amp_enabled,
            grad_scaler=scaler,
        )

        batch_size = len(batch)
        detached = {
            name: getattr(batch_breakdown, name).detach() * batch_size
            for name in (
                "reconstruction",
                "forward",
                "backward",
                "rollout",
                "eigenvalue",
                "lie",
                "pde",
                "sparsity",
                "worst_case",
                "vamp2",
                "topology",
                "presence",
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
    sequences: SnapshotSequence | Sequence[SnapshotSequence],
    loss_weights: LossWeights,
    *,
    extra_losses: ExtraLosses | None = None,
    rollout_horizon: int | None = None,
    rollout_start_indices: Sequence[int] | None = None,
    batch_graphs: bool = False,
) -> TrainingLossBreakdown:
    """Compute validation loss for one epoch without parameter updates.

    Parameters
    ----------
    model : TrainableKoopmanModel
        Model to evaluate.
    sequences : SnapshotSequence or sequence of SnapshotSequence
        One or more validation trajectories.
    loss_weights : LossWeights
        Weights for reconstruction and consistency terms.
    rollout_horizon : int or None, optional
        Number of rollout steps when ``loss_weights.rollout`` is non-zero.
    rollout_start_indices : sequence of int or None, optional
        Rollout origin indices for this epoch.
    extra_losses : ExtraLosses | None
        See the function signature / summary for ``extra_losses``.
    batch_graphs : bool, optional
        When ``True``, use the collated multi-graph loss path. Default is
        ``False``.

    Returns
    -------
    TrainingLossBreakdown
        Mean loss breakdown across trajectories.
    """
    if isinstance(sequences, (GraphSnapshotSequence, HeteroGraphSnapshotSequence)):
        trajectory_list = [sequences]
    else:
        trajectory_list = list(sequences)

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            if batch_graphs:
                return compute_batched_training_loss(
                    model,
                    trajectory_list,
                    loss_weights,
                    extra_losses=extra_losses,
                    rollout_horizon=rollout_horizon,
                    rollout_start_indices=rollout_start_indices,
                )
            breakdowns = [
                compute_training_loss(
                    model,
                    sequence,
                    loss_weights,
                    extra_losses=extra_losses,
                    rollout_horizon=rollout_horizon,
                    rollout_start_indices=rollout_start_indices,
                )
                for sequence in trajectory_list
            ]
    finally:
        model.train(was_training)
    return mean_training_loss_breakdown(breakdowns)
