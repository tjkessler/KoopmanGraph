"""Single-epoch training and evaluation helpers."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from koopman_graph.data import (
    GraphSnapshotSequence,
    RolloutStartIndices,
    WindowSampler,
    resolve_rollout_start_indices,
)
from koopman_graph.protocols import TrainableKoopmanModel
from koopman_graph.training.history import (
    ExtraLosses,
    LossWeights,
    TrainingLossBreakdown,
    mean_training_loss_breakdown,
)
from koopman_graph.training.objectives import compute_training_loss


def train_one_epoch(
    model: TrainableKoopmanModel,
    sequences: GraphSnapshotSequence | Sequence[GraphSnapshotSequence],
    optimizer: Optimizer,
    loss_weights: LossWeights,
    *,
    extra_losses: ExtraLosses | None = None,
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
            extra_losses=extra_losses,
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
    extra_losses: ExtraLosses | None = None,
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
                    extra_losses=extra_losses,
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
                "lie",
                "pde",
                "sparsity",
                "worst_case",
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
    extra_losses: ExtraLosses | None = None,
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
                    extra_losses=extra_losses,
                    rollout_horizon=rollout_horizon,
                    rollout_start_indices=rollout_start_indices,
                )
                for sequence in trajectory_list
            ]
    finally:
        model.train(was_training)
    return mean_training_loss_breakdown(breakdowns)
