"""Training utilities for GraphKoopmanModel."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch_geometric.data import Data

from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.losses import (
    BackwardConsistencyLoss,
    ForwardConsistencyLoss,
    rollout_sequence_loss,
)

_FORWARD_CONSISTENCY_LOSS = ForwardConsistencyLoss()
_BACKWARD_CONSISTENCY_LOSS = BackwardConsistencyLoss()

PairLossFn = Callable[[nn.Module, Data, Data], Tensor]


@dataclass(frozen=True)
class LossWeights:
    """Weights for reconstruction and consistency loss terms.

    Attributes
    ----------
    reconstruction : float
        Weight on the one-step reconstruction (MSE) loss.
    forward : float
        Weight on the forward consistency loss.
    backward : float
        Weight on the backward consistency loss.
    rollout : float
        Weight on the autoregressive rollout reconstruction loss.
    """

    reconstruction: float = 1.0
    forward: float = 0.0
    backward: float = 0.0
    rollout: float = 0.0


LossWeightSchedule = Callable[[int], LossWeights]


@dataclass
class FitHistory:
    """Training history returned by :meth:`GraphKoopmanModel.fit`.

    Attributes
    ----------
    loss : list of float
        Per-epoch average combined training loss.
    epochs : int
        Number of training epochs completed (may be less than requested when
        early stopping triggers).
    stopped_early : bool
        Whether training stopped before the requested epoch count.
    """

    loss: list[float]
    epochs: int
    stopped_early: bool = False


def constant_loss_weights(
    *,
    reconstruction: float = 1.0,
    forward: float = 0.0,
    backward: float = 0.0,
    rollout: float = 0.0,
) -> LossWeights:
    """Build static loss weights for all training epochs.

    Parameters
    ----------
    reconstruction : float, optional
        Weight on the reconstruction loss. Default is ``1.0``.
    forward : float, optional
        Weight on the forward consistency loss. Default is ``0.0``.
    backward : float, optional
        Weight on the backward consistency loss. Default is ``0.0``.
    rollout : float, optional
        Weight on the rollout reconstruction loss. Default is ``0.0``.

    Returns
    -------
    :class:`~koopman_graph.training.LossWeights`
        Fixed weights applied uniformly across epochs.
    """
    return LossWeights(
        reconstruction=reconstruction,
        forward=forward,
        backward=backward,
        rollout=rollout,
    )


def linear_ramp_loss_weights(
    start: LossWeights,
    end: LossWeights,
    ramp_epochs: int,
) -> LossWeightSchedule:
    """Linearly interpolate loss weights over the first ``ramp_epochs``.

    Parameters
    ----------
    start : :class:`~koopman_graph.training.LossWeights`
        Weights at epoch ``0``.
    end : :class:`~koopman_graph.training.LossWeights`
        Weights reached at epoch ``ramp_epochs - 1`` and held thereafter.
    ramp_epochs : int
        Number of epochs over which to interpolate. Must be >= 1.

    Returns
    -------
    callable
        Schedule mapping epoch index to :class:`~koopman_graph.training.LossWeights`.

    Raises
    ------
    ValueError
        If ``ramp_epochs < 1``.
    """
    if ramp_epochs < 1:
        msg = f"ramp_epochs must be >= 1, got {ramp_epochs}"
        raise ValueError(msg)

    def schedule(epoch: int) -> LossWeights:
        """Return interpolated loss weights for a training epoch.

        Parameters
        ----------
        epoch : int
            Zero-based epoch index.

        Returns
        -------
        :class:`~koopman_graph.training.LossWeights`
            Interpolated weights for the given epoch.
        """
        if ramp_epochs == 1 or epoch >= ramp_epochs - 1:
            return end
        t = epoch / (ramp_epochs - 1)
        return LossWeights(
            reconstruction=start.reconstruction
            + t * (end.reconstruction - start.reconstruction),
            forward=start.forward + t * (end.forward - start.forward),
            backward=start.backward + t * (end.backward - start.backward),
            rollout=start.rollout + t * (end.rollout - start.rollout),
        )

    return schedule


def one_step_loss(
    model: nn.Module,
    snapshot_t: Data,
    snapshot_t1: Data,
) -> Tensor:
    """Compute one-step MSE between model prediction and the next snapshot.

    Parameters
    ----------
    model : nn.Module
        Model implementing a single-step forward pass (e.g.
        :class:`~koopman_graph.model.GraphKoopmanModel`).
    snapshot_t : Data
        Graph snapshot at time ``t``.
    snapshot_t1 : Data
        Graph snapshot at time ``t+1`` (prediction target).

    Returns
    -------
    Tensor
        Scalar mean-squared error loss.
    """
    prediction = model(snapshot_t)
    target = snapshot_t1.x
    return nn.functional.mse_loss(prediction, target)


def _forward_consistency_pair(
    model: nn.Module,
    snapshot_t: Data,
    snapshot_t1: Data,
) -> Tensor:
    """Compute forward consistency loss for one consecutive snapshot pair.

    Parameters
    ----------
    model : nn.Module
        Model with ``encoder`` and ``koopman`` attributes.
    snapshot_t : Data
        Graph snapshot at time ``t``.
    snapshot_t1 : Data
        Graph snapshot at time ``t+1``.

    Returns
    -------
    Tensor
        Scalar forward consistency loss for the pair.
    """
    edge_index = snapshot_t.edge_index
    z_t = model.encoder(snapshot_t, edge_index)
    z_t1 = model.encoder(snapshot_t1, edge_index)
    return _FORWARD_CONSISTENCY_LOSS(z_t, z_t1, model.koopman)


def _backward_consistency_pair(
    model: nn.Module,
    snapshot_t: Data,
    snapshot_t1: Data,
) -> Tensor:
    """Compute backward consistency loss for one consecutive snapshot pair.

    Parameters
    ----------
    model : nn.Module
        Model with ``encoder`` and ``koopman`` attributes.
    snapshot_t : Data
        Graph snapshot at time ``t``.
    snapshot_t1 : Data
        Graph snapshot at time ``t+1``.

    Returns
    -------
    Tensor
        Scalar backward consistency loss for the pair.
    """
    edge_index = snapshot_t.edge_index
    z_t = model.encoder(snapshot_t, edge_index)
    z_t1 = model.encoder(snapshot_t1, edge_index)
    return _BACKWARD_CONSISTENCY_LOSS(z_t, z_t1, model.koopman)


def _mean_pair_sequence_loss(
    model: nn.Module,
    sequence: GraphSnapshotSequence,
    pair_fn: PairLossFn,
) -> Tensor:
    """Average a pair-wise loss function over consecutive snapshots.

    Parameters
    ----------
    model : nn.Module
        Model passed through to ``pair_fn``.
    sequence : :class:`~koopman_graph.data.GraphSnapshotSequence`
        Time-ordered snapshots with at least two timesteps.
    pair_fn : callable
        Function mapping ``(model, snapshot_t, snapshot_t1)`` to a scalar loss.

    Returns
    -------
    Tensor
        Scalar average loss over all consecutive pairs.

    Raises
    ------
    ValueError
        If ``sequence`` contains fewer than two snapshots.
    """
    if sequence.num_timesteps < 2:
        msg = "GraphSnapshotSequence must contain at least 2 snapshots for training"
        raise ValueError(msg)

    total_loss = torch.zeros((), device=next(model.parameters()).device)
    num_pairs = sequence.num_timesteps - 1
    for t in range(num_pairs):
        total_loss = total_loss + pair_fn(model, sequence[t], sequence[t + 1])
    return total_loss / num_pairs


def compute_sequence_loss(
    model: nn.Module,
    sequence: GraphSnapshotSequence,
) -> Tensor:
    """Average one-step prediction loss over consecutive snapshot pairs.

    Parameters
    ----------
    model : nn.Module
        Model implementing a single-step forward pass.
    sequence : :class:`~koopman_graph.data.GraphSnapshotSequence`
        Time-ordered snapshots with at least two timesteps.

    Returns
    -------
    Tensor
        Scalar average loss over all consecutive pairs.

    Raises
    ------
    ValueError
        If ``sequence`` contains fewer than two snapshots.
    """
    return _mean_pair_sequence_loss(model, sequence, one_step_loss)


def compute_forward_consistency_sequence_loss(
    model: nn.Module,
    sequence: GraphSnapshotSequence,
) -> Tensor:
    """Average forward consistency loss over consecutive snapshot pairs.

    Parameters
    ----------
    model : nn.Module
        Model with ``encoder`` and ``koopman`` attributes.
    sequence : :class:`~koopman_graph.data.GraphSnapshotSequence`
        Time-ordered snapshots with at least two timesteps.

    Returns
    -------
    Tensor
        Scalar average forward consistency loss.

    Raises
    ------
    ValueError
        If ``sequence`` contains fewer than two snapshots.
    """
    return _mean_pair_sequence_loss(
        model,
        sequence,
        _forward_consistency_pair,
    )


def compute_backward_consistency_sequence_loss(
    model: nn.Module,
    sequence: GraphSnapshotSequence,
) -> Tensor:
    """Average backward consistency loss over consecutive snapshot pairs.

    Parameters
    ----------
    model : nn.Module
        Model with ``encoder`` and ``koopman`` attributes.
    sequence : :class:`~koopman_graph.data.GraphSnapshotSequence`
        Time-ordered snapshots with at least two timesteps.

    Returns
    -------
    Tensor
        Scalar average backward consistency loss.

    Raises
    ------
    ValueError
        If ``sequence`` contains fewer than two snapshots.
    """
    return _mean_pair_sequence_loss(
        model,
        sequence,
        _backward_consistency_pair,
    )


def compute_training_loss(
    model: nn.Module,
    sequence: GraphSnapshotSequence,
    loss_weights: LossWeights,
    *,
    rollout_horizon: int | None = None,
) -> Tensor:
    """Compute combined reconstruction and consistency losses.

    Parameters
    ----------
    model : nn.Module
        Model implementing a single-step forward pass with encoder and Koopman
        operator.
    sequence : :class:`~koopman_graph.data.GraphSnapshotSequence`
        Time-ordered snapshots with at least two timesteps.
    loss_weights : :class:`~koopman_graph.training.LossWeights`
        Weights for reconstruction, forward, backward, and rollout terms.
    rollout_horizon : int or None, optional
        Number of rollout steps when ``loss_weights.rollout`` is non-zero.
        Defaults to ``sequence.num_timesteps - 1``.
    Returns
    -------
    Tensor
        Scalar combined training loss.
    """
    total_loss = torch.zeros((), device=next(model.parameters()).device)
    if loss_weights.reconstruction != 0.0:
        total_loss = total_loss + loss_weights.reconstruction * compute_sequence_loss(
            model, sequence
        )
    if loss_weights.forward != 0.0:
        total_loss = total_loss + loss_weights.forward * (
            compute_forward_consistency_sequence_loss(model, sequence)
        )
    if loss_weights.backward != 0.0:
        total_loss = total_loss + loss_weights.backward * (
            compute_backward_consistency_sequence_loss(model, sequence)
        )
    if loss_weights.rollout != 0.0:
        horizon = (
            sequence.num_timesteps - 1 if rollout_horizon is None else rollout_horizon
        )
        total_loss = total_loss + loss_weights.rollout * rollout_sequence_loss(
            model,
            sequence,
            horizon=horizon,
        )
    return total_loss


def train_one_epoch(
    model: nn.Module,
    sequence: GraphSnapshotSequence,
    optimizer: Optimizer,
    loss_weights: LossWeights,
    *,
    max_grad_norm: float | None = None,
    rollout_horizon: int | None = None,
) -> float:
    """Run one training epoch and return the average loss.

    Parameters
    ----------
    model : nn.Module
        Model to train.
    sequence : :class:`~koopman_graph.data.GraphSnapshotSequence`
        Training snapshots with at least two timesteps.
    optimizer : Optimizer
        PyTorch optimizer used for the parameter update.
    loss_weights : :class:`~koopman_graph.training.LossWeights`
        Weights for reconstruction and consistency terms this epoch.
    max_grad_norm : float or None, optional
        When set, clip the global gradient norm to this value before
        ``optimizer.step()``.
    rollout_horizon : int or None, optional
        Number of rollout steps when ``loss_weights.rollout`` is non-zero.

    Returns
    -------
    float
        Detached scalar loss for the epoch.
    """
    model.train()
    optimizer.zero_grad()
    loss = compute_training_loss(
        model,
        sequence,
        loss_weights,
        rollout_horizon=rollout_horizon,
    )
    loss.backward()
    if max_grad_norm is not None:
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    return float(loss.detach().cpu())


def resolve_sequence(
    data_sequence: GraphSnapshotSequence | Sequence[Data],
) -> GraphSnapshotSequence:
    """Normalize input into a validated snapshot sequence.

    Wraps input in :class:`~koopman_graph.data.GraphSnapshotSequence` when
    needed.

    Parameters
    ----------
    data_sequence : :class:`~koopman_graph.data.GraphSnapshotSequence` or \
sequence of Data
        Raw snapshot input from a training or inference API.

    Returns
    -------
    :class:`~koopman_graph.data.GraphSnapshotSequence`
        Validated sequence container.
    """
    if isinstance(data_sequence, GraphSnapshotSequence):
        return data_sequence
    return GraphSnapshotSequence(data_sequence)


def resolve_device(
    model: nn.Module,
    device: str | torch.device | None,
) -> torch.device:
    """Resolve the training device from an explicit argument or model parameters.

    Parameters
    ----------
    model : nn.Module
        Model whose parameter device is used as fallback.
    device : str, torch.device, or None
        Explicit device. When ``None``, uses the device of the first model
        parameter, or CPU if the model has no parameters.

    Returns
    -------
    torch.device
        Resolved device for training or inference.
    """
    if device is not None:
        return torch.device(device)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def resolve_loss_weights_for_epoch(
    epoch: int,
    *,
    loss_weights: LossWeights | None,
    loss_weight_schedule: LossWeightSchedule | None,
) -> LossWeights:
    """Resolve per-epoch loss weights from static weights or a schedule.

    Parameters
    ----------
    epoch : int
        Zero-based epoch index.
    loss_weights : :class:`~koopman_graph.training.LossWeights` or None
        Explicit static weights. When ``None`` and no schedule is provided,
        defaults to reconstruction-only weights.
    loss_weight_schedule : callable or None
        Optional per-epoch schedule. Takes precedence over ``loss_weights``.

    Returns
    -------
    :class:`~koopman_graph.training.LossWeights`
        Weights to use for the given epoch.
    """
    if loss_weight_schedule is not None:
        return loss_weight_schedule(epoch)
    if loss_weights is not None:
        return loss_weights
    return constant_loss_weights()


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
