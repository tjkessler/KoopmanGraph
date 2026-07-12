"""Training utilities for GraphKoopmanModel."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch_geometric.data import Data

from koopman_graph.data import GraphSnapshotSequence, WindowSampler
from koopman_graph.losses import (
    BackwardConsistencyLoss,
    EigenvalueRegularizationLoss,
    ForwardConsistencyLoss,
    rollout_multi_start_loss,
    rollout_sequence_loss,
)

_FORWARD_CONSISTENCY_LOSS = ForwardConsistencyLoss()
_BACKWARD_CONSISTENCY_LOSS = BackwardConsistencyLoss()
_EIGENVALUE_REGULARIZATION_LOSS = EigenvalueRegularizationLoss()

PairLossFn = Callable[[nn.Module, GraphSnapshotSequence, int], Tensor]


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
    eigenvalue : float
        Weight on the unit-circle eigenvalue hinge penalty.
    """

    reconstruction: float = 1.0
    forward: float = 0.0
    backward: float = 0.0
    rollout: float = 0.0
    eigenvalue: float = 0.0


LossWeightSchedule = Callable[[int], LossWeights]
EarlyStoppingMonitor = Literal["auto", "train", "val"]
RolloutStartIndices = Sequence[int] | Literal["all"] | None
LRSchedulerFactory = Callable[[Optimizer], LRScheduler]
TrainingInput = GraphSnapshotSequence | Sequence[Data] | Sequence[GraphSnapshotSequence]
ValidationInput = (
    GraphSnapshotSequence | Sequence[Data] | Sequence[GraphSnapshotSequence] | None
)


@dataclass
class TrainingLossBreakdown:
    """Unweighted and weighted training loss terms for one batch or epoch.

    Attributes
    ----------
    reconstruction : Tensor
        Mean one-step reconstruction loss.
    forward : Tensor
        Mean forward consistency loss.
    backward : Tensor
        Mean backward consistency loss.
    rollout : Tensor
        Mean rollout reconstruction loss.
    eigenvalue : Tensor
        Eigenvalue hinge regularization loss.
    total : Tensor
        Weighted sum of all active loss terms.
    """

    reconstruction: Tensor
    forward: Tensor
    backward: Tensor
    rollout: Tensor
    eigenvalue: Tensor
    total: Tensor

    @classmethod
    def zeros(cls, device: torch.device) -> TrainingLossBreakdown:
        """Return a zero breakdown on ``device``.

        Parameters
        ----------
        device : torch.device
            Device for the zero tensors.

        Returns
        -------
        TrainingLossBreakdown
            Breakdown with all scalar terms set to zero.
        """
        zero = torch.zeros((), device=device)
        return cls(zero, zero, zero, zero, zero, zero)

    def to_floats(self) -> dict[str, float]:
        """Convert tensor terms to detached Python floats.

        Returns
        -------
        dict of str to float
            Mapping with keys ``reconstruction``, ``forward``, ``backward``,
            ``rollout``, ``eigenvalue``, and ``total``.
        """
        return {
            "reconstruction": float(self.reconstruction.detach().cpu()),
            "forward": float(self.forward.detach().cpu()),
            "backward": float(self.backward.detach().cpu()),
            "rollout": float(self.rollout.detach().cpu()),
            "eigenvalue": float(self.eigenvalue.detach().cpu()),
            "total": float(self.total.detach().cpu()),
        }


def mean_training_loss_breakdown(
    breakdowns: Sequence[TrainingLossBreakdown],
) -> TrainingLossBreakdown:
    """Average loss breakdowns across multiple trajectories.

    Parameters
    ----------
    breakdowns : sequence of TrainingLossBreakdown
        Per-trajectory breakdowns to average.

    Returns
    -------
    TrainingLossBreakdown
        Element-wise mean across ``breakdowns``.

    Raises
    ------
    ValueError
        If ``breakdowns`` is empty.
    """
    if not breakdowns:
        msg = "breakdowns must contain at least one entry"
        raise ValueError(msg)
    count = len(breakdowns)
    return TrainingLossBreakdown(
        reconstruction=sum(b.reconstruction for b in breakdowns) / count,
        forward=sum(b.forward for b in breakdowns) / count,
        backward=sum(b.backward for b in breakdowns) / count,
        rollout=sum(b.rollout for b in breakdowns) / count,
        eigenvalue=sum(b.eigenvalue for b in breakdowns) / count,
        total=sum(b.total for b in breakdowns) / count,
    )


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
    reconstruction_loss : list of float
        Per-epoch unweighted reconstruction loss.
    forward_loss : list of float
        Per-epoch unweighted forward consistency loss.
    backward_loss : list of float
        Per-epoch unweighted backward consistency loss.
    rollout_loss : list of float
        Per-epoch unweighted rollout reconstruction loss.
    eigenvalue_loss : list of float
        Per-epoch unweighted eigenvalue regularization loss.
    val_loss : list of float or None
        Per-epoch validation loss when a validation sequence is provided.
    val_reconstruction_loss : list of float or None
        Per-epoch unweighted validation reconstruction loss.
    val_forward_loss : list of float or None
        Per-epoch unweighted validation forward consistency loss.
    val_backward_loss : list of float or None
        Per-epoch unweighted validation backward consistency loss.
    val_rollout_loss : list of float or None
        Per-epoch unweighted validation rollout loss.
    val_eigenvalue_loss : list of float or None
        Per-epoch unweighted validation eigenvalue loss.
    stopped_early : bool
        Whether training stopped before the requested epoch count.
    best_epoch : int or None
        Zero-based index of the lowest-loss epoch when best-epoch tracking is
        enabled in :meth:`~koopman_graph.model.GraphKoopmanModel.fit`.
    best_loss : float or None
        Lowest monitored loss observed when best-epoch tracking is enabled.
    """

    loss: list[float]
    epochs: int
    reconstruction_loss: list[float] = field(default_factory=list)
    forward_loss: list[float] = field(default_factory=list)
    backward_loss: list[float] = field(default_factory=list)
    rollout_loss: list[float] = field(default_factory=list)
    eigenvalue_loss: list[float] = field(default_factory=list)
    val_loss: list[float] | None = None
    val_reconstruction_loss: list[float] | None = None
    val_forward_loss: list[float] | None = None
    val_backward_loss: list[float] | None = None
    val_rollout_loss: list[float] | None = None
    val_eigenvalue_loss: list[float] | None = None
    stopped_early: bool = False
    best_epoch: int | None = None
    best_loss: float | None = None


def constant_loss_weights(
    *,
    reconstruction: float = 1.0,
    forward: float = 0.0,
    backward: float = 0.0,
    rollout: float = 0.0,
    eigenvalue: float = 0.0,
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
    eigenvalue : float, optional
        Weight on the eigenvalue hinge penalty. Default is ``0.0``.

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
        eigenvalue=eigenvalue,
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
            eigenvalue=start.eigenvalue + t * (end.eigenvalue - start.eigenvalue),
        )

    return schedule


def one_step_loss(
    model: nn.Module,
    snapshot_t: Data,
    snapshot_t1: Data,
    *,
    control: Tensor | None = None,
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
    control : Tensor or None, optional
        Control input driving the transition from ``t`` to ``t+1``.

    Returns
    -------
    Tensor
        Scalar mean-squared error loss.
    """
    prediction = model(snapshot_t, control=control)
    target = snapshot_t1.x
    return nn.functional.mse_loss(prediction, target)


def _pair_control(sequence: GraphSnapshotSequence, timestep: int) -> Tensor | None:
    """Return the control input for transition ``timestep -> timestep + 1``.

    Parameters
    ----------
    sequence : GraphSnapshotSequence
        Snapshot sequence that may carry controls.
    timestep : int
        Index of the source snapshot in the transition pair.

    Returns
    -------
    Tensor or None
        Control tensor when present, otherwise ``None``.
    """
    if not sequence.has_controls:
        return None
    return sequence.control_at(timestep)


def _forward_consistency_pair(
    model: nn.Module,
    sequence: GraphSnapshotSequence,
    timestep: int,
) -> Tensor:
    """Compute forward consistency loss for one consecutive snapshot pair.

    Parameters
    ----------
    model : nn.Module
        Model with ``encoder`` and ``koopman`` attributes.
    sequence : GraphSnapshotSequence
        Snapshot sequence containing the consecutive pair.
    timestep : int
        Index of the source snapshot ``t`` in the pair ``(t, t+1)``.

    Returns
    -------
    Tensor
        Scalar forward consistency loss for the pair.
    """
    snapshot_t = sequence[timestep]
    snapshot_t1 = sequence[timestep + 1]
    edge_index_t = snapshot_t.edge_index
    edge_weight_t = getattr(snapshot_t, "edge_weight", None)
    edge_index_t1 = snapshot_t1.edge_index
    edge_weight_t1 = getattr(snapshot_t1, "edge_weight", None)
    z_t = model.encoder(snapshot_t, edge_index_t, edge_weight_t)
    z_t1 = model.encoder(snapshot_t1, edge_index_t1, edge_weight_t1)
    return _FORWARD_CONSISTENCY_LOSS(
        z_t,
        z_t1,
        model.koopman,
        control=_pair_control(sequence, timestep),
    )


def _backward_consistency_pair(
    model: nn.Module,
    sequence: GraphSnapshotSequence,
    timestep: int,
    *,
    inverse_matrix: Tensor | None = None,
) -> Tensor:
    """Compute backward consistency loss for one consecutive snapshot pair.

    Parameters
    ----------
    model : nn.Module
        Model with ``encoder`` and ``koopman`` attributes.
    sequence : GraphSnapshotSequence
        Snapshot sequence containing the consecutive pair.
    timestep : int
        Index of the source snapshot ``t`` in the pair ``(t, t+1)``.
    inverse_matrix : Tensor or None, optional
        Precomputed dense inverse matrix reused across pair evaluations.

    Returns
    -------
    Tensor
        Scalar backward consistency loss for the pair.
    """
    snapshot_t = sequence[timestep]
    snapshot_t1 = sequence[timestep + 1]
    edge_index_t = snapshot_t.edge_index
    edge_weight_t = getattr(snapshot_t, "edge_weight", None)
    edge_index_t1 = snapshot_t1.edge_index
    edge_weight_t1 = getattr(snapshot_t1, "edge_weight", None)
    z_t = model.encoder(snapshot_t, edge_index_t, edge_weight_t)
    z_t1 = model.encoder(snapshot_t1, edge_index_t1, edge_weight_t1)
    return _BACKWARD_CONSISTENCY_LOSS(
        z_t,
        z_t1,
        model.koopman,
        control=_pair_control(sequence, timestep),
        inverse_matrix=inverse_matrix,
    )


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
        total_loss = total_loss + pair_fn(model, sequence, t)
    return total_loss / num_pairs


def _one_step_pair(
    model: nn.Module,
    sequence: GraphSnapshotSequence,
    timestep: int,
) -> Tensor:
    """Compute one-step loss for snapshot pair ``(timestep, timestep + 1)``.

    Parameters
    ----------
    model : nn.Module
        Model implementing a single-step forward pass.
    sequence : GraphSnapshotSequence
        Snapshot sequence that may carry control inputs.
    timestep : int
        Index of the source snapshot in the transition pair.

    Returns
    -------
    Tensor
        Scalar one-step reconstruction loss.
    """
    return one_step_loss(
        model,
        sequence[timestep],
        sequence[timestep + 1],
        control=_pair_control(sequence, timestep),
    )


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
    return _mean_pair_sequence_loss(model, sequence, _one_step_pair)


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
    if sequence.num_timesteps < 2:
        msg = "GraphSnapshotSequence must contain at least 2 snapshots for training"
        raise ValueError(msg)

    inverse_matrix = None
    if model.koopman.parameterization == "dense":
        inverse_matrix = model.koopman.dense_inverse_matrix()

    total_loss = torch.zeros((), device=next(model.parameters()).device)
    num_pairs = sequence.num_timesteps - 1
    for t in range(num_pairs):
        total_loss = total_loss + _backward_consistency_pair(
            model,
            sequence,
            t,
            inverse_matrix=inverse_matrix,
        )
    return total_loss / num_pairs


def compute_eigenvalue_regularization_loss(model: nn.Module) -> Tensor:
    """Compute the eigenvalue hinge penalty for the model Koopman operator.

    Parameters
    ----------
    model : nn.Module
        Model with a ``koopman`` attribute.

    Returns
    -------
    Tensor
        Scalar eigenvalue regularization loss.
    """
    return _EIGENVALUE_REGULARIZATION_LOSS(model.koopman)


def resolve_rollout_start_indices(
    sequence: GraphSnapshotSequence,
    *,
    horizon: int,
    rollout_start_indices: RolloutStartIndices = None,
    rollout_starts_per_epoch: int | None = None,
    rollout_start_seed: int | None = None,
    epoch: int = 0,
) -> list[int]:
    """Resolve rollout-loss origin indices for one training epoch.

    Parameters
    ----------
    sequence : GraphSnapshotSequence
        Training snapshots.
    horizon : int
        Rollout horizon in steps.
    rollout_start_indices : sequence of int, ``"all"``, or None, optional
        Explicit origins, all valid origins, or ``None`` for the default
        ``[0]``.
    rollout_starts_per_epoch : int or None, optional
        When set, randomly sample this many valid origins each epoch.
        Overrides ``rollout_start_indices`` except when that argument is an
        explicit non-empty index list.
    rollout_start_seed : int or None, optional
        Base seed for random origin sampling. The effective seed is
        ``rollout_start_seed + epoch`` when provided.
    epoch : int, optional
        Zero-based epoch index mixed into the random seed. Default is ``0``.

    Returns
    -------
    list of int
        Valid zero-based rollout origin indices.

    Raises
    ------
    ValueError
        If ``horizon`` is invalid, no origins are available, or an origin is
        out of range.
    """
    if horizon < 1:
        msg = f"horizon must be >= 1, got {horizon}"
        raise ValueError(msg)
    if sequence.num_timesteps <= horizon:
        msg = (
            f"sequence has {sequence.num_timesteps} timesteps but needs more "
            f"than {horizon} for rollout loss"
        )
        raise ValueError(msg)

    upper = sequence.num_timesteps - horizon
    valid_origins = list(range(upper))

    if rollout_start_indices == "all":
        origins = valid_origins
    elif rollout_start_indices is not None:
        origins = []
        for start in rollout_start_indices:
            if start < 0 or start >= upper:
                msg = (
                    f"start index {start} is out of range for horizon "
                    f"{horizon}; valid range is [0, {upper - 1}]"
                )
                raise ValueError(msg)
            origins.append(start)
        if not origins:
            msg = "rollout_start_indices must contain at least one valid origin"
            raise ValueError(msg)
    else:
        origins = [0]

    if rollout_starts_per_epoch is not None:
        if rollout_starts_per_epoch < 1:
            msg = (
                f"rollout_starts_per_epoch must be >= 1 when set, "
                f"got {rollout_starts_per_epoch}"
            )
            raise ValueError(msg)
        generator: torch.Generator | None = None
        if rollout_start_seed is not None:
            generator = torch.Generator()
            generator.manual_seed(rollout_start_seed + epoch)
        sampled = torch.randint(
            low=0,
            high=len(valid_origins),
            size=(rollout_starts_per_epoch,),
            generator=generator,
        )
        origins = [valid_origins[int(index)] for index in sampled.tolist()]

    return origins


def compute_rollout_loss(
    model: nn.Module,
    sequence: GraphSnapshotSequence,
    *,
    horizon: int,
    start_indices: Sequence[int],
) -> Tensor:
    """Compute rollout reconstruction loss averaged over start origins.

    Parameters
    ----------
    model : nn.Module
        Model with encoder, Koopman operator, and decoder.
    sequence : GraphSnapshotSequence
        Training snapshots.
    horizon : int
        Number of rollout steps.
    start_indices : sequence of int
        Rollout origin indices.

    Returns
    -------
    Tensor
        Scalar mean rollout loss across origins.
    """
    if len(start_indices) == 1:
        return rollout_sequence_loss(
            model,
            sequence,
            horizon=horizon,
            start=start_indices[0],
        )
    return rollout_multi_start_loss(
        model,
        sequence,
        horizon=horizon,
        start_indices=start_indices,
    )


def compute_training_loss(
    model: nn.Module,
    sequence: GraphSnapshotSequence,
    loss_weights: LossWeights,
    *,
    rollout_horizon: int | None = None,
    rollout_start_indices: Sequence[int] | None = None,
) -> TrainingLossBreakdown:
    """Compute reconstruction, consistency, and rollout losses.

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
    rollout_start_indices : sequence of int or None, optional
        Rollout origin indices. Defaults to ``[0]`` when ``None``.

    Returns
    -------
    TrainingLossBreakdown
        Unweighted per-term losses and the weighted total.
    """
    device = next(model.parameters()).device
    reconstruction = compute_sequence_loss(model, sequence)
    forward = compute_forward_consistency_sequence_loss(model, sequence)
    backward = compute_backward_consistency_sequence_loss(model, sequence)
    eigenvalue = compute_eigenvalue_regularization_loss(model)

    if loss_weights.rollout != 0.0:
        horizon = (
            sequence.num_timesteps - 1 if rollout_horizon is None else rollout_horizon
        )
        starts = [0] if rollout_start_indices is None else list(rollout_start_indices)
        rollout = compute_rollout_loss(
            model,
            sequence,
            horizon=horizon,
            start_indices=starts,
        )
    else:
        rollout = torch.zeros((), device=device)

    total = (
        loss_weights.reconstruction * reconstruction
        + loss_weights.forward * forward
        + loss_weights.backward * backward
        + loss_weights.rollout * rollout
        + loss_weights.eigenvalue * eigenvalue
    )
    return TrainingLossBreakdown(
        reconstruction=reconstruction,
        forward=forward,
        backward=backward,
        rollout=rollout,
        eigenvalue=eigenvalue,
        total=total,
    )


def train_one_epoch(
    model: nn.Module,
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
    model : nn.Module
        Model to train.
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
    model: nn.Module,
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
    model : nn.Module
        Model to train.
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
    model: nn.Module,
    sequences: GraphSnapshotSequence | Sequence[GraphSnapshotSequence],
    loss_weights: LossWeights,
    *,
    rollout_horizon: int | None = None,
    rollout_start_indices: Sequence[int] | None = None,
) -> TrainingLossBreakdown:
    """Compute validation loss for one epoch without parameter updates.

    Parameters
    ----------
    model : nn.Module
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


def is_sequence_of_sequences(
    data: TrainingInput | ValidationInput,
) -> bool:
    """Return whether ``data`` is a list of snapshot sequences.

    Parameters
    ----------
    data : TrainingInput or ValidationInput
        Training or validation input passed to :meth:`fit`.

    Returns
    -------
    bool
        ``True`` when the first element is a :class:`GraphSnapshotSequence`.
    """
    if data is None or isinstance(data, GraphSnapshotSequence):
        return False
    if not isinstance(data, Sequence) or isinstance(data, Data):
        return False
    if len(data) == 0:
        return False
    return isinstance(data[0], GraphSnapshotSequence)


def resolve_training_sequences(
    data_sequence: TrainingInput,
) -> list[GraphSnapshotSequence]:
    """Normalize training input into one or more snapshot sequences.

    Parameters
    ----------
    data_sequence : TrainingInput
        Single sequence, list of ``Data`` snapshots, or list of sequences.

    Returns
    -------
    list of GraphSnapshotSequence
        One or more validated training trajectories.

    Raises
    ------
    ValueError
        If a multi-sequence input is empty.
    """
    if isinstance(data_sequence, GraphSnapshotSequence):
        return [data_sequence]
    if is_sequence_of_sequences(data_sequence):
        assert not isinstance(data_sequence, GraphSnapshotSequence)
        sequences = [resolve_sequence(sequence) for sequence in data_sequence]
        if not sequences:
            msg = "data_sequence must contain at least one trajectory"
            raise ValueError(msg)
        return sequences
    return [resolve_sequence(data_sequence)]


def resolve_validation_sequences(
    validation_sequence: ValidationInput,
    *,
    num_training_sequences: int,
) -> list[GraphSnapshotSequence] | None:
    """Normalize validation input for :meth:`fit`.

    A single validation sequence is reused for all training trajectories. A
    list of validation sequences must match the training trajectory count.

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
        If a validation list length does not match ``num_training_sequences``.
    """
    if validation_sequence is None:
        return None
    if is_sequence_of_sequences(validation_sequence):
        assert not isinstance(validation_sequence, GraphSnapshotSequence)
        sequences = [resolve_sequence(sequence) for sequence in validation_sequence]
        if len(sequences) != num_training_sequences:
            msg = (
                "validation_sequence list length must match the number of "
                f"training trajectories ({num_training_sequences}), "
                f"got {len(sequences)}"
            )
            raise ValueError(msg)
        return sequences
    return [resolve_sequence(validation_sequence)]


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
