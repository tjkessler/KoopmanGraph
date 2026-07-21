"""Pair and sequence reconstruction / consistency objective composition.

Owns one-step reconstruction and forward/backward consistency helpers used by
:func:`~koopman_graph.training.objectives.compute_training_loss`. Shared pair
utilities used by :mod:`~koopman_graph.training.extra_objectives` are documented
non-private names (no cross-module leading-underscore imports).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch_geometric.data import Data

from koopman_graph.data import (
    GraphSnapshotSequence,
    resolve_pair_delta_t,
)
from koopman_graph.graph_utils import propagate_latent
from koopman_graph.losses import (
    BackwardConsistencyLoss,
    ForwardConsistencyLoss,
    masked_mse_loss,
)
from koopman_graph.operators import GraphKoopmanOperator
from koopman_graph.protocols import TrainableKoopmanModel

_FORWARD_CONSISTENCY_LOSS = ForwardConsistencyLoss()
_BACKWARD_CONSISTENCY_LOSS = BackwardConsistencyLoss()

PairLossFn = Callable[[TrainableKoopmanModel, GraphSnapshotSequence, int], Tensor]


def model_default_delta_t(model: TrainableKoopmanModel) -> float:
    """Return the model-backed continuous default interval.

    Uses :meth:`~koopman_graph.protocols.TrainableKoopmanModel.resolve_delta_t`
    so training matches the model forward/env policy.

    Parameters
    ----------
    model : TrainableKoopmanModel
        Trainable model exposing ``resolve_delta_t``.

    Returns
    -------
    float
        Default continuous integration interval.
    """
    return float(model.resolve_delta_t(None))


def pair_control(sequence: GraphSnapshotSequence, timestep: int) -> Tensor | None:
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


def mean_pair_sequence_loss(
    model: TrainableKoopmanModel,
    sequence: GraphSnapshotSequence,
    pair_fn: PairLossFn,
) -> Tensor:
    """Average a pair-wise loss function over consecutive snapshots.

    Parameters
    ----------
    model : TrainableKoopmanModel
        Model passed through to ``pair_fn``.
    sequence : :class:`~koopman_graph.data.GraphSnapshotSequence`
        Time-ordered snapshots with at least two timesteps.
    pair_fn : callable
        Function mapping ``(model, sequence, timestep)`` to a scalar loss.

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


def _encode_at(
    model: TrainableKoopmanModel,
    sequence: GraphSnapshotSequence,
    index: int,
) -> Tensor:
    """Encode with delay history when the model exposes ``encode_at``.

    Parameters
    ----------
    model : TrainableKoopmanModel
        Trainable model; may implement ``encode_at(sequence, index)``.
    sequence : GraphSnapshotSequence
        Source trajectory.
    index : int
        Timestep to encode (window end).

    Returns
    -------
    Tensor
        Latent node features.
    """
    encode_at = getattr(model, "encode_at", None)
    if callable(encode_at):
        return encode_at(sequence, index)
    return model.encode(sequence[index])


def one_step_loss(
    model: TrainableKoopmanModel,
    snapshot_t: Data,
    snapshot_t1: Data,
    *,
    control: Tensor | None = None,
    delta_t: float | Tensor | None = None,
    target_mask: Tensor | None = None,
) -> Tensor:
    """Compute one-step MSE between model prediction and the next snapshot.

    Parameters
    ----------
    model : TrainableKoopmanModel
        Model satisfying :class:`~koopman_graph.protocols.TrainableKoopmanModel`.
    snapshot_t : Data
        Graph snapshot at time ``t``.
    snapshot_t1 : Data
        Graph snapshot at time ``t+1`` (prediction target).
    control : Tensor or None, optional
        Control input driving the transition from ``t`` to ``t+1``.
    delta_t : float, Tensor, or None, optional
        Integration interval for continuous-time models.
    target_mask : Tensor or None, optional
        Boolean node mask with shape ``(num_nodes,)``. When provided, the loss
        averages only over observed nodes at the target snapshot.

    Returns
    -------
    Tensor
        Scalar mean-squared error loss.
    """
    prediction = model(snapshot_t, control=control, delta_t=delta_t)
    target = snapshot_t1.x
    if target_mask is None:
        return nn.functional.mse_loss(prediction, target)
    return masked_mse_loss(prediction, target, target_mask)


def _forward_consistency_pair(
    model: TrainableKoopmanModel,
    sequence: GraphSnapshotSequence,
    timestep: int,
) -> Tensor:
    """Compute forward consistency loss for one consecutive snapshot pair.

    Parameters
    ----------
    model : TrainableKoopmanModel
        Model satisfying :class:`~koopman_graph.protocols.TrainableKoopmanModel`.
    sequence : GraphSnapshotSequence
        Snapshot sequence containing the consecutive pair.
    timestep : int
        Index of the source snapshot ``t`` in the pair ``(t, t+1)``.

    Returns
    -------
    Tensor
        Scalar forward consistency loss for the pair.
    """
    snapshot_t1 = sequence[timestep + 1]
    z_t = _encode_at(model, sequence, timestep)
    z_t1 = _encode_at(model, sequence, timestep + 1)
    default_delta_t = model_default_delta_t(model)
    delta_t = resolve_pair_delta_t(
        sequence,
        timestep,
        default_time_step=default_delta_t,
    )
    control = pair_control(sequence, timestep)
    pair_mask = (
        sequence.pair_observation_mask(timestep)
        if sequence.has_observation_masks
        else None
    )
    # Align with rollout decode policy: advance under the target snapshot topology.
    edge_index = snapshot_t1.edge_index
    edge_weight = getattr(snapshot_t1, "edge_weight", None)
    return _FORWARD_CONSISTENCY_LOSS(
        z_t,
        z_t1,
        model.koopman,
        control=control,
        delta_t=delta_t,
        default_delta_t=default_delta_t,
        mask=pair_mask,
        edge_index=edge_index,
        edge_weight=edge_weight,
    )


def _backward_consistency_pair(
    model: TrainableKoopmanModel,
    sequence: GraphSnapshotSequence,
    timestep: int,
    *,
    inverse_matrix: Tensor | None = None,
) -> Tensor:
    """Compute backward consistency loss for one consecutive snapshot pair.

    Parameters
    ----------
    model : TrainableKoopmanModel
        Model satisfying :class:`~koopman_graph.protocols.TrainableKoopmanModel`.
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
    snapshot_t1 = sequence[timestep + 1]
    z_t = _encode_at(model, sequence, timestep)
    z_t1 = _encode_at(model, sequence, timestep + 1)
    default_delta_t = model_default_delta_t(model)
    delta_t = resolve_pair_delta_t(
        sequence,
        timestep,
        default_time_step=default_delta_t,
    )
    control = pair_control(sequence, timestep)
    pair_mask = (
        sequence.pair_observation_mask(timestep)
        if sequence.has_observation_masks
        else None
    )
    edge_index = snapshot_t1.edge_index
    edge_weight = getattr(snapshot_t1, "edge_weight", None)
    return _BACKWARD_CONSISTENCY_LOSS(
        z_t,
        z_t1,
        model.koopman,
        control=control,
        inverse_matrix=inverse_matrix,
        delta_t=delta_t,
        default_delta_t=default_delta_t,
        mask=pair_mask,
        edge_index=edge_index,
        edge_weight=edge_weight,
    )


def _one_step_pair(
    model: TrainableKoopmanModel,
    sequence: GraphSnapshotSequence,
    timestep: int,
) -> Tensor:
    """Compute one-step loss for snapshot pair ``(timestep, timestep + 1)``.

    Parameters
    ----------
    model : TrainableKoopmanModel
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
    target_mask = None
    if sequence.has_observation_masks:
        target_mask = sequence.observation_mask_at(timestep + 1)

    n_delays = int(getattr(model, "n_delays", 1))
    if n_delays > 1 and callable(getattr(model, "encode_at", None)):
        snapshot_t = sequence[timestep]
        snapshot_t1 = sequence[timestep + 1]
        z = _encode_at(model, sequence, timestep)
        default_delta_t = model_default_delta_t(model)
        delta_t = resolve_pair_delta_t(
            sequence,
            timestep,
            default_time_step=default_delta_t,
        )
        z_next = propagate_latent(
            model.koopman,
            z,
            control=pair_control(sequence, timestep),
            delta_t=delta_t,
            default_delta_t=default_delta_t,
            edge_index=snapshot_t1.edge_index,
            edge_weight=getattr(snapshot_t1, "edge_weight", None),
        )
        prediction = model.decoder(
            z_next,
            snapshot_t.edge_index,
            getattr(snapshot_t, "edge_weight", None),
        )
        target = snapshot_t1.x
        if target_mask is None:
            return nn.functional.mse_loss(prediction, target)
        return masked_mse_loss(prediction, target, target_mask)

    return one_step_loss(
        model,
        sequence[timestep],
        sequence[timestep + 1],
        control=pair_control(sequence, timestep),
        delta_t=resolve_pair_delta_t(
            sequence,
            timestep,
            default_time_step=model_default_delta_t(model),
        ),
        target_mask=target_mask,
    )


def compute_sequence_loss(
    model: TrainableKoopmanModel,
    sequence: GraphSnapshotSequence,
) -> Tensor:
    """Average one-step prediction loss over consecutive snapshot pairs.

    Parameters
    ----------
    model : TrainableKoopmanModel
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
    return mean_pair_sequence_loss(model, sequence, _one_step_pair)


def compute_forward_consistency_sequence_loss(
    model: TrainableKoopmanModel,
    sequence: GraphSnapshotSequence,
) -> Tensor:
    """Average forward consistency loss over consecutive snapshot pairs.

    Parameters
    ----------
    model : TrainableKoopmanModel
        Model satisfying :class:`~koopman_graph.protocols.TrainableKoopmanModel`.
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
    return mean_pair_sequence_loss(
        model,
        sequence,
        _forward_consistency_pair,
    )


def compute_backward_consistency_sequence_loss(
    model: TrainableKoopmanModel,
    sequence: GraphSnapshotSequence,
) -> Tensor:
    """Average backward consistency loss over consecutive snapshot pairs.

    Parameters
    ----------
    model : TrainableKoopmanModel
        Model satisfying :class:`~koopman_graph.protocols.TrainableKoopmanModel`.
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

    # Optional built-in optimization: precompute ``K^{-1}`` once per sequence when
    # the operator exposes ``dense_inverse_matrix`` and does not need topology
    # (networked operators invert the effective ``N·d`` map per pair instead).
    inverse_matrix = None
    if (
        model.dynamics_mode == "discrete"
        and model.koopman.parameterization == "dense"
        and not isinstance(model.koopman, GraphKoopmanOperator)
    ):
        dense_inverse = getattr(model.koopman, "dense_inverse_matrix", None)
        if callable(dense_inverse):
            inverse_matrix = dense_inverse()

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
