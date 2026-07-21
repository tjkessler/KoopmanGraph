"""Extra-loss objective composition over ``ExtraLosses`` / ``LossWeights``.

Owns Lie / PDE / sparsity / worst-case training-side composition helpers used by
:func:`~koopman_graph.training.objectives.compute_training_loss`. Shared pair
utilities come from :mod:`~koopman_graph.training.pair_objectives` as documented
non-private names.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch_geometric.data import Data

from koopman_graph.data import (
    GraphSnapshotSequence,
    resolve_pair_delta_t,
)
from koopman_graph.losses import (
    KoopmanSparsityLoss,
    LieConsistencyLoss,
    PDEResidualLoss,
    WorstCaseReconstructionLoss,
)
from koopman_graph.protocols import TrainableKoopmanModel
from koopman_graph.training.history import ExtraLosses
from koopman_graph.training.pair_objectives import (
    mean_pair_sequence_loss,
    model_default_delta_t,
    pair_control,
)

_LIE_CONSISTENCY_LOSS = LieConsistencyLoss()
_PDE_RESIDUAL_LOSS = PDEResidualLoss()
_KOOPMAN_SPARSITY_LOSS = KoopmanSparsityLoss()
_WORST_CASE_RECONSTRUCTION_LOSS = WorstCaseReconstructionLoss()


def _snapshot_with_x(snapshot: Data, x: Tensor) -> Data:
    """Clone a snapshot and replace its node features.

    Parameters
    ----------
    snapshot : Data
        Source graph metadata.
    x : Tensor
        Replacement node features.

    Returns
    -------
    Data
        Cloned snapshot whose ``x`` attribute is ``x``.
    """
    updated = snapshot.clone()
    updated.x = x
    return updated


def compute_lie_consistency_loss(
    model: TrainableKoopmanModel,
    sequence: GraphSnapshotSequence,
    *,
    weight: float,
    extra_losses: ExtraLosses | None,
) -> Tensor:
    """Average Lie consistency across snapshots when enabled.

    Parameters
    ----------
    model : TrainableKoopmanModel
        Continuous model supplying observables and latent generator.
    sequence : GraphSnapshotSequence
        Snapshots at which to evaluate the known vector field.
    weight : float
        Active Lie loss weight; zero skips evaluation.
    extra_losses : ExtraLosses or None
        Fit-time known-vector-field configuration.

    Returns
    -------
    Tensor
        Scalar mean Lie consistency loss.
    """
    device = next(model.parameters()).device
    if weight == 0.0:
        return torch.zeros((), device=device)
    if model.dynamics_mode != "continuous":
        msg = "Lie consistency requires model dynamics_mode='continuous'"
        raise ValueError(msg)
    if extra_losses is None or extra_losses.lie_dynamics_fn is None:
        msg = (
            "loss_weights.lie is non-zero but "
            "extra_losses.lie_dynamics_fn is not configured"
        )
        raise ValueError(msg)

    dynamics_from_snapshot = extra_losses.lie_dynamics_fn
    losses: list[Tensor] = []
    for snapshot in sequence:
        state = snapshot.x
        if state is None:
            msg = "Lie consistency requires snapshot.x"
            raise ValueError(msg)

        def observable_fn(x: Tensor, context: Data = snapshot) -> Tensor:
            return model.encode(_snapshot_with_x(context, x))

        def dynamics_fn(x: Tensor, context: Data = snapshot) -> Tensor:
            return dynamics_from_snapshot(_snapshot_with_x(context, x))

        losses.append(
            _LIE_CONSISTENCY_LOSS(
                state,
                observable_fn=observable_fn,
                dynamics_fn=dynamics_fn,
                koopman=model.koopman,
            )
        )
    return torch.stack(losses).mean()


def compute_pde_residual_loss(
    model: TrainableKoopmanModel,
    sequence: GraphSnapshotSequence,
    *,
    weight: float,
    extra_losses: ExtraLosses | None,
) -> Tensor:
    """Average decoded-field PDE residuals across one-step predictions.

    Parameters
    ----------
    model : TrainableKoopmanModel
        Model producing decoded one-step forecasts.
    sequence : GraphSnapshotSequence
        Source/target pairs and graph context.
    weight : float
        Active PDE loss weight; zero skips evaluation.
    extra_losses : ExtraLosses or None
        Fit-time PDE residual configuration.

    Returns
    -------
    Tensor
        Scalar mean PDE residual loss.
    """
    device = next(model.parameters()).device
    if weight == 0.0:
        return torch.zeros((), device=device)
    if extra_losses is None or extra_losses.pde_residual_fn is None:
        msg = (
            "loss_weights.pde is non-zero but "
            "extra_losses.pde_residual_fn is not configured"
        )
        raise ValueError(msg)

    residual_fn = extra_losses.pde_residual_fn
    losses: list[Tensor] = []
    default_delta_t = model_default_delta_t(model)
    for timestep in range(sequence.num_timesteps - 1):
        source = sequence[timestep]
        target = sequence[timestep + 1]
        delta_t = resolve_pair_delta_t(
            sequence,
            timestep,
            default_time_step=default_delta_t,
        )
        prediction = model(
            source,
            control=pair_control(sequence, timestep),
            delta_t=delta_t,
        )
        mask = (
            sequence.observation_mask_at(timestep + 1)
            if sequence.has_observation_masks
            else None
        )
        losses.append(
            _PDE_RESIDUAL_LOSS(
                prediction,
                target,
                pde_fn=residual_fn,
                mask=mask,
            )
        )
    return torch.stack(losses).mean()


def compute_sparsity_loss(
    model: TrainableKoopmanModel,
    *,
    weight: float,
) -> Tensor:
    """Compute the Koopman-matrix sparsity penalty when enabled.

    Parameters
    ----------
    model : TrainableKoopmanModel
        Model whose public Koopman factors are penalized.
    weight : float
        Active sparsity weight; zero skips evaluation.

    Returns
    -------
    Tensor
        Scalar sparsity penalty.
    """
    device = next(model.parameters()).device
    if weight == 0.0:
        return torch.zeros((), device=device)
    return _KOOPMAN_SPARSITY_LOSS(model.koopman)


def _worst_case_pair(
    model: TrainableKoopmanModel,
    sequence: GraphSnapshotSequence,
    timestep: int,
) -> Tensor:
    """Compute worst-case reconstruction loss for one consecutive pair.

    Parameters
    ----------
    model : TrainableKoopmanModel
        Model implementing a single-step forward pass.
    sequence : GraphSnapshotSequence
        Snapshot sequence containing the consecutive pair.
    timestep : int
        Index of the source snapshot in the transition pair.

    Returns
    -------
    Tensor
        Scalar max-over-nodes mean-squared error for the pair.
    """
    target_mask = (
        sequence.observation_mask_at(timestep + 1)
        if sequence.has_observation_masks
        else None
    )
    prediction = model(
        sequence[timestep],
        control=pair_control(sequence, timestep),
        delta_t=resolve_pair_delta_t(
            sequence,
            timestep,
            default_time_step=model_default_delta_t(model),
        ),
    )
    return _WORST_CASE_RECONSTRUCTION_LOSS(
        prediction,
        sequence[timestep + 1].x,
        target_mask,
    )


def compute_worst_case_reconstruction_loss(
    model: TrainableKoopmanModel,
    sequence: GraphSnapshotSequence,
    *,
    weight: float,
) -> Tensor:
    """Average worst-case reconstruction over consecutive pairs when enabled.

    Parameters
    ----------
    model : TrainableKoopmanModel
        Model producing one-step forecasts.
    sequence : GraphSnapshotSequence
        Source/target pairs.
    weight : float
        Active worst-case weight; zero skips evaluation.

    Returns
    -------
    Tensor
        Scalar mean of per-pair max-over-nodes MSE.
    """
    device = next(model.parameters()).device
    if weight == 0.0:
        return torch.zeros((), device=device)
    return mean_pair_sequence_loss(model, sequence, _worst_case_pair)
