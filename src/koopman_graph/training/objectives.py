"""Training-side objective orchestration over pair and extra peers.

Owns eigenvalue / rollout composition and
:func:`compute_training_loss`. Pair reconstruction/consistency helpers live in
:mod:`~koopman_graph.training.pair_objectives`; Lie / PDE / sparsity /
worst-case helpers live in :mod:`~koopman_graph.training.extra_objectives`.
Public pair helpers are re-exported here so existing deep-import monkeypatches
against this module remain stable.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.graph_utils import snapshot_edge_weight
from koopman_graph.losses import (
    EigenvalueRegularizationLoss,
    rollout_multi_start_loss,
    rollout_sequence_loss,
)
from koopman_graph.operators import GraphKoopmanOperator
from koopman_graph.protocols import TrainableKoopmanModel
from koopman_graph.training.extra_objectives import (
    compute_lie_consistency_loss,
    compute_pde_residual_loss,
    compute_sparsity_loss,
    compute_worst_case_reconstruction_loss,
)
from koopman_graph.training.history import (
    ExtraLosses,
    LossWeights,
    TrainingLossBreakdown,
)
from koopman_graph.training.pair_objectives import (
    compute_backward_consistency_sequence_loss,
    compute_forward_consistency_sequence_loss,
    compute_sequence_loss,
    one_step_loss,
)

_EIGENVALUE_REGULARIZATION_LOSS = EigenvalueRegularizationLoss()


def _topologies_equal(
    edge_index_a: Tensor,
    edge_weight_a: Tensor | None,
    edge_index_b: Tensor,
    edge_weight_b: Tensor | None,
) -> bool:
    """Return whether two topology payloads are numerically identical.

    Parameters
    ----------
    edge_index_a : Tensor
        First edge index.
    edge_weight_a : Tensor or None
        First optional edge weights.
    edge_index_b : Tensor
        Second edge index.
    edge_weight_b : Tensor or None
        Second optional edge weights.

    Returns
    -------
    bool
        ``True`` when indices match and weights match (including both absent).
    """
    if not torch.equal(edge_index_a, edge_index_b):
        return False
    if (edge_weight_a is None) != (edge_weight_b is None):
        return False
    if edge_weight_a is None:
        return True
    assert edge_weight_b is not None
    return torch.allclose(edge_weight_a, edge_weight_b, equal_nan=True)


def _graph_eigenvalue_regularization_over_sequence(
    model: TrainableKoopmanModel,
    sequence: GraphSnapshotSequence,
) -> Tensor:
    """Average graph dense/ODO eigenvalue hinges over pair-target topologies.

    Static sequences evaluate the effective operator once. Dynamic sequences
    average the hinge over each forward-consistency pair target
    (``sequence[t + 1]``), reusing cached penalties for identical topologies.

    Parameters
    ----------
    model : TrainableKoopmanModel
        Model whose ``koopman`` is a :class:`GraphKoopmanOperator`.
    sequence : GraphSnapshotSequence
        Training window or trajectory supplying topology.

    Returns
    -------
    Tensor
        Scalar mean eigenvalue hinge.

    Raises
    ------
    ValueError
        If ``sequence`` has fewer than two snapshots.
    """
    if sequence.num_timesteps < 2:
        msg = (
            "GraphSnapshotSequence must contain at least 2 snapshots for "
            "graph eigenvalue regularization"
        )
        raise ValueError(msg)

    num_nodes = sequence.num_nodes
    if not sequence.is_dynamic_topology:
        return _EIGENVALUE_REGULARIZATION_LOSS(
            model.koopman,
            dynamics_mode=model.dynamics_mode,
            edge_index=sequence.edge_index,
            num_nodes=num_nodes,
            edge_weight=sequence.edge_weight,
        )

    num_pairs = sequence.num_timesteps - 1
    device = next(model.parameters()).device
    total = torch.zeros((), device=device)
    cache: list[tuple[Tensor, Tensor | None, Tensor]] = []
    for t in range(num_pairs):
        snapshot = sequence[t + 1]
        edge_index = snapshot.edge_index
        edge_weight = snapshot_edge_weight(snapshot)
        cached: Tensor | None = None
        for cached_index, cached_weight, cached_penalty in cache:
            if _topologies_equal(
                edge_index,
                edge_weight,
                cached_index,
                cached_weight,
            ):
                cached = cached_penalty
                break
        if cached is None:
            penalty = _EIGENVALUE_REGULARIZATION_LOSS(
                model.koopman,
                dynamics_mode=model.dynamics_mode,
                edge_index=edge_index,
                num_nodes=num_nodes,
                edge_weight=edge_weight,
            )
            cache.append((edge_index, edge_weight, penalty))
        else:
            penalty = cached
        total = total + penalty
    return total / num_pairs


def compute_eigenvalue_regularization_loss(
    model: TrainableKoopmanModel,
    sequence: GraphSnapshotSequence | None = None,
) -> Tensor:
    """Compute the eigenvalue hinge penalty for the model Koopman operator.

    Ordinary / custom operators use the per-node contract matrix (or structural
    ``bound_metric``). For :class:`~koopman_graph.operators.GraphKoopmanOperator`
    dense/ODO modes, regularizes the topology-coupled effective operator:
    pass ``sequence`` so training can resolve pair/window topology. Structural
    graph modes still use factor-level ``bound_metric`` and do not require
    topology (they are **not** whole-network certificates).

    Parameters
    ----------
    model : TrainableKoopmanModel
        Model satisfying :class:`~koopman_graph.protocols.TrainableKoopmanModel`.
    sequence : GraphSnapshotSequence or None, optional
        Trajectory or window providing topology for graph dense/ODO
        regularization. Required when ``model.koopman`` is a dense/ODO
        :class:`~koopman_graph.operators.GraphKoopmanOperator`.

    Returns
    -------
    Tensor
        Scalar eigenvalue regularization loss.

    Raises
    ------
    ValueError
        If a graph dense/ODO operator is regularized without ``sequence``.
    """
    koopman = model.koopman
    if isinstance(koopman, GraphKoopmanOperator) and koopman.parameterization in {
        "dense",
        "odo",
    }:
        if sequence is None:
            msg = (
                "sequence is required for eigenvalue regularization of "
                "GraphKoopmanOperator dense/odo modes (topology-coupled "
                "effective operator); pass the training sequence/window"
            )
            raise ValueError(msg)
        return _graph_eigenvalue_regularization_over_sequence(model, sequence)
    return _EIGENVALUE_REGULARIZATION_LOSS(
        koopman,
        dynamics_mode=model.dynamics_mode,
    )


def compute_rollout_loss(
    model: TrainableKoopmanModel,
    sequence: GraphSnapshotSequence,
    *,
    horizon: int,
    start_indices: Sequence[int],
) -> Tensor:
    """Compute rollout reconstruction loss averaged over start origins.

    Parameters
    ----------
    model : TrainableKoopmanModel
        Model satisfying :class:`~koopman_graph.protocols.TrainableKoopmanModel`.
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
    model: TrainableKoopmanModel,
    sequence: GraphSnapshotSequence,
    loss_weights: LossWeights,
    *,
    extra_losses: ExtraLosses | None = None,
    rollout_horizon: int | None = None,
    rollout_start_indices: Sequence[int] | None = None,
) -> TrainingLossBreakdown:
    """Compute reconstruction, consistency, and rollout losses.

    Parameters
    ----------
    model : TrainableKoopmanModel
        Model satisfying :class:`~koopman_graph.protocols.TrainableKoopmanModel`.
    sequence : :class:`~koopman_graph.data.GraphSnapshotSequence`
        Time-ordered snapshots with at least two timesteps.
    loss_weights : :class:`~koopman_graph.training.LossWeights`
        Weights for reconstruction, forward, backward, and rollout terms.
    extra_losses : :class:`~koopman_graph.training.ExtraLosses` or None, optional
        Fit-time vector-field and PDE-residual callables. Required when the
        corresponding ``lie`` or ``pde`` weight is non-zero.
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

    if loss_weights.reconstruction != 0.0:
        reconstruction = compute_sequence_loss(model, sequence)
    else:
        reconstruction = torch.zeros((), device=device)

    if loss_weights.forward != 0.0:
        forward = compute_forward_consistency_sequence_loss(model, sequence)
    else:
        forward = torch.zeros((), device=device)

    if loss_weights.backward != 0.0:
        backward = compute_backward_consistency_sequence_loss(model, sequence)
    else:
        backward = torch.zeros((), device=device)

    if loss_weights.eigenvalue != 0.0:
        eigenvalue = compute_eigenvalue_regularization_loss(model, sequence)
    else:
        eigenvalue = torch.zeros((), device=device)

    lie = compute_lie_consistency_loss(
        model,
        sequence,
        weight=loss_weights.lie,
        extra_losses=extra_losses,
    )
    pde = compute_pde_residual_loss(
        model,
        sequence,
        weight=loss_weights.pde,
        extra_losses=extra_losses,
    )
    sparsity = compute_sparsity_loss(model, weight=loss_weights.sparsity)
    worst_case = compute_worst_case_reconstruction_loss(
        model,
        sequence,
        weight=loss_weights.worst_case,
    )

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
        + loss_weights.lie * lie
        + loss_weights.pde * pde
        + loss_weights.sparsity * sparsity
        + loss_weights.worst_case * worst_case
    )
    return TrainingLossBreakdown(
        reconstruction=reconstruction,
        forward=forward,
        backward=backward,
        rollout=rollout,
        eigenvalue=eigenvalue,
        lie=lie,
        pde=pde,
        sparsity=sparsity,
        worst_case=worst_case,
        total=total,
    )


__all__ = [
    "compute_backward_consistency_sequence_loss",
    "compute_eigenvalue_regularization_loss",
    "compute_forward_consistency_sequence_loss",
    "compute_rollout_loss",
    "compute_sequence_loss",
    "compute_training_loss",
    "one_step_loss",
]
