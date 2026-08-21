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
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from koopman_graph.data import (
    GraphSnapshotSequence,
    HeteroGraphSnapshotSequence,
    SnapshotSequence,
)
from koopman_graph.graph_utils import snapshot_edge_weight
from koopman_graph.losses import (
    EigenvalueRegularizationLoss,
    rollout_multi_start_loss,
    rollout_sequence_loss,
)
from koopman_graph.nn.predicted_topology import (
    PredictedTopologyHead,
    SparseCandidateTopologyHead,
    build_supervision_index,
    candidate_edge_labels,
    dense_offdiag_index,
)
from koopman_graph.operators import (
    ContinuousGraphKoopmanOperator,
    ContinuousHeteroGraphKoopmanOperator,
    GraphKoopmanOperator,
    HeteroGraphKoopmanOperator,
    HypergraphKoopmanOperator,
)
from koopman_graph.protocols import TrainableKoopmanModel
from koopman_graph.training.extra_objectives import (
    compute_lie_consistency_loss,
    compute_pde_residual_loss,
    compute_sparsity_loss,
    compute_vamp2_loss,
    compute_worst_case_reconstruction_loss,
)
from koopman_graph.training.history import (
    ExtraLosses,
    LossWeights,
    TrainingLossBreakdown,
)
from koopman_graph.training.latent_cache import encode_sequence_latents
from koopman_graph.training.pair_objectives import (
    compute_backward_consistency_sequence_loss,
    compute_forward_consistency_sequence_loss,
    compute_sequence_loss,
    one_step_loss,
    one_step_predictions,
)

if TYPE_CHECKING:
    from koopman_graph.training.latent_cache import SequenceLatentCache

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


def _hetero_relation_banks(
    sequence: HeteroGraphSnapshotSequence,
) -> tuple[list[Tensor], list[Tensor | None]]:
    """Return ordered relation edge banks matching RelGraph / hetero advance.

    Banks are ordered by sorted edge-type ``repr`` (same convention as
    :class:`~koopman_graph.nn.heterogeneous.RelGraphEncoder`).

    Parameters
    ----------
    sequence : HeteroGraphSnapshotSequence
        Static-topology multiplex sequence.

    Returns
    -------
    tuple of (list of Tensor, list of Tensor or None)
        Ordered ``edge_indices`` and optional ``edge_weights``.

    Raises
    ------
    ValueError
        If the sequence uses dynamic topology.
    """
    if sequence.is_dynamic_topology:
        msg = (
            "dynamic-topology HeteroGraphSnapshotSequence eigenvalue "
            "regularization is not supported yet; use static relation banks"
        )
        raise ValueError(msg)
    edge_types = tuple(sorted(sequence.edge_types, key=repr))
    edge_index_dict = sequence.edge_index_dict
    edge_indices = [edge_index_dict[edge_type] for edge_type in edge_types]
    snapshot = sequence[0]
    edge_weights: list[Tensor | None] = [
        snapshot[edge_type].get("edge_weight", None) for edge_type in edge_types
    ]
    return edge_indices, edge_weights


def _hetero_eigenvalue_regularization_over_sequence(
    model: TrainableKoopmanModel,
    sequence: HeteroGraphSnapshotSequence,
) -> Tensor:
    """Evaluate hetero dense/ODO eigenvalue hinge on static relation banks.

    Parameters
    ----------
    model : TrainableKoopmanModel
        Model whose ``koopman`` is a :class:`HeteroGraphKoopmanOperator` or a
        :class:`ContinuousHeteroGraphKoopmanOperator`.
    sequence : HeteroGraphSnapshotSequence
        Training window or trajectory supplying relation topology.

    Returns
    -------
    Tensor
        Scalar eigenvalue hinge (may be zero above the assembled-size ceiling).

    Raises
    ------
    ValueError
        If ``sequence`` has fewer than two snapshots, dynamic topology, or
        relation count mismatches the operator.
    """
    if sequence.num_timesteps < 2:
        msg = (
            "HeteroGraphSnapshotSequence must contain at least 2 snapshots for "
            "hetero eigenvalue regularization"
        )
        raise ValueError(msg)
    koopman = model.koopman
    assert isinstance(
        koopman, (HeteroGraphKoopmanOperator, ContinuousHeteroGraphKoopmanOperator)
    )
    edge_indices, edge_weights = _hetero_relation_banks(sequence)
    if len(edge_indices) != koopman.num_relations:
        msg = (
            f"{type(koopman).__name__} has num_relations="
            f"{koopman.num_relations}, but sequence provides "
            f"{len(edge_indices)} edge types"
        )
        raise ValueError(msg)
    return _EIGENVALUE_REGULARIZATION_LOSS(
        koopman,
        dynamics_mode=model.dynamics_mode,
        edge_indices=edge_indices,
        num_nodes=sequence.num_nodes_total,
        edge_weights=edge_weights,
        num_nodes_dict=sequence.num_nodes_dict,
    )


def compute_eigenvalue_regularization_loss(
    model: TrainableKoopmanModel,
    sequence: GraphSnapshotSequence | HeteroGraphSnapshotSequence | None = None,
) -> Tensor:
    """Compute the eigenvalue hinge penalty for the model Koopman operator.

    Ordinary / custom operators use the per-node contract matrix (or structural
    ``bound_metric``). For networked dense/ODO modes
    (:class:`~koopman_graph.operators.GraphKoopmanOperator`,
    :class:`~koopman_graph.operators.HypergraphKoopmanOperator`,
    :class:`~koopman_graph.operators.HeteroGraphKoopmanOperator`), regularizes
    the topology-coupled effective operator: pass ``sequence`` so training can
    resolve topology. Structural modes still use factor-level
    ``bound_metric`` and do not require topology (they are **not**
    whole-network / joint ``ρ(K_eff)`` certificates).

    Parameters
    ----------
    model : TrainableKoopmanModel
        Model satisfying :class:`~koopman_graph.protocols.TrainableKoopmanModel`.
    sequence : GraphSnapshotSequence or HeteroGraphSnapshotSequence or None, optional
        Trajectory or window providing topology for networked dense/ODO
        regularization. Required when ``model.koopman`` is a dense/ODO
        networked operator.

    Returns
    -------
    Tensor
        Scalar eigenvalue regularization loss.

    Raises
    ------
    ValueError
        If a networked dense/ODO operator is regularized without ``sequence``,
        or a hetero operator is paired with a homogeneous sequence (or vice
        versa).
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
        if isinstance(sequence, HeteroGraphSnapshotSequence):
            msg = (
                "GraphKoopmanOperator eigenvalue regularization requires a "
                "homogeneous GraphSnapshotSequence, not HeteroGraphSnapshotSequence"
            )
            raise ValueError(msg)
        return _graph_eigenvalue_regularization_over_sequence(model, sequence)

    if isinstance(
        koopman, ContinuousGraphKoopmanOperator
    ) and koopman.parameterization in {"dense", "odo"}:
        if sequence is None:
            msg = (
                "sequence is required for eigenvalue regularization of "
                "ContinuousGraphKoopmanOperator dense/odo modes "
                "(topology-coupled effective generator); pass the training "
                "sequence/window"
            )
            raise ValueError(msg)
        if isinstance(sequence, HeteroGraphSnapshotSequence):
            msg = (
                "ContinuousGraphKoopmanOperator eigenvalue regularization "
                "requires a homogeneous GraphSnapshotSequence"
            )
            raise ValueError(msg)
        return _EIGENVALUE_REGULARIZATION_LOSS(
            koopman,
            dynamics_mode=model.dynamics_mode,
            edge_index=sequence.edge_index,
            num_nodes=sequence.num_nodes,
            edge_weight=sequence.edge_weight,
        )

    if isinstance(koopman, HypergraphKoopmanOperator) and koopman.parameterization in {
        "dense",
        "odo",
    }:
        if sequence is None:
            msg = (
                "sequence is required for eigenvalue regularization of "
                "HypergraphKoopmanOperator dense/odo modes (topology-coupled "
                "effective operator); pass the training sequence/window"
            )
            raise ValueError(msg)
        if isinstance(sequence, HeteroGraphSnapshotSequence):
            msg = (
                "HypergraphKoopmanOperator eigenvalue regularization requires "
                "a homogeneous GraphSnapshotSequence with hyperedges"
            )
            raise ValueError(msg)
        if not sequence.has_hyperedges:
            msg = (
                "HypergraphKoopmanOperator eigenvalue regularization requires "
                "a hyperedge-carrying sequence"
            )
            raise ValueError(msg)
        return _EIGENVALUE_REGULARIZATION_LOSS(
            koopman,
            dynamics_mode=model.dynamics_mode,
            hyperedge_index=sequence.hyperedge_index,
            num_nodes=sequence.num_nodes,
            hyperedge_weight=sequence.hyperedge_weight,
        )

    if isinstance(koopman, HeteroGraphKoopmanOperator) and koopman.parameterization in {
        "dense",
        "odo",
    }:
        if sequence is None:
            msg = (
                "sequence is required for eigenvalue regularization of "
                "HeteroGraphKoopmanOperator dense/odo modes (topology-coupled "
                "effective operator); pass a HeteroGraphSnapshotSequence"
            )
            raise ValueError(msg)
        if not isinstance(sequence, HeteroGraphSnapshotSequence):
            msg = (
                "HeteroGraphKoopmanOperator eigenvalue regularization requires "
                "a HeteroGraphSnapshotSequence"
            )
            raise ValueError(msg)
        return _hetero_eigenvalue_regularization_over_sequence(model, sequence)

    if isinstance(
        koopman, ContinuousHeteroGraphKoopmanOperator
    ) and koopman.parameterization in {"dense", "odo"}:
        if sequence is None:
            msg = (
                "sequence is required for eigenvalue regularization of "
                "ContinuousHeteroGraphKoopmanOperator dense/odo modes "
                "(topology-coupled effective generator); pass a "
                "HeteroGraphSnapshotSequence"
            )
            raise ValueError(msg)
        if not isinstance(sequence, HeteroGraphSnapshotSequence):
            msg = (
                "ContinuousHeteroGraphKoopmanOperator eigenvalue "
                "regularization requires a HeteroGraphSnapshotSequence"
            )
            raise ValueError(msg)
        return _hetero_eigenvalue_regularization_over_sequence(model, sequence)

    return _EIGENVALUE_REGULARIZATION_LOSS(
        koopman,
        dynamics_mode=model.dynamics_mode,
    )


def _validate_hetero_fit_surface(
    sequence: HeteroGraphSnapshotSequence,
    loss_weights: LossWeights,
    *,
    extra_losses: ExtraLosses | None,
) -> None:
    """Reject hetero fit features that would silently use wrong topology.

    Parameters
    ----------
    sequence : HeteroGraphSnapshotSequence
        Multiplex or typed training trajectory.
    loss_weights : LossWeights
        Active loss weights for this step.
    extra_losses : ExtraLosses or None
        Optional Lie / PDE callables.

    Raises
    ------
    ValueError
        If unsupported hetero fit options are requested.
    """
    if sequence.is_dynamic_topology:
        msg = (
            "dynamic-topology HeteroGraphSnapshotSequence fit is unsupported; "
            "use static relation banks"
        )
        raise ValueError(msg)
    if sequence.has_controls:
        msg = (
            "controlled HeteroGraphSnapshotSequence fit is unsupported; "
            "omit control_inputs"
        )
        raise ValueError(msg)
    if sequence.has_timestamps:
        msg = (
            "timestamped HeteroGraphSnapshotSequence fit is unsupported; "
            "omit timestamps"
        )
        raise ValueError(msg)
    if loss_weights.backward != 0.0:
        msg = (
            "backward consistency is unsupported for koopman='hetero_graph'; "
            "set loss_weights.backward=0"
        )
        raise ValueError(msg)
    if (
        loss_weights.lie != 0.0
        or loss_weights.pde != 0.0
        or loss_weights.worst_case != 0.0
        or loss_weights.vamp2 != 0.0
    ):
        msg = (
            "lie / pde / worst_case / vamp2 losses are unsupported for "
            "HeteroGraphSnapshotSequence fit"
        )
        raise ValueError(msg)
    if extra_losses is not None and (
        extra_losses.lie_dynamics_fn is not None
        or extra_losses.pde_residual_fn is not None
    ):
        msg = (
            "extra_losses lie_dynamics_fn / pde_residual_fn are unsupported for "
            "HeteroGraphSnapshotSequence fit"
        )
        raise ValueError(msg)


def compute_graph_dynamics_losses(
    model: TrainableKoopmanModel,
    sequence: SnapshotSequence,
    *,
    cache: SequenceLatentCache | None,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Return unweighted topology and presence BCE terms.

    Weights come from :class:`~koopman_graph.data.GraphDynamicsConfig`, not
    :class:`~koopman_graph.training.LossWeights`. Presence is skipped when
    the sequence has no presence masks.

    Parameters
    ----------
    model : TrainableKoopmanModel
        Model exposing ``graph_dynamics``, ``predicted_topology``, and
        optional ``presence_head``.
    sequence : GraphSnapshotSequence or HeteroGraphSnapshotSequence
        Training snapshots.
    cache : SequenceLatentCache or None
        Teacher-forced latents. Required when a topology or presence
        head is active on a homogeneous sequence.
    device : torch.device
        Device for zero scalars when the terms are inactive.

    Returns
    -------
    topology : Tensor
        Mean structural BCE (unweighted).
    presence : Tensor
        Mean presence BCE (unweighted).
    """
    zero = torch.zeros((), device=device)
    config = getattr(model, "graph_dynamics", None)
    if config is None or not isinstance(sequence, GraphSnapshotSequence):
        return zero, zero
    head = getattr(model, "predicted_topology", None)
    presence_head = getattr(model, "presence_head", None)
    topology_weight = float(config.topology_loss_weight)
    presence_weight = float(config.presence_loss_weight)
    need_topology = head is not None and topology_weight > 0.0
    need_presence = (
        presence_head is not None
        and presence_weight > 0.0
        and sequence.has_presence_masks
    )
    if not need_topology and not need_presence:
        return zero, zero
    if cache is None:
        msg = "graph-state losses require a SequenceLatentCache of teacher latents"
        raise ValueError(msg)
    topology_terms: list[Tensor] = []
    presence_terms: list[Tensor] = []
    num_pairs = sequence.num_timesteps - 1
    for timestep in range(num_pairs):
        z_t = cache.z[timestep]
        snapshot_t = sequence[timestep]
        snapshot_t1 = sequence[timestep + 1]
        if need_topology:
            if isinstance(head, SparseCandidateTopologyHead):
                candidates = build_supervision_index(
                    int(z_t.shape[0]),
                    head.candidate_k,
                    snapshot_t.edge_index,
                    snapshot_t1.edge_index,
                    device=z_t.device,
                )
                logits = head.pair_logits(z_t, candidates)
                labels = candidate_edge_labels(
                    candidates,
                    snapshot_t1.edge_index,
                    int(z_t.shape[0]),
                )
                topology_terms.append(
                    torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
                )
            elif isinstance(head, PredictedTopologyHead):
                logits = head.pairwise_logits(z_t)
                index = dense_offdiag_index(int(z_t.shape[0]), device=z_t.device)
                labels = candidate_edge_labels(
                    index,
                    snapshot_t1.edge_index,
                    int(z_t.shape[0]),
                )
                topology_terms.append(
                    torch.nn.functional.binary_cross_entropy_with_logits(
                        logits[index[0], index[1]],
                        labels,
                    )
                )
        if need_presence:
            target = sequence.presence_mask_at(timestep + 1).to(
                dtype=z_t.dtype, device=z_t.device
            )
            logits = presence_head(z_t)
            presence_terms.append(
                torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
            )
    topology = torch.stack(topology_terms).mean() if topology_terms else zero
    presence = torch.stack(presence_terms).mean() if presence_terms else zero
    return topology, presence


def compute_rollout_loss(
    model: TrainableKoopmanModel,
    sequence: SnapshotSequence,
    *,
    horizon: int,
    start_indices: Sequence[int],
    cache: SequenceLatentCache | None = None,
) -> Tensor:
    """Compute rollout reconstruction loss averaged over start origins.

    Parameters
    ----------
    model : TrainableKoopmanModel
        Model satisfying :class:`~koopman_graph.protocols.TrainableKoopmanModel`.
    sequence : GraphSnapshotSequence or HeteroGraphSnapshotSequence
        Training snapshots.
    horizon : int
        Number of rollout steps.
    start_indices : sequence of int
        Rollout origin indices.
    cache : SequenceLatentCache or None, optional
        Shared teacher-forced latents for rollout origins.

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
            cache=cache,
        )
    return rollout_multi_start_loss(
        model,
        sequence,
        horizon=horizon,
        start_indices=start_indices,
        cache=cache,
    )


def compute_training_loss(
    model: TrainableKoopmanModel,
    sequence: SnapshotSequence,
    loss_weights: LossWeights,
    *,
    extra_losses: ExtraLosses | None = None,
    rollout_horizon: int | None = None,
    rollout_start_indices: Sequence[int] | None = None,
) -> TrainingLossBreakdown:
    """Compute reconstruction, consistency, and rollout losses.

    When reconstruction, forward, backward, rollout, PDE, or worst-case
    weights are non-zero, encodes the sequence once into a shared
    :class:`~koopman_graph.training.latent_cache.SequenceLatentCache` and
    reuses those latents across those terms. Reconstruction, PDE, and
    worst-case additionally share one decoded one-step prediction per pair.

    Parameters
    ----------
    model : TrainableKoopmanModel
        Model satisfying :class:`~koopman_graph.protocols.TrainableKoopmanModel`.
    sequence : GraphSnapshotSequence or HeteroGraphSnapshotSequence
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
    # Evaluation-scoped continuous L_eff / Φ caches must not span optimizer steps.
    koopman = model.koopman
    if isinstance(
        koopman, (ContinuousGraphKoopmanOperator, ContinuousHeteroGraphKoopmanOperator)
    ):
        koopman.clear_transition_cache()

    if isinstance(sequence, HeteroGraphSnapshotSequence):
        _validate_hetero_fit_surface(
            sequence,
            loss_weights,
            extra_losses=extra_losses,
        )

    device = next(model.parameters()).device
    needs_shared_predictions = (
        loss_weights.reconstruction != 0.0
        or loss_weights.pde != 0.0
        or loss_weights.worst_case != 0.0
    )
    needs_latent_cache = (
        needs_shared_predictions
        or loss_weights.forward != 0.0
        or loss_weights.backward != 0.0
        or loss_weights.rollout != 0.0
        or loss_weights.vamp2 != 0.0
        or getattr(model, "graph_dynamics", None) is not None
    )
    cache = encode_sequence_latents(model, sequence) if needs_latent_cache else None
    predictions = (
        one_step_predictions(model, sequence, cache=cache)
        if needs_shared_predictions and cache is not None
        else None
    )

    if loss_weights.reconstruction != 0.0:
        reconstruction = compute_sequence_loss(
            model,
            sequence,
            cache=cache,
            predictions=predictions,
        )
    else:
        reconstruction = torch.zeros((), device=device)

    if loss_weights.forward != 0.0:
        forward = compute_forward_consistency_sequence_loss(
            model, sequence, cache=cache
        )
    else:
        forward = torch.zeros((), device=device)

    if loss_weights.backward != 0.0:
        backward = compute_backward_consistency_sequence_loss(
            model, sequence, cache=cache
        )
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
        predictions=predictions,
    )
    sparsity = compute_sparsity_loss(model, weight=loss_weights.sparsity)
    worst_case = compute_worst_case_reconstruction_loss(
        model,
        sequence,
        weight=loss_weights.worst_case,
        predictions=predictions,
    )
    if isinstance(sequence, GraphSnapshotSequence):
        vamp2 = compute_vamp2_loss(
            model,
            sequence,
            weight=loss_weights.vamp2,
            cache=cache,
        )
    else:
        vamp2 = torch.zeros((), device=device)

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
            cache=cache,
        )
    else:
        rollout = torch.zeros((), device=device)

    topology, presence = compute_graph_dynamics_losses(
        model,
        sequence,
        cache=cache,
        device=device,
    )
    config = getattr(model, "graph_dynamics", None)
    topology_weight = 0.0 if config is None else float(config.topology_loss_weight)
    presence_weight = 0.0 if config is None else float(config.presence_loss_weight)

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
        + loss_weights.vamp2 * vamp2
        + topology_weight * topology
        + presence_weight * presence
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
        vamp2=vamp2,
        topology=topology,
        presence=presence,
        total=total,
    )


__all__ = [
    "compute_backward_consistency_sequence_loss",
    "compute_eigenvalue_regularization_loss",
    "compute_forward_consistency_sequence_loss",
    "compute_graph_dynamics_losses",
    "compute_rollout_loss",
    "compute_sequence_loss",
    "compute_training_loss",
    "one_step_loss",
]
