"""Opt-in batched reconstruction / forward losses over a PyG ``Batch``.

Vectorizes the default :func:`~koopman_graph.training.compute_training_loss`
hot path (reconstruction, and forward consistency when weighted) across
independent homogeneous trajectories. Other terms stay the per-sequence mean
so totals still match :func:`~koopman_graph.training.mean_training_loss_breakdown`.
Eigenvalue hinges are never evaluated on the disconnected union graph.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import torch
from torch import Tensor, nn
from torch_geometric.data import Batch

from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.data.batching import collate_graph_trajectories
from koopman_graph.graph_utils import propagate_latent, snapshot_edge_weight
from koopman_graph.losses import masked_mse_loss
from koopman_graph.nn import DelayEmbeddingEncoder, HypergraphEncoder, RelGraphEncoder
from koopman_graph.operators import (
    ContinuousGraphKoopmanOperator,
    ContinuousHeteroGraphKoopmanOperator,
    ContinuousKoopmanOperator,
    GlobalLocalKoopmanOperator,
    HeteroGraphKoopmanOperator,
    HypergraphKoopmanOperator,
)
from koopman_graph.protocols import TrainableKoopmanModel
from koopman_graph.training.history import (
    ExtraLosses,
    LossWeights,
    TrainingLossBreakdown,
    mean_training_loss_breakdown,
)
from koopman_graph.training.objectives import compute_training_loss
from koopman_graph.training.pair_objectives import model_default_delta_t

__all__ = [
    "compute_batched_training_loss",
    "validate_graph_batching_request",
]


def _unwrap_encoder(model: TrainableKoopmanModel) -> nn.Module:
    """Return the base encoder, unwrapping a delay wrapper when present.

    Parameters
    ----------
    model : TrainableKoopmanModel
        Model whose encoder is inspected.

    Returns
    -------
    nn.Module
        Delay base encoder when wrapped, else ``model.encoder``.
    """
    encoder = model.encoder
    if isinstance(encoder, DelayEmbeddingEncoder):
        return encoder.base_encoder
    return encoder


def validate_graph_batching_request(
    model: TrainableKoopmanModel,
    sequences: Sequence[GraphSnapshotSequence],
) -> None:
    """Refuse ``batch_graphs=True`` combinations that change ``N`` or mix families.

    Parameters
    ----------
    model : TrainableKoopmanModel
        Model whose operator, encoder, and delay / control flags are checked.
    sequences : sequence of GraphSnapshotSequence
        Trajectories that would be collated.

    Raises
    ------
    TypeError
        If any trajectory is not a :class:`GraphSnapshotSequence`.
    ValueError
        If a refused feature is active (hetero / hypergraph / continuous /
        controls / delay / orbits / learned topology / timestamps).
    """
    if not sequences:
        msg = "batch_graphs=True requires at least one GraphSnapshotSequence"
        raise ValueError(msg)
    for index, sequence in enumerate(sequences):
        if not isinstance(sequence, GraphSnapshotSequence):
            msg = (
                "batch_graphs=True requires homogeneous GraphSnapshotSequence "
                f"trajectories; index {index} has type {type(sequence).__name__}"
            )
            raise TypeError(msg)
        if sequence.has_controls:
            msg = (
                "batch_graphs=True is unsupported with control_inputs; "
                f"sequence {index} has controls"
            )
            raise ValueError(msg)
        if sequence.has_hyperedges:
            msg = (
                "batch_graphs=True is unsupported with hyperedge incidence; "
                f"sequence {index} has hyperedges"
            )
            raise ValueError(msg)
        if sequence.has_timestamps:
            msg = (
                "batch_graphs=True is unsupported with timestamps; "
                f"sequence {index} has timestamps"
            )
            raise ValueError(msg)

    if int(getattr(model, "n_delays", 1)) > 1:
        msg = "batch_graphs=True is unsupported with delay embeddings (n_delays>1)"
        raise ValueError(msg)
    if int(getattr(model, "control_dim", 0)) > 0:
        msg = "batch_graphs=True is unsupported when control_dim > 0"
        raise ValueError(msg)
    if getattr(model, "learns_pairwise_topology", False):
        msg = "batch_graphs=True is unsupported with learned pairwise topology"
        raise ValueError(msg)
    if getattr(model, "adaptive_topology", None) is not None:
        msg = "batch_graphs=True is unsupported with adaptive topology"
        raise ValueError(msg)
    if getattr(model, "predicted_topology", None) is not None:
        config = getattr(model, "graph_dynamics", None)
        if config is not None and bool(config.recursive_training):
            msg = (
                "batch_graphs=True is unsupported with recursive graph-state "
                "topology (predicted Â feeds the operator step)"
            )
            raise ValueError(msg)
    if getattr(model, "uses_hypergraph_koopman", False):
        msg = "batch_graphs=True is unsupported with hypergraph Koopman operators"
        raise ValueError(msg)

    encoder = _unwrap_encoder(model)
    if isinstance(encoder, (RelGraphEncoder, HypergraphEncoder)):
        msg = (
            "batch_graphs=True is unsupported with RelGraph or hypergraph "
            f"encoders; got {type(encoder).__name__}"
        )
        raise ValueError(msg)

    koopman = model.koopman
    if isinstance(
        koopman,
        (
            ContinuousKoopmanOperator,
            ContinuousGraphKoopmanOperator,
            ContinuousHeteroGraphKoopmanOperator,
            HeteroGraphKoopmanOperator,
            HypergraphKoopmanOperator,
            GlobalLocalKoopmanOperator,
        ),
    ):
        msg = (
            "batch_graphs=True requires a discrete per-node or graph Koopman "
            f"operator; got {type(koopman).__name__}"
        )
        raise ValueError(msg)
    if getattr(koopman, "auto_orbits", False) or getattr(
        koopman, "isotypic_symmetry", False
    ):
        msg = "batch_graphs=True is unsupported with orbit-tied or isotypic operators"
        raise ValueError(msg)
    if getattr(koopman, "orbit_partition", None) is not None:
        msg = "batch_graphs=True is unsupported with orbit-tied operators"
        raise ValueError(msg)
    if getattr(koopman, "uses_orbit_selves", False):
        msg = "batch_graphs=True is unsupported with orbit-tied operators"
        raise ValueError(msg)


def _per_graph_mse(
    prediction: Tensor,
    target: Tensor,
    time_batch: Batch,
    node_masks: Sequence[Tensor | None] | None,
) -> Tensor:
    """Return per-graph mean squared error over nodes and features.

    Each graph is reduced independently (mean over that graph's contributing
    entries), matching per-sequence ``mse_loss`` / ``masked_mse_loss`` rather
    than a node-weighted union mean.

    Parameters
    ----------
    prediction : Tensor
        Concatenated predictions with shape ``(sum_g N_g, F)``.
    target : Tensor
        Concatenated targets with the same shape.
    time_batch : Batch
        Time-slice batch whose ``ptr`` splits concatenated node rows.
    node_masks : sequence of Tensor or None, or None
        Optional per-graph boolean masks with length ``time_batch.num_graphs``.

    Returns
    -------
    Tensor
        Per-graph scalars with shape ``(num_graphs,)``.
    """
    ptr = time_batch.ptr
    num_graphs = int(time_batch.num_graphs)
    losses: list[Tensor] = []
    for graph_id in range(num_graphs):
        start = int(ptr[graph_id].item())
        stop = int(ptr[graph_id + 1].item())
        pred_g = prediction[start:stop]
        tgt_g = target[start:stop]
        mask_g = None if node_masks is None else node_masks[graph_id]
        if mask_g is None:
            losses.append(nn.functional.mse_loss(pred_g, tgt_g))
        else:
            losses.append(masked_mse_loss(pred_g, tgt_g, mask_g))
    return torch.stack(losses)


def _optional_edge_weight(snapshot: Batch) -> Tensor | None:
    """Return ``edge_weight`` when present on ``snapshot``.

    Parameters
    ----------
    snapshot : Batch
        Time-slice batch.

    Returns
    -------
    Tensor or None
        Edge weights, or ``None`` when absent.
    """
    return snapshot_edge_weight(snapshot)


def _needs_non_vectorized_terms(
    loss_weights: LossWeights,
    model: TrainableKoopmanModel | None = None,
) -> bool:
    """Return whether any non-recon/forward term is active.

    Parameters
    ----------
    loss_weights : LossWeights
        Active weights.
    model : TrainableKoopmanModel or None, optional
        Used to detect graph-state topology / presence terms.

    Returns
    -------
    bool
        ``True`` when backward, rollout, eigenvalue, extras, sparsity,
        or graph-state terms are active.
    """
    if (
        loss_weights.backward != 0.0
        or loss_weights.rollout != 0.0
        or loss_weights.eigenvalue != 0.0
        or loss_weights.lie != 0.0
        or loss_weights.pde != 0.0
        or loss_weights.sparsity != 0.0
        or loss_weights.worst_case != 0.0
        or loss_weights.vamp2 != 0.0
    ):
        return True
    config = None if model is None else getattr(model, "graph_dynamics", None)
    if config is None:
        return False
    return (
        float(config.topology_loss_weight) > 0.0
        or float(config.presence_loss_weight) > 0.0
    )


def compute_batched_training_loss(
    model: TrainableKoopmanModel,
    sequences: Sequence[GraphSnapshotSequence],
    loss_weights: LossWeights,
    *,
    extra_losses: ExtraLosses | None = None,
    rollout_horizon: int | None = None,
    rollout_start_indices: Sequence[int] | None = None,
) -> TrainingLossBreakdown:
    """Compute training loss on a collated multi-graph ``Batch``.

    Reconstruction (and forward consistency when weighted) reduce **per
    graph then mean over graphs**, matching
    :func:`~koopman_graph.training.mean_training_loss_breakdown`. Remaining
    terms use the existing per-sequence helpers so the total still matches
    a Python ``MultiTrajectory`` loop on modest batches.

    Parameters
    ----------
    model : TrainableKoopmanModel
        Homogeneous discrete per-node or graph model.
    sequences : sequence of GraphSnapshotSequence
        Trajectories sharing ``num_timesteps`` (node counts may differ).
    loss_weights : LossWeights
        Active term weights.
    extra_losses : ExtraLosses or None, optional
        Fit-time Lie / PDE callables for leftover per-sequence terms.
    rollout_horizon : int or None, optional
        Rollout horizon forwarded to leftover per-sequence losses.
    rollout_start_indices : sequence of int or None, optional
        Rollout origins forwarded to leftover per-sequence losses.

    Returns
    -------
    TrainingLossBreakdown
        Unweighted per-term losses and the weighted total.

    Raises
    ------
    TypeError, ValueError
        If :func:`validate_graph_batching_request` or collate rejects the
        request, or a trajectory has fewer than two snapshots.
    """
    validate_graph_batching_request(model, sequences)
    if sequences[0].num_timesteps < 2:
        msg = "snapshot sequence must contain at least 2 snapshots for training"
        raise ValueError(msg)

    device = next(model.parameters()).device
    reconstruction = torch.zeros((), device=device)
    forward = torch.zeros((), device=device)
    vectorize_recon = loss_weights.reconstruction != 0.0
    vectorize_forward = loss_weights.forward != 0.0

    if vectorize_recon or vectorize_forward:
        batched = collate_graph_trajectories(sequences)
        num_times = batched.num_timesteps
        time_batches = [
            batched.graphs_at_time(timestep) for timestep in range(num_times)
        ]
        latents = [model.encode(time_batch) for time_batch in time_batches]
        default_delta_t = model_default_delta_t(model)
        num_pairs = num_times - 1
        recon_pairs: list[Tensor] = []
        forward_pairs: list[Tensor] = []
        for timestep in range(num_pairs):
            source = time_batches[timestep]
            target_batch = time_batches[timestep + 1]
            z_next = propagate_latent(
                model.koopman,
                latents[timestep],
                default_delta_t=default_delta_t,
                edge_index=target_batch.edge_index,
                edge_weight=_optional_edge_weight(target_batch),
            )
            if vectorize_recon:
                prediction = model.decoder(
                    z_next,
                    source.edge_index,
                    _optional_edge_weight(source),
                )
                recon_masks = [
                    sequence.loss_mask_at(timestep + 1) for sequence in sequences
                ]
                recon_pairs.append(
                    _per_graph_mse(
                        prediction,
                        target_batch.x,
                        source,
                        recon_masks,
                    )
                )
            if vectorize_forward:
                pair_masks = [
                    sequence.pair_loss_mask(timestep) for sequence in sequences
                ]
                forward_pairs.append(
                    _per_graph_mse(
                        z_next,
                        latents[timestep + 1],
                        source,
                        pair_masks,
                    )
                )
        if vectorize_recon:
            reconstruction = torch.stack(recon_pairs, dim=0).mean(dim=0).mean()
        if vectorize_forward:
            forward = torch.stack(forward_pairs, dim=0).mean(dim=0).mean()

    backward = torch.zeros((), device=device)
    rollout = torch.zeros((), device=device)
    eigenvalue = torch.zeros((), device=device)
    lie = torch.zeros((), device=device)
    pde = torch.zeros((), device=device)
    sparsity = torch.zeros((), device=device)
    worst_case = torch.zeros((), device=device)
    vamp2 = torch.zeros((), device=device)
    topology = torch.zeros((), device=device)
    presence = torch.zeros((), device=device)
    if _needs_non_vectorized_terms(loss_weights, model):
        rest_weights = replace(
            loss_weights,
            reconstruction=0.0,
            forward=0.0,
        )
        rest = mean_training_loss_breakdown(
            [
                compute_training_loss(
                    model,
                    sequence,
                    rest_weights,
                    extra_losses=extra_losses,
                    rollout_horizon=rollout_horizon,
                    rollout_start_indices=rollout_start_indices,
                )
                for sequence in sequences
            ]
        )
        backward = rest.backward
        rollout = rest.rollout
        eigenvalue = rest.eigenvalue
        lie = rest.lie
        pde = rest.pde
        sparsity = rest.sparsity
        worst_case = rest.worst_case
        vamp2 = rest.vamp2
        topology = rest.topology
        presence = rest.presence

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
