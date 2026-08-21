"""Autoregressive rollout reconstruction losses."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch_geometric.data import HeteroData

from koopman_graph.data import (
    GraphSnapshotSequence,
    HeteroGraphSnapshotSequence,
    SnapshotSequence,
    resolve_pair_delta_t,
)
from koopman_graph.graph_utils import (
    autoregressive_hetero_latent_rollout,
    autoregressive_latent_rollout,
    snapshot_topology_at,
)
from koopman_graph.losses.reconstruction import masked_mse_loss
from koopman_graph.nn import HypergraphDecoder, RelGraphDecoder, bind_hypergraph_decoder
from koopman_graph.nn.heterogeneous import (
    resolve_multiplex_relation_inputs,
    resolve_typed_relation_inputs,
)
from koopman_graph.nn.predicted_topology import (
    make_recursive_topology_at,
    recursive_training_enabled,
)
from koopman_graph.operators import HeteroGraphKoopmanOperator
from koopman_graph.protocols import TrainableKoopmanModel


def _multiplex_target_features(snapshot: HeteroData) -> Tensor:
    """Return sole-node-type features from a multiplex target snapshot.

    Parameters
    ----------
    snapshot : HeteroData
        Multiplex teacher target (exactly one node type).

    Returns
    -------
    Tensor
        Node features with shape ``(num_nodes, in_channels)``.

    Raises
    ------
    ValueError
        If ``snapshot`` is not multiplex or is missing ``x``.
    """
    node_types = tuple(snapshot.node_types)
    if len(node_types) != 1:
        msg = (
            "hetero rollout loss requires multiplex HeteroData with exactly "
            f"one node type; got {node_types!r}"
        )
        raise ValueError(msg)
    features = snapshot[node_types[0]].x
    if features is None:
        msg = f"HeteroData node type {node_types[0]!r} is missing feature matrix x"
        raise ValueError(msg)
    return features


def _relation_topology_at_from_targets(
    targets: Sequence[HeteroData],
    *,
    num_relations: int,
    node_types: Sequence[str] | None = None,
    edge_types: Sequence[Sequence[str]] | None = None,
):
    """Build a per-step relation-bank schedule from teacher target snapshots.

    Parameters
    ----------
    targets : sequence of HeteroData
        Teacher targets aligned with autoregressive decode steps.
    num_relations : int
        Expected relation-bank length ``|R|``.
    node_types : sequence of str or None, optional
        Typed stacking order; when given, banks use stacked global numbering.
    edge_types : sequence of sequence of str or None, optional
        Explicit relation order for the typed path.

    Returns
    -------
    callable
        ``topology_at(step)`` returning ``(edge_indices, edge_weights)``.
    """

    def topology_at(step: int) -> tuple[list[Tensor], list[Tensor | None]]:
        """Return relation banks from the teacher target at ``step``.

        Parameters
        ----------
        step : int
            Zero-based index into ``targets``.

        Returns
        -------
        tuple of list
            Ordered ``edge_indices`` and optional ``edge_weights`` banks.
        """
        if node_types is not None:
            _, edge_indices, edge_weights, _ = resolve_typed_relation_inputs(
                targets[step],
                node_types=node_types,
                edge_types=edge_types,
                num_relations=num_relations,
            )
            return edge_indices, edge_weights
        _, edge_indices, edge_weights = resolve_multiplex_relation_inputs(
            targets[step],
            num_relations=num_relations,
        )
        return edge_indices, edge_weights

    return topology_at


def _bind_hetero_decoder(
    decoder: RelGraphDecoder,
    num_nodes_dict: Mapping[str, int] | None,
):
    """Adapt a hetero decoder to the rollout decode signature.

    Parameters
    ----------
    decoder : RelGraphDecoder
        Multiplex or typed relational decoder.
    num_nodes_dict : mapping of str to int or None
        Per-type node counts required by typed decoders.

    Returns
    -------
    callable
        ``decode(z, edge_indices, edge_weights)``.
    """
    if not decoder.is_typed:
        return decoder

    def decode(
        latent: Tensor,
        edge_indices: Sequence[Tensor],
        edge_weights: Sequence[Tensor | None],
    ) -> Tensor | dict[str, Tensor]:
        """Decode one rollout step with typed per-type heads.

        Parameters
        ----------
        latent : Tensor
            Advanced stacked latent block.
        edge_indices : sequence of Tensor
            Ordered relation banks for this step.
        edge_weights : sequence of Tensor or None
            Optional per-relation weights.

        Returns
        -------
        Tensor or dict of str to Tensor
            Reconstructed features for this step.
        """
        return decoder(
            latent,
            edge_indices,
            edge_weights,
            num_nodes_dict=num_nodes_dict,
        )

    return decode


def _typed_target_features(
    snapshot: HeteroData,
    node_types: Sequence[str],
) -> dict[str, Tensor]:
    """Return per-type features from a typed teacher target snapshot.

    Parameters
    ----------
    snapshot : HeteroData
        Typed teacher target.
    node_types : sequence of str
        Ordered node-type names to extract.

    Returns
    -------
    dict of str to Tensor
        Feature matrix ``(N_τ, F_τ)`` per node type.

    Raises
    ------
    ValueError
        If a node type or its ``x`` matrix is missing.
    """
    features: dict[str, Tensor] = {}
    present = set(snapshot.node_types)
    for name in node_types:
        if name not in present:
            msg = (
                f"hetero rollout target is missing node type {name!r}; "
                f"present types are {sorted(present)!r}"
            )
            raise ValueError(msg)
        block = snapshot[name].x
        if block is None:
            msg = f"HeteroData node type {name!r} is missing feature matrix x"
            raise ValueError(msg)
        features[name] = block
    return features


def _hetero_rollout_step_loss(
    prediction: Tensor | Mapping[str, Tensor],
    target: HeteroData,
    *,
    node_types: Sequence[str] | None,
    masks: Mapping[str, Tensor] | None,
) -> Tensor:
    """Return the reconstruction loss for one hetero rollout step.

    Parameters
    ----------
    prediction : Tensor or mapping of str to Tensor
        Decoded features for this step.
    target : HeteroData
        Teacher target snapshot.
    node_types : sequence of str or None
        Typed stacking order, or ``None`` for the multiplex path.
    masks : mapping of str to Tensor or None
        Optional per-type observation masks at the target snapshot.

    Returns
    -------
    Tensor
        Scalar step loss (mean over node types for the typed path).
    """
    if node_types is None or isinstance(prediction, Tensor):
        assert isinstance(prediction, Tensor)
        features = _multiplex_target_features(target)
        if masks is None:
            return nn.functional.mse_loss(prediction, features)
        sole = tuple(target.node_types)[0]
        return masked_mse_loss(prediction, features, masks[sole])
    targets = _typed_target_features(target, node_types)
    total = None
    for name in node_types:
        block = (
            nn.functional.mse_loss(prediction[name], targets[name])
            if masks is None
            else masked_mse_loss(prediction[name], targets[name], masks[name])
        )
        total = block if total is None else total + block
    assert total is not None
    return total / len(tuple(node_types))


def rollout_sequence_loss(
    model: TrainableKoopmanModel,
    sequence: SnapshotSequence,
    *,
    horizon: int,
    start: int = 0,
    cache: Any | None = None,
    origin_latent: Tensor | None = None,
) -> Tensor:
    """Compute autoregressive rollout reconstruction loss from one start snapshot.

    Encodes ``sequence[start]`` once via
    :meth:`~koopman_graph.protocols.TrainableKoopmanModel.encode`, advances the
    latent state with the model's Koopman operator for ``horizon`` steps, and
    compares decoded predictions to the observed snapshots
    ``sequence[start + 1 : start + horizon + 1]``. This term aligns training
    with :meth:`~koopman_graph.model.GraphKoopmanModel.predict` via the shared
    :func:`~koopman_graph.graph_utils.autoregressive_latent_rollout` primitive.
    Decode topology uses **teacher target** edges (per-step snapshot topology),
    whereas ``predict`` uses hold-last unless ``future_topologies`` are supplied
    — see :mod:`koopman_graph.graph_utils`.

    Parameters
    ----------
    model : :class:`~koopman_graph.protocols.TrainableKoopmanModel`
        Trainable model exposing ``encode``, ``resolve_delta_t``, ``koopman``,
        and ``decoder``. :class:`~koopman_graph.model.GraphKoopmanModel` is the
        intended implementer; no encoder-only fallback is used.
    sequence : GraphSnapshotSequence or HeteroGraphSnapshotSequence
        Time-ordered snapshots. For dynamic-topology homogeneous sequences,
        each decode step uses the target snapshot's ``edge_index``. Multiplex
        sequences use per-step relation banks from teacher targets.
    horizon : int
        Number of rollout steps (must be >= 1).
    start : int, optional
        Index of the initial snapshot. Default is ``0``.
    cache : object or None, optional
        Optional
        :class:`~koopman_graph.training.latent_cache.SequenceLatentCache`
        (or duck-typed object with ``.z``). When set, the origin uses
        ``cache.z[start]``; subsequent autoregressive latents are still
        produced by Koopman advance (not replaced by ``cache.z[t]``).
        Typed as ``Any`` so :mod:`koopman_graph.losses` does not import
        :mod:`koopman_graph.training`.
    origin_latent : Tensor or None, optional
        Pre-encoded origin latent. When set, skips encode / cache lookup
        (used by :func:`rollout_multi_start_loss` to share origins).

    Returns
    -------
    Tensor
        Scalar mean rollout reconstruction loss over ``horizon`` steps.

    Raises
    ------
    ValueError
        If ``horizon < 1``, ``start < 0``, or the sequence is too short.
    """
    if horizon < 1:
        msg = f"horizon must be >= 1, got {horizon}"
        raise ValueError(msg)
    if start < 0:
        msg = f"start must be >= 0, got {start}"
        raise ValueError(msg)
    if start + horizon >= sequence.num_timesteps:
        msg = (
            f"sequence too short for rollout from start={start} "
            f"with horizon={horizon} (num_timesteps={sequence.num_timesteps})"
        )
        raise ValueError(msg)

    if origin_latent is not None:
        z = origin_latent
    elif cache is not None:
        z = cache.z[start]
    else:
        initial = sequence[start]
        encode_at = getattr(model, "encode_at", None)
        z = encode_at(sequence, start) if callable(encode_at) else model.encode(initial)

    time_step = float(model.resolve_delta_t(None))
    targets = [sequence[start + step] for step in range(1, horizon + 1)]

    if isinstance(sequence, HeteroGraphSnapshotSequence):
        if sequence.has_controls:
            msg = (
                "hetero rollout loss requires an uncontrolled "
                "HeteroGraphSnapshotSequence"
            )
            raise ValueError(msg)
        decoder = model.decoder
        if not isinstance(decoder, RelGraphDecoder):
            msg = "hetero rollout loss requires RelGraphDecoder"
            raise TypeError(msg)
        koopman = model.koopman
        if isinstance(koopman, HeteroGraphKoopmanOperator):
            num_relations = int(koopman.num_relations)
        else:
            num_relations = int(decoder.num_relations)
        hetero_targets = [
            target for target in targets if isinstance(target, HeteroData)
        ]
        if len(hetero_targets) != len(targets):
            msg = "hetero rollout targets must all be HeteroData"
            raise TypeError(msg)
        node_types = tuple(decoder.node_types) if decoder.is_typed else None
        num_nodes_dict = sequence.num_nodes_dict if decoder.is_typed else None
        rollout = autoregressive_hetero_latent_rollout(
            model.koopman,
            _bind_hetero_decoder(decoder, num_nodes_dict),
            z,
            steps=horizon,
            topology_at=_relation_topology_at_from_targets(
                hetero_targets,
                num_relations=num_relations,
                node_types=node_types,
                edge_types=decoder.edge_types,
            ),
            control_at=None,
            delta_t_at=lambda step: resolve_pair_delta_t(
                sequence,
                start + step,
                default_time_step=time_step,
            ),
            default_delta_t=time_step,
            num_nodes_dict=num_nodes_dict,
        )
        total_loss = torch.zeros((), device=z.device)
        for step, (prediction, _, _) in enumerate(rollout):
            masks = (
                sequence.observation_mask_at(start + step + 1)
                if sequence.has_observation_masks
                else None
            )
            total_loss = total_loss + _hetero_rollout_step_loss(
                prediction,
                hetero_targets[step],
                node_types=node_types,
                masks=masks,
            )
        return total_loss / horizon

    decoder_fn = model.decoder
    if isinstance(model.decoder, HypergraphDecoder):
        hyperedge_index = sequence.hyperedge_index
        if hyperedge_index is None:
            msg = (
                "HypergraphDecoder rollout loss requires a hyperedge-carrying sequence"
            )
            raise ValueError(msg)
        decoder_fn = bind_hypergraph_decoder(
            model.decoder,
            hyperedge_index,
            sequence.hyperedge_weight,
        )
    origin = sequence[start]
    if recursive_training_enabled(model):
        head = getattr(model, "predicted_topology", None)
        if head is None:
            msg = "recursive graph-state training requires a predicted topology head"
            raise ValueError(msg)
        topology_at = make_recursive_topology_at(head, origin.edge_index)
    else:
        topology_at = snapshot_topology_at(targets)
    rollout = autoregressive_latent_rollout(
        model.koopman,
        decoder_fn,
        z,
        steps=horizon,
        topology_at=topology_at,
        control_at=(
            None
            if not sequence.has_controls
            else (lambda step: sequence.control_at(start + step))
        ),
        parameters_at=(
            None
            if getattr(sequence, "parameter_trajectory", None) is None
            else (lambda step: sequence.parameter_trajectory[start + step])
        ),
        delta_t_at=lambda step: resolve_pair_delta_t(
            sequence,
            start + step,
            default_time_step=time_step,
        ),
        default_delta_t=time_step,
        hyperedge_index=sequence.hyperedge_index,
        hyperedge_weight=sequence.hyperedge_weight,
    )

    total_loss = torch.zeros((), device=z.device)
    for step, (prediction, _, _) in enumerate(rollout):
        target = targets[step]
        node_mask = (
            sequence.loss_mask_at(start + step + 1)
            if isinstance(sequence, GraphSnapshotSequence)
            else (
                sequence.observation_mask_at(start + step + 1)
                if sequence.has_observation_masks
                else None
            )
        )
        if node_mask is None:
            step_loss = nn.functional.mse_loss(prediction, target.x)
        else:
            step_loss = masked_mse_loss(prediction, target.x, node_mask)
        total_loss = total_loss + step_loss
    return total_loss / horizon


def _encode_rollout_origin_latent(
    model: TrainableKoopmanModel,
    sequence: SnapshotSequence,
    start: int,
) -> Tensor:
    """Encode the rollout origin at ``start``.

    Parameters
    ----------
    model : TrainableKoopmanModel
        Model exposing ``encode_at`` or ``encode``.
    sequence : GraphSnapshotSequence or HeteroGraphSnapshotSequence
        Source trajectory.
    start : int
        Origin timestep index.

    Returns
    -------
    Tensor
        Latent features at the rollout origin.
    """
    encode_at = getattr(model, "encode_at", None)
    if callable(encode_at):
        return encode_at(sequence, start)
    return model.encode(sequence[start])


def rollout_multi_start_loss(
    model: TrainableKoopmanModel,
    sequence: SnapshotSequence,
    *,
    horizon: int,
    start_indices: Sequence[int],
    cache: Any | None = None,
) -> Tensor:
    """Average rollout reconstruction loss over multiple start snapshots.

    When ``cache`` is omitted, each distinct origin is encoded once and shared
    across duplicate starts; autoregressive decode remains per-origin.

    Parameters
    ----------
    model : :class:`~koopman_graph.protocols.TrainableKoopmanModel`
        Trainable model accepted by :func:`rollout_sequence_loss` (also uses
        ``parameters`` for device placement).
        :class:`~koopman_graph.model.GraphKoopmanModel` is the intended
        implementer.
    sequence : GraphSnapshotSequence or HeteroGraphSnapshotSequence
        Time-ordered snapshots.
    horizon : int
        Number of rollout steps (must be >= 1).
    start_indices : sequence of int
        Zero-based origin indices for each rollout.
    cache : object or None, optional
        Optional shared latent cache (see :func:`rollout_sequence_loss`).

    Returns
    -------
    Tensor
        Scalar mean rollout loss across origins.

    Raises
    ------
    ValueError
        If ``start_indices`` is empty or any origin is invalid.
    """
    if not start_indices:
        msg = "start_indices must contain at least one origin"
        raise ValueError(msg)

    device = next(model.parameters()).device
    origin_latents: dict[int, Tensor] | None = None
    if cache is None:
        origin_latents = {}
        for start in start_indices:
            if start not in origin_latents:
                origin_latents[start] = _encode_rollout_origin_latent(
                    model, sequence, start
                )

    total_loss = torch.zeros((), device=device)
    for start in start_indices:
        total_loss = total_loss + rollout_sequence_loss(
            model,
            sequence,
            horizon=horizon,
            start=start,
            cache=cache,
            origin_latent=(None if origin_latents is None else origin_latents[start]),
        )
    return total_loss / len(start_indices)
