"""Pair and sequence reconstruction / consistency objective composition.

Owns one-step reconstruction and forward/backward consistency helpers used by
:func:`~koopman_graph.training.objectives.compute_training_loss`. Shared pair
utilities used by :mod:`~koopman_graph.training.extra_objectives` are documented
non-private names (no cross-module leading-underscore imports).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
from torch_geometric.data import Data, HeteroData

from koopman_graph.data import (
    GraphSnapshotSequence,
    HeteroGraphSnapshotSequence,
    SnapshotSequence,
    resolve_pair_delta_t,
)
from koopman_graph.graph_utils import (
    propagate_latent,
    snapshot_edge_weight,
    snapshot_hyperedge_index,
    snapshot_hyperedge_weight,
)
from koopman_graph.losses import (
    BackwardConsistencyLoss,
    ForwardConsistencyLoss,
    masked_mse_loss,
)
from koopman_graph.nn import HypergraphDecoder, RelGraphDecoder, RelGraphEncoder
from koopman_graph.nn.heterogeneous import (
    resolve_multiplex_relation_inputs,
    resolve_typed_relation_inputs,
)
from koopman_graph.operators import (
    GlobalLocalKoopmanOperator,
    GraphKoopmanOperator,
    HeteroGraphKoopmanOperator,
    HypergraphKoopmanOperator,
)
from koopman_graph.operators.global_local import stack_latent_window
from koopman_graph.protocols import TrainableKoopmanModel
from koopman_graph.training.latent_cache import latent_window_from_cache
from koopman_graph.training.timestep_encode import encode_at_timestep

if TYPE_CHECKING:
    from koopman_graph.training.latent_cache import SequenceLatentCache

_FORWARD_CONSISTENCY_LOSS = ForwardConsistencyLoss()
_BACKWARD_CONSISTENCY_LOSS = BackwardConsistencyLoss()

PairLossFn = Callable[[TrainableKoopmanModel, SnapshotSequence, int], Tensor]


def _is_hetero_sequence(sequence: SnapshotSequence) -> bool:
    """Return whether ``sequence`` is a multiplex hetero container.

    Parameters
    ----------
    sequence : GraphSnapshotSequence or HeteroGraphSnapshotSequence
        Candidate training sequence.

    Returns
    -------
    bool
        ``True`` when ``sequence`` is a
        :class:`~koopman_graph.data.HeteroGraphSnapshotSequence`.
    """
    return isinstance(sequence, HeteroGraphSnapshotSequence)


def multiplex_node_features(snapshot: HeteroData) -> Tensor:
    """Return the sole node-type feature matrix from a multiplex snapshot.

    Parameters
    ----------
    snapshot : HeteroData
        Multiplex hetero snapshot (exactly one node type).

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
            "multiplex training requires HeteroData with exactly one node "
            f"type; got {node_types!r}"
        )
        raise ValueError(msg)
    features = snapshot[node_types[0]].x
    if features is None:
        msg = f"HeteroData node type {node_types[0]!r} is missing feature matrix x"
        raise ValueError(msg)
    return features


def _relation_banks_from_snapshot(
    snapshot: HeteroData,
    *,
    num_relations: int,
) -> tuple[list[Tensor], list[Tensor | None]]:
    """Resolve ordered relation banks from a multiplex ``HeteroData`` snapshot.

    Parameters
    ----------
    snapshot : HeteroData
        Multiplex snapshot supplying edge types.
    num_relations : int
        Expected relation-bank length ``|R|``.

    Returns
    -------
    tuple of list
        Ordered ``edge_indices`` and optional ``edge_weights`` banks.
    """
    _, edge_indices, edge_weights = resolve_multiplex_relation_inputs(
        snapshot,
        num_relations=num_relations,
    )
    return edge_indices, edge_weights


def _hetero_num_relations(model: TrainableKoopmanModel) -> int:
    """Return ``|R|`` from the active hetero operator or RelGraph decoder.

    Parameters
    ----------
    model : TrainableKoopmanModel
        Model exposing a hetero Koopman operator or RelGraph decoder.

    Returns
    -------
    int
        Number of relations ``|R|``.

    Raises
    ------
    TypeError
        If neither a hetero operator nor RelGraph decoder is present.
    """
    koopman = model.koopman
    if isinstance(koopman, HeteroGraphKoopmanOperator):
        return int(koopman.num_relations)
    decoder = getattr(model, "decoder", None)
    if isinstance(decoder, RelGraphDecoder):
        return int(decoder.num_relations)
    msg = (
        "hetero training requires HeteroGraphKoopmanOperator or RelGraphDecoder "
        "to resolve num_relations"
    )
    raise TypeError(msg)


def typed_hetero_node_types(model: TrainableKoopmanModel) -> tuple[str, ...] | None:
    """Return ordered node types when ``model`` uses the typed hetero path.

    Parameters
    ----------
    model : TrainableKoopmanModel
        Model whose RelGraph peers are inspected.

    Returns
    -------
    tuple of str or None
        Stacking order of node types, or ``None`` for the multiplex path.
    """
    for module in (getattr(model, "encoder", None), getattr(model, "decoder", None)):
        if (
            isinstance(module, (RelGraphEncoder, RelGraphDecoder))
            and module.is_typed
            and module.node_types is not None
        ):
            return tuple(module.node_types)
    return None


def _typed_hetero_edge_types(
    model: TrainableKoopmanModel,
) -> tuple[tuple[str, str, str], ...] | None:
    """Return the declared relation-bank order for a typed hetero model.

    Parameters
    ----------
    model : TrainableKoopmanModel
        Model whose encoder / decoder / operator declare relation order.

    Returns
    -------
    tuple of tuple of str or None
        Ordered ``(src, rel, dst)`` triples, or ``None`` when unavailable.
    """
    for module in (getattr(model, "encoder", None), getattr(model, "decoder", None)):
        declared = getattr(module, "edge_types", None)
        if declared is not None:
            return tuple(tuple(triple) for triple in declared)  # type: ignore[misc]
    koopman = model.koopman
    if isinstance(koopman, HeteroGraphKoopmanOperator):
        return tuple(tuple(triple) for triple in koopman.edge_types)  # type: ignore[misc]
    return None


def typed_node_features(
    snapshot: HeteroData,
    node_types: Sequence[str],
) -> dict[str, Tensor]:
    """Return per-type feature matrices from a typed hetero snapshot.

    Parameters
    ----------
    snapshot : HeteroData
        Typed hetero snapshot.
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
    available = set(snapshot.node_types)
    for name in node_types:
        if name not in available:
            msg = (
                f"HeteroData snapshot is missing node type {name!r}; "
                f"available types are {sorted(available)!r}"
            )
            raise ValueError(msg)
        block = snapshot[name].x
        if block is None:
            msg = f"HeteroData node type {name!r} is missing feature matrix x"
            raise ValueError(msg)
        features[name] = block
    return features


def _hetero_relation_banks(
    model: TrainableKoopmanModel,
    snapshot: HeteroData,
    *,
    num_relations: int,
) -> tuple[list[Tensor], list[Tensor | None], dict[str, int] | None]:
    """Resolve relation banks (and typed counts) from a hetero snapshot.

    Parameters
    ----------
    model : TrainableKoopmanModel
        Model providing typed metadata.
    snapshot : HeteroData
        Snapshot supplying edge stores.
    num_relations : int
        Expected relation-bank length ``|R|``.

    Returns
    -------
    tuple
        ``(edge_indices, edge_weights, num_nodes_dict)`` with
        ``num_nodes_dict`` set only for the typed path.
    """
    node_types = typed_hetero_node_types(model)
    if node_types is None:
        edge_indices, edge_weights = _relation_banks_from_snapshot(
            snapshot,
            num_relations=num_relations,
        )
        return edge_indices, edge_weights, None
    _, edge_indices, edge_weights, num_nodes_dict = resolve_typed_relation_inputs(
        snapshot,
        node_types=node_types,
        edge_types=_typed_hetero_edge_types(model),
        num_relations=num_relations,
    )
    return edge_indices, edge_weights, num_nodes_dict


def stack_typed_masks(
    masks: Mapping[str, Tensor],
    node_types: Sequence[str],
) -> Tensor:
    """Stack per-type boolean node masks into the shared stacked order.

    Parameters
    ----------
    masks : mapping of str to Tensor
        Boolean masks with shape ``(N_τ,)`` per node type.
    node_types : sequence of str
        Stacking order.

    Returns
    -------
    Tensor
        Boolean mask with shape ``(N,)`` where ``N = Σ_τ N_τ``.

    Raises
    ------
    ValueError
        If a node type is missing from ``masks``.
    """
    blocks: list[Tensor] = []
    for name in node_types:
        if name not in masks:
            msg = f"observation masks are missing node type {name!r}"
            raise ValueError(msg)
        blocks.append(masks[name].reshape(-1))
    return torch.cat(blocks, dim=0)


def _hetero_pair_mask(
    sequence: SnapshotSequence,
    timestep: int,
    node_types: Sequence[str] | None,
) -> Tensor | None:
    """Return the stacked latent pair mask for a hetero sequence.

    Parameters
    ----------
    sequence : HeteroGraphSnapshotSequence
        Hetero training sequence.
    timestep : int
        Index of the source snapshot in the pair.
    node_types : sequence of str or None
        Typed stacking order, or ``None`` for the multiplex path.

    Returns
    -------
    Tensor or None
        Boolean mask over stacked rows, or ``None`` when masks are absent.
    """
    if not sequence.has_observation_masks:
        return None
    masks = sequence.pair_observation_mask(timestep)
    assert isinstance(masks, Mapping)
    order = tuple(masks) if node_types is None else tuple(node_types)
    return stack_typed_masks(masks, order)


def _hetero_target_masks(
    sequence: SnapshotSequence,
    timestep: int,
) -> dict[str, Tensor] | None:
    """Return per-type target masks at ``timestep`` for a hetero sequence.

    Parameters
    ----------
    sequence : HeteroGraphSnapshotSequence
        Hetero training sequence.
    timestep : int
        Snapshot index of the prediction target.

    Returns
    -------
    dict of str to Tensor or None
        Per-type boolean masks, or ``None`` when masks are absent.
    """
    if not sequence.has_observation_masks:
        return None
    masks = sequence.observation_mask_at(timestep)
    assert isinstance(masks, Mapping)
    return dict(masks)


def _hetero_prediction_loss(
    prediction: Tensor | Mapping[str, Tensor],
    target_snapshot: HeteroData,
    *,
    node_types: Sequence[str] | None,
    target_masks: Mapping[str, Tensor] | None,
) -> Tensor:
    """Average reconstruction MSE for a multiplex or typed prediction.

    Parameters
    ----------
    prediction : Tensor or mapping of str to Tensor
        Decoded features (per-type mapping for typed decoders).
    target_snapshot : HeteroData
        Target snapshot supplying ``x`` per node type.
    node_types : sequence of str or None
        Typed stacking order, or ``None`` for the multiplex path.
    target_masks : mapping of str to Tensor or None
        Optional per-type observation masks at the target snapshot.

    Returns
    -------
    Tensor
        Scalar loss (mean over node types for the typed path).
    """
    if node_types is None or isinstance(prediction, Tensor):
        target = multiplex_node_features(target_snapshot)
        assert isinstance(prediction, Tensor)
        if target_masks is None:
            return nn.functional.mse_loss(prediction, target)
        sole = tuple(target_snapshot.node_types)[0]
        return masked_mse_loss(prediction, target, target_masks[sole])
    targets = typed_node_features(target_snapshot, node_types)
    total = None
    for name in node_types:
        block = (
            nn.functional.mse_loss(prediction[name], targets[name])
            if target_masks is None
            else masked_mse_loss(prediction[name], targets[name], target_masks[name])
        )
        total = block if total is None else total + block
    assert total is not None
    return total / len(tuple(node_types))


def teacher_forced_latent_window(
    model: TrainableKoopmanModel,
    sequence: GraphSnapshotSequence,
    timestep: int,
) -> Tensor | None:
    """Build a teacher-forced latent window for global/local operators.

    Encodes snapshots ``[t-w+1, …, t]`` (left-padded by the earliest available
    encoding) when ``model.koopman`` is a
    :class:`~koopman_graph.operators.GlobalLocalKoopmanOperator`. Returns
    ``None`` for other operator kinds.

    Parameters
    ----------
    model : TrainableKoopmanModel
        Trainable model with an encoder.
    sequence : GraphSnapshotSequence
        Source trajectory.
    timestep : int
        Index ``t`` of the state being advanced.

    Returns
    -------
    Tensor or None
        Window with shape ``(w, ..., d)``, or ``None``.
    """
    koopman = model.koopman
    if not isinstance(koopman, GlobalLocalKoopmanOperator):
        return None
    window = koopman.local_window
    start = max(0, timestep - window + 1)
    frames = [
        _encode_at(model, sequence, index) for index in range(start, timestep + 1)
    ]
    return stack_latent_window(frames[:-1], window=window, current=frames[-1])


def _pair_latents(
    model: TrainableKoopmanModel,
    sequence: SnapshotSequence,
    timestep: int,
    *,
    cache: SequenceLatentCache | None,
) -> tuple[Tensor, Tensor]:
    """Return ``(z_t, z_{t+1})`` from ``cache`` or by encoding.

    Parameters
    ----------
    model : TrainableKoopmanModel
        Model whose encoder produces latents when ``cache`` is absent.
    sequence : GraphSnapshotSequence or HeteroGraphSnapshotSequence
        Snapshot window aligned with ``cache`` when provided.
    timestep : int
        Index ``t`` of the first latent in the pair.
    cache : SequenceLatentCache or None
        Shared teacher-forced latents, or ``None`` to encode on demand.

    Returns
    -------
    tuple of Tensor
        Latents ``(z_t, z_{t+1})``.
    """
    if cache is not None:
        return cache.z[timestep], cache.z[timestep + 1]
    return (
        encode_at_timestep(model, sequence, timestep),
        encode_at_timestep(model, sequence, timestep + 1),
    )


def _pair_latent_window(
    model: TrainableKoopmanModel,
    sequence: SnapshotSequence,
    timestep: int,
    *,
    cache: SequenceLatentCache | None,
) -> Tensor | None:
    """Return a teacher-forced global/local window from cache or encodes.

    Parameters
    ----------
    model : TrainableKoopmanModel
        Model used when ``cache`` is ``None`` or to detect global/local
        operators.
    sequence : GraphSnapshotSequence or HeteroGraphSnapshotSequence
        Snapshot window aligned with ``cache`` when provided.
    timestep : int
        Index ``t`` of the state being advanced.
    cache : SequenceLatentCache or None
        Shared teacher-forced latents, or ``None`` to encode on demand.

    Returns
    -------
    Tensor or None
        Stacked latent window for
        :class:`~koopman_graph.operators.GlobalLocalKoopmanOperator`, else
        ``None``.
    """
    if cache is None:
        return teacher_forced_latent_window(model, sequence, timestep)
    koopman = model.koopman
    if not isinstance(koopman, GlobalLocalKoopmanOperator):
        return None
    return latent_window_from_cache(cache, timestep, koopman.local_window)


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


def pair_control(sequence: SnapshotSequence, timestep: int) -> Tensor | None:
    """Return the control input for transition ``timestep -> timestep + 1``.

    Parameters
    ----------
    sequence : GraphSnapshotSequence or HeteroGraphSnapshotSequence
        Snapshot sequence that may carry controls.
    timestep : int
        Index of the source snapshot in the transition pair.

    Returns
    -------
    Tensor or None
        Control tensor when present, otherwise ``None``.

    Raises
    ------
    ValueError
        If a hetero sequence carries controls (``control_at`` is not
        implemented for multiplex containers yet).
    """
    if not sequence.has_controls:
        return None
    control_at = getattr(sequence, "control_at", None)
    if not callable(control_at):
        msg = (
            "controlled HeteroGraphSnapshotSequence training is unsupported; "
            "omit control_inputs for multiplex fit"
        )
        raise ValueError(msg)
    return control_at(timestep)


def mean_pair_sequence_loss(
    model: TrainableKoopmanModel,
    sequence: SnapshotSequence,
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


# Backward-compatible alias for in-module call sites / tests.
_encode_at = encode_at_timestep


def one_step_loss(
    model: TrainableKoopmanModel,
    snapshot_t: Data | HeteroData,
    snapshot_t1: Data | HeteroData,
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
    snapshot_t : Data or HeteroData
        Graph snapshot at time ``t``.
    snapshot_t1 : Data or HeteroData
        Graph snapshot at time ``t+1`` (prediction target).
    control : Tensor or None, optional
        Control input driving the transition from ``t`` to ``t+1``.
    delta_t : float, Tensor, or None, optional
        Integration interval for continuous-time models.
    target_mask : Tensor or None, optional
        Boolean node mask with shape ``(num_nodes,)``. When provided, the loss
        averages only over observed nodes at the target snapshot.
        Homogeneous-only today.

    Returns
    -------
    Tensor
        Scalar mean-squared error loss.
    """
    prediction = model(snapshot_t, control=control, delta_t=delta_t)
    if isinstance(snapshot_t1, HeteroData):
        if target_mask is not None:
            msg = "target_mask is unsupported for HeteroData one-step loss"
            raise ValueError(msg)
        return _hetero_prediction_loss(
            prediction,
            snapshot_t1,
            node_types=typed_hetero_node_types(model),
            target_masks=None,
        )
    target = snapshot_t1.x
    if target_mask is None:
        return nn.functional.mse_loss(prediction, target)
    return masked_mse_loss(prediction, target, target_mask)


def _forward_consistency_pair(
    model: TrainableKoopmanModel,
    sequence: SnapshotSequence,
    timestep: int,
    *,
    cache: SequenceLatentCache | None = None,
) -> Tensor:
    """Compute forward consistency loss for one consecutive snapshot pair.

    Parameters
    ----------
    model : TrainableKoopmanModel
        Model satisfying :class:`~koopman_graph.protocols.TrainableKoopmanModel`.
    sequence : GraphSnapshotSequence or HeteroGraphSnapshotSequence
        Snapshot sequence containing the consecutive pair.
    timestep : int
        Index of the source snapshot ``t`` in the pair ``(t, t+1)``.
    cache : SequenceLatentCache or None, optional
        Shared teacher-forced latents. When set, skips per-pair encoding.

    Returns
    -------
    Tensor
        Scalar forward consistency loss for the pair.
    """
    snapshot_t1 = sequence[timestep + 1]
    z_t, z_t1 = _pair_latents(model, sequence, timestep, cache=cache)
    default_delta_t = model_default_delta_t(model)
    delta_t = resolve_pair_delta_t(
        sequence,
        timestep,
        default_time_step=default_delta_t,
    )
    control = pair_control(sequence, timestep)
    if _is_hetero_sequence(sequence):
        assert isinstance(snapshot_t1, HeteroData)
        node_types = typed_hetero_node_types(model)
        edge_indices, edge_weights, num_nodes_dict = _hetero_relation_banks(
            model,
            snapshot_t1,
            num_relations=_hetero_num_relations(model),
        )
        return _FORWARD_CONSISTENCY_LOSS(
            z_t,
            z_t1,
            model.koopman,
            control=control,
            delta_t=delta_t,
            default_delta_t=default_delta_t,
            mask=_hetero_pair_mask(sequence, timestep, node_types),
            edge_indices=edge_indices,
            edge_weights=edge_weights,
            num_nodes_dict=num_nodes_dict,
            latent_window=_pair_latent_window(model, sequence, timestep, cache=cache),
        )

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
        hyperedge_index=snapshot_hyperedge_index(snapshot_t1),
        hyperedge_weight=snapshot_hyperedge_weight(snapshot_t1),
        latent_window=_pair_latent_window(model, sequence, timestep, cache=cache),
    )


def _backward_consistency_pair(
    model: TrainableKoopmanModel,
    sequence: SnapshotSequence,
    timestep: int,
    *,
    inverse_matrix: Tensor | None = None,
    cache: SequenceLatentCache | None = None,
) -> Tensor:
    """Compute backward consistency loss for one consecutive snapshot pair.

    Parameters
    ----------
    model : TrainableKoopmanModel
        Model satisfying :class:`~koopman_graph.protocols.TrainableKoopmanModel`.
    sequence : GraphSnapshotSequence or HeteroGraphSnapshotSequence
        Snapshot sequence containing the consecutive pair.
    timestep : int
        Index of the source snapshot ``t`` in the pair ``(t, t+1)``.
    inverse_matrix : Tensor or None, optional
        Precomputed dense inverse matrix reused across pair evaluations.
    cache : SequenceLatentCache or None, optional
        Shared teacher-forced latents. When set, skips per-pair encoding.

    Returns
    -------
    Tensor
        Scalar backward consistency loss for the pair.

    Raises
    ------
    ValueError
        If ``sequence`` is multiplex hetero (unsupported in this release).
    """
    if _is_hetero_sequence(sequence):
        msg = (
            "backward consistency is unsupported for HeteroGraphSnapshotSequence "
            "/ koopman='hetero_graph'; set loss_weights.backward=0"
        )
        raise ValueError(msg)
    snapshot_t1 = sequence[timestep + 1]
    z_t, z_t1 = _pair_latents(model, sequence, timestep, cache=cache)
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
        hyperedge_index=snapshot_hyperedge_index(snapshot_t1),
        hyperedge_weight=snapshot_hyperedge_weight(snapshot_t1),
    )


def one_step_prediction(
    model: TrainableKoopmanModel,
    sequence: SnapshotSequence,
    timestep: int,
    *,
    cache: SequenceLatentCache | None = None,
) -> Tensor | dict[str, Tensor]:
    """Decode the one-step forecast for pair ``(timestep, timestep + 1)``.

    When ``cache`` is set (or delay / global-local requires it), uses
    encode→advance→decode with cached or freshly encoded ``z_t``. Otherwise
    falls back to ``model(source, …)``.

    Parameters
    ----------
    model : TrainableKoopmanModel
        Model implementing encode / advance / decode (or a single-step forward).
    sequence : GraphSnapshotSequence or HeteroGraphSnapshotSequence
        Snapshot sequence that may carry control inputs.
    timestep : int
        Index of the source snapshot in the transition pair.
    cache : SequenceLatentCache or None, optional
        Shared teacher-forced latents. When set, always uses the
        encode→advance→decode path so reconstruction reuses cached ``z_t``.

    Returns
    -------
    Tensor or dict of str to Tensor
        Decoded node features at the predicted next step (one tensor per node
        type for typed hetero models).
    """
    if _is_hetero_sequence(sequence):
        snapshot_t = sequence[timestep]
        snapshot_t1 = sequence[timestep + 1]
        assert isinstance(snapshot_t, HeteroData)
        assert isinstance(snapshot_t1, HeteroData)
        if cache is not None:
            z = cache.z[timestep]
        else:
            z = encode_at_timestep(model, sequence, timestep)
        default_delta_t = model_default_delta_t(model)
        delta_t = resolve_pair_delta_t(
            sequence,
            timestep,
            default_time_step=default_delta_t,
        )
        num_relations = _hetero_num_relations(model)
        edge_indices_t1, edge_weights_t1, num_nodes_dict = _hetero_relation_banks(
            model,
            snapshot_t1,
            num_relations=num_relations,
        )
        edge_indices_t, edge_weights_t, _ = _hetero_relation_banks(
            model,
            snapshot_t,
            num_relations=num_relations,
        )
        z_next = propagate_latent(
            model.koopman,
            z,
            control=pair_control(sequence, timestep),
            delta_t=delta_t,
            default_delta_t=default_delta_t,
            edge_indices=edge_indices_t1,
            edge_weights=edge_weights_t1,
            num_nodes_dict=num_nodes_dict,
        )
        if not isinstance(model.decoder, RelGraphDecoder):
            msg = "hetero one-step prediction requires RelGraphDecoder"
            raise TypeError(msg)
        if model.decoder.is_typed:
            return model.decoder(
                z_next,
                edge_indices_t,
                edge_weights_t,
                num_nodes_dict=num_nodes_dict,
            )
        return model.decoder(z_next, edge_indices_t, edge_weights_t)

    n_delays = int(getattr(model, "n_delays", 1))
    uses_global_local = isinstance(model.koopman, GlobalLocalKoopmanOperator)
    use_encode_path = (
        cache is not None
        or uses_global_local
        or (n_delays > 1 and callable(getattr(model, "encode_at", None)))
    )
    if use_encode_path:
        snapshot_t = sequence[timestep]
        snapshot_t1 = sequence[timestep + 1]
        if cache is not None:
            z = cache.z[timestep]
        else:
            z = encode_at_timestep(model, sequence, timestep)
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
            hyperedge_index=snapshot_hyperedge_index(snapshot_t1),
            hyperedge_weight=snapshot_hyperedge_weight(snapshot_t1),
            latent_window=_pair_latent_window(model, sequence, timestep, cache=cache),
        )
        if isinstance(model.decoder, HypergraphDecoder):
            hyperedge_index = snapshot_hyperedge_index(snapshot_t)
            if hyperedge_index is None:
                msg = "HypergraphDecoder requires hyperedge_index on training snapshots"
                raise ValueError(msg)
            return model.decoder(
                z_next,
                hyperedge_index,
                snapshot_hyperedge_weight(snapshot_t),
            )
        return model.decoder(
            z_next,
            snapshot_t.edge_index,
            getattr(snapshot_t, "edge_weight", None),
        )

    return model(
        sequence[timestep],
        control=pair_control(sequence, timestep),
        delta_t=resolve_pair_delta_t(
            sequence,
            timestep,
            default_time_step=model_default_delta_t(model),
        ),
    )


def one_step_predictions(
    model: TrainableKoopmanModel,
    sequence: SnapshotSequence,
    *,
    cache: SequenceLatentCache,
) -> list[Tensor | dict[str, Tensor]]:
    """Build one-step decoded forecasts for every consecutive pair.

    Parameters
    ----------
    model : TrainableKoopmanModel
        Model used for advance / decode.
    sequence : GraphSnapshotSequence or HeteroGraphSnapshotSequence
        Time-ordered snapshots with at least two timesteps.
    cache : SequenceLatentCache
        Shared teacher-forced latents (required so each pair reuses ``z_t``).

    Returns
    -------
    list of Tensor or dict of str to Tensor
        Length ``sequence.num_timesteps - 1``; index ``t`` is the forecast
        from timestep ``t`` to ``t + 1``.

    Raises
    ------
    ValueError
        If ``sequence`` contains fewer than two snapshots.
    """
    if sequence.num_timesteps < 2:
        msg = "snapshot sequence must contain at least 2 snapshots for training"
        raise ValueError(msg)
    return [
        one_step_prediction(model, sequence, timestep, cache=cache)
        for timestep in range(sequence.num_timesteps - 1)
    ]


def _reconstruction_from_predictions(
    model: TrainableKoopmanModel,
    sequence: SnapshotSequence,
    predictions: Sequence[Tensor | Mapping[str, Tensor]],
) -> Tensor:
    """Average masked MSE between shared predictions and target snapshots.

    Parameters
    ----------
    model : TrainableKoopmanModel
        Model used only for device placement of the accumulator.
    sequence : GraphSnapshotSequence or HeteroGraphSnapshotSequence
        Target snapshots and optional observation masks.
    predictions : sequence of Tensor
        One-step decoded forecasts with length ``num_timesteps - 1``.

    Returns
    -------
    Tensor
        Scalar mean reconstruction loss over consecutive pairs.

    Raises
    ------
    ValueError
        If ``sequence`` has fewer than two snapshots or ``predictions`` length
        does not match the number of pairs.
    """
    if sequence.num_timesteps < 2:
        msg = "snapshot sequence must contain at least 2 snapshots for training"
        raise ValueError(msg)
    num_pairs = sequence.num_timesteps - 1
    if len(predictions) != num_pairs:
        msg = (
            f"predictions length ({len(predictions)}) must equal "
            f"num_pairs ({num_pairs})"
        )
        raise ValueError(msg)

    total_loss = torch.zeros((), device=next(model.parameters()).device)
    node_types = (
        typed_hetero_node_types(model) if _is_hetero_sequence(sequence) else None
    )
    for timestep in range(num_pairs):
        target_snapshot = sequence[timestep + 1]
        prediction = predictions[timestep]
        if isinstance(target_snapshot, HeteroData):
            total_loss = total_loss + _hetero_prediction_loss(
                prediction,
                target_snapshot,
                node_types=node_types,
                target_masks=_hetero_target_masks(sequence, timestep + 1),
            )
            continue
        assert isinstance(prediction, Tensor)
        target = target_snapshot.x
        target_mask = (
            sequence.observation_mask_at(timestep + 1)
            if sequence.has_observation_masks
            else None
        )
        if target_mask is None:
            total_loss = total_loss + nn.functional.mse_loss(prediction, target)
        else:
            total_loss = total_loss + masked_mse_loss(prediction, target, target_mask)
    return total_loss / num_pairs


def _one_step_pair(
    model: TrainableKoopmanModel,
    sequence: SnapshotSequence,
    timestep: int,
    *,
    cache: SequenceLatentCache | None = None,
) -> Tensor:
    """Compute one-step loss for snapshot pair ``(timestep, timestep + 1)``.

    Parameters
    ----------
    model : TrainableKoopmanModel
        Model implementing a single-step forward pass.
    sequence : GraphSnapshotSequence or HeteroGraphSnapshotSequence
        Snapshot sequence that may carry control inputs.
    timestep : int
        Index of the source snapshot in the transition pair.
    cache : SequenceLatentCache or None, optional
        Shared teacher-forced latents. When set, always uses the
        encode→advance→decode path so reconstruction reuses cached ``z_t``.

    Returns
    -------
    Tensor
        Scalar one-step reconstruction loss.
    """
    if _is_hetero_sequence(sequence):
        target_masks = _hetero_target_masks(sequence, timestep + 1)
        if cache is None and target_masks is None:
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
            )
        prediction = one_step_prediction(model, sequence, timestep, cache=cache)
        return _hetero_prediction_loss(
            prediction,
            sequence[timestep + 1],
            node_types=typed_hetero_node_types(model),
            target_masks=target_masks,
        )

    target_mask = None
    if sequence.has_observation_masks:
        target_mask = sequence.observation_mask_at(timestep + 1)

    # Without a cache on the plain one-step path, keep ``one_step_loss`` so
    # deep-import monkeypatches of that helper remain effective.
    n_delays = int(getattr(model, "n_delays", 1))
    uses_global_local = isinstance(model.koopman, GlobalLocalKoopmanOperator)
    use_encode_path = (
        cache is not None
        or uses_global_local
        or (n_delays > 1 and callable(getattr(model, "encode_at", None)))
    )
    if not use_encode_path:
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

    prediction = one_step_prediction(model, sequence, timestep, cache=cache)
    target = sequence[timestep + 1].x
    if target_mask is None:
        return nn.functional.mse_loss(prediction, target)
    return masked_mse_loss(prediction, target, target_mask)


def compute_sequence_loss(
    model: TrainableKoopmanModel,
    sequence: SnapshotSequence,
    *,
    cache: SequenceLatentCache | None = None,
    predictions: Sequence[Tensor] | None = None,
) -> Tensor:
    """Average one-step prediction loss over consecutive snapshot pairs.

    Parameters
    ----------
    model : TrainableKoopmanModel
        Model implementing a single-step forward pass.
    sequence : :class:`~koopman_graph.data.GraphSnapshotSequence`
        Time-ordered snapshots with at least two timesteps.
    cache : SequenceLatentCache or None, optional
        Shared teacher-forced latents. When omitted, encodes per pair as before.
    predictions : sequence of Tensor or None, optional
        Precomputed one-step decoded forecasts (length ``T - 1``). When set,
        reconstruction MSE uses these tensors and skips a second decode.

    Returns
    -------
    Tensor
        Scalar average loss over all consecutive pairs.

    Raises
    ------
    ValueError
        If ``sequence`` contains fewer than two snapshots.
    """
    if predictions is not None:
        return _reconstruction_from_predictions(model, sequence, predictions)
    if cache is None:
        return mean_pair_sequence_loss(model, sequence, _one_step_pair)
    return mean_pair_sequence_loss(
        model,
        sequence,
        lambda m, s, t: _one_step_pair(m, s, t, cache=cache),
    )


def compute_forward_consistency_sequence_loss(
    model: TrainableKoopmanModel,
    sequence: SnapshotSequence,
    *,
    cache: SequenceLatentCache | None = None,
) -> Tensor:
    """Average forward consistency loss over consecutive snapshot pairs.

    Parameters
    ----------
    model : TrainableKoopmanModel
        Model satisfying :class:`~koopman_graph.protocols.TrainableKoopmanModel`.
    sequence : GraphSnapshotSequence or HeteroGraphSnapshotSequence
        Time-ordered snapshots with at least two timesteps.
    cache : SequenceLatentCache or None, optional
        Shared teacher-forced latents. When omitted, encodes per pair as before.

    Returns
    -------
    Tensor
        Scalar average forward consistency loss.

    Raises
    ------
    ValueError
        If ``sequence`` contains fewer than two snapshots.
    """
    if cache is None:
        return mean_pair_sequence_loss(
            model,
            sequence,
            _forward_consistency_pair,
        )
    return mean_pair_sequence_loss(
        model,
        sequence,
        lambda m, s, t: _forward_consistency_pair(m, s, t, cache=cache),
    )


def topologies_equal(
    edge_index_a: Tensor,
    edge_weight_a: Tensor | None,
    edge_index_b: Tensor,
    edge_weight_b: Tensor | None,
    *,
    hyperedge_index_a: Tensor | None = None,
    hyperedge_weight_a: Tensor | None = None,
    hyperedge_index_b: Tensor | None = None,
    hyperedge_weight_b: Tensor | None = None,
) -> bool:
    """Return whether two topology payloads are numerically identical.

    Used as the evaluation-scoped fingerprint for dense networked inverse
    reuse: cloned tensors with equal values share a cache entry even when
    ``data_ptr`` differs. In-place edits that keep pointers but change values
    are detected by content comparison on the next lookup.

    Parameters
    ----------
    edge_index_a, edge_index_b : Tensor
        Pairwise COO edge indices.
    edge_weight_a, edge_weight_b : Tensor or None
        Optional pairwise edge weights.
    hyperedge_index_a, hyperedge_index_b : Tensor or None, optional
        Optional bipartite hyperedge incidence (hypergraph operators).
    hyperedge_weight_a, hyperedge_weight_b : Tensor or None, optional
        Optional hyperedge weights.

    Returns
    -------
    bool
        ``True`` when pairwise (and hyperedge, when present) payloads match.

    Notes
    -----
    Fingerprint is **content equality** (``torch.equal`` / ``allclose``), not
    storage-pointer identity. Bilinear / pair-local self maps must not use a
    shared inverse across pairs even when topology matches.
    """
    if not torch.equal(edge_index_a, edge_index_b):
        return False
    if (edge_weight_a is None) != (edge_weight_b is None):
        return False
    if edge_weight_a is not None:
        assert edge_weight_b is not None
        if not torch.allclose(edge_weight_a, edge_weight_b, equal_nan=True):
            return False
    if (hyperedge_index_a is None) != (hyperedge_index_b is None):
        return False
    if hyperedge_index_a is not None:
        assert hyperedge_index_b is not None
        if not torch.equal(hyperedge_index_a, hyperedge_index_b):
            return False
    if (hyperedge_weight_a is None) != (hyperedge_weight_b is None):
        return False
    if hyperedge_weight_a is not None:
        assert hyperedge_weight_b is not None
        if not torch.allclose(hyperedge_weight_a, hyperedge_weight_b, equal_nan=True):
            return False
    return True


def _dense_networked_inverse_for_snapshot(
    koopman: GraphKoopmanOperator | HypergraphKoopmanOperator,
    snapshot: Data,
) -> Tensor | None:
    """Build ``dense_effective_inverse`` for a pair-target snapshot topology.

    Parameters
    ----------
    koopman : GraphKoopmanOperator or HypergraphKoopmanOperator
        Dense networked operator (``sparsity="dense"``).
    snapshot : Data
        Target snapshot (``sequence[t + 1]``) supplying topology.

    Returns
    -------
    Tensor or None
        Dense inverse, or ``None`` when hypergraph incidence is missing.
    """
    num_nodes = int(snapshot.x.shape[0])
    if isinstance(koopman, GraphKoopmanOperator):
        return koopman.dense_effective_inverse(
            snapshot.edge_index,
            num_nodes,
            edge_weight=snapshot_edge_weight(snapshot),
        )
    hyperedge_index = snapshot_hyperedge_index(snapshot)
    if hyperedge_index is None:
        return None
    return koopman.dense_effective_inverse(
        hyperedge_index,
        num_nodes,
        hyperedge_weight=snapshot_hyperedge_weight(snapshot),
    )


def _lookup_networked_inverse(
    koopman: GraphKoopmanOperator | HypergraphKoopmanOperator,
    snapshot: Data,
    cache: list[
        tuple[
            Tensor,
            Tensor | None,
            Tensor | None,
            Tensor | None,
            Tensor,
        ]
    ],
) -> Tensor | None:
    """Return a cached or freshly built dense inverse for ``snapshot``.

    Parameters
    ----------
    koopman : GraphKoopmanOperator or HypergraphKoopmanOperator
        Dense networked operator.
    snapshot : Data
        Pair-target topology carrier.
    cache : list of tuple
        Evaluation-scoped entries
        ``(edge_index, edge_weight, hyperedge_index, hyperedge_weight, inverse)``.

    Returns
    -------
    Tensor or None
        Inverse for this topology, or ``None`` if it cannot be built.
    """
    edge_index = snapshot.edge_index
    edge_weight = snapshot_edge_weight(snapshot)
    hyperedge_index = snapshot_hyperedge_index(snapshot)
    hyperedge_weight = snapshot_hyperedge_weight(snapshot)
    for (
        cached_index,
        cached_weight,
        cached_hyper_index,
        cached_hyper_weight,
        cached_inverse,
    ) in cache:
        if topologies_equal(
            edge_index,
            edge_weight,
            cached_index,
            cached_weight,
            hyperedge_index_a=hyperedge_index,
            hyperedge_weight_a=hyperedge_weight,
            hyperedge_index_b=cached_hyper_index,
            hyperedge_weight_b=cached_hyper_weight,
        ):
            return cached_inverse
    inverse = _dense_networked_inverse_for_snapshot(koopman, snapshot)
    if inverse is not None:
        cache.append(
            (edge_index, edge_weight, hyperedge_index, hyperedge_weight, inverse)
        )
    return inverse


def compute_backward_consistency_sequence_loss(
    model: TrainableKoopmanModel,
    sequence: SnapshotSequence,
    *,
    cache: SequenceLatentCache | None = None,
) -> Tensor:
    """Average backward consistency loss over consecutive snapshot pairs.

    For discrete dense networked operators, reuses evaluation-scoped dense
    inverses keyed by topology content (static: one inverse; dynamic: one per
    distinct pair-target topology). Bilinear control never shares an inverse
    across pairs because pair-local ``K_self`` makes the map control-dependent.

    Parameters
    ----------
    model : TrainableKoopmanModel
        Model satisfying :class:`~koopman_graph.protocols.TrainableKoopmanModel`.
    sequence : GraphSnapshotSequence or HeteroGraphSnapshotSequence
        Time-ordered snapshots with at least two timesteps. Multiplex hetero
        sequences raise (unsupported).
    cache : SequenceLatentCache or None, optional
        Shared teacher-forced latents. When omitted, encodes per pair as before.

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

    shared_inverse: Tensor | None = None
    networked_inverse_cache: (
        list[
            tuple[
                Tensor,
                Tensor | None,
                Tensor | None,
                Tensor | None,
                Tensor,
            ]
        ]
        | None
    ) = None
    koopman = model.koopman
    if model.dynamics_mode in {"discrete", "stochastic"}:
        if isinstance(koopman, (GraphKoopmanOperator, HypergraphKoopmanOperator)):
            # Bilinear: pair-local effective K — never share inverses (TASK-1504).
            if (
                koopman.sparsity == "dense"
                and getattr(koopman, "control_mode", "additive") != "bilinear"
            ):
                if sequence.is_dynamic_topology:
                    networked_inverse_cache = []
                else:
                    shared_inverse = _dense_networked_inverse_for_snapshot(
                        koopman, sequence[1]
                    )
        elif koopman.parameterization == "dense":
            dense_inverse = getattr(koopman, "dense_inverse_matrix", None)
            if callable(dense_inverse):
                shared_inverse = dense_inverse()

    total_loss = torch.zeros((), device=next(model.parameters()).device)
    num_pairs = sequence.num_timesteps - 1
    for t in range(num_pairs):
        inverse_matrix = shared_inverse
        if networked_inverse_cache is not None and isinstance(
            koopman, (GraphKoopmanOperator, HypergraphKoopmanOperator)
        ):
            inverse_matrix = _lookup_networked_inverse(
                koopman, sequence[t + 1], networked_inverse_cache
            )
        total_loss = total_loss + _backward_consistency_pair(
            model,
            sequence,
            t,
            inverse_matrix=inverse_matrix,
            cache=cache,
        )
    return total_loss / num_pairs
