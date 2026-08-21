"""Snapshot coercion and validation helpers for graph sequences.

Power-user helpers shared by :class:`~koopman_graph.data.GraphSnapshotSequence`
and :class:`~koopman_graph.data.HeteroGraphSnapshotSequence` construction.
Prefer constructing sequences via the container APIs; import these symbols only
when validating or coercing inputs outside that path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data, HeteroData

from koopman_graph.graph_utils import (
    snapshot_edge_weight,
    snapshot_hyperedge_index,
    snapshot_hyperedge_weight,
)

ArrayLike = Tensor | np.ndarray
EdgeTypeTriple = tuple[str, str, str]


def _hetero_edge_weight(store: object) -> Tensor | None:
    """Return optional ``edge_weight`` on a PyG edge store, if present.

    Parameters
    ----------
    store : object
        Edge store exposing optional ``edge_weight``.

    Returns
    -------
    Tensor or None
        Weight tensor when present, otherwise ``None``.

    Raises
    ------
    TypeError
        If ``edge_weight`` is present but not a :class:`~torch.Tensor`.
    """
    weight = getattr(store, "edge_weight", None)
    if weight is None:
        return None
    if not isinstance(weight, Tensor):
        msg = f"edge_weight must be a Tensor when present, got {type(weight).__name__}"
        raise TypeError(msg)
    return weight


def _require_hetero_node_x(
    snapshot: HeteroData,
    node_type: str,
    *,
    index: int,
) -> Tensor:
    """Return node features for ``node_type``, raising a named error if absent.

    Parameters
    ----------
    snapshot : HeteroData
        Heterogeneous snapshot to inspect.
    node_type : str
        Node type name.
    index : int
        Snapshot index used in error messages.

    Returns
    -------
    Tensor
        Feature matrix with shape ``(N_τ, F_τ)``.

    Raises
    ------
    ValueError
        If the node type or feature tensor is missing or malformed.
    """
    if node_type not in snapshot.node_types:
        msg = (
            f"Snapshot {index} is missing node type {node_type!r}; "
            f"expected types {list(snapshot.node_types)}"
        )
        raise ValueError(msg)
    features = snapshot[node_type].x
    if features is None:
        msg = f"Snapshot {index} node type {node_type!r} has no feature tensor x"
        raise ValueError(msg)
    if features.ndim != 2:
        msg = (
            f"Snapshot {index} node type {node_type!r} features must have shape "
            f"(num_nodes, in_channels), got {tuple(features.shape)}"
        )
        raise ValueError(msg)
    return features


def infer_hetero_schema(
    snapshot: HeteroData,
    *,
    index: int = 0,
) -> tuple[dict[str, int], dict[str, int], list[EdgeTypeTriple]]:
    """Infer node feature dims, node counts, and ordered edge types.

    Parameters
    ----------
    snapshot : HeteroData
        Reference heterogeneous snapshot.
    index : int, optional
        Snapshot index used in error messages. Default is ``0``.

    Returns
    -------
    node_feature_dims : dict of str to int
        Mapping from node type name to feature dimension ``F_τ``.
    num_nodes : dict of str to int
        Mapping from node type name to node count ``N_τ``.
    edge_types : list of (str, str, str)
        Ordered ``(src_type, relation, dst_type)`` triples from the snapshot.

    Raises
    ------
    ValueError
        If the snapshot has no node types, no edge types, or missing features.
    """
    if not snapshot.node_types:
        msg = f"Snapshot {index} has no node types"
        raise ValueError(msg)
    if not snapshot.edge_types:
        msg = (
            f"Snapshot {index} has no edge types; "
            "HeteroGraphSnapshotSequence requires |R| >= 1"
        )
        raise ValueError(msg)

    node_feature_dims: dict[str, int] = {}
    num_nodes: dict[str, int] = {}
    for node_type in snapshot.node_types:
        features = _require_hetero_node_x(snapshot, node_type, index=index)
        node_feature_dims[node_type] = int(features.shape[1])
        num_nodes[node_type] = int(features.shape[0])

    edge_types = [tuple(edge_type) for edge_type in snapshot.edge_types]
    for edge_type in edge_types:
        src_type, _relation, dst_type = edge_type
        if src_type not in node_feature_dims:
            msg = (
                f"Snapshot {index} edge type {edge_type!r} references unknown "
                f"source node type {src_type!r}"
            )
            raise ValueError(msg)
        if dst_type not in node_feature_dims:
            msg = (
                f"Snapshot {index} edge type {edge_type!r} references unknown "
                f"destination node type {dst_type!r}"
            )
            raise ValueError(msg)
        store = snapshot[edge_type]
        if store.edge_index is None:
            msg = f"Snapshot {index} edge type {edge_type!r} has no edge_index"
            raise ValueError(msg)
        edge_index = store.edge_index
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            msg = (
                f"Snapshot {index} edge type {edge_type!r} edge_index must have "
                f"shape (2, num_edges), got {tuple(edge_index.shape)}"
            )
            raise ValueError(msg)
    return node_feature_dims, num_nodes, edge_types


def validate_hetero_snapshot_metadata(
    snapshots: Sequence[HeteroData],
) -> tuple[dict[str, int], dict[str, int], list[EdgeTypeTriple]]:
    """Verify fixed node types, counts, feature dims, and edge-type set.

    Parameters
    ----------
    snapshots : sequence of HeteroData
        Heterogeneous graph snapshots to validate.

    Returns
    -------
    node_feature_dims : dict of str to int
        Mapping from node type to feature dimension from snapshot 0.
    num_nodes : dict of str to int
        Mapping from node type to node count from snapshot 0.
    edge_types : list of (str, str, str)
        Ordered edge-type triples from snapshot 0.

    Raises
    ------
    ValueError
        If the sequence is empty or any snapshot drifts in cardinality,
        feature dimension, or edge-type set.
    """
    if not snapshots:
        msg = "HeteroGraphSnapshotSequence requires at least one snapshot"
        raise ValueError(msg)

    node_feature_dims, num_nodes, edge_types = infer_hetero_schema(
        snapshots[0],
        index=0,
    )
    ref_node_types = set(node_feature_dims)
    ref_edge_types = set(edge_types)

    for idx, snapshot in enumerate(snapshots[1:], start=1):
        other_dims, other_counts, other_edges = infer_hetero_schema(
            snapshot,
            index=idx,
        )
        if set(other_dims) != ref_node_types:
            msg = (
                f"Snapshot {idx} node types {sorted(other_dims)!r} differ from "
                f"snapshot 0 types {sorted(ref_node_types)!r}"
            )
            raise ValueError(msg)
        if set(other_edges) != ref_edge_types:
            msg = (
                f"Snapshot {idx} edge-type set {sorted(other_edges)!r} differs "
                f"from snapshot 0 set {sorted(ref_edge_types)!r}"
            )
            raise ValueError(msg)
        for node_type in node_feature_dims:
            if other_counts[node_type] != num_nodes[node_type]:
                msg = (
                    f"Snapshot {idx} node type {node_type!r} has "
                    f"{other_counts[node_type]} nodes, expected "
                    f"{num_nodes[node_type]}"
                )
                raise ValueError(msg)
            if other_dims[node_type] != node_feature_dims[node_type]:
                msg = (
                    f"Snapshot {idx} node type {node_type!r} has feature "
                    f"dimension {other_dims[node_type]}, expected "
                    f"{node_feature_dims[node_type]}"
                )
                raise ValueError(msg)
    return node_feature_dims, num_nodes, edge_types


def hetero_snapshots_have_dynamic_topology(
    snapshots: Sequence[HeteroData],
    edge_types: Sequence[EdgeTypeTriple],
) -> bool:
    """Return whether any relation's ``edge_index`` differs from snapshot 0.

    Parameters
    ----------
    snapshots : sequence of HeteroData
        Heterogeneous snapshots to inspect.
    edge_types : sequence of (str, str, str)
        Edge types to compare.

    Returns
    -------
    bool
        ``True`` when at least one relation differs in ``edge_index``.
    """
    if not snapshots:
        return False
    reference = snapshots[0]
    for snapshot in snapshots[1:]:
        for edge_type in edge_types:
            if not torch.equal(
                snapshot[edge_type].edge_index,
                reference[edge_type].edge_index,
            ):
                return True
    return False


def validate_shared_hetero_topology(
    snapshots: Sequence[HeteroData],
) -> tuple[dict[str, int], dict[str, int], list[EdgeTypeTriple]]:
    """Verify shared node schema and per-relation topology across snapshots.

    Parameters
    ----------
    snapshots : sequence of HeteroData
        Heterogeneous graph snapshots to validate.

    Returns
    -------
    node_feature_dims, num_nodes, edge_types
        Schema inferred from snapshot 0 (see
        :func:`validate_hetero_snapshot_metadata`).

    Raises
    ------
    ValueError
        If metadata drifts or any relation's ``edge_index`` /
        ``edge_weight`` differs from snapshot 0.
    """
    node_feature_dims, num_nodes, edge_types = validate_hetero_snapshot_metadata(
        snapshots
    )
    reference = snapshots[0]
    for idx, snapshot in enumerate(snapshots[1:], start=1):
        for edge_type in edge_types:
            if not torch.equal(
                snapshot[edge_type].edge_index,
                reference[edge_type].edge_index,
            ):
                msg = (
                    f"Snapshot {idx} edge type {edge_type!r} has a different "
                    "edge_index than snapshot 0"
                )
                raise ValueError(msg)
            ref_weight = _hetero_edge_weight(reference[edge_type])
            weight = _hetero_edge_weight(snapshot[edge_type])
            if (ref_weight is None) != (weight is None):
                msg = (
                    f"Snapshot {idx} edge type {edge_type!r} edge_weight "
                    "presence does not match snapshot 0"
                )
                raise ValueError(msg)
            if ref_weight is not None and not torch.allclose(
                weight,
                ref_weight,
                equal_nan=True,
            ):
                msg = (
                    f"Snapshot {idx} edge type {edge_type!r} has a different "
                    "edge_weight than snapshot 0"
                )
                raise ValueError(msg)
    return node_feature_dims, num_nodes, edge_types


def validate_hetero_observation_masks(
    observation_masks: Mapping[str, Tensor],
    *,
    num_timesteps: int,
    num_nodes: Mapping[str, int],
) -> dict[str, Tensor]:
    """Validate per-type observation masks for a hetero sequence.

    Parameters
    ----------
    observation_masks : mapping of str to Tensor
        Masks keyed by node type; each value has shape ``(T, N_τ)``.
    num_timesteps : int
        Expected number of snapshots.
    num_nodes : mapping of str to int
        Expected node count per type.

    Returns
    -------
    dict of str to Tensor
        Boolean masks for every node type in ``num_nodes``.

    Raises
    ------
    ValueError
        If keys, shapes, or observed-node constraints fail.
    """
    expected_types = set(num_nodes)
    mask_types = set(observation_masks)
    if mask_types != expected_types:
        msg = (
            "observation_masks keys must match node types exactly; "
            f"got {sorted(mask_types)!r}, expected {sorted(expected_types)!r}"
        )
        raise ValueError(msg)
    validated: dict[str, Tensor] = {}
    for node_type, mask in observation_masks.items():
        try:
            validated[node_type] = validate_observation_masks(
                mask,
                num_timesteps=num_timesteps,
                num_nodes=int(num_nodes[node_type]),
            )
        except ValueError as exc:
            msg = f"observation_masks[{node_type!r}]: {exc}"
            raise ValueError(msg) from exc
    return validated


def validate_hetero_presence_masks(
    presence_masks: Mapping[str, Tensor],
    *,
    num_timesteps: int,
    num_nodes: Mapping[str, int],
) -> dict[str, Tensor]:
    """Validate per-type entity presence masks for a hetero sequence.

    Presence marks whether an entity **exists** in each type's fixed union
    (``N_τ``). Distinct from :func:`validate_hetero_observation_masks`
    (measured vs present).

    Parameters
    ----------
    presence_masks : mapping of str to Tensor
        Masks keyed by node type; each value has shape ``(T, N_τ)``.
    num_timesteps : int
        Expected number of snapshots.
    num_nodes : mapping of str to int
        Expected node count per type.

    Returns
    -------
    dict of str to Tensor
        Boolean masks for every node type in ``num_nodes``.

    Raises
    ------
    ValueError
        If keys, shapes, or present-entity constraints fail.
    """
    expected_types = set(num_nodes)
    mask_types = set(presence_masks)
    if mask_types != expected_types:
        msg = (
            "presence_masks keys must match node types exactly; "
            f"got {sorted(mask_types)!r}, expected {sorted(expected_types)!r}"
        )
        raise ValueError(msg)
    validated: dict[str, Tensor] = {}
    for node_type, mask in presence_masks.items():
        try:
            validated[node_type] = validate_presence_masks(
                mask,
                num_timesteps=num_timesteps,
                num_nodes=int(num_nodes[node_type]),
            )
        except ValueError as exc:
            msg = f"presence_masks[{node_type!r}]: {exc}"
            raise ValueError(msg) from exc
    return validated


def validate_hetero_control_inputs(
    control_inputs: Tensor | Mapping[str, Tensor],
    *,
    num_timesteps: int,
    num_nodes: Mapping[str, int],
) -> None:
    """Validate global or per-type control inputs for a hetero sequence.

    Parameters
    ----------
    control_inputs : Tensor or mapping of str to Tensor
        Global controls use shape ``(T, C)``. A single-node-type sequence may
        also use ``(T, N, C)``. Per-type dict values use ``(T, C)`` or
        ``(T, N_τ, C)``.
    num_timesteps : int
        Expected number of snapshots.
    num_nodes : mapping of str to int
        Node counts per type.

    Raises
    ------
    ValueError
        If shapes or dict keys are invalid.
    """
    if isinstance(control_inputs, Mapping):
        if not control_inputs:
            msg = "control_inputs dict must be non-empty when provided"
            raise ValueError(msg)
        unknown = set(control_inputs) - set(num_nodes)
        if unknown:
            msg = (
                "control_inputs dict has unknown node types "
                f"{sorted(unknown)!r}; valid types are {sorted(num_nodes)!r}"
            )
            raise ValueError(msg)
        control_dims: set[int] = set()
        for node_type, controls in control_inputs.items():
            try:
                validate_control_inputs(
                    controls,
                    num_timesteps=num_timesteps,
                    num_nodes=int(num_nodes[node_type]),
                )
            except ValueError as exc:
                msg = f"control_inputs[{node_type!r}]: {exc}"
                raise ValueError(msg) from exc
            if controls.ndim == 2:
                control_dims.add(int(controls.shape[1]))
            else:
                control_dims.add(int(controls.shape[2]))
        if len(control_dims) != 1:
            msg = (
                "per-type control_inputs must share a common control_dim, "
                f"got dimensions {sorted(control_dims)!r}"
            )
            raise ValueError(msg)
        return

    if control_inputs.ndim == 3 and len(num_nodes) != 1:
        msg = (
            "tensor control_inputs with shape (T, N, C) require a single node "
            f"type; got {len(num_nodes)} types {sorted(num_nodes)!r} "
            "(use a dict of per-type controls or global (T, C))"
        )
        raise ValueError(msg)
    single_n = next(iter(num_nodes.values())) if len(num_nodes) == 1 else 0
    validate_control_inputs(
        control_inputs,
        num_timesteps=num_timesteps,
        num_nodes=int(single_n),
    )


def coerce_hyperedge_index(
    hyperedge_index: ArrayLike,
    *,
    num_nodes: int,
) -> Tensor:
    """Coerce and validate a PyG bipartite hyperedge incidence index.

    Parameters
    ----------
    hyperedge_index : array-like
        Incidence with shape ``(2, nnz)``: row 0 node indices, row 1
        hyperedge indices.
    num_nodes : int
        Number of nodes in the graph; used to bound node indices.

    Returns
    -------
    Tensor
        Long tensor with shape ``(2, nnz)``.

    Raises
    ------
    ValueError
        If shape is invalid or node indices are out of range.
    """
    edges = as_tensor(hyperedge_index, dtype=torch.long)
    if edges.ndim != 2 or edges.shape[0] != 2:
        msg = (
            "hyperedge_index must have shape (2, nnz) with row 0 = nodes "
            f"and row 1 = hyperedges, got {tuple(edges.shape)}"
        )
        raise ValueError(msg)
    if edges.numel() == 0:
        return edges
    if int(edges[0].min().item()) < 0 or int(edges[0].max().item()) >= num_nodes:
        msg = (
            "hyperedge_index node indices must lie in "
            f"[0, {num_nodes - 1}], got min={int(edges[0].min().item())}, "
            f"max={int(edges[0].max().item())}"
        )
        raise ValueError(msg)
    if int(edges[1].min().item()) < 0:
        msg = (
            "hyperedge_index hyperedge indices must be non-negative, "
            f"got min={int(edges[1].min().item())}"
        )
        raise ValueError(msg)
    return edges


def coerce_hyperedge_weight(
    hyperedge_weight: ArrayLike | None,
    *,
    hyperedge_index: Tensor,
    dtype: torch.dtype,
) -> Tensor | None:
    """Coerce and validate optional per-hyperedge weights.

    Parameters
    ----------
    hyperedge_weight : array-like or None
        Weights with shape ``(num_hyperedges,)``, indexed by the distinct
        hyperedge ids in ``hyperedge_index[1]``.
    hyperedge_index : Tensor
        Validated bipartite incidence with shape ``(2, nnz)``.
    dtype : torch.dtype
        Floating dtype for the returned weights.

    Returns
    -------
    Tensor or None
        Coerced weights, or ``None`` when ``hyperedge_weight`` is ``None``.

    Raises
    ------
    ValueError
        If weight shape does not match the number of hyperedges.
    """
    if hyperedge_weight is None:
        return None
    weights = as_tensor(hyperedge_weight, dtype=dtype)
    if weights.ndim != 1:
        msg = (
            "hyperedge_weight must have shape (num_hyperedges,), "
            f"got {tuple(weights.shape)}"
        )
        raise ValueError(msg)
    if hyperedge_index.numel() == 0:
        num_hyperedges = 0
    else:
        num_hyperedges = int(hyperedge_index[1].max().item()) + 1
    if weights.shape[0] != num_hyperedges:
        msg = (
            f"hyperedge_weight length {weights.shape[0]} does not match "
            f"num_hyperedges {num_hyperedges}"
        )
        raise ValueError(msg)
    return weights


def validate_static_hyperedges(snapshots: Sequence[Data]) -> None:
    """Require shared (static) hyperedge incidence across snapshots.

    When any snapshot carries ``hyperedge_index``, every snapshot must carry
    the same incidence and optional ``hyperedge_weight``. Time-varying
    hyperedges are rejected in this release.

    Parameters
    ----------
    snapshots : sequence of Data
        Graph snapshots to validate.

    Raises
    ------
    ValueError
        If hyperedge presence, indices, or weights differ across snapshots.
    """
    if not snapshots:
        return

    reference = snapshots[0]
    ref_index = snapshot_hyperedge_index(reference)
    ref_weight = snapshot_hyperedge_weight(reference)
    has_hyperedges = ref_index is not None

    for idx, snapshot in enumerate(snapshots[1:], start=1):
        hyperedge_index = snapshot_hyperedge_index(snapshot)
        hyperedge_weight = snapshot_hyperedge_weight(snapshot)
        if has_hyperedges != (hyperedge_index is not None):
            msg = (
                f"Snapshot {idx} hyperedge_index presence does not match "
                "snapshot 0; time-varying hyperedges are not supported"
            )
            raise ValueError(msg)
        if not has_hyperedges:
            continue
        if not torch.equal(hyperedge_index, ref_index):
            msg = (
                f"Snapshot {idx} has a different hyperedge_index than "
                "snapshot 0; time-varying hyperedges are not supported"
            )
            raise ValueError(msg)
        if (ref_weight is None) != (hyperedge_weight is None):
            msg = f"Snapshot {idx} hyperedge_weight presence does not match snapshot 0"
            raise ValueError(msg)
        if ref_weight is not None and not torch.allclose(
            hyperedge_weight,
            ref_weight,
            equal_nan=True,
        ):
            msg = (
                f"Snapshot {idx} has a different hyperedge_weight than "
                "snapshot 0; time-varying hyperedges are not supported"
            )
            raise ValueError(msg)


def sequence_has_hyperedges(snapshots: Sequence[Data]) -> bool:
    """Return whether any snapshot carries a ``hyperedge_index``.

    Parameters
    ----------
    snapshots : sequence of Data
        Graph snapshots to inspect.

    Returns
    -------
    bool
        ``True`` when at least one snapshot has ``hyperedge_index``.
    """
    return any(snapshot_hyperedge_index(snapshot) is not None for snapshot in snapshots)


def require_no_hyperedges(sequence: object) -> None:
    """Reject hyperedge-carrying sequences for non-hypergraph consumers.

    Classical baselines, :class:`~koopman_graph.env.GraphKoopmanEnv`, and the
    current neural ``fit`` path do not consume ``hyperedge_index``. Passing a
    hyperedge-carrying sequence would otherwise silently ignore the incidence.

    Parameters
    ----------
    sequence : GraphSnapshotSequence
        Candidate sequence exposing :attr:`has_hyperedges`.

    Raises
    ------
    ValueError
        If ``sequence.has_hyperedges`` is ``True``.
    """
    has_hyperedges = getattr(sequence, "has_hyperedges", False)
    if callable(has_hyperedges):
        flagged = bool(has_hyperedges())
    else:
        flagged = bool(has_hyperedges)
    if flagged:
        msg = (
            "this API does not support hyperedge-carrying sequences; "
            "got hyperedge_index on one or more snapshots "
            "(use a sequence without hyperedges, or a hypergraph-aware "
            "encoder/operator when available)"
        )
        raise ValueError(msg)


def as_tensor(value: ArrayLike, *, dtype: torch.dtype | None = None) -> Tensor:
    """Convert an array-like value to a :class:`torch.Tensor`.

    Parameters
    ----------
    value : Tensor or ndarray
        Input array or tensor.
    dtype : torch.dtype, optional
        Target dtype. When ``value`` is already a tensor, conversion is applied
        only if the dtypes differ.

    Returns
    -------
    Tensor
        Tensor representation of ``value``.
    """
    if isinstance(value, Tensor):
        if dtype is not None and value.dtype != dtype:
            return value.to(dtype=dtype)
        return value
    return torch.as_tensor(value, dtype=dtype)


def validate_parameter_trajectory(
    parameter_trajectory: Tensor,
    *,
    num_timesteps: int,
) -> None:
    """Validate optional per-snapshot regime coordinates.

    Parameters
    ----------
    parameter_trajectory : Tensor
        Regime coordinates :math:`\\mu_t` with shape
        ``(num_timesteps, d_mu)`` and ``d_mu >= 1``. Entries must be
        finite. Units are caller-defined (dimensionless if unspecified).
    num_timesteps : int
        Expected number of snapshots.

    Raises
    ------
    ValueError
        If rank, length, width, dtype, or finiteness is invalid.
    """
    if parameter_trajectory.ndim != 2:
        msg = (
            "parameter_trajectory must have shape (num_timesteps, d_mu), "
            f"got {tuple(parameter_trajectory.shape)}"
        )
        raise ValueError(msg)
    if int(parameter_trajectory.shape[0]) != num_timesteps:
        msg = (
            f"parameter_trajectory has {parameter_trajectory.shape[0]} "
            f"timesteps, expected {num_timesteps}"
        )
        raise ValueError(msg)
    if int(parameter_trajectory.shape[1]) < 1:
        msg = (
            "parameter_trajectory must have shape (num_timesteps, d_mu) "
            f"with d_mu >= 1, got {tuple(parameter_trajectory.shape)}"
        )
        raise ValueError(msg)
    if parameter_trajectory.dtype == torch.bool:
        msg = (
            "parameter_trajectory must be a real numeric tensor, "
            f"got dtype {parameter_trajectory.dtype}"
        )
        raise ValueError(msg)
    if parameter_trajectory.is_floating_point() and not torch.all(
        torch.isfinite(parameter_trajectory)
    ):
        msg = "parameter_trajectory must be finite"
        raise ValueError(msg)


def validate_timestamps(
    timestamps: Tensor,
    *,
    num_timesteps: int,
) -> None:
    """Validate optional per-snapshot timestamps.

    Parameters
    ----------
    timestamps : Tensor
        Monotone increasing timestamps with shape ``(num_timesteps,)``.
    num_timesteps : int
        Expected number of snapshots.

    Raises
    ------
    ValueError
        If ``timestamps`` has invalid shape or is not strictly increasing.
    """
    if timestamps.ndim != 1:
        msg = (
            "timestamps must have shape (num_timesteps,), "
            f"got {tuple(timestamps.shape)}"
        )
        raise ValueError(msg)
    if timestamps.shape[0] != num_timesteps:
        msg = f"timestamps has {timestamps.shape[0]} entries, expected {num_timesteps}"
        raise ValueError(msg)
    if num_timesteps >= 2:
        deltas = timestamps[1:] - timestamps[:-1]
        if not torch.all(deltas > 0):
            msg = "timestamps must be strictly monotone increasing"
            raise ValueError(msg)


def validate_observation_masks(
    observation_masks: Tensor,
    *,
    num_timesteps: int,
    num_nodes: int,
) -> Tensor:
    """Validate optional per-snapshot node observation masks.

    Parameters
    ----------
    observation_masks : Tensor
        Boolean or 0/1 mask with shape ``(num_timesteps, num_nodes)``.
        ``True`` (or ``1``) means the node is observed at that timestep.
    num_timesteps : int
        Expected number of snapshots.
    num_nodes : int
        Expected node count.

    Returns
    -------
    Tensor
        Boolean mask with shape ``(num_timesteps, num_nodes)``.

    Raises
    ------
    ValueError
        If ``observation_masks`` has invalid shape, dtype, or no observed nodes
        at any timestep.
    """
    if observation_masks.ndim != 2:
        msg = (
            "observation_masks must have shape (num_timesteps, num_nodes), "
            f"got {tuple(observation_masks.shape)}"
        )
        raise ValueError(msg)
    if observation_masks.shape != (num_timesteps, num_nodes):
        msg = (
            "observation_masks shape "
            f"{tuple(observation_masks.shape)} does not match "
            f"(num_timesteps={num_timesteps}, num_nodes={num_nodes})"
        )
        raise ValueError(msg)
    if observation_masks.dtype not in (torch.bool, torch.float, torch.int, torch.long):
        msg = (
            "observation_masks must be boolean or numeric 0/1, "
            f"got dtype {observation_masks.dtype}"
        )
        raise ValueError(msg)

    if observation_masks.dtype != torch.bool:
        unique = torch.unique(observation_masks)
        if not torch.all((unique == 0) | (unique == 1)):
            msg = "numeric observation_masks must contain only 0 and 1"
            raise ValueError(msg)

    mask = observation_masks.bool()
    empty_timesteps = torch.where(~mask.any(dim=1))[0]
    if empty_timesteps.numel() > 0:
        msg = (
            "observation_masks must have at least one observed node per timestep; "
            f"timesteps with no observations: {empty_timesteps.tolist()}"
        )
        raise ValueError(msg)
    return mask


def validate_presence_masks(
    presence_masks: Tensor,
    *,
    num_timesteps: int,
    num_nodes: int,
) -> Tensor:
    """Validate optional per-snapshot entity presence masks.

    Presence marks whether an entity **exists** in the fixed union universe
    at a timestep (shape ``(T, N_max)``). This is distinct from
    :func:`validate_observation_masks`, which marks whether an existing
    entity was **measured**. Inactive entities may carry padded zero feature
    rows; enforcing that padding is a separate churn-contract concern.

    Parameters
    ----------
    presence_masks : Tensor
        Boolean or 0/1 mask with shape ``(num_timesteps, num_nodes)``.
        ``True`` (or ``1``) means the entity is present at that timestep.
    num_timesteps : int
        Expected number of snapshots.
    num_nodes : int
        Expected union universe size ``N_max``.

    Returns
    -------
    Tensor
        Boolean mask with shape ``(num_timesteps, num_nodes)``.

    Raises
    ------
    ValueError
        If ``presence_masks`` has invalid shape, dtype, or no present entities
        at any timestep.
    """
    if presence_masks.ndim != 2:
        msg = (
            "presence_masks must have shape (num_timesteps, num_nodes), "
            f"got {tuple(presence_masks.shape)}"
        )
        raise ValueError(msg)
    if presence_masks.shape != (num_timesteps, num_nodes):
        msg = (
            "presence_masks shape "
            f"{tuple(presence_masks.shape)} does not match "
            f"(num_timesteps={num_timesteps}, num_nodes={num_nodes})"
        )
        raise ValueError(msg)
    if presence_masks.dtype not in (torch.bool, torch.float, torch.int, torch.long):
        msg = (
            "presence_masks must be boolean or numeric 0/1, "
            f"got dtype {presence_masks.dtype}"
        )
        raise ValueError(msg)

    if presence_masks.dtype != torch.bool:
        unique = torch.unique(presence_masks)
        if not torch.all((unique == 0) | (unique == 1)):
            msg = "numeric presence_masks must contain only 0 and 1"
            raise ValueError(msg)

    mask = presence_masks.bool()
    empty_timesteps = torch.where(~mask.any(dim=1))[0]
    if empty_timesteps.numel() > 0:
        msg = (
            "presence_masks must have at least one present entity per timestep; "
            f"timesteps with no present entities: {empty_timesteps.tolist()}"
        )
        raise ValueError(msg)
    return mask


def validate_entity_ids(
    entity_ids: Sequence[object],
    *,
    num_nodes: int,
) -> tuple[str | int, ...]:
    """Validate optional stable entity keys for a fixed union universe.

    Parameters
    ----------
    entity_ids : sequence
        One key per row of the fixed universe (``N_max``). Keys must be
        ``str`` or ``int`` and unique.
    num_nodes : int
        Expected universe size ``N_max``.

    Returns
    -------
    tuple of str or int
        Frozen unique entity keys in row order.

    Raises
    ------
    ValueError
        If length mismatches ``num_nodes``, keys are not ``str``/``int``, or
        duplicates appear.
    """
    ids = tuple(entity_ids)
    if len(ids) != num_nodes:
        msg = (
            f"entity_ids length {len(ids)} does not match "
            f"num_nodes={num_nodes} (fixed union N_max)"
        )
        raise ValueError(msg)
    typed_ids: list[str | int] = []
    for index, key in enumerate(ids):
        if not isinstance(key, (str, int)) or isinstance(key, bool):
            msg = (
                "entity_ids entries must be str or int; "
                f"got {type(key).__name__} at index {index}"
            )
            raise ValueError(msg)
        typed_ids.append(key)
    if len(set(typed_ids)) != len(typed_ids):
        seen: set[str | int] = set()
        duplicates: list[str | int] = []
        for key in typed_ids:
            if key in seen and key not in duplicates:
                duplicates.append(key)
            seen.add(key)
        msg = f"entity_ids must be unique; duplicates: {duplicates}"
        raise ValueError(msg)
    return tuple(typed_ids)


def validate_node_churn_policy(
    *,
    allow_node_churn: bool,
    presence_masks: Tensor | Mapping[str, Tensor] | None,
) -> None:
    """Enforce the fixed-union presence / ``allow_node_churn`` contract.

    Default ``allow_node_churn=False`` rejects any presence drop (a ``False``
    entry). Churn mode requires presence masks; when enabled, inactive feature
    rows may be padded zeros (zeros are conventional, not required here).
    Homogeneous sequences pass a ``(T, N_max)`` tensor; hetero sequences pass
    a per-type mapping of ``(T, N_τ)`` masks.

    Parameters
    ----------
    allow_node_churn : bool
        Whether entity drop-in/out via presence masks is permitted.
    presence_masks : Tensor, mapping of str to Tensor, or None
        Validated boolean presence masks, or ``None``.

    Raises
    ------
    ValueError
        If churn is enabled without presence masks, or if churn is disabled
        but presence masks drop any entity.
    """
    if allow_node_churn and presence_masks is None:
        msg = (
            "allow_node_churn=True requires presence_masks; "
            "churn without a presence contract is undefined"
        )
        raise ValueError(msg)
    if presence_masks is None:
        return
    if isinstance(presence_masks, Mapping):
        if allow_node_churn:
            return
        for node_type, mask in presence_masks.items():
            if not bool(mask.all()):
                inactive = torch.nonzero(~mask, as_tuple=False)
                sample = inactive[:5].tolist()
                msg = (
                    f"presence_masks[{node_type!r}] drop one or more entities but "
                    "allow_node_churn=False; set allow_node_churn=True for "
                    "fixed-union churn, or keep all entities present. Sample "
                    f"inactive (timestep, node) pairs: {sample}"
                )
                raise ValueError(msg)
        return
    if not allow_node_churn and not bool(presence_masks.all()):
        inactive = torch.nonzero(~presence_masks, as_tuple=False)
        # Report a short sample of (t, n) pairs for debugging.
        sample = inactive[:5].tolist()
        msg = (
            "presence_masks drop one or more entities but allow_node_churn=False; "
            "set allow_node_churn=True for fixed-union churn, or keep all "
            f"entities present. Sample inactive (timestep, node) pairs: {sample}"
        )
        raise ValueError(msg)


def validate_control_inputs(
    control_inputs: Tensor,
    *,
    num_timesteps: int,
    num_nodes: int,
) -> None:
    """Validate optional per-timestep control inputs.

    Parameters
    ----------
    control_inputs : Tensor
        Control tensor with shape ``(num_timesteps, control_dim)`` for global
        controls or ``(num_timesteps, num_nodes, control_dim)`` for per-node
        controls.
    num_timesteps : int
        Expected number of timesteps in the sequence.
    num_nodes : int
        Expected node count for per-node controls.

    Raises
    ------
    ValueError
        If ``control_inputs`` has invalid rank or shape.
    """
    if control_inputs.ndim not in (2, 3):
        msg = (
            "control_inputs must have shape (num_timesteps, control_dim) or "
            "(num_timesteps, num_nodes, control_dim), "
            f"got {tuple(control_inputs.shape)}"
        )
        raise ValueError(msg)
    if control_inputs.shape[0] != num_timesteps:
        msg = (
            f"control_inputs has {control_inputs.shape[0]} timesteps, "
            f"expected {num_timesteps}"
        )
        raise ValueError(msg)
    if control_inputs.ndim == 3 and control_inputs.shape[1] != num_nodes:
        msg = (
            f"per-node control_inputs has {control_inputs.shape[1]} nodes, "
            f"expected {num_nodes}"
        )
        raise ValueError(msg)


def validate_snapshot_metadata(snapshots: Sequence[Data]) -> None:
    """Verify consistent node count and feature dimension across snapshots.

    Parameters
    ----------
    snapshots : sequence of Data
        Graph snapshots to validate.

    Raises
    ------
    ValueError
        If the sequence is empty or any snapshot differs in node count or
        feature dimension from the first snapshot. A node-count mismatch
        is unbounded growth without
        :class:`~koopman_graph.data.EntityRemap`; pad into a shared
        :math:`N_{\\max}` first.
    """
    if not snapshots:
        msg = "GraphSnapshotSequence requires at least one snapshot"
        raise ValueError(msg)

    reference = snapshots[0]
    ref_num_nodes = reference.num_nodes
    ref_in_channels = reference.x.shape[1]

    for idx, snapshot in enumerate(snapshots[1:], start=1):
        if snapshot.num_nodes != ref_num_nodes:
            msg = (
                f"Snapshot {idx} has {snapshot.num_nodes} nodes, "
                f"expected {ref_num_nodes} (fixed N_max). "
                "Use EntityRemap / remap_node_features into a shared union; "
                "unbounded node growth without remap is refused."
            )
            raise ValueError(msg)
        if snapshot.x.shape[1] != ref_in_channels:
            msg = (
                f"Snapshot {idx} has feature dimension {snapshot.x.shape[1]}, "
                f"expected {ref_in_channels}"
            )
            raise ValueError(msg)


def snapshots_have_dynamic_topology(snapshots: Sequence[Data]) -> bool:
    """Return whether any snapshot differs in ``edge_index`` from the first.

    Parameters
    ----------
    snapshots : sequence of Data
        Graph snapshots to inspect.

    Returns
    -------
    bool
        ``True`` when at least one snapshot uses a different ``edge_index``.
    """
    if not snapshots:
        return False
    reference = snapshots[0].edge_index
    return any(
        not torch.equal(snapshot.edge_index, reference) for snapshot in snapshots[1:]
    )


def validate_shared_topology(snapshots: Sequence[Data]) -> None:
    """Verify that all snapshots share node count, features, and topology.

    Parameters
    ----------
    snapshots : sequence of Data
        Graph snapshots to validate.

    Raises
    ------
    ValueError
        If the sequence is empty or any snapshot differs in ``edge_index``,
        optional ``edge_weight``, node count, or feature dimension from the
        first snapshot.
    """
    validate_snapshot_metadata(snapshots)

    reference = snapshots[0]
    ref_edge_index = reference.edge_index
    ref_edge_weight = snapshot_edge_weight(reference)

    for idx, snapshot in enumerate(snapshots[1:], start=1):
        if not torch.equal(snapshot.edge_index, ref_edge_index):
            msg = f"Snapshot {idx} has a different edge_index than snapshot 0"
            raise ValueError(msg)
        edge_weight = snapshot_edge_weight(snapshot)
        if (ref_edge_weight is None) != (edge_weight is None):
            msg = f"Snapshot {idx} edge_weight presence does not match snapshot 0"
            raise ValueError(msg)
        if ref_edge_weight is not None and not torch.allclose(
            edge_weight,
            ref_edge_weight,
            equal_nan=True,
        ):
            msg = f"Snapshot {idx} has a different edge_weight than snapshot 0"
            raise ValueError(msg)
