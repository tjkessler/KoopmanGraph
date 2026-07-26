"""Snapshot coercion and validation helpers for graph sequences.

Power-user helpers shared by :class:`~koopman_graph.data.GraphSnapshotSequence`
construction. Prefer constructing sequences via the container APIs; import these
symbols only when validating or coercing inputs outside that path.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data

from koopman_graph.graph_utils import (
    snapshot_edge_weight,
    snapshot_hyperedge_index,
    snapshot_hyperedge_weight,
)

ArrayLike = Tensor | np.ndarray


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
        feature dimension from the first snapshot.
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
                f"expected {ref_num_nodes}"
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
