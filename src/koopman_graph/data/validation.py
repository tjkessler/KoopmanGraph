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

from koopman_graph.graph_utils import snapshot_edge_weight

ArrayLike = Tensor | np.ndarray

ArrayLike = Tensor | np.ndarray


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
