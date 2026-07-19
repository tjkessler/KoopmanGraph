"""Array and delay-window builders for graph snapshot sequences.

Power-user helpers that turn array / dynamic-array / Hankel inputs into
validated ``Data`` lists and optional metadata. Prefer constructing sequences
via :class:`~koopman_graph.data.GraphSnapshotSequence` classmethods; import
these symbols only when building snapshots outside that path.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch_geometric.data import Data

from koopman_graph.data.validation import ArrayLike, as_tensor

if TYPE_CHECKING:
    from koopman_graph.data.containers import GraphSnapshotSequence


@dataclass(frozen=True)
class ConstructedSnapshots:
    """Snapshot list plus optional sequence metadata for container init.

    Attributes
    ----------
    snapshots : list of Data
        Time-ordered graph snapshots.
    control_inputs : Tensor or None
        Coerced control tensor, or ``None``.
    timestamps : Tensor or None
        Coerced timestamp tensor, or ``None``.
    observation_masks : Tensor or None
        Coerced boolean observation mask, or ``None``.
    allow_dynamic_topology : bool
        Whether the container should permit per-snapshot topology.
    """

    snapshots: list[Data]
    control_inputs: Tensor | None = None
    timestamps: Tensor | None = None
    observation_masks: Tensor | None = None
    allow_dynamic_topology: bool = False


def build_snapshots_from_arrays(
    node_features: ArrayLike,
    edge_index: ArrayLike,
    *,
    edge_weight: ArrayLike | None = None,
    control_inputs: ArrayLike | None = None,
    timestamps: ArrayLike | None = None,
    observation_masks: ArrayLike | None = None,
    dtype: torch.dtype = torch.float32,
) -> ConstructedSnapshots:
    """Build static-topology snapshots from node feature arrays.

    Parameters
    ----------
    node_features : array-like
        Array with shape ``(num_timesteps, num_nodes, in_channels)``.
    edge_index : array-like
        Shared edge index with shape ``(2, num_edges)``.
    edge_weight : array-like, optional
        Shared scalar edge weights with shape ``(num_edges,)``.
    control_inputs : array-like, optional
        Per-timestep control inputs with shape ``(num_timesteps,
        control_dim)`` or ``(num_timesteps, num_nodes, control_dim)``.
    timestamps : array-like, optional
        Strictly increasing physical timestamps with shape
        ``(num_timesteps,)``.
    observation_masks : array-like, optional
        Per-timestep node observation mask with shape
        ``(num_timesteps, num_nodes)``.
    dtype : torch.dtype, optional
        Floating dtype used when converting numpy inputs to torch tensors.
        Default is ``torch.float32``.

    Returns
    -------
    ConstructedSnapshots
        Snapshot list and coerced optional metadata (static topology).

    Raises
    ------
    ValueError
        If ``node_features``, ``edge_index``, or ``edge_weight`` have
        invalid shape.
    """
    features = as_tensor(node_features, dtype=dtype)
    edges = as_tensor(edge_index, dtype=torch.long)
    weights = None if edge_weight is None else as_tensor(edge_weight, dtype=dtype)
    controls = (
        None if control_inputs is None else as_tensor(control_inputs, dtype=dtype)
    )
    times = None if timestamps is None else as_tensor(timestamps, dtype=dtype)
    masks = (
        None
        if observation_masks is None
        else as_tensor(observation_masks, dtype=torch.bool)
    )

    if features.ndim != 3:
        msg = (
            f"node_features must have shape "
            f"(num_timesteps, num_nodes, in_channels), got {tuple(features.shape)}"
        )
        raise ValueError(msg)
    if edges.ndim != 2 or edges.shape[0] != 2:
        msg = f"edge_index must have shape (2, num_edges), got {tuple(edges.shape)}"
        raise ValueError(msg)
    if weights is not None and weights.ndim != 1:
        msg = f"edge_weight must have shape (num_edges,), got {tuple(weights.shape)}"
        raise ValueError(msg)
    if weights is not None and weights.shape[0] != edges.shape[1]:
        msg = (
            f"edge_weight length {weights.shape[0]} does not match "
            f"num_edges {edges.shape[1]}"
        )
        raise ValueError(msg)
    if features.shape[0] < 1:
        msg = "node_features must contain at least one timestep"
        raise ValueError(msg)

    snapshots = []
    for t in range(features.shape[0]):
        if weights is None:
            snapshots.append(Data(x=features[t], edge_index=edges))
        else:
            snapshots.append(
                Data(
                    x=features[t],
                    edge_index=edges,
                    edge_weight=weights.clone(),
                )
            )
    return ConstructedSnapshots(
        snapshots=snapshots,
        control_inputs=controls,
        timestamps=times,
        observation_masks=masks,
        allow_dynamic_topology=False,
    )


def build_snapshots_from_dynamic_arrays(
    node_features: ArrayLike,
    edge_indices: Sequence[ArrayLike],
    *,
    edge_weights: Sequence[ArrayLike | None] | None = None,
    control_inputs: ArrayLike | None = None,
    timestamps: ArrayLike | None = None,
    observation_masks: ArrayLike | None = None,
    dtype: torch.dtype = torch.float32,
) -> ConstructedSnapshots:
    """Build dynamic-topology snapshots from per-timestep edge indices.

    Parameters
    ----------
    node_features : array-like
        Array with shape ``(num_timesteps, num_nodes, in_channels)``.
    edge_indices : sequence of array-like
        One edge index per timestep, each with shape ``(2, num_edges_t)``.
    edge_weights : sequence of array-like or None, optional
        Optional per-timestep scalar edge weights aligned with
        ``edge_indices``. When provided, must have the same length as
        ``edge_indices``.
    control_inputs : array-like, optional
        Per-timestep control inputs with shape ``(num_timesteps,
        control_dim)`` or ``(num_timesteps, num_nodes, control_dim)``.
    timestamps : array-like, optional
        Strictly increasing physical timestamps with shape
        ``(num_timesteps,)``.
    observation_masks : array-like, optional
        Per-timestep node observation mask with shape
        ``(num_timesteps, num_nodes)``.
    dtype : torch.dtype, optional
        Floating dtype used when converting numpy inputs to torch tensors.
        Default is ``torch.float32``.

    Returns
    -------
    ConstructedSnapshots
        Snapshot list and coerced optional metadata with
        ``allow_dynamic_topology=True``.

    Raises
    ------
    ValueError
        If shapes are inconsistent or ``edge_indices`` length mismatches
        ``num_timesteps``.
    """
    features = as_tensor(node_features, dtype=dtype)
    if features.ndim != 3:
        msg = (
            f"node_features must have shape "
            f"(num_timesteps, num_nodes, in_channels), got {tuple(features.shape)}"
        )
        raise ValueError(msg)
    if features.shape[0] < 1:
        msg = "node_features must contain at least one timestep"
        raise ValueError(msg)

    num_timesteps = int(features.shape[0])
    if len(edge_indices) != num_timesteps:
        msg = (
            f"edge_indices has length {len(edge_indices)}, expected "
            f"{num_timesteps} to match node_features timesteps"
        )
        raise ValueError(msg)
    if edge_weights is not None and len(edge_weights) != num_timesteps:
        msg = f"edge_weights has length {len(edge_weights)}, expected {num_timesteps}"
        raise ValueError(msg)

    snapshots: list[Data] = []
    for t in range(num_timesteps):
        edges = as_tensor(edge_indices[t], dtype=torch.long)
        if edges.ndim != 2 or edges.shape[0] != 2:
            msg = (
                f"edge_indices[{t}] must have shape (2, num_edges), "
                f"got {tuple(edges.shape)}"
            )
            raise ValueError(msg)
        weight = None
        if edge_weights is not None:
            weight_value = edge_weights[t]
            if weight_value is not None:
                weight = as_tensor(weight_value, dtype=dtype)
                if weight.ndim != 1:
                    msg = (
                        f"edge_weights[{t}] must have shape (num_edges,), "
                        f"got {tuple(weight.shape)}"
                    )
                    raise ValueError(msg)
                if weight.shape[0] != edges.shape[1]:
                    msg = (
                        f"edge_weights[{t}] length {weight.shape[0]} does not "
                        f"match num_edges {edges.shape[1]}"
                    )
                    raise ValueError(msg)
        if weight is None:
            snapshots.append(Data(x=features[t], edge_index=edges))
        else:
            snapshots.append(
                Data(
                    x=features[t],
                    edge_index=edges,
                    edge_weight=weight,
                )
            )

    controls = (
        None if control_inputs is None else as_tensor(control_inputs, dtype=dtype)
    )
    times = None if timestamps is None else as_tensor(timestamps, dtype=dtype)
    masks = (
        None
        if observation_masks is None
        else as_tensor(observation_masks, dtype=torch.bool)
    )
    return ConstructedSnapshots(
        snapshots=snapshots,
        control_inputs=controls,
        timestamps=times,
        observation_masks=masks,
        allow_dynamic_topology=True,
    )


def build_windowed_snapshots(
    sequence: GraphSnapshotSequence,
    n_delays: int,
    *,
    stride: int = 1,
    pad: bool = True,
    zero_unobserved: bool = True,
) -> ConstructedSnapshots:
    """Build Hankel-stacked delay-window snapshots from a source sequence.

    Parameters
    ----------
    sequence : GraphSnapshotSequence
        Source trajectory.
    n_delays : int
        Delay window length (must be >= 1).
    stride : int, optional
        Step between successive window ends. Default is ``1``.
    pad : bool, optional
        Zero-pad missing history at the sequence start. Default is ``True``.
    zero_unobserved : bool, optional
        Zero unobserved node rows inside each delay slot when observation
        masks are present. Default is ``True``.

    Returns
    -------
    ConstructedSnapshots
        Windowed snapshots with metadata taken from each window end.

    Raises
    ------
    ValueError
        If ``n_delays < 1``, ``stride < 1``, the sequence is too short when
        ``pad=False``, or a window spans a topology change on a static
        sequence.
    """
    from koopman_graph.data.delay_windows import (
        flatten_delay_window,
        stack_delay_features,
    )

    if n_delays < 1:
        msg = f"n_delays must be >= 1, got {n_delays}"
        raise ValueError(msg)
    if stride < 1:
        msg = f"stride must be >= 1, got {stride}"
        raise ValueError(msg)

    first_end = 0 if pad else n_delays - 1
    if first_end >= sequence.num_timesteps:
        msg = (
            f"sequence too short for n_delays={n_delays} with pad={pad} "
            f"(num_timesteps={sequence.num_timesteps})"
        )
        raise ValueError(msg)

    ends = list(range(first_end, sequence.num_timesteps, stride))
    stacked_snapshots: list[Data] = []
    for end in ends:
        x_window, edge_index, edge_weight, _history_mask = stack_delay_features(
            sequence,
            end,
            n_delays,
            pad=pad,
            zero_unobserved=zero_unobserved,
        )
        payload: dict[str, object] = {
            "x": flatten_delay_window(x_window),
            "edge_index": edge_index,
        }
        if edge_weight is not None:
            payload["edge_weight"] = edge_weight
        stacked_snapshots.append(Data(**payload))

    control_inputs = None
    if sequence.control_inputs is not None:
        control_inputs = sequence.control_inputs[ends]

    timestamps = None
    if sequence.timestamps is not None:
        timestamps = sequence.timestamps[ends]

    observation_masks = None
    if sequence.observation_masks is not None:
        observation_masks = sequence.observation_masks[ends]

    return ConstructedSnapshots(
        snapshots=stacked_snapshots,
        control_inputs=control_inputs,
        timestamps=timestamps,
        observation_masks=observation_masks,
        allow_dynamic_topology=sequence.allow_dynamic_topology,
    )
