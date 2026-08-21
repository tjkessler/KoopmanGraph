"""Device resolution helpers for training."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor

from koopman_graph.data import (
    GraphSnapshotSequence,
    HeteroGraphSnapshotSequence,
    SnapshotSequence,
)
from koopman_graph.graph_utils import snapshot_to_device
from koopman_graph.protocols import TrainableKoopmanModel


def resolve_device(
    model: TrainableKoopmanModel,
    device: str | torch.device | None,
) -> torch.device:
    """Resolve the training device from an explicit argument or model parameters.

    Parameters
    ----------
    model : TrainableKoopmanModel
        Model whose parameter device is used as fallback.
    device : str, torch.device, or None
        Explicit device. When ``None``, uses the device of the first model
        parameter, or CPU if the model has no parameters.

    Returns
    -------
    torch.device
        Resolved device for training or inference.
    """
    if device is not None:
        return torch.device(device)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _tensor_or_mapping_to_device(
    value: Tensor | Mapping[str, Tensor] | None,
    device: torch.device,
) -> Tensor | dict[str, Tensor] | None:
    """Move an optional tensor or string→tensor mapping to ``device``.

    Parameters
    ----------
    value : Tensor or mapping of str to Tensor or None
        Control tensor, observation-mask tensor, named control / mask map,
        or ``None``.
    device : torch.device
        Target device.

    Returns
    -------
    Tensor or dict of str to Tensor or None
        Device-local copy of ``value``, or ``None``.
    """
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {key: tensor.to(device) for key, tensor in value.items()}
    return value.to(device)


def sequence_to_device(
    sequence: SnapshotSequence,
    device: torch.device,
) -> SnapshotSequence:
    """Move a snapshot sequence and optional controls/timestamps/parameters.

    Preserves homogeneous vs multiplex container type (no silent cast from
    :class:`~koopman_graph.data.HeteroGraphSnapshotSequence` to
    :class:`~koopman_graph.data.GraphSnapshotSequence`).

    Parameters
    ----------
    sequence : GraphSnapshotSequence or HeteroGraphSnapshotSequence
        Sequence to move.
    device : torch.device
        Target device.

    Returns
    -------
    GraphSnapshotSequence or HeteroGraphSnapshotSequence
        Device-local copy of ``sequence`` with the same container family.
    """
    moved_snapshots = [snapshot_to_device(snapshot, device) for snapshot in sequence]
    if isinstance(sequence, HeteroGraphSnapshotSequence):
        return HeteroGraphSnapshotSequence(
            moved_snapshots,  # type: ignore[arg-type]
            allow_dynamic_topology=sequence.allow_dynamic_topology,
            control_inputs=_tensor_or_mapping_to_device(
                sequence.control_inputs, device
            ),
            timestamps=(
                None if sequence.timestamps is None else sequence.timestamps.to(device)
            ),
            observation_masks=_tensor_or_mapping_to_device(
                sequence.observation_masks, device
            ),
            presence_masks=_tensor_or_mapping_to_device(
                sequence.presence_masks, device
            ),
            allow_node_churn=sequence.allow_node_churn,
        )
    return GraphSnapshotSequence(
        moved_snapshots,  # type: ignore[arg-type]
        allow_dynamic_topology=sequence.allow_dynamic_topology,
        control_inputs=(
            None
            if sequence.control_inputs is None
            else sequence.control_inputs.to(device)
        ),
        timestamps=(
            None if sequence.timestamps is None else sequence.timestamps.to(device)
        ),
        parameter_trajectory=(
            None
            if sequence.parameter_trajectory is None
            else sequence.parameter_trajectory.to(device)
        ),
        observation_masks=(
            None
            if sequence.observation_masks is None
            else sequence.observation_masks.to(device)
        ),
        presence_masks=(
            None
            if sequence.presence_masks is None
            else sequence.presence_masks.to(device)
        ),
        entity_ids=sequence.entity_ids,
        allow_node_churn=sequence.allow_node_churn,
    )
