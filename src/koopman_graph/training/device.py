"""Device resolution helpers for training."""

from __future__ import annotations

import torch

from koopman_graph.data import GraphSnapshotSequence
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


def sequence_to_device(
    sequence: GraphSnapshotSequence,
    device: torch.device,
) -> GraphSnapshotSequence:
    """Move a snapshot sequence and optional controls/timestamps to ``device``.

    Parameters
    ----------
    sequence : GraphSnapshotSequence
        Sequence to move.
    device : torch.device
        Target device.

    Returns
    -------
    GraphSnapshotSequence
        Device-local copy of ``sequence``.
    """
    return GraphSnapshotSequence(
        [snapshot_to_device(snapshot, device) for snapshot in sequence],
        allow_dynamic_topology=sequence.allow_dynamic_topology,
        control_inputs=(
            None
            if sequence.control_inputs is None
            else sequence.control_inputs.to(device)
        ),
        timestamps=(
            None if sequence.timestamps is None else sequence.timestamps.to(device)
        ),
    )
