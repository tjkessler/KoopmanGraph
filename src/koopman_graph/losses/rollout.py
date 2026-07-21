"""Autoregressive rollout reconstruction losses."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from koopman_graph.data import (
    GraphSnapshotSequence,
    resolve_pair_delta_t,
)
from koopman_graph.graph_utils import (
    autoregressive_latent_rollout,
    snapshot_topology_at,
)
from koopman_graph.losses.reconstruction import masked_mse_loss
from koopman_graph.protocols import TrainableKoopmanModel


def rollout_sequence_loss(
    model: TrainableKoopmanModel,
    sequence: GraphSnapshotSequence,
    *,
    horizon: int,
    start: int = 0,
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
    sequence : :class:`~koopman_graph.data.GraphSnapshotSequence`
        Time-ordered snapshots. For dynamic-topology sequences, each decode step
        uses the target snapshot's ``edge_index``.
    horizon : int
        Number of rollout steps (must be >= 1).
    start : int, optional
        Index of the initial snapshot. Default is ``0``.

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

    initial = sequence[start]
    encode_at = getattr(model, "encode_at", None)
    z = encode_at(sequence, start) if callable(encode_at) else model.encode(initial)

    time_step = float(model.resolve_delta_t(None))
    targets = [sequence[start + step] for step in range(1, horizon + 1)]

    rollout = autoregressive_latent_rollout(
        model.koopman,
        model.decoder,
        z,
        steps=horizon,
        topology_at=snapshot_topology_at(targets),
        control_at=(
            None
            if not sequence.has_controls
            else (lambda step: sequence.control_at(start + step))
        ),
        delta_t_at=lambda step: resolve_pair_delta_t(
            sequence,
            start + step,
            default_time_step=time_step,
        ),
        default_delta_t=time_step,
    )

    total_loss = torch.zeros((), device=z.device)
    for step, (prediction, _, _) in enumerate(rollout):
        target = targets[step]
        if sequence.has_observation_masks:
            node_mask = sequence.observation_mask_at(start + step + 1)
            step_loss = masked_mse_loss(prediction, target.x, node_mask)
        else:
            step_loss = nn.functional.mse_loss(prediction, target.x)
        total_loss = total_loss + step_loss
    return total_loss / horizon


def rollout_multi_start_loss(
    model: TrainableKoopmanModel,
    sequence: GraphSnapshotSequence,
    *,
    horizon: int,
    start_indices: Sequence[int],
) -> Tensor:
    """Average rollout reconstruction loss over multiple start snapshots.

    Parameters
    ----------
    model : :class:`~koopman_graph.protocols.TrainableKoopmanModel`
        Trainable model accepted by :func:`rollout_sequence_loss` (also uses
        ``parameters`` for device placement).
        :class:`~koopman_graph.model.GraphKoopmanModel` is the intended
        implementer.
    sequence : GraphSnapshotSequence
        Time-ordered snapshots.
    horizon : int
        Number of rollout steps (must be >= 1).
    start_indices : sequence of int
        Zero-based origin indices for each rollout.

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
    total_loss = torch.zeros((), device=device)
    for start in start_indices:
        total_loss = total_loss + rollout_sequence_loss(
            model,
            sequence,
            horizon=horizon,
            start=start,
        )
    return total_loss / len(start_indices)
