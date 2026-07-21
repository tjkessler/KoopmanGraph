"""Delay-window tensor and snapshot transforms (data-layer helpers).

Owns Hankel-style stacking / flattening and observation-mask zeroing used by
:class:`~koopman_graph.data.containers.GraphSnapshotSequence`,
:mod:`koopman_graph.nn.delay`, and model encode paths. Feature modules import
**down** into this peer; :mod:`koopman_graph.data` must not import
:mod:`koopman_graph.nn`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch_geometric.data import Data

if TYPE_CHECKING:
    from koopman_graph.data.containers import GraphSnapshotSequence


def apply_observation_mask_to_features(
    x: Tensor,
    mask: Tensor | None,
) -> Tensor:
    """Zero node rows that are marked unobserved.

    Parameters
    ----------
    x : Tensor
        Node features with shape ``(num_nodes, feature_dim)``.
    mask : Tensor or None
        Boolean observation mask with shape ``(num_nodes,)``. ``True`` means
        observed. When ``None``, ``x`` is returned unchanged.

    Returns
    -------
    Tensor
        Masked features (cloned when a mask is applied).
    """
    if mask is None:
        return x
    if mask.shape != (x.size(0),):
        msg = (
            f"observation mask shape {tuple(mask.shape)} does not match "
            f"num_nodes={x.size(0)}"
        )
        raise ValueError(msg)
    out = x.clone()
    out[~mask.bool()] = 0.0
    return out


def stack_delay_features(
    sequence: GraphSnapshotSequence,
    index: int,
    n_delays: int,
    *,
    pad: bool = True,
    zero_unobserved: bool = True,
) -> tuple[Tensor, Tensor, Tensor | None, Tensor]:
    """Build a delay window of node features ending at ``index``.

    Parameters
    ----------
    sequence : GraphSnapshotSequence
        Source trajectory.
    index : int
        Inclusive end index of the delay window.
    n_delays : int
        Window length (must be >= 1).
    pad : bool, optional
        When ``True``, zero-pad missing history before the sequence start.
        When ``False``, require ``index >= n_delays - 1``. Default is ``True``.
    zero_unobserved : bool, optional
        When ``True`` and the sequence carries observation masks, zero
        unobserved node rows in each delay slot before stacking. Default is
        ``True``.

    Returns
    -------
    x_window : Tensor
        Delay stack with shape ``(n_delays, num_nodes, feature_dim)``, ordered
        oldest → newest (index ``-1`` is time ``index``).
    edge_index : Tensor
        Topology at the window end (time ``index``).
    edge_weight : Tensor or None
        Optional edge weights at the window end.
    history_mask : Tensor
        Boolean mask with shape ``(n_delays,)``. ``True`` means that delay slot
        came from a real snapshot (not zero-padding).

    Raises
    ------
    ValueError
        If indices are out of range, ``n_delays < 1``, padding is disabled and
        history is insufficient, or topology changes across the window when
        dynamic topology is not allowed.
    IndexError
        If ``index`` is outside ``[0, num_timesteps)``.
    """
    if n_delays < 1:
        msg = f"n_delays must be >= 1, got {n_delays}"
        raise ValueError(msg)
    if index < 0 or index >= sequence.num_timesteps:
        msg = f"index must satisfy 0 <= index < {sequence.num_timesteps}, got {index}"
        raise IndexError(msg)

    start = index - n_delays + 1
    if start < 0 and not pad:
        msg = (
            f"insufficient history for n_delays={n_delays} at index={index}; "
            "pass pad=True or start later in the sequence"
        )
        raise ValueError(msg)

    end_snapshot = sequence[index]
    feature_dim = int(end_snapshot.x.size(-1))
    num_nodes = int(end_snapshot.x.size(0))
    edge_index = end_snapshot.edge_index
    edge_weight = getattr(end_snapshot, "edge_weight", None)

    slots: list[Tensor] = []
    history_flags: list[bool] = []
    ref_edge_index = edge_index
    ref_edge_weight = edge_weight

    for delay_offset in range(n_delays):
        t = start + delay_offset
        if t < 0:
            slots.append(
                torch.zeros(
                    num_nodes,
                    feature_dim,
                    dtype=end_snapshot.x.dtype,
                    device=end_snapshot.x.device,
                )
            )
            history_flags.append(False)
            continue

        snapshot = sequence[t]
        if snapshot.x.size(0) != num_nodes or snapshot.x.size(-1) != feature_dim:
            msg = (
                "delay window snapshots must share num_nodes and feature_dim; "
                f"got {tuple(snapshot.x.shape)} vs ({num_nodes}, {feature_dim})"
            )
            raise ValueError(msg)

        if not sequence.allow_dynamic_topology:
            same_edges = torch.equal(snapshot.edge_index, ref_edge_index)
            snap_weight = getattr(snapshot, "edge_weight", None)
            if snap_weight is None and ref_edge_weight is None:
                same_weights = True
            elif snap_weight is None or ref_edge_weight is None:
                same_weights = False
            else:
                same_weights = torch.equal(snap_weight, ref_edge_weight)
            if not (same_edges and same_weights):
                msg = (
                    "topology changed across delay window "
                    f"[{max(start, 0)}, {index}]; set allow_dynamic_topology=True "
                    "or use a static-topology sequence"
                )
                raise ValueError(msg)

        x_t = snapshot.x
        if zero_unobserved and sequence.has_observation_masks:
            x_t = apply_observation_mask_to_features(
                x_t,
                sequence.observation_mask_at(t),
            )
        slots.append(x_t)
        history_flags.append(True)

    x_window = torch.stack(slots, dim=0)
    history_mask = torch.tensor(
        history_flags,
        dtype=torch.bool,
        device=end_snapshot.x.device,
    )
    return x_window, edge_index, edge_weight, history_mask


def flatten_delay_window(x_window: Tensor) -> Tensor:
    """Concatenate delay slots along the feature axis.

    Parameters
    ----------
    x_window : Tensor
        Delay stack with shape ``(n_delays, num_nodes, feature_dim)``.

    Returns
    -------
    Tensor
        Flattened features with shape ``(num_nodes, n_delays * feature_dim)``.
    """
    if x_window.ndim != 3:
        msg = (
            "x_window must have shape (n_delays, num_nodes, feature_dim), "
            f"got {tuple(x_window.shape)}"
        )
        raise ValueError(msg)
    n_delays, num_nodes, feature_dim = x_window.shape
    return x_window.permute(1, 0, 2).reshape(num_nodes, n_delays * feature_dim)


def history_from_snapshots(
    snapshots: Sequence[Data],
    n_delays: int,
    *,
    pad: bool = True,
) -> tuple[Tensor, Tensor, Tensor | None, Tensor]:
    """Build a delay window from an explicit snapshot list (oldest → newest).

    Parameters
    ----------
    snapshots : sequence of Data
        Observed history. The last entry is the newest state. Length may be
        shorter than ``n_delays`` when ``pad=True``.
    n_delays : int
        Target window length.
    pad : bool, optional
        Zero-pad on the left when fewer than ``n_delays`` snapshots are given.

    Returns
    -------
    tuple
        Same layout as :func:`stack_delay_features`.
    """
    if n_delays < 1:
        msg = f"n_delays must be >= 1, got {n_delays}"
        raise ValueError(msg)
    if not snapshots:
        msg = "snapshots must contain at least one Data object"
        raise ValueError(msg)
    if len(snapshots) > n_delays:
        snapshots = list(snapshots)[-n_delays:]
    if len(snapshots) < n_delays and not pad:
        msg = f"need {n_delays} snapshots when pad=False, got {len(snapshots)}"
        raise ValueError(msg)

    newest = snapshots[-1]
    feature_dim = int(newest.x.size(-1))
    num_nodes = int(newest.x.size(0))
    pad_count = n_delays - len(snapshots)
    slots: list[Tensor] = [
        torch.zeros(
            num_nodes,
            feature_dim,
            dtype=newest.x.dtype,
            device=newest.x.device,
        )
        for _ in range(pad_count)
    ]
    history_flags = [False] * pad_count
    for snapshot in snapshots:
        slots.append(snapshot.x)
        history_flags.append(True)

    x_window = torch.stack(slots, dim=0)
    history_mask = torch.tensor(
        history_flags,
        dtype=torch.bool,
        device=newest.x.device,
    )
    edge_weight = getattr(newest, "edge_weight", None)
    return x_window, newest.edge_index, edge_weight, history_mask
