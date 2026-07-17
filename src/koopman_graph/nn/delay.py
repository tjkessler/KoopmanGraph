"""Delay-coordinate encoder wrapper for partial observability.

Stacks a short history of node features into encoder channels (Takens-style
delay coordinates). Related Hankel delay matrices appear in HAVOK / Hankel-DMD;
this module does not implement those full algorithms.

Composition style (see architecture docs): callers size the base encoder with
``in_channels = n_delays * feature_dim`` and wrap it explicitly, or pass
``n_delays`` to :class:`~koopman_graph.model.GraphKoopmanModel` so the model
wraps without rebuilding layers.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
from torch_geometric.data import Data

from koopman_graph.graph_utils import resolve_edge_index, resolve_edge_weight

if TYPE_CHECKING:
    from koopman_graph.data.containers import GraphSnapshotSequence


def _feature_dim_from_base(base_encoder: nn.Module, n_delays: int) -> int:
    """Infer per-snapshot feature width from a stacked-channel base encoder.

    Parameters
    ----------
    base_encoder : nn.Module
        Encoder exposing ``in_channels`` equal to ``n_delays * feature_dim``.
    n_delays : int
        Number of delay coordinates (must be >= 1).

    Returns
    -------
    int
        Per-snapshot feature dimension ``F``.

    Raises
    ------
    ValueError
        If ``n_delays < 1``, the encoder lacks ``in_channels``, or
        ``in_channels`` is not divisible by ``n_delays``.
    """
    if n_delays < 1:
        msg = f"n_delays must be >= 1, got {n_delays}"
        raise ValueError(msg)
    in_channels = getattr(base_encoder, "in_channels", None)
    if in_channels is None:
        msg = "base_encoder must expose in_channels for delay embedding"
        raise ValueError(msg)
    if in_channels % n_delays != 0:
        msg = (
            f"base_encoder.in_channels ({in_channels}) must be divisible by "
            f"n_delays ({n_delays}); size the encoder as "
            f"in_channels = n_delays * feature_dim"
        )
        raise ValueError(msg)
    return int(in_channels) // int(n_delays)


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

    # Local import avoids a containers ↔ nn.delay cycle (windowed lives on
    # GraphSnapshotSequence and calls into this helper).
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


class DelayEmbeddingEncoder(nn.Module):
    """Wrap a GNN/GAT encoder with delay-coordinate channel stacking.

    Stacks the last ``n_delays`` per-node feature vectors along the channel
    axis before calling ``base_encoder``. The base encoder must already be
    sized with ``in_channels = n_delays * feature_dim`` (composition; this
    wrapper does not rebuild layers).

    Attributes
    ----------
    base_encoder : nn.Module
        Underlying topology-aware encoder.
    n_delays : int
        Number of delay coordinates.
    feature_dim : int
        Per-snapshot feature width ``F``.
    in_channels : int
        Stacked channel width ``n_delays * feature_dim`` (matches base).
    latent_dim : int
        Latent width exposed by ``base_encoder``.
    """

    def __init__(self, base_encoder: nn.Module, n_delays: int = 3) -> None:
        """Initialize the delay-embedding wrapper.

        Parameters
        ----------
        base_encoder : nn.Module
            Encoder with ``in_channels = n_delays * feature_dim`` and
            ``latent_dim``.
        n_delays : int, optional
            Delay window length. Default is ``3``.

        Raises
        ------
        ValueError
            If ``n_delays < 1`` or base ``in_channels`` is incompatible.
        """
        super().__init__()
        feature_dim = _feature_dim_from_base(base_encoder, n_delays)
        self.base_encoder = base_encoder
        self.n_delays = int(n_delays)
        self.feature_dim = feature_dim
        self.in_channels = int(base_encoder.in_channels)
        self.latent_dim = int(base_encoder.latent_dim)
        # Mirror common checkpoint fields from GCN/GAT bases when present.
        for attr in (
            "hidden_channels",
            "num_layers",
            "activation_name",
            "heads",
            "dropout",
        ):
            if hasattr(base_encoder, attr):
                setattr(self, attr, getattr(base_encoder, attr))

    def forward(
        self,
        x_or_data: Tensor | Data,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        """Lift a delay window or already-stacked features into latent space.

        Parameters
        ----------
        x_or_data : Tensor or Data
            One of:

            * Delay window ``(n_delays, num_nodes, feature_dim)``
            * Stacked features ``(num_nodes, n_delays * feature_dim)``
            * ``Data`` whose ``x`` is already stacked
        edge_index : Tensor or None, optional
            Edge index when ``x_or_data`` is a bare tensor window/stack.
        edge_weight : Tensor or None, optional
            Optional scalar edge weights.

        Returns
        -------
        Tensor
            Latent node features with shape ``(num_nodes, latent_dim)``.
        """
        if isinstance(x_or_data, Data):
            x = x_or_data.x
            edge_index = resolve_edge_index(x_or_data, edge_index)
            edge_weight = resolve_edge_weight(x_or_data, edge_weight)
        else:
            x = x_or_data
            if edge_index is None:
                msg = "edge_index is required when x_or_data is a tensor"
                raise ValueError(msg)

        if x.ndim == 3:
            if x.size(0) != self.n_delays:
                msg = (
                    f"delay window leading dim must equal n_delays="
                    f"{self.n_delays}, got {x.size(0)}"
                )
                raise ValueError(msg)
            if x.size(-1) != self.feature_dim:
                msg = (
                    f"delay window feature_dim must equal {self.feature_dim}, "
                    f"got {x.size(-1)}"
                )
                raise ValueError(msg)
            x = flatten_delay_window(x)
        elif x.ndim == 2:
            if x.size(-1) != self.in_channels:
                msg = (
                    f"stacked features must have in_channels={self.in_channels}, "
                    f"got {x.size(-1)}"
                )
                raise ValueError(msg)
        else:
            msg = (
                "expected delay window (n_delays, N, F) or stacked (N, n_delays*F), "
                f"got shape {tuple(x.shape)}"
            )
            raise ValueError(msg)

        return self.base_encoder(x, edge_index, edge_weight)


def resolve_delay_encoder(
    encoder: nn.Module,
    n_delays: int,
) -> tuple[nn.Module, int]:
    """Resolve composition of a base or delay encoder with ``n_delays``.

    Parameters
    ----------
    encoder : nn.Module
        Bare GNN/GAT encoder or an existing :class:`DelayEmbeddingEncoder`.
    n_delays : int
        Requested delay count. ``1`` leaves a bare encoder unchanged.

    Returns
    -------
    encoder : nn.Module
        Possibly wrapped encoder.
    n_delays : int
        Effective delay count.

    Raises
    ------
    ValueError
        If ``n_delays`` conflicts with an existing wrapper or base sizing.
    TypeError
        If ``encoder`` is not a supported type for auto-wrapping.
    """
    from koopman_graph.nn.encoder import GATEncoder, GNNEncoder

    if n_delays < 1:
        msg = f"n_delays must be >= 1, got {n_delays}"
        raise ValueError(msg)

    if isinstance(encoder, DelayEmbeddingEncoder):
        if n_delays != encoder.n_delays and n_delays != 1:
            msg = (
                f"n_delays={n_delays} conflicts with DelayEmbeddingEncoder."
                f"n_delays={encoder.n_delays}"
            )
            raise ValueError(msg)
        return encoder, encoder.n_delays

    if n_delays == 1:
        return encoder, 1

    if not isinstance(encoder, (GNNEncoder, GATEncoder)):
        msg = (
            "n_delays > 1 requires a GNNEncoder/GATEncoder (to wrap) or an "
            f"existing DelayEmbeddingEncoder; got {type(encoder).__name__}"
        )
        raise TypeError(msg)

    _feature_dim_from_base(encoder, n_delays)
    return DelayEmbeddingEncoder(encoder, n_delays=n_delays), n_delays


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
