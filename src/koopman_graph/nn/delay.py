"""Delay-coordinate encoder wrapper for partial observability.

Stacks a short history of node features into encoder channels (Takens-style
delay coordinates). Related Hankel delay matrices appear in HAVOK / Hankel-DMD;
this module does not implement those full algorithms.

Composition style (see architecture docs): callers size the base encoder with
``in_channels = n_delays * feature_dim`` and wrap it explicitly, or pass
``n_delays`` to :class:`~koopman_graph.model.GraphKoopmanModel` so the model
wraps without rebuilding layers.

Pure snapshot / tensor delay-window helpers live in
:mod:`koopman_graph.data.delay_windows` and are thin-re-exported here for
stable ``nn.delay`` import paths.
"""

from __future__ import annotations

from torch import Tensor, nn
from torch_geometric.data import Data

from koopman_graph.data.delay_windows import (
    apply_observation_mask_to_features,
    flatten_delay_window,
    history_from_snapshots,
    stack_delay_features,
)
from koopman_graph.graph_utils import resolve_edge_index, resolve_edge_weight

__all__ = [
    "DelayEmbeddingEncoder",
    "apply_observation_mask_to_features",
    "flatten_delay_window",
    "history_from_snapshots",
    "resolve_delay_encoder",
    "stack_delay_features",
]


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
    from koopman_graph.nn.encoder import (
        DiffConvEncoder,
        GATEncoder,
        GNNEncoder,
        GraphTransformerEncoder,
        SAGEEncoder,
    )

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

    if not isinstance(
        encoder,
        (
            GNNEncoder,
            GATEncoder,
            SAGEEncoder,
            DiffConvEncoder,
            GraphTransformerEncoder,
        ),
    ):
        msg = (
            "n_delays > 1 requires a GNNEncoder/GATEncoder/SAGEEncoder/"
            "DiffConvEncoder/GraphTransformerEncoder (to wrap) or an existing "
            "DelayEmbeddingEncoder; "
            f"got {type(encoder).__name__}"
        )
        raise TypeError(msg)

    _feature_dim_from_base(encoder, n_delays)
    return DelayEmbeddingEncoder(encoder, n_delays=n_delays), n_delays
