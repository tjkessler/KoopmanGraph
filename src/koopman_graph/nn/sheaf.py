"""Sheaf encoder / decoder peers (diagonal default; optional general maps).

Implements a teaching sheaf Laplacian on oriented ``edge_index`` (same
``B_1`` edge convention as the simplicial-1 peers). Restriction maps default
to **diagonal** per feature channel (``O(C)`` params/layer). Opt-in
``restriction_maps="general"`` learns dense ``C×C`` maps (``O(C^2)``
params/layer) with a hard channel ceiling. Checkpoint type strings
``sheaf_enc`` / ``sheaf_dec`` are registered in the format-1 schema. Factory
kind ``encoder="sheaf"`` builds a matched peer pair via
:func:`~koopman_graph.model.factory.build_encoder_peers`.

Latent Koopman operators stay linear — sheaf nonlinearities live only in
encode / decode.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch
from torch import Tensor, nn
from torch_geometric.data import Data

from koopman_graph.nn.gnn import (
    ActivationName,
    BaseGNNModule,
    validate_positive_dims,
)
from koopman_graph.observables import (
    diagonal_sheaf_laplacian_matvec,
    general_sheaf_laplacian_matvec,
)

DecoderFn = Callable[[Tensor, Tensor, Tensor | None], Tensor]
RestrictionMaps = Literal["diagonal", "general"]

# Refuse general maps above this feature width (``2 C^2`` params/layer).
MAX_GENERAL_SHEAF_CHANNELS = 32


def _validate_restriction_maps(restriction_maps: str) -> RestrictionMaps:
    """Normalize and validate the restriction-map kind string.

    Parameters
    ----------
    restriction_maps : str
        Requested restriction-map mode.

    Returns
    -------
    RestrictionMaps
        Validated ``\"diagonal\"`` or ``\"general\"`` literal.

    Raises
    ------
    ValueError
        If ``restriction_maps`` is not supported.
    """
    if restriction_maps not in ("diagonal", "general"):
        msg = (
            f"restriction_maps must be 'diagonal' or 'general', "
            f"got {restriction_maps!r}"
        )
        raise ValueError(msg)
    return restriction_maps  # type: ignore[return-value]


def _check_general_channel_ceiling(channels: int) -> None:
    """Raise if general restriction maps exceed the documented size ceiling.

    Parameters
    ----------
    channels
        See signature."""
    if channels > MAX_GENERAL_SHEAF_CHANNELS:
        msg = (
            f"restriction_maps='general' requires channels ≤ "
            f"{MAX_GENERAL_SHEAF_CHANNELS} (got {channels}); "
            f"parameter cost is 2·C² per SheafConv layer — use "
            f"restriction_maps='diagonal' or reduce width"
        )
        raise ValueError(msg)


class SheafConv(nn.Module):
    """One sheaf mixing layer: ``Linear(L_F @ x)``.

    Parameters
    ----------
    in_channels : int
        Input feature width.
    out_channels : int
        Output feature width.
    residual : bool, optional
        Add a residual skip when ``in_channels == out_channels``. Default
        is ``False``.
    restriction_maps : {"diagonal", "general"}, optional
        ``\"diagonal\"`` (default) uses ``O(C)`` diagonal maps;
        ``\"general\"`` uses dense ``C×C`` maps (``O(C^2)``) with ceiling
        :data:`MAX_GENERAL_SHEAF_CHANNELS`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        residual: bool = False,
        restriction_maps: RestrictionMaps = "diagonal",
    ) -> None:
        """Initialize restriction maps and the post-mix linear map.

        Parameters
        ----------
        in_channels : int
            Input feature width.
        out_channels : int
            Output feature width.
        residual : bool, optional
            Residual skip when widths match.
        restriction_maps : {"diagonal", "general"}, optional
            Restriction-map parameterization.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.residual = bool(residual) and in_channels == out_channels
        self.restriction_maps = _validate_restriction_maps(restriction_maps)
        if self.restriction_maps == "general":
            _check_general_channel_ceiling(in_channels)
            # Identity init → L_F matches combinatorial L_1 at step 0.
            self.source_map = nn.Parameter(torch.eye(in_channels))
            self.target_map = nn.Parameter(torch.eye(in_channels))
            self.source_diag = None
            self.target_diag = None
        else:
            self.source_diag = nn.Parameter(torch.ones(in_channels))
            self.target_diag = nn.Parameter(torch.ones(in_channels))
            self.source_map = None
            self.target_map = None
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Apply sheaf ``L_F`` then a learned linear map.

        Parameters
        ----------
        x : Tensor
            Node features ``(num_nodes, in_channels)``.
        edge_index : Tensor
            Oriented edges ``(2, num_edges)``.

        Returns
        -------
        Tensor
            Mixed features ``(num_nodes, out_channels)``.
        """
        if self.restriction_maps == "general":
            assert self.source_map is not None and self.target_map is not None
            mixed = general_sheaf_laplacian_matvec(
                edge_index,
                x,
                self.source_map,
                self.target_map,
                num_nodes=int(x.size(0)),
            )
        else:
            assert self.source_diag is not None and self.target_diag is not None
            mixed = diagonal_sheaf_laplacian_matvec(
                edge_index,
                x,
                self.source_diag,
                self.target_diag,
                num_nodes=int(x.size(0)),
            )
        out = self.lin(mixed)
        if self.residual:
            out = out + x
        return out


def build_sheaf_convs(
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    num_layers: int,
    *,
    residual: bool = False,
    restriction_maps: RestrictionMaps = "diagonal",
) -> nn.ModuleList:
    """Build a stack of :class:`SheafConv` layers.

    Parameters
    ----------
    in_channels : int
        Input width.
    hidden_channels : int
        Hidden width.
    out_channels : int
        Output width.
    num_layers : int
        Stack depth.
    residual : bool, optional
        Residual skips on hidden layers when widths match.
    restriction_maps : {"diagonal", "general"}, optional
        Restriction-map parameterization for every layer.

    Returns
    -------
    nn.ModuleList
        Ordered sheaf convolution stack.
    """
    kind = _validate_restriction_maps(restriction_maps)
    if num_layers == 1:
        return nn.ModuleList(
            [
                SheafConv(
                    in_channels,
                    out_channels,
                    residual=residual,
                    restriction_maps=kind,
                )
            ]
        )
    layers: list[nn.Module] = [
        SheafConv(
            in_channels,
            hidden_channels,
            residual=residual,
            restriction_maps=kind,
        )
    ]
    for _ in range(num_layers - 2):
        layers.append(
            SheafConv(
                hidden_channels,
                hidden_channels,
                residual=residual,
                restriction_maps=kind,
            )
        )
    layers.append(
        SheafConv(
            hidden_channels,
            out_channels,
            residual=False,
            restriction_maps=kind,
        )
    )
    return nn.ModuleList(layers)


def _resolve_sheaf_forward_inputs(
    x_or_data: Tensor | Data,
    edge_index: Tensor | None,
    edge_weight: Tensor | None,
) -> tuple[Tensor, Tensor]:
    """Resolve node features and oriented edges for sheaf peers.

    Parameters
    ----------
    x_or_data
        See signature.
    edge_index
        See signature.
    edge_weight
        See signature.

    Returns
    -------
        See signature."""
    del edge_weight  # unused — combinatorial / sheaf L_F on topology only
    if isinstance(x_or_data, Data):
        if x_or_data.x is None:
            msg = "data.x is required for SheafGNNEncoder / SheafGNNDecoder"
            raise ValueError(msg)
        if x_or_data.edge_index is None:
            msg = "data.edge_index is required for SheafGNNEncoder / SheafGNNDecoder"
            raise ValueError(msg)
        return x_or_data.x, x_or_data.edge_index
    if edge_index is None:
        msg = (
            "edge_index is required when x_or_data is a tensor for "
            "SheafGNNEncoder / SheafGNNDecoder"
        )
        raise ValueError(msg)
    return x_or_data, edge_index


def _sheaf_message_passing(
    module: BaseGNNModule,
    x: Tensor,
    edge_index: Tensor,
) -> Tensor:
    """Run stacked :class:`SheafConv` layers with hidden activations.

    Parameters
    ----------
    module
        See signature.
    x
        See signature.
    edge_index
        See signature.

    Returns
    -------
        See signature."""
    if x.ndim != 2:
        msg = (
            f"Expected x with shape (num_nodes, {module.input_dim_name}), "
            f"got {tuple(x.shape)}"
        )
        raise ValueError(msg)
    if x.shape[1] != module.input_channels:
        msg = (
            f"Expected {module.input_dim_name}={module.input_channels}, "
            f"got feature dimension {x.shape[1]}"
        )
        raise ValueError(msg)
    for layer_idx, conv in enumerate(module.convs):
        if not isinstance(conv, SheafConv):
            msg = f"expected SheafConv in sheaf stack, got {type(conv).__name__}"
            raise TypeError(msg)
        x = conv(x, edge_index)
        if layer_idx < len(module.convs) - 1:
            x = module.activation(x)
    return x


class SheafGNNEncoder(BaseGNNModule):
    """Sheaf encoder with diagonal (default) or general restriction maps.

    Diagonal maps cost ``2 C`` parameters per :class:`SheafConv` layer; general
    maps cost ``2 C²`` and are refused when any layer input width exceeds
    :data:`MAX_GENERAL_SHEAF_CHANNELS`.

    Attributes
    ----------
    in_channels : int
        Input node feature dimension.
    hidden_channels : int
        Hidden channel width.
    latent_dim : int
        Output latent dimension per node.
    restriction_maps : {"diagonal", "general"}
        Restriction-map parameterization.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        latent_dim: int,
        *,
        num_layers: int = 2,
        activation: ActivationName = "relu",
        residual: bool = False,
        restriction_maps: RestrictionMaps = "diagonal",
    ) -> None:
        """Initialize the sheaf encoder stack.

        Parameters
        ----------
        in_channels : int
            Input node feature dimension.
        hidden_channels : int
            Hidden channel width.
        latent_dim : int
            Output latent dimension per node.
        num_layers : int, optional
            Stack depth. Default is ``2``.
        activation : str, optional
            Hidden activation name. Default is ``\"relu\"``.
        residual : bool, optional
            Residual skips when widths match. Default is ``False``.
        restriction_maps : {"diagonal", "general"}, optional
            Restriction-map parameterization. Default is ``\"diagonal\"``.
        """
        validate_positive_dims(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            latent_dim=latent_dim,
            num_layers=num_layers,
        )
        kind = _validate_restriction_maps(restriction_maps)
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.latent_dim = latent_dim
        self.residual = bool(residual)
        self.restriction_maps = kind
        super().__init__(
            input_channels=in_channels,
            input_dim_name="in_channels",
            num_layers=num_layers,
            activation=activation,
            convs=build_sheaf_convs(
                in_channels,
                hidden_channels,
                latent_dim,
                num_layers,
                residual=residual,
                restriction_maps=kind,
            ),
        )

    def forward(
        self,
        x_or_data: Tensor | Data,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        """Encode node features with a sheaf Laplacian mix.

        Parameters
        ----------
        x_or_data : Tensor or Data
            Node features or a PyG ``Data`` graph.
        edge_index : Tensor or None, optional
            Oriented edges when ``x_or_data`` is a tensor.
        edge_weight : Tensor or None, optional
            Ignored (topology-only sheaf mix).

        Returns
        -------
        Tensor
            Latents with shape ``(num_nodes, latent_dim)``.
        """
        x, resolved_edges = _resolve_sheaf_forward_inputs(
            x_or_data,
            edge_index,
            edge_weight,
        )
        return _sheaf_message_passing(self, x, resolved_edges)


class SheafGNNDecoder(BaseGNNModule):
    """Sheaf decoder with diagonal (default) or general restriction maps.

    Attributes
    ----------
    latent_dim : int
        Input latent dimension per node.
    hidden_channels : int
        Hidden channel width.
    out_channels : int
        Output physical feature dimension per node.
    restriction_maps : {"diagonal", "general"}
        Restriction-map parameterization.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_channels: int,
        out_channels: int,
        *,
        num_layers: int = 2,
        activation: ActivationName = "relu",
        residual: bool = False,
        restriction_maps: RestrictionMaps = "diagonal",
    ) -> None:
        """Initialize the sheaf decoder stack.

        Parameters
        ----------
        latent_dim : int
            Input latent dimension per node.
        hidden_channels : int
            Hidden channel width.
        out_channels : int
            Output feature dimension.
        num_layers : int, optional
            Stack depth. Default is ``2``.
        activation : str, optional
            Hidden activation name. Default is ``\"relu\"``.
        residual : bool, optional
            Residual skips when widths match. Default is ``False``.
        restriction_maps : {"diagonal", "general"}, optional
            Restriction-map parameterization. Default is ``\"diagonal\"``.
        """
        validate_positive_dims(
            latent_dim=latent_dim,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
        )
        kind = _validate_restriction_maps(restriction_maps)
        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.residual = bool(residual)
        self.restriction_maps = kind
        super().__init__(
            input_channels=latent_dim,
            input_dim_name="latent_dim",
            num_layers=num_layers,
            activation=activation,
            convs=build_sheaf_convs(
                latent_dim,
                hidden_channels,
                out_channels,
                num_layers,
                residual=residual,
                restriction_maps=kind,
            ),
        )

    def forward(
        self,
        x_or_data: Tensor | Data,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        """Decode latent features with a sheaf Laplacian mix.

        Parameters
        ----------
        x_or_data : Tensor or Data
            Latents or a PyG ``Data`` graph.
        edge_index : Tensor or None, optional
            Oriented edges when ``x_or_data`` is a tensor.
        edge_weight : Tensor or None, optional
            Ignored (topology-only sheaf mix).

        Returns
        -------
        Tensor
            Reconstructions with shape ``(num_nodes, out_channels)``.
        """
        x, resolved_edges = _resolve_sheaf_forward_inputs(
            x_or_data,
            edge_index,
            edge_weight,
        )
        return _sheaf_message_passing(self, x, resolved_edges)


def bind_sheaf_decoder(
    decoder: SheafGNNDecoder,
    edge_index: Tensor,
) -> DecoderFn:
    """Bind static oriented edges into a decoder closure.

    Parameters
    ----------
    decoder : SheafGNNDecoder
        Sheaf decoder instance.
    edge_index : Tensor
        Static oriented edges ``(2, num_edges)``.

    Returns
    -------
    callable
        ``decoder(z, edge_index, edge_weight) -> Tensor`` compatible with
        autoregressive rollout helpers.
    """
    return lambda z, _edge_index, _edge_weight: decoder(z, edge_index, None)
