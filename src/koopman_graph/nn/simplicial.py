"""Simplicial-1 encoder / decoder peers (combinatorial ``L_1 = B_1 B_1^T``).

Uses oriented ``edge_index`` (one column per undirected edge) and optional
triangular ``face_index`` on ``Data`` (validated only; does not alter
``B_1``). Checkpoint type strings ``sim_enc`` / ``sim_dec`` are registered in
the format-1 schema.

These peers are **not** sheaf Laplacians, cell-complex layers, or 2-form
Hodge operators — see :doc:`limitations` (Sphinx honesty in Phase 61).
"""

from __future__ import annotations

from collections.abc import Callable

from torch import Tensor, nn
from torch_geometric.data import Data

from koopman_graph.nn.gnn import (
    ActivationName,
    BaseGNNModule,
    validate_positive_dims,
)
from koopman_graph.observables import (
    coerce_face_index,
    simplicial_one_laplacian_matvec,
)

DecoderFn = Callable[[Tensor, Tensor, Tensor | None], Tensor]


class SimplicialConv(nn.Module):
    """One combinatorial ``L_1`` mixing layer: ``Linear(L_1 @ x)``.

    Parameters
    ----------
    in_channels : int
        Input feature width.
    out_channels : int
        Output feature width.
    residual : bool, optional
        Add a residual skip when ``in_channels == out_channels``. Default
        is ``False``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        residual: bool = False,
    ) -> None:
        """Initialize the linear map after ``L_1`` mixing.

        Parameters
        ----------
        in_channels
            Value for ``in_channels``.
        out_channels
            Value for ``out_channels``.
        residual
            Value for ``residual``.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.residual = bool(residual) and in_channels == out_channels
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Apply ``L_1`` then a learned linear map.

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
        mixed = simplicial_one_laplacian_matvec(
            edge_index,
            x,
            num_nodes=int(x.size(0)),
        )
        out = self.lin(mixed)
        if self.residual:
            out = out + x
        return out


def build_simplicial_convs(
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    num_layers: int,
    *,
    residual: bool = False,
) -> nn.ModuleList:
    """Build a stack of :class:`SimplicialConv` layers.

    Parameters
    ----------
    in_channels
        Value for ``in_channels``.
    hidden_channels
        Value for ``hidden_channels``.
    out_channels
        Value for ``out_channels``.
    num_layers
        Value for ``num_layers``.
    residual
        Value for ``residual``.

    Returns
    -------
    object
        Function result.
    """
    if num_layers == 1:
        return nn.ModuleList(
            [SimplicialConv(in_channels, out_channels, residual=residual)]
        )
    layers: list[nn.Module] = [
        SimplicialConv(in_channels, hidden_channels, residual=residual)
    ]
    for _ in range(num_layers - 2):
        layers.append(
            SimplicialConv(hidden_channels, hidden_channels, residual=residual)
        )
    layers.append(SimplicialConv(hidden_channels, out_channels, residual=False))
    return nn.ModuleList(layers)


def _resolve_simplicial_forward_inputs(
    x_or_data: Tensor | Data,
    edge_index: Tensor | None,
    edge_weight: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Resolve node features, oriented edges, and optional faces.

    Parameters
    ----------
    x_or_data
        Value for ``x_or_data``.
    edge_index
        Value for ``edge_index``.
    edge_weight
        Value for ``edge_weight``.

    Returns
    -------
    object
        Function result.
    """
    del edge_weight  # unused — combinatorial L1
    if isinstance(x_or_data, Data):
        if x_or_data.x is None:
            msg = "data.x is required for SimplicialEncoder / SimplicialDecoder"
            raise ValueError(msg)
        if x_or_data.edge_index is None:
            msg = (
                "data.edge_index is required for SimplicialEncoder / SimplicialDecoder"
            )
            raise ValueError(msg)
        face_index = getattr(x_or_data, "face_index", None)
        if face_index is not None:
            face_index = coerce_face_index(
                face_index,
                num_nodes=int(x_or_data.x.size(0)),
            )
        return x_or_data.x, x_or_data.edge_index, face_index
    if edge_index is None:
        msg = (
            "edge_index is required when x_or_data is a tensor for "
            "SimplicialEncoder / SimplicialDecoder"
        )
        raise ValueError(msg)
    return x_or_data, edge_index, None


def _simplicial_message_passing(
    module: BaseGNNModule,
    x: Tensor,
    edge_index: Tensor,
) -> Tensor:
    """Run stacked :class:`SimplicialConv` layers with hidden activations.

    Parameters
    ----------
    module
        Value for ``module``.
    x
        Value for ``x``.
    edge_index
        Value for ``edge_index``.

    Returns
    -------
    object
        Function result.
    """
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
        if not isinstance(conv, SimplicialConv):
            msg = (
                f"expected SimplicialConv in simplicial stack, "
                f"got {type(conv).__name__}"
            )
            raise TypeError(msg)
        x = conv(x, edge_index)
        if layer_idx < len(module.convs) - 1:
            x = module.activation(x)
    return x


class SimplicialEncoder(BaseGNNModule):
    """Simplicial-1 encoder using combinatorial ``L_1`` mixing.

    Attributes
    ----------
    in_channels : int
        Input node feature dimension.
    hidden_channels : int
        Hidden channel width.
    latent_dim : int
        Output latent dimension per node.
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
    ) -> None:
        """Initialize the simplicial-1 encoder stack.

        Parameters
        ----------
        in_channels
            Value for ``in_channels``.
        hidden_channels
            Value for ``hidden_channels``.
        latent_dim
            Value for ``latent_dim``.
        num_layers
            Value for ``num_layers``.
        activation
            Value for ``activation``.
        residual
            Value for ``residual``.
        """
        validate_positive_dims(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            latent_dim=latent_dim,
            num_layers=num_layers,
        )
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.latent_dim = latent_dim
        self.residual = bool(residual)
        super().__init__(
            input_channels=in_channels,
            input_dim_name="in_channels",
            num_layers=num_layers,
            activation=activation,
            convs=build_simplicial_convs(
                in_channels,
                hidden_channels,
                latent_dim,
                num_layers,
                residual=residual,
            ),
        )

    def forward(
        self,
        x_or_data: Tensor | Data,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        """Encode node features with combinatorial ``L_1`` mixing.

        Parameters
        ----------
        x_or_data
            Value for ``x_or_data``.
        edge_index
            Value for ``edge_index``.
        edge_weight
            Value for ``edge_weight``.

        Returns
        -------
        object
            Function result.
        """
        x, resolved_edges, _face = _resolve_simplicial_forward_inputs(
            x_or_data,
            edge_index,
            edge_weight,
        )
        return _simplicial_message_passing(self, x, resolved_edges)


class SimplicialDecoder(BaseGNNModule):
    """Simplicial-1 decoder using combinatorial ``L_1`` mixing.

    Attributes
    ----------
    latent_dim : int
        Input latent dimension per node.
    hidden_channels : int
        Hidden channel width.
    out_channels : int
        Output physical feature dimension per node.
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
    ) -> None:
        """Initialize the simplicial-1 decoder stack.

        Parameters
        ----------
        latent_dim
            Value for ``latent_dim``.
        hidden_channels
            Value for ``hidden_channels``.
        out_channels
            Value for ``out_channels``.
        num_layers
            Value for ``num_layers``.
        activation
            Value for ``activation``.
        residual
            Value for ``residual``.
        """
        validate_positive_dims(
            latent_dim=latent_dim,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
        )
        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.residual = bool(residual)
        super().__init__(
            input_channels=latent_dim,
            input_dim_name="latent_dim",
            num_layers=num_layers,
            activation=activation,
            convs=build_simplicial_convs(
                latent_dim,
                hidden_channels,
                out_channels,
                num_layers,
                residual=residual,
            ),
        )

    def forward(
        self,
        x_or_data: Tensor | Data,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        """Decode latent features with combinatorial ``L_1`` mixing.

        Parameters
        ----------
        x_or_data
            Value for ``x_or_data``.
        edge_index
            Value for ``edge_index``.
        edge_weight
            Value for ``edge_weight``.

        Returns
        -------
        object
            Function result.
        """
        x, resolved_edges, _face = _resolve_simplicial_forward_inputs(
            x_or_data,
            edge_index,
            edge_weight,
        )
        return _simplicial_message_passing(self, x, resolved_edges)


def bind_simplicial_decoder(
    decoder: SimplicialDecoder,
    edge_index: Tensor,
    face_index: Tensor | None = None,
) -> DecoderFn:
    """Bind static oriented edges (and optional faces) into a decoder closure.

    Parameters
    ----------
    decoder : SimplicialDecoder
        Simplicial decoder instance.
    edge_index : Tensor
        Static oriented edges ``(2, num_edges)``.
    face_index : Tensor or None, optional
        Optional triangular faces ``(3, num_faces)`` for validation on
        ``Data``-style call sites; unused for tensor decode.

    Returns
    -------
    callable
        ``decoder(z, edge_index, edge_weight) -> Tensor`` compatible with
        autoregressive rollout helpers.
    """
    del face_index  # reserved for Data validation; tensor path uses bound edges

    return lambda z, _edge_index, _edge_weight: decoder(z, edge_index, None)
