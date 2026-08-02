"""Hypergraph encoder / decoder peers (PyG ``HypergraphConv``).

Encoder and decoder import shared builders from :mod:`koopman_graph.nn.gnn`
only; they do not import each other. Checkpoint type strings ``hyper_enc`` /
``hyper_dec`` are registered in the format-1 schema.

These stacks are **undirected**: message passing uses bipartite
``hyperedge_index`` (and optional ``hyperedge_weight``) through
:class:`~torch_geometric.nn.HypergraphConv`. They do not consume directed
``tail_index`` / ``head_index``. Pairing them with a
:class:`~koopman_graph.operators.HypergraphKoopmanOperator` that uses a
directed ``incidence_mode`` is supported and intentional — encode and
latent advance need not share an orientation. Directed operator modes
implement one documented random-walk normalization; they are not claimed
to be the unique literature choice, nor equivalent to simplicial or Hodge
operators.
"""

from __future__ import annotations

from collections.abc import Callable

from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import HypergraphConv

from koopman_graph.graph_utils import (
    snapshot_hyperedge_index,
    snapshot_hyperedge_weight,
)
from koopman_graph.nn.gnn import (
    ActivationName,
    BaseGNNModule,
    build_hypergraph_convs,
    validate_positive_dims,
)

DecoderFn = Callable[[Tensor, Tensor, Tensor | None], Tensor]


def _resolve_hypergraph_forward_inputs(
    x_or_data: Tensor | Data,
    edge_index: Tensor | None,
    edge_weight: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Resolve node features and hyperedge incidence for a forward pass.

    When ``x_or_data`` is a :class:`~torch_geometric.data.Data` snapshot, the
    incidence is read from ``hyperedge_index`` / ``hyperedge_weight`` and any
    positional topology arguments are ignored (call sites often pass pairwise
    ``edge_index`` for API symmetry with GNN peers). When ``x_or_data`` is a
    tensor, the positional arguments are the hyperedge incidence and optional
    hyperedge weights.

    Parameters
    ----------
    x_or_data : Tensor or Data
        Node features or a PyG snapshot carrying ``hyperedge_index``.
    edge_index : Tensor or None
        Hyperedge incidence for tensor input; ignored for ``Data`` input.
    edge_weight : Tensor or None
        Optional hyperedge weights for tensor input; ignored for ``Data``.

    Returns
    -------
    tuple of (Tensor, Tensor, Tensor or None)
        Node features, hyperedge incidence, and optional hyperedge weights.

    Raises
    ------
    ValueError
        If hyperedge incidence is missing for the given input form.
    """
    if isinstance(x_or_data, Data):
        hyperedge_index = snapshot_hyperedge_index(x_or_data)
        if hyperedge_index is None:
            msg = (
                "hyperedge_index is required on Data input for "
                "HypergraphEncoder / HypergraphDecoder"
            )
            raise ValueError(msg)
        return (
            x_or_data.x,
            hyperedge_index,
            snapshot_hyperedge_weight(x_or_data),
        )
    if edge_index is None:
        msg = (
            "hyperedge_index is required when x_or_data is a tensor for "
            "HypergraphEncoder / HypergraphDecoder"
        )
        raise ValueError(msg)
    return x_or_data, edge_index, edge_weight


def _hypergraph_message_passing(
    module: BaseGNNModule,
    x: Tensor,
    hyperedge_index: Tensor,
    hyperedge_weight: Tensor | None,
) -> Tensor:
    """Run stacked :class:`~torch_geometric.nn.HypergraphConv` layers.

    Parameters
    ----------
    module : BaseGNNModule
        Hypergraph encoder or decoder owning ``convs`` and activation.
    x : Tensor
        Node features with shape ``(num_nodes, channels)``.
    hyperedge_index : Tensor
        Bipartite incidence with shape ``(2, nnz)``.
    hyperedge_weight : Tensor or None
        Optional hyperedge weights with shape ``(num_hyperedges,)``.

    Returns
    -------
    Tensor
        Transformed node features after the stack.

    Raises
    ------
    ValueError
        If ``x`` has the wrong rank or channel dimension.
    TypeError
        If ``module.convs`` contains a non-``HypergraphConv`` layer.
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
        if not isinstance(conv, HypergraphConv):
            msg = (
                f"expected HypergraphConv in hypergraph stack, "
                f"got {type(conv).__name__}"
            )
            raise TypeError(msg)
        x = conv(x, hyperedge_index, hyperedge_weight=hyperedge_weight)
        if layer_idx < len(module.convs) - 1:
            x = module.activation(x)
    return x


class HypergraphEncoder(BaseGNNModule):
    """Hypergraph encoder that lifts node features into a latent space.

    Applies stacked :class:`~torch_geometric.nn.HypergraphConv` layers with a
    configurable hidden activation. The final layer maps to ``latent_dim``
    without an activation. Pairwise ``edge_index`` on ``Data`` inputs is
    ignored; incidence comes from undirected ``hyperedge_index`` /
    ``hyperedge_weight`` only (directed ``tail_index`` / ``head_index`` are
    not used here). When the paired Koopman operator uses a directed
    ``incidence_mode``, encode and advance orientations may differ; see
    :class:`~koopman_graph.operators.HypergraphKoopmanOperator`.

    Attributes
    ----------
    in_channels : int
        Input node feature dimension.
    hidden_channels : int
        Hidden hypergraph channel width.
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
    ) -> None:
        """Initialize the hypergraph encoder stack.

        Parameters
        ----------
        in_channels : int
            Input node feature dimension.
        hidden_channels : int
            Hidden channel width for intermediate layers.
        latent_dim : int
            Output latent dimension per node.
        num_layers : int, optional
            Number of hypergraph convolution layers. Default is ``2``.
        activation : {"relu", "sigmoid", "tanh"}, optional
            Hidden-layer activation. Default is ``"relu"``.

        Raises
        ------
        ValueError
            If any dimension argument is not positive.
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

        super().__init__(
            input_channels=in_channels,
            input_dim_name="in_channels",
            num_layers=num_layers,
            activation=activation,
            convs=build_hypergraph_convs(
                in_channels,
                hidden_channels,
                latent_dim,
                num_layers,
            ),
        )

    def forward(
        self,
        x_or_data: Tensor | Data,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        """Encode node features with hyperedge-mediated message passing.

        Parameters
        ----------
        x_or_data : Tensor or Data
            Node features or a PyG ``Data`` carrying ``hyperedge_index``.
        edge_index : Tensor or None, optional
            Hyperedge incidence ``(2, nnz)`` when ``x_or_data`` is a tensor.
            Ignored for ``Data`` input (use ``Data.hyperedge_index``).
        edge_weight : Tensor or None, optional
            Hyperedge weights ``(num_hyperedges,)`` for tensor input; ignored
            for ``Data`` input (use ``Data.hyperedge_weight``).

        Returns
        -------
        Tensor
            Latent node features with shape ``(num_nodes, latent_dim)``.
        """
        x, hyperedge_index, hyperedge_weight = _resolve_hypergraph_forward_inputs(
            x_or_data,
            edge_index,
            edge_weight,
        )
        return _hypergraph_message_passing(
            self,
            x,
            hyperedge_index,
            hyperedge_weight,
        )


class HypergraphDecoder(BaseGNNModule):
    """Hypergraph decoder that maps latent node features to physical space.

    Applies stacked :class:`~torch_geometric.nn.HypergraphConv` layers with a
    configurable hidden activation. The final layer maps to ``out_channels``
    without an activation. Like the encoder, decoding uses undirected
    ``hyperedge_index`` / ``hyperedge_weight`` only; it does not consume
    directed incidence. Orientation asymmetry relative to a directed
    Koopman ``incidence_mode`` is therefore possible and intentional.

    Attributes
    ----------
    latent_dim : int
        Input latent dimension per node.
    hidden_channels : int
        Hidden hypergraph channel width.
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
    ) -> None:
        """Initialize the hypergraph decoder stack.

        Parameters
        ----------
        latent_dim : int
            Input latent dimension per node.
        hidden_channels : int
            Hidden channel width for intermediate layers.
        out_channels : int
            Output physical feature dimension per node.
        num_layers : int, optional
            Number of hypergraph convolution layers. Default is ``2``.
        activation : {"relu", "sigmoid", "tanh"}, optional
            Hidden-layer activation. Default is ``"relu"``.

        Raises
        ------
        ValueError
            If any dimension argument is not positive.
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

        super().__init__(
            input_channels=latent_dim,
            input_dim_name="latent_dim",
            num_layers=num_layers,
            activation=activation,
            convs=build_hypergraph_convs(
                latent_dim,
                hidden_channels,
                out_channels,
                num_layers,
            ),
        )

    def forward(
        self,
        x_or_data: Tensor | Data,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        """Decode latent features with hyperedge-mediated message passing.

        Parameters
        ----------
        x_or_data : Tensor or Data
            Latent node features or a ``Data`` carrying ``hyperedge_index``.
        edge_index : Tensor or None, optional
            Hyperedge incidence ``(2, nnz)`` when ``x_or_data`` is a tensor.
            Ignored for ``Data`` input.
        edge_weight : Tensor or None, optional
            Hyperedge weights for tensor input; ignored for ``Data`` input.

        Returns
        -------
        Tensor
            Physical node features with shape ``(num_nodes, out_channels)``.
        """
        x, hyperedge_index, hyperedge_weight = _resolve_hypergraph_forward_inputs(
            x_or_data,
            edge_index,
            edge_weight,
        )
        return _hypergraph_message_passing(
            self,
            x,
            hyperedge_index,
            hyperedge_weight,
        )


def bind_hypergraph_decoder(
    decoder: HypergraphDecoder,
    hyperedge_index: Tensor,
    hyperedge_weight: Tensor | None = None,
) -> DecoderFn:
    """Bind static hyperedge incidence into a ``DecoderFn`` closure.

    Autoregressive helpers pass pairwise ``(edge_index, edge_weight)`` to
    decoders; hypergraph peers ignore those pairwise args and use the bound
    incidence instead.

    Parameters
    ----------
    decoder : HypergraphDecoder
        Hypergraph decoder instance.
    hyperedge_index : Tensor
        Static bipartite incidence with shape ``(2, nnz)``.
    hyperedge_weight : Tensor or None, optional
        Optional hyperedge weights with shape ``(num_hyperedges,)``.

    Returns
    -------
    callable
        ``decoder(z, edge_index, edge_weight) -> Tensor`` compatible with
        :func:`~koopman_graph.graph_utils.autoregressive_latent_rollout`.
    """

    return lambda z, _edge_index, _edge_weight: decoder(
        z,
        hyperedge_index,
        hyperedge_weight,
    )
