"""Graph Neural Network decoders for latent-to-physical reconstruction."""

from __future__ import annotations

from koopman_graph.nn.gnn import (
    ActivationName,
    BaseGNNModule,
    build_gat_convs,
    build_gcn_convs,
    validate_gat_attention,
    validate_positive_dims,
)


class GNNDecoder(BaseGNNModule):
    """GCN decoder that maps latent node features back to physical space.

    Applies stacked Graph Convolutional Network layers with configurable hidden
    activation. The final layer maps directly to ``out_channels`` without an
    activation, producing per-node physical feature predictions.

    Attributes
    ----------
    latent_dim : int
        Input latent dimension per node.
    hidden_channels : int
        Hidden GCN channel width.
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
        """Initialize the GCN decoder stack.

        Parameters
        ----------
        latent_dim : int
            Input latent dimension per node.
        hidden_channels : int
            Hidden GCN channel width for intermediate layers.
        out_channels : int
            Output physical feature dimension per node.
        num_layers : int, optional
            Number of GCN layers. Default is ``2``.
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
            convs=build_gcn_convs(
                latent_dim,
                hidden_channels,
                out_channels,
                num_layers,
            ),
        )


class GATDecoder(BaseGNNModule):
    """GAT decoder that maps latent node features back to physical space.

    Applies stacked Graph Attention Network layers with configurable hidden
    activation. The final layer maps directly to ``out_channels`` without an
    activation, producing per-node physical feature predictions.

    Constructor arguments mirror :class:`~koopman_graph.nn.encoder.GATEncoder`
    with the I/O dimensions swapped (``latent_dim`` in, ``out_channels`` out)
    and the same ``heads`` / ``dropout`` / activation defaults.

    Scalar ``edge_weight`` arguments are accepted for API symmetry with
    :class:`~koopman_graph.nn.decoder.GNNDecoder` but are ignored because
    :class:`~torch_geometric.nn.GATConv` does not consume scalar edge weights.
    Multi-dimensional edge *feature* vectors remain unsupported.

    Attributes
    ----------
    latent_dim : int
        Input latent dimension per node.
    hidden_channels : int
        Hidden GAT channel width.
    out_channels : int
        Output physical feature dimension per node.
    heads : int
        Number of attention heads per GAT layer.
    dropout : float
        Dropout probability inside GAT attention.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_channels: int,
        out_channels: int,
        *,
        num_layers: int = 2,
        activation: ActivationName = "relu",
        heads: int = 1,
        dropout: float = 0.0,
    ) -> None:
        """Initialize the GAT decoder stack.

        Parameters
        ----------
        latent_dim : int
            Input latent dimension per node.
        hidden_channels : int
            Hidden GAT channel width for intermediate layers.
        out_channels : int
            Output physical feature dimension per node.
        num_layers : int, optional
            Number of GAT layers. Default is ``2``.
        activation : {"relu", "sigmoid", "tanh"}, optional
            Hidden-layer activation. Default is ``"relu"``.
        heads : int, optional
            Number of attention heads per GAT layer. Default is ``1``.
        dropout : float, optional
            Dropout probability inside GAT attention. Default is ``0.0``.

        Raises
        ------
        ValueError
            If any dimension argument is not positive, ``heads < 1``, or
            ``dropout`` is outside ``[0, 1]``.
        """
        validate_positive_dims(
            latent_dim=latent_dim,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
        )
        validate_gat_attention(heads=heads, dropout=dropout)

        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout

        super().__init__(
            input_channels=latent_dim,
            input_dim_name="latent_dim",
            num_layers=num_layers,
            activation=activation,
            convs=build_gat_convs(
                latent_dim,
                hidden_channels,
                out_channels,
                num_layers,
                heads=heads,
                dropout=dropout,
            ),
        )
