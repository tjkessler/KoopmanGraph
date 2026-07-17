"""Graph Neural Network encoders for topology-aware latent lifting."""

from __future__ import annotations

from koopman_graph.nn.gnn import (
    ActivationName,
    BaseGNNModule,
    build_gat_convs,
    build_gcn_convs,
    validate_gat_attention,
    validate_positive_dims,
)


class GNNEncoder(BaseGNNModule):
    """GCN encoder that lifts node features into a latent space.

    Applies stacked Graph Convolutional Network layers with configurable hidden
    activation. The final layer maps directly to ``latent_dim`` without an
    activation, producing per-node latent vectors suitable for Koopman propagation.

    Attributes
    ----------
    in_channels : int
        Input node feature dimension.
    hidden_channels : int
        Hidden GCN channel width.
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
        """Initialize the GCN encoder stack.

        Parameters
        ----------
        in_channels : int
            Input node feature dimension.
        hidden_channels : int
            Hidden GCN channel width for intermediate layers.
        latent_dim : int
            Output latent dimension per node.
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
            convs=build_gcn_convs(
                in_channels,
                hidden_channels,
                latent_dim,
                num_layers,
            ),
        )


class GATEncoder(BaseGNNModule):
    """GAT encoder that lifts node features into a latent space.

    Applies stacked Graph Attention Network layers with configurable hidden
    activation. The final layer maps directly to ``latent_dim`` without an
    activation, producing per-node latent vectors suitable for Koopman propagation.

    Scalar ``edge_weight`` arguments are accepted for API symmetry with
    :class:`~koopman_graph.nn.encoder.GNNEncoder` but are ignored because
    :class:`~torch_geometric.nn.GATConv` does not consume scalar edge weights.

    Attributes
    ----------
    in_channels : int
        Input node feature dimension.
    hidden_channels : int
        Hidden GAT channel width.
    latent_dim : int
        Output latent dimension per node.
    heads : int
        Number of attention heads per GAT layer.
    dropout : float
        Dropout probability inside GAT attention.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        latent_dim: int,
        *,
        num_layers: int = 2,
        activation: ActivationName = "relu",
        heads: int = 1,
        dropout: float = 0.0,
    ) -> None:
        """Initialize the GAT encoder stack.

        Parameters
        ----------
        in_channels : int
            Input node feature dimension.
        hidden_channels : int
            Hidden GAT channel width for intermediate layers.
        latent_dim : int
            Output latent dimension per node.
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
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            latent_dim=latent_dim,
            num_layers=num_layers,
        )
        validate_gat_attention(heads=heads, dropout=dropout)

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.latent_dim = latent_dim
        self.heads = heads
        self.dropout = dropout

        super().__init__(
            input_channels=in_channels,
            input_dim_name="in_channels",
            num_layers=num_layers,
            activation=activation,
            convs=build_gat_convs(
                in_channels,
                hidden_channels,
                latent_dim,
                num_layers,
                heads=heads,
                dropout=dropout,
            ),
        )
