"""Graph Neural Network encoders for topology-aware latent lifting."""

from __future__ import annotations

from koopman_graph.nn.gnn import (
    ActivationName,
    BaseGNNModule,
    build_diff_convs,
    build_gat_convs,
    build_gcn_convs,
    build_sage_convs,
    build_transformer_convs,
    validate_diffusion_steps,
    validate_gat_attention,
    validate_optional_edge_dim,
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


class SAGEEncoder(BaseGNNModule):
    """GraphSAGE encoder that lifts node features into a latent space.

    Applies stacked :class:`~torch_geometric.nn.SAGEConv` layers (Hamilton et
    al., NeurIPS 2017). Scalar ``edge_weight`` arguments are accepted for API
    symmetry with :class:`~koopman_graph.nn.encoder.GNNEncoder` but are ignored
    because SAGEConv does not consume scalar edge weights.

    Attributes
    ----------
    in_channels : int
        Input node feature dimension.
    hidden_channels : int
        Hidden SAGE channel width.
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
        """Initialize the GraphSAGE encoder stack.

        Parameters
        ----------
        in_channels : int
            Input node feature dimension.
        hidden_channels : int
            Hidden SAGE channel width for intermediate layers.
        latent_dim : int
            Output latent dimension per node.
        num_layers : int, optional
            Number of SAGE layers. Default is ``2``.
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
            convs=build_sage_convs(
                in_channels,
                hidden_channels,
                latent_dim,
                num_layers,
            ),
        )


class DiffConvEncoder(BaseGNNModule):
    """Diffusion-convolution encoder for directional spatial mixing.

    Stacks DCRNN-style bidirectional diffusion convolutions (Li et al., ICLR
    2018). Unlike GCN, DiffConv builds separate forward/backward random-walk
    supports and therefore respects asymmetric ``edge_weight`` structure on
    directed or anisotropically weighted graphs.

    Attributes
    ----------
    in_channels : int
        Input node feature dimension.
    hidden_channels : int
        Hidden DiffConv channel width.
    latent_dim : int
        Output latent dimension per node.
    diffusion_steps : int
        Number of forward/backward random-walk hops per layer.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        latent_dim: int,
        *,
        num_layers: int = 2,
        activation: ActivationName = "relu",
        diffusion_steps: int = 2,
    ) -> None:
        """Initialize the DiffConv encoder stack.

        Parameters
        ----------
        in_channels : int
            Input node feature dimension.
        hidden_channels : int
            Hidden DiffConv channel width for intermediate layers.
        latent_dim : int
            Output latent dimension per node.
        num_layers : int, optional
            Number of DiffConv layers. Default is ``2``.
        activation : {"relu", "sigmoid", "tanh"}, optional
            Hidden-layer activation. Default is ``"relu"``.
        diffusion_steps : int, optional
            Number of forward/backward random-walk hops (excluding identity).
            Default is ``2``.

        Raises
        ------
        ValueError
            If any dimension argument is not positive or
            ``diffusion_steps < 1``.
        """
        validate_positive_dims(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            latent_dim=latent_dim,
            num_layers=num_layers,
        )
        validate_diffusion_steps(diffusion_steps)

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.latent_dim = latent_dim
        self.diffusion_steps = diffusion_steps

        super().__init__(
            input_channels=in_channels,
            input_dim_name="in_channels",
            num_layers=num_layers,
            activation=activation,
            convs=build_diff_convs(
                in_channels,
                hidden_channels,
                latent_dim,
                num_layers,
                diffusion_steps=diffusion_steps,
            ),
        )


class GraphTransformerEncoder(BaseGNNModule):
    """Graph Transformer encoder for longer-range spatial attention.

    Stacks :class:`~torch_geometric.nn.TransformerConv` layers (masked
    attention on graph edges; Shi et al. / PyG lineage; Vaswani et al. 2017
    background). Multi-head attention is denser than GCN/DiffConv per edge and
    typically more expensive; prefer GCN/GAT/DiffConv when local message
    passing suffices.

    Optional ``edge_dim`` enables edge-feature conditioning: when set, forward
    requires ``edge_weight`` (scalar when ``edge_dim=1``, or dense
    ``(num_edges, edge_dim)``) which is passed as ``edge_attr``.

    Attributes
    ----------
    in_channels : int
        Input node feature dimension.
    hidden_channels : int
        Hidden Transformer channel width.
    latent_dim : int
        Output latent dimension per node.
    heads : int
        Number of attention heads per layer.
    dropout : float
        Dropout probability inside attention.
    edge_dim : int or None
        Edge feature width, or ``None`` when edge conditioning is disabled.
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
        edge_dim: int | None = None,
    ) -> None:
        """Initialize the graph Transformer encoder stack.

        Parameters
        ----------
        in_channels : int
            Input node feature dimension.
        hidden_channels : int
            Hidden Transformer channel width for intermediate layers.
        latent_dim : int
            Output latent dimension per node.
        num_layers : int, optional
            Number of Transformer layers. Default is ``2``.
        activation : {"relu", "sigmoid", "tanh"}, optional
            Hidden-layer activation. Default is ``"relu"``.
        heads : int, optional
            Number of attention heads per layer. Default is ``1``.
        dropout : float, optional
            Dropout probability inside attention. Default is ``0.0``.
        edge_dim : int or None, optional
            Edge feature width for optional conditioning. Default is ``None``.

        Raises
        ------
        ValueError
            If any dimension argument is not positive, ``heads < 1``,
            ``dropout`` is outside ``[0, 1]``, or ``edge_dim`` is invalid.
        """
        validate_positive_dims(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            latent_dim=latent_dim,
            num_layers=num_layers,
        )
        validate_gat_attention(heads=heads, dropout=dropout)
        validate_optional_edge_dim(edge_dim)

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.latent_dim = latent_dim
        self.heads = heads
        self.dropout = dropout
        self.edge_dim = edge_dim

        super().__init__(
            input_channels=in_channels,
            input_dim_name="in_channels",
            num_layers=num_layers,
            activation=activation,
            convs=build_transformer_convs(
                in_channels,
                hidden_channels,
                latent_dim,
                num_layers,
                heads=heads,
                dropout=dropout,
                edge_dim=edge_dim,
            ),
        )
