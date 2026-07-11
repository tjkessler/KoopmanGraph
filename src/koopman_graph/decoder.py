"""Graph Neural Network decoders for latent-to-physical reconstruction."""

from __future__ import annotations

from koopman_graph.encoder import (
    ActivationName,
    BaseGNNModule,
    _build_gcn_convs,
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
        if latent_dim < 1:
            msg = f"latent_dim must be positive, got {latent_dim}"
            raise ValueError(msg)
        if hidden_channels < 1:
            msg = f"hidden_channels must be positive, got {hidden_channels}"
            raise ValueError(msg)
        if out_channels < 1:
            msg = f"out_channels must be positive, got {out_channels}"
            raise ValueError(msg)
        if num_layers < 1:
            msg = f"num_layers must be positive, got {num_layers}"
            raise ValueError(msg)

        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        super().__init__(
            input_channels=latent_dim,
            input_dim_name="latent_dim",
            num_layers=num_layers,
            activation=activation,
            convs=_build_gcn_convs(
                latent_dim,
                hidden_channels,
                out_channels,
                num_layers,
            ),
        )
