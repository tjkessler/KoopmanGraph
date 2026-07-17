"""Shared GNN message-passing primitives (power-user layer).

Encoder and decoder are peer components. Shared activation typing, node-feature
validation, constructor validators, convolution builders, and
:class:`BaseGNNModule` live here so neither peer owns the other. Importable as
``koopman_graph.nn.gnn`` but **not** part of the stable public façade (not
listed in ``koopman_graph.__all__``).

Prefer :class:`~koopman_graph.nn.encoder.GNNEncoder` /
:class:`~koopman_graph.nn.encoder.GATEncoder` and
:class:`~koopman_graph.nn.decoder.GNNDecoder` /
:class:`~koopman_graph.nn.decoder.GATDecoder` for application code.
"""

from __future__ import annotations

from typing import Literal

from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv

from koopman_graph.graph_utils import resolve_graph_inputs

ActivationName = Literal["relu", "sigmoid", "tanh"]


def validate_positive_dims(**dims: int) -> None:
    """Require each named constructor dimension to be a positive integer.

    Parameters
    ----------
    **dims
        Mapping of parameter name to integer value (for example
        ``in_channels=3``, ``num_layers=2``).

    Raises
    ------
    ValueError
        If any value is less than ``1``.
    """
    for name, value in dims.items():
        if value < 1:
            msg = f"{name} must be positive, got {value}"
            raise ValueError(msg)


def validate_gat_attention(*, heads: int, dropout: float) -> None:
    """Validate GAT attention hyperparameters.

    Parameters
    ----------
    heads : int
        Number of attention heads per GAT layer.
    dropout : float
        Dropout probability inside GAT attention.

    Raises
    ------
    ValueError
        If ``heads < 1`` or ``dropout`` is outside ``[0, 1]``.
    """
    if heads < 1:
        msg = f"heads must be positive, got {heads}"
        raise ValueError(msg)
    if not 0.0 <= dropout <= 1.0:
        msg = f"dropout must be in [0, 1], got {dropout}"
        raise ValueError(msg)


def _resolve_activation(name: ActivationName) -> nn.Module:
    """Return a PyTorch activation module for the given name.

    Parameters
    ----------
    name : {"relu", "sigmoid", "tanh"}
        Activation identifier.

    Returns
    -------
    nn.Module
        Instantiated activation layer.

    Raises
    ------
    ValueError
        If ``name`` is not a supported activation.
    """
    if name == "relu":
        return nn.ReLU()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "tanh":
        return nn.Tanh()
    msg = f"Unknown activation: {name!r}"
    raise ValueError(msg)


def _validate_node_features(
    x: Tensor,
    expected_channels: int,
    dim_name: str,
) -> None:
    """Validate node feature tensor rank and channel dimension.

    Parameters
    ----------
    x : Tensor
        Node feature tensor to validate.
    expected_channels : int
        Expected feature dimension.
    dim_name : str
        Name of the dimension used in error messages.

    Raises
    ------
    ValueError
        If ``x`` is not rank-2 or its channel dimension does not match
        ``expected_channels``.
    """
    if x.ndim != 2:
        msg = f"Expected x with shape (num_nodes, {dim_name}), got {tuple(x.shape)}"
        raise ValueError(msg)
    if x.shape[1] != expected_channels:
        msg = (
            f"Expected {dim_name}={expected_channels}, "
            f"got feature dimension {x.shape[1]}"
        )
        raise ValueError(msg)


def build_gcn_convs(
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    num_layers: int,
) -> nn.ModuleList:
    """Build a stacked GCN module list.

    Parameters
    ----------
    in_channels : int
        Input node feature dimension.
    hidden_channels : int
        Hidden GCN channel width for intermediate layers.
    out_channels : int
        Output node feature dimension.
    num_layers : int
        Number of GCN layers.

    Returns
    -------
    nn.ModuleList
        Ordered GCN convolution layers.
    """
    convs: list[GCNConv] = []
    if num_layers == 1:
        convs.append(GCNConv(in_channels, out_channels))
    else:
        convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            convs.append(GCNConv(hidden_channels, hidden_channels))
        convs.append(GCNConv(hidden_channels, out_channels))
    return nn.ModuleList(convs)


def build_gat_convs(
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    num_layers: int,
    *,
    heads: int,
    dropout: float,
) -> nn.ModuleList:
    """Build a stacked GAT module list.

    Parameters
    ----------
    in_channels : int
        Input node feature dimension.
    hidden_channels : int
        Hidden GAT channel width for intermediate layers.
    out_channels : int
        Output node feature dimension.
    num_layers : int
        Number of GAT layers.
    heads : int
        Number of attention heads per layer.
    dropout : float
        Dropout probability inside GAT attention.

    Returns
    -------
    nn.ModuleList
        Ordered GAT convolution layers.
    """
    convs: list[GATConv] = []
    if num_layers == 1:
        convs.append(
            GATConv(
                in_channels,
                out_channels,
                heads=heads,
                concat=False,
                dropout=dropout,
            )
        )
    else:
        convs.append(
            GATConv(
                in_channels,
                hidden_channels,
                heads=heads,
                concat=False,
                dropout=dropout,
            )
        )
        for _ in range(num_layers - 2):
            convs.append(
                GATConv(
                    hidden_channels,
                    hidden_channels,
                    heads=heads,
                    concat=False,
                    dropout=dropout,
                )
            )
        convs.append(
            GATConv(
                hidden_channels,
                out_channels,
                heads=heads,
                concat=False,
                dropout=dropout,
            )
        )
    return nn.ModuleList(convs)


class BaseGNNModule(nn.Module):
    """Shared message-passing stack for GNN encoders and decoders.

    Attributes
    ----------
    input_channels : int
        Expected input node feature dimension.
    input_dim_name : str
        Name of the input dimension used in validation errors.
    num_layers : int
        Number of message-passing layers.
    activation_name : str
        Identifier for the hidden-layer activation.
    activation : nn.Module
        Instantiated hidden-layer activation.
    convs : nn.ModuleList
        Ordered graph convolution layers.
    """

    def __init__(
        self,
        *,
        input_channels: int,
        input_dim_name: str,
        num_layers: int,
        activation: ActivationName,
        convs: nn.ModuleList,
    ) -> None:
        """Initialize shared GNN module state.

        Parameters
        ----------
        input_channels : int
            Expected input node feature dimension.
        input_dim_name : str
            Name of the input dimension used in validation errors.
        num_layers : int
            Number of message-passing layers.
        activation : {"relu", "sigmoid", "tanh"}
            Hidden-layer activation identifier.
        convs : nn.ModuleList
            Pre-built graph convolution layers.
        """
        super().__init__()
        self.input_channels = input_channels
        self.input_dim_name = input_dim_name
        self.num_layers = num_layers
        self.activation_name = activation
        self.activation = _resolve_activation(activation)
        self.convs = convs

    def forward(
        self,
        x_or_data: Tensor | Data,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        """Run the stacked message-passing layers on graph node features.

        Parameters
        ----------
        x_or_data : Tensor or Data
            Either a PyG ``Data`` object or node features ``x``.
        edge_index : Tensor or None, optional
            Edge index required when ``x_or_data`` is a tensor.
        edge_weight : Tensor or None, optional
            Scalar edge weights with shape ``(num_edges,)``. Passed to
            :class:`~torch_geometric.nn.GCNConv` when present. Ignored by
            GAT layers.

        Returns
        -------
        Tensor
            Transformed node features with shape ``(num_nodes, out_channels)``.
        """
        x, edge_index, edge_weight = resolve_graph_inputs(
            x_or_data,
            edge_index,
            edge_weight,
        )
        _validate_node_features(x, self.input_channels, self.input_dim_name)

        for layer_idx, conv in enumerate(self.convs):
            if edge_weight is not None and isinstance(conv, GCNConv):
                x = conv(x, edge_index, edge_weight=edge_weight)
            else:
                x = conv(x, edge_index)
            if layer_idx < len(self.convs) - 1:
                x = self.activation(x)
        return x
