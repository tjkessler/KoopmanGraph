"""Shared GNN message-passing primitives (power-user layer).

Encoder and decoder are peer components. Shared activation typing, node-feature
validation, constructor validators, convolution builders, and
:class:`BaseGNNModule` live here so neither peer owns the other. Importable as
``koopman_graph.nn.gnn`` but **not** part of the stable public façade (not
listed in ``koopman_graph.__all__``).

Prefer :class:`~koopman_graph.nn.encoder.GNNEncoder` /
:class:`~koopman_graph.nn.encoder.GATEncoder` /
:class:`~koopman_graph.nn.encoder.SAGEEncoder` /
:class:`~koopman_graph.nn.encoder.DiffConvEncoder` /
:class:`~koopman_graph.nn.encoder.GraphTransformerEncoder` and the matching
decoders for application code.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, TransformerConv

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


def validate_optional_edge_dim(edge_dim: int | None) -> None:
    """Validate optional Transformer edge-feature dimension.

    Parameters
    ----------
    edge_dim : int or None
        Edge feature width. ``None`` disables edge conditioning; otherwise
        must be a positive integer.

    Raises
    ------
    ValueError
        If ``edge_dim`` is not ``None`` and is less than ``1``.
    """
    if edge_dim is not None and edge_dim < 1:
        msg = f"edge_dim must be positive when set, got {edge_dim}"
        raise ValueError(msg)


def _edge_weight_as_attr(edge_weight: Tensor, edge_dim: int) -> Tensor:
    """Reshape scalar or dense edge weights into Transformer ``edge_attr``.

    Parameters
    ----------
    edge_weight : Tensor
        Edge features with shape ``(num_edges,)`` (requires ``edge_dim == 1``)
        or ``(num_edges, edge_dim)``.
    edge_dim : int
        Expected edge feature width configured on ``TransformerConv``.

    Returns
    -------
    Tensor
        Edge attributes with shape ``(num_edges, edge_dim)``.

    Raises
    ------
    ValueError
        If ``edge_weight`` rank/shape is incompatible with ``edge_dim``.
    """
    if edge_weight.ndim == 1:
        if edge_dim != 1:
            msg = f"1-D edge_weight requires edge_dim=1, got edge_dim={edge_dim}"
            raise ValueError(msg)
        return edge_weight.unsqueeze(-1)
    if edge_weight.ndim == 2 and edge_weight.shape[1] == edge_dim:
        return edge_weight
    msg = (
        "edge_weight for TransformerConv must have shape (num_edges,) "
        f"when edge_dim=1 or (num_edges, {edge_dim}), got {tuple(edge_weight.shape)}"
    )
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


def validate_diffusion_steps(diffusion_steps: int) -> None:
    """Validate DiffConv diffusion-hop count.

    Parameters
    ----------
    diffusion_steps : int
        Number of forward/backward random-walk hops (excluding identity).

    Raises
    ------
    ValueError
        If ``diffusion_steps < 1``.
    """
    if diffusion_steps < 1:
        msg = f"diffusion_steps must be positive, got {diffusion_steps}"
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


def build_sage_convs(
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    num_layers: int,
) -> nn.ModuleList:
    """Build a stacked GraphSAGE module list.

    Parameters
    ----------
    in_channels : int
        Input node feature dimension.
    hidden_channels : int
        Hidden SAGE channel width for intermediate layers.
    out_channels : int
        Output node feature dimension.
    num_layers : int
        Number of SAGE layers.

    Returns
    -------
    nn.ModuleList
        Ordered SAGE convolution layers.
    """
    convs: list[SAGEConv] = []
    if num_layers == 1:
        convs.append(SAGEConv(in_channels, out_channels))
    else:
        convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            convs.append(SAGEConv(hidden_channels, hidden_channels))
        convs.append(SAGEConv(hidden_channels, out_channels))
    return nn.ModuleList(convs)


def build_transformer_convs(
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    num_layers: int,
    *,
    heads: int,
    dropout: float,
    edge_dim: int | None = None,
) -> nn.ModuleList:
    """Build a stacked graph Transformer module list.

    Uses :class:`~torch_geometric.nn.TransformerConv` with ``concat=False`` so
    output channel widths match ``hidden_channels`` / ``out_channels``
    independent of ``heads`` (same stable-width convention as GAT builders).

    Parameters
    ----------
    in_channels : int
        Input node feature dimension.
    hidden_channels : int
        Hidden Transformer channel width for intermediate layers.
    out_channels : int
        Output node feature dimension.
    num_layers : int
        Number of Transformer layers.
    heads : int
        Number of attention heads per layer.
    dropout : float
        Dropout probability inside attention.
    edge_dim : int or None, optional
        Edge feature width for optional edge conditioning. Default is
        ``None`` (no edge attributes).

    Returns
    -------
    nn.ModuleList
        Ordered Transformer convolution layers.
    """

    def _layer(in_ch: int, out_ch: int) -> TransformerConv:
        """Build one ``TransformerConv`` with shared head/dropout settings.

        Parameters
        ----------
        in_ch : int
            Input channel count.
        out_ch : int
            Output channel count.

        Returns
        -------
        TransformerConv
            Configured attention convolution layer.
        """
        return TransformerConv(
            in_ch,
            out_ch,
            heads=heads,
            concat=False,
            dropout=dropout,
            edge_dim=edge_dim,
        )

    convs: list[TransformerConv] = []
    if num_layers == 1:
        convs.append(_layer(in_channels, out_channels))
    else:
        convs.append(_layer(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            convs.append(_layer(hidden_channels, hidden_channels))
        convs.append(_layer(hidden_channels, out_channels))
    return nn.ModuleList(convs)


def _dense_adjacency(
    edge_index: Tensor,
    edge_weight: Tensor | None,
    num_nodes: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """Assemble a dense adjacency matrix from sparse COO topology.

    Parameters
    ----------
    edge_index : Tensor
        COO edge index with shape ``(2, num_edges)``.
    edge_weight : Tensor or None
        Optional scalar edge weights with shape ``(num_edges,)``.
    num_nodes : int
        Number of nodes in the graph.
    dtype : torch.dtype
        Floating dtype for the assembled matrix.
    device : torch.device
        Device for the assembled matrix.

    Returns
    -------
    Tensor
        Dense adjacency with shape ``(num_nodes, num_nodes)``.
    """
    weights = (
        torch.ones(edge_index.shape[1], dtype=dtype, device=device)
        if edge_weight is None
        else edge_weight.to(dtype=dtype, device=device)
    )
    adjacency = torch.zeros((num_nodes, num_nodes), dtype=dtype, device=device)
    adjacency.index_put_((edge_index[0], edge_index[1]), weights, accumulate=True)
    return adjacency


def _random_walk_normalize(adjacency: Tensor) -> Tensor:
    """Row-normalize adjacency (``D^{-1} A``) with a small degree floor.

    Parameters
    ----------
    adjacency : Tensor
        Dense adjacency with shape ``(num_nodes, num_nodes)``.

    Returns
    -------
    Tensor
        Random-walk normalized adjacency of the same shape.
    """
    degree = adjacency.sum(dim=1).clamp_min(1e-6)
    return adjacency / degree.unsqueeze(1)


def _diffusion_supports(
    edge_index: Tensor,
    edge_weight: Tensor | None,
    num_nodes: int,
    diffusion_steps: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> list[Tensor]:
    """Build identity + bidirectional random-walk diffusion supports.

    Follows the DCRNN support set (Li et al., ICLR 2018): ``I`` plus powers of
    the forward and backward row-normalized adjacency up to
    ``diffusion_steps``.

    Parameters
    ----------
    edge_index : Tensor
        COO edge index with shape ``(2, num_edges)``.
    edge_weight : Tensor or None
        Optional scalar edge weights with shape ``(num_edges,)``.
    num_nodes : int
        Number of nodes in the graph.
    diffusion_steps : int
        Number of forward/backward random-walk hops (excluding identity).
    dtype : torch.dtype
        Floating dtype for support matrices.
    device : torch.device
        Device for support matrices.

    Returns
    -------
    list of Tensor
        Dense supports of length ``1 + 2 * diffusion_steps``.
    """
    adjacency = _dense_adjacency(
        edge_index,
        edge_weight,
        num_nodes,
        dtype=dtype,
        device=device,
    )
    forward = _random_walk_normalize(adjacency)
    backward = _random_walk_normalize(adjacency.transpose(0, 1))
    supports: list[Tensor] = [torch.eye(num_nodes, dtype=dtype, device=device)]
    support = forward
    for _ in range(diffusion_steps):
        supports.append(support)
        support = support @ forward
    support = backward
    for _ in range(diffusion_steps):
        supports.append(support)
        support = support @ backward
    return supports


class DiffusionConv(nn.Module):
    """Bidirectional diffusion convolution (DCRNN-style supports).

    Applies a linear mix of identity and forward/backward random-walk
    adjacency powers. Scalar ``edge_weight`` values are respected when
    building the adjacency, so directional grids (e.g.
    :class:`~koopman_graph.datasets.grid.AnisotropicAdvectionGridBenchmark`)
    can bias spatial mixing.

    Attributes
    ----------
    in_channels : int
        Input node feature dimension.
    out_channels : int
        Output node feature dimension.
    diffusion_steps : int
        Number of forward/backward random-walk hops per forward pass.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        diffusion_steps: int = 2,
    ) -> None:
        """Initialize a diffusion convolution layer.

        Parameters
        ----------
        in_channels : int
            Input node feature dimension.
        out_channels : int
            Output node feature dimension.
        diffusion_steps : int, optional
            Number of forward/backward random-walk hops (excluding identity).
            Default is ``2``.
        """
        super().__init__()
        validate_positive_dims(
            in_channels=in_channels,
            out_channels=out_channels,
        )
        validate_diffusion_steps(diffusion_steps)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.diffusion_steps = int(diffusion_steps)
        num_supports = 1 + 2 * self.diffusion_steps
        self.weights = nn.Parameter(
            torch.empty(num_supports, in_channels, out_channels)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.xavier_uniform_(self.weights)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        """Apply diffusion convolution on node features.

        Parameters
        ----------
        x : Tensor
            Node features with shape ``(num_nodes, in_channels)``.
        edge_index : Tensor
            COO edge index with shape ``(2, num_edges)``.
        edge_weight : Tensor or None, optional
            Scalar edge weights with shape ``(num_edges,)``.

        Returns
        -------
        Tensor
            Diffused features with shape ``(num_nodes, out_channels)``.
        """
        supports = _diffusion_supports(
            edge_index,
            edge_weight,
            x.shape[0],
            self.diffusion_steps,
            dtype=x.dtype,
            device=x.device,
        )
        out = x.new_zeros(x.shape[0], self.out_channels)
        for support, weight in zip(supports, self.weights, strict=True):
            out = out + (support @ x) @ weight
        return out + self.bias


def build_diff_convs(
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    num_layers: int,
    *,
    diffusion_steps: int,
) -> nn.ModuleList:
    """Build a stacked DiffConv module list.

    Parameters
    ----------
    in_channels : int
        Input node feature dimension.
    hidden_channels : int
        Hidden DiffConv channel width for intermediate layers.
    out_channels : int
        Output node feature dimension.
    num_layers : int
        Number of DiffConv layers.
    diffusion_steps : int
        Number of forward/backward random-walk hops per layer.

    Returns
    -------
    nn.ModuleList
        Ordered diffusion convolution layers.
    """
    convs: list[DiffusionConv] = []
    if num_layers == 1:
        convs.append(
            DiffusionConv(
                in_channels,
                out_channels,
                diffusion_steps=diffusion_steps,
            )
        )
    else:
        convs.append(
            DiffusionConv(
                in_channels,
                hidden_channels,
                diffusion_steps=diffusion_steps,
            )
        )
        for _ in range(num_layers - 2):
            convs.append(
                DiffusionConv(
                    hidden_channels,
                    hidden_channels,
                    diffusion_steps=diffusion_steps,
                )
            )
        convs.append(
            DiffusionConv(
                hidden_channels,
                out_channels,
                diffusion_steps=diffusion_steps,
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
            Scalar edge weights with shape ``(num_edges,)``, or dense edge
            features ``(num_edges, edge_dim)`` for Transformer peers configured
            with ``edge_dim``. Passed to
            :class:`~torch_geometric.nn.GCNConv` and
            :class:`~koopman_graph.nn.gnn.DiffusionConv` when present. For
            :class:`~torch_geometric.nn.TransformerConv` with ``edge_dim`` set,
            reshaped into ``edge_attr`` (required whenever ``edge_dim`` is
            configured). Ignored by GAT, SAGE, and Transformer layers without
            ``edge_dim``.

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
            if isinstance(conv, TransformerConv) and conv.edge_dim is not None:
                if edge_weight is None:
                    msg = (
                        "TransformerConv with edge_dim requires edge_weight "
                        "(or Data.edge_weight) as edge features"
                    )
                    raise ValueError(msg)
                edge_attr = _edge_weight_as_attr(edge_weight, conv.edge_dim)
                x = conv(x, edge_index, edge_attr=edge_attr)
            elif edge_weight is not None and isinstance(conv, (GCNConv, DiffusionConv)):
                x = conv(x, edge_index, edge_weight=edge_weight)
            else:
                x = conv(x, edge_index)
            if layer_idx < len(self.convs) - 1:
                x = self.activation(x)
        return x
