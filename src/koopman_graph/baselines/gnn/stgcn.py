"""STGCN-style spatiotemporal GNN forecaster baseline."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.nn import GCNConv

from koopman_graph.baselines.gnn.base import GNNForecasterBaseline


class _TemporalGatedConv(nn.Module):
    """Causal temporal convolution with GLU gating (STGCN-style).

    Notes
    -----
    Pads on the left so the temporal kernel is causal, then applies a gated
    linear unit to the split convolution channels.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        """Initialize the temporal gated convolution.

        Parameters
        ----------
        in_channels : int
            Input channel count (feature dim).
        out_channels : int
            Output channel count.
        kernel_size : int
            Temporal kernel width.
        """
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.conv = nn.Conv2d(
            in_channels,
            2 * out_channels,
            kernel_size=(1, self.kernel_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply causal temporal GLU convolution.

        Parameters
        ----------
        x : Tensor
            Input with shape ``(batch, channels, num_nodes, time)``.

        Returns
        -------
        Tensor
            Output with shape ``(batch, out_channels, num_nodes, time)``.
        """
        if self.kernel_size > 1:
            x = nn.functional.pad(x, (self.kernel_size - 1, 0))
        split = self.conv(x)
        residual, gate = split.chunk(2, dim=1)
        return residual * torch.sigmoid(gate)


class _STConvBlock(nn.Module):
    """One ST-Conv block: temporal → spatial GCN → temporal.

    Notes
    -----
    Residual connection is identity when channel counts match, otherwise a
    ``1x1`` projection.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        kernel_size: int,
    ) -> None:
        """Initialize an ST-Conv block.

        Parameters
        ----------
        in_channels : int
            Block input channels.
        hidden_channels : int
            Intermediate width after the first temporal conv.
        out_channels : int
            Block output channels.
        kernel_size : int
            Temporal kernel width.
        """
        super().__init__()
        self.temporal_in = _TemporalGatedConv(in_channels, hidden_channels, kernel_size)
        self.spatial = GCNConv(hidden_channels, hidden_channels)
        self.temporal_out = _TemporalGatedConv(
            hidden_channels, out_channels, kernel_size
        )
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor | None,
    ) -> Tensor:
        """Forward an ST-Conv block.

        Parameters
        ----------
        x : Tensor
            Input with shape ``(batch, channels, num_nodes, time)``.
        edge_index : Tensor
            Graph connectivity.
        edge_weight : Tensor or None
            Optional edge weights.

        Returns
        -------
        Tensor
            Block output with the same layout as ``x``.
        """
        residual = self.residual(x)
        h = self.temporal_in(x)
        batch, channels, num_nodes, time = h.shape
        # Flatten (batch, time) graph copies for one batched GCNConv call.
        spatial_in = h.permute(0, 3, 2, 1).reshape(batch * time * num_nodes, channels)
        offsets = (
            torch.arange(batch * time, device=edge_index.device).repeat_interleave(
                edge_index.shape[1]
            )
            * num_nodes
        )
        edge_index_b = edge_index.repeat(1, batch * time) + offsets.unsqueeze(0)
        edge_weight_b = (
            None if edge_weight is None else edge_weight.repeat(batch * time)
        )
        spatial_out = self.spatial(spatial_in, edge_index_b, edge_weight_b)
        spatial_out = spatial_out.reshape(batch, time, num_nodes, channels)
        spatial_out = spatial_out.permute(0, 3, 2, 1).contiguous()
        return self.temporal_out(spatial_out) + residual


class STGCNBaseline(GNNForecasterBaseline):
    """Lightweight STGCN-style forecaster (Yu, Yin & Zhu, IJCAI 2018).

    In-repo reference implementation for apples-to-apples comparisons with
    :class:`~koopman_graph.model.GraphKoopmanModel`. Not a replacement for
    dedicated traffic-forecasting libraries; training budgets and lookbacks are
    intentionally modest.

    Satisfies :class:`~koopman_graph.protocols.ForecastModel` for ``fit`` /
    ``predict`` method presence. :meth:`spectrum` raises :class:`RuntimeError`
    because there is no linear Koopman operator.

    Notes
    -----
    Constructor arguments are documented on :meth:`__init__`.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        *,
        num_st_blocks: int = 2,
        kernel_size: int = 3,
        history_len: int = 3,
        time_step: float = 1.0,
    ) -> None:
        """Initialize the STGCN baseline.

        Parameters
        ----------
        in_channels : int
            Input node-feature dimension.
        hidden_channels : int
            Internal width of ST-Conv blocks.
        out_channels : int
            Output node-feature dimension (must equal ``in_channels`` for
            autoregressive rollout).
        num_st_blocks : int, optional
            Number of ST-Conv blocks. Default is ``2``.
        kernel_size : int, optional
            Temporal kernel width. Default is ``3``.
        history_len : int, optional
            Lookback window. Default is ``3``.
        time_step : float, optional
            Snapshot interval metadata. Default is ``1.0``.
        """
        super().__init__(
            in_channels,
            hidden_channels,
            out_channels,
            history_len=history_len,
            time_step=time_step,
        )
        if num_st_blocks < 1:
            msg = f"num_st_blocks must be positive, got {num_st_blocks}"
            raise ValueError(msg)
        if kernel_size < 1:
            msg = f"kernel_size must be positive, got {kernel_size}"
            raise ValueError(msg)
        blocks: list[nn.Module] = []
        channels = in_channels
        for _ in range(num_st_blocks):
            blocks.append(
                _STConvBlock(channels, hidden_channels, hidden_channels, kernel_size)
            )
            channels = hidden_channels
        self.blocks = nn.ModuleList(blocks)
        self.readout = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

    def predict_next(
        self,
        history: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        """Predict the next frame from an STGCN history window.

        Parameters
        ----------
        history : Tensor
            History with shape ``(history_len, num_nodes, in_channels)``.
        edge_index : Tensor
            Graph connectivity.
        edge_weight : Tensor or None, optional
            Optional edge weights.

        Returns
        -------
        Tensor
            Next-step features with shape ``(num_nodes, out_channels)``.
        """
        # (1, C, N, T)
        x = history.permute(2, 1, 0).unsqueeze(0)
        for block in self.blocks:
            x = block(x, edge_index, edge_weight)
        x = self.readout(x)
        return x[0, :, :, -1].transpose(0, 1).contiguous()
