"""Graph WaveNet–style dilated spatiotemporal forecaster baseline."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch_geometric.data import Data

from koopman_graph.baselines.gnn.base import (
    GNNForecasterBaseline,
    dense_adjacency,
    random_walk_normalize,
)
from koopman_graph.data import GraphSnapshotSequence, resolve_sequence


class _DilatedTemporalConv(nn.Module):
    """Dilated causal temporal convolution with gated activation.

    Notes
    -----
    Uses kernel width 2 with left padding equal to the dilation so the
    receptive field grows exponentially across stacked layers.
    """

    def __init__(self, channels: int, dilation: int) -> None:
        """Initialize a dilated gated temporal convolution.

        Parameters
        ----------
        channels : int
            Feature channel count.
        dilation : int
            Temporal dilation factor.
        """
        super().__init__()
        self.dilation = int(dilation)
        self.conv = nn.Conv2d(
            channels,
            2 * channels,
            kernel_size=(1, 2),
            dilation=(1, self.dilation),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply gated dilated temporal convolution.

        Parameters
        ----------
        x : Tensor
            Input with shape ``(batch, channels, num_nodes, time)``.

        Returns
        -------
        Tensor
            Output with shape ``(batch, channels, num_nodes, time)``.
        """
        x = nn.functional.pad(x, (self.dilation, 0))
        split = self.conv(x)
        filter_gate, gate = split.chunk(2, dim=1)
        return torch.tanh(filter_gate) * torch.sigmoid(gate)


class _GraphWaveNetLayer(nn.Module):
    """One Graph WaveNet layer: dilated TCN + fixed/adaptive spatial mix.

    Notes
    -----
    Residual and skip projections are ``1x1`` convolutions. Adaptive adjacency
    embeddings are allocated lazily via :meth:`set_num_nodes` when enabled.
    """

    def __init__(self, channels: int, dilation: int, num_nodes: int | None) -> None:
        """Initialize a WaveNet layer.

        Parameters
        ----------
        channels : int
            Feature channel count.
        dilation : int
            Temporal dilation.
        num_nodes : int or None
            Optional node count for adaptive adjacency embeddings. When
            ``None``, adaptive adjacency is disabled until fit-time resize.
        """
        super().__init__()
        self.temporal = _DilatedTemporalConv(channels, dilation)
        self.spatial_fixed = nn.Linear(channels, channels, bias=False)
        self.spatial_adaptive = nn.Linear(channels, channels, bias=False)
        self.residual = nn.Conv2d(channels, channels, kernel_size=1)
        self.skip = nn.Conv2d(channels, channels, kernel_size=1)
        self.num_nodes = num_nodes
        if num_nodes is not None:
            self.adaptive_source = nn.Parameter(torch.randn(num_nodes, 8) * 0.1)
            self.adaptive_target = nn.Parameter(torch.randn(num_nodes, 8) * 0.1)
        else:
            self.register_parameter("adaptive_source", None)
            self.register_parameter("adaptive_target", None)

    def set_num_nodes(self, num_nodes: int) -> None:
        """Allocate or resize adaptive adjacency embeddings.

        Parameters
        ----------
        num_nodes : int
            Graph node count.
        """
        if self.num_nodes == num_nodes and self.adaptive_source is not None:
            return
        device = self.residual.weight.device
        self.num_nodes = num_nodes
        self.adaptive_source = nn.Parameter(
            torch.randn(num_nodes, 8, device=device) * 0.1
        )
        self.adaptive_target = nn.Parameter(
            torch.randn(num_nodes, 8, device=device) * 0.1
        )

    def forward(
        self,
        x: Tensor,
        fixed_adj: Tensor,
        *,
        use_adaptive: bool,
    ) -> tuple[Tensor, Tensor]:
        """Forward one WaveNet layer.

        Parameters
        ----------
        x : Tensor
            Input with shape ``(batch, channels, num_nodes, time)``.
        fixed_adj : Tensor
            Dense normalized adjacency ``(N, N)``.
        use_adaptive : bool
            Whether to mix in the adaptive adjacency branch.

        Returns
        -------
        tuple of Tensor
            Residual stream and skip connection, both shaped like ``x``.
        """
        temporal = self.temporal(x)
        # (B, C, N, T) -> (B, T, N, C)
        spatial_in = temporal.permute(0, 3, 2, 1)
        fixed = torch.einsum("ij,btjc->btic", fixed_adj, spatial_in)
        fixed = self.spatial_fixed(fixed)
        mixed = fixed
        if (
            use_adaptive
            and self.adaptive_source is not None
            and self.adaptive_target is not None
        ):
            adaptive_adj = torch.softmax(
                torch.relu(self.adaptive_source @ self.adaptive_target.transpose(0, 1)),
                dim=1,
            )
            adaptive = torch.einsum("ij,btjc->btic", adaptive_adj, spatial_in)
            mixed = mixed + self.spatial_adaptive(adaptive)
        mixed = mixed.permute(0, 3, 2, 1).contiguous()
        residual = x + self.residual(mixed)
        skip = self.skip(mixed)
        return residual, skip


class GraphWaveNetBaseline(GNNForecasterBaseline):
    """Lightweight Graph WaveNet–style forecaster (Wu et al., IJCAI 2019).

    Dilated causal temporal convolutions with optional adaptive adjacency.
    Reference-scale implementation for in-repo comparisons — not a substitute
    for production traffic libraries.

    :meth:`spectrum` raises :class:`RuntimeError` (no linear Koopman operator).

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
        num_layers: int = 4,
        adaptive_adj: bool = True,
        history_len: int = 4,
        time_step: float = 1.0,
        num_nodes: int | None = None,
    ) -> None:
        """Initialize the Graph WaveNet baseline.

        Parameters
        ----------
        in_channels : int
            Input node-feature dimension.
        hidden_channels : int
            Internal channel width.
        out_channels : int
            Output node-feature dimension.
        num_layers : int, optional
            Number of dilated layers. Default is ``4``.
        adaptive_adj : bool, optional
            Enable adaptive adjacency. Default is ``True``.
        history_len : int, optional
            Lookback window. Default is ``4``.
        time_step : float, optional
            Snapshot interval metadata. Default is ``1.0``.
        num_nodes : int or None, optional
            Optional node count to preallocate adaptive embeddings.
        """
        super().__init__(
            in_channels,
            hidden_channels,
            out_channels,
            history_len=history_len,
            time_step=time_step,
        )
        if num_layers < 1:
            msg = f"num_layers must be positive, got {num_layers}"
            raise ValueError(msg)
        self.adaptive_adj = bool(adaptive_adj)
        self.input_proj = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.layers = nn.ModuleList(
            [
                _GraphWaveNetLayer(
                    hidden_channels, dilation=2**layer, num_nodes=num_nodes
                )
                for layer in range(num_layers)
            ]
        )
        self.output_proj = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )

    def fit(
        self,
        sequence: GraphSnapshotSequence | Sequence[Data],
        *,
        epochs: int = 40,
        lr: float = 1e-3,
        batch_size: int | None = None,
        device: torch.device | str | None = None,
    ) -> GraphWaveNetBaseline:
        """Fit after allocating adaptive adjacency for the graph size.

        Parameters
        ----------
        sequence : GraphSnapshotSequence or sequence of Data
            Training snapshots with fixed topology.
        epochs : int, optional
            Number of Adam epochs. Default is ``40``.
        lr : float, optional
            Adam learning rate. Default is ``1e-3``.
        batch_size : int or None, optional
            Mini-batch size over sliding windows.
        device : torch.device, str, or None, optional
            Training device.

        Returns
        -------
        GraphWaveNetBaseline
            ``self`` for sklearn-style chaining.
        """
        resolved = resolve_sequence(sequence)
        if self.adaptive_adj:
            for layer in self.layers:
                layer.set_num_nodes(resolved.num_nodes)
        return super().fit(
            resolved,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            device=device,
        )

    def predict_next(
        self,
        history: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        """Predict the next frame with dilated temporal and spatial mixing.

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
        num_nodes = history.shape[1]
        if self.adaptive_adj:
            for layer in self.layers:
                layer.set_num_nodes(num_nodes)
        adjacency = random_walk_normalize(
            dense_adjacency(edge_index, edge_weight, num_nodes)
        )
        x = history.permute(2, 1, 0).unsqueeze(0)
        x = self.input_proj(x)
        skip_sum = torch.zeros_like(x)
        for layer in self.layers:
            x, skip = layer(x, adjacency, use_adaptive=self.adaptive_adj)
            skip_sum = skip_sum + skip
        out = self.output_proj(skip_sum)
        return out[0, :, :, -1].transpose(0, 1).contiguous()
