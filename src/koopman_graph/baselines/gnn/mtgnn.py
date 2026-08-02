"""MTGNN-style learned-graph dilated temporal forecaster baseline.

Teaching-scale port of Wu et al., KDD 2020 ("Connecting the Dots: Multivariate
Time Series Forecasting with Graph Neural Networks"): asymmetric graph learning
from node embeddings plus dilated temporal convolution. Not a drop-in
replacement for LibCity / the authors' training scripts.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch_geometric.data import Data

from koopman_graph.baselines.gnn.base import GNNForecasterBaseline
from koopman_graph.baselines.gnn.protocol import ForecasterProtocol
from koopman_graph.data import GraphSnapshotSequence, resolve_sequence


def _learned_adjacency(source: Tensor, target: Tensor) -> Tensor:
    """Build a soft uni-directed adjacency from dual node embeddings.

    Parameters
    ----------
    source : Tensor
        Source embeddings with shape ``(num_nodes, embed_dim)``.
    target : Tensor
        Target embeddings with shape ``(num_nodes, embed_dim)``.

    Returns
    -------
    Tensor
        Row-stochastic adjacency with shape ``(num_nodes, num_nodes)``.
    """
    # Asymmetric scoring (paper graph-learning spirit); softmax replaces top-k.
    score = source @ target.transpose(0, 1) - target @ source.transpose(0, 1)
    return torch.softmax(torch.relu(torch.tanh(score)), dim=-1)


class _DilatedTemporalConv(nn.Module):
    """Single-kernel dilated causal temporal convolution with gated activation.

    Notes
    -----
    See class definition.
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
        """One teaching MTGNN block: dilated TCN then learned-graph spatial mix.

Notes
-----
See class definition.
        """
        filter_gate, gate = self.conv(x).chunk(2, dim=1)
        return torch.tanh(filter_gate) * torch.sigmoid(gate)


class _MTGNNLayer(nn.Module):
    """One teaching MTGNN block: dilated TCN then learned-graph spatial mix.

    Notes
    -----
    See class definition."""

    def __init__(self, channels: int, dilation: int) -> None:
        """Initialize temporal and spatial projections.

        Parameters
        ----------
        channels : int
            Feature channel count.
        dilation : int
            Temporal dilation for this block.
        """
        super().__init__()
        self.temporal = _DilatedTemporalConv(channels, dilation)
        self.spatial = nn.Linear(channels, channels, bias=False)
        self.residual = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: Tensor, adjacency: Tensor) -> Tensor:
        """Apply temporal then single-hop spatial mixing.

        Parameters
        ----------
        x : Tensor
            Input with shape ``(batch, channels, num_nodes, time)``.
        adjacency : Tensor
            Learned adjacency ``(num_nodes, num_nodes)``.

        Returns
        -------
        Tensor
            Residual-updated features with the same shape as ``x``.
        """
        temporal = self.temporal(x)
        # (B, C, N, T) -> (B, T, N, C)
        spatial_in = temporal.permute(0, 3, 2, 1)
        mixed = self.spatial(torch.einsum("ij,btjc->btic", adjacency, spatial_in))
        mixed = mixed.permute(0, 3, 2, 1).contiguous()
        return x + self.residual(mixed)


class MTGNNBaseline(GNNForecasterBaseline):
    """Lightweight MTGNN-style forecaster (Wu et al., KDD 2020).

    Combines asymmetric graph learning from dual node embeddings with a short
    stack of dilated temporal convolutions and single-hop spatial mixing. Sized
    for in-repo teaching comparisons with
    :class:`~koopman_graph.model.GraphKoopmanModel` — not a substitute for the
    authors' / LibCity training stacks.

    Fitted ``edge_index`` is accepted for contract parity with other GNN
    baselines but spatial mixing uses the learned adjacency.

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
        embed_dim: int = 8,
        num_layers: int = 2,
        history_len: int = 4,
        time_step: float = 1.0,
        num_nodes: int | None = None,
    ) -> None:
        """Initialize the MTGNN teaching baseline.

        Parameters
        ----------
        in_channels : int
            Input node-feature dimension.
        hidden_channels : int
            Internal channel width.
        out_channels : int
            Output node-feature dimension.
        embed_dim : int, optional
            Graph-learning embedding width. Default is ``8``.
        num_layers : int, optional
            Number of dilated temporal / spatial blocks. Default is ``2``.
        history_len : int, optional
            Encoder lookback. Default is ``4``.
        time_step : float, optional
            Snapshot interval metadata. Default is ``1.0``.
        num_nodes : int or None, optional
            Optional node count to preallocate embeddings; otherwise allocated
            on the first :meth:`fit` / :meth:`predict_next` call.
        """
        super().__init__(
            in_channels,
            hidden_channels,
            out_channels,
            history_len=history_len,
            time_step=time_step,
        )
        if embed_dim < 1:
            msg = f"embed_dim must be positive, got {embed_dim}"
            raise ValueError(msg)
        if num_layers < 1:
            msg = f"num_layers must be positive, got {num_layers}"
            raise ValueError(msg)
        self.embed_dim = int(embed_dim)
        self.input_proj = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.layers = nn.ModuleList(
            [
                _MTGNNLayer(hidden_channels, dilation=2**layer)
                for layer in range(num_layers)
            ]
        )
        self.output_proj = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )
        self.source_embeddings: nn.Parameter | None
        self.target_embeddings: nn.Parameter | None
        self.register_parameter("source_embeddings", None)
        self.register_parameter("target_embeddings", None)
        if num_nodes is not None:
            self.set_num_nodes(num_nodes)

    def set_num_nodes(self, num_nodes: int) -> None:
        """Allocate or validate graph-learning embeddings for ``num_nodes``.

        Parameters
        ----------
        num_nodes : int
            Fixed union node count for dual embeddings.

        Raises
        ------
        ValueError
            If ``num_nodes`` is not positive or conflicts with a prior binding.
        """
        if num_nodes < 1:
            msg = f"num_nodes must be positive, got {num_nodes}"
            raise ValueError(msg)
        if self.source_embeddings is not None:
            if self.source_embeddings.shape[0] == num_nodes:
                return
            msg = (
                "MTGNNBaseline node embeddings are static per fit: "
                f"num_nodes was {self.source_embeddings.shape[0]}, got {num_nodes}"
            )
            raise ValueError(msg)
        device = next(self.parameters()).device
        self.source_embeddings = nn.Parameter(
            torch.randn(num_nodes, self.embed_dim, device=device) * 0.1
        )
        self.target_embeddings = nn.Parameter(
            torch.randn(num_nodes, self.embed_dim, device=device) * 0.1
        )

    def protocol(self) -> ForecasterProtocol:
        """Return the MTGNN teaching protocol (non-empty deviations).

        Returns
        -------
        ForecasterProtocol
            Lookback, claimed evaluation horizon, split ratios, and deviations
            versus Wu et al. / LibCity-style MTGNN scripts.
        """
        return self._teaching_protocol(
            name="mtgnn",
            deviations=(
                "architecture: short single-kernel gated dilated TCN; not Wu et "
                "al. dilated-inception multi-kernel capacity",
                "architecture: single learned-adjacency hop; mix-hop "
                "propagation / retain-ratio stacks omitted",
                "architecture: graph learning uses soft softmax adjacency "
                "without top-k sparsification or external node attributes; "
                "fitted edge_index is ignored for message passing",
            ),
        )

    def fit(
        self,
        sequence: GraphSnapshotSequence | Sequence[Data],
        *,
        epochs: int = 40,
        lr: float = 1e-3,
        batch_size: int | None = None,
        device: torch.device | str | None = None,
    ) -> MTGNNBaseline:
        """Fit after allocating graph-learning embeddings for the graph size.

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
        MTGNNBaseline
            ``self`` for sklearn-style chaining.
        """
        resolved = resolve_sequence(sequence)
        self.set_num_nodes(resolved.num_nodes)
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
        """Predict the next frame with dilated TCN and learned-graph mixing.

        Parameters
        ----------
        history : Tensor
            History with shape ``(history_len, num_nodes, in_channels)`` or
            ``(batch, history_len, num_nodes, in_channels)``.
        edge_index : Tensor
            Accepted for API parity; unused (learned adjacency from embeddings).
        edge_weight : Tensor or None, optional
            Accepted for API parity; unused.

        Returns
        -------
        Tensor
            Next-step features with shape ``(num_nodes, out_channels)`` or
            ``(batch, num_nodes, out_channels)`` when ``history`` is batched.
        """
        del edge_index, edge_weight  # learned graph; contract parity only
        squeeze = history.dim() == 3
        if squeeze:
            history = history.unsqueeze(0)
        elif history.dim() != 4:
            msg = (
                "history must have shape (history_len, N, C) or "
                f"(batch, history_len, N, C), got {tuple(history.shape)}"
            )
            raise ValueError(msg)

        num_nodes = history.shape[2]
        self.set_num_nodes(num_nodes)
        assert self.source_embeddings is not None
        assert self.target_embeddings is not None
        adjacency = _learned_adjacency(self.source_embeddings, self.target_embeddings)

        # (B, C, N, T)
        x = history.permute(0, 3, 2, 1)
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x, adjacency)
        out = self.output_proj(x)
        result = out[:, :, :, -1].transpose(1, 2).contiguous()
        return result[0] if squeeze else result
