"""STAEformer-class teaching spatiotemporal attention baseline.

A single temporal self-attention block plus GCN mixing. Documented
deviations versus Guo / Liu STAEformer-class scripts.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.nn import GCNConv

from koopman_graph.baselines.gnn.base import GNNForecasterBaseline
from koopman_graph.baselines.gnn.protocol import ForecasterProtocol


class STAEformerBaseline(GNNForecasterBaseline):
    """Teaching temporal-attention + GCN forecaster.

    Parameters
    ----------
    in_channels, hidden_channels, out_channels : int
        Feature widths.
    history_len : int, optional
        Lookback. Default is 4.
    n_heads : int, optional
        Attention heads. Default is 2.
    time_step : float, optional
        Snapshot interval. Default is 1.0.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        *,
        history_len: int = 4,
        n_heads: int = 2,
        time_step: float = 1.0,
    ) -> None:
        """Initialize the teaching attention baseline.

        Parameters
        ----------
        in_channels, hidden_channels, out_channels : int
            Feature widths.
        history_len : int, optional
            Lookback.
        n_heads : int, optional
            Attention heads.
        time_step : float, optional
            Snapshot interval.
        """
        super().__init__(
            in_channels,
            hidden_channels,
            out_channels,
            history_len=history_len,
            time_step=time_step,
        )
        if n_heads < 1:
            raise ValueError(f"n_heads must be positive, got {n_heads}")
        self.n_heads = int(n_heads)
        self.lift = nn.Linear(in_channels, hidden_channels)
        self.attn = nn.MultiheadAttention(
            hidden_channels,
            self.n_heads,
            batch_first=True,
        )
        self.spatial = GCNConv(hidden_channels, hidden_channels)
        self.readout = nn.Linear(hidden_channels, out_channels)

    def protocol(self) -> ForecasterProtocol:
        """Return the teaching protocol (non-empty deviations).

        Returns
        -------
        ForecasterProtocol
            Named teaching protocol.
        """
        return self._teaching_protocol(
            name="staeformer",
            deviations=(
                "architecture: one temporal MultiheadAttention + GCN; not "
                "STAEformer encoder stacks or adaptive embeddings",
                "training: next-frame windowed fit, not paper 12-step MAE "
                "schedules on METR-LA / PEMS",
            ),
        )

    def predict_next(
        self,
        history: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        """Predict the next frame from a history window.

        Parameters
        ----------
        history : Tensor
            ``(history_len, N, F)`` or batched ``(B, history_len, N, F)``.
        edge_index : Tensor
            Graph connectivity.
        edge_weight : Tensor or None, optional
            Optional edge weights.

        Returns
        -------
        Tensor
            Next-frame node features.
        """
        batched = history.ndim == 4
        if not batched:
            history = history.unsqueeze(0)
        batch, lookback, num_nodes, _ = history.shape
        tokens = self.lift(
            history.permute(0, 2, 1, 3).reshape(batch * num_nodes, lookback, -1)
        )
        attended, _ = self.attn(tokens, tokens, tokens)
        last = attended[:, -1].view(batch, num_nodes, self.hidden_channels)
        mixed = []
        for index in range(batch):
            mixed.append(self.spatial(last[index], edge_index, edge_weight))
        stacked = torch.stack(mixed, dim=0)
        out = self.readout(stacked)
        if not batched:
            return out.squeeze(0)
        return out
