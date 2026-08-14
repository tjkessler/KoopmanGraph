"""Linear state-space / linear-attention teaching spatiotemporal baseline.

A diagonal SSM mix along time plus GCN spatial mixing. Documented teaching
deviations versus dedicated SSM traffic models.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.nn import GCNConv

from koopman_graph.baselines.gnn.base import GNNForecasterBaseline
from koopman_graph.baselines.gnn.protocol import ForecasterProtocol


class SpatiotemporalSSMBaseline(GNNForecasterBaseline):
    """Teaching diagonal SSM + GCN forecaster.

    Parameters
    ----------
    in_channels, hidden_channels, out_channels : int
        Feature widths.
    history_len : int, optional
        Lookback. Default is 4.
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
        time_step: float = 1.0,
    ) -> None:
        """Initialize the teaching SSM baseline.

        Parameters
        ----------
        in_channels, hidden_channels, out_channels : int
            Feature widths.
        history_len : int, optional
            Lookback.
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
        self.lift = nn.Linear(in_channels, hidden_channels)
        self.log_decay = nn.Parameter(torch.zeros(hidden_channels))
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
            name="st_ssm",
            deviations=(
                "architecture: diagonal SSM recurrence + one GCN; not S4/Mamba "
                "or linear-attention traffic papers",
                "training: next-frame windowed fit, not paper multi-horizon "
                "leaderboard schedules",
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
        lifted = self.lift(history)
        decay = torch.sigmoid(self.log_decay).view(1, 1, 1, -1)
        state = torch.zeros_like(lifted[:, 0])
        for step in range(lifted.shape[1]):
            state = decay.squeeze(1) * state + lifted[:, step]
        mixed = []
        for index in range(state.shape[0]):
            mixed.append(self.spatial(state[index], edge_index, edge_weight))
        stacked = torch.stack(mixed, dim=0)
        out = self.readout(stacked)
        if not batched:
            return out.squeeze(0)
        return out
