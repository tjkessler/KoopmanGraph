"""AGCRN-style adaptive graph convolutional recurrent forecaster baseline.

Teaching-scale port of Bai et al., NeurIPS 2020 (Adaptive Graph Convolutional
Recurrent Network): data-adaptive graph generation (DAGG) via node embeddings
and node-adaptive parameter learning (NAPL) inside a GRU cell. Not a drop-in
replacement for LibCity / the authors' PEMS training scripts.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch_geometric.data import Data

from koopman_graph.baselines.gnn.base import GNNForecasterBaseline
from koopman_graph.baselines.gnn.protocol import ForecasterProtocol
from koopman_graph.data import GraphSnapshotSequence, resolve_sequence


def _adaptive_adjacency(node_embeddings: Tensor) -> Tensor:
    """Build a row-stochastic adaptive adjacency from node embeddings (DAGG).

    Parameters
    ----------
    node_embeddings : Tensor
        Learnable embeddings with shape ``(num_nodes, embed_dim)``.

    Returns
    -------
    Tensor
        Soft adjacency with shape ``(num_nodes, num_nodes)``.
    """
    raw = torch.relu(node_embeddings @ node_embeddings.transpose(0, 1))
    return torch.softmax(raw, dim=-1)


class _NAPLGCN(nn.Module):
    """Node-adaptive graph convolution via shared weight / bias pools (NAPL).

    Notes
    -----
    See class definition.
    """

    def __init__(self, in_channels: int, out_channels: int, embed_dim: int) -> None:
        """Initialize NAPL pools.

        Parameters
        ----------
        in_channels : int
            Input feature width.
        out_channels : int
            Output feature width.
        embed_dim : int
            Node embedding dimension shared with DAGG.
        """
        super().__init__()
        self.weights_pool = nn.Parameter(
            torch.empty(embed_dim, in_channels, out_channels)
        )
        self.bias_pool = nn.Parameter(torch.zeros(embed_dim, out_channels))
        nn.init.xavier_uniform_(self.weights_pool)

    def forward(
        self,
        x: Tensor,
        node_embeddings: Tensor,
        adjacency: Tensor,
    ) -> Tensor:
        """Apply adaptive-graph aggregation then node-specific linear maps.

        Parameters
        ----------
        x : Tensor
            Features ``(num_nodes, in_channels)`` or
            ``(batch, num_nodes, in_channels)``.
        node_embeddings : Tensor
            Embeddings ``(num_nodes, embed_dim)``.
        adjacency : Tensor
            Adaptive adjacency ``(num_nodes, num_nodes)``.

        Returns
        -------
        Tensor
            Transformed features with the same leading layout as ``x``.
        """
        weights = torch.einsum("nd,dio->nio", node_embeddings, self.weights_pool)
        bias = node_embeddings @ self.bias_pool
        if x.dim() == 2:
            aggregated = adjacency @ x
            return torch.einsum("ni,nio->no", aggregated, weights) + bias
        if x.dim() != 3:
            msg = f"x must have shape (N, C) or (batch, N, C), got {tuple(x.shape)}"
            raise ValueError(msg)
            """GRU cell whose gates use NAPL-GCN (teaching AGCRN core).

Notes
-----
See class definition.
            """
        aggregated = torch.einsum("ij,bjc->bic", adjacency, x)
        return torch.einsum("bni,nio->bno", aggregated, weights) + bias


class _AGCRNCell(nn.Module):
    """GRU cell whose gates use NAPL-GCN (teaching AGCRN core).

    Notes
    -----
    See class definition."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        embed_dim: int,
    ) -> None:
        """Initialize reset/update and candidate NAPL-GCN maps.

        Parameters
        ----------
        in_channels : int
            Input feature dimension.
        hidden_channels : int
            Recurrent hidden width.
        embed_dim : int
            Node embedding dimension.
        """
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        self.gate = _NAPLGCN(
            in_channels + hidden_channels,
            2 * hidden_channels,
            embed_dim,
        )
        self.candidate = _NAPLGCN(
            in_channels + hidden_channels,
            hidden_channels,
            embed_dim,
        )

    def forward(
        self,
        x: Tensor,
        hidden: Tensor,
        node_embeddings: Tensor,
        adjacency: Tensor,
    ) -> Tensor:
        """Advance one AGCRN step.

        Parameters
        ----------
        x : Tensor
            Input ``(N, Cin)`` or ``(batch, N, Cin)``.
        hidden : Tensor
            Previous hidden state with matching layout.
        node_embeddings : Tensor
            Node embeddings ``(N, embed_dim)``.
        adjacency : Tensor
            Adaptive adjacency ``(N, N)``.

        Returns
        -------
        Tensor
            Updated hidden state.
        """
        gates = torch.sigmoid(
            self.gate(torch.cat([x, hidden], dim=-1), node_embeddings, adjacency)
        )
        reset, update = gates.chunk(2, dim=-1)
        candidate = torch.tanh(
            self.candidate(
                torch.cat([x, reset * hidden], dim=-1),
                node_embeddings,
                adjacency,
            )
        )
        return update * hidden + (1.0 - update) * candidate


class AGCRNBaseline(GNNForecasterBaseline):
    """Lightweight AGCRN-style forecaster (Bai et al., NeurIPS 2020).

    Combines data-adaptive graph generation (node embeddings → soft adjacency)
    with node-adaptive parameter learning inside a single GRU cell. Sized for
    in-repo teaching comparisons with
    :class:`~koopman_graph.model.GraphKoopmanModel` — not a substitute for the
    authors' PEMS / LibCity training stacks.

    Fitted ``edge_index`` is accepted for contract parity with other GNN
    baselines but spatial mixing uses the learned adaptive adjacency.

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
        history_len: int = 4,
        time_step: float = 1.0,
        num_nodes: int | None = None,
    ) -> None:
        """Initialize the AGCRN teaching baseline.

        Parameters
        ----------
        in_channels : int
            Input node-feature dimension.
        hidden_channels : int
            Recurrent hidden width.
        out_channels : int
            Output node-feature dimension.
        embed_dim : int, optional
            DAGG / NAPL embedding width. Default is ``8``.
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
        self.embed_dim = int(embed_dim)
        self.cell = _AGCRNCell(in_channels, hidden_channels, self.embed_dim)
        self.readout = nn.Linear(hidden_channels, out_channels)
        self.node_embeddings: nn.Parameter | None
        self.register_parameter("node_embeddings", None)
        if num_nodes is not None:
            self.set_num_nodes(num_nodes)

    def set_num_nodes(self, num_nodes: int) -> None:
        """Allocate or validate node embeddings for universe size ``num_nodes``.

        Parameters
        ----------
        num_nodes : int
            Fixed union node count for DAGG embeddings.

        Raises
        ------
        ValueError
            If ``num_nodes`` is not positive or conflicts with a prior binding.
        """
        if num_nodes < 1:
            msg = f"num_nodes must be positive, got {num_nodes}"
            raise ValueError(msg)
        if self.node_embeddings is not None:
            if self.node_embeddings.shape[0] == num_nodes:
                return
            msg = (
                "AGCRNBaseline node embeddings are static per fit: "
                f"num_nodes was {self.node_embeddings.shape[0]}, got {num_nodes}"
            )
            raise ValueError(msg)
        device = next(self.parameters()).device
        self.node_embeddings = nn.Parameter(
            torch.randn(num_nodes, self.embed_dim, device=device) * 0.1
        )

    def protocol(self) -> ForecasterProtocol:
        """Return the AGCRN teaching protocol (non-empty deviations).

        Returns
        -------
        ForecasterProtocol
            Lookback, claimed evaluation horizon, split ratios, and deviations
            versus Bai et al. / LibCity-style AGCRN scripts.
        """
        return self._teaching_protocol(
            name="agcrn",
            deviations=(
                "architecture: single AGCRN cell with teaching-scale embed_dim / "
                "hidden width; not Bai et al. PEMS-D4 / PEMS-D8 capacity",
                "architecture: spatial mixing uses simplified DAGG "
                "softmax(ReLU(EE^T)); fitted edge_index is ignored for message "
                "passing",
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
    ) -> AGCRNBaseline:
        """Fit after allocating DAGG embeddings for the graph size.

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
        AGCRNBaseline
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
        """Predict the next frame by unrolling the AGCRN cell over ``history``.

        Parameters
        ----------
        history : Tensor
            History with shape ``(history_len, num_nodes, in_channels)`` or
            ``(batch, history_len, num_nodes, in_channels)``.
        edge_index : Tensor
            Accepted for API parity; unused (adaptive adjacency from embeddings).
        edge_weight : Tensor or None, optional
            Accepted for API parity; unused.

        Returns
        -------
        Tensor
            Next-step features with shape ``(num_nodes, out_channels)`` or
            ``(batch, num_nodes, out_channels)`` when ``history`` is batched.
        """
        del edge_index, edge_weight  # adaptive graph; contract parity only
        squeeze = history.dim() == 3
        if squeeze:
            history = history.unsqueeze(0)
        elif history.dim() != 4:
            msg = (
                "history must have shape (history_len, N, C) or "
                f"(batch, history_len, N, C), got {tuple(history.shape)}"
            )
            raise ValueError(msg)

        batch, history_len, num_nodes, _channels = history.shape
        self.set_num_nodes(num_nodes)
        assert self.node_embeddings is not None
        adjacency = _adaptive_adjacency(self.node_embeddings)

        hidden = history.new_zeros(batch, num_nodes, self.hidden_channels)
        for step in range(history_len):
            hidden = self.cell(
                history[:, step],
                hidden,
                self.node_embeddings,
                adjacency,
            )
        output = self.readout(hidden)
        if squeeze:
            return output.squeeze(0)
        return output
