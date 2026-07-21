"""DCRNN-style diffusion-convolutional recurrent forecaster baseline."""

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
from koopman_graph.graph_utils import snapshot_edge_weight


class _DiffusionConv(nn.Module):
    """Dense diffusion convolution with bidirectional random-walk supports.

    Notes
    -----
    Supports are ``I`` plus powers of forward and backward random-walk
    normalized adjacency up to ``diffusion_steps``.
    """

    def __init__(
        self, in_channels: int, out_channels: int, diffusion_steps: int
    ) -> None:
        """Initialize a diffusion convolution layer.

        Parameters
        ----------
        in_channels : int
            Input feature dimension.
        out_channels : int
            Output feature dimension.
        diffusion_steps : int
            Number of diffusion hops (excluding the identity support).
        """
        super().__init__()
        self.diffusion_steps = int(diffusion_steps)
        # Identity + K forward + K backward supports.
        num_supports = 1 + 2 * self.diffusion_steps
        self.weights = nn.Parameter(
            torch.empty(num_supports, in_channels, out_channels)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.xavier_uniform_(self.weights)

    def forward(self, x: Tensor, supports: list[Tensor]) -> Tensor:
        """Apply diffusion convolution.

        Parameters
        ----------
        x : Tensor
            Node features with shape ``(num_nodes, in_channels)`` or
            ``(batch, num_nodes, in_channels)``.
        supports : list of Tensor
            Dense support matrices, length ``1 + 2 * diffusion_steps``.

        Returns
        -------
        Tensor
            Diffused features with shape ``(num_nodes, out_channels)`` or
            ``(batch, num_nodes, out_channels)``.
        """
        if x.dim() == 2:
            out = x.new_zeros(x.shape[0], self.weights.shape[-1])
            for support, weight in zip(supports, self.weights, strict=True):
                out = out + (support @ x) @ weight
            return out + self.bias
        if x.dim() != 3:
            msg = (
                "x must have shape (N, C) or (batch, N, C), "
                f"got {tuple(x.shape)}"
            )
            raise ValueError(msg)
        out = x.new_zeros(x.shape[0], x.shape[1], self.weights.shape[-1])
        for support, weight in zip(supports, self.weights, strict=True):
            # (B, N, Cin) -> (B, N, Cout) with shared dense support (N, N).
            out = out + torch.einsum("ij,bjc,cd->bid", support, x, weight)
        return out + self.bias


class _DCGRUCell(nn.Module):
    """Diffusion Convolutional GRU cell (DCRNN-style).

    Notes
    -----
    Reset/update gates and the candidate state each use
    :class:`_DiffusionConv` over the shared support set.
    """

    def __init__(
        self, in_channels: int, hidden_channels: int, diffusion_steps: int
    ) -> None:
        """Initialize a DCGRU cell.

        Parameters
        ----------
        in_channels : int
            Input feature dimension.
        hidden_channels : int
            Recurrent hidden dimension.
        diffusion_steps : int
            Diffusion hops for each gate.
        """
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        self.gate = _DiffusionConv(
            in_channels + hidden_channels,
            2 * hidden_channels,
            diffusion_steps,
        )
        self.candidate = _DiffusionConv(
            in_channels + hidden_channels,
            hidden_channels,
            diffusion_steps,
        )

    def forward(
        self,
        x: Tensor,
        hidden: Tensor,
        supports: list[Tensor],
    ) -> Tensor:
        """Advance one DCGRU step.

        Parameters
        ----------
        x : Tensor
            Input features ``(num_nodes, in_channels)`` or
            ``(batch, num_nodes, in_channels)``.
        hidden : Tensor
            Previous hidden state with the same leading layout as ``x``.
        supports : list of Tensor
            Diffusion support matrices.

        Returns
        -------
        Tensor
            Updated hidden state.
        """
        gates = torch.sigmoid(self.gate(torch.cat([x, hidden], dim=-1), supports))
        reset, update = gates.chunk(2, dim=-1)
        candidate = torch.tanh(
            self.candidate(torch.cat([x, reset * hidden], dim=-1), supports)
        )
        return update * hidden + (1.0 - update) * candidate


def _build_diffusion_supports(
    edge_index: Tensor,
    edge_weight: Tensor | None,
    num_nodes: int,
    diffusion_steps: int,
) -> list[Tensor]:
    """Build identity + forward/backward random-walk diffusion supports.

    Parameters
    ----------
    edge_index : Tensor
        COO edge index.
    edge_weight : Tensor or None
        Optional edge weights.
    num_nodes : int
        Node count.
    diffusion_steps : int
        Number of diffusion hops per direction.

    Returns
    -------
    list of Tensor
        Dense supports starting with the identity.
    """
    adjacency = dense_adjacency(edge_index, edge_weight, num_nodes)
    forward = random_walk_normalize(adjacency)
    backward = random_walk_normalize(adjacency.transpose(0, 1))
    identity = torch.eye(num_nodes, device=adjacency.device, dtype=adjacency.dtype)
    supports = [identity]
    forward_power = identity
    backward_power = identity
    for _ in range(diffusion_steps):
        forward_power = forward_power @ forward
        backward_power = backward_power @ backward
        supports.append(forward_power)
        supports.append(backward_power)
    return supports


class DCRNNBaseline(GNNForecasterBaseline):
    """Lightweight DCRNN-style forecaster (Li et al., ICLR 2018).

    Uses a diffusion-convolutional GRU core over dense random-walk supports.
    Sized for in-repo benchmarks; not a drop-in replacement for the original
    DCRNN traffic codebase.

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
        diffusion_steps: int = 2,
        history_len: int = 1,
        time_step: float = 1.0,
    ) -> None:
        """Initialize the DCRNN baseline.

        Parameters
        ----------
        in_channels : int
            Input node-feature dimension.
        hidden_channels : int
            Recurrent hidden width.
        out_channels : int
            Output node-feature dimension.
        diffusion_steps : int, optional
            Diffusion hops per direction. Default is ``2``.
        history_len : int, optional
            Encoder lookback. Default is ``1``.
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
        if diffusion_steps < 1:
            msg = f"diffusion_steps must be positive, got {diffusion_steps}"
            raise ValueError(msg)
        self.diffusion_steps = int(diffusion_steps)
        self.encoder = _DCGRUCell(in_channels, hidden_channels, diffusion_steps)
        self.decoder = _DCGRUCell(out_channels, hidden_channels, diffusion_steps)
        self.readout = nn.Linear(hidden_channels, out_channels)
        self._cached_supports: list[Tensor] | None = None

    def fit(
        self,
        sequence: GraphSnapshotSequence | Sequence[Data],
        *,
        epochs: int = 40,
        lr: float = 1e-3,
        batch_size: int | None = None,
        device: torch.device | str | None = None,
    ) -> DCRNNBaseline:
        """Fit with diffusion supports cached for the static training topology.

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
        DCRNNBaseline
            ``self`` for sklearn-style chaining.
        """
        resolved = resolve_sequence(sequence)
        train_device = (
            torch.device(device)
            if device is not None
            else next(self.parameters()).device
        )
        edge_index = resolved.edge_index.to(train_device)
        edge_weight = snapshot_edge_weight(resolved[0])
        if edge_weight is not None:
            edge_weight = edge_weight.to(train_device)
        self._cached_supports = _build_diffusion_supports(
            edge_index,
            edge_weight,
            resolved.num_nodes,
            self.diffusion_steps,
        )
        try:
            return super().fit(
                resolved,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
                device=device,
            )
        finally:
            self._cached_supports = None

    def predict_next(
        self,
        history: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        """Encode history with DCGRU and decode one future step.

        Parameters
        ----------
        history : Tensor
            History with shape ``(history_len, num_nodes, in_channels)`` or
            ``(batch, history_len, num_nodes, in_channels)``.
        edge_index : Tensor
            Graph connectivity.
        edge_weight : Tensor or None, optional
            Optional edge weights.

        Returns
        -------
        Tensor
            Next-step features with shape ``(num_nodes, out_channels)`` or
            ``(batch, num_nodes, out_channels)`` when ``history`` is batched.
        """
        squeeze = history.dim() == 3
        if squeeze:
            history = history.unsqueeze(0)
        elif history.dim() != 4:
            msg = (
                "history must have shape (history_len, N, C) or "
                f"(batch, history_len, N, C), got {tuple(history.shape)}"
            )
            raise ValueError(msg)
        batch, _history_len, num_nodes, _channels = history.shape
        cached = self._cached_supports
        if cached is not None and cached[0].shape[0] == num_nodes:
            supports = cached
        else:
            supports = _build_diffusion_supports(
                edge_index,
                edge_weight,
                num_nodes,
                self.diffusion_steps,
            )
        hidden = history.new_zeros(batch, num_nodes, self.hidden_channels)
        for step in range(history.shape[1]):
            hidden = self.encoder(history[:, step], hidden, supports)
        # One-step decoder from the last observed frame as teacher input.
        decoder_input = history[:, -1]
        if self.out_channels != self.in_channels:
            decoder_input = self.readout(
                history.new_zeros(batch, num_nodes, self.hidden_channels)
            )
        hidden = self.decoder(decoder_input, hidden, supports)
        out = self.readout(hidden)
        return out[0] if squeeze else out
