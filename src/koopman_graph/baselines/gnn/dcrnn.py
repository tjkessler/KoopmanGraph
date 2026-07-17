"""DCRNN-style diffusion-convolutional recurrent forecaster baseline."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from koopman_graph.baselines.gnn.base import (
    GNNForecasterBaseline,
    dense_adjacency,
    random_walk_normalize,
)


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
            Node features with shape ``(num_nodes, in_channels)``.
        supports : list of Tensor
            Dense support matrices, length ``1 + 2 * diffusion_steps``.

        Returns
        -------
        Tensor
            Diffused features with shape ``(num_nodes, out_channels)``.
        """
        out = x.new_zeros(x.shape[0], self.weights.shape[-1])
        for support, weight in zip(supports, self.weights, strict=True):
            out = out + (support @ x) @ weight
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
            Input features ``(num_nodes, in_channels)``.
        hidden : Tensor
            Previous hidden state ``(num_nodes, hidden_channels)``.
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
        supports = _build_diffusion_supports(
            edge_index,
            edge_weight,
            num_nodes,
            self.diffusion_steps,
        )
        hidden = history.new_zeros(num_nodes, self.hidden_channels)
        for step in range(history.shape[0]):
            hidden = self.encoder(history[step], hidden, supports)
        # One-step decoder from the last observed frame as teacher input.
        decoder_input = history[-1]
        if self.out_channels != self.in_channels:
            decoder_input = self.readout(
                history.new_zeros(num_nodes, self.hidden_channels)
            )
        hidden = self.decoder(decoder_input, hidden, supports)
        return self.readout(hidden)
