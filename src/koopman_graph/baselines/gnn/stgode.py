"""STGODE-style continuous spatial-temporal forecaster baseline.

Teaching-scale port of Fang et al., KDD 2021 ("Spatial-Temporal Graph ODE
Networks for Traffic Flow Forecasting"): a short temporal encoder, then a
graph-coupled ODE integrated with ``torchdiffeq``. Requires the optional
``[baselines-ode]`` extra. Not a drop-in replacement for LibCity / the
authors' PEMS training scripts.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch_geometric.data import Data

from koopman_graph.baselines.gnn.base import (
    GNNForecasterBaseline,
    dense_adjacency,
    random_walk_normalize,
)
from koopman_graph.baselines.gnn.protocol import ForecasterProtocol
from koopman_graph.data import GraphSnapshotSequence, resolve_sequence
from koopman_graph.graph_utils import snapshot_edge_weight

try:
    from torchdiffeq import odeint as _odeint
except ImportError:  # pragma: no cover - exercised via monkeypatch / missing extra
    _odeint = None

# Teaching solver defaults (also listed in protocol deviations).
_SOLVER_METHOD = "dopri5"
_SOLVER_RTOL = 1e-3
_SOLVER_ATOL = 1e-4
_INTEGRATION_TIME = 1.0

_TORCHDIFFEQ_IMPORT_ERROR = (
    "torchdiffeq is required for STGODEBaseline. "
    "Install with: pip install koopman-graph[baselines-ode]"
)


def _require_odeint() -> Callable[..., Any]:
    """Return ``torchdiffeq.odeint`` or raise a clear missing-extra error.

    Returns
    -------
    callable
        ``odeint`` from ``torchdiffeq``.

    Raises
    ------
    ImportError
        If ``torchdiffeq`` is not installed.
    """
    if _odeint is None:
        raise ImportError(_TORCHDIFFEQ_IMPORT_ERROR)
    return _odeint


class _GraphODEFunc(nn.Module):
    """Continuous graph dynamics ``dh/dt = tanh(Â h W + b)``.

    Notes
    -----
    See class definition.
    """

    def __init__(self, channels: int) -> None:
        """Initialize the linear map inside the ODE right-hand side.

        Parameters
        ----------
        channels : int
            Latent channel width.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.empty(channels, channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        nn.init.xavier_uniform_(self.weight)
        self._adjacency: Tensor | None = None

    def set_adjacency(self, adjacency: Tensor) -> None:
        """Bind the dense RW-normalized adjacency used by the RHS.

        Parameters
        ----------
        adjacency : Tensor
            Adjacency with shape ``(num_nodes, num_nodes)``.
        """
        self._adjacency = adjacency

    def forward(self, _t: Tensor, hidden: Tensor) -> Tensor:
        """Evaluate the ODE right-hand side.

        Parameters
        ----------
        _t : Tensor
            Integration time (unused; autonomous dynamics).
        hidden : Tensor
            Latent state ``(batch, num_nodes, channels)``.

        Returns
        -------
        Tensor
            Time derivative with the same shape as ``hidden``.
        """
        if self._adjacency is None:
            msg = "Graph ODE adjacency must be set before integration"
            raise RuntimeError(msg)
        mixed = torch.einsum("ij,bjc->bic", self._adjacency, hidden)
        return torch.tanh(mixed @ self.weight + self.bias)


class STGODEBaseline(GNNForecasterBaseline):
    """Lightweight STGODE-style forecaster (Fang et al., KDD 2021).

    Encodes a lookback window with a short dilated temporal stack, integrates a
    graph-coupled ODE with ``torchdiffeq.odeint``, and reads out the next frame.
    Requires ``pip install koopman-graph[baselines-ode]``. Sized for in-repo
    teaching comparisons — not a substitute for the authors' / LibCity stacks.

    Spatial mixing uses the fitted topology (random-walk normalized). Semantic
    adjacency from the paper is omitted.

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
        num_layers: int = 2,
        history_len: int = 4,
        time_step: float = 1.0,
        integration_time: float = _INTEGRATION_TIME,
        solver_method: str = _SOLVER_METHOD,
        solver_rtol: float = _SOLVER_RTOL,
        solver_atol: float = _SOLVER_ATOL,
    ) -> None:
        """Initialize the STGODE teaching baseline.

        Parameters
        ----------
        in_channels : int
            Input node-feature dimension.
        hidden_channels : int
            Latent / ODE channel width.
        out_channels : int
            Output node-feature dimension.
        num_layers : int, optional
            Number of dilated temporal encoder layers. Default is ``2``.
        history_len : int, optional
            Encoder lookback. Default is ``4``.
        time_step : float, optional
            Snapshot interval metadata. Default is ``1.0``.
        integration_time : float, optional
            ODE integration end time (start is ``0``). Default is ``1.0``.
        solver_method : str, optional
            ``torchdiffeq.odeint`` method. Default is ``\"dopri5\"``.
        solver_rtol : float, optional
            Relative solver tolerance. Default is ``1e-3``.
        solver_atol : float, optional
            Absolute solver tolerance. Default is ``1e-4``.
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
        if integration_time <= 0:
            msg = f"integration_time must be positive, got {integration_time}"
            raise ValueError(msg)
        if solver_rtol <= 0 or solver_atol <= 0:
            msg = (
                "solver_rtol and solver_atol must be positive, "
                f"got rtol={solver_rtol}, atol={solver_atol}"
            )
            raise ValueError(msg)
        self.integration_time = float(integration_time)
        self.solver_method = str(solver_method)
        self.solver_rtol = float(solver_rtol)
        self.solver_atol = float(solver_atol)
        self.input_proj = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        # Causal dilated TCN: left-pad in forward; no built-in Conv2d padding.
        self.temporal_layers = nn.ModuleList(
            [
                nn.Conv2d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=(1, 2),
                    dilation=(1, 2**layer),
                )
                for layer in range(num_layers)
            ]
        )
        self.ode_func = _GraphODEFunc(hidden_channels)
        self.readout = nn.Linear(hidden_channels, out_channels)
        self._cached_rw_adj: Tensor | None = None

    def protocol(self) -> ForecasterProtocol:
        """Return the STGODE teaching protocol (non-empty deviations).

        Returns
        -------
        ForecasterProtocol
            Lookback, claimed evaluation horizon, split ratios, and deviations
            versus Fang et al. / LibCity-style STGODE scripts.
        """
        return self._teaching_protocol(
            name="stgode",
            deviations=(
                f"solver: torchdiffeq.odeint method={self.solver_method!r} with "
                f"rtol={self.solver_rtol}, atol={self.solver_atol}, "
                f"integration_time={self.integration_time}; not Fang et al. "
                "solver settings or PEMS training hyperparameters",
                "architecture: simplified node-feature graph ODE "
                "(dh/dt = tanh(Â h W + b)); paper tensor STGODE capacity "
                "omitted",
                "architecture: semantic adjacency omitted; spatial mix uses "
                "fitted random-walk topology only",
                "architecture: short dilated temporal encoder, not paper "
                "temporal block capacity",
                "dependency: requires optional [baselines-ode] (torchdiffeq)",
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
    ) -> STGODEBaseline:
        """Fit after caching RW-normalized adjacency for the static graph.

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
        STGODEBaseline
            ``self`` for sklearn-style chaining.
        """
        _require_odeint()
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
        self._cached_rw_adj = random_walk_normalize(
            dense_adjacency(edge_index, edge_weight, resolved.num_nodes)
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
            self._cached_rw_adj = None

    def predict_next(
        self,
        history: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        """Predict the next frame via temporal encode + graph ODE integrate.

        Parameters
        ----------
        history : Tensor
            History with shape ``(history_len, num_nodes, in_channels)`` or
            ``(batch, history_len, num_nodes, in_channels)``.
        edge_index : Tensor
            Graph connectivity for RW-normalized spatial mixing.
        edge_weight : Tensor or None, optional
            Optional edge weights.

        Returns
        -------
        Tensor
            Next-step features with shape ``(num_nodes, out_channels)`` or
            ``(batch, num_nodes, out_channels)`` when ``history`` is batched.
        """
        odeint = _require_odeint()
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
        cached = self._cached_rw_adj
        if cached is not None and cached.shape[0] == num_nodes:
            adjacency = cached
        else:
            adjacency = random_walk_normalize(
                dense_adjacency(edge_index, edge_weight, num_nodes)
            )
        self.ode_func.set_adjacency(adjacency)

        # (B, C, N, T) with causal left-padding per dilated layer.
        x = self.input_proj(history.permute(0, 3, 2, 1))
        for layer in self.temporal_layers:
            dilation = int(layer.dilation[1])
            x = torch.relu(layer(nn.functional.pad(x, (dilation, 0))))
        hidden0 = x[:, :, :, -1].transpose(1, 2).contiguous()
        times = hidden0.new_tensor([0.0, self.integration_time])
        trajectory = odeint(
            self.ode_func,
            hidden0,
            times,
            method=self.solver_method,
            rtol=self.solver_rtol,
            atol=self.solver_atol,
        )
        output = self.readout(trajectory[-1])
        return output[0] if squeeze else output
