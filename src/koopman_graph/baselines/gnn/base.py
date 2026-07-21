"""Shared scaffolding for spatiotemporal GNN forecaster baselines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch_geometric.data import Data

from koopman_graph.baselines.base import (
    check_initial_graph,
    copy_topology,
    require_static_topology,
)
from koopman_graph.data import GraphSnapshotSequence, resolve_sequence
from koopman_graph.graph_utils import snapshot_edge_weight, snapshot_to_device
from koopman_graph.spectrum_types import KoopmanSpectrum


def stack_node_features(sequence: GraphSnapshotSequence) -> Tensor:
    """Stack snapshot node features into a ``(T, N, C)`` tensor.

    Parameters
    ----------
    sequence : GraphSnapshotSequence
        Graph snapshots with shared feature layout.

    Returns
    -------
    Tensor
        Stacked features with shape ``(num_timesteps, num_nodes, in_channels)``.
    """
    return torch.stack([snapshot.x for snapshot in sequence], dim=0)


def dense_adjacency(
    edge_index: Tensor,
    edge_weight: Tensor | None,
    num_nodes: int,
) -> Tensor:
    """Build a dense adjacency matrix from sparse topology.

    Parameters
    ----------
    edge_index : Tensor
        COO edge index with shape ``(2, E)``.
    edge_weight : Tensor or None
        Optional edge weights with shape ``(E,)``. ``None`` uses ones.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    Tensor
        Dense adjacency with shape ``(num_nodes, num_nodes)``.
    """
    weights = (
        torch.ones(edge_index.shape[1], dtype=torch.float32, device=edge_index.device)
        if edge_weight is None
        else edge_weight.to(dtype=torch.float32, device=edge_index.device)
    )
    adjacency = torch.zeros(
        (num_nodes, num_nodes),
        dtype=weights.dtype,
        device=edge_index.device,
    )
    adjacency.index_put_((edge_index[0], edge_index[1]), weights, accumulate=True)
    return adjacency


def random_walk_normalize(adjacency: Tensor) -> Tensor:
    """Row-normalize an adjacency matrix (``D^{-1} A``).

    Parameters
    ----------
    adjacency : Tensor
        Dense adjacency with shape ``(N, N)``.

    Returns
    -------
    Tensor
        Random-walk normalized adjacency.
    """
    degree = adjacency.sum(dim=1).clamp_min(1e-6)
    return adjacency / degree.unsqueeze(1)


class GNNForecasterBaseline(nn.Module, ABC):
    """Shared ``fit`` / ``predict`` scaffolding for GNN forecaster baselines.

    Concrete models implement :meth:`predict_next` for one-step forecasting.
    Training uses sliding windows of length :attr:`history_len` to predict the
    next frame. Autoregressive :meth:`predict` pads missing history by repeating
    the initial snapshot (a documented simplification vs paper lookbacks).

    Attributes
    ----------
    in_channels : int
        Input node-feature dimension.
    hidden_channels : int
        Internal width.
    out_channels : int
        Output node-feature dimension.
    history_len : int
        Temporal lookback used during training and rollout.
    time_step : float
        Physical duration of one snapshot transition (metadata only).
    control_dim : int
        Always ``0``; these baselines are uncontrolled.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        *,
        history_len: int = 1,
        time_step: float = 1.0,
    ) -> None:
        """Initialize shared forecaster hyperparameters.

        Parameters
        ----------
        in_channels : int
            Input node-feature dimension.
        hidden_channels : int
            Internal width.
        out_channels : int
            Output node-feature dimension.
        history_len : int, optional
            Temporal lookback length. Default is ``1``.
        time_step : float, optional
            Physical duration of one snapshot transition. Default is ``1.0``.

        Raises
        ------
        ValueError
            If any dimension or ``time_step`` is invalid.
        """
        super().__init__()
        for name, value in {
            "in_channels": in_channels,
            "hidden_channels": hidden_channels,
            "out_channels": out_channels,
            "history_len": history_len,
        }.items():
            if value < 1:
                msg = f"{name} must be positive, got {value}"
                raise ValueError(msg)
        if time_step <= 0:
            msg = f"time_step must be positive, got {time_step}"
            raise ValueError(msg)
        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.out_channels = int(out_channels)
        self.history_len = int(history_len)
        self.time_step = float(time_step)
        self.control_dim = 0
        self.num_nodes: int | None = None
        self._fitted = False
        self._topology: dict[str, Tensor] | None = None

    @abstractmethod
    def predict_next(
        self,
        history: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        """Predict the next node-feature frame from a history window.

        Parameters
        ----------
        history : Tensor
            History with shape ``(history_len, num_nodes, in_channels)`` or a
            batched stack ``(batch, history_len, num_nodes, in_channels)``.
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

    def fit(
        self,
        sequence: GraphSnapshotSequence | Sequence[Data],
        *,
        epochs: int = 40,
        lr: float = 1e-3,
        batch_size: int | None = None,
        device: torch.device | str | None = None,
    ) -> GNNForecasterBaseline:
        """Fit the forecaster on consecutive snapshot windows.

        Parameters
        ----------
        sequence : GraphSnapshotSequence or sequence of Data
            Training snapshots with fixed topology.
        epochs : int, optional
            Number of Adam epochs. Default is ``40``.
        lr : float, optional
            Adam learning rate. Default is ``1e-3``.
        batch_size : int or None, optional
            Mini-batch size over sliding windows. ``None`` uses full-batch.
            Each batch is forwarded in one vectorized :meth:`predict_next`
            call (not a Python loop over windows).
        device : torch.device, str, or None, optional
            Training device. Defaults to the module parameter device.

        Returns
        -------
        GNNForecasterBaseline
            ``self`` for sklearn-style chaining.

        Raises
        ------
        ValueError
            If the sequence is too short, has dynamic topology, or feature
            dimensions do not match the constructor.
        """
        resolved = resolve_sequence(sequence)
        require_static_topology(resolved)
        if resolved.num_timesteps < self.history_len + 1:
            msg = (
                f"{type(self).__name__}.fit requires at least "
                f"{self.history_len + 1} snapshots for history_len="
                f"{self.history_len}, got {resolved.num_timesteps}"
            )
            raise ValueError(msg)
        if resolved.in_channels != self.in_channels:
            msg = (
                f"sequence in_channels={resolved.in_channels} does not match "
                f"model in_channels={self.in_channels}"
            )
            raise ValueError(msg)
        if epochs < 1:
            msg = f"epochs must be positive, got {epochs}"
            raise ValueError(msg)
        if lr <= 0:
            msg = f"lr must be positive, got {lr}"
            raise ValueError(msg)

        train_device = (
            torch.device(device)
            if device is not None
            else next(self.parameters()).device
        )
        self.to(train_device)
        features = stack_node_features(resolved).to(train_device)
        edge_index = resolved.edge_index.to(train_device)
        edge_weight = snapshot_edge_weight(resolved[0])
        if edge_weight is not None:
            edge_weight = edge_weight.to(train_device)

        if self.out_channels != self.in_channels:
            msg = (
                "out_channels must equal in_channels for autoregressive "
                f"training, got {self.out_channels} vs {self.in_channels}"
            )
            raise ValueError(msg)

        # (n_windows, history_len, N, C) and (n_windows, N, C)
        histories = torch.stack(
            [
                features[start : start + self.history_len]
                for start in range(0, resolved.num_timesteps - self.history_len)
            ],
            dim=0,
        )
        targets = torch.stack(
            [
                features[start + self.history_len]
                for start in range(0, resolved.num_timesteps - self.history_len)
            ],
            dim=0,
        )

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.train()
        n_windows = histories.shape[0]
        effective_batch = (
            n_windows if batch_size is None else min(batch_size, n_windows)
        )
        if effective_batch < 1:
            msg = "no training windows available"
            raise ValueError(msg)

        for _ in range(epochs):
            permutation = torch.randperm(n_windows, device=train_device)
            for batch_start in range(0, n_windows, effective_batch):
                indices = permutation[batch_start : batch_start + effective_batch]
                prediction = self.predict_next(
                    histories[indices],
                    edge_index,
                    edge_weight,
                )
                loss = nn.functional.mse_loss(prediction, targets[indices])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.num_nodes = resolved.num_nodes
        self._topology = {
            key: value.detach().cpu()
            for key, value in copy_topology(resolved[0]).items()
        }
        self._fitted = True
        self.eval()
        return self

    def predict(
        self,
        initial_graph: Data,
        steps: int,
        controls: Sequence[Tensor] | Tensor | None = None,
        future_topologies: Sequence[Data] | None = None,
    ) -> list[Data]:
        """Autoregressively predict future graph snapshots.

        Parameters
        ----------
        initial_graph : Data
            Initial PyG snapshot (and history seed when ``history_len > 1``).
        steps : int
            Number of future snapshots to predict.
        controls : ignored
            Must be ``None``; these baselines are uncontrolled.
        future_topologies : ignored
            Must be ``None``; topology is frozen from the fitted / initial graph.

        Returns
        -------
        list of Data
            Predicted snapshots of length ``steps``.

        Raises
        ------
        RuntimeError
            If the baseline has not been fit.
        ValueError
            If ``steps`` is invalid, controls/topologies are provided, or the
            initial graph shape does not match fit-time metadata.
        """
        self._check_fitted()
        if steps < 1:
            msg = f"steps must be >= 1, got {steps}"
            raise ValueError(msg)
        if controls is not None:
            msg = f"{type(self).__name__} does not support controls"
            raise ValueError(msg)
        if future_topologies is not None:
            msg = (
                f"{type(self).__name__} does not support future_topologies; "
                "topology is frozen from the initial graph"
            )
            raise ValueError(msg)
        assert self.num_nodes is not None  # guarded by _check_fitted
        check_initial_graph(
            initial_graph,
            num_nodes=self.num_nodes,
            in_channels=self.in_channels,
        )

        device = next(self.parameters()).device
        graph = snapshot_to_device(initial_graph, device)
        edge_index = graph.edge_index
        edge_weight = snapshot_edge_weight(graph)
        history = graph.x.unsqueeze(0).repeat(self.history_len, 1, 1)
        topology = copy_topology(initial_graph)

        predictions: list[Data] = []
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                for _ in range(steps):
                    next_x = self.predict_next(history, edge_index, edge_weight)
                    predictions.append(
                        Data(
                            x=next_x.detach().cpu(),
                            **{key: value.clone() for key, value in topology.items()},
                        )
                    )
                    history = torch.cat([history[1:], next_x.unsqueeze(0)], dim=0)
        finally:
            self.train(was_training)
        return predictions

    def spectrum(self, *args: object, **kwargs: object) -> KoopmanSpectrum:
        """GNN forecasters have no linear Koopman operator spectrum.

        Raises
        ------
        RuntimeError
            Always. Use :class:`~koopman_graph.model.GraphKoopmanModel` or
            classical DMD/EDMD baselines for spectral analysis.
        """
        msg = (
            f"{type(self).__name__} has no linear Koopman operator spectrum; "
            "use GraphKoopmanModel or classical DMD/EDMD baselines for "
            "spectral analysis"
        )
        raise RuntimeError(msg)

    def _check_fitted(self) -> None:
        """Raise if :meth:`fit` has not been called.

        Raises
        ------
        RuntimeError
            If fit-time metadata is missing.
        """
        if not self._fitted or self.num_nodes is None:
            msg = f"{type(self).__name__} must be fit before prediction"
            raise RuntimeError(msg)
