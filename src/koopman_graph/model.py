"""GraphKoopmanModel: encoder, Koopman operator, and decoder composition."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch_geometric.data import Data

from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.decoder import GNNDecoder
from koopman_graph.encoder import GATEncoder, GNNEncoder
from koopman_graph.operator import InitMode, KoopmanOperator
from koopman_graph.training import (
    FitHistory,
    LossWeights,
    LossWeightSchedule,
    resolve_device,
    resolve_loss_weights_for_epoch,
    resolve_sequence,
    should_stop_early,
    train_one_epoch,
)

Encoder = GNNEncoder | GATEncoder


class GraphKoopmanModel(nn.Module):
    """Topology-aware Koopman dynamics model for graph snapshots.

    Composes a GNN encoder (lifting), a finite-dimensional Koopman operator
    (linear latent evolution), and a symmetric GNN decoder (reconstruction).

    Attributes
    ----------
    encoder : GNNEncoder or GATEncoder
        Topology-aware encoder for latent lifting.
    decoder : GNNDecoder
        Symmetric GNN decoder for physical reconstruction.
    latent_dim : int
        Latent space dimension shared by encoder, operator, and decoder.
    time_step : float
        Physical time increment associated with one model step.
    koopman : KoopmanOperator
        Learnable linear propagator in latent space.
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: GNNDecoder,
        latent_dim: int,
        time_step: float,
        *,
        koopman_init_mode: InitMode = "identity_noise",
        koopman_init_scale: float = 1e-2,
    ) -> None:
        """Initialize encoder, decoder, and Koopman operator.

        Parameters
        ----------
        encoder : GNNEncoder or GATEncoder
            Topology-aware encoder for latent lifting.
        decoder : GNNDecoder
            Symmetric GNN decoder for physical reconstruction.
        latent_dim : int
            Latent space dimension shared by encoder, operator, and decoder.
        time_step : float
            Physical time increment associated with one model step.
        koopman_init_mode : {"identity", "identity_noise", "xavier"}, optional
            Initialization strategy for the Koopman matrix. Default is
            ``"identity_noise"``.
        koopman_init_scale : float, optional
            Noise scale when ``koopman_init_mode="identity_noise"``.
            Default is ``1e-2``.

        Raises
        ------
        ValueError
            If ``latent_dim`` is not positive, ``time_step <= 0``, or encoder/
            decoder latent dimensions do not match ``latent_dim``.
        """
        super().__init__()
        if latent_dim < 1:
            msg = f"latent_dim must be positive, got {latent_dim}"
            raise ValueError(msg)
        if time_step <= 0:
            msg = f"time_step must be positive, got {time_step}"
            raise ValueError(msg)
        if encoder.latent_dim != latent_dim:
            msg = (
                f"encoder.latent_dim ({encoder.latent_dim}) must match "
                f"latent_dim ({latent_dim})"
            )
            raise ValueError(msg)
        if decoder.latent_dim != latent_dim:
            msg = (
                f"decoder.latent_dim ({decoder.latent_dim}) must match "
                f"latent_dim ({latent_dim})"
            )
            raise ValueError(msg)

        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.time_step = time_step
        self.koopman = KoopmanOperator(
            latent_dim,
            init_mode=koopman_init_mode,
            init_scale=koopman_init_scale,
        )

    def forward(
        self,
        x_or_data: Tensor | Data,
        edge_index: Tensor | None = None,
    ) -> Tensor:
        """Predict the next graph snapshot from the current one.

        Performs encode → linear Koopman advance → decode for a single step.

        Parameters
        ----------
        x_or_data : Tensor or Data
            Either a PyG ``Data`` object or node features ``x`` of shape
            ``(num_nodes, in_channels)``.
        edge_index : Tensor, optional
            Edge index with shape ``(2, num_edges)``. Required when
            ``x_or_data`` is a tensor; ignored for ``Data`` input.

        Returns
        -------
        Tensor
            Predicted node features of shape ``(num_nodes, out_channels)``.
        """
        edge_index = self._resolve_edge_index(x_or_data, edge_index)
        z = self.encoder(x_or_data, edge_index)
        z_next = self.koopman(z)
        return self.decoder(z_next, edge_index)

    def _rollout(
        self,
        x_or_data: Tensor | Data,
        steps: int,
        edge_index: Tensor | None = None,
    ) -> list[Tensor]:
        """Autoregressively advance latent state and decode for multiple steps.

        Encodes the initial graph once, then applies the Koopman operator
        repeatedly in latent space, decoding after each step.

        Parameters
        ----------
        x_or_data : Tensor or Data
            Either a PyG ``Data`` object or node features ``x``.
        steps : int
            Number of rollout steps (must be >= 1).
        edge_index : Tensor, optional
            Edge index with shape ``(2, num_edges)``. Required when
            ``x_or_data`` is a tensor; ignored for ``Data`` input.

        Returns
        -------
        list of Tensor
            Decoded predictions for each step, each of shape
            ``(num_nodes, out_channels)``.

        Raises
        ------
        ValueError
            If ``steps < 1``.
        """
        if steps < 1:
            msg = f"steps must be >= 1, got {steps}"
            raise ValueError(msg)

        edge_index = self._resolve_edge_index(x_or_data, edge_index)
        z = self.encoder(x_or_data, edge_index)

        predictions: list[Tensor] = []
        for _ in range(steps):
            z = self.koopman(z)
            predictions.append(self.decoder(z, edge_index))
        return predictions

    def predict(
        self,
        initial_graph: Tensor | Data,
        steps: int,
        edge_index: Tensor | None = None,
    ) -> list[Data]:
        """Autoregressively predict future graph snapshots.

        Encodes the initial graph once, advances the latent state with the
        Koopman operator for ``steps`` iterations, and decodes after each step.
        Runs in evaluation mode without gradient tracking.

        Parameters
        ----------
        initial_graph : Tensor or Data
            Either a PyG ``Data`` object or node features ``x`` of shape
            ``(num_nodes, in_channels)``.
        steps : int
            Number of future snapshots to predict (must be >= 1).
        edge_index : Tensor, optional
            Edge index with shape ``(2, num_edges)``. Required when
            ``initial_graph`` is a tensor; ignored for ``Data`` input.

        Returns
        -------
        list of Data
            ``steps`` predicted graph snapshots sharing the input topology.
            Each ``Data.x`` has shape ``(num_nodes, out_channels)``.

        Raises
        ------
        ValueError
            If ``steps < 1``.
        """
        edge_index = self._resolve_edge_index(initial_graph, edge_index)
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                predictions = self._rollout(initial_graph, steps, edge_index)
        finally:
            self.train(was_training)

        return [Data(x=pred, edge_index=edge_index) for pred in predictions]

    def fit(
        self,
        data_sequence: GraphSnapshotSequence | Sequence[Data],
        *,
        epochs: int = 100,
        lr: float = 1e-3,
        optimizer: Callable[..., Optimizer] = torch.optim.Adam,
        device: str | torch.device | None = None,
        loss_weights: LossWeights | None = None,
        loss_weight_schedule: LossWeightSchedule | None = None,
        rollout_horizon: int | None = None,
        max_grad_norm: float | None = None,
        early_stopping_patience: int | None = None,
        early_stopping_min_delta: float = 0.0,
        **optimizer_kwargs: Any,
    ) -> FitHistory:
        """Train encoder, Koopman operator, and decoder end-to-end.

        Minimizes a weighted sum of one-step MSE and optional forward and
        backward consistency terms::

            loss = w_r * recon_loss
                 + w_f * ||K z_t - z_{t+1}||^2
                 + w_b * ||z_t - z_{t+1} K^{\\dagger}||^2

        where ``z_t`` and ``z_{t+1}`` are encoder outputs for consecutive
        snapshots and weights ``(w_r, w_f, w_b)`` come from a
        :class:`~koopman_graph.training.LossWeights` object or an optional
        per-epoch schedule.

        Parameters
        ----------
        data_sequence : GraphSnapshotSequence or sequence of Data
            Time-ordered graph snapshots with shared topology. Must contain at
            least two snapshots.
        epochs : int, optional
            Number of training epochs. Default is ``100``.
        lr : float, optional
            Learning rate passed to the optimizer. Default is ``1e-3``.
        optimizer : callable, optional
            Optimizer class. Default is :class:`torch.optim.Adam`.
        device : str, torch.device, or None, optional
            Device for training. Defaults to the model's current device, or CPU
            if the model has no parameters.
        loss_weights : LossWeights or None, optional
            Static loss weights for all epochs. When ``None`` and no schedule is
            provided, defaults to reconstruction-only training.
        loss_weight_schedule : callable or None, optional
            Callable ``epoch -> LossWeights`` applied each epoch. Overrides
            ``loss_weights`` when set.
        rollout_horizon : int or None, optional
            Number of autoregressive rollout steps used when
            ``loss_weights.rollout`` is non-zero. Defaults to
            ``num_timesteps - 1``.
        max_grad_norm : float or None, optional
            When set, clip the global gradient norm before each optimizer step.
        early_stopping_patience : int or None, optional
            Stop training when training loss fails to improve for this many
            consecutive epochs. Disabled when ``None``.
        early_stopping_min_delta : float, optional
            Minimum decrease in training loss to count as improvement. Default
            is ``0.0``.
        **optimizer_kwargs
            Additional keyword arguments forwarded to the optimizer constructor.

        Returns
        -------
        :class:`~koopman_graph.training.FitHistory`
            Per-epoch training losses and early-stop metadata.

        Raises
        ------
        ValueError
            If ``epochs < 1``, ``early_stopping_patience < 1`` when set, or
            fewer than two snapshots are provided.
        """
        if epochs < 1:
            msg = f"epochs must be >= 1, got {epochs}"
            raise ValueError(msg)
        if early_stopping_patience is not None and early_stopping_patience < 1:
            msg = (
                f"early_stopping_patience must be >= 1 when set, "
                f"got {early_stopping_patience}"
            )
            raise ValueError(msg)

        sequence = resolve_sequence(data_sequence)
        if sequence.num_timesteps < 2:
            msg = "data_sequence must contain at least 2 snapshots for training"
            raise ValueError(msg)

        train_device = resolve_device(self, device)
        self.to(train_device)
        sequence = GraphSnapshotSequence(
            [
                Data(
                    x=snapshot.x.to(train_device),
                    edge_index=snapshot.edge_index.to(train_device),
                )
                for snapshot in sequence
            ]
        )

        optim = optimizer(self.parameters(), lr=lr, **optimizer_kwargs)
        losses: list[float] = []
        best_loss = float("inf")
        epochs_without_improvement = 0
        stopped_early = False

        for epoch in range(epochs):
            epoch_weights = resolve_loss_weights_for_epoch(
                epoch,
                loss_weights=loss_weights,
                loss_weight_schedule=loss_weight_schedule,
            )
            epoch_loss = train_one_epoch(
                self,
                sequence,
                optim,
                epoch_weights,
                max_grad_norm=max_grad_norm,
                rollout_horizon=rollout_horizon,
            )
            losses.append(epoch_loss)

            if early_stopping_patience is not None:
                stop, best_loss, epochs_without_improvement = should_stop_early(
                    epoch_loss=epoch_loss,
                    best_loss=best_loss,
                    epochs_without_improvement=epochs_without_improvement,
                    patience=early_stopping_patience,
                    min_delta=early_stopping_min_delta,
                )
                if stop:
                    stopped_early = True
                    break

        return FitHistory(
            loss=losses,
            epochs=len(losses),
            stopped_early=stopped_early,
        )

    @staticmethod
    def _resolve_edge_index(
        x_or_data: Tensor | Data,
        edge_index: Tensor | None,
    ) -> Tensor:
        """Extract or validate ``edge_index`` from input arguments.

        Parameters
        ----------
        x_or_data : Tensor or Data
            Graph input; when a ``Data`` object, its ``edge_index`` is returned.
        edge_index : Tensor or None
            Explicit edge index for tensor input.

        Returns
        -------
        Tensor
            Edge index with shape ``(2, num_edges)``.

        Raises
        ------
        ValueError
            If ``x_or_data`` is a tensor and ``edge_index`` is ``None``.
        """
        if isinstance(x_or_data, Data):
            return x_or_data.edge_index
        if edge_index is None:
            msg = "edge_index is required when x_or_data is a tensor"
            raise ValueError(msg)
        return edge_index
