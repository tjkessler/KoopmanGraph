"""Coverage and error-path tests for :mod:`koopman_graph.training`."""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from torch_geometric.data import Data

from koopman_graph import (
    GNNDecoder,
    GNNEncoder,
    GraphKoopmanModel,
    GraphSnapshotSequence,
)
from koopman_graph.data import WindowSampler
from koopman_graph.training import LossWeights
from koopman_graph.training.epochs import prepare_training_amp, train_windowed_epoch


def _path_edges(num_nodes: int = 3) -> torch.Tensor:
    sources = list(range(num_nodes - 1)) + list(range(1, num_nodes))
    targets = list(range(1, num_nodes)) + list(range(num_nodes - 1))
    return torch.tensor([sources, targets], dtype=torch.long)


def _tiny_model(
    *, n_delays: int = 1, learn_topology: str | None = None
) -> GraphKoopmanModel:
    in_channels = n_delays * 2
    return GraphKoopmanModel(
        encoder=GNNEncoder(in_channels, 8, 3, num_layers=1),
        decoder=GNNDecoder(3, 8, 2, num_layers=1),
        latent_dim=3,
        time_step=1.0,
        n_delays=n_delays,
        learn_topology=learn_topology,
    )


def _sequence(num_timesteps: int = 3, *, num_nodes: int = 3) -> GraphSnapshotSequence:
    edge_index = _path_edges(num_nodes)
    return GraphSnapshotSequence(
        [
            Data(x=torch.randn(num_nodes, 2), edge_index=edge_index)
            for _ in range(num_timesteps)
        ]
    )


def test_prepare_training_amp_cuda_path_without_gpu() -> None:
    """CUDA device type enables AMP even when no GPU is present (mocked)."""
    fake_cuda = SimpleNamespace(type="cuda")
    stub_scaler = MagicMock(name="GradScaler")
    with patch("torch.amp.GradScaler", return_value=stub_scaler) as ctor:
        enabled, dtype, scaler = prepare_training_amp(
            True,
            fake_cuda,  # type: ignore[arg-type]
            amp_dtype=torch.bfloat16,
        )
    assert enabled is True
    assert dtype is torch.bfloat16
    assert scaler is stub_scaler
    ctor.assert_called_once()

    with patch("torch.amp.GradScaler") as ctor2:
        enabled2, dtype2, scaler2 = prepare_training_amp(
            True,
            fake_cuda,  # type: ignore[arg-type]
            grad_scaler=stub_scaler,
        )
    assert enabled2 is True
    assert dtype2 is torch.float16
    assert scaler2 is stub_scaler
    ctor2.assert_not_called()


def test_train_windowed_epoch_amp_autocast_branch() -> None:
    """Forced AMP path exercises the windowed autocast body."""
    model = _tiny_model()
    sequence = _sequence(4)
    sampler = WindowSampler(sequence, window_length=3, batch_size=1, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    stub_scaler = MagicMock()
    stub_scaler.scale.side_effect = lambda loss: loss
    stub_scaler.step.side_effect = lambda opt: opt.step()

    with (
        patch(
            "koopman_graph.training.epochs.prepare_training_amp",
            return_value=(True, torch.float16, stub_scaler),
        ),
        patch(
            "torch.amp.autocast",
            side_effect=lambda *args, **kwargs: contextlib.nullcontext(),
        ),
    ):
        breakdown = train_windowed_epoch(
            model,
            sampler,
            optimizer,
            LossWeights(),
            epoch=0,
        )
    assert torch.isfinite(breakdown.total)
