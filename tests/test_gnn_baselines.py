"""Tests for spatiotemporal GNN forecaster baselines."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data

from koopman_graph import GraphSnapshotSequence, evaluate_forecast
from koopman_graph.baselines.gnn import (
    DCRNNBaseline,
    GraphWaveNetBaseline,
    STGCNBaseline,
)
from koopman_graph.protocols import ForecastModel


def _ring_edge_index(num_nodes: int) -> torch.Tensor:
    """Build an undirected ring graph edge index."""
    sources = torch.arange(num_nodes)
    targets = (sources + 1) % num_nodes
    forward = torch.stack([sources, targets], dim=0)
    return torch.cat([forward, forward.flip(0)], dim=1)


def _diffusion_sequence(
    *,
    num_nodes: int = 4,
    timesteps: int = 16,
    seed: int = 0,
) -> GraphSnapshotSequence:
    """Generate a smooth diffusion trajectory on a ring for overfit smoke tests."""
    generator = torch.Generator().manual_seed(seed)
    edge_index = _ring_edge_index(num_nodes)
    state = torch.randn(num_nodes, 1, generator=generator)
    adjacency = torch.zeros(num_nodes, num_nodes)
    adjacency[edge_index[0], edge_index[1]] = 1.0
    degree = adjacency.sum(dim=1).clamp_min(1.0)
    normalized = adjacency / degree.unsqueeze(1)
    snapshots: list[Data] = []
    for _ in range(timesteps):
        snapshots.append(Data(x=state.clone(), edge_index=edge_index))
        state = 0.85 * state + 0.15 * (normalized @ state)
    return GraphSnapshotSequence(snapshots)


@pytest.mark.parametrize(
    ("cls", "kwargs"),
    [
        (STGCNBaseline, {"history_len": 2, "num_st_blocks": 1, "kernel_size": 2}),
        (DCRNNBaseline, {"history_len": 2, "diffusion_steps": 1}),
        (
            GraphWaveNetBaseline,
            {"history_len": 2, "num_layers": 2, "adaptive_adj": True},
        ),
    ],
)
def test_gnn_baseline_forecast_protocol_and_shapes(
    cls: type,
    kwargs: dict,
) -> None:
    """Verify ForecastModel presence, predict shapes, and evaluate_forecast."""
    sequence = _diffusion_sequence()
    model = cls(1, 8, 1, **kwargs)
    assert isinstance(model, ForecastModel)
    assert model.control_dim == 0

    model.fit(sequence, epochs=3, lr=1e-2)
    predictions = model.predict(sequence[0], steps=3)
    assert len(predictions) == 3
    for prediction in predictions:
        assert prediction.x.shape == (sequence.num_nodes, 1)
        assert torch.equal(prediction.edge_index, sequence.edge_index)

    result = evaluate_forecast(model, sequence, horizons=(1, 2))
    assert result.num_origins > 0
    assert len(result.horizons) == 2


@pytest.mark.parametrize(
    ("cls", "kwargs"),
    [
        (STGCNBaseline, {"history_len": 2, "num_st_blocks": 1, "kernel_size": 2}),
        (DCRNNBaseline, {"history_len": 1, "diffusion_steps": 1}),
        (
            GraphWaveNetBaseline,
            {"history_len": 2, "num_layers": 2, "adaptive_adj": False},
        ),
    ],
)
def test_gnn_baseline_overfits_small_synthetic(
    cls: type,
    kwargs: dict,
) -> None:
    """Smoke-check that training loss drops on a tiny synthetic sequence."""
    sequence = _diffusion_sequence(timesteps=12, seed=1)
    model = cls(1, 16, 1, **kwargs)

    features = torch.stack([snapshot.x for snapshot in sequence])
    edge_index = sequence.edge_index
    history_len = model.history_len

    def _mean_train_loss() -> float:
        model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for start in range(0, sequence.num_timesteps - history_len):
                history = features[start : start + history_len]
                target = features[start + history_len]
                prediction = model.predict_next(history, edge_index, None)
                total += float(torch.nn.functional.mse_loss(prediction, target))
                count += 1
        return total / count

    before = _mean_train_loss()
    model.fit(sequence, epochs=40, lr=5e-3)
    after = _mean_train_loss()
    assert after < before
    assert after < 0.15


@pytest.mark.parametrize("cls", [STGCNBaseline, DCRNNBaseline, GraphWaveNetBaseline])
def test_gnn_baseline_spectrum_raises(cls: type) -> None:
    """Verify spectrum is intentionally unsupported for nonlinear GNN baselines."""
    sequence = _diffusion_sequence(timesteps=6)
    model = cls(
        1,
        4,
        1,
        history_len=1,
        **(
            {"num_st_blocks": 1, "kernel_size": 1}
            if cls is STGCNBaseline
            else {"num_layers": 1, "adaptive_adj": False}
            if cls is GraphWaveNetBaseline
            else {"diffusion_steps": 1}
        ),
    )
    model.fit(sequence, epochs=1, lr=1e-2)
    with pytest.raises(RuntimeError, match="no linear Koopman operator spectrum"):
        model.spectrum()


def test_gnn_baseline_rejects_controls_and_unfitted_predict() -> None:
    """Verify control rejection and unfitted predict guard."""
    sequence = _diffusion_sequence(timesteps=6)
    model = DCRNNBaseline(1, 4, 1, history_len=1, diffusion_steps=1)
    with pytest.raises(RuntimeError, match="must be fit"):
        model.predict(sequence[0], steps=1)

    model.fit(sequence, epochs=1, lr=1e-2)
    with pytest.raises(ValueError, match="does not support controls"):
        model.predict(sequence[0], steps=1, controls=torch.zeros(1, 1))
