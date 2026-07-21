"""Tests for spatiotemporal GNN forecaster baselines."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data

from koopman_graph import GraphSnapshotSequence
from koopman_graph.baselines.gnn import (
    DCRNNBaseline,
    GraphWaveNetBaseline,
    STGCNBaseline,
)
from koopman_graph.metrics import evaluate_forecast
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


def test_gnn_forecaster_base_helpers_and_validation() -> None:
    """Cover dense adjacency helpers and fit/predict validation branches."""
    from koopman_graph.baselines.gnn.base import dense_adjacency, random_walk_normalize

    edge_index = _ring_edge_index(4)
    adj = dense_adjacency(edge_index, None, 4)
    assert adj.shape == (4, 4)
    weighted = dense_adjacency(
        edge_index,
        torch.ones(edge_index.shape[1]) * 2.0,
        4,
    )
    assert torch.allclose(
        weighted[edge_index[0], edge_index[1]],
        torch.full((edge_index.shape[1],), 2.0),
    )
    normalized = random_walk_normalize(adj)
    assert torch.allclose(normalized.sum(dim=1), torch.ones(4), atol=1e-5)

    with pytest.raises(ValueError, match="in_channels"):
        STGCNBaseline(0, 4, 1)
    with pytest.raises(ValueError, match="time_step"):
        STGCNBaseline(1, 4, 1, time_step=0.0)

    sequence = _diffusion_sequence(timesteps=6)
    model = DCRNNBaseline(1, 4, 1, history_len=2, diffusion_steps=1)
    with pytest.raises(ValueError, match="at least"):
        model.fit(_diffusion_sequence(timesteps=2), epochs=1)
    with pytest.raises(ValueError, match="in_channels"):
        DCRNNBaseline(2, 4, 2, history_len=1, diffusion_steps=1).fit(sequence, epochs=1)
    with pytest.raises(ValueError, match="epochs"):
        model.fit(sequence, epochs=0)
    with pytest.raises(ValueError, match="lr"):
        model.fit(sequence, epochs=1, lr=0.0)

    mismatched = DCRNNBaseline(1, 4, 2, history_len=1, diffusion_steps=1)
    with pytest.raises(ValueError, match="out_channels must equal"):
        mismatched.fit(sequence, epochs=1)

    model.fit(sequence, epochs=2, lr=1e-2, batch_size=2)
    with pytest.raises(ValueError, match="steps must be"):
        model.predict(sequence[0], steps=0)
    with pytest.raises(ValueError, match="future_topologies"):
        model.predict(sequence[0], steps=1, future_topologies=[sequence[0]])

    # Fit path with edge weights present.
    weighted_snaps = [
        Data(
            x=snap.x.clone(),
            edge_index=snap.edge_index.clone(),
            edge_weight=torch.ones(snap.edge_index.shape[1]),
        )
        for snap in sequence
    ]
    weighted = GraphSnapshotSequence(weighted_snaps)
    weighted_model = DCRNNBaseline(1, 4, 1, history_len=1, diffusion_steps=1)
    weighted_model.fit(weighted, epochs=1, lr=1e-2)
    preds = weighted_model.predict(weighted[0], steps=1)
    assert len(preds) == 1


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
def test_gnn_baseline_batched_predict_next_matches_single(
    cls: type,
    kwargs: dict,
) -> None:
    """Batched predict_next must match stacked single-window calls."""
    sequence = _diffusion_sequence(timesteps=8, seed=2)
    model = cls(1, 8, 1, **kwargs)
    model.fit(sequence, epochs=1, lr=1e-2)
    features = torch.stack([snapshot.x for snapshot in sequence])
    h0 = features[0:2]
    h1 = features[1:3]
    edge_index = sequence.edge_index
    single = torch.stack(
        [
            model.predict_next(h0, edge_index, None),
            model.predict_next(h1, edge_index, None),
        ]
    )
    batched = model.predict_next(torch.stack([h0, h1]), edge_index, None)
    assert batched.shape == single.shape
    assert torch.allclose(batched, single, atol=1e-5, rtol=1e-5)


def test_gnn_baseline_constructor_and_rank_validation() -> None:
    """Cover constructor guards and invalid history ranks for GNN baselines."""
    from koopman_graph.baselines.gnn.dcrnn import _DiffusionConv

    with pytest.raises(ValueError, match="diffusion_steps"):
        DCRNNBaseline(1, 4, 1, diffusion_steps=0)
    with pytest.raises(ValueError, match="num_st_blocks"):
        STGCNBaseline(1, 4, 1, num_st_blocks=0)
    with pytest.raises(ValueError, match="kernel_size"):
        STGCNBaseline(1, 4, 1, kernel_size=0)
    with pytest.raises(ValueError, match="num_layers"):
        GraphWaveNetBaseline(1, 4, 1, num_layers=0)

    sequence = _diffusion_sequence(timesteps=6)
    edge_index = sequence.edge_index
    bad_history = torch.randn(4, 1)  # rank-2, not 3 or 4

    dcrnn = DCRNNBaseline(1, 4, 1, history_len=1, diffusion_steps=1)
    dcrnn.fit(sequence, epochs=1, lr=1e-2)
    with pytest.raises(ValueError, match="history must have shape"):
        dcrnn.predict_next(bad_history, edge_index, None)

    # Mismatched in/out channels forces the decoder zero-input branch.
    mismatched = DCRNNBaseline(1, 4, 2, history_len=1, diffusion_steps=1)
    history = torch.stack([sequence[0].x])
    # Bypass fit validation by calling predict_next after manual support cache.
    mismatched._cached_supports = None
    out = mismatched.predict_next(history, edge_index, None)
    assert out.shape == (sequence.num_nodes, 2)

    stgcn = STGCNBaseline(1, 4, 1, history_len=2, num_st_blocks=1, kernel_size=2)
    stgcn.fit(sequence, epochs=1, lr=1e-2)
    with pytest.raises(ValueError, match="history must have shape"):
        stgcn.predict_next(bad_history, edge_index, None)

    wavenet = GraphWaveNetBaseline(
        1, 4, 1, history_len=2, num_layers=1, adaptive_adj=False
    )
    wavenet.fit(sequence, epochs=1, lr=1e-2)
    with pytest.raises(ValueError, match="history must have shape"):
        wavenet.predict_next(bad_history, edge_index, None)

    # Weighted WaveNet fit exercises the edge_weight.to(device) branch.
    weighted_snaps = [
        Data(
            x=snap.x.clone(),
            edge_index=snap.edge_index.clone(),
            edge_weight=torch.ones(snap.edge_index.shape[1]),
        )
        for snap in sequence
    ]
    GraphWaveNetBaseline(1, 4, 1, history_len=2, num_layers=1, adaptive_adj=False).fit(
        GraphSnapshotSequence(weighted_snaps), epochs=1, lr=1e-2
    )

    # Diffusion conv 2-D path and invalid rank.
    conv = _DiffusionConv(1, 2, diffusion_steps=1)
    support = torch.eye(4)
    supports = [support, support, support]
    y2d = conv(torch.randn(4, 1), supports)
    assert y2d.shape == (4, 2)
    with pytest.raises(ValueError, match="x must have shape"):
        conv(torch.randn(4), supports)
