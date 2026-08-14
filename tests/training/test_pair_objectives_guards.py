"""Coverage and error-path tests for :mod:`koopman_graph.training`."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data

from koopman_graph import (
    GNNDecoder,
    GNNEncoder,
    GraphKoopmanModel,
    GraphSnapshotSequence,
)
from koopman_graph.training.pair_objectives import (
    _reconstruction_from_predictions,
    compute_sequence_loss,
    one_step_prediction,
)
from koopman_graph.training.timestep_encode import encode_at_timestep


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


def test_encode_at_timestep_falls_back_to_encode() -> None:
    """Models without ``encode_at`` use ``encode(snapshot)``."""
    model = _tiny_model()
    model.encode_at = None  # type: ignore[method-assign, assignment]
    sequence = _sequence(2)
    with torch.no_grad():
        z = encode_at_timestep(model, sequence, 0)
    assert z.shape == (3, 3)


def test_one_step_prediction_model_forward_fallback() -> None:
    """n_delays==1 without cache uses ``model(source, …)``."""
    model = _tiny_model()
    sequence = _sequence(2)
    with torch.no_grad():
        pred = one_step_prediction(model, sequence, 0)
    assert pred.shape == sequence[1].x.shape


def test_reconstruction_and_sequence_loss_guard_short_sequences() -> None:
    """Prediction helpers reject sequences shorter than two snapshots."""
    model = _tiny_model()
    short = _sequence(1)
    with pytest.raises(ValueError, match="at least 2 snapshots"):
        _reconstruction_from_predictions(model, short, [])
    with pytest.raises(ValueError, match="at least 2 snapshots"):
        compute_sequence_loss(model, short)


def test_reconstruction_rejects_prediction_length_mismatch() -> None:
    """``predictions`` length must equal the number of consecutive pairs."""
    model = _tiny_model()
    sequence = _sequence(3)
    with pytest.raises(ValueError, match="predictions length"):
        _reconstruction_from_predictions(model, sequence, [torch.zeros(3, 2)])


def test_sequence_loss_without_cache() -> None:
    """``cache is None`` routes through ``mean_pair_sequence_loss``."""
    model = _tiny_model()
    sequence = _sequence(3)
    loss = compute_sequence_loss(model, sequence, cache=None)
    assert torch.isfinite(loss)
