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
from koopman_graph.training.extra_objectives import (
    compute_worst_case_reconstruction_loss,
)


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


def test_worst_case_predictions_guard_short_sequence() -> None:
    """Worst-case path with precomputed predictions rejects short sequences."""
    model = _tiny_model()
    short = _sequence(1)
    with pytest.raises(ValueError, match="at least 2 snapshots"):
        compute_worst_case_reconstruction_loss(model, short, weight=1.0, predictions=[])
