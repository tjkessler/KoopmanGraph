"""Coverage and error-path tests for :mod:`koopman_graph.model`."""

from __future__ import annotations

import torch
from torch_geometric.data import Data

from koopman_graph import (
    GNNDecoder,
    GNNEncoder,
    GraphKoopmanModel,
    GraphSnapshotSequence,
)
from koopman_graph.losses.rollout import _encode_rollout_origin_latent


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


def test_encode_rollout_origin_delay_learned_topology() -> None:
    """Delay + self-adaptive origin uses history_from_snapshots + encode."""
    model = _tiny_model(n_delays=2, learn_topology="self_adaptive")
    edge_index = _path_edges(3)
    history = [Data(x=torch.randn(3, 2), edge_index=edge_index)]
    origin = Data(x=torch.randn(3, 2), edge_index=edge_index)
    model.eval()
    with torch.no_grad():
        z, resolved_ei, resolved_ew = model.encode_rollout_origin(
            origin, history=history
        )
    assert z.shape[0] == 3
    assert resolved_ei.shape[0] == 2
    assert resolved_ew is not None


def test_rollout_encode_origin_falls_back_without_encode_at() -> None:
    """Rollout helper falls back to ``encode`` when ``encode_at`` is absent."""
    model = _tiny_model()
    model.encode_at = None  # type: ignore[method-assign, assignment]
    sequence = _sequence(2)
    with torch.no_grad():
        z = _encode_rollout_origin_latent(model, sequence, 0)
    assert z.shape == (3, 3)
