"""Coverage and error-path tests for :mod:`koopman_graph.analysis`."""

from __future__ import annotations

import torch
from torch_geometric.data import Data

from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
from koopman_graph.data import (
    GraphSnapshotSequence,
)


def _path_edges(num_nodes: int = 4) -> torch.Tensor:
    """Undirected path ``edge_index``."""
    src = list(range(num_nodes - 1)) + list(range(1, num_nodes))
    dst = list(range(1, num_nodes)) + list(range(num_nodes - 1))
    return torch.tensor([src, dst], dtype=torch.long)


def _tiny_model(*, koopman: str = "pernode", parameterization: str = "dense", **kwargs):
    """Build a tiny homogeneous GCN model."""
    return GraphKoopmanModel(
        encoder=GNNEncoder(2, 4, 4, num_layers=1),
        decoder=GNNDecoder(4, 4, 2, num_layers=1),
        latent_dim=4,
        time_step=0.1,
        koopman=koopman,
        koopman_parameterization=parameterization,
        **kwargs,
    )


def _tiny_sequence(*, num_nodes: int = 4, timesteps: int = 6) -> GraphSnapshotSequence:
    """Path-graph snapshot sequence."""
    edge_index = _path_edges(num_nodes)
    snapshots = [
        Data(x=torch.randn(num_nodes, 2), edge_index=edge_index)
        for _ in range(timesteps)
    ]
    return GraphSnapshotSequence(snapshots)


def test_evaluate_include_resdmd() -> None:
    """``evaluate(..., include_resdmd=True)`` attaches a finite-dictionary report."""
    model = _tiny_model()
    sequence = _tiny_sequence()
    result = model.evaluate(sequence, horizons=(1,), include_resdmd=True)
    assert result.resdmd is not None
