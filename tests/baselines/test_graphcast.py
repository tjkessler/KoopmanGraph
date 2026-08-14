"""Coverage and error-path tests for :mod:`koopman_graph.baselines`."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor
from torch_geometric.data import Data

from koopman_graph.baselines.gnn.graphcast import GraphCastBaseline
from koopman_graph.data import (
    GraphSnapshotSequence,
)


def _path_edges(num_nodes: int = 4) -> Tensor:
    """Undirected path ``edge_index``."""
    src = list(range(num_nodes - 1)) + list(range(1, num_nodes))
    dst = list(range(1, num_nodes)) + list(range(num_nodes - 1))
    return torch.tensor([src, dst], dtype=torch.long)


def test_graphcast_edge_weight_and_predict_validation() -> None:
    """GraphCast fit honors ``edge_weight`` and validates history rank."""
    edge_index = _path_edges(4)
    weight = torch.ones(edge_index.shape[1])
    snapshots = [
        Data(x=torch.randn(4, 1), edge_index=edge_index, edge_weight=weight)
        for _ in range(4)
    ]
    sequence = GraphSnapshotSequence(snapshots)
    model = GraphCastBaseline(1, 4, 1, history_len=2, num_processor_layers=1)
    model.fit(sequence, epochs=1, lr=1e-2)
    history = torch.stack([snap.x for snap in sequence[:2]])
    with pytest.raises(ValueError, match="history must have shape"):
        model.predict_next(history[0], edge_index)
    with pytest.raises(ValueError, match="history_len mismatch"):
        model.predict_next(torch.randn(3, 4, 1), edge_index)
    with pytest.raises(ValueError, match="in_channels mismatch"):
        model.predict_next(torch.randn(2, 4, 3), edge_index)
