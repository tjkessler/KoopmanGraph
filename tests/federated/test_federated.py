"""Coverage and error-path tests for :mod:`koopman_graph.federated`."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor
from torch_geometric.data import Data

from koopman_graph.federated import federated_average
from koopman_graph.robustness import corrupt_node_features


def _path_edges(num_nodes: int = 4) -> Tensor:
    """Undirected path ``edge_index``."""
    src = list(range(num_nodes - 1)) + list(range(1, num_nodes))
    dst = list(range(1, num_nodes)) + list(range(num_nodes - 1))
    return torch.tensor([src, dst], dtype=torch.long)


def test_federated_mismatched_keys_and_robustness() -> None:
    """FedAvg and FDI helpers reject mismatched keys / missing features."""
    with pytest.raises(ValueError, match="identical keys"):
        federated_average([{"a": torch.ones(2)}, {"b": torch.zeros(2)}])
    with pytest.raises(ValueError, match="requires Data.x"):
        corrupt_node_features(Data(edge_index=_path_edges(3)), magnitude=0.1)
    with pytest.raises(ValueError, match="non-negative"):
        corrupt_node_features(
            Data(x=torch.randn(3, 2), edge_index=_path_edges(3)),
            magnitude=-0.1,
        )


def test_federated_average_rejects_empty() -> None:
    """FedAvg refuses an empty client list."""
    with pytest.raises(ValueError, match="non-empty"):
        federated_average([])
