"""Coverage and error-path tests for :mod:`koopman_graph.graph_utils`."""

from __future__ import annotations

import pytest
import torch

from koopman_graph.graph_utils.topology import random_walk_normalized_adjacency_matvec


def _path_edges(num_nodes: int = 3) -> torch.Tensor:
    sources = list(range(num_nodes - 1)) + list(range(1, num_nodes))
    targets = list(range(1, num_nodes)) + list(range(num_nodes - 1))
    return torch.tensor([sources, targets], dtype=torch.long)


def test_topology_matvec_validation() -> None:
    """Row-walk matvec rejects bad feature rank and direction."""
    edge_index = _path_edges(3)
    with pytest.raises(ValueError, match="x must be 2D"):
        random_walk_normalized_adjacency_matvec(edge_index, torch.randn(3))
    with pytest.raises(ValueError, match="direction must be"):
        random_walk_normalized_adjacency_matvec(
            edge_index,
            torch.randn(3, 2),
            direction="sideways",  # type: ignore[arg-type]
        )
