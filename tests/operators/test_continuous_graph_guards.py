"""Coverage and error-path tests for :mod:`koopman_graph.operators`."""

from __future__ import annotations

import torch

from koopman_graph.operators.continuous_graph import ContinuousGraphKoopmanOperator


def _path_edges(num_nodes: int = 3) -> torch.Tensor:
    sources = list(range(num_nodes - 1)) + list(range(1, num_nodes))
    targets = list(range(1, num_nodes)) + list(range(num_nodes - 1))
    return torch.tensor([sources, targets], dtype=torch.long)


def test_continuous_graph_topology_payload_equal_mismatch_branches() -> None:
    """``_topology_payload_equal`` false paths for index/weight mismatches."""
    op = ContinuousGraphKoopmanOperator(latent_dim=2)
    edges = _path_edges(3)
    other = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    weights = torch.ones(edges.shape[1])
    assert op._topology_payload_equal(edges, None, other, None) is False
    assert op._topology_payload_equal(edges, weights, edges, None) is False
    assert op._topology_payload_equal(edges, None, edges, None) is True
    assert op._topology_payload_equal(edges, weights, edges, 2.0 * weights) is False
