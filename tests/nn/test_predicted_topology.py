"""Coverage and error-path tests for :mod:`koopman_graph.nn`."""

from __future__ import annotations

import pytest
import torch

from koopman_graph.nn import (
    PredictedTopologyHead,
)


def test_predicted_topology_validation_empty_edges_and_forward() -> None:
    """Topology head covers invalid inputs, empty thresholding, and ``forward``."""
    with pytest.raises(ValueError, match="latent_dim"):
        PredictedTopologyHead(0)
    head = PredictedTopologyHead(3, hidden_dim=8)
    with pytest.raises(ValueError, match="shape \\(N, d\\)"):
        head.pairwise_logits(torch.randn(3))
    z = torch.randn(4, 3)
    empty = head.edge_index(z, threshold=1e9)
    assert empty.shape == (2, 4)
    assert head(z).shape == (4, 4)


def test_predicted_topology_head_logits() -> None:
    """Predicted topology emits pairwise logits and a COO index."""
    head = PredictedTopologyHead(4, hidden_dim=8)
    z = torch.randn(5, 4)
    logits = head.pairwise_logits(z)
    assert logits.shape == (5, 5)
    edges = head.edge_index(z, top_k=2)
    assert edges.shape[0] == 2
    assert edges.shape[1] == 10
