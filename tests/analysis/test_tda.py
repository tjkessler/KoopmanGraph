"""Coverage and error-path tests for :mod:`koopman_graph.analysis`."""

from __future__ import annotations

import pytest
import torch

from koopman_graph.analysis import (
    betti_curve,
    persistence_diagram_0d,
)
from koopman_graph.analysis.tda import pairwise_distance_filtration
from koopman_graph.losses import PersistenceRegularizer


def test_tda_empty_pairs_cycle_edge_and_betti() -> None:
    """Union-find covers 1-D input, cycles, empty diagrams, and empty Betti."""
    with pytest.raises(ValueError, match="shape \\(n, dim\\)"):
        pairwise_distance_filtration(torch.randn(4))
    singleton = persistence_diagram_0d(torch.zeros(1, 2))
    assert singleton.pairs.numel() == 0
    curve = betti_curve(singleton, torch.tensor([0.0, 1.0]))
    assert torch.equal(curve, torch.tensor([1.0, 1.0]))
    clustered = torch.tensor(
        [[0.0, 0.0], [0.01, 0.0], [0.0, 0.01], [5.0, 5.0]],
        dtype=torch.float32,
    )
    diagram = persistence_diagram_0d(clustered)
    assert diagram.betti_0 == 4
    assert diagram.pairs.shape[0] == 3


def test_tda_persistence_and_regularizer() -> None:
    """0-d diagrams and the persistence regularizer run without extras."""
    points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    diagram = persistence_diagram_0d(points)
    assert diagram.betti_0 == 3
    assert diagram.pairs.shape[0] == 2
    curve = betti_curve(diagram, torch.tensor([0.0, 10.0]))
    assert curve.shape == (2,)
    loss = PersistenceRegularizer(weight=0.5)(points)
    assert loss.ndim == 0
