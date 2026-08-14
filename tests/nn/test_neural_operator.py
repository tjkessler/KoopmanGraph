"""Coverage and error-path tests for :mod:`koopman_graph.nn`."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor
from torch_geometric.data import Data

from koopman_graph.nn import (
    FourierNeuralOperatorEncoder,
)


def _path_edges(num_nodes: int = 4) -> Tensor:
    """Undirected path ``edge_index``."""
    src = list(range(num_nodes - 1)) + list(range(1, num_nodes))
    dst = list(range(1, num_nodes)) + list(range(num_nodes - 1))
    return torch.tensor([src, dst], dtype=torch.long)


def test_fno_requires_features() -> None:
    """Fourier lifting refuses snapshots without ``x``."""
    encoder = FourierNeuralOperatorEncoder(2, 4, 3, n_modes=2)
    with pytest.raises(ValueError, match="requires Data.x"):
        encoder(Data(edge_index=_path_edges(4)))
