"""Coverage and error-path tests for :mod:`koopman_graph.datasets`."""

from __future__ import annotations

import torch
from torch_geometric.data import Data

from koopman_graph.datasets import era5_slice_card, generate_tiny_era5_mesh
from koopman_graph.datasets.molecular import alanine_dipeptide_card
from koopman_graph.nn import FourierNeuralOperatorEncoder
from koopman_graph.operators import (
    sample_graphon_adjacency,
)


def _path_edges(num_nodes: int = 4) -> torch.Tensor:
    """Undirected path ``edge_index``."""
    src = list(range(num_nodes - 1)) + list(range(1, num_nodes))
    dst = list(range(1, num_nodes)) + list(range(num_nodes - 1))
    return torch.tensor([src, dst], dtype=torch.long)


def test_fno_era5_alanine_graphon() -> None:
    """Neural-operator encoder, ERA5 mesh, alanine card, graphon sample."""
    encoder = FourierNeuralOperatorEncoder(2, 4, 3, n_modes=2)
    data = Data(x=torch.randn(8, 2), edge_index=_path_edges(8))
    lifted = encoder(data)
    assert lifted.shape == (8, 3)
    card = era5_slice_card()
    assert card.num_nodes == 8
    mesh = generate_tiny_era5_mesh(num_nodes=6, num_timesteps=4, seed=1)
    assert mesh.num_timesteps == 4
    ala = alanine_dipeptide_card()
    assert ala["n_atoms"] == 22
    edges = sample_graphon_adjacency(6, kernel="constant", density=0.5)
    assert edges.shape[0] == 2
