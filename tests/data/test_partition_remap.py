"""Coverage and error-path tests for :mod:`koopman_graph.data`."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor
from torch_geometric.data import Data

from koopman_graph.data import (
    cluster_node_partition,
    induced_cluster_subgraph,
    iter_cluster_subgraphs,
    remap_node_features,
)


def _path_edges(num_nodes: int = 4) -> Tensor:
    """Undirected path ``edge_index``."""
    src = list(range(num_nodes - 1)) + list(range(1, num_nodes))
    dst = list(range(1, num_nodes)) + list(range(num_nodes - 1))
    return torch.tensor([src, dst], dtype=torch.long)


def _path_edges_v014_remaining(num_nodes: int = 4) -> torch.Tensor:
    """Undirected path ``edge_index``."""
    src = list(range(num_nodes - 1)) + list(range(1, num_nodes))
    dst = list(range(1, num_nodes)) + list(range(num_nodes - 1))
    return torch.tensor([src, dst], dtype=torch.long)


def test_partition_and_remap_error_paths() -> None:
    """Cluster partitions and open-world remap reject invalid arguments."""
    with pytest.raises(ValueError, match="num_clusters"):
        cluster_node_partition(4, 0)
    with pytest.raises(ValueError, match="num_nodes"):
        cluster_node_partition(0, 2)
    data = Data(edge_index=_path_edges(3))
    assignment = torch.zeros(3, dtype=torch.long)
    with pytest.raises(ValueError, match="requires Data.x"):
        induced_cluster_subgraph(data, 0, assignment)
    filled = Data(x=torch.randn(3, 2), edge_index=_path_edges(3))
    with pytest.raises(ValueError, match="empty"):
        induced_cluster_subgraph(filled, 9, assignment)
    with pytest.raises(ValueError, match="requires Data.x"):
        list(iter_cluster_subgraphs(data, 2))
    generator = torch.Generator().manual_seed(0)
    subgraphs = list(iter_cluster_subgraphs(filled, 8, generator=generator))
    assert subgraphs
    features = torch.randn(3, 2)
    with pytest.raises(ValueError, match="shape \\(N, F\\)"):
        remap_node_features(torch.randn(3), old_index=torch.arange(3), new_capacity=4)
    with pytest.raises(ValueError, match="1-D"):
        remap_node_features(features, old_index=torch.arange(2), new_capacity=4)
    with pytest.raises(ValueError, match="at least N_old"):
        remap_node_features(features, old_index=torch.arange(3), new_capacity=2)
    with pytest.raises(ValueError, match="outside"):
        remap_node_features(features, old_index=torch.tensor([0, 1, 9]), new_capacity=4)


def test_remap_injective_and_merge_reject() -> None:
    """Open-world remap requires an injective index map."""
    features = torch.arange(6, dtype=torch.float32).view(3, 2)
    mapped = remap_node_features(
        features, old_index=torch.tensor([0, 2, 5]), new_capacity=6
    )
    assert mapped.shape == (6, 2)
    assert torch.equal(mapped[2], features[1])
    with pytest.raises(ValueError, match="injective"):
        remap_node_features(features, old_index=torch.tensor([0, 0, 1]), new_capacity=4)


def test_cluster_partition_and_iter() -> None:
    """ClusterGCN-style partitions yield induced subgraphs."""
    data = Data(x=torch.randn(6, 2), edge_index=_path_edges_v014_remaining(6))
    gen = torch.Generator().manual_seed(0)
    assignment = cluster_node_partition(6, 2, generator=gen)
    assert assignment.shape == (6,)
    subgraph = induced_cluster_subgraph(data, int(assignment[0].item()), assignment)
    assert subgraph.x is not None
    graphs = list(
        iter_cluster_subgraphs(data, 2, generator=torch.Generator().manual_seed(1))
    )
    assert graphs


def test_remap_refuses_silent_merge() -> None:
    """Open-world remap refuses colliding destination indices."""
    features = torch.randn(3, 2)
    with pytest.raises(ValueError, match="injective"):
        remap_node_features(
            features,
            old_index=torch.tensor([0, 0, 1]),
            new_capacity=4,
        )
