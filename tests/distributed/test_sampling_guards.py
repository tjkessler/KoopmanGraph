"""Coverage and error-path tests for :mod:`koopman_graph.distributed`."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor
from torch_geometric.data import Data

from koopman_graph.data import (
    GraphSnapshotSequence,
)
from koopman_graph.distributed import (
    DistributedWindowSampler,
    shard_sequences_for_rank,
)


def _path_edges(num_nodes: int = 4) -> Tensor:
    """Undirected path ``edge_index``."""
    src = list(range(num_nodes - 1)) + list(range(1, num_nodes))
    dst = list(range(1, num_nodes)) + list(range(num_nodes - 1))
    return torch.tensor([src, dst], dtype=torch.long)


def _tiny_sequence(*, num_nodes: int = 4, timesteps: int = 4) -> GraphSnapshotSequence:
    """Path-graph snapshot sequence."""
    edge_index = _path_edges(num_nodes)
    snapshots = [
        Data(x=torch.randn(num_nodes, 2), edge_index=edge_index)
        for _ in range(timesteps)
    ]
    return GraphSnapshotSequence(snapshots)


def test_distributed_sampling_validation_and_iter() -> None:
    """Sampler and shard helpers reject invalid ranks and iterate epoch 0."""
    sequence = _tiny_sequence(timesteps=5)
    with pytest.raises(ValueError, match="rank must be >= 0"):
        shard_sequences_for_rank([sequence], rank=-1, world_size=1)
    with pytest.raises(ValueError, match="world_size must be >= 1"):
        shard_sequences_for_rank([sequence], rank=0, world_size=0)
    with pytest.raises(ValueError, match="rank must be < world_size"):
        shard_sequences_for_rank([sequence], rank=2, world_size=2)
    with pytest.raises(ValueError, match="batch_size"):
        DistributedWindowSampler([sequence], window_length=2, batch_size=0)
    with pytest.raises(ValueError, match="windows_per_epoch"):
        DistributedWindowSampler(
            [sequence],
            window_length=2,
            windows_per_epoch=0,
        )
    with pytest.raises(ValueError, match="rank must be >= 0"):
        DistributedWindowSampler([sequence], window_length=2, rank=-1, world_size=1)
    with pytest.raises(ValueError, match="world_size must be >= 1"):
        DistributedWindowSampler([sequence], window_length=2, rank=0, world_size=0)
    with pytest.raises(ValueError, match="rank must be < world_size"):
        DistributedWindowSampler([sequence], window_length=2, rank=1, world_size=1)
    with pytest.raises(ValueError, match="requires seed"):
        DistributedWindowSampler(
            [sequence],
            window_length=2,
            shuffle=True,
            world_size=2,
            rank=0,
        )
    sampler = DistributedWindowSampler(
        [sequence],
        window_length=2,
        shuffle=True,
        seed=0,
        rank=0,
        world_size=2,
        windows_per_epoch=3,
    )
    with pytest.raises(ValueError, match="epoch must be >= 0"):
        sampler.rank_origin_indices(epoch=-1)
    batches = list(sampler)
    assert batches
    _ = sampler.rank_origin_indices(epoch=1)
