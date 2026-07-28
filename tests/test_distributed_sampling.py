"""Tests for window-origin planning and ``DistributedWindowSampler``."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data

import koopman_graph
from koopman_graph.data import (
    GraphSnapshotSequence,
    WindowOrigin,
    WindowSampler,
    build_window_index_list,
)
from koopman_graph.distributed import (
    DistributedWindowSampler,
    shard_sequences_for_rank,
)


def _feature_sequence(
    timesteps: int,
    *,
    edge_index: torch.Tensor,
) -> GraphSnapshotSequence:
    """Build a short sequence with timestep-valued node features."""
    num_nodes = int(edge_index.max().item() + 1)
    snapshots = [
        Data(
            x=torch.full((num_nodes, 1), float(t)),
            edge_index=edge_index,
            num_nodes=num_nodes,
        )
        for t in range(timesteps)
    ]
    return GraphSnapshotSequence(snapshots)


@pytest.fixture
def short_sequence(synthetic_edge_index: torch.Tensor) -> GraphSnapshotSequence:
    """Five-step path-graph sequence for window indexing tests."""
    return _feature_sequence(5, edge_index=synthetic_edge_index)


def test_build_window_index_list_enumerates_origins(
    short_sequence: GraphSnapshotSequence,
) -> None:
    """Pure planner lists every valid ``(sequence_index, start)``."""
    origins = build_window_index_list([short_sequence], window_length=3)
    assert origins == [
        WindowOrigin(0, 0),
        WindowOrigin(0, 1),
        WindowOrigin(0, 2),
    ]


def test_build_window_index_list_pools_trajectories(
    short_sequence: GraphSnapshotSequence,
) -> None:
    """Origins are pooled across trajectories in order."""
    origins = build_window_index_list(
        [short_sequence, short_sequence],
        window_length=4,
    )
    assert len(origins) == 4
    assert origins[0] == WindowOrigin(0, 0)
    assert origins[2] == WindowOrigin(1, 0)


def test_distributed_shards_are_disjoint_and_cover_global_cap(
    short_sequence: GraphSnapshotSequence,
) -> None:
    """Global ``windows_per_epoch`` then ``rank::world_size`` partitions."""
    rank0 = DistributedWindowSampler(
        short_sequence,
        window_length=2,
        batch_size=8,
        windows_per_epoch=5,
        shuffle=False,
        seed=0,
        rank=0,
        world_size=2,
    )
    rank1 = DistributedWindowSampler(
        short_sequence,
        window_length=2,
        batch_size=8,
        windows_per_epoch=5,
        shuffle=False,
        seed=0,
        rank=1,
        world_size=2,
    )
    idx0 = rank0.rank_origin_indices(0)
    idx1 = rank1.rank_origin_indices(0)
    assert set(idx0).isdisjoint(idx1)
    assert sorted(idx0 + idx1) == list(range(4))


def test_windows_per_epoch_global_then_shard(
    short_sequence: GraphSnapshotSequence,
) -> None:
    """Cap applies globally before sharding (3 windows → ranks get 2 and 1)."""
    rank0 = DistributedWindowSampler(
        short_sequence,
        window_length=2,
        shuffle=False,
        seed=1,
        windows_per_epoch=3,
        rank=0,
        world_size=2,
    )
    rank1 = DistributedWindowSampler(
        short_sequence,
        window_length=2,
        shuffle=False,
        seed=1,
        windows_per_epoch=3,
        rank=1,
        world_size=2,
    )
    assert len(rank0.rank_origin_indices(0)) == 2
    assert len(rank1.rank_origin_indices(0)) == 1
    assert sorted(rank0.rank_origin_indices(0) + rank1.rank_origin_indices(0)) == [
        0,
        1,
        2,
    ]


def test_seeded_shuffle_reproducible_per_rank(
    short_sequence: GraphSnapshotSequence,
) -> None:
    """Same seed and rank yields identical shards; ranks differ."""
    kwargs: dict[str, object] = {
        "window_length": 2,
        "batch_size": 1,
        "shuffle": True,
        "seed": 11,
        "world_size": 2,
    }
    a = DistributedWindowSampler(short_sequence, rank=0, **kwargs)  # type: ignore[arg-type]
    b = DistributedWindowSampler(short_sequence, rank=0, **kwargs)  # type: ignore[arg-type]
    other = DistributedWindowSampler(short_sequence, rank=1, **kwargs)  # type: ignore[arg-type]
    assert a.rank_origin_indices(0) == b.rank_origin_indices(0)
    assert a.rank_origin_indices(0) != other.rank_origin_indices(0)


def test_world_size_one_matches_window_sampler_slices(
    short_sequence: GraphSnapshotSequence,
) -> None:
    """World size 1 reproduces ``WindowSampler`` window starts (no shuffle)."""
    baseline = WindowSampler(
        short_sequence,
        window_length=3,
        batch_size=2,
        shuffle=False,
    )
    distributed = DistributedWindowSampler(
        short_sequence,
        window_length=3,
        batch_size=2,
        shuffle=False,
        seed=0,
        rank=0,
        world_size=1,
    )
    base_batches = list(baseline.iter_epoch(0))
    dist_batches = list(distributed.iter_epoch(0))
    assert len(base_batches) == len(dist_batches)
    for left_batch, right_batch in zip(base_batches, dist_batches, strict=True):
        assert len(left_batch) == len(right_batch)
        for left, right in zip(left_batch, right_batch, strict=True):
            assert torch.equal(left[0].x, right[0].x)


def test_shuffle_requires_seed_when_world_size_gt_one(
    short_sequence: GraphSnapshotSequence,
) -> None:
    """Unseeded shuffle is rejected for multi-rank jobs."""
    with pytest.raises(ValueError, match="requires seed"):
        DistributedWindowSampler(
            short_sequence,
            window_length=2,
            shuffle=True,
            seed=None,
            rank=0,
            world_size=2,
        )


def test_distributed_sampler_not_on_root_all() -> None:
    """Distributed sampler stays off root ``__all__``."""
    assert "DistributedWindowSampler" not in koopman_graph.__all__
    assert "build_window_index_list" not in koopman_graph.__all__
    assert "shard_sequences_for_rank" not in koopman_graph.__all__


def test_shard_sequences_world_size_one(
    short_sequence: GraphSnapshotSequence,
) -> None:
    """Single trajectory with world size 1 returns that trajectory."""
    shard = shard_sequences_for_rank(short_sequence, rank=0, world_size=1)
    assert len(shard) == 1
    assert shard[0] is short_sequence


def test_shard_sequences_disjoint_cover(
    short_sequence: GraphSnapshotSequence,
) -> None:
    """Four trajectories split evenly across two ranks."""
    trajectories = [short_sequence, short_sequence, short_sequence, short_sequence]
    rank0 = shard_sequences_for_rank(trajectories, rank=0, world_size=2)
    rank1 = shard_sequences_for_rank(trajectories, rank=1, world_size=2)
    assert len(rank0) == 2
    assert len(rank1) == 2
    assert rank0[0] is trajectories[0]
    assert rank0[1] is trajectories[2]
    assert rank1[0] is trajectories[1]
    assert rank1[1] is trajectories[3]


def test_shard_sequences_raises_when_fewer_than_world_size(
    short_sequence: GraphSnapshotSequence,
) -> None:
    """Undersized trajectory lists raise with windowed-mode guidance."""
    with pytest.raises(ValueError, match="DistributedWindowSampler"):
        shard_sequences_for_rank(
            [short_sequence],
            rank=0,
            world_size=2,
        )


def test_shard_sequences_rejects_empty() -> None:
    """Empty trajectory lists are rejected."""
    with pytest.raises(ValueError, match="at least one trajectory"):
        shard_sequences_for_rank([], rank=0, world_size=1)
