"""Tests for window-origin planning and ``DistributedWindowSampler``."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data, HeteroData

import koopman_graph
from koopman_graph.data import (
    GraphSnapshotSequence,
    HeteroGraphSnapshotSequence,
    MultiTrajectory,
    NeighborWindowSampler,
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


def _multiplex_snapshot(*, seed: int = 0, timesteps_mark: float = 0.0) -> HeteroData:
    """Build a multiplex snapshot with marked node features."""
    del seed  # features are deterministic from timesteps_mark
    data = HeteroData()
    data["node"].x = torch.full((4, 2), float(timesteps_mark))
    data["node", "r1", "node"].edge_index = torch.tensor(
        [[0, 1, 2], [1, 2, 0]],
        dtype=torch.long,
    )
    data["node", "r2", "node"].edge_index = torch.tensor(
        [[0, 2], [2, 3]],
        dtype=torch.long,
    )
    return data


def _hetero_feature_sequence(timesteps: int = 5) -> HeteroGraphSnapshotSequence:
    """Build a multiplex sequence with timestep-valued node features."""
    snapshots = [_multiplex_snapshot(timesteps_mark=float(t)) for t in range(timesteps)]
    return HeteroGraphSnapshotSequence(snapshots)


def test_as_trajectory_list_keeps_single_hetero_sequence() -> None:
    """A lone hetero sequence is one trajectory, not a list of HeteroData."""
    from koopman_graph.data.sampling import as_trajectory_list

    sequence = _hetero_feature_sequence(3)
    trajectories = as_trajectory_list(sequence)
    assert len(trajectories) == 1
    assert trajectories[0] is sequence


def test_hetero_distributed_shards_are_disjoint() -> None:
    """Hetero window shards are disjoint and cover the global capped set."""
    sequence = _hetero_feature_sequence(5)
    rank0 = DistributedWindowSampler(
        sequence,
        window_length=2,
        batch_size=8,
        windows_per_epoch=5,
        shuffle=False,
        seed=0,
        rank=0,
        world_size=2,
    )
    rank1 = DistributedWindowSampler(
        sequence,
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


def test_hetero_seeded_shuffle_reproducible_per_rank() -> None:
    """Hetero seeded shuffle is reproducible per rank and differs across ranks."""
    sequence = _hetero_feature_sequence(5)
    kwargs: dict[str, object] = {
        "window_length": 2,
        "batch_size": 1,
        "shuffle": True,
        "seed": 11,
        "world_size": 2,
    }
    a = DistributedWindowSampler(sequence, rank=0, **kwargs)  # type: ignore[arg-type]
    b = DistributedWindowSampler(sequence, rank=0, **kwargs)  # type: ignore[arg-type]
    other = DistributedWindowSampler(sequence, rank=1, **kwargs)  # type: ignore[arg-type]
    assert a.rank_origin_indices(0) == b.rank_origin_indices(0)
    assert a.rank_origin_indices(0) != other.rank_origin_indices(0)


def test_hetero_world_size_one_window_slices() -> None:
    """World-size-1 hetero DistributedWindowSampler yields hetero windows."""
    sequence = _hetero_feature_sequence(5)
    sampler = DistributedWindowSampler(
        sequence,
        window_length=3,
        batch_size=2,
        shuffle=False,
        seed=0,
        rank=0,
        world_size=1,
    )
    batches = list(sampler.iter_epoch(0))
    assert batches
    window = batches[0][0]
    assert isinstance(window, HeteroGraphSnapshotSequence)
    assert window.num_timesteps == 3
    assert torch.equal(window[0]["node"].x, sequence[0]["node"].x)


def test_shard_hetero_multi_trajectory_disjoint() -> None:
    """Hetero MultiTrajectory shards via ``shard_sequences_for_rank``."""
    trajectories = [_hetero_feature_sequence(4) for _ in range(4)]
    multi = MultiTrajectory(tuple(trajectories))
    rank0 = shard_sequences_for_rank(multi, rank=0, world_size=2)
    rank1 = shard_sequences_for_rank(multi, rank=1, world_size=2)
    assert len(rank0) == 2
    assert len(rank1) == 2
    assert rank0[0] is trajectories[0]
    assert rank0[1] is trajectories[2]
    assert rank1[0] is trajectories[1]
    assert rank1[1] is trajectories[3]


def test_shard_hetero_raises_when_fewer_than_world_size() -> None:
    """Hetero undersized trajectory lists still raise (0.8 Q6)."""
    with pytest.raises(ValueError, match="DistributedWindowSampler"):
        shard_sequences_for_rank(
            [_hetero_feature_sequence(3)],
            rank=0,
            world_size=2,
        )


def test_neighbor_window_sampler_rejects_hetero() -> None:
    """NeighborWindowSampler stays homogeneous-only."""
    sequence = _hetero_feature_sequence(4)
    with pytest.raises(
        ValueError, match="does not support HeteroGraphSnapshotSequence"
    ):
        NeighborWindowSampler(
            sequence,
            window_length=2,
            num_nodes=2,
            num_hops=1,
        )


def test_build_window_index_list_accepts_hetero() -> None:
    """Origin planner enumerates hetero trajectories like homogeneous ones."""
    sequence = _hetero_feature_sequence(4)
    origins = build_window_index_list([sequence], window_length=3)
    assert origins == [WindowOrigin(0, 0), WindowOrigin(0, 1)]
