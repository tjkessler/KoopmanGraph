"""Tests for shared ``resolve_window_sampler`` construction policy."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data, HeteroData

from koopman_graph.data import (
    GraphSnapshotSequence,
    HeteroGraphSnapshotSequence,
    NeighborWindowSampler,
    WindowSampler,
    resolve_window_sampler,
)
from koopman_graph.distributed import DistributedWindowSampler
from koopman_graph.distributed._fit_epochs import resolve_distributed_window_sampler
from koopman_graph.model import GraphKoopmanModel
from koopman_graph.nn import RelGraphDecoder, RelGraphEncoder


def _homo_sequence(timesteps: int = 5) -> GraphSnapshotSequence:
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    snapshots = [
        Data(x=torch.full((2, 1), float(t)), edge_index=edge_index, num_nodes=2)
        for t in range(timesteps)
    ]
    return GraphSnapshotSequence(snapshots)


def _hetero_sequence(timesteps: int = 5) -> HeteroGraphSnapshotSequence:
    snapshots = []
    for t in range(timesteps):
        data = HeteroData()
        data["node"].x = torch.full((2, 1), float(t))
        data["node", "rel", "node"].edge_index = torch.tensor(
            [[0, 1], [1, 0]],
            dtype=torch.long,
        )
        snapshots.append(data)
    return HeteroGraphSnapshotSequence(snapshots)


def test_resolve_local_none_without_window_or_sampler() -> None:
    """Neither sampler nor window_length ⇒ full-sequence training."""
    assert (
        resolve_window_sampler(
            [_homo_sequence()],
            window_length=None,
            sampler=None,
            distributed=False,
        )
        is None
    )


def test_resolve_local_builds_window_sampler() -> None:
    """Local path constructs WindowSampler from window_length."""
    sequence = _homo_sequence()
    resolved = resolve_window_sampler(
        [sequence],
        window_length=3,
        batch_size=2,
        windows_per_epoch=4,
        window_seed=0,
        distributed=False,
    )
    assert isinstance(resolved, WindowSampler)
    assert resolved.window_length == 3
    assert resolved.batch_size == 2
    assert resolved.windows_per_epoch == 4
    assert resolved.seed == 0


def test_resolve_local_passthrough_neighbor_sampler() -> None:
    """Local path accepts a pre-built NeighborWindowSampler."""
    sequence = _homo_sequence()
    sampler = NeighborWindowSampler(
        sequence,
        window_length=2,
        num_nodes=1,
        num_hops=1,
        batch_size=1,
        shuffle=False,
    )
    resolved = resolve_window_sampler(
        [sequence],
        window_length=None,
        sampler=sampler,
        distributed=False,
    )
    assert resolved is sampler


def test_resolve_rejects_sampler_and_window_length() -> None:
    """sampler and window_length remain mutually exclusive."""
    sequence = _homo_sequence()
    sampler = WindowSampler(sequence, window_length=2, batch_size=1, shuffle=False)
    with pytest.raises(ValueError, match="sampler or window_length"):
        resolve_window_sampler(
            [sequence],
            window_length=2,
            sampler=sampler,
            distributed=False,
        )


def test_resolve_distributed_builds_distributed_sampler() -> None:
    """Distributed path constructs DistributedWindowSampler from window_length."""
    sequence = _homo_sequence()
    resolved = resolve_window_sampler(
        [sequence],
        window_length=3,
        batch_size=2,
        window_seed=0,
        distributed=True,
        rank=0,
        world_size=1,
    )
    assert isinstance(resolved, DistributedWindowSampler)
    assert resolved.window_length == 3
    assert resolved.rank == 0
    assert resolved.world_size == 1


def test_resolve_distributed_rejects_plain_window_sampler() -> None:
    """Distributed path rejects non-rank-aware local samplers."""
    sequence = _homo_sequence()
    sampler = WindowSampler(sequence, window_length=2, batch_size=1, shuffle=False)
    with pytest.raises(ValueError, match="not rank-aware"):
        resolve_window_sampler(
            [sequence],
            window_length=None,
            sampler=sampler,
            distributed=True,
            api_name="fit_ddp",
        )


def test_resolve_distributed_rejects_neighbor_sampler() -> None:
    """Distributed path rejects NeighborWindowSampler."""
    sequence = _homo_sequence()
    sampler = NeighborWindowSampler(
        sequence,
        window_length=2,
        num_nodes=1,
        num_hops=1,
        batch_size=1,
        shuffle=False,
    )
    with pytest.raises(ValueError, match="not rank-aware"):
        resolve_window_sampler(
            [sequence],
            window_length=None,
            sampler=sampler,
            distributed=True,
            api_name="fit_ddp",
        )


def test_resolve_distributed_wrapper_delegates() -> None:
    """``resolve_distributed_window_sampler`` shares the distributed branch."""
    sequence = _homo_sequence()
    resolved = resolve_distributed_window_sampler(
        [sequence],
        window_length=3,
        batch_size=2,
        windows_per_epoch=None,
        window_seed=0,
        sampler=None,
        api_name="fit_ddp",
        rank=0,
        world_size=1,
    )
    assert isinstance(resolved, DistributedWindowSampler)


def test_neighbor_sampler_still_rejects_hetero() -> None:
    """NeighborWindowSampler remains homogeneous-only at construction."""
    with pytest.raises(
        ValueError,
        match="does not support HeteroGraphSnapshotSequence",
    ):
        NeighborWindowSampler(
            _hetero_sequence(),
            window_length=2,
            num_nodes=1,
            num_hops=1,
            batch_size=1,
            shuffle=False,
        )


def test_run_fit_loop_rejects_neighbor_sampler_on_hetero() -> None:
    """NeighborWindowSampler remains rejected on hetero fit paths."""
    # Construction already rejects hetero sequences; fit-path guard is covered
    # in tests/test_hetero_windowed_fit.py. Here ensure homo NeighborSampler
    # cannot be paired with hetero train_sequences via resolve + fit.
    homo = _homo_sequence()
    sampler = NeighborWindowSampler(
        homo,
        window_length=2,
        num_nodes=1,
        num_hops=1,
        batch_size=1,
        shuffle=False,
    )
    model = GraphKoopmanModel(
        encoder=RelGraphEncoder(1, 8, 4, num_relations=1, num_layers=1),
        decoder=RelGraphDecoder(4, 8, 1, num_relations=1, num_layers=1),
        latent_dim=4,
        time_step=1.0,
        koopman="hetero_graph",
    )
    with pytest.raises(ValueError, match="NeighborWindowSampler is homogeneous-only"):
        model.fit(_hetero_sequence(), epochs=1, sampler=sampler)
