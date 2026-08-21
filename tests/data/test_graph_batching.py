"""Tests for collating independent homogeneous snapshot sequences."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data, HeteroData

import koopman_graph
from koopman_graph.data import (
    BatchedGraphTrajectory,
    GraphSnapshotSequence,
    HeteroGraphSnapshotSequence,
    collate_graph_trajectories,
)


def _path_edge_index(num_nodes: int) -> torch.Tensor:
    """Return a bidirectional path graph on ``num_nodes`` vertices.

    Parameters
    ----------
    num_nodes : int
        Node count (at least 2).

    Returns
    -------
    Tensor
        COO index with shape ``(2, 2 * (num_nodes - 1))``.
    """
    src = list(range(num_nodes - 1)) + list(range(1, num_nodes))
    dst = list(range(1, num_nodes)) + list(range(num_nodes - 1))
    return torch.tensor([src, dst], dtype=torch.long)


def _sequence(
    *,
    num_nodes: int,
    num_timesteps: int,
    in_channels: int = 3,
    seed: int = 0,
) -> GraphSnapshotSequence:
    """Build a static-topology homogeneous sequence.

    Parameters
    ----------
    num_nodes : int
        Nodes per snapshot.
    num_timesteps : int
        Snapshot count.
    in_channels : int, optional
        Feature width. Default is ``3``.
    seed : int, optional
        RNG seed. Default is ``0``.

    Returns
    -------
    GraphSnapshotSequence
        Random snapshots on a path graph.
    """
    torch.manual_seed(seed)
    edge_index = _path_edge_index(num_nodes)
    snapshots = [
        Data(x=torch.randn(num_nodes, in_channels), edge_index=edge_index)
        for _ in range(num_timesteps)
    ]
    return GraphSnapshotSequence(snapshots)


def test_collate_time_major_indices_and_disconnected_union() -> None:
    """Collate uses time-major components and keeps graphs disconnected."""
    seq_a = _sequence(num_nodes=3, num_timesteps=4, seed=1)
    seq_b = _sequence(num_nodes=5, num_timesteps=4, seed=2)
    batched = collate_graph_trajectories((seq_a, seq_b))
    assert isinstance(batched, BatchedGraphTrajectory)
    assert batched.num_trajectories == 2
    assert batched.num_timesteps == 4
    assert batched.num_components == 8
    assert batched.time_index.tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert batched.graph_index.tolist() == [0, 1, 0, 1, 0, 1, 0, 1]
    src, dst = batched.batch.edge_index
    assert torch.equal(batched.batch.batch[src], batched.batch.batch[dst])
    slice0 = batched.graphs_at_time(0)
    assert int(slice0.num_graphs) == 2
    assert int(slice0.num_nodes) == 8
    with pytest.raises(ValueError, match="timestep"):
        batched.graphs_at_time(4)


def test_collate_does_not_mutate_source_snapshots() -> None:
    """Source ``Data`` objects are cloned rather than tagged with ``batch``."""
    seq_a = _sequence(num_nodes=3, num_timesteps=2, seed=3)
    original = seq_a[0]
    collate_graph_trajectories((seq_a,))
    assert seq_a[0] is original
    assert getattr(original, "batch", None) is None


def test_collate_rejects_unequal_lengths_and_hetero() -> None:
    """Unequal ``T`` and hetero sequences are refused."""
    short = _sequence(num_nodes=3, num_timesteps=3, seed=4)
    long = _sequence(num_nodes=3, num_timesteps=4, seed=5)
    with pytest.raises(ValueError, match="equal num_timesteps"):
        collate_graph_trajectories((short, long))
    with pytest.raises(ValueError, match="at least one sequence"):
        collate_graph_trajectories(())

    hetero = HeteroData()
    hetero["node"].x = torch.randn(4, 3)
    hetero["node", "r1", "node"].edge_index = torch.tensor(
        [[0, 1], [1, 2]], dtype=torch.long
    )
    hetero_seq = HeteroGraphSnapshotSequence([hetero, hetero])
    with pytest.raises(TypeError, match="GraphSnapshotSequence"):
        collate_graph_trajectories((hetero_seq,))  # type: ignore[arg-type]


def test_collate_rejects_controls_hyperedges_timestamps_and_parameters() -> None:
    """Controls, hyperedges, timestamps, and mu stacks are out of scope."""
    edge_index = _path_edge_index(3)
    snapshots = [Data(x=torch.randn(3, 3), edge_index=edge_index) for _ in range(2)]
    controlled = GraphSnapshotSequence(
        snapshots,
        control_inputs=torch.zeros(2, 1),
    )
    with pytest.raises(ValueError, match="control_inputs"):
        collate_graph_trajectories((controlled,))

    hyper_snaps = [
        Data(
            x=torch.randn(3, 3),
            edge_index=edge_index,
            hyperedge_index=torch.tensor([[0, 1, 2], [0, 0, 0]], dtype=torch.long),
        )
        for _ in range(2)
    ]
    hyper = GraphSnapshotSequence(hyper_snaps)
    with pytest.raises(ValueError, match="hyperedge"):
        collate_graph_trajectories((hyper,))

    stamped = GraphSnapshotSequence(
        snapshots,
        timestamps=torch.tensor([0.0, 1.0]),
    )
    with pytest.raises(ValueError, match="timestamps"):
        collate_graph_trajectories((stamped,))

    parameterized = GraphSnapshotSequence(
        snapshots,
        parameter_trajectory=torch.zeros(2, 1),
    )
    with pytest.raises(ValueError, match="parameter_trajectory"):
        collate_graph_trajectories((parameterized,))


def test_batching_symbols_are_not_on_root_all() -> None:
    """Collate types stay on the data package, not the root façade."""
    assert "BatchedGraphTrajectory" not in koopman_graph.__all__
    assert "collate_graph_trajectories" not in koopman_graph.__all__
    from koopman_graph.data import BatchedGraphTrajectory as imported

    assert imported is BatchedGraphTrajectory
