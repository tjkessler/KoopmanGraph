"""Injective entity remap into a fixed union capacity."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data, HeteroData

from koopman_graph.data import (
    EntityRemap,
    GraphSnapshotSequence,
    remap_node_features,
)


def _path_edge_index(num_nodes: int) -> torch.Tensor:
    """Return a bidirectional path.

    Parameters
    ----------
    num_nodes : int
        Node count (at least 2).

    Returns
    -------
    Tensor
        COO index.
    """
    src = list(range(num_nodes - 1)) + list(range(1, num_nodes))
    dst = list(range(1, num_nodes)) + list(range(num_nodes - 1))
    return torch.tensor([src, dst], dtype=torch.long)


def test_entity_remap_scatters_features_and_edges() -> None:
    """Mapped rows keep features and remapped endpoints."""
    remap = EntityRemap(
        entity_ids=("a", "b", "c", "d"),
        index=torch.tensor([0, 2, 3]),
    )
    features = torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    mapped = remap.apply_features(features)
    assert mapped.shape == (4, 2)
    torch.testing.assert_close(mapped[2], features[1])
    assert torch.equal(mapped[1], torch.zeros(2))
    assert torch.equal(remap.presence_mask, torch.tensor([True, False, True, True]))
    snapshot = Data(x=features, edge_index=_path_edge_index(3))
    union = remap.apply_snapshot(snapshot)
    assert int(union.num_nodes) == 4
    expected_edges = torch.tensor(
        [[0, 2, 2, 3], [2, 0, 3, 2]],
        dtype=torch.long,
    )
    # Path 0-1-2 becomes 0-2-3 (bidirectional).
    got = set(map(tuple, union.edge_index.t().tolist()))
    expected = set(map(tuple, expected_edges.t().tolist()))
    assert got == expected


def test_entity_remap_apply_snapshots_builds_fixed_union_sequence() -> None:
    """Remapped snapshots construct a presence-masked sequence."""
    remap = EntityRemap(
        entity_ids=("u0", "u1", "u2"),
        index=torch.tensor([0, 2]),
    )
    edges = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    snapshots = [
        Data(x=torch.ones(2, 1), edge_index=edges),
        Data(x=2 * torch.ones(2, 1), edge_index=edges),
    ]
    remapped, presence = remap.apply_snapshots(snapshots)
    sequence = GraphSnapshotSequence(
        remapped,
        presence_masks=presence,
        entity_ids=remap.entity_ids,
        allow_node_churn=True,
    )
    assert sequence.num_nodes == 3
    assert sequence.entity_ids == remap.entity_ids
    assert torch.equal(sequence.presence_mask_at(0), remap.presence_mask)


def test_entity_remap_refuses_colliding_index_and_ids() -> None:
    """Colliding destinations or union names raise."""
    with pytest.raises(ValueError, match="injective"):
        EntityRemap(
            entity_ids=("a", "b", "c"),
            index=torch.tensor([0, 0]),
        )
    with pytest.raises(ValueError, match="duplicate"):
        EntityRemap(
            entity_ids=("a", "a"),
            index=torch.tensor([0]),
        )


def test_mismatched_node_count_refuses_unbounded_growth() -> None:
    """Sequences may not grow N without an explicit remap."""
    path3 = _path_edge_index(3)
    path4 = _path_edge_index(4)
    snapshots = [
        Data(x=torch.randn(3, 2), edge_index=path3),
        Data(x=torch.randn(4, 2), edge_index=path4),
    ]
    with pytest.raises(ValueError, match="EntityRemap"):
        GraphSnapshotSequence(snapshots)
    with pytest.raises(ValueError, match="unbounded"):
        GraphSnapshotSequence(snapshots)


def test_entity_remap_rejects_hetero_and_bad_edges() -> None:
    """Hetero graphs and out-of-range endpoints are refused."""
    remap = EntityRemap(entity_ids=("a", "b"), index=torch.tensor([0, 1]))
    hetero = HeteroData()
    hetero["node"].x = torch.ones(2, 1)
    with pytest.raises(TypeError, match="homogeneous Data"):
        remap.apply_snapshot(hetero)
    bad = Data(
        x=torch.ones(2, 1),
        edge_index=torch.tensor([[0, 2], [1, 0]], dtype=torch.long),
    )
    with pytest.raises(ValueError, match="edge_index endpoints"):
        remap.apply_snapshot(bad)


def test_remap_node_features_primitive_still_injective() -> None:
    """Tensor helper remains the scatter primitive."""
    features = torch.arange(6, dtype=torch.float32).view(3, 2)
    mapped = remap_node_features(
        features,
        old_index=torch.tensor([0, 2, 5]),
        new_capacity=6,
    )
    assert mapped.shape == (6, 2)
    torch.testing.assert_close(mapped[5], features[2])
