"""Tests for graph-state snapshots and frozen dynamics config."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch
from torch_geometric.data import Data

from koopman_graph.data import (
    DEFAULT_CANDIDATE_K,
    GraphDynamicsConfig,
    GraphSnapshotSequence,
    GraphStateSnapshot,
    graph_dynamics_from_mapping,
    graph_state_at,
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
    num_nodes: int = 4,
    num_timesteps: int = 3,
    in_channels: int = 2,
    presence: torch.Tensor | None = None,
    entity_ids: tuple[str, ...] | None = None,
    edge_weight: torch.Tensor | None = None,
) -> GraphSnapshotSequence:
    """Build a static-topology homogeneous sequence.

    Parameters
    ----------
    num_nodes : int, optional
        Nodes per snapshot.
    num_timesteps : int, optional
        Snapshot count.
    in_channels : int, optional
        Feature width.
    presence : Tensor or None, optional
        Optional ``(T, N)`` presence masks.
    entity_ids : tuple of str or None, optional
        Optional universe row names.
    edge_weight : Tensor or None, optional
        Optional shared edge weights.

    Returns
    -------
    GraphSnapshotSequence
        Sequence on a path graph.
    """
    torch.manual_seed(0)
    edge_index = _path_edge_index(num_nodes)
    snapshots = []
    for _ in range(num_timesteps):
        data = Data(
            x=torch.randn(num_nodes, in_channels),
            edge_index=edge_index,
        )
        if edge_weight is not None:
            data.edge_weight = edge_weight
        snapshots.append(data)
    return GraphSnapshotSequence(
        snapshots,
        presence_masks=presence,
        entity_ids=entity_ids,
        allow_node_churn=bool(presence is not None and not bool(presence.all())),
    )


def test_graph_dynamics_config_defaults_are_sparse_candidate() -> None:
    """Default config selects the sparse-candidate head with ``k=8``."""
    config = GraphDynamicsConfig()
    assert config.topology_head == "sparse_candidate"
    assert config.recursive_training is True
    assert config.topology_loss_weight == 1.0
    assert config.presence_loss_weight == 1.0
    assert config.candidate_k == DEFAULT_CANDIDATE_K == 8
    mapping = config.to_mapping()
    restored = graph_dynamics_from_mapping(mapping)
    assert restored == config


def test_graph_dynamics_config_rejects_invalid_fields() -> None:
    """Head kind, flags, weights, and ``candidate_k`` are validated."""
    with pytest.raises(ValueError, match="topology_head"):
        GraphDynamicsConfig(topology_head="self_adaptive")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="recursive_training"):
        GraphDynamicsConfig(recursive_training=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="topology_loss_weight"):
        GraphDynamicsConfig(topology_loss_weight=-0.1)
    with pytest.raises(ValueError, match="presence_loss_weight"):
        GraphDynamicsConfig(presence_loss_weight=float("inf"))
    with pytest.raises(ValueError, match="candidate_k"):
        GraphDynamicsConfig(candidate_k=0)
    frozen = GraphDynamicsConfig()
    with pytest.raises(FrozenInstanceError):
        frozen.topology_head = "none"  # type: ignore[misc]


def test_graph_dynamics_from_mapping_none_unknown_and_type() -> None:
    """``None`` is the 0.14 default; unknown keys and non-mappings raise."""
    assert graph_dynamics_from_mapping(None) is None
    empty = graph_dynamics_from_mapping({})
    assert empty is not None
    assert empty.topology_head == "sparse_candidate"
    with pytest.raises(TypeError, match="mapping or None"):
        graph_dynamics_from_mapping(["sparse_candidate"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown graph_dynamics keys"):
        graph_dynamics_from_mapping({"topology_head": "none", "dense_max_nodes": 64})


def test_graph_state_snapshot_validates_shapes() -> None:
    """Snapshot construction rejects rank and length mismatches."""
    x = torch.randn(3, 2)
    edge_index = _path_edge_index(3)
    presence = torch.ones(3, dtype=torch.bool)
    GraphStateSnapshot(
        x=x,
        edge_index=edge_index,
        edge_weight=None,
        presence=presence,
    )
    with pytest.raises(ValueError, match="x must have shape"):
        GraphStateSnapshot(
            x=torch.randn(3),
            edge_index=edge_index,
            edge_weight=None,
            presence=presence,
        )
    with pytest.raises(ValueError, match="edge_index must have shape"):
        GraphStateSnapshot(
            x=x,
            edge_index=torch.arange(3),
            edge_weight=None,
            presence=presence,
        )
    with pytest.raises(ValueError, match="edge_weight must have shape"):
        GraphStateSnapshot(
            x=x,
            edge_index=edge_index,
            edge_weight=torch.ones(1),
            presence=presence,
        )
    with pytest.raises(ValueError, match="presence must have shape"):
        GraphStateSnapshot(
            x=x,
            edge_index=edge_index,
            edge_weight=None,
            presence=torch.ones(2, dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="entity_ids length"):
        GraphStateSnapshot(
            x=x,
            edge_index=edge_index,
            edge_weight=None,
            presence=presence,
            entity_ids=("a", "b"),
        )


def test_graph_state_at_copies_features_edges_and_presence() -> None:
    """Helper returns the requested timestep, including negative indices."""
    weights = torch.ones(_path_edge_index(4).shape[1])
    presence = torch.ones(3, 4, dtype=torch.bool)
    presence[1, 3] = False
    sequence = _sequence(
        presence=presence,
        entity_ids=("a", "b", "c", "d"),
        edge_weight=weights,
    )
    mid = graph_state_at(sequence, 1)
    assert torch.equal(mid.x, sequence[1].x)
    assert torch.equal(mid.edge_index, sequence[1].edge_index)
    assert mid.edge_weight is not None
    assert torch.equal(mid.edge_weight, weights)
    assert torch.equal(mid.presence, presence[1])
    assert mid.entity_ids == ("a", "b", "c", "d")
    last = graph_state_at(sequence, -1)
    assert torch.equal(last.x, sequence[2].x)
    plain = _sequence()
    filled = graph_state_at(plain, 0)
    assert bool(filled.presence.all())
    assert filled.entity_ids is None
    with pytest.raises(IndexError):
        graph_state_at(sequence, 9)
    with pytest.raises(TypeError, match="GraphSnapshotSequence"):
        graph_state_at([sequence[0]], 0)  # type: ignore[arg-type]
