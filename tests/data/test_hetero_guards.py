"""Coverage and error-path tests for :mod:`koopman_graph.data`."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch_geometric.data import HeteroData

from koopman_graph.data.validation import (
    _hetero_edge_weight,
    _require_hetero_node_x,
    hetero_snapshots_have_dynamic_topology,
    infer_hetero_schema,
    validate_hetero_observation_masks,
    validate_hetero_snapshot_metadata,
    validate_shared_hetero_topology,
)


def _multiplex_snapshot(*, num_nodes: int = 4, in_channels: int = 3) -> HeteroData:
    """Build a one-type, two-relation multiplex snapshot."""
    data = HeteroData()
    data["node"].x = torch.randn(num_nodes, in_channels)
    data["node", "r1", "node"].edge_index = torch.tensor(
        [[0, 1, 2], [1, 2, 3]],
        dtype=torch.long,
    )
    data["node", "r2", "node"].edge_index = torch.tensor(
        [[0, 2], [2, 0]],
        dtype=torch.long,
    )
    return data


def _typed_snapshot() -> HeteroData:
    """Build a two-type snapshot with one cross relation."""
    data = HeteroData()
    data["gen"].x = torch.randn(2, 3)
    data["load"].x = torch.randn(3, 2)
    data["gen", "feeds", "load"].edge_index = torch.tensor(
        [[0, 1], [0, 2]],
        dtype=torch.long,
    )
    return data


def test_hetero_edge_weight_rejects_non_tensor() -> None:
    """Non-tensor edge_weight on a store raises TypeError."""
    store = SimpleNamespace(edge_weight=[1.0, 2.0])
    with pytest.raises(TypeError, match="must be a Tensor"):
        _hetero_edge_weight(store)


def test_require_hetero_node_x_error_paths() -> None:
    """Missing type / None features / wrong ndim raise named ValueErrors."""
    snap = _typed_snapshot()
    with pytest.raises(ValueError, match="missing node type"):
        _require_hetero_node_x(snap, "ghost", index=2)

    class _Store:
        def __init__(self, features: torch.Tensor | None) -> None:
            self.x = features

    class _Snap:
        node_types = ("gen",)

        def __getitem__(self, _key: str) -> _Store:
            return _Store(None)

    with pytest.raises(ValueError, match="has no feature tensor x"):
        _require_hetero_node_x(_Snap(), "gen", index=0)  # type: ignore[arg-type]

    snap["gen"].x = torch.randn(4)
    with pytest.raises(ValueError, match="must have shape"):
        _require_hetero_node_x(snap, "gen", index=0)


def test_infer_hetero_schema_error_paths() -> None:
    """Schema inference rejects empty / malformed hetero snapshots."""
    with pytest.raises(ValueError, match="has no node types"):
        infer_hetero_schema(HeteroData())

    no_edges = HeteroData()
    no_edges["node"].x = torch.randn(3, 2)
    with pytest.raises(ValueError, match=r"\|R\| >= 1"):
        infer_hetero_schema(no_edges)

    bad_src = HeteroData()
    bad_src["load"].x = torch.randn(2, 2)
    bad_src["ghost", "r", "load"].edge_index = torch.tensor(
        [[0], [0]],
        dtype=torch.long,
    )
    with pytest.raises(ValueError, match="unknown source node type"):
        infer_hetero_schema(bad_src)

    bad_dst = HeteroData()
    bad_dst["gen"].x = torch.randn(2, 2)
    bad_dst["gen", "r", "ghost"].edge_index = torch.tensor(
        [[0], [0]],
        dtype=torch.long,
    )
    with pytest.raises(ValueError, match="unknown destination"):
        infer_hetero_schema(bad_dst)

    bad_shape = HeteroData()
    bad_shape["node"].x = torch.randn(2, 2)
    bad_shape["node", "r", "node"].edge_index = torch.ones(3, 2, dtype=torch.long)
    with pytest.raises(ValueError, match=r"\(2, num_edges\)"):
        infer_hetero_schema(bad_shape)


def test_validate_hetero_metadata_and_shared_topology_errors() -> None:
    """Metadata / shared-topology helpers reject drift and weight mismatches."""
    snap0 = _typed_snapshot()
    snap1 = HeteroData()
    snap1["gen"].x = torch.randn(2, 3)
    snap1["load"].x = torch.randn(3, 2)
    snap1["bus"].x = torch.randn(1, 2)
    snap1["gen", "feeds", "load"].edge_index = torch.tensor(
        [[0, 1], [0, 2]],
        dtype=torch.long,
    )
    with pytest.raises(ValueError, match="node types"):
        validate_hetero_snapshot_metadata([snap0, snap1])

    snap1b = _typed_snapshot()
    feeds = snap1b["gen", "feeds", "load"].edge_index
    snap1b["gen", "other", "load"].edge_index = feeds
    del snap1b["gen", "feeds", "load"]
    with pytest.raises(ValueError, match="edge-type set"):
        validate_hetero_snapshot_metadata([snap0, snap1b])

    edge_types = [("node", "r1", "node"), ("node", "r2", "node")]
    assert hetero_snapshots_have_dynamic_topology([], edge_types) is False
    m0 = _multiplex_snapshot()
    m1 = _multiplex_snapshot()
    m1["node", "r1", "node"].edge_index = torch.tensor(
        [[0, 1], [1, 0]],
        dtype=torch.long,
    )
    assert hetero_snapshots_have_dynamic_topology([m0, m1], edge_types) is True

    w0 = _multiplex_snapshot()
    w1 = _multiplex_snapshot()
    w0["node", "r1", "node"].edge_weight = torch.ones(3)
    with pytest.raises(ValueError, match="presence does not match"):
        validate_shared_hetero_topology([w0, w1])
    w1["node", "r1", "node"].edge_weight = torch.full((3,), 2.0)
    with pytest.raises(ValueError, match="different edge_weight"):
        validate_shared_hetero_topology([w0, w1])


def test_validate_hetero_observation_masks_errors() -> None:
    """Per-type observation mask validation wraps key / shape failures."""
    with pytest.raises(ValueError, match="keys must match node types"):
        validate_hetero_observation_masks(
            {"gen": torch.ones(2, 2, dtype=torch.bool)},
            num_timesteps=2,
            num_nodes={"gen": 2, "load": 3},
        )
    with pytest.raises(ValueError, match=r"observation_masks\['gen'\]"):
        validate_hetero_observation_masks(
            {
                "gen": torch.ones(2, 5, dtype=torch.bool),
                "load": torch.ones(2, 3, dtype=torch.bool),
            },
            num_timesteps=2,
            num_nodes={"gen": 2, "load": 3},
        )
