"""Tests for :class:`~koopman_graph.data.HeteroGraphSnapshotSequence`."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data, HeteroData

from koopman_graph.data import HeteroGraphSnapshotSequence


def _multiplex_snapshot(
    *,
    num_nodes: int = 4,
    in_channels: int = 3,
    edge_index_r1: torch.Tensor | None = None,
    edge_index_r2: torch.Tensor | None = None,
) -> HeteroData:
    """Build a one-type, two-relation multiplex ``HeteroData`` snapshot."""
    data = HeteroData()
    data["node"].x = torch.randn(num_nodes, in_channels)
    data["node", "r1", "node"].edge_index = (
        torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        if edge_index_r1 is None
        else edge_index_r1
    )
    data["node", "r2", "node"].edge_index = (
        torch.tensor([[0, 2], [2, 0]], dtype=torch.long)
        if edge_index_r2 is None
        else edge_index_r2
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


def test_construct_multiplex_sequence() -> None:
    """Verify static multiplex construction and schema accessors."""
    snapshots = [_multiplex_snapshot() for _ in range(3)]
    sequence = HeteroGraphSnapshotSequence(snapshots)

    assert sequence.num_timesteps == 3
    assert sequence.num_nodes == 4
    assert sequence.in_channels == 3
    assert sequence.num_nodes_total == 4
    assert sequence.node_type_names == ("node",)
    assert sequence.node_types == {"node": 3}
    assert sequence.num_nodes_dict == {"node": 4}
    assert sequence.num_nodes_of("node") == 4
    assert sequence.edge_types == (
        ("node", "r1", "node"),
        ("node", "r2", "node"),
    )
    assert not sequence.is_dynamic_topology
    assert torch.equal(
        sequence.edge_index_dict[("node", "r1", "node")],
        snapshots[0]["node", "r1", "node"].edge_index,
    )
    assert len(sequence) == 3
    assert sequence[0] is snapshots[0]
    assert list(sequence) == snapshots


def test_single_snapshot_allowed() -> None:
    """Verify a single-snapshot hetero sequence is accepted."""
    sequence = HeteroGraphSnapshotSequence([_multiplex_snapshot()])
    assert sequence.num_timesteps == 1


def test_rejects_homogeneous_data() -> None:
    """Verify homogeneous ``Data`` snapshots raise ``TypeError``."""
    with pytest.raises(TypeError, match="must be HeteroData"):
        HeteroGraphSnapshotSequence(
            [Data(x=torch.randn(3, 2), edge_index=torch.tensor([[0], [1]]))]
        )


def test_rejects_empty_sequence() -> None:
    """Verify an empty snapshot list raises ``ValueError``."""
    with pytest.raises(ValueError, match="at least one snapshot"):
        HeteroGraphSnapshotSequence([])


def test_rejects_no_edge_types() -> None:
    """Verify snapshots without edge types are rejected."""
    data = HeteroData()
    data["node"].x = torch.randn(3, 2)
    with pytest.raises(ValueError, match=r"\|R\| >= 1"):
        HeteroGraphSnapshotSequence([data])


def test_rejects_node_count_churn() -> None:
    """Verify node-count churn names the failing type."""
    snapshots = [_multiplex_snapshot(num_nodes=4) for _ in range(2)]
    snapshots[1] = _multiplex_snapshot(num_nodes=5)
    with pytest.raises(ValueError, match=r"node type 'node'.*5 nodes.*expected 4"):
        HeteroGraphSnapshotSequence(snapshots)


def test_rejects_feature_dim_drift() -> None:
    """Verify feature-dimension drift names the failing type."""
    snapshots = [_multiplex_snapshot(in_channels=3) for _ in range(2)]
    snapshots[1] = _multiplex_snapshot(in_channels=4)
    with pytest.raises(
        ValueError,
        match=r"node type 'node'.*feature dimension 4.*expected 3",
    ):
        HeteroGraphSnapshotSequence(snapshots)


def test_rejects_edge_type_set_drift() -> None:
    """Verify edge-type set drift raises a named error."""
    snapshots = [_multiplex_snapshot() for _ in range(2)]
    alt = HeteroData()
    alt["node"].x = torch.randn(4, 3)
    alt["node", "r1", "node"].edge_index = snapshots[0]["node", "r1", "node"].edge_index
    alt["node", "r3", "node"].edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    snapshots[1] = alt
    with pytest.raises(ValueError, match="edge-type set"):
        HeteroGraphSnapshotSequence(snapshots)


def test_rejects_static_topology_edge_index_drift() -> None:
    """Verify static mode rejects per-relation edge_index drift."""
    snapshots = [_multiplex_snapshot() for _ in range(2)]
    alt_edges = torch.tensor([[0, 3], [3, 1]], dtype=torch.long)
    snapshots[1] = _multiplex_snapshot(edge_index_r1=alt_edges)
    with pytest.raises(
        ValueError,
        match=r"edge type \('node', 'r1', 'node'\).*different edge_index",
    ):
        HeteroGraphSnapshotSequence(snapshots)


def test_dynamic_topology_allowed_with_flag() -> None:
    """Verify dynamic topology is accepted when explicitly enabled."""
    snapshots = [_multiplex_snapshot() for _ in range(2)]
    alt_edges = torch.tensor([[0, 3], [3, 1]], dtype=torch.long)
    snapshots[1] = _multiplex_snapshot(edge_index_r2=alt_edges)
    sequence = HeteroGraphSnapshotSequence(snapshots, allow_dynamic_topology=True)

    assert sequence.allow_dynamic_topology
    assert sequence.is_dynamic_topology
    with pytest.raises(ValueError, match="edge_index_dict is undefined"):
        _ = sequence.edge_index_dict


def test_typed_multi_node_schema() -> None:
    """Verify multi-node-type sequences expose dict accessors."""
    snapshots = [_typed_snapshot() for _ in range(2)]
    sequence = HeteroGraphSnapshotSequence(snapshots)

    assert sequence.num_nodes_dict == {"gen": 2, "load": 3}
    assert sequence.num_nodes_total == 5
    assert sequence.node_types == {"gen": 3, "load": 2}
    assert sequence.edge_types == (("gen", "feeds", "load"),)
    with pytest.raises(ValueError, match="num_nodes is defined only"):
        _ = sequence.num_nodes
    with pytest.raises(ValueError, match="in_channels is defined only"):
        _ = sequence.in_channels


def test_no_homogeneous_edge_index_attribute() -> None:
    """Verify there is no silent homogeneous ``edge_index`` property."""
    sequence = HeteroGraphSnapshotSequence([_multiplex_snapshot()])
    assert not hasattr(sequence, "edge_index")
    assert "edge_index" not in dir(type(sequence))


def test_slice_preserves_controls_masks_and_topology() -> None:
    """Verify contiguous slices preserve metadata and topology."""
    snapshots = [_multiplex_snapshot() for _ in range(4)]
    controls = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    timestamps = torch.tensor([0.0, 0.1, 0.2, 0.3])
    masks = {
        "node": torch.tensor(
            [
                [True, True, False, True],
                [False, True, True, True],
                [True, False, True, True],
                [True, True, True, False],
            ]
        )
    }
    sequence = HeteroGraphSnapshotSequence(
        snapshots,
        control_inputs=controls,
        timestamps=timestamps,
        observation_masks=masks,
    )

    window = sequence.slice(1, 4)

    assert window.num_timesteps == 3
    assert torch.equal(window[0]["node"].x, snapshots[1]["node"].x)
    assert torch.equal(window.control_inputs, controls[1:4])
    assert torch.equal(window.timestamps, timestamps[1:4])
    assert torch.equal(window.observation_masks["node"], masks["node"][1:4])
    assert window.edge_types == sequence.edge_types


@pytest.mark.parametrize("start, stop", [(-1, 2), (1, 1), (3, 2), (0, 6)])
def test_slice_rejects_invalid_bounds(start: int, stop: int) -> None:
    """Verify invalid temporal slice bounds raise a clear error."""
    sequence = HeteroGraphSnapshotSequence([_multiplex_snapshot() for _ in range(4)])
    with pytest.raises(ValueError, match="slice bounds"):
        sequence.slice(start, stop)


def test_per_type_control_dict() -> None:
    """Verify per-type control dicts validate and expose control_dim."""
    snapshots = [_typed_snapshot() for _ in range(2)]
    controls = {
        "gen": torch.randn(2, 2, 1),
        "load": torch.randn(2, 1),
    }
    sequence = HeteroGraphSnapshotSequence(snapshots, control_inputs=controls)
    assert sequence.control_dim == 1
    assert sequence.has_controls


def test_tensor_controls_reject_multi_type_per_node_layout() -> None:
    """Verify ``(T, N, C)`` tensor controls require a single node type."""
    snapshots = [_typed_snapshot() for _ in range(2)]
    with pytest.raises(ValueError, match="single node type"):
        HeteroGraphSnapshotSequence(
            snapshots,
            control_inputs=torch.randn(2, 5, 1),
        )


def test_exported_from_data_package_not_root() -> None:
    """Verify capability-module export without root ``__all__`` growth."""
    import koopman_graph
    from koopman_graph import data as kg_data

    assert "HeteroGraphSnapshotSequence" in kg_data.__all__
    assert "HeteroGraphSnapshotSequence" not in koopman_graph.__all__
