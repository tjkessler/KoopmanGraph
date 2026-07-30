"""Device movers for homogeneous and multiplex snapshot sequences."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data, HeteroData

from koopman_graph.data import GraphSnapshotSequence, HeteroGraphSnapshotSequence
from koopman_graph.graph_utils import snapshot_to_device
from koopman_graph.training.device import sequence_to_device

_CPU = torch.device("cpu")
_CUDA = torch.device("cuda")


def _multiplex_snapshot(
    *,
    num_nodes: int = 4,
    in_channels: int = 3,
    seed: int = 0,
) -> HeteroData:
    generator = torch.Generator().manual_seed(seed)
    data = HeteroData()
    data["node"].x = torch.randn(num_nodes, in_channels, generator=generator)
    data["node", "r1", "node"].edge_index = torch.tensor(
        [[0, 1, 2], [1, 2, 0]],
        dtype=torch.long,
    )
    data["node", "r2", "node"].edge_index = torch.tensor(
        [[0, 2], [2, 3]],
        dtype=torch.long,
    )
    return data


def _homo_snapshot(*, num_nodes: int = 3, in_channels: int = 2, seed: int = 0) -> Data:
    generator = torch.Generator().manual_seed(seed)
    return Data(
        x=torch.randn(num_nodes, in_channels, generator=generator),
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        edge_weight=torch.tensor([0.5, 1.5]),
    )


def test_snapshot_to_device_preserves_hetero_edge_types_and_features() -> None:
    """HeteroData moves without casting to Data; stores stay intact."""
    snapshot = _multiplex_snapshot(seed=1)
    moved = snapshot_to_device(snapshot, _CPU)
    assert type(moved) is HeteroData
    assert not isinstance(moved, Data)
    assert set(moved.node_types) == {"node"}
    assert set(moved.edge_types) == {
        ("node", "r1", "node"),
        ("node", "r2", "node"),
    }
    torch.testing.assert_close(moved["node"].x, snapshot["node"].x)
    torch.testing.assert_close(
        moved["node", "r1", "node"].edge_index,
        snapshot["node", "r1", "node"].edge_index,
    )
    torch.testing.assert_close(
        moved["node", "r2", "node"].edge_index,
        snapshot["node", "r2", "node"].edge_index,
    )


def test_sequence_to_device_preserves_hetero_masks_and_schema() -> None:
    """Hetero sequence keeps container type, edge schema, and mask dict."""
    snapshots = [_multiplex_snapshot(seed=i) for i in range(3)]
    masks = {
        "node": torch.tensor(
            [
                [True, True, False, True],
                [True, False, True, True],
                [False, True, True, True],
            ]
        )
    }
    controls = torch.randn(3, 1)
    timestamps = torch.tensor([0.0, 0.1, 0.2])
    sequence = HeteroGraphSnapshotSequence(
        snapshots,
        control_inputs=controls,
        timestamps=timestamps,
        observation_masks=masks,
    )

    moved = sequence_to_device(sequence, _CPU)
    assert type(moved) is HeteroGraphSnapshotSequence
    assert not isinstance(moved, GraphSnapshotSequence)
    assert moved.num_timesteps == 3
    assert moved.edge_types == sequence.edge_types
    assert moved.observation_masks is not None
    assert set(moved.observation_masks) == {"node"}
    assert torch.equal(moved.observation_masks["node"], masks["node"])
    assert moved.observation_masks["node"].device == _CPU
    assert moved.control_inputs is not None
    torch.testing.assert_close(moved.control_inputs, controls)
    torch.testing.assert_close(moved.timestamps, timestamps)
    torch.testing.assert_close(moved[0]["node"].x, snapshots[0]["node"].x)
    assert type(moved[0]) is HeteroData


def test_sequence_to_device_preserves_hetero_per_type_control_dict() -> None:
    """Per-type hetero control dicts move via the shared mapping helper."""
    snapshots = [_multiplex_snapshot(seed=i) for i in range(2)]
    controls = {"node": torch.randn(2, 4, 1)}
    sequence = HeteroGraphSnapshotSequence(snapshots, control_inputs=controls)
    moved = sequence_to_device(sequence, _CPU)
    assert type(moved) is HeteroGraphSnapshotSequence
    assert isinstance(moved.control_inputs, dict)
    torch.testing.assert_close(moved.control_inputs["node"], controls["node"])
    assert moved.control_inputs["node"].device == _CPU


def test_sequence_to_device_preserves_homogeneous_observation_masks() -> None:
    """Homogeneous observation_masks survive device transfer."""
    snapshots = [_homo_snapshot(seed=i) for i in range(3)]
    masks = torch.tensor(
        [
            [True, False, True],
            [True, True, False],
            [False, True, True],
        ]
    )
    sequence = GraphSnapshotSequence(snapshots, observation_masks=masks)
    moved = sequence_to_device(sequence, _CPU)
    assert isinstance(moved, GraphSnapshotSequence)
    assert moved.observation_masks is not None
    assert torch.equal(moved.observation_masks, masks)
    assert moved.observation_masks.device == _CPU
    torch.testing.assert_close(moved[0].x, snapshots[0].x)
    torch.testing.assert_close(moved[0].edge_weight, snapshots[0].edge_weight)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_hetero_sequence_cuda_cpu_round_trip() -> None:
    """Hetero sequence round-trips CPU → CUDA → CPU with matching stores."""
    snapshots = [_multiplex_snapshot(seed=i) for i in range(2)]
    masks = {"node": torch.ones(2, 4, dtype=torch.bool)}
    sequence = HeteroGraphSnapshotSequence(snapshots, observation_masks=masks)

    on_cuda = sequence_to_device(sequence, _CUDA)
    assert isinstance(on_cuda, HeteroGraphSnapshotSequence)
    assert on_cuda[0]["node"].x.device.type == "cuda"
    assert on_cuda.observation_masks is not None
    assert on_cuda.observation_masks["node"].device.type == "cuda"

    back = sequence_to_device(on_cuda, _CPU)
    assert isinstance(back, HeteroGraphSnapshotSequence)
    assert back.edge_types == sequence.edge_types
    torch.testing.assert_close(back[0]["node"].x, snapshots[0]["node"].x)
    assert back.observation_masks is not None
    assert torch.equal(back.observation_masks["node"], masks["node"])
    assert back[0]["node"].x.device == _CPU
