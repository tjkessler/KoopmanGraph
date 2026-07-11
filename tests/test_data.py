"""Unit tests for graph snapshot sequence utilities."""

from collections.abc import Callable

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from koopman_graph.data import GraphSnapshotSequence


def test_construct_from_data_list(
    synthetic_edge_index: torch.Tensor,
    make_snapshots: Callable[..., list[Data]],
) -> None:
    """Verify construction from a list of ``Data`` snapshots."""
    snapshots = make_snapshots(synthetic_edge_index)
    sequence = GraphSnapshotSequence(snapshots)

    assert sequence.num_timesteps == 3
    assert sequence.num_nodes == 5
    assert sequence.in_channels == 3
    assert torch.equal(sequence.edge_index, synthetic_edge_index)
    assert len(sequence) == 3
    assert sequence[0] is snapshots[0]
    assert list(sequence) == snapshots


def test_single_snapshot_allowed(
    synthetic_edge_index: torch.Tensor,
    make_snapshots: Callable[..., list[Data]],
) -> None:
    """Verify a single-snapshot sequence is accepted."""
    snapshots = make_snapshots(synthetic_edge_index, num_timesteps=1)
    sequence = GraphSnapshotSequence(snapshots)
    assert sequence.num_timesteps == 1


def test_from_arrays_torch(synthetic_edge_index: torch.Tensor) -> None:
    """Verify ``from_arrays`` accepts torch feature arrays."""
    features = torch.randn(4, 5, 3)
    sequence = GraphSnapshotSequence.from_arrays(features, synthetic_edge_index)

    assert sequence.num_timesteps == 4
    assert sequence.num_nodes == 5
    assert sequence.in_channels == 3
    for t, snapshot in enumerate(sequence):
        assert torch.equal(snapshot.x, features[t])
        assert torch.equal(snapshot.edge_index, synthetic_edge_index)


def test_from_arrays_numpy(synthetic_edge_index: torch.Tensor) -> None:
    """Verify ``from_arrays`` accepts numpy feature arrays."""
    features = np.random.randn(2, 5, 3).astype(np.float32)
    edges = synthetic_edge_index.numpy()
    sequence = GraphSnapshotSequence.from_arrays(features, edges)

    assert sequence.num_timesteps == 2
    assert sequence.in_channels == 3
    assert sequence[0].x.dtype == torch.float32


def test_empty_sequence_raises() -> None:
    """Verify empty input raises ``ValueError``."""
    with pytest.raises(ValueError, match="at least one snapshot"):
        GraphSnapshotSequence([])


def test_mismatched_edge_index_raises(
    synthetic_edge_index: torch.Tensor,
    make_snapshots: Callable[..., list[Data]],
) -> None:
    """Verify mismatched topology raises ``ValueError``."""
    snapshots = make_snapshots(synthetic_edge_index)
    snapshots[1] = Data(
        x=torch.randn(5, 3),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
    )
    with pytest.raises(ValueError, match="different edge_index"):
        GraphSnapshotSequence(snapshots)


def test_mismatched_num_nodes_raises(
    synthetic_edge_index: torch.Tensor,
    make_snapshots: Callable[..., list[Data]],
) -> None:
    """Verify mismatched node counts raise ``ValueError``."""
    snapshots = make_snapshots(synthetic_edge_index)
    snapshots[2] = Data(x=torch.randn(4, 3), edge_index=synthetic_edge_index)
    with pytest.raises(ValueError, match="has 4 nodes"):
        GraphSnapshotSequence(snapshots)


def test_mismatched_feature_dim_raises(
    synthetic_edge_index: torch.Tensor,
    make_snapshots: Callable[..., list[Data]],
) -> None:
    """Verify mismatched feature dims raise ``ValueError``."""
    snapshots = make_snapshots(synthetic_edge_index)
    snapshots[1] = Data(x=torch.randn(5, 2), edge_index=synthetic_edge_index)
    with pytest.raises(ValueError, match="feature dimension 2"):
        GraphSnapshotSequence(snapshots)


@pytest.mark.parametrize(
    ("features", "match"),
    [
        (torch.randn(5, 3), "node_features must have shape"),
        (torch.randn(0, 5, 3), "at least one timestep"),
    ],
)
def test_from_arrays_invalid_feature_shape(
    synthetic_edge_index: torch.Tensor,
    features: torch.Tensor,
    match: str,
) -> None:
    """Verify invalid feature shapes raise ``ValueError``."""
    with pytest.raises(ValueError, match=match):
        GraphSnapshotSequence.from_arrays(features, synthetic_edge_index)


def test_from_arrays_invalid_edge_index_shape() -> None:
    """Verify invalid edge index shape raises ``ValueError``."""
    features = torch.randn(2, 5, 3)
    with pytest.raises(ValueError, match="edge_index must have shape"):
        GraphSnapshotSequence.from_arrays(features, torch.tensor([0, 1, 2]))


def test_exported_from_package() -> None:
    """Verify the symbol is exported from the package root."""
    from koopman_graph import GraphSnapshotSequence as ExportedSequence

    assert ExportedSequence is GraphSnapshotSequence


def test_snapshots_property(
    synthetic_edge_index: torch.Tensor,
    make_snapshots: Callable[..., list[Data]],
) -> None:
    """Verify the ``snapshots`` property exposes underlying storage."""
    snapshots = make_snapshots(synthetic_edge_index)
    sequence = GraphSnapshotSequence(snapshots)
    assert sequence.snapshots == snapshots
    assert sequence.snapshots[0] is snapshots[0]


def test_as_tensor_converts_dtype() -> None:
    """Verify ``_as_tensor`` converts tensor dtype when requested."""
    from koopman_graph.data import _as_tensor

    value = torch.randn(2, 3, dtype=torch.float64)
    converted = _as_tensor(value, dtype=torch.float32)
    assert converted.dtype == torch.float32
    assert torch.equal(converted, value.to(dtype=torch.float32))


def test_from_arrays_converts_numpy_dtype(synthetic_edge_index: torch.Tensor) -> None:
    """Verify ``from_arrays`` converts numpy arrays to the requested dtype."""
    features = np.random.randn(2, 5, 3).astype(np.float64)
    sequence = GraphSnapshotSequence.from_arrays(
        features,
        synthetic_edge_index,
        dtype=torch.float32,
    )
    assert sequence[0].x.dtype == torch.float32
