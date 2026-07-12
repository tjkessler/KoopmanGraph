"""Unit tests for graph snapshot sequence utilities."""

from collections.abc import Callable

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from koopman_graph.data import (
    GraphSnapshotSequence,
    TemporalSplit,
    WindowSampler,
    temporal_split,
)


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


def test_sequence_slice_preserves_controls_and_topology(
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Verify contiguous slices preserve snapshots, controls, and topology."""
    features = torch.arange(60, dtype=torch.float32).reshape(4, 5, 3)
    controls = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    sequence = GraphSnapshotSequence.from_arrays(
        features,
        synthetic_edge_index,
        control_inputs=controls,
    )

    window = sequence.slice(1, 4)

    assert window.num_timesteps == 3
    assert torch.equal(window[0].x, features[1])
    assert torch.equal(window.edge_index, synthetic_edge_index)
    assert torch.equal(window.control_inputs, controls[1:4])


@pytest.mark.parametrize("start, stop", [(-1, 2), (1, 1), (3, 2), (0, 6)])
def test_sequence_slice_rejects_invalid_bounds(
    scaling_sequence: GraphSnapshotSequence,
    start: int,
    stop: int,
) -> None:
    """Verify invalid temporal slice bounds raise a clear error."""
    with pytest.raises(ValueError, match="slice bounds"):
        scaling_sequence.slice(start, stop)


def test_window_sampler_enumerates_and_batches_all_windows(
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Verify the default sampler covers all valid windows in batches."""
    sampler = WindowSampler(
        scaling_sequence,
        window_length=3,
        batch_size=2,
        shuffle=False,
    )

    batches = list(sampler)

    assert sampler.num_windows == scaling_sequence.num_timesteps - 2
    assert [len(batch) for batch in batches] == [2, 1]
    assert all(window.num_timesteps == 3 for batch in batches for window in batch)
    assert torch.equal(batches[0][0][0].x, scaling_sequence[0].x)
    assert torch.equal(batches[0][1][0].x, scaling_sequence[1].x)


def test_window_sampler_pools_multiple_sequences(
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Verify valid windows are pooled across every source trajectory."""
    sampler = WindowSampler(
        [scaling_sequence, scaling_sequence],
        window_length=4,
        batch_size=10,
        shuffle=False,
    )

    assert sampler.num_windows == 4
    assert len(list(sampler)[0]) == 4


def test_window_sampler_seed_is_reproducible_per_epoch(
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Verify seeded shuffling is repeatable and changes between epochs."""
    sampler = WindowSampler(
        scaling_sequence,
        window_length=2,
        batch_size=1,
        seed=7,
    )

    epoch_zero = [batch[0][0].x for batch in sampler.iter_epoch(0)]
    repeated = [batch[0][0].x for batch in sampler.iter_epoch(0)]
    assert all(
        torch.equal(left, right)
        for left, right in zip(epoch_zero, repeated, strict=True)
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"window_length": 1}, "window_length"),
        ({"window_length": 2, "batch_size": 0}, "batch_size"),
        (
            {"window_length": 2, "windows_per_epoch": 0},
            "windows_per_epoch",
        ),
    ],
)
def test_window_sampler_rejects_invalid_configuration(
    scaling_sequence: GraphSnapshotSequence,
    kwargs: dict[str, int],
    message: str,
) -> None:
    """Verify invalid sampler arguments raise clear errors."""
    with pytest.raises(ValueError, match=message):
        WindowSampler(scaling_sequence, **kwargs)


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


def test_dynamic_topology_allowed_with_flag(
    synthetic_edge_index: torch.Tensor,
    make_snapshots: Callable[..., list[Data]],
) -> None:
    """Verify dynamic topology is accepted when explicitly enabled."""
    snapshots = make_snapshots(synthetic_edge_index)
    alt_edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    snapshots[1] = Data(x=torch.randn(5, 3), edge_index=alt_edges)
    sequence = GraphSnapshotSequence(snapshots, allow_dynamic_topology=True)

    assert sequence.is_dynamic_topology
    assert sequence.allow_dynamic_topology
    assert torch.equal(sequence[0].edge_index, synthetic_edge_index)
    assert torch.equal(sequence[1].edge_index, alt_edges)


def test_dynamic_topology_edge_index_property_raises(
    synthetic_edge_index: torch.Tensor,
    make_snapshots: Callable[..., list[Data]],
) -> None:
    """Verify shared topology properties are unavailable when dynamic."""
    snapshots = make_snapshots(synthetic_edge_index)
    snapshots[1] = Data(
        x=torch.randn(5, 3),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
    )
    sequence = GraphSnapshotSequence(snapshots, allow_dynamic_topology=True)

    with pytest.raises(ValueError, match="edge_index is undefined"):
        _ = sequence.edge_index
    with pytest.raises(ValueError, match="edge_weight is undefined"):
        _ = sequence.edge_weight


def test_dynamic_flag_without_topology_changes_is_not_dynamic(
    synthetic_edge_index: torch.Tensor,
    make_snapshots: Callable[..., list[Data]],
) -> None:
    """Verify identical topology keeps ``is_dynamic_topology`` false."""
    snapshots = make_snapshots(synthetic_edge_index)
    sequence = GraphSnapshotSequence(snapshots, allow_dynamic_topology=True)
    assert not sequence.is_dynamic_topology
    assert torch.equal(sequence.edge_index, synthetic_edge_index)


def test_from_dynamic_arrays_builds_sequence() -> None:
    """Verify per-timestep edge indices are attached correctly."""
    features = torch.randn(3, 4, 2)
    edge_indices = [
        torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        torch.tensor([[0, 2], [2, 0]], dtype=torch.long),
    ]
    sequence = GraphSnapshotSequence.from_dynamic_arrays(features, edge_indices)

    assert sequence.allow_dynamic_topology
    assert sequence.is_dynamic_topology
    assert sequence.num_timesteps == 3
    for t, edges in enumerate(edge_indices):
        assert torch.equal(sequence[t].edge_index, edges)


def test_from_dynamic_arrays_rejects_length_mismatch() -> None:
    """Verify edge index count must match timesteps."""
    features = torch.randn(2, 4, 2)
    edge_indices = [torch.tensor([[0, 1], [1, 0]], dtype=torch.long)]
    with pytest.raises(ValueError, match="edge_indices has length"):
        GraphSnapshotSequence.from_dynamic_arrays(features, edge_indices)


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


def test_temporal_split_default_ratios(
    synthetic_edge_index: torch.Tensor,
    make_snapshots: Callable[..., list[Data]],
) -> None:
    """Verify default temporal split produces contiguous partitions."""
    snapshots = make_snapshots(synthetic_edge_index, num_timesteps=20)
    sequence = GraphSnapshotSequence(snapshots)
    split = temporal_split(sequence)

    assert isinstance(split, TemporalSplit)
    assert split.train.num_timesteps == 14
    assert split.val.num_timesteps == 2
    assert split.test.num_timesteps == 4
    assert split.train[0] is snapshots[0]
    assert split.val[0] is snapshots[14]
    assert split.test[0] is snapshots[16]
    assert torch.equal(split.train.edge_index, sequence.edge_index)


def test_temporal_split_rejects_invalid_ratios(
    synthetic_edge_index: torch.Tensor,
    make_snapshots: Callable[..., list[Data]],
) -> None:
    """Verify invalid ratio sums raise ``ValueError``."""
    snapshots = make_snapshots(synthetic_edge_index, num_timesteps=10)
    sequence = GraphSnapshotSequence(snapshots)
    with pytest.raises(ValueError, match="must equal 1.0"):
        temporal_split(sequence, train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)


def test_temporal_split_rejects_short_sequence(
    synthetic_edge_index: torch.Tensor,
    make_snapshots: Callable[..., list[Data]],
) -> None:
    """Verify sequences that are too short raise ``ValueError``."""
    snapshots = make_snapshots(synthetic_edge_index, num_timesteps=4)
    sequence = GraphSnapshotSequence(snapshots)
    with pytest.raises(ValueError, match="needs at least"):
        temporal_split(sequence)


def test_from_arrays_with_edge_weight(synthetic_edge_index: torch.Tensor) -> None:
    """Verify ``from_arrays`` attaches shared edge weights to every snapshot."""
    features = torch.randn(3, 5, 2)
    edge_weight = torch.ones(synthetic_edge_index.shape[1], dtype=torch.float32)
    sequence = GraphSnapshotSequence.from_arrays(
        features,
        synthetic_edge_index,
        edge_weight=edge_weight,
    )
    assert sequence.edge_weight is not None
    assert torch.equal(sequence.edge_weight, edge_weight)
    for snapshot in sequence:
        assert torch.equal(snapshot.edge_weight, edge_weight)


def test_edge_weight_property_none_for_unweighted(
    synthetic_edge_index: torch.Tensor,
    make_snapshots: Callable[..., list[Data]],
) -> None:
    """Verify unweighted sequences expose ``edge_weight=None``."""
    sequence = GraphSnapshotSequence(make_snapshots(synthetic_edge_index))
    assert sequence.edge_weight is None


def test_rejects_mismatched_edge_weight(
    synthetic_edge_index: torch.Tensor,
    make_snapshots: Callable[..., list[Data]],
) -> None:
    """Verify inconsistent edge weights across snapshots raise ``ValueError``."""
    snapshots = make_snapshots(synthetic_edge_index)
    weight = torch.ones(synthetic_edge_index.shape[1], dtype=torch.float32)
    snapshots[0] = Data(
        x=snapshots[0].x,
        edge_index=synthetic_edge_index,
        edge_weight=weight,
    )
    snapshots[1] = Data(
        x=snapshots[1].x,
        edge_index=synthetic_edge_index,
        edge_weight=weight * 2,
    )
    with pytest.raises(ValueError, match="different edge_weight"):
        GraphSnapshotSequence(snapshots)


def test_rejects_partial_edge_weight_presence(
    synthetic_edge_index: torch.Tensor,
    make_snapshots: Callable[..., list[Data]],
) -> None:
    """Verify mixed weighted/unweighted snapshots raise ``ValueError``."""
    snapshots = make_snapshots(synthetic_edge_index)
    weight = torch.ones(synthetic_edge_index.shape[1], dtype=torch.float32)
    snapshots[0] = Data(
        x=snapshots[0].x,
        edge_index=synthetic_edge_index,
        edge_weight=weight,
    )
    with pytest.raises(ValueError, match="edge_weight presence"):
        GraphSnapshotSequence(snapshots)


def test_from_arrays_invalid_edge_weight_shape(
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Verify invalid edge weight shapes raise ``ValueError``."""
    features = torch.randn(2, 5, 3)
    with pytest.raises(ValueError, match="edge_weight must have shape"):
        GraphSnapshotSequence.from_arrays(
            features,
            synthetic_edge_index,
            edge_weight=torch.ones(2, 3),
        )
    with pytest.raises(ValueError, match="edge_weight length"):
        GraphSnapshotSequence.from_arrays(
            features,
            synthetic_edge_index,
            edge_weight=torch.ones(3),
        )
