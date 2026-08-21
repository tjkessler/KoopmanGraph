"""Tests for homogeneous regime coordinates and ConditioningContext."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch
from torch_geometric.data import Data, HeteroData

import koopman_graph
from koopman_graph.data import (
    ConditioningContext,
    GraphSnapshotSequence,
    HeteroGraphSnapshotSequence,
    collate_graph_trajectories,
    conditioning_at,
    diurnal_control_features,
    diurnal_phase_index,
    temporal_split,
)
from koopman_graph.data.sampling import induce_neighbor_subgraph_sequence
from koopman_graph.model import GraphKoopmanModel
from koopman_graph.nn import GNNDecoder, GNNEncoder
from koopman_graph.training.device import sequence_to_device


def _path_edge_index(num_nodes: int) -> torch.Tensor:
    """Return a bidirectional path graph.

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


def _features(
    *,
    timesteps: int = 10,
    nodes: int = 4,
    channels: int = 2,
) -> torch.Tensor:
    """Return a deterministic feature cube.

    Parameters
    ----------
    timesteps, nodes, channels : int
        Cube dimensions.

    Returns
    -------
    Tensor
        Features with shape ``(T, N, F)``.
    """
    return torch.arange(timesteps * nodes * channels, dtype=torch.float32).reshape(
        timesteps, nodes, channels
    )


def test_from_arrays_round_trip_slice_split_and_windowed() -> None:
    """parameter_trajectory survives from_arrays, slice, split, and windowed."""
    features = _features()
    edge_index = _path_edge_index(4)
    mu = torch.stack(
        [torch.tensor([0.1 * t, 1.0 + 0.1 * t], dtype=torch.float32) for t in range(10)]
    )
    times = torch.arange(10, dtype=torch.float32)
    controls = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    sequence = GraphSnapshotSequence.from_arrays(
        features,
        edge_index,
        timestamps=times,
        parameter_trajectory=mu,
        control_inputs=controls,
    )
    assert sequence.has_parameter_trajectory
    torch.testing.assert_close(sequence.parameter_trajectory, mu)

    window = sequence.slice(1, 4)
    torch.testing.assert_close(window.parameter_trajectory, mu[1:4])
    torch.testing.assert_close(window.timestamps, times[1:4])
    torch.testing.assert_close(window.control_inputs, controls[1:4])

    split = temporal_split(sequence)
    assert split.train.parameter_trajectory is not None
    torch.testing.assert_close(
        split.train.parameter_trajectory,
        mu[: split.train.num_timesteps],
    )
    assert split.val.parameter_trajectory is not None
    train_end = split.train.num_timesteps
    val_end = train_end + split.val.num_timesteps
    torch.testing.assert_close(split.val.parameter_trajectory, mu[train_end:val_end])
    assert split.test.parameter_trajectory is not None
    torch.testing.assert_close(split.test.parameter_trajectory, mu[val_end:])

    stacked = sequence.windowed(n_delays=2, stride=2, pad=True)
    torch.testing.assert_close(stacked.parameter_trajectory, mu[[0, 2, 4, 6, 8]])
    torch.testing.assert_close(stacked.timestamps, times[[0, 2, 4, 6, 8]])


def test_from_dynamic_arrays_preserves_parameters() -> None:
    """Dynamic-topology arrays keep the aligned mu rows."""
    features = _features(timesteps=3, nodes=3, channels=1)
    edges = [_path_edge_index(3) for _ in range(3)]
    mu = torch.tensor([[1.0], [2.0], [3.0]])
    sequence = GraphSnapshotSequence.from_dynamic_arrays(
        features,
        edges,
        parameter_trajectory=mu,
    )
    assert sequence.allow_dynamic_topology
    torch.testing.assert_close(sequence.parameter_trajectory, mu)


def test_conditioning_at_fills_time_parameters_and_control() -> None:
    """conditioning_at copies time, mu, and control; phase_index stays None."""
    features = _features(timesteps=3, nodes=3, channels=2)
    sequence = GraphSnapshotSequence.from_arrays(
        features,
        _path_edge_index(3),
        timestamps=torch.tensor([10.0, 20.0, 30.0]),
        parameter_trajectory=torch.tensor([[0.5, -1.0], [1.5, -2.0], [2.5, -3.0]]),
        control_inputs=torch.arange(6, dtype=torch.float32).reshape(3, 2),
    )
    ctx = conditioning_at(sequence, 1)
    assert ctx.time == 20.0
    assert ctx.parameters is not None
    torch.testing.assert_close(ctx.parameters, torch.tensor([1.5, -2.0]))
    assert ctx.control is not None
    torch.testing.assert_close(ctx.control, torch.tensor([2.0, 3.0]))
    assert ctx.phase_index is None


def test_conditioning_at_all_none_is_valid() -> None:
    """Missing timestamps, mu, and controls yield an empty record."""
    features = _features(timesteps=2, nodes=3, channels=1)
    sequence = GraphSnapshotSequence.from_arrays(features, _path_edge_index(3))
    ctx = conditioning_at(sequence, 0)
    assert ctx.time is None
    assert ctx.parameters is None
    assert ctx.control is None
    assert ctx.phase_index is None


def test_conditioning_at_per_node_control_keeps_node_rows() -> None:
    """Per-node controls at an index have shape (N, C)."""
    features = _features(timesteps=2, nodes=3, channels=1)
    controls = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
    sequence = GraphSnapshotSequence.from_arrays(
        features,
        _path_edge_index(3),
        control_inputs=controls,
    )
    ctx = conditioning_at(sequence, 1)
    assert ctx.control is not None
    torch.testing.assert_close(ctx.control, controls[1])


def test_conditioning_at_optional_phase_index_from_timestamps() -> None:
    """period and num_phases fill phase_index; default stays None."""
    features = _features(timesteps=3, nodes=3, channels=1)
    sequence = GraphSnapshotSequence.from_arrays(
        features,
        _path_edge_index(3),
        timestamps=torch.tensor([0.0, 6.0, 12.0]),
    )
    default = conditioning_at(sequence, 1)
    assert default.phase_index is None
    filled = conditioning_at(sequence, 1, period=24.0, num_phases=4)
    assert filled.phase_index == 1
    with pytest.raises(ValueError, match="together"):
        conditioning_at(sequence, 1, period=24.0)
    with pytest.raises(ValueError, match="together"):
        conditioning_at(sequence, 1, num_phases=4)
    no_time = GraphSnapshotSequence.from_arrays(features, _path_edge_index(3))
    with pytest.raises(ValueError, match="timestamps"):
        conditioning_at(no_time, 0, period=24.0, num_phases=4)


def test_conditioning_at_rejects_bad_index_and_hetero() -> None:
    """Out-of-range indices and hetero sequences are refused."""
    features = _features(timesteps=2, nodes=3, channels=1)
    sequence = GraphSnapshotSequence.from_arrays(features, _path_edge_index(3))
    with pytest.raises(ValueError, match="out of range"):
        conditioning_at(sequence, 2)
    hetero = HeteroData()
    hetero["node"].x = torch.randn(3, 1)
    hetero["node", "r", "node"].edge_index = _path_edge_index(3)
    hetero_seq = HeteroGraphSnapshotSequence([hetero, hetero])
    with pytest.raises(TypeError, match="GraphSnapshotSequence"):
        conditioning_at(hetero_seq, 0)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad, match",
    [
        (torch.arange(10, dtype=torch.float32), "shape"),
        (torch.zeros(10, 0), "d_mu"),
        (torch.zeros(5, 2), "timesteps"),
        (torch.tensor([[1.0, float("nan")]]), "finite"),
    ],
)
def test_parameter_trajectory_rejects_invalid_tensors(
    bad: torch.Tensor,
    match: str,
) -> None:
    """Invalid mu stacks raise with a shape or finiteness message."""
    features = _features(timesteps=10, nodes=3, channels=1)
    if bad.shape[0] == 1:
        features = _features(timesteps=1, nodes=3, channels=1)
    with pytest.raises(ValueError, match=match):
        GraphSnapshotSequence.from_arrays(
            features,
            _path_edge_index(3),
            parameter_trajectory=bad,
        )


def test_parameter_trajectory_rejects_bool_dtype() -> None:
    """Boolean mu stacks are refused at the sequence boundary."""
    snapshots = [
        Data(x=torch.randn(3, 1), edge_index=_path_edge_index(3)) for _ in range(2)
    ]
    with pytest.raises(ValueError, match="real numeric"):
        GraphSnapshotSequence(
            snapshots,
            parameter_trajectory=torch.zeros(2, 1, dtype=torch.bool),
        )


def test_conditioning_context_validates_fields() -> None:
    """Frozen context rejects non-finite time, empty mu, and bad phase."""
    with pytest.raises(ValueError, match="finite float"):
        ConditioningContext(time=float("inf"), parameters=None, control=None)
    with pytest.raises(ValueError, match="d_mu"):
        ConditioningContext(
            time=None,
            parameters=torch.tensor([]),
            control=None,
        )
    with pytest.raises(ValueError, match="phase_index"):
        ConditioningContext(time=None, parameters=None, control=None, phase_index=-1)
    with pytest.raises(FrozenInstanceError):
        ctx = ConditioningContext(time=1.0, parameters=None, control=None)
        ctx.time = 2.0  # type: ignore[misc]


def test_collate_rejects_parameter_trajectory() -> None:
    """Batched collate refuses regime coordinates, like timestamps."""
    features = _features(timesteps=2, nodes=3, channels=1)
    sequence = GraphSnapshotSequence.from_arrays(
        features,
        _path_edge_index(3),
        parameter_trajectory=torch.zeros(2, 1),
    )
    with pytest.raises(ValueError, match="parameter_trajectory"):
        collate_graph_trajectories((sequence,))


def test_sequence_to_device_preserves_parameter_trajectory() -> None:
    """Device copy keeps mu on the target device."""
    features = _features(timesteps=2, nodes=3, channels=1)
    mu = torch.tensor([[0.25], [0.75]])
    sequence = GraphSnapshotSequence.from_arrays(
        features,
        _path_edge_index(3),
        parameter_trajectory=mu,
    )
    moved = sequence_to_device(sequence, torch.device("cpu"))
    assert moved.parameter_trajectory is not None
    torch.testing.assert_close(moved.parameter_trajectory, mu)
    assert moved.parameter_trajectory.device.type == "cpu"


def test_neighbor_subgraph_copies_global_parameter_trajectory() -> None:
    """Induced subgraphs copy mu in full; they do not subset columns."""
    features = _features(timesteps=3, nodes=5, channels=1)
    mu = torch.tensor([[0.0, 9.0], [1.0, 8.0], [2.0, 7.0]])
    sequence = GraphSnapshotSequence.from_arrays(
        features,
        _path_edge_index(5),
        parameter_trajectory=mu,
    )
    induced = induce_neighbor_subgraph_sequence(
        sequence,
        seed_nodes=torch.tensor([0, 1], dtype=torch.long),
        num_hops=1,
    )
    torch.testing.assert_close(induced.parameter_trajectory, mu)
    assert induced.num_nodes < sequence.num_nodes


def test_factory_builds_parametric_kind() -> None:
    """switched/mixture stay LTI banks; parametric is the interpolant kind."""
    encoder = GNNEncoder(2, 4, 2)
    decoder = GNNDecoder(2, 4, 2)
    model = GraphKoopmanModel(
        encoder,
        decoder,
        latent_dim=2,
        time_step=0.1,
        koopman="parametric",
    )
    assert model.koopman_kind == "parametric"


def test_conditioning_symbols_are_not_on_root_all() -> None:
    """Q6: ConditioningContext stays off the root façade."""
    assert "ConditioningContext" not in koopman_graph.__all__
    assert "conditioning_at" not in koopman_graph.__all__
    assert "parameter_trajectory" not in koopman_graph.__all__
    assert "diurnal_control_features" not in koopman_graph.__all__
    assert "diurnal_phase_index" not in koopman_graph.__all__
    assert callable(diurnal_control_features)
    assert callable(diurnal_phase_index)
