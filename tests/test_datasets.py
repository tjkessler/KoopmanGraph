"""Tests for benchmark datasets."""

from pathlib import Path

import numpy as np
import pytest
import torch

from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.datasets import (
    AnisotropicAdvectionGridBenchmark,
    GridDynamicGraphBenchmark,
    IEEE118DynamicBenchmark,
    MetrLaTrafficBenchmark,
    SyntheticDynamicGraphBenchmark,
)
from koopman_graph.datasets.grid import _grid_edge_index, grid_node_index


def test_generate_returns_graph_snapshot_sequence() -> None:
    """Verify the benchmark generator returns a validated sequence."""
    sequence = SyntheticDynamicGraphBenchmark.generate(
        num_nodes=10,
        num_timesteps=12,
        in_channels=4,
        seed=0,
    )
    assert isinstance(sequence, GraphSnapshotSequence)
    assert sequence.num_nodes == 10
    assert sequence.num_timesteps == 12
    assert sequence.in_channels == 4
    assert sequence.edge_index.shape[0] == 2


def test_generate_is_reproducible_with_seed() -> None:
    """Verify identical seeds produce identical node features."""
    kwargs = {
        "num_nodes": 8,
        "num_timesteps": 6,
        "in_channels": 2,
        "noise_std": 0.05,
        "seed": 123,
    }
    first = SyntheticDynamicGraphBenchmark.generate(**kwargs)
    second = SyntheticDynamicGraphBenchmark.generate(**kwargs)
    for left, right in zip(first, second, strict=True):
        assert torch.equal(left.x, right.x)


def test_generate_path_topology_edge_count() -> None:
    """Verify path topology has two directed edges per adjacent pair."""
    sequence = SyntheticDynamicGraphBenchmark.generate(
        num_nodes=5,
        num_timesteps=3,
        topology="path",
        seed=0,
    )
    assert sequence.edge_index.shape[1] == 2 * (sequence.num_nodes - 1)


def test_generate_ring_topology_edge_count() -> None:
    """Verify ring topology has two directed edges per node."""
    sequence = SyntheticDynamicGraphBenchmark.generate(
        num_nodes=6,
        num_timesteps=3,
        topology="ring",
        seed=0,
    )
    assert sequence.edge_index.shape[1] == 2 * sequence.num_nodes


def test_advection_grid_generate_returns_graph_snapshot_sequence() -> None:
    """Verify the advection benchmark generator returns a validated sequence."""
    sequence = AnisotropicAdvectionGridBenchmark.generate(
        num_rows=4,
        num_cols=4,
        num_timesteps=10,
        seed=0,
    )
    assert isinstance(sequence, GraphSnapshotSequence)
    assert sequence.num_nodes == 16
    assert sequence.num_timesteps == 10


def test_advection_grid_rejects_invalid_weights() -> None:
    """Verify directional weights must leave mass for other neighbors."""
    with pytest.raises(ValueError, match="west_weight"):
        AnisotropicAdvectionGridBenchmark.generate(west_weight=0.8, north_weight=0.3)


def test_advection_grid_gat_beats_gcn_on_rollout() -> None:
    """Verify attention helps on asymmetric advection dynamics."""
    from koopman_graph import GATEncoder, GNNDecoder, GNNEncoder, GraphKoopmanModel
    from koopman_graph.training import constant_loss_weights

    sequence = AnisotropicAdvectionGridBenchmark.generate(
        num_rows=8,
        num_cols=8,
        num_timesteps=30,
        seed=0,
    )
    in_channels = sequence.in_channels
    hidden = 64
    latent = 64
    weights = constant_loss_weights(reconstruction=1.0, forward=1.0, rollout=2.0)
    fit_kwargs = {
        "epochs": 80,
        "lr": 1e-3,
        "loss_weights": weights,
        "rollout_horizon": 5,
        "max_grad_norm": 1.0,
    }

    gat_model = GraphKoopmanModel(
        GATEncoder(in_channels, hidden, latent, heads=4),
        GNNDecoder(latent, hidden, in_channels),
        latent_dim=latent,
        time_step=0.1,
    )
    gcn_model = GraphKoopmanModel(
        GNNEncoder(in_channels, hidden, latent),
        GNNDecoder(latent, hidden, in_channels),
        latent_dim=latent,
        time_step=0.1,
    )
    torch.manual_seed(0)
    gat_model.fit(sequence, **fit_kwargs)
    torch.manual_seed(0)
    gcn_model.fit(sequence, **fit_kwargs)

    gat_preds = gat_model.predict(sequence[0], steps=5)
    gcn_preds = gcn_model.predict(sequence[0], steps=5)
    ground_truth = sequence[1:6]
    gat_mse = torch.mean(
        torch.stack(
            [
                torch.mean((pred.x - truth.x) ** 2)
                for pred, truth in zip(gat_preds, ground_truth, strict=True)
            ]
        )
    )
    gcn_mse = torch.mean(
        torch.stack(
            [
                torch.mean((pred.x - truth.x) ** 2)
                for pred, truth in zip(gcn_preds, ground_truth, strict=True)
            ]
        )
    )
    assert gat_mse < gcn_mse


def test_grid_generate_returns_graph_snapshot_sequence() -> None:
    """Verify the grid benchmark generator returns a validated sequence."""
    sequence = GridDynamicGraphBenchmark.generate(
        num_rows=4,
        num_cols=5,
        num_timesteps=12,
        in_channels=2,
        seed=0,
    )
    assert isinstance(sequence, GraphSnapshotSequence)
    assert sequence.num_nodes == 20
    assert sequence.num_timesteps == 12
    assert sequence.in_channels == 2


def test_grid_edge_count_for_small_lattice() -> None:
    """Verify 4-connected grid has expected directed edge count."""
    sequence = GridDynamicGraphBenchmark.generate(
        num_rows=2,
        num_cols=3,
        num_timesteps=3,
        seed=0,
    )
    num_rows, num_cols = 2, 3
    horizontal = num_rows * (num_cols - 1) * 2
    vertical = (num_rows - 1) * num_cols * 2
    assert sequence.edge_index.shape[1] == horizontal + vertical


def test_grid_node_index_helper() -> None:
    """Verify flattened grid coordinates map to node indices."""
    assert grid_node_index(1, 2, num_cols=5) == 7


def test_grid_generate_is_reproducible_with_seed() -> None:
    """Verify identical seeds produce identical grid node features."""
    kwargs = {
        "num_rows": 3,
        "num_cols": 3,
        "num_timesteps": 6,
        "in_channels": 2,
        "seed": 11,
    }
    first = GridDynamicGraphBenchmark.generate(**kwargs)
    second = GridDynamicGraphBenchmark.generate(**kwargs)
    for left, right in zip(first, second, strict=True):
        assert torch.equal(left.x, right.x)


@pytest.mark.parametrize(
    ("param_name", "kwargs"),
    [
        ("num_rows", {"num_rows": 0}),
        ("num_cols", {"num_cols": 0}),
        ("num_timesteps", {"num_timesteps": 0}),
        ("diffusion_rate", {"diffusion_rate": 1.5}),
        ("initial_state", {"initial_state": "invalid"}),
    ],
)
def test_grid_generate_invalid_parameters_raise(
    param_name: str,
    kwargs: dict[str, object],
) -> None:
    """Verify invalid grid generation parameters raise clear errors."""
    with pytest.raises(ValueError, match=param_name):
        GridDynamicGraphBenchmark.generate(**kwargs)


def test_generate_zero_noise_is_deterministic_given_seed() -> None:
    """Verify zero noise with fixed seed yields repeatable dynamics."""
    kwargs = {
        "num_nodes": 6,
        "num_timesteps": 5,
        "in_channels": 3,
        "noise_std": 0.0,
        "seed": 7,
        "initial_state": "random",
    }
    first = SyntheticDynamicGraphBenchmark.generate(**kwargs)
    second = SyntheticDynamicGraphBenchmark.generate(**kwargs)
    assert torch.equal(first[2].x, second[2].x)


def test_generate_ones_initial_state() -> None:
    """Verify ones initial state starts from an all-ones feature matrix."""
    sequence = SyntheticDynamicGraphBenchmark.generate(
        num_nodes=4,
        num_timesteps=3,
        in_channels=2,
        initial_state="ones",
        noise_std=0.0,
        seed=0,
    )
    assert torch.allclose(sequence[0].x, torch.ones(4, 2))


@pytest.mark.parametrize(
    ("param_name", "kwargs"),
    [
        ("num_nodes", {"num_nodes": 0}),
        ("num_timesteps", {"num_timesteps": 0}),
        ("in_channels", {"in_channels": 0}),
        ("diffusion_rate", {"diffusion_rate": 1.5}),
        ("decay_rate", {"decay_rate": 0.0}),
        ("noise_std", {"noise_std": -0.1}),
        ("initial_state", {"initial_state": "invalid"}),
        ("topology", {"topology": "star"}),
    ],
)
def test_generate_invalid_parameters_raise(
    param_name: str,
    kwargs: dict[str, object],
) -> None:
    """Verify invalid generation parameters raise clear errors."""
    match_pattern = "topology" if param_name == "topology" else param_name
    with pytest.raises(ValueError, match=match_pattern):
        SyntheticDynamicGraphBenchmark.generate(**kwargs)


def test_ieee118_generate_returns_graph_snapshot_sequence() -> None:
    """Verify IEEE 118 benchmark returns a validated 118-bus sequence."""
    sequence = IEEE118DynamicBenchmark.generate(num_timesteps=6, seed=0)
    assert isinstance(sequence, GraphSnapshotSequence)
    assert sequence.num_nodes == IEEE118DynamicBenchmark.NUM_BUSES
    assert sequence.num_timesteps == 6
    assert sequence.in_channels == IEEE118DynamicBenchmark.IN_CHANNELS
    assert sequence.edge_index.shape[0] == 2


def test_ieee118_generate_is_reproducible_with_seed() -> None:
    """Verify identical seeds produce identical IEEE 118 node features."""
    kwargs = {"num_timesteps": 8, "noise_std": 0.01, "seed": 7}
    first = IEEE118DynamicBenchmark.generate(**kwargs)
    second = IEEE118DynamicBenchmark.generate(**kwargs)
    for left, right in zip(first, second, strict=True):
        assert torch.equal(left.x, right.x)


def test_ieee118_load_topology_has_expected_fields() -> None:
    """Verify cached topology exposes bus tables and edge index."""
    topology = IEEE118DynamicBenchmark.load_topology()
    assert topology["num_nodes"] == IEEE118DynamicBenchmark.NUM_BUSES
    assert topology["initial_features"].shape == (118, 4)
    assert topology["edge_index"].dtype == torch.long


@pytest.mark.parametrize(
    ("param_name", "kwargs"),
    [
        ("num_timesteps", {"num_timesteps": 0}),
        ("diffusion_rate", {"diffusion_rate": 1.5}),
        ("decay_rate", {"decay_rate": 0.0}),
        ("noise_std", {"noise_std": -0.1}),
        ("load_ramp_amplitude", {"load_ramp_amplitude": -0.1}),
        ("load_ramp_period", {"load_ramp_period": 0.0}),
    ],
)
def test_ieee118_generate_invalid_parameters_raise(
    param_name: str,
    kwargs: dict[str, object],
) -> None:
    """Verify invalid IEEE 118 generation parameters raise clear errors."""
    with pytest.raises(ValueError, match=param_name):
        IEEE118DynamicBenchmark.generate(**kwargs)


def test_metr_la_load_sequence_returns_graph_snapshot_sequence() -> None:
    """Verify METR-LA benchmark returns a validated 207-sensor sequence."""
    sequence = MetrLaTrafficBenchmark.load_sequence()
    assert isinstance(sequence, GraphSnapshotSequence)
    assert sequence.num_nodes == MetrLaTrafficBenchmark.NUM_SENSORS
    assert sequence.in_channels == MetrLaTrafficBenchmark.IN_CHANNELS
    assert sequence.num_timesteps >= 2
    assert sequence.edge_index.shape[0] == 2


def test_metr_la_load_sequence_is_reproducible() -> None:
    """Verify repeated cache loads produce identical node features."""
    first = MetrLaTrafficBenchmark.load_sequence()
    second = MetrLaTrafficBenchmark.load_sequence()
    for left, right in zip(first, second, strict=True):
        assert torch.equal(left.x, right.x)


def test_metr_la_load_topology_has_expected_fields() -> None:
    """Verify cached METR-LA topology exposes sensor IDs and edge index."""
    topology = MetrLaTrafficBenchmark.load_topology()
    assert topology["num_nodes"] == MetrLaTrafficBenchmark.NUM_SENSORS
    assert len(topology["sensor_ids"]) == MetrLaTrafficBenchmark.NUM_SENSORS
    assert topology["edge_index"].dtype == torch.long


def test_metr_la_build_adjacency_matrix_shape() -> None:
    """Verify adjacency construction yields a square matrix for sensor IDs."""
    from koopman_graph.datasets.metr_la import build_adjacency_matrix

    sensor_ids = ["a", "b", "c"]
    distance_csv = "from,to,cost\na,a,0.0\na,b,10.0\nb,b,0.0\n"
    adj = build_adjacency_matrix(distance_csv, sensor_ids, normalized_k=0.0)
    assert adj.shape == (3, 3)
    assert adj[0, 1] > 0.0


@pytest.mark.parametrize(
    ("param_name", "kwargs"),
    [
        ("normalized_k", {"normalized_k": 1.5}),
    ],
)
def test_metr_la_invalid_parameters_raise(
    param_name: str,
    kwargs: dict[str, object],
) -> None:
    """Verify invalid METR-LA helper parameters raise clear errors."""
    from koopman_graph.datasets.metr_la import build_adjacency_matrix

    with pytest.raises(ValueError, match=param_name):
        build_adjacency_matrix("from,to,cost\n", ["a"], **kwargs)


def test_metr_la_preprocess_speeds_fills_missing_zeros() -> None:
    """Verify zero readings are imputed before normalization."""
    from koopman_graph.datasets.metr_la import normalize_speeds, preprocess_speeds

    speeds = np.full((6, 3), 60.0, dtype=np.float32)
    speeds[2:4, :] = 0.0
    speeds[5, 1] = 0.0

    cleaned = preprocess_speeds(speeds)
    assert np.all(cleaned[2:4, :] == 60.0)
    assert cleaned[5, 1] == 60.0

    normalized = normalize_speeds(cleaned)
    assert normalized[2:4].min() > -3.0
    assert normalized[2:4].max() < 3.0


def test_generate_single_node_path_has_no_edges() -> None:
    """Verify a one-node path graph has an empty edge index."""
    sequence = SyntheticDynamicGraphBenchmark.generate(
        num_nodes=1,
        num_timesteps=2,
        topology="path",
        seed=0,
    )
    assert sequence.num_nodes == 1
    assert sequence.edge_index.shape[1] == 0


def test_generate_single_node_ring_has_no_edges() -> None:
    """Verify a one-node ring graph has an empty edge index."""
    sequence = SyntheticDynamicGraphBenchmark.generate(
        num_nodes=1,
        num_timesteps=2,
        topology="ring",
        seed=0,
    )
    assert sequence.num_nodes == 1
    assert sequence.edge_index.shape[1] == 0


def test_grid_edge_index_empty_for_invalid_dimensions() -> None:
    """Verify invalid lattice dimensions yield an empty edge index."""
    edge_index = _grid_edge_index(0, 5)
    assert edge_index.shape == (2, 0)


@pytest.mark.parametrize(
    ("param_name", "kwargs"),
    [
        ("in_channels", {"in_channels": 0}),
        ("decay_rate", {"decay_rate": 0.0}),
        ("noise_std", {"noise_std": -0.1}),
    ],
)
def test_grid_generate_additional_invalid_parameters_raise(
    param_name: str,
    kwargs: dict[str, object],
) -> None:
    """Verify additional invalid grid parameters raise clear errors."""
    with pytest.raises(ValueError, match=param_name):
        GridDynamicGraphBenchmark.generate(**kwargs)


@pytest.mark.parametrize(
    ("param_name", "kwargs"),
    [
        ("num_rows", {"num_rows": 0}),
        ("num_cols", {"num_cols": 0}),
        ("num_timesteps", {"num_timesteps": 0}),
        ("in_channels", {"in_channels": 0}),
        ("decay_rate", {"decay_rate": 1.0}),
        ("decay_rate", {"decay_rate": 0.0}),
        ("west_weight", {"west_weight": -0.1}),
        ("initial_state", {"initial_state": "invalid"}),
    ],
)
def test_advection_grid_invalid_parameters_raise(
    param_name: str,
    kwargs: dict[str, object],
) -> None:
    """Verify invalid advection benchmark parameters raise clear errors."""
    with pytest.raises(ValueError, match=param_name):
        AnisotropicAdvectionGridBenchmark.generate(**kwargs)


def test_advection_grid_single_node_corner_has_no_neighbors() -> None:
    """Verify a 1x1 advection grid still produces a valid sequence."""
    sequence = AnisotropicAdvectionGridBenchmark.generate(
        num_rows=1,
        num_cols=1,
        num_timesteps=3,
        seed=0,
    )
    assert sequence.num_nodes == 1
    assert sequence.num_timesteps == 3


def test_metr_la_read_h5_speed_window_invalid_window(tmp_path: Path) -> None:
    """Verify invalid HDF5 window parameters raise clear errors."""
    h5py = pytest.importorskip("h5py")

    h5_path = tmp_path / "metr-la.h5"
    with h5py.File(h5_path, "w") as handle:
        group = handle.create_group("df")
        group.create_dataset("block0_values", data=np.zeros((5, 3), dtype=np.float32))

    from koopman_graph.datasets.metr_la import read_h5_speed_window

    with pytest.raises(ValueError, match="num_timesteps"):
        read_h5_speed_window(h5_path, num_timesteps=0)
    with pytest.raises(ValueError, match="offset"):
        read_h5_speed_window(h5_path, num_timesteps=2, offset=-1)
    with pytest.raises(ValueError, match="exceeds available rows"):
        read_h5_speed_window(h5_path, num_timesteps=10, offset=0)
