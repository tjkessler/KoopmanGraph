"""Tests for benchmark datasets."""

import inspect
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest
import torch

from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.datasets import (
    AnisotropicAdvectionGridBenchmark,
    CylinderWakeBenchmark,
    EpidemicNetworkBenchmark,
    GridDynamicGraphBenchmark,
    IEEE118DynamicBenchmark,
    KuramotoSivashinskyBenchmark,
    Lorenz96GraphBenchmark,
    MetrLaTrafficBenchmark,
    SyntheticDynamicGraphBenchmark,
    TopologyPayload,
)
from koopman_graph.datasets import kuramoto_sivashinsky as ks_mod
from koopman_graph.datasets import lorenz96 as lorenz96_mod
from koopman_graph.datasets import synthetic as synthetic_mod
from koopman_graph.datasets.dynamics import (
    validate_advection_decay_rate,
    validate_diffusion_generation_params,
)
from koopman_graph.datasets.grid import (
    _grid_edge_index,
    anisotropic_advection_step,
    grid_node_index,
)
from koopman_graph.datasets.topology import path_edge_index, ring_edge_index


def test_path_ring_edge_builders_are_shared() -> None:
    """Verify synthetic and chaotic peers reuse the same topology builders."""
    assert synthetic_mod.path_edge_index is path_edge_index
    assert synthetic_mod.ring_edge_index is ring_edge_index
    assert lorenz96_mod.ring_edge_index is ring_edge_index
    assert ks_mod.path_edge_index is path_edge_index
    assert ks_mod.ring_edge_index is ring_edge_index
    assert not hasattr(synthetic_mod, "_path_edge_index")
    assert not hasattr(synthetic_mod, "_ring_edge_index")
    assert not hasattr(lorenz96_mod, "_path_edge_index")
    assert not hasattr(lorenz96_mod, "_ring_edge_index")
    assert not hasattr(ks_mod, "_path_edge_index")
    assert not hasattr(ks_mod, "_ring_edge_index")

    assert path_edge_index(1).shape == (2, 0)
    assert ring_edge_index(1).shape == (2, 0)
    assert path_edge_index(5).shape == (2, 8)
    assert ring_edge_index(5).shape == (2, 10)
    assert torch.equal(path_edge_index(4), synthetic_mod._build_topology("path", 4))
    assert torch.equal(ring_edge_index(4), synthetic_mod._build_topology("ring", 4))


def test_retired_nonlinear_module_unavailable() -> None:
    """Verify the hard migration removed datasets.nonlinear."""
    import importlib

    import koopman_graph.datasets as datasets_pkg

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("koopman_graph.datasets.nonlinear")

    assert datasets_pkg.EpidemicNetworkBenchmark is EpidemicNetworkBenchmark
    assert datasets_pkg.Lorenz96GraphBenchmark is Lorenz96GraphBenchmark
    assert datasets_pkg.KuramotoSivashinskyBenchmark is KuramotoSivashinskyBenchmark
    assert datasets_pkg.CylinderWakeBenchmark is CylinderWakeBenchmark
    for name in (
        "EpidemicNetworkBenchmark",
        "Lorenz96GraphBenchmark",
        "KuramotoSivashinskyBenchmark",
        "CylinderWakeBenchmark",
    ):
        assert name in datasets_pkg.__all__


@pytest.mark.parametrize(
    "benchmark_cls",
    [
        SyntheticDynamicGraphBenchmark,
        GridDynamicGraphBenchmark,
        AnisotropicAdvectionGridBenchmark,
        IEEE118DynamicBenchmark,
        EpidemicNetworkBenchmark,
        Lorenz96GraphBenchmark,
        KuramotoSivashinskyBenchmark,
    ],
)
def test_generate_defaults_seed_to_none(benchmark_cls: type) -> None:
    """Verify simulated generate() methods share an unseeded default."""
    assert inspect.signature(benchmark_cls.generate).parameters["seed"].default is None


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"decay_rate": 0.0, "noise_std": 0.0}, "decay_rate must be > 0"),
        ({"decay_rate": 0.5, "noise_std": -0.1}, "noise_std must be >= 0"),
        (
            {"decay_rate": 0.5, "noise_std": 0.0, "diffusion_rate": 1.5},
            r"diffusion_rate must be in \[0, 1\]",
        ),
        (
            {"decay_rate": 0.5, "noise_std": 0.0, "initial_state": "invalid"},
            "initial_state must be 'random' or 'ones'",
        ),
    ],
)
def test_validate_diffusion_generation_params_raises(
    kwargs: dict[str, object],
    match: str,
) -> None:
    """Verify shared diffusion validators raise explicit range errors."""
    with pytest.raises(ValueError, match=match):
        validate_diffusion_generation_params(**kwargs)  # type: ignore[arg-type]


def test_validate_diffusion_allows_decay_rate_at_or_above_one() -> None:
    """Verify diffusion decay_rate > 0 accepts values outside (0, 1)."""
    validate_diffusion_generation_params(decay_rate=1.0, noise_std=0.0)
    validate_diffusion_generation_params(decay_rate=1.5, noise_std=0.0)


@pytest.mark.parametrize("decay_rate", [0.0, 1.0, -0.1])
def test_validate_advection_decay_rate_raises(decay_rate: float) -> None:
    """Verify advection self-retention uses the open (0, 1) interval."""
    with pytest.raises(ValueError, match=r"decay_rate must be in \(0, 1\)"):
        validate_advection_decay_rate(decay_rate)


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


def test_advection_grid_one_step_matches_closed_form_mixture() -> None:
    """Verify one advection step against hand-computed 2x2 neighbor mixtures."""
    decay_rate = 0.8
    west_weight = 0.5
    north_weight = 0.2
    remaining = 1.0 - west_weight - north_weight
    sequence = AnisotropicAdvectionGridBenchmark.generate(
        num_rows=2,
        num_cols=2,
        num_timesteps=2,
        in_channels=1,
        decay_rate=decay_rate,
        west_weight=west_weight,
        north_weight=north_weight,
        noise_std=0.0,
        seed=0,
        initial_state="random",
    )
    state = sequence[0].x
    # Node layout: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3.
    # Border preferred directions reserve weight from the leftover budget but
    # are not assigned when the neighbor is missing.
    mix0 = 0.5 * state[1] + 0.5 * state[2]  # east+south share remaining only
    mix1 = (west_weight * state[0] + remaining * state[3]) / (west_weight + remaining)
    mix2 = (north_weight * state[0] + remaining * state[3]) / (north_weight + remaining)
    mix3 = (west_weight * state[2] + north_weight * state[1]) / (
        west_weight + north_weight
    )
    expected = decay_rate * state + (1.0 - decay_rate) * torch.stack(
        [mix0, mix1, mix2, mix3]
    )
    assert torch.allclose(sequence[1].x, expected, atol=1e-6)
    assert torch.allclose(
        anisotropic_advection_step(
            state,
            num_rows=2,
            num_cols=2,
            decay_rate=decay_rate,
            west_weight=west_weight,
            north_weight=north_weight,
        ),
        expected,
        atol=1e-6,
    )


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


def test_advection_zero_preferred_weights_keeps_self_retention_without_nan() -> None:
    """Corner nodes with zero preferred weights must not NaN; pure decay applies."""
    decay_rate = 0.75
    sequence = AnisotropicAdvectionGridBenchmark.generate(
        num_rows=2,
        num_cols=2,
        num_timesteps=2,
        in_channels=1,
        decay_rate=decay_rate,
        west_weight=0.0,
        north_weight=0.0,
        noise_std=0.0,
        seed=0,
        initial_state="random",
    )
    assert torch.isfinite(sequence[1].x).all()
    # SE corner (node 3) only has west+north neighbors; both weights are 0, so
    # the assigned-weight sum is zero and the update is pure self-retention.
    expected_corner = decay_rate * sequence[0].x[3]
    assert torch.allclose(sequence[1].x[3], expected_corner, atol=1e-6)


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


def test_advection_grid_diffconv_beats_gcn_on_rollout() -> None:
    """Verify DiffConv beats GCN on asymmetric advection at matched capacity."""
    from koopman_graph import (
        DiffConvDecoder,
        DiffConvEncoder,
        GNNDecoder,
        GNNEncoder,
        GraphKoopmanModel,
    )
    from koopman_graph.training import constant_loss_weights

    sequence = AnisotropicAdvectionGridBenchmark.generate(
        num_rows=8,
        num_cols=8,
        num_timesteps=30,
        seed=0,
    )
    in_channels = sequence.in_channels
    hidden = 32
    latent = 32
    weights = constant_loss_weights(reconstruction=1.0, forward=1.0, rollout=2.0)
    fit_kwargs = {
        "epochs": 60,
        "lr": 1e-3,
        "loss_weights": weights,
        "rollout_horizon": 5,
        "max_grad_norm": 1.0,
    }

    diff_model = GraphKoopmanModel(
        DiffConvEncoder(
            in_channels,
            hidden,
            latent,
            diffusion_steps=2,
        ),
        DiffConvDecoder(
            latent,
            hidden,
            in_channels,
            diffusion_steps=2,
        ),
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
    diff_model.fit(sequence, **fit_kwargs)
    torch.manual_seed(0)
    gcn_model.fit(sequence, **fit_kwargs)

    diff_preds = diff_model.predict(sequence[0], steps=5)
    gcn_preds = gcn_model.predict(sequence[0], steps=5)
    ground_truth = sequence[1:6]
    diff_mse = torch.mean(
        torch.stack(
            [
                torch.mean((pred.x - truth.x) ** 2)
                for pred, truth in zip(diff_preds, ground_truth, strict=True)
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
    assert diff_mse < gcn_mse


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
    assert isinstance(topology, TopologyPayload)
    assert topology.num_nodes == IEEE118DynamicBenchmark.NUM_BUSES
    assert topology["num_nodes"] == IEEE118DynamicBenchmark.NUM_BUSES
    assert topology.initial_features is not None
    assert topology.initial_features.shape == (118, 4)
    assert topology["initial_features"].shape == (118, 4)
    assert topology.edge_index.dtype == torch.long
    with pytest.raises(FrozenInstanceError):
        topology.num_nodes = 0  # type: ignore[misc]


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


def test_ieee118_generate_accepts_decay_rate_above_one() -> None:
    """Verify IEEE 118 uses shared diffusion decay_rate > 0 (not (0, 1))."""
    sequence = IEEE118DynamicBenchmark.generate(
        num_timesteps=3,
        decay_rate=1.0,
        noise_std=0.0,
        seed=0,
    )
    assert sequence.num_timesteps == 3


def test_advection_rejects_decay_rate_one_with_open_interval_message() -> None:
    """Verify advection rejects decay_rate=1.0 via the named open-interval validator."""
    with pytest.raises(ValueError, match=r"decay_rate must be in \(0, 1\)"):
        AnisotropicAdvectionGridBenchmark.generate(decay_rate=1.0)


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
    assert isinstance(topology, TopologyPayload)
    assert topology.num_nodes == MetrLaTrafficBenchmark.NUM_SENSORS
    assert topology.sensor_ids is not None
    assert len(topology.sensor_ids) == MetrLaTrafficBenchmark.NUM_SENSORS
    assert len(topology["sensor_ids"]) == MetrLaTrafficBenchmark.NUM_SENSORS
    assert topology.edge_index.dtype == torch.long
    assert "initial_features" not in topology


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


def test_epidemic_network_generate_shapes_and_simplex() -> None:
    """Verify SIR features stay on the simplex and infection spreads."""
    sequence = EpidemicNetworkBenchmark.generate(
        num_nodes=20,
        num_timesteps=25,
        topology="ring",
        seed=0,
    )
    assert sequence.num_nodes == 20
    assert sequence.in_channels == EpidemicNetworkBenchmark.IN_CHANNELS
    assert sequence.num_timesteps == 25
    for snapshot in sequence:
        totals = snapshot.x.sum(dim=1)
        assert torch.allclose(totals, torch.ones_like(totals), atol=1e-5)
        assert torch.all(snapshot.x >= -1e-6)
    assert float(sequence[12].x[:, 1].mean()) > float(sequence[0].x[:, 1].mean())


def test_epidemic_network_seed_determinism_and_topologies() -> None:
    """Verify fixed-seed identity and small-world / custom topologies."""
    kwargs = {
        "num_nodes": 18,
        "num_timesteps": 12,
        "topology": "small_world",
        "seed": 11,
    }
    first = EpidemicNetworkBenchmark.generate(**kwargs)
    second = EpidemicNetworkBenchmark.generate(**kwargs)
    for left, right in zip(first, second, strict=True):
        assert torch.equal(left.x, right.x)
        assert torch.equal(left.edge_index, right.edge_index)

    custom_edges = first[0].edge_index.clone()
    custom = EpidemicNetworkBenchmark.generate(
        num_nodes=18,
        num_timesteps=5,
        topology="custom",
        edge_index=custom_edges,
        seed=0,
    )
    assert custom.num_nodes == 18
    assert torch.equal(custom[0].edge_index, custom_edges)


def test_epidemic_network_rejects_invalid_params() -> None:
    """Verify epidemic generator validates topology and rates."""
    with pytest.raises(ValueError, match="custom"):
        EpidemicNetworkBenchmark.generate(topology="custom")
    with pytest.raises(ValueError, match="beta"):
        EpidemicNetworkBenchmark.generate(beta=-0.1)
    with pytest.raises(ValueError, match="Unsupported topology"):
        EpidemicNetworkBenchmark.generate(topology="lattice")  # type: ignore[arg-type]


def test_epidemic_intervention_controls_and_effect() -> None:
    """Verify contact-reduction controls attach and suppress infection."""
    baseline = EpidemicNetworkBenchmark.generate(
        num_nodes=24,
        num_timesteps=40,
        topology="ring",
        seed=2,
    )
    zero_u = EpidemicNetworkBenchmark.generate(
        num_nodes=24,
        num_timesteps=40,
        topology="ring",
        seed=2,
        expose_intervention_control=True,
        intervention=torch.zeros(40, 1),
    )
    for left, right in zip(baseline, zero_u, strict=True):
        assert torch.allclose(left.x, right.x, atol=1e-6)
    assert zero_u.has_controls
    assert zero_u.control_dim == 1
    assert zero_u.control_inputs is not None
    assert zero_u.control_inputs.shape == (40, 1)

    strong = EpidemicNetworkBenchmark.generate(
        num_nodes=24,
        num_timesteps=40,
        topology="ring",
        seed=2,
        expose_intervention_control=True,
        intervention=torch.full((40, 1), 0.85),
    )
    peak_base = max(float(snapshot.x[:, 1].mean()) for snapshot in baseline)
    peak_strong = max(float(snapshot.x[:, 1].mean()) for snapshot in strong)
    assert peak_strong < peak_base

    default = EpidemicNetworkBenchmark.generate(
        num_nodes=20,
        num_timesteps=30,
        topology="ring",
        seed=0,
        expose_intervention_control=True,
    )
    assert default.control_inputs is not None
    assert float(default.control_inputs[:10].max()) == 0.0
    assert float(default.control_inputs[-1]) > 0.0


def test_epidemic_intervention_rejects_bad_controls() -> None:
    """Verify intervention argument validation."""
    with pytest.raises(ValueError, match="expose_intervention_control"):
        EpidemicNetworkBenchmark.generate(intervention=torch.zeros(10))
    with pytest.raises(ValueError, match="shape"):
        EpidemicNetworkBenchmark.generate(
            num_timesteps=10,
            expose_intervention_control=True,
            intervention=torch.zeros(5, 1),
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        EpidemicNetworkBenchmark.generate(
            num_timesteps=10,
            expose_intervention_control=True,
            intervention=torch.full((10, 1), 1.5),
        )


def test_epidemic_intervention_bilinear_beats_additive_frozen_k() -> None:
    """Soft check: bilinear lowers post-onset MSE vs additive with frozen K."""
    from koopman_graph.operators import KoopmanOperator

    torch.manual_seed(0)
    num_timesteps = 80
    onset_frac = 0.25
    onset = int(onset_frac * num_timesteps)
    schedule = EpidemicNetworkBenchmark.default_intervention_schedule(
        num_timesteps,
        onset_fraction=onset_frac,
        max_reduction=0.85,
    )
    sequence = EpidemicNetworkBenchmark.generate(
        num_nodes=24,
        num_timesteps=num_timesteps,
        topology="ring",
        beta=0.5,
        gamma=0.1,
        seed=3,
        expose_intervention_control=True,
        intervention=schedule,
    )
    assert sequence.control_inputs is not None
    states = [sequence[t].x for t in range(len(sequence) - 1)]
    next_states = [sequence[t + 1].x for t in range(len(sequence) - 1)]
    controls = [sequence.control_inputs[t] for t in range(len(sequence) - 1)]
    design = torch.cat(states[: max(onset, 5)], dim=0)
    targets = torch.cat(next_states[: max(onset, 5)], dim=0)
    k0 = torch.linalg.lstsq(design, targets).solution.T
    post = list(range(onset, len(states)))

    def _fit(mode: str) -> float:
        operator = KoopmanOperator(
            3,
            control_dim=1,
            control_mode=mode,  # type: ignore[arg-type]
            init_mode="identity",
        )
        with torch.no_grad():
            kwargs: dict = {"control_matrix": torch.zeros(1, 3)}
            if mode == "bilinear":
                kwargs["bilinear_matrices"] = torch.zeros(1, 3, 3)
            operator.set_dense_matrix(k0, **kwargs)
        operator._parameters["K"].requires_grad_(False)
        trainable = [
            parameter for parameter in operator.parameters() if parameter.requires_grad
        ]
        opt = torch.optim.Adam(trainable, lr=3e-2)
        final = 0.0
        for _ in range(400):
            opt.zero_grad()
            loss = torch.zeros(())
            for index in post:
                pred = operator(states[index], control=controls[index])
                loss = loss + (pred - next_states[index]).square().mean()
            loss = loss / len(post)
            loss.backward()
            opt.step()
            final = float(loss.detach())
        return final

    additive_loss = _fit("additive")
    bilinear_loss = _fit("bilinear")
    assert bilinear_loss <= additive_loss
    assert (additive_loss - bilinear_loss) / max(additive_loss, 1e-12) >= 0.03


def test_lorenz96_seed_determinism_and_chaos_scale() -> None:
    """Verify Lorenz-96 ring trajectories are finite and deterministic."""
    kwargs = {
        "num_nodes": 16,
        "num_timesteps": 40,
        "burn_in": 30,
        "forcing": 8.0,
        "seed": 4,
    }
    first = Lorenz96GraphBenchmark.generate(**kwargs)
    second = Lorenz96GraphBenchmark.generate(**kwargs)
    assert first.num_nodes == 16
    assert first.in_channels == 1
    for left, right in zip(first, second, strict=True):
        assert torch.equal(left.x, right.x)
    assert torch.isfinite(first[-1].x).all()
    # Chaotic F=8 should not collapse to a near-constant field.
    assert float(first[-1].x.std()) > 0.5


def test_kuramoto_sivashinsky_seed_determinism_and_energy() -> None:
    """Verify KS ETDRK4 trajectories are finite and deterministic."""
    kwargs = {
        "num_nodes": 32,
        "num_timesteps": 30,
        "burn_in": 20,
        "domain_length": 22.0,
        "dt": 0.25,
        "seed": 2,
    }
    first = KuramotoSivashinskyBenchmark.generate(**kwargs)
    second = KuramotoSivashinskyBenchmark.generate(**kwargs)
    assert first.in_channels == 1
    for left, right in zip(first, second, strict=True):
        assert torch.equal(left.x, right.x)
    assert torch.isfinite(first[-1].x).all()
    assert float(first[-1].x.std()) > 0.1
    path_seq = KuramotoSivashinskyBenchmark.generate(
        num_nodes=32,
        num_timesteps=5,
        burn_in=5,
        topology="path",
        seed=0,
    )
    assert path_seq[0].edge_index.shape[1] == 2 * (32 - 1)


def test_cylinder_wake_cache_load(tmp_path: Path) -> None:
    """Verify cylinder-wake cache build/load mirrors the METR-LA pattern."""
    from koopman_graph.datasets.cylinder_wake import ensure_wake_cache

    cache_path = ensure_wake_cache(tmp_path, force=True)
    assert cache_path.exists()
    sequence = CylinderWakeBenchmark.load_sequence(tmp_path)
    topology = CylinderWakeBenchmark.load_topology(tmp_path)
    assert sequence.num_nodes == topology.num_nodes
    assert sequence.in_channels == CylinderWakeBenchmark.IN_CHANNELS
    assert sequence.num_timesteps >= 2
    assert torch.isfinite(sequence[0].x).all()


def test_cylinder_wake_default_cache_loads() -> None:
    """Verify the shipped default wake.pt cache is readable."""
    sequence = CylinderWakeBenchmark.load_sequence()
    assert sequence.num_nodes == CylinderWakeBenchmark.NUM_NODES
    assert sequence.num_timesteps >= 2
    again = CylinderWakeBenchmark.load_sequence()
    for left, right in zip(sequence, again, strict=True):
        assert torch.equal(left.x, right.x)
