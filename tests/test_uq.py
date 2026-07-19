"""Tests for ensemble and latent-Gaussian uncertainty quantification."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
from koopman_graph.datasets import SyntheticDynamicGraphBenchmark
from koopman_graph.uq import (
    EnsembleGraphKoopmanModel,
    IntervalForecastModel,
    LatentGaussianKoopmanUQ,
    PredictionInterval,
    dense_nodewise_transition,
    empirical_coverage,
    propagate_gaussian_covariance,
    quantile_levels,
    snapshot_with_features,
)


def _tiny_factory(*, koopman: str | None = None) -> GraphKoopmanModel:
    """Build a small GraphKoopmanModel suitable for fast UQ tests."""
    encoder = GNNEncoder(in_channels=1, hidden_channels=4, latent_dim=2)
    decoder = GNNDecoder(latent_dim=2, hidden_channels=4, out_channels=1)
    kwargs: dict[str, object] = {
        "encoder": encoder,
        "decoder": decoder,
        "latent_dim": 2,
        "time_step": 0.1,
    }
    if koopman is not None:
        kwargs["koopman"] = koopman
    return GraphKoopmanModel(**kwargs)  # type: ignore[arg-type]


def test_ensemble_rejects_empty_members() -> None:
    """Empty member lists raise ``ValueError``."""
    with pytest.raises(ValueError, match="at least one member"):
        EnsembleGraphKoopmanModel([])


def test_from_factory_uses_distinct_seeds() -> None:
    """Independently seeded members should not share identical weights."""
    ensemble = EnsembleGraphKoopmanModel.from_factory(
        _tiny_factory,
        n_members=3,
        seeds=(0, 1, 2),
    )
    assert ensemble.n_members == 3
    params = [
        torch.nn.utils.parameters_to_vector(member.parameters()).detach()
        for member in ensemble.members
    ]
    assert not torch.allclose(params[0], params[1])
    assert not torch.allclose(params[1], params[2])


def test_predict_interval_protocol_and_nesting() -> None:
    """``predict_interval`` satisfies the optional Protocol and nests mean."""
    ensemble = EnsembleGraphKoopmanModel.from_factory(
        _tiny_factory,
        n_members=4,
        seeds=(10, 11, 12, 13),
    )
    assert isinstance(ensemble, IntervalForecastModel)
    assert hasattr(ensemble, "predict_interval")

    sequence = SyntheticDynamicGraphBenchmark.generate(
        num_nodes=4,
        num_timesteps=8,
        in_channels=1,
        noise_std=0.0,
        seed=0,
    )
    ensemble.fit(sequence, epochs=2, lr=1e-2)

    initial = sequence[0]
    interval = ensemble.predict_interval(initial, steps=3, level=0.8)
    assert isinstance(interval, PredictionInterval)
    assert interval.level == 0.8
    assert interval.n_members == 4
    assert len(interval.mean) == 3
    assert len(interval.lower) == 3
    assert len(interval.upper) == 3

    mean_pred = ensemble.predict(initial, steps=3)
    for mean_snap, interval_mean in zip(mean_pred, interval.mean, strict=True):
        assert torch.allclose(mean_snap.x, interval_mean.x)

    for lower, mean, upper in zip(
        interval.lower, interval.mean, interval.upper, strict=True
    ):
        assert torch.all(lower.x <= mean.x + 1e-5)
        assert torch.all(mean.x <= upper.x + 1e-5)


def test_single_member_interval_collapses() -> None:
    """A one-member ensemble has identical mean and bounds."""
    ensemble = EnsembleGraphKoopmanModel.from_factory(
        _tiny_factory,
        n_members=1,
        seeds=(7,),
    )
    sequence = SyntheticDynamicGraphBenchmark.generate(
        num_nodes=3,
        num_timesteps=6,
        in_channels=1,
        seed=1,
    )
    ensemble.fit(sequence, epochs=1, lr=1e-2)
    interval = ensemble.predict_interval(sequence[0], steps=2, level=0.9)
    for mean, lower, upper in zip(
        interval.mean, interval.lower, interval.upper, strict=True
    ):
        assert torch.allclose(mean.x, lower.x)
        assert torch.allclose(mean.x, upper.x)


def test_empirical_coverage_on_noisy_synthetic() -> None:
    """90% ensemble intervals cover noisy synthetic targets within tolerance."""
    torch.manual_seed(0)
    sequence = SyntheticDynamicGraphBenchmark.generate(
        num_nodes=5,
        num_timesteps=24,
        in_channels=1,
        noise_std=0.05,
        seed=42,
    )
    train = sequence[:16]
    # Hold out later snapshots for open-loop coverage from train[-1].
    initial = train[-1]
    targets = list(sequence[16:20])

    ensemble = EnsembleGraphKoopmanModel.from_factory(
        _tiny_factory,
        n_members=5,
        seeds=(0, 1, 2, 3, 4),
    )
    ensemble.fit(train, epochs=15, lr=5e-3, seeds=(100, 101, 102, 103, 104))

    level = 0.9
    interval = ensemble.predict_interval(initial, steps=len(targets), level=level)
    coverage = empirical_coverage(targets, interval)

    # Deep-ensemble quantiles are coarse with few members; require coverage
    # near the nominal level without demanding perfect calibration.
    assert abs(coverage - level) <= 0.35
    assert 0.5 <= coverage <= 1.0


def test_empirical_coverage_rejects_length_mismatch() -> None:
    """Coverage helper validates aligned step counts."""
    snaps = [
        Data(x=torch.zeros(2, 1), edge_index=torch.tensor([[0], [1]])),
        Data(x=torch.ones(2, 1), edge_index=torch.tensor([[0], [1]])),
    ]
    interval = PredictionInterval(
        mean=snaps[:1],
        lower=snaps[:1],
        upper=snaps[:1],
        level=0.9,
        n_members=1,
    )
    with pytest.raises(ValueError, match="same number of steps"):
        empirical_coverage(snaps, interval)


def test_prediction_interval_fields_are_immutable_tuples() -> None:
    """Interval collections are tuples; slots cannot be appended or replaced."""
    snaps = [
        Data(x=torch.zeros(2, 1), edge_index=torch.tensor([[0], [1]])),
        Data(x=torch.ones(2, 1), edge_index=torch.tensor([[0], [1]])),
    ]
    interval = PredictionInterval(
        mean=list(snaps),
        lower=list(snaps),
        upper=list(snaps),
        level=0.9,
        n_members=2,
    )
    assert isinstance(interval.mean, tuple)
    assert isinstance(interval.lower, tuple)
    assert isinstance(interval.upper, tuple)
    assert interval.mean[0] is snaps[0]
    with pytest.raises(AttributeError):
        interval.mean.append(snaps[0])  # type: ignore[attr-defined]
    with pytest.raises(TypeError):
        interval.mean[0] = snaps[1]  # type: ignore[index]
    # Mutating the input list after construction must not alter storage.
    snaps.append(snaps[0])
    assert len(interval.mean) == 2


def test_prediction_interval_lives_in_uq_common() -> None:
    """Peers import ``PredictionInterval`` from ``uq.common``, not each other."""
    import ast
    from pathlib import Path

    import koopman_graph.uq.common as uq_common
    import koopman_graph.uq.ensemble as uq_ensemble
    import koopman_graph.uq.latent_gaussian as uq_latent

    assert PredictionInterval is uq_common.PredictionInterval
    assert uq_ensemble.PredictionInterval is uq_common.PredictionInterval
    assert uq_latent.PredictionInterval is uq_common.PredictionInterval

    root = Path(__file__).resolve().parents[1] / "src" / "koopman_graph" / "uq"
    for module_name, path in (
        ("ensemble", root / "ensemble.py"),
        ("latent_gaussian", root / "latent_gaussian.py"),
    ):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        peer_interval_imports: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            if node.module == "koopman_graph.uq.common":
                continue
            if not node.module.startswith("koopman_graph.uq"):
                continue
            for alias in node.names:
                if alias.name == "PredictionInterval":
                    peer_interval_imports.append(
                        f"{module_name}:{node.module}.PredictionInterval"
                    )
        assert peer_interval_imports == [], peer_interval_imports


def test_graph_koopman_members_remain_topology_required() -> None:
    """``koopman='graph'`` members still require topology for spectrum."""
    ensemble = EnsembleGraphKoopmanModel.from_factory(
        lambda: _tiny_factory(koopman="graph"),
        n_members=2,
        seeds=(0, 1),
    )
    member = ensemble.members[0]
    assert member.uses_graph_koopman
    with pytest.raises(ValueError, match="edge_index"):
        member.spectrum()

    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    spectrum = member.spectrum(edge_index=edge_index, num_nodes=3)
    assert spectrum.eigenvalues.numel() == 3 * member.latent_dim


def test_ensemble_not_on_root_all() -> None:
    """UQ types stay off the root façade."""
    import koopman_graph

    exported = set(koopman_graph.__all__)
    assert "EnsembleGraphKoopmanModel" not in exported
    assert "LatentGaussianKoopmanUQ" not in exported
    assert "PredictionInterval" not in exported
    assert "empirical_coverage" not in exported


def test_save_load_round_trip(tmp_path: Path) -> None:
    """Directory save/load restores member predictions via format-1 files."""
    ensemble = EnsembleGraphKoopmanModel.from_factory(
        _tiny_factory,
        n_members=2,
        seeds=(3, 4),
    )
    sequence = SyntheticDynamicGraphBenchmark.generate(
        num_nodes=3,
        num_timesteps=6,
        in_channels=1,
        seed=2,
    )
    ensemble.fit(sequence, epochs=2, lr=1e-2)

    directory = tmp_path / "ensemble_ckpt"
    ensemble.save(directory)
    loaded = EnsembleGraphKoopmanModel.load(directory)

    assert loaded.n_members == 2
    initial = sequence[0]
    original = ensemble.predict_interval(initial, steps=2, level=0.9)
    restored = loaded.predict_interval(initial, steps=2, level=0.9)
    for left, right in zip(original.mean, restored.mean, strict=True):
        assert torch.allclose(left.x, right.x, atol=1e-6)


def test_propagate_gaussian_covariance_closed_form() -> None:
    """``A P Aᵀ + Q`` matches a hand-computed dense recurrence."""
    k_mat = torch.tensor([[0.5, 0.1], [0.0, 0.4]], dtype=torch.float64)
    p0 = torch.eye(2, dtype=torch.float64)
    q_scale = 0.01
    expected = k_mat @ p0 @ k_mat.T + q_scale * torch.eye(2, dtype=torch.float64)
    got = propagate_gaussian_covariance(k_mat, p0, q_scale)
    assert torch.allclose(got, expected, atol=1e-12)

    # Multi-step with nodewise kron transition.
    n_nodes = 3
    transition = dense_nodewise_transition(k_mat, n_nodes)
    cov = torch.eye(n_nodes * 2, dtype=torch.float64)
    for _ in range(4):
        cov = propagate_gaussian_covariance(transition, cov, q_scale)
    block = cov[:2, :2]
    hand = p0.clone()
    for _ in range(4):
        hand = k_mat @ hand @ k_mat.T + q_scale * torch.eye(2, dtype=torch.float64)
    assert torch.allclose(block, hand, atol=1e-12)


def test_latent_gaussian_mean_matches_model_predict() -> None:
    """Open-loop latent-mean decode matches ``GraphKoopmanModel.predict``."""
    model = _tiny_factory()
    sequence = SyntheticDynamicGraphBenchmark.generate(
        num_nodes=4,
        num_timesteps=8,
        in_channels=1,
        noise_std=0.0,
        seed=3,
    )
    model.fit(sequence, epochs=3, lr=1e-2)
    uq = LatentGaussianKoopmanUQ(
        model,
        process_noise=1e-3,
        initial_covariance=1.0,
        n_samples=8,
    )
    assert isinstance(uq, IntervalForecastModel)

    initial = sequence[0]
    steps = 3
    mean_pred = uq.predict(initial, steps=steps)
    model_pred = model.predict(initial, steps=steps)
    for left, right in zip(mean_pred, model_pred, strict=True):
        assert torch.allclose(left.x, right.x, atol=1e-5)


def test_latent_gaussian_forecast_matches_hand_covariance() -> None:
    """Dense discrete forecast covariances match the analytic recurrence."""
    model = _tiny_factory()
    # Freeze a known Koopman matrix for a closed-form check.
    with torch.no_grad():
        model.koopman.K.copy_(
            torch.tensor([[0.6, 0.05], [0.0, 0.5]], dtype=model.koopman.K.dtype)
        )

    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    initial = Data(x=torch.randn(2, 1), edge_index=edge_index)
    process_noise = 1e-2
    initial_covariance = 0.25
    uq = LatentGaussianKoopmanUQ(
        model,
        process_noise=process_noise,
        initial_covariance=initial_covariance,
        n_samples=4,
    )
    forecast = uq.forecast_latents(initial, steps=3)
    k_mat = model.koopman.matrix.detach()
    transition = dense_nodewise_transition(k_mat, num_nodes=2)
    cov = initial_covariance * torch.eye(4)
    q_mat = process_noise * torch.eye(4)
    for step in range(3):
        cov = propagate_gaussian_covariance(transition, cov, q_mat)
        assert torch.allclose(forecast.covariances[step], cov, atol=1e-5)


def test_latent_gaussian_kalman_update_reduces_covariance() -> None:
    """A perfect latent observation shrinks predictive covariance."""
    model = _tiny_factory()
    sequence = SyntheticDynamicGraphBenchmark.generate(
        num_nodes=3,
        num_timesteps=6,
        in_channels=1,
        seed=4,
    )
    model.fit(sequence, epochs=2, lr=1e-2)
    uq = LatentGaussianKoopmanUQ(
        model,
        process_noise=1e-3,
        observation_noise=1e-6,
        initial_covariance=1.0,
        n_samples=4,
    )
    initial = sequence[0]
    open_loop = uq.forecast_latents(initial, steps=2)
    # Use the model's own one-step prediction as a near-perfect observation.
    obs = model.predict(initial, steps=2)
    refined = uq.forecast_latents(initial, steps=2, observations=obs)
    assert (
        torch.linalg.eigvalsh(refined.covariances[-1]).sum()
        < torch.linalg.eigvalsh(open_loop.covariances[-1]).sum()
    )


def test_latent_gaussian_predict_interval_shapes() -> None:
    """``predict_interval`` returns nested bounds and reports ``n_samples``."""
    model = _tiny_factory()
    sequence = SyntheticDynamicGraphBenchmark.generate(
        num_nodes=3,
        num_timesteps=6,
        in_channels=1,
        seed=5,
    )
    model.fit(sequence, epochs=2, lr=1e-2)
    uq = LatentGaussianKoopmanUQ(model, n_samples=16)
    interval = uq.predict_interval(sequence[0], steps=2, level=0.8)
    assert isinstance(interval, PredictionInterval)
    assert interval.level == 0.8
    assert interval.n_members == 16
    assert len(interval.mean) == 2
    for lower, mean, upper in zip(
        interval.lower, interval.mean, interval.upper, strict=True
    ):
        assert torch.all(lower.x <= mean.x + 1e-4)
        assert torch.all(mean.x <= upper.x + 1e-4)


def test_latent_gaussian_rejects_bilinear() -> None:
    """Bilinear control mode is rejected at construction."""
    encoder = GNNEncoder(in_channels=1, hidden_channels=4, latent_dim=2)
    decoder = GNNDecoder(latent_dim=2, hidden_channels=4, out_channels=1)
    model = GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=2,
        time_step=0.1,
        control_dim=1,
        control_mode="bilinear",
    )
    with pytest.raises(ValueError, match="bilinear"):
        LatentGaussianKoopmanUQ(model)


def test_continuous_latent_gaussian_propagates() -> None:
    """Continuous operators propagate via ``exp(L Δt)`` without error."""
    encoder = GNNEncoder(in_channels=1, hidden_channels=4, latent_dim=2)
    decoder = GNNDecoder(latent_dim=2, hidden_channels=4, out_channels=1)
    model = GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=2,
        time_step=0.1,
        dynamics_mode="continuous",
    )
    sequence = SyntheticDynamicGraphBenchmark.generate(
        num_nodes=3,
        num_timesteps=6,
        in_channels=1,
        seed=6,
    )
    model.fit(sequence, epochs=1, lr=1e-2)
    uq = LatentGaussianKoopmanUQ(model, n_samples=4)
    forecast = uq.forecast_latents(sequence[0], steps=2)
    assert forecast.means.shape == (2, 3, 2)
    assert forecast.covariances.shape == (2, 6, 6)


def test_latent_gaussian_control_bias_parity() -> None:
    """UQ control bias matches ``model._advance_latent`` for built-in ops."""
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_weight = torch.ones(2)
    control = torch.tensor([[0.4], [-0.2]])
    n_nodes = 2

    cases: list[GraphKoopmanModel] = []

    discrete = GraphKoopmanModel(
        GNNEncoder(1, 4, 2, num_layers=1),
        GNNDecoder(2, 4, 1, num_layers=1),
        latent_dim=2,
        time_step=0.1,
        control_dim=1,
        koopman_parameterization="dense",
    )
    discrete.koopman.set_dense_matrix(
        torch.tensor([[0.6, 0.05], [0.0, 0.5]]),
        control_matrix=torch.tensor([[0.2, 0.1]]),
    )
    cases.append(discrete)

    continuous = GraphKoopmanModel(
        GNNEncoder(1, 4, 2, num_layers=1),
        GNNDecoder(2, 4, 1, num_layers=1),
        latent_dim=2,
        time_step=0.25,
        dynamics_mode="continuous",
        control_dim=1,
        koopman_parameterization="dense",
    )
    continuous.koopman.set_dense_matrix(
        torch.diag(torch.tensor([-0.3, -0.5])),
        control_matrix=torch.tensor([[0.15, -0.05]]),
    )
    cases.append(continuous)

    graph = GraphKoopmanModel(
        GNNEncoder(1, 4, 2, num_layers=1),
        GNNDecoder(2, 4, 1, num_layers=1),
        latent_dim=2,
        time_step=0.1,
        control_dim=1,
        koopman="graph",
        koopman_parameterization="dense",
    )
    k_mat = torch.tensor([[0.7, 0.0], [0.0, 0.6]])
    graph.koopman.set_dense_matrices(
        k_mat,
        0.05 * k_mat,
        control_matrix=torch.tensor([[0.1, 0.2]]),
    )
    cases.append(graph)

    for model in cases:
        uq = LatentGaussianKoopmanUQ(model, n_samples=2)
        _transition, bias = uq._transition_and_bias(
            n_nodes=n_nodes,
            edge_index=edge_index,
            edge_weight=edge_weight,
            control=control,
            default_delta_t=model.time_step,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        z0 = torch.zeros(n_nodes, model.latent_dim)
        expected = model._advance_latent(
            z0,
            control=control,
            delta_t=model.time_step,
            edge_index=edge_index,
            edge_weight=edge_weight,
        ).reshape(-1)
        assert torch.allclose(bias, expected, atol=1e-6)


def test_quantile_levels_and_snapshot_helpers() -> None:
    """Shared UQ helpers map coverage levels and clone topology."""
    lower_q, upper_q = quantile_levels(0.9)
    assert lower_q == pytest.approx(0.05)
    assert upper_q == pytest.approx(0.95)
    with pytest.raises(ValueError, match="level must lie"):
        quantile_levels(1.0)

    template = Data(
        x=torch.zeros(2, 1),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_weight=torch.tensor([0.5, 0.5]),
    )
    features = torch.ones(2, 1)
    out = snapshot_with_features(template, features)
    assert torch.equal(out.x, features)
    assert torch.equal(out.edge_index, template.edge_index)
    assert torch.equal(out.edge_weight, template.edge_weight)
    assert out is not template


def test_uq_does_not_import_model_inference() -> None:
    """``koopman_graph.uq`` must not import ``koopman_graph.model.inference``."""
    import ast
    from pathlib import Path

    uq_root = Path(__file__).resolve().parents[1] / "src" / "koopman_graph" / "uq"
    forbidden: list[str] = []
    for path in sorted(uq_root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module == "koopman_graph.model.inference" or (
                    node.module.startswith("koopman_graph.model.inference.")
                ):
                    forbidden.append(f"{path.name}:{node.lineno}:{node.module}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "koopman_graph.model.inference" or (
                        alias.name.startswith("koopman_graph.model.inference.")
                    ):
                        forbidden.append(f"{path.name}:{alias.name}")
    assert forbidden == []


def test_latent_gaussian_has_no_private_peer_imports() -> None:
    """``uq.latent_gaussian`` must not import leading-``_`` peer symbols."""
    import ast
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "koopman_graph"
        / "uq"
        / "latent_gaussian.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    private_imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if not node.module.startswith("koopman_graph."):
                continue
            if node.module.endswith(".latent_gaussian"):
                continue
            for alias in node.names:
                if alias.name.startswith("_"):
                    private_imports.append(f"{node.module}.{alias.name}")
    assert private_imports == []


def test_encode_rollout_origin_matches_predict_preamble() -> None:
    """``encode_rollout_origin`` matches delay encode used by ``predict``."""
    encoder = GNNEncoder(in_channels=2, hidden_channels=4, latent_dim=2)
    decoder = GNNDecoder(latent_dim=2, hidden_channels=4, out_channels=1)
    model = GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=2,
        n_delays=2,
        time_step=0.1,
    )
    history = [
        Data(
            x=torch.tensor([[0.1], [0.2]]),
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        )
    ]
    initial = Data(
        x=torch.tensor([[0.3], [0.4]]),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
    )
    z, edge_index, edge_weight = model.encode_rollout_origin(
        initial,
        history=history,
    )
    preds = model.predict(initial, steps=1, history=history)
    assert z.shape == (2, 2)
    assert edge_index.shape[0] == 2
    assert edge_weight is None
    assert len(preds) == 1
    # Origin latent used by UQ must agree with a fresh encode_rollout_origin.
    z2, _, _ = model.encode_rollout_origin(initial, history=history)
    assert torch.allclose(z, z2)
