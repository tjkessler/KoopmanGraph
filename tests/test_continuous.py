"""Tests for continuous-time Koopman dynamics."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data

from koopman_graph import (
    ContinuousKoopmanOperator,
    GNNDecoder,
    GNNEncoder,
    GraphKoopmanModel,
    GraphSnapshotSequence,
)
from koopman_graph.analysis import (
    compute_generator_spectrum,
    discrete_spectrum_at_delta_t,
)
from koopman_graph.data import resolve_pair_delta_t


@pytest.fixture
def tiny_continuous_model(synthetic_edge_index: torch.Tensor) -> GraphKoopmanModel:
    """Return a small continuous-time model for synthetic tests."""
    encoder = GNNEncoder(in_channels=3, hidden_channels=8, latent_dim=4, num_layers=2)
    decoder = GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=3, num_layers=2)
    return GraphKoopmanModel(
        encoder,
        decoder,
        latent_dim=4,
        time_step=0.1,
        dynamics_mode="continuous",
        koopman_parameterization="dissipative",
        control_dim=0,
    )


def test_transition_matrix_at_zero_is_identity() -> None:
    """Verify exp(L·0) ≈ I."""
    op = ContinuousKoopmanOperator(
        4, parameterization="dense", init_mode="identity_noise"
    )
    transition = op.transition_matrix(0.0)
    identity = torch.eye(4)
    assert torch.allclose(transition, identity, atol=1e-5)


def test_advance_at_zero_is_identity() -> None:
    """Verify advance(z, 0) returns z unchanged."""
    op = ContinuousKoopmanOperator(3, parameterization="schur")
    z = torch.randn(5, 3)
    advanced = op.advance(z, 0.0)
    assert torch.allclose(advanced, z)


@pytest.mark.parametrize(
    "parameterization", ["dense", "odo", "schur", "dissipative", "lyapunov"]
)
def test_semigroup_property(parameterization: str) -> None:
    """Verify advance(z, dt1+dt2) matches composed advances."""
    op = ContinuousKoopmanOperator(4, parameterization=parameterization)  # type: ignore[arg-type]
    z = torch.randn(2, 4)
    dt1, dt2 = 0.07, 0.13
    composed = op.advance(op.advance(z, dt1), dt2)
    direct = op.advance(z, dt1 + dt2)
    assert torch.allclose(composed, direct, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("parameterization", ["dense", "dissipative"])
def test_inverse_advance_round_trip_uncontrolled(parameterization: str) -> None:
    """Verify inverse_advance recovers z after uncontrolled advance."""
    op = ContinuousKoopmanOperator(
        4,
        parameterization=parameterization,  # type: ignore[arg-type]
        init_mode="identity_noise",
        init_scale=1e-2,
    )
    z = torch.randn(3, 4)
    delta_t = 0.15
    z_next = op.advance(z, delta_t)
    recovered = op.inverse_advance(z_next, delta_t)
    assert torch.allclose(recovered, z, atol=1e-4)


def test_inverse_advance_round_trip_controlled() -> None:
    """Verify inverse_advance recovers z after controlled Van Loan advance."""
    torch.manual_seed(0)
    op = ContinuousKoopmanOperator(
        4,
        parameterization="dense",
        control_dim=1,
        init_mode="identity_noise",
        init_scale=1e-3,
    )
    z = torch.randn(2, 4)
    control = torch.tensor([0.3])
    delta_t = 0.2
    z_next = op.advance(z, delta_t, control=control)
    recovered = op.inverse_advance(z_next, delta_t, control=control)
    assert torch.allclose(recovered, z, atol=1e-3, rtol=1e-4)


@pytest.mark.parametrize("parameterization", ["schur", "dissipative", "lyapunov"])
def test_continuous_structured_max_real_part_non_positive(
    parameterization: str,
) -> None:
    """Verify structured generators report non-positive max real-part bounds."""
    op = ContinuousKoopmanOperator(
        4,
        parameterization=parameterization,  # type: ignore[arg-type]
        init_mode="identity",
    )
    assert op.max_real_part().item() <= 0


@pytest.mark.parametrize("parameterization", ["schur", "dissipative", "lyapunov"])
def test_continuous_structured_stability_certificate_positive_margin(
    parameterization: str,
) -> None:
    """Verify continuous structural modes expose positive stability margins."""
    op = ContinuousKoopmanOperator(
        4,
        parameterization=parameterization,  # type: ignore[arg-type]
        init_mode="identity",
    )
    certificate = op.stability_certificate()
    assert certificate is not None
    assert certificate["margin"].item() > 0
    if parameterization == "lyapunov":
        assert "lyapunov_matrix" in certificate


def test_controlled_van_loan_matches_manual_integral() -> None:
    """Verify controlled advance matches a small-system manual integral."""
    latent_dim = 2
    control_dim = 1
    op = ContinuousKoopmanOperator(
        latent_dim,
        parameterization="dense",
        control_dim=control_dim,
        init_mode="identity_noise",
        init_scale=1e-2,
    )
    delta_t = 0.25
    z = torch.tensor([[1.0, 0.5]])
    u = torch.tensor([0.3])

    phi11, phi12 = op._van_loan_factors(torch.tensor(delta_t))
    expected = z @ phi11.T + u @ phi12.T
    actual = op.advance(z, delta_t, control=u)
    assert torch.allclose(actual, expected, atol=1e-5)


def test_timestamps_validation_rejects_non_monotone() -> None:
    """Verify invalid timestamps raise on construction."""
    snapshots = [
        Data(x=torch.randn(5, 3), edge_index=torch.tensor([[0], [1]])),
        Data(x=torch.randn(5, 3), edge_index=torch.tensor([[0], [1]])),
    ]
    with pytest.raises(ValueError, match="strictly monotone"):
        GraphSnapshotSequence(
            snapshots,
            timestamps=torch.tensor([0.0, 0.0]),
        )


def test_predict_at_query_times_and_step_deltas_mutually_exclusive(
    tiny_continuous_model: GraphKoopmanModel,
    synthetic_graph: Data,
) -> None:
    """Verify predict_at enforces exactly one time specification."""
    with pytest.raises(ValueError, match="exactly one"):
        tiny_continuous_model.predict_at(
            synthetic_graph,
            query_times=[0.1, 0.2],
            step_deltas=[0.1, 0.1],
        )
    with pytest.raises(ValueError, match="exactly one"):
        tiny_continuous_model.predict_at(synthetic_graph)


def test_predict_at_returns_expected_shapes(
    tiny_continuous_model: GraphKoopmanModel,
    synthetic_graph: Data,
) -> None:
    """Verify predict_at output shapes for query_times and step_deltas."""
    by_query = tiny_continuous_model.predict_at(
        synthetic_graph,
        query_times=[0.1, 0.25, 0.4],
    )
    by_deltas = tiny_continuous_model.predict_at(
        synthetic_graph,
        step_deltas=[0.1, 0.15, 0.15],
    )
    assert len(by_query) == 3
    assert len(by_deltas) == 3
    for snapshot in by_query:
        assert snapshot.x.shape == synthetic_graph.x.shape


def test_discrete_predict_at_rejects_irregular_increments(
    synthetic_graph: Data,
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Verify discrete models reject non-uniform query schedules."""
    encoder = GNNEncoder(in_channels=3, hidden_channels=8, latent_dim=4, num_layers=2)
    decoder = GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=3, num_layers=2)
    model = GraphKoopmanModel(
        encoder,
        decoder,
        latent_dim=4,
        time_step=0.1,
        dynamics_mode="discrete",
    )
    with pytest.raises(ValueError, match="uniform increments"):
        model.predict_at(synthetic_graph, step_deltas=[0.1, 0.2])


def test_irregular_timestamp_training_recovers_generator(
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Train a continuous model on irregular timestamps from a known linear system."""
    torch.manual_seed(0)
    latent_dim = 3
    generator = torch.tensor(
        [
            [-0.5, 0.0, 0.0],
            [0.0, -0.5, 0.0],
            [0.0, 0.0, -0.5],
        ]
    )
    timestamps = torch.tensor([0.0, 0.05, 0.18, 0.20, 0.55, 0.90])
    z_path = [torch.randn(5, latent_dim)]
    for index in range(len(timestamps) - 1):
        delta = timestamps[index + 1] - timestamps[index]
        transition = torch.linalg.matrix_exp(generator * delta)
        z_path.append(z_path[-1] @ transition.T)

    snapshots = [Data(x=z, edge_index=synthetic_edge_index) for z in z_path]
    sequence = GraphSnapshotSequence(snapshots, timestamps=timestamps)

    encoder = GNNEncoder(
        in_channels=latent_dim,
        hidden_channels=32,
        latent_dim=latent_dim,
        num_layers=1,
    )
    decoder = GNNDecoder(
        latent_dim=latent_dim,
        hidden_channels=32,
        out_channels=latent_dim,
        num_layers=1,
    )
    model = GraphKoopmanModel(
        encoder,
        decoder,
        latent_dim=latent_dim,
        time_step=0.1,
        dynamics_mode="continuous",
        koopman_parameterization="dissipative",
        koopman_init_mode="identity_noise",
        koopman_init_scale=1e-3,
    )

    history = model.fit(
        sequence,
        epochs=400,
        lr=1e-2,
        loss_weights=__import__("koopman_graph").LossWeights(
            reconstruction=1.0,
            forward=5.0,
        ),
    )
    assert history.loss[-1] < history.loss[0]

    learned_spec = compute_generator_spectrum(model.koopman.L)
    assert torch.all(learned_spec.growth_rates < 0.0)

    for timestep in range(sequence.num_timesteps - 1):
        delta = resolve_pair_delta_t(
            sequence, timestep, default_time_step=model.time_step
        )
        expected = torch.linalg.matrix_exp(generator * delta)
        learned = model.koopman.transition_matrix(delta)
        assert torch.allclose(learned, expected, atol=0.2, rtol=0.2)


def test_generator_spectrum_and_discrete_conversion() -> None:
    """Verify generator and discrete-at-delta spectra are consistent."""
    generator = torch.tensor([[-0.4, 0.0], [0.0, -0.2]])
    delta_t = 0.5
    discrete = discrete_spectrum_at_delta_t(generator, delta_t)
    transition = torch.linalg.matrix_exp(generator * delta_t)
    assert torch.allclose(
        discrete.eigenvalues.abs().sort().values,
        torch.linalg.eigvals(transition).abs().sort().values,
        atol=1e-4,
    )


def test_continuous_model_serialization_round_trip(
    tiny_continuous_model: GraphKoopmanModel,
    synthetic_graph: Data,
    tmp_path,
) -> None:
    """Verify continuous checkpoints round-trip through save/load."""
    path = tmp_path / "continuous_model.pt"
    with torch.no_grad():
        before = tiny_continuous_model.predict_at(
            synthetic_graph, step_deltas=[0.1, 0.1]
        )
    tiny_continuous_model.save(path)
    loaded = GraphKoopmanModel.load(path)
    assert loaded.dynamics_mode == "continuous"
    assert loaded.koopman.parameterization == "dissipative"
    with torch.no_grad():
        after = loaded.predict_at(synthetic_graph, step_deltas=[0.1, 0.1])
    for left, right in zip(before, after, strict=True):
        assert torch.allclose(left.x, right.x)


@pytest.mark.parametrize(
    "parameterization",
    ["dense", "odo", "schur", "dissipative", "lyapunov"],
)
def test_continuous_eigenvalue_regularization_resolves_parameters(
    parameterization: str,
) -> None:
    """Structured continuous parameterizations must not crash eigenvalue-reg lookup.

    A previous bug eagerly built a diagnostic dict referencing every structured
    parameter name (``diag_raw``, ``schur_diag_raw``, ``lyap_diag_raw``), so any
    structured continuous operator other than the active one raised AttributeError.
    """
    from koopman_graph.losses import EigenvalueRegularizationLoss

    operator = ContinuousKoopmanOperator(
        latent_dim=6,
        parameterization=parameterization,  # type: ignore[arg-type]
    )
    penalty = EigenvalueRegularizationLoss()(operator)
    assert penalty.ndim == 0
    assert torch.isfinite(penalty)


def _masked_one_step_rmse(
    model: GraphKoopmanModel,
    sequence: GraphSnapshotSequence,
) -> float:
    """Average masked one-step RMSE over consecutive pairs."""
    from koopman_graph.metrics import masked_rmse

    model.eval()
    total = 0.0
    with torch.no_grad():
        for timestep in range(sequence.num_timesteps - 1):
            delta_t = resolve_pair_delta_t(
                sequence,
                timestep,
                default_time_step=model.time_step,
            )
            prediction = model(sequence[timestep], delta_t=delta_t)
            target = sequence[timestep + 1].x
            mask = sequence.observation_mask_at(timestep + 1)
            total += float(masked_rmse(prediction, target, mask).cpu())
    return total / (sequence.num_timesteps - 1)


def test_masked_irregular_training_continuous_matches_or_beats_discrete(
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Masked irregular training favors continuous dynamics over fixed-step discrete."""
    torch.manual_seed(7)
    latent_dim = 3
    num_nodes = 5
    generator = torch.tensor(
        [
            [-0.45, 0.0, 0.0],
            [0.0, -0.45, 0.0],
            [0.0, 0.0, -0.45],
        ]
    )
    timestamps = torch.tensor([0.0, 0.04, 0.11, 0.15, 0.29, 0.42, 0.50, 0.68])
    z_path = [torch.randn(num_nodes, latent_dim)]
    for index in range(len(timestamps) - 1):
        delta = timestamps[index + 1] - timestamps[index]
        transition = torch.linalg.matrix_exp(generator * delta)
        z_path.append(z_path[-1] @ transition.T)

    snapshots = [Data(x=state, edge_index=synthetic_edge_index) for state in z_path]
    masks = torch.rand(len(snapshots), num_nodes) > 0.35
    for timestep in range(masks.shape[0]):
        if masks[timestep].sum() < 2:
            masks[timestep, :2] = True

    sequence = GraphSnapshotSequence(
        snapshots,
        timestamps=timestamps,
        observation_masks=masks,
    )
    loss_weights = __import__("koopman_graph").LossWeights(
        reconstruction=1.0,
        forward=4.0,
    )

    def _train(dynamics_mode: str) -> GraphKoopmanModel:
        encoder = GNNEncoder(
            in_channels=latent_dim,
            hidden_channels=32,
            latent_dim=latent_dim,
            num_layers=1,
        )
        decoder = GNNDecoder(
            latent_dim=latent_dim,
            hidden_channels=32,
            out_channels=latent_dim,
            num_layers=1,
        )
        model = GraphKoopmanModel(
            encoder,
            decoder,
            latent_dim=latent_dim,
            time_step=0.1,
            dynamics_mode=dynamics_mode,  # type: ignore[arg-type]
            koopman_parameterization="dissipative",
            koopman_init_mode="identity_noise",
            koopman_init_scale=1e-3,
        )
        model.fit(sequence, epochs=500, lr=1e-2, loss_weights=loss_weights)
        return model

    continuous_model = _train("continuous")
    discrete_model = _train("discrete")

    continuous_rmse = _masked_one_step_rmse(continuous_model, sequence)
    discrete_rmse = _masked_one_step_rmse(discrete_model, sequence)
    assert continuous_rmse <= discrete_rmse * 1.05


def test_unmasked_sequence_matches_prior_one_step_loss(
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Verify sequences without masks use the full-graph loss path."""
    from koopman_graph.training import one_step_loss

    torch.manual_seed(0)
    snapshots = [
        Data(x=torch.randn(5, 3), edge_index=synthetic_edge_index),
        Data(x=torch.randn(5, 3), edge_index=synthetic_edge_index),
    ]
    sequence = GraphSnapshotSequence(snapshots)
    encoder = GNNEncoder(in_channels=3, hidden_channels=8, latent_dim=4, num_layers=1)
    decoder = GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=3, num_layers=1)
    model = GraphKoopmanModel(encoder, decoder, latent_dim=4, time_step=0.1)
    loss = one_step_loss(model, sequence[0], sequence[1])
    full = torch.nn.functional.mse_loss(model(sequence[0]), sequence[1].x)
    assert torch.allclose(loss, full)
