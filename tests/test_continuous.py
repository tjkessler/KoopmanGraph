"""Tests for continuous-time Koopman dynamics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import Tensor
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
from koopman_graph.operators import (
    STABILITY_EPS_MARGIN,
    GraphKoopmanOperator,
    KoopmanOperator,
    matrix_log,
    van_loan_factors,
    van_loan_generator_from_discrete,
)
from koopman_graph.operators.continuous import negative_strict_diagonal_values
from koopman_graph.training import LossWeights


def test_matrix_log_round_trips_spd_matrix() -> None:
    """Principal log should satisfy exp(log(M)) ≈ M for well-conditioned SPD M."""
    torch.manual_seed(1)
    factor = torch.randn(4, 4)
    spd = factor @ factor.T + 0.5 * torch.eye(4)
    recovered = torch.linalg.matrix_exp(matrix_log(spd))
    assert torch.allclose(recovered, spd, atol=1e-5)


def test_matrix_log_of_matrix_exp_recovers_symmetric_generator() -> None:
    """log(exp(L)) ≈ L for a symmetric Hurwitz generator."""
    generator = torch.tensor(
        [
            [-0.5, 0.1, 0.0],
            [0.1, -0.4, 0.05],
            [0.0, 0.05, -0.3],
        ]
    )
    recovered = matrix_log(torch.linalg.matrix_exp(generator))
    assert torch.allclose(recovered, generator, atol=1e-5)


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


def test_zero_control_matches_uncontrolled_for_nonsymmetric_generator() -> None:
    """Van Loan with B=0 must match uncontrolled advance for nonsymmetric L."""
    torch.manual_seed(0)
    L = torch.tensor(
        [
            [0.0, 2.0, -0.5],
            [-0.5, 0.0, 1.0],
            [0.3, -1.0, -0.2],
        ]
    )
    delta_t = 0.2
    z = torch.randn(5, 3)

    uncontrolled = ContinuousKoopmanOperator(
        3,
        parameterization="dense",
        init_mode="identity",
    )
    uncontrolled.set_dense_matrix(L)

    controlled = ContinuousKoopmanOperator(
        3,
        control_dim=2,
        parameterization="dense",
        init_mode="identity",
    )
    controlled.set_dense_matrix(L, control_matrix=torch.zeros(2, 3))

    z_unc = uncontrolled.advance(z, delta_t)
    z_ctrl = controlled.advance(z, delta_t, control=torch.zeros(2))
    assert torch.allclose(z_unc, z_ctrl, atol=1e-5)

    # Phi11 must equal exp(L Δt) so z @ Phi11.T matches uncontrolled.
    phi11, _ = van_loan_factors(L, torch.zeros(2, 3), delta_t)
    assert torch.allclose(phi11, torch.linalg.matrix_exp(L * delta_t), atol=1e-5)


def test_controlled_inverse_advance_nonsymmetric_generator() -> None:
    """Controlled inverse must round-trip for strongly nonsymmetric L."""
    torch.manual_seed(1)
    L = torch.tensor(
        [
            [0.0, 2.0, 0.0, 0.0],
            [-0.5, 0.0, 0.0, 0.0],
            [0.0, 0.0, -0.3, 0.4],
            [0.0, 0.0, -0.2, -0.25],
        ]
    )
    B = torch.zeros(1, 4)
    B[0, 0] = 1.0
    op = ContinuousKoopmanOperator(
        4,
        control_dim=1,
        parameterization="dense",
        init_mode="identity",
    )
    op.set_dense_matrix(L, control_matrix=B)
    z = torch.randn(3, 4)
    control = torch.tensor([0.4])
    delta_t = 0.2
    z_next = op.advance(z, delta_t, control=control)
    recovered = op.inverse_advance(z_next, delta_t, control=control)
    assert torch.allclose(recovered, z, atol=1e-4)


def test_van_loan_round_trip_recovers_nonsymmetric_generator() -> None:
    """Discrete↔generator Van Loan recovery must preserve nonsymmetric L."""
    torch.manual_seed(2)
    L = torch.randn(3, 3) * 0.15
    B = torch.randn(2, 3) * 0.1
    delta_t = 0.25
    phi11, phi12 = van_loan_factors(L, B, delta_t)
    L_rec, B_rec = van_loan_generator_from_discrete(phi11, phi12.T, delta_t)
    assert torch.allclose(L_rec, L, atol=1e-5)
    assert torch.allclose(B_rec, B, atol=1e-5)


@pytest.mark.parametrize("parameterization", ["schur", "dissipative", "lyapunov"])
def test_continuous_structured_max_real_part_strictly_negative(
    parameterization: str,
) -> None:
    """Verify structured generators are strictly Hurwitz (max Re(λ) < 0)."""
    op = ContinuousKoopmanOperator(
        4,
        parameterization=parameterization,  # type: ignore[arg-type]
        init_mode="identity",
    )
    assert op.bound_metric().item() < 0
    assert op.max_real_part().item() < 0
    assert torch.allclose(
        op.max_real_part(),
        torch.linalg.eigvals(op.L).real.max(),
    )


def test_negative_strict_diagonal_stays_strict_at_raw_zero() -> None:
    """Verify raw=0 maps into (-bound, 0), not onto the imaginary axis."""
    raw = torch.zeros(3)
    values = negative_strict_diagonal_values(raw, max_real_eigenvalue=1.0)
    bound = 1.0 - STABILITY_EPS_MARGIN
    assert torch.all(values < 0)
    assert torch.all(values > -bound)
    assert torch.allclose(values, torch.full_like(values, -STABILITY_EPS_MARGIN))


def test_continuous_odo_bound_metric_vs_true_spectrum() -> None:
    """Verify continuous ODO bound_metric is a factor bound, not max Re(λ)."""
    torch.manual_seed(42)
    op = ContinuousKoopmanOperator(8, parameterization="odo", max_real_eigenvalue=1.0)
    found_unstable = False
    for _ in range(40):
        with torch.no_grad():
            op.cayley_O1.copy_(torch.randn_like(op.cayley_O1) * 8.0)
            op.cayley_O2.copy_(torch.randn_like(op.cayley_O2) * 8.0)
            op.diag_raw.copy_(torch.randn_like(op.diag_raw) * 2.0 + 3.0)
        factor = op.bound_metric()
        true_max_re = op.max_real_part()
        assert torch.allclose(true_max_re, torch.linalg.eigvals(op.L).real.max())
        if factor.item() < 0 and true_max_re.item() > 0:
            found_unstable = True
            break
    assert found_unstable
    assert op.stability_certificate() is None


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
    assert certificate.margin.item() > 0
    if parameterization == "lyapunov":
        assert certificate.lyapunov_matrix is not None
    else:
        assert certificate.lyapunov_matrix is None


def test_continuous_lyapunov_stability_certificate() -> None:
    """Verify continuous Lyapunov certificate satisfies LᵀP + PL ≺ 0."""
    op = ContinuousKoopmanOperator(4, parameterization="lyapunov", init_mode="identity")
    certificate = op.stability_certificate()
    assert certificate is not None
    assert certificate.lyapunov_matrix is not None
    assert certificate.margin.item() > 0
    p = certificate.lyapunov_matrix
    residual = op.L.T @ p + p @ op.L
    eigenvalues = torch.linalg.eigvalsh(residual)
    assert eigenvalues.max().item() < -1e-5


def test_controlled_van_loan_matches_analytic_diagonal_integral() -> None:
    """Verify Φ12 against ∫₀^Δt exp(L s) Bᵀ ds for a diagonal generator."""
    dtype = torch.float64
    delta_t = 0.25
    diag = torch.tensor([-0.5, -1.25, 0.0], dtype=dtype)
    generator = torch.diag(diag)
    control_matrix = torch.tensor(
        [[0.4, -0.1, 0.2], [0.0, 0.3, -0.5]],
        dtype=dtype,
    )
    phi11, phi12 = van_loan_factors(generator, control_matrix, delta_t)

    expected_phi11 = torch.diag(torch.exp(diag * delta_t))
    scales = torch.where(
        diag.abs() < 1e-12,
        torch.full_like(diag, delta_t),
        (torch.exp(diag * delta_t) - 1.0) / diag,
    )
    expected_phi12 = torch.diag(scales) @ control_matrix.T

    assert torch.allclose(phi11, expected_phi11, atol=1e-10)
    assert torch.allclose(phi12, expected_phi12, atol=1e-10)

    z = torch.tensor([[1.0, -0.5, 0.25]], dtype=dtype)
    u = torch.tensor([[0.3, -0.2]], dtype=dtype)
    expected_advance = z @ expected_phi11.T + u @ expected_phi12.T
    actual_advance = z @ phi11.T + u @ phi12.T
    assert torch.allclose(actual_advance, expected_advance, atol=1e-10)


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


def test_predict_at_matches_matrix_exp_propagation(
    synthetic_graph: Data,
) -> None:
    """predict_at must match encode → exp(L·t) → decode composition."""
    encoder = GNNEncoder(in_channels=3, hidden_channels=8, latent_dim=4, num_layers=2)
    decoder = GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=3, num_layers=2)
    operator = ContinuousKoopmanOperator(
        4,
        parameterization="dense",
        init_mode="identity_noise",
    )
    with torch.no_grad():
        operator.L.copy_(
            torch.tensor(
                [
                    [-0.5, 0.2, 0.0, 0.0],
                    [-0.2, -0.5, 0.0, 0.0],
                    [0.0, 0.0, -0.3, 0.1],
                    [0.0, 0.0, -0.1, -0.3],
                ]
            )
        )
    model = GraphKoopmanModel(
        encoder,
        decoder,
        latent_dim=4,
        time_step=0.1,
        dynamics_mode="continuous",
        koopman=operator,
    )
    model.eval()
    query_times = [0.1, 0.25, 0.4]
    with torch.no_grad():
        predictions = model.predict_at(synthetic_graph, query_times=query_times)
        z0 = model.encode(synthetic_graph)
        for prediction, absolute_time in zip(predictions, query_times, strict=True):
            expected_z = operator.advance(z0, absolute_time)
            expected_x = model.decoder(
                expected_z,
                synthetic_graph.edge_index,
            )
            assert torch.allclose(prediction.x, expected_x, atol=1e-5)

        by_deltas = model.predict_at(
            synthetic_graph,
            step_deltas=[0.1, 0.15, 0.15],
        )
        for query_pred, delta_pred in zip(predictions, by_deltas, strict=True):
            assert torch.allclose(query_pred.x, delta_pred.x, atol=1e-5)


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
        loss_weights=LossWeights(
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


def test_compute_generator_spectrum_known_rotation_decay() -> None:
    """Generator spectrum should match analytic Re/Im for a 2x2 rotation block."""
    omega = 2.0 * torch.pi * 0.25  # 0.25 cycles per unit time
    decay = -0.5
    generator = torch.tensor(
        [
            [decay, -omega],
            [omega, decay],
        ],
        dtype=torch.float64,
    )
    spectrum = compute_generator_spectrum(generator)
    assert spectrum.time_step == 1.0
    assert torch.allclose(
        spectrum.growth_rates,
        torch.full((2,), decay, dtype=torch.float64),
        atol=1e-8,
    )
    assert torch.allclose(
        spectrum.frequencies.abs(),
        torch.full((2,), 0.25, dtype=torch.float64),
        atol=1e-8,
    )

    delta_t = 0.5
    discrete = discrete_spectrum_at_delta_t(generator, delta_t)
    assert torch.allclose(
        discrete.growth_rates,
        spectrum.growth_rates,
        atol=1e-6,
    )
    assert torch.allclose(
        discrete.frequencies.abs(),
        spectrum.frequencies.abs(),
        atol=1e-6,
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
    penalty = EigenvalueRegularizationLoss()(
        operator,
        dynamics_mode="continuous",
    )
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
    loss_weights = LossWeights(
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


def test_continuous_operator_satisfies_contract() -> None:
    """Verify continuous operators implement KoopmanOperatorContract."""
    from koopman_graph.operators import (
        GeneratorParameterization,
        KoopmanOperatorContract,
        Parameterization,
    )

    assert GeneratorParameterization is Parameterization
    op = ContinuousKoopmanOperator(4, parameterization="dense", init_mode="identity")
    assert isinstance(op, KoopmanOperatorContract)
    z = torch.randn(3, 4)
    delta_t = 0.25
    assert torch.allclose(op.matrix, op.L)
    assert torch.allclose(op.bound_metric(), op.max_real_part())
    advanced = op.advance(z, delta_t)
    recovered = op.inverse_advance(advanced, delta_t)
    assert torch.allclose(recovered, z, atol=1e-4)


def test_continuous_advance_requires_delta_t() -> None:
    """Verify continuous advance rejects a missing integration interval."""
    op = ContinuousKoopmanOperator(3, parameterization="dense")
    z = torch.randn(2, 3)
    with pytest.raises(ValueError, match="delta_t is required"):
        op.advance(z)


def test_continuous_delta_t_policy_aligned_across_call_sites() -> None:
    """Verify model, training, losses, and env share resolve_delta_t policy."""
    torch.manual_seed(0)
    time_step = 0.25
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    snapshots = [Data(x=torch.randn(2, 2), edge_index=edge_index) for _ in range(4)]
    controls = torch.randn(4, 1)
    sequence = GraphSnapshotSequence(snapshots, control_inputs=controls)

    encoder = GNNEncoder(in_channels=2, hidden_channels=8, latent_dim=4, num_layers=1)
    decoder = GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=2, num_layers=1)
    model = GraphKoopmanModel(
        encoder,
        decoder,
        latent_dim=4,
        time_step=time_step,
        dynamics_mode="continuous",
        control_dim=1,
        koopman_parameterization="dissipative",
    )

    assert model.resolve_delta_t(None) == time_step
    assert model.resolve_delta_t(0.5) == 0.5

    from koopman_graph.graph_utils import resolve_delta_t as resolve_helper
    from koopman_graph.losses import ForwardConsistencyLoss
    from koopman_graph.training import one_step_loss
    from koopman_graph.training.pair_objectives import model_default_delta_t

    assert resolve_helper(None) == 1.0
    assert model_default_delta_t(model) == pytest.approx(time_step)
    assert resolve_pair_delta_t(
        sequence,
        0,
        default_time_step=model_default_delta_t(model),
    ) == pytest.approx(time_step)

    control0 = sequence.control_at(0)
    with torch.no_grad():
        pred = model(sequence[0], control=control0)
        z = model.encode(sequence[0])
        expected_z = model.koopman.advance(z, time_step, control=control0)
        expected = model.decoder(expected_z, sequence[0].edge_index)
        assert torch.allclose(pred, expected)

        wrong_z = model.koopman.advance(z, 1.0, control=control0)
        wrong = model.decoder(wrong_z, sequence[0].edge_index)
        assert not torch.allclose(pred, wrong, atol=1e-3)

        z_t1 = model.encode(sequence[1])
        loss_fn = ForwardConsistencyLoss()
        loss = loss_fn(
            z,
            z_t1,
            model.koopman,
            control=control0,
            delta_t=None,
            default_delta_t=model.resolve_delta_t(None),
        )
        expected_loss = torch.nn.functional.mse_loss(expected_z, z_t1)
        assert torch.allclose(loss, expected_loss)

        step_loss = one_step_loss(
            model,
            sequence[0],
            sequence[1],
            control=control0,
            delta_t=model.resolve_delta_t(None),
        )
        assert torch.allclose(
            step_loss,
            torch.nn.functional.mse_loss(pred, sequence[1].x),
        )

    from koopman_graph.env import GraphKoopmanEnv, flatten_latent

    env = GraphKoopmanEnv(
        model,
        sequence,
        lambda _s, _i: 0.0,
        random_start=False,
        start_index=0,
    )
    assert env._delta_t is None
    action = control0.detach().cpu().numpy().astype(np.float32)
    env.reset(seed=0)
    with torch.no_grad():
        z0 = model.encode(sequence[0])
        z_env = model.koopman.advance(z0, time_step, control=control0)
        expected_obs = flatten_latent(z_env)
    next_obs, *_ = env.step(action)
    np.testing.assert_allclose(next_obs, expected_obs, rtol=1e-5, atol=1e-5)


def test_auxiliary_spectral_rejected_on_discrete_and_graph() -> None:
    """auxiliary_spectral is continuous-only."""
    with pytest.raises(ValueError, match="continuous-only"):
        KoopmanOperator(4, parameterization="auxiliary_spectral")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="continuous-only"):
        GraphKoopmanOperator(4, parameterization="auxiliary_spectral")  # type: ignore[arg-type]


def test_auxiliary_spectral_generator_at_and_odd_dim() -> None:
    """Block-diagonal L(z) covers even and odd latent dims."""
    even = ContinuousKoopmanOperator(4, parameterization="auxiliary_spectral")
    z = torch.randn(4)
    generator = even.generator_at(z)
    assert generator.shape == (4, 4)
    spectrum = even.instantaneous_spectrum(z)
    assert spectrum.shape == (4,)
    with pytest.raises(ValueError, match="generator_at"):
        _ = even.L

    odd = ContinuousKoopmanOperator(
        3,
        parameterization="auxiliary_spectral",
        auxiliary_hidden_dims=(32,),
    )
    assert odd.auxiliary_hidden_dims == (32,)
    g_odd = odd.generator_at(torch.randn(3))
    assert g_odd.shape == (3, 3)


def test_auxiliary_spectral_batched_advance_gradients() -> None:
    """State-dependent advance is differentiable for batched latents."""
    op = ContinuousKoopmanOperator(4, parameterization="auxiliary_spectral")
    z = torch.randn(5, 4, requires_grad=True)
    next_z = op.advance(z, 0.1)
    assert next_z.shape == z.shape
    next_z.sum().backward()
    assert z.grad is not None


def test_auxiliary_spectral_controlled_additive_and_bilinear() -> None:
    """Controlled Van Loan advance works with frozen L(z)."""
    additive = ContinuousKoopmanOperator(
        4,
        parameterization="auxiliary_spectral",
        control_dim=2,
        control_mode="additive",
    )
    z = torch.randn(4)
    u = torch.randn(2)
    advanced = additive.advance(z, 0.05, control=u)
    assert advanced.shape == z.shape
    recovered = additive.inverse_advance(advanced, 0.05, control=u)
    assert recovered.shape == z.shape

    bilinear = ContinuousKoopmanOperator(
        4,
        parameterization="auxiliary_spectral",
        control_dim=1,
        control_mode="bilinear",
    )
    u_b = torch.tensor([0.2])
    advanced_b = bilinear.advance(z, 0.05, control=u_b)
    assert advanced_b.shape == z.shape


def test_auxiliary_spectral_configurable_hidden_dims_round_trip(
    tmp_path: Path,
) -> None:
    """Custom aux-net widths serialize under format-1 and reload."""
    encoder = GNNEncoder(in_channels=2, hidden_channels=8, latent_dim=4, num_layers=2)
    decoder = GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=2, num_layers=2)
    model = GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=4,
        time_step=0.1,
        dynamics_mode="continuous",
        koopman_parameterization="auxiliary_spectral",
        koopman_auxiliary_hidden_dims=(16, 8),
    )
    assert model.koopman.auxiliary_hidden_dims == (16, 8)
    path = tmp_path / "aux.pt"
    model.save(path)
    loaded = GraphKoopmanModel.load(path)
    assert loaded.koopman.parameterization == "auxiliary_spectral"
    assert loaded.koopman.auxiliary_hidden_dims == (16, 8)
    for key in model.state_dict():
        assert torch.allclose(model.state_dict()[key], loaded.state_dict()[key])


def _amplitude_dependent_oscillator_pairs(
    n_pairs: int = 256,
    delta_t: float = 0.1,
    seed: int = 0,
) -> tuple[Tensor, Tensor]:
    """Sample (z_t, z_{t+Δt}) from an amplitude-dependent planar oscillator."""
    generator = torch.Generator().manual_seed(seed)
    radii = 0.5 + 1.5 * torch.rand(n_pairs, generator=generator)
    angles = 2 * torch.pi * torch.rand(n_pairs, generator=generator)
    z0 = torch.stack([radii * torch.cos(angles), radii * torch.sin(angles)], dim=-1)
    omega = 1.0 + 0.75 * radii
    damp = 0.05
    # Locally exact step under frozen ω(r) at t0.
    mu = -damp
    advanced = []
    for idx in range(n_pairs):
        w = float(omega[idx])
        block = torch.tensor([[mu, -w], [w, mu]], dtype=torch.float32)
        advanced.append(z0[idx] @ torch.linalg.matrix_exp(block * delta_t).T)
    return z0, torch.stack(advanced, dim=0)


def _fit_operator_pairs(
    op: ContinuousKoopmanOperator,
    z0: Tensor,
    z1: Tensor,
    *,
    delta_t: float,
    epochs: int = 200,
    lr: float = 1e-2,
) -> float:
    """Minimize one-step MSE; return long-horizon rollout MSE on held-out starts."""
    opt = torch.optim.Adam(op.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        pred = op.advance(z0, delta_t)
        loss = torch.nn.functional.mse_loss(pred, z1)
        loss.backward()
        opt.step()

    with torch.no_grad():
        starts = z0[:32]
        horizon = 20
        roll_aux = starts.clone()
        target = starts.clone()
        for _ in range(horizon):
            # Ground-truth local step with frozen ω(r) each sub-step.
            radii = target.norm(dim=-1)
            omega = 1.0 + 0.75 * radii
            next_states = []
            for idx in range(target.shape[0]):
                w = float(omega[idx])
                block = torch.tensor([[-0.05, -w], [w, -0.05]], dtype=torch.float32)
                next_states.append(
                    target[idx] @ torch.linalg.matrix_exp(block * delta_t).T
                )
            target = torch.stack(next_states, dim=0)
            roll_aux = op.advance(roll_aux, delta_t)
        return float(torch.nn.functional.mse_loss(roll_aux, target).item())


def test_auxiliary_spectral_beats_dense_on_amplitude_oscillator() -> None:
    """Auxiliary mode should beat fixed dense L on amplitude-dependent dynamics."""
    delta_t = 0.1
    z0, z1 = _amplitude_dependent_oscillator_pairs(n_pairs=512, delta_t=delta_t)

    torch.manual_seed(0)
    aux = ContinuousKoopmanOperator(
        2,
        parameterization="auxiliary_spectral",
        auxiliary_hidden_dims=(64, 64),
        init_mode="identity",
    )
    torch.manual_seed(0)
    dense = ContinuousKoopmanOperator(2, parameterization="dense", init_mode="identity")

    aux_err = _fit_operator_pairs(aux, z0, z1, delta_t=delta_t)
    dense_err = _fit_operator_pairs(dense, z0, z1, delta_t=delta_t)
    assert aux_err < dense_err

    with torch.no_grad():
        radii = z0.norm(dim=-1)
        _mu, omega, _real = aux.auxiliary_net(z0)
        r_c = torch.corrcoef(torch.stack([radii, omega.reshape(-1)]))[0, 1]
        assert r_c.item() > 0.3


def test_model_spectrum_rejects_auxiliary_spectral() -> None:
    """Global model.spectrum must not pretend a fixed generator exists."""
    encoder = GNNEncoder(in_channels=2, hidden_channels=8, latent_dim=4, num_layers=2)
    decoder = GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=2, num_layers=2)
    model = GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=4,
        time_step=0.1,
        dynamics_mode="continuous",
        koopman_parameterization="auxiliary_spectral",
    )
    with pytest.raises(ValueError, match="generator_at"):
        model.spectrum()
