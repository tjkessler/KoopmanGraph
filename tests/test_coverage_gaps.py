"""Fill remaining branch/error-path coverage for the v0.3.0 95% gate."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn
from torch_geometric.data import Data

from koopman_graph import (
    ContinuousKoopmanOperator,
    GNNDecoder,
    GNNEncoder,
    GraphKoopmanEnv,
    GraphKoopmanModel,
    KoopmanOperator,
)
from koopman_graph.adaptation import RecursiveKoopmanAdapter
from koopman_graph.analysis import (
    calibrate_anomaly_threshold,
    compute_generator_spectrum,
    detect_anomaly,
    discrete_spectrum_at_delta_t,
    plot_spectrum,
)
from koopman_graph.analysis.anomaly import AnomalyDetectionResult
from koopman_graph.analysis.similarity import (
    _resolve_num_modes,
    _subspace_angle_distance,
    _wasserstein_magnitude_distance,
    resolve_spectrum,
)
from koopman_graph.baselines import DMDcBaseline
from koopman_graph.data import GraphSnapshotSequence, MultiTrajectory
from koopman_graph.datasets.topology import TopologyPayload
from koopman_graph.env import unflatten_latent
from koopman_graph.losses import (
    BackwardConsistencyLoss,
    EigenvalueRegularizationLoss,
    ForwardConsistencyLoss,
    masked_mse_loss,
    rollout_sequence_loss,
)
from koopman_graph.metrics import (
    evaluate_forecast,
    masked_mae,
    masked_mape,
    masked_rmse,
)
from koopman_graph.observables import (
    concatenate_observables,
    graph_laplacian_features,
    resolve_physics_lifting_fn,
    validate_physics_output,
)
from koopman_graph.operators import matrix_log, van_loan_generator_from_discrete
from koopman_graph.protocols import accepts_uncontrolled_data_predict
from koopman_graph.spectrum_types import KoopmanSpectrum
from koopman_graph.training.loop import (
    _classify_trajectory_items,
    resolve_validation_sequences,
)

gymnasium = pytest.importorskip("gymnasium")


def _edge_index() -> torch.Tensor:
    return torch.tensor([[0, 1], [1, 0]], dtype=torch.long)


def _tiny_model(
    *,
    latent_dim: int = 4,
    control_dim: int = 0,
    dynamics_mode: str = "discrete",
    parameterization: str = "dense",
    physics_dim: int = 0,
    physics_preset: str | None = None,
) -> GraphKoopmanModel:
    gnn_dim = latent_dim - physics_dim
    encoder = GNNEncoder(2, 8, gnn_dim)
    decoder = GNNDecoder(latent_dim, 8, 2)
    return GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=latent_dim,
        time_step=0.1,
        control_dim=control_dim,
        dynamics_mode=dynamics_mode,
        koopman_parameterization=parameterization,
        physics_dim=physics_dim,
        physics_preset=physics_preset,
    )


def _sequence(
    *,
    num_timesteps: int = 4,
    with_weights: bool = False,
    with_masks: bool = False,
    with_timestamps: bool = False,
    control_dim: int = 0,
) -> GraphSnapshotSequence:
    edge = _edge_index()
    weight = torch.ones(edge.shape[1]) if with_weights else None
    snapshots = [
        Data(
            x=torch.randn(2, 2),
            edge_index=edge,
            **({"edge_weight": weight} if weight is not None else {}),
        )
        for _ in range(num_timesteps)
    ]
    kwargs: dict[str, Any] = {}
    if with_masks:
        kwargs["observation_masks"] = torch.ones(num_timesteps, 2, dtype=torch.bool)
    if with_timestamps:
        kwargs["timestamps"] = torch.arange(num_timesteps, dtype=torch.float32)
    if control_dim > 0:
        kwargs["control_inputs"] = torch.randn(num_timesteps, control_dim)
    return GraphSnapshotSequence(snapshots, **kwargs)


# ---------------------------------------------------------------------------
# continuous.py
# ---------------------------------------------------------------------------


def test_matrix_log_preserves_complex_dtype() -> None:
    matrix = torch.eye(2, dtype=torch.complex64) * (0.5 + 0.1j)
    result = matrix_log(matrix)
    assert result.is_complex()


def test_van_loan_generator_rejects_nonpositive_delta_t() -> None:
    with pytest.raises(ValueError, match="delta_t must be positive"):
        van_loan_generator_from_discrete(torch.eye(2), torch.zeros(1, 2), 0.0)


def test_continuous_operator_constructor_validation() -> None:
    with pytest.raises(ValueError, match="latent_dim must be positive"):
        ContinuousKoopmanOperator(0)
    with pytest.raises(ValueError, match="init_scale must be non-negative"):
        ContinuousKoopmanOperator(2, init_scale=-0.1)
    with pytest.raises(ValueError, match="max_real_eigenvalue must be positive"):
        ContinuousKoopmanOperator(2, max_real_eigenvalue=0.0)
    with pytest.raises(ValueError, match="control_dim must be non-negative"):
        ContinuousKoopmanOperator(2, control_dim=-1)
    with pytest.raises(ValueError, match="Unknown parameterization"):
        ContinuousKoopmanOperator(2, parameterization="not-a-mode")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "parameterization",
    ["dense", "odo", "schur", "dissipative", "lyapunov"],
)
def test_continuous_xavier_init_modes(parameterization: str) -> None:
    op = ContinuousKoopmanOperator(
        3,
        parameterization=parameterization,  # type: ignore[arg-type]
        init_mode="xavier",
    )
    assert op.L.shape == (3, 3)


@pytest.mark.parametrize(
    "parameterization",
    ["dense", "odo", "schur", "lyapunov"],
)
def test_continuous_unknown_init_mode(parameterization: str) -> None:
    with pytest.raises(ValueError, match="Unknown init_mode"):
        ContinuousKoopmanOperator(
            2,
            parameterization=parameterization,  # type: ignore[arg-type]
            init_mode="not-real",  # type: ignore[arg-type]
        )


def test_continuous_reset_control_parameters_noop_without_control() -> None:
    op = ContinuousKoopmanOperator(2, control_dim=0)
    op.reset_control_parameters()


def test_continuous_control_term_and_advance_validation() -> None:
    uncontrolled = ContinuousKoopmanOperator(3, init_mode="identity")
    with pytest.raises(ValueError, match="control_term requires control_dim > 0"):
        uncontrolled.control_term(torch.ones(1))

    controlled = ContinuousKoopmanOperator(
        3, control_dim=2, init_mode="identity", parameterization="dense"
    )
    with pytest.raises(ValueError, match="Expected global control shape"):
        controlled.control_term(torch.ones(3))
    with pytest.raises(ValueError, match="Expected per-node control shape"):
        controlled.control_term(torch.ones(2, 3), num_nodes=2)
    with pytest.raises(ValueError, match="control input must have shape"):
        controlled.control_term(torch.ones(2, 2, 2), num_nodes=2)

    z = torch.randn(2, 3)
    with pytest.raises(ValueError, match="delta_t is required"):
        controlled.advance(z, None, control=torch.ones(2))
    with pytest.raises(ValueError, match="Expected trailing dimension"):
        controlled.advance(torch.randn(2, 4), 0.1, control=torch.ones(2))
    with pytest.raises(ValueError, match="ignored when delta_t is zero"):
        controlled.advance(z, 0.0, control=torch.ones(2))
    with pytest.raises(ValueError, match="uncontrolled operator"):
        uncontrolled.advance(z, 0.1, control=torch.ones(1))
    with pytest.raises(ValueError, match="control input is required"):
        controlled.advance(z, 0.1, control=None)
    with pytest.raises(ValueError, match="control input must have shape"):
        controlled._advance_controlled(z, torch.tensor(0.1), torch.ones(2, 2, 2))

    offset = controlled.control_term(torch.ones(2))
    assert offset.shape == (3,)
    per_node = controlled.control_term(torch.ones(2, 2), num_nodes=2)
    assert per_node.shape == (2, 3)
    advanced = controlled.advance(z, 0.2, control=torch.ones(2))
    assert advanced.shape == z.shape
    advanced_rows = controlled.advance(z, 0.2, control=torch.ones(2, 2))
    assert advanced_rows.shape == z.shape
    assert torch.allclose(controlled(z, control=torch.ones(2), delta_t=0.1), advanced)


def test_continuous_set_dense_matrix_and_inverse_paths() -> None:
    structured = ContinuousKoopmanOperator(2, parameterization="schur")
    with pytest.raises(ValueError, match="parameterization='dense'"):
        structured.set_dense_matrix(torch.eye(2))

    uncontrolled = ContinuousKoopmanOperator(2, init_mode="identity")
    with pytest.raises(ValueError, match="Expected generator shape"):
        uncontrolled.set_dense_matrix(torch.ones(2, 3))
    with pytest.raises(ValueError, match="control_matrix provided"):
        uncontrolled.set_dense_matrix(torch.eye(2), control_matrix=torch.ones(1, 2))
    uncontrolled._parameters.pop("L")
    with pytest.raises(AttributeError, match="L"):
        uncontrolled.set_dense_matrix(torch.eye(2))

    controlled = ContinuousKoopmanOperator(2, control_dim=1, init_mode="identity")
    with pytest.raises(ValueError, match="control_matrix is required"):
        controlled.set_dense_matrix(torch.eye(2))
    with pytest.raises(ValueError, match="Expected control_matrix shape"):
        controlled.set_dense_matrix(torch.eye(2), control_matrix=torch.ones(2, 2))
    controlled.set_dense_matrix(
        -torch.eye(2),
        control_matrix=torch.zeros(1, 2),
    )

    z = torch.randn(2, 2)
    with pytest.raises(ValueError, match="delta_t is required"):
        controlled.inverse_advance(z, None, control=torch.ones(1))
    with pytest.raises(ValueError, match="control input is required"):
        controlled.inverse_advance(z, 0.1, control=None)
    recovered = controlled.inverse_advance(z, 0.1, control=torch.ones(1))
    assert recovered.shape == z.shape
    recovered_rows = controlled.inverse_advance(z, 0.1, control=torch.ones(2, 1))
    assert recovered_rows.shape == z.shape


# ---------------------------------------------------------------------------
# adaptation.py
# ---------------------------------------------------------------------------


def test_adapter_constructor_validation_and_defaults() -> None:
    with pytest.raises(ValueError, match="latent_dim must be positive"):
        RecursiveKoopmanAdapter(0)
    with pytest.raises(ValueError, match="control_dim must be non-negative"):
        RecursiveKoopmanAdapter(2, control_dim=-1)
    with pytest.raises(ValueError, match="forgetting_factor"):
        RecursiveKoopmanAdapter(2, forgetting_factor=0.0)
    with pytest.raises(ValueError, match="regularization must be positive"):
        RecursiveKoopmanAdapter(2, regularization=0.0)

    discrete = RecursiveKoopmanAdapter(3, mode="discrete")
    assert torch.allclose(discrete.discrete_matrix, torch.eye(3))

    continuous = RecursiveKoopmanAdapter(3, mode="continuous", control_dim=1)
    assert continuous.generator_matrix.shape == (3, 3)
    assert continuous.control_matrix is not None
    assert continuous.generator_control_matrix is not None

    uncontrolled = RecursiveKoopmanAdapter(2, mode="continuous")
    assert uncontrolled.generator_control_matrix is None


def test_adapter_from_operator_continuous_and_type_errors() -> None:
    continuous = ContinuousKoopmanOperator(3, control_dim=1, init_mode="identity")
    adapter = RecursiveKoopmanAdapter.from_operator(continuous, mode="continuous")
    assert adapter.mode == "continuous"
    assert adapter.control_dim == 1

    class FakeOp(nn.Module):
        parameterization = "dense"
        latent_dim = 2
        control_dim = 0

    fake = FakeOp()
    with pytest.raises(TypeError, match="built-in"):
        RecursiveKoopmanAdapter.from_operator(fake, mode="discrete")  # type: ignore[arg-type]

    controlled = RecursiveKoopmanAdapter(
        2,
        control_dim=1,
        mode="discrete",
        initial_b=torch.zeros(1, 2),
    )
    with pytest.raises(ValueError, match="initial_b is required"):
        controlled._set_from_discrete(torch.eye(2), None)


def test_adapter_update_and_apply_validation() -> None:
    adapter = RecursiveKoopmanAdapter(3, mode="continuous")
    z = torch.randn(3)
    with pytest.raises(ValueError, match="delta_t is required"):
        adapter.update(z, z)
    with pytest.raises(ValueError, match="matching shapes"):
        adapter.update(torch.randn(2, 3), torch.randn(3, 3), delta_t=0.1)
    with pytest.raises(ValueError, match="delta_t must be positive"):
        adapter.update(z, z, delta_t=0.0)

    adapter.update(z, z, delta_t=0.2)
    continuous = ContinuousKoopmanOperator(3, init_mode="identity")
    adapter.apply_to(continuous)

    class FakeOp(nn.Module):
        parameterization = "dense"
        latent_dim = 3
        control_dim = 0

    with pytest.raises(TypeError, match="built-in"):
        adapter.apply_to(FakeOp())  # type: ignore[arg-type]


def test_adapter_row_and_control_validation() -> None:
    adapter = RecursiveKoopmanAdapter(3, control_dim=0)
    with pytest.raises(ValueError, match="Expected latent vector"):
        adapter._as_rows(torch.randn(2))
    with pytest.raises(ValueError, match="Expected trailing latent dimension"):
        adapter._as_rows(torch.randn(2, 2))
    with pytest.raises(ValueError, match="latent tensors must have shape"):
        adapter._as_rows(torch.randn(2, 2, 3))
    with pytest.raises(ValueError, match="uncontrolled adapter"):
        adapter._resolve_control_row(torch.ones(1), batch_rows=1)

    controlled = RecursiveKoopmanAdapter(
        3,
        control_dim=2,
        initial_b=torch.zeros(2, 3),
    )
    with pytest.raises(ValueError, match="control input is required"):
        controlled._resolve_control_row(None, batch_rows=1)
    with pytest.raises(ValueError, match="Expected global control shape"):
        controlled._resolve_control_row(torch.ones(3), batch_rows=1)
    with pytest.raises(ValueError, match="Expected per-node control shape"):
        controlled._resolve_control_row(torch.ones(2, 3), batch_rows=2)
    with pytest.raises(ValueError, match="must match latent rows"):
        controlled._resolve_control_row(torch.ones(3, 2), batch_rows=2)
    with pytest.raises(ValueError, match="control input must have shape"):
        controlled._resolve_control_row(torch.ones(2, 2, 2), batch_rows=2)

    assert controlled._resolve_control_row(torch.ones(2), batch_rows=4).shape == (2,)
    assert controlled._resolve_control_row(torch.ones(4, 2), batch_rows=4).shape == (
        4,
        2,
    )


# ---------------------------------------------------------------------------
# model.py
# ---------------------------------------------------------------------------


def test_model_constructor_and_injection_gaps() -> None:
    encoder = GNNEncoder(2, 8, 4)
    decoder = GNNDecoder(4, 8, 2)
    with pytest.raises(ValueError, match="dynamics_mode must be"):
        GraphKoopmanModel(
            encoder,
            decoder,
            latent_dim=4,
            time_step=0.1,
            dynamics_mode="hybrid",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="physics_dim must be non-negative"):
        GraphKoopmanModel(
            encoder,
            decoder,
            latent_dim=4,
            time_step=0.1,
            physics_dim=-1,
        )

    with pytest.raises(ValueError, match="Unknown physics_preset"):
        GraphKoopmanModel(
            encoder,
            decoder,
            latent_dim=4,
            time_step=0.1,
            physics_dim=2,
            physics_preset="missing",
        )

    injected = KoopmanOperator(4, init_mode="identity")
    with pytest.raises(ValueError, match="mutually exclusive"):
        GraphKoopmanModel(
            encoder,
            decoder,
            latent_dim=4,
            time_step=0.1,
            koopman=injected,
            koopman_init_mode="xavier",
            koopman_init_scale=0.2,
            koopman_max_spectral_radius=0.5,
        )
    with pytest.raises(ValueError, match="dynamics_mode='discrete'"):
        GraphKoopmanModel(
            encoder,
            decoder,
            latent_dim=4,
            time_step=0.1,
            dynamics_mode="continuous",
            koopman=injected,
        )


def test_model_adaptation_spectrum_and_predict_gaps() -> None:
    model = _tiny_model(dynamics_mode="continuous", parameterization="dense")
    assert model.uses_physics_observables is False
    assert model.online_adaptation_enabled is False
    with pytest.raises(RuntimeError, match="enable_online_adaptation"):
        model.adapt_step(
            Data(x=torch.randn(2, 2), edge_index=_edge_index()),
            Data(x=torch.randn(2, 2), edge_index=_edge_index()),
        )

    model.enable_online_adaptation()
    assert model.online_adaptation_enabled is True
    step = model.adapt_step(
        Data(x=torch.randn(2, 2), edge_index=_edge_index()),
        Data(x=torch.randn(2, 2), edge_index=_edge_index()),
    )
    assert torch.isfinite(step.operator_change_norm)
    model.disable_online_adaptation()
    assert model.online_adaptation_enabled is False

    spectrum = model.spectrum(delta_t=0.25)
    assert spectrum.eigenvalues.numel() == model.latent_dim

    with pytest.raises(ValueError, match="step_deltas for rollout"):
        model._rollout(
            Data(x=torch.randn(2, 2), edge_index=_edge_index()),
            steps=2,
            step_deltas=[0.1],
        )
    with pytest.raises(ValueError, match="step_deltas must be non-empty"):
        GraphKoopmanModel._resolve_time_increments(query_times=None, step_deltas=[])
    with pytest.raises(ValueError, match="query_times must be non-empty"):
        GraphKoopmanModel._resolve_time_increments(query_times=[], step_deltas=None)
    with pytest.raises(ValueError, match="strictly increasing"):
        GraphKoopmanModel._resolve_time_increments(
            query_times=[0.2, 0.1],
            step_deltas=None,
        )

    weighted = Data(
        x=torch.randn(2, 2),
        edge_index=_edge_index(),
        edge_weight=torch.ones(2),
    )
    preds = model.predict_at(weighted, query_times=[0.1, 0.2])
    assert len(preds) == 2
    assert preds[0].edge_weight is not None

    as_data = model._as_data(torch.randn(2, 2), _edge_index(), torch.ones(2))
    assert as_data.edge_weight is not None

    with pytest.raises(ValueError, match="control_dim > 0"):
        model.to_latent_env(_sequence(), reward_fn=lambda _s, _i: 0.0)


# ---------------------------------------------------------------------------
# env.py
# ---------------------------------------------------------------------------


def test_env_validation_reset_and_weighted_packaging() -> None:
    with pytest.raises(ValueError, match="flattened observation size"):
        unflatten_latent(np.zeros(3, dtype=np.float32), num_nodes=2, latent_dim=2)

    model = _tiny_model(control_dim=1)
    sequence = _sequence(with_weights=True)
    with pytest.raises(TypeError, match="GraphKoopmanModel"):
        GraphKoopmanEnv(SimpleNamespace(control_dim=1), sequence, lambda _s, _i: 0.0)  # type: ignore[arg-type]

    empty = SimpleNamespace(num_timesteps=0, is_dynamic_topology=False)
    with pytest.raises(ValueError, match="at least one snapshot"):
        GraphKoopmanEnv(model, empty, lambda _s, _i: 0.0)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="max_episode_steps"):
        GraphKoopmanEnv(model, sequence, lambda _s, _i: 0.0, max_episode_steps=0)
    with pytest.raises(ValueError, match="start_index must be in"):
        GraphKoopmanEnv(model, sequence, lambda _s, _i: 0.0, start_index=99)
    with pytest.raises(ValueError, match="delta_t must be positive"):
        GraphKoopmanEnv(model, sequence, lambda _s, _i: 0.0, delta_t=0.0)
    model2 = _tiny_model(control_dim=2)
    sequence2 = _sequence(control_dim=2, with_weights=True)
    with pytest.raises(ValueError, match="control_low must have shape"):
        GraphKoopmanEnv(model2, sequence2, lambda _s, _i: 0.0, control_low=[-1.0])
    with pytest.raises(ValueError, match="control_high must have shape"):
        GraphKoopmanEnv(model2, sequence2, lambda _s, _i: 0.0, control_high=[1.0])

    env = GraphKoopmanEnv(
        model,
        sequence,
        lambda _s, _i: 0.0,
        control_low=[-1.5],
        control_high=[1.5],
        start_index=None,
        random_start=False,
    )
    obs, _info = env.reset()
    assert obs.shape == (env.num_nodes * env.latent_dim,)
    reshaped = env.reshape_observation(obs)
    assert reshaped.shape == (env.num_nodes, env.latent_dim)

    obs2, info2 = env.reset(options={"start_index": 1})
    assert info2["start_index"] == 1
    assert obs2.shape == obs.shape
    with pytest.raises(ValueError, match="options\\['start_index'\\]"):
        env.reset(options={"start_index": 99})

    decoded = env._package_decoded(torch.randn(2, 2))
    assert decoded.edge_weight is not None


# ---------------------------------------------------------------------------
# residual mid-coverage modules
# ---------------------------------------------------------------------------


def test_containers_timestamp_and_mask_validation() -> None:
    edge = _edge_index()
    snapshots = [Data(x=torch.randn(2, 2), edge_index=edge) for _ in range(3)]
    with pytest.raises(ValueError, match="timestamps must have shape"):
        GraphSnapshotSequence(snapshots, timestamps=torch.ones(2, 2))
    with pytest.raises(ValueError, match="entries, expected"):
        GraphSnapshotSequence(snapshots, timestamps=torch.arange(2.0))
    with pytest.raises(ValueError, match="observation_masks"):
        GraphSnapshotSequence(
            snapshots,
            observation_masks=torch.ones(2, 2, dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="observation_masks"):
        GraphSnapshotSequence(
            snapshots,
            observation_masks=torch.ones(3, 3, dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="only 0 and 1"):
        GraphSnapshotSequence(
            snapshots,
            observation_masks=torch.full((3, 2), 2, dtype=torch.int64),
        )

    sequence = GraphSnapshotSequence(snapshots)
    with pytest.raises(ValueError, match="does not contain timestamps"):
        sequence.delta_t_at(0)
    with pytest.raises(ValueError, match="does not contain observation_masks"):
        sequence.observation_mask_at(0)
    with pytest.raises(ValueError, match="does not contain observation_masks"):
        sequence.pair_observation_mask(0)

    stamped = GraphSnapshotSequence(
        snapshots,
        timestamps=torch.tensor([0.0, 0.5, 1.0]),
    )
    with pytest.raises(ValueError, match="out of range"):
        stamped.delta_t_at(2)

    masked = GraphSnapshotSequence(
        snapshots,
        observation_masks=torch.ones(3, 2, dtype=torch.bool),
    )
    with pytest.raises(ValueError, match="out of range"):
        masked.observation_mask_at(3)
    with pytest.raises(ValueError, match="pair_observation_mask|out of range"):
        masked.pair_observation_mask(2)


def test_discrete_set_dense_matrix_and_inverse_gaps() -> None:
    structured = KoopmanOperator(2, parameterization="schur")
    with pytest.raises(ValueError, match="parameterization='dense'"):
        structured.set_dense_matrix(torch.eye(2))

    uncontrolled = KoopmanOperator(2, init_mode="identity")
    with pytest.raises(ValueError, match="Expected matrix shape"):
        uncontrolled.set_dense_matrix(torch.ones(2, 3))
    with pytest.raises(ValueError, match="control_matrix provided"):
        uncontrolled.set_dense_matrix(torch.eye(2), control_matrix=torch.ones(1, 2))
    uncontrolled._parameters.pop("K")
    with pytest.raises(AttributeError, match="K"):
        uncontrolled.set_dense_matrix(torch.eye(2))

    controlled = KoopmanOperator(2, control_dim=1, init_mode="identity")
    with pytest.raises(ValueError, match="control_matrix is required"):
        controlled.set_dense_matrix(torch.eye(2))
    with pytest.raises(ValueError, match="Expected control_matrix shape"):
        controlled.set_dense_matrix(torch.eye(2), control_matrix=torch.ones(2, 2))

    with pytest.raises(ValueError, match="Unknown init_mode"):
        KoopmanOperator(2, parameterization="schur", init_mode="nope")  # type: ignore[arg-type]

    op = KoopmanOperator(2, init_mode="identity")
    op.parameterization = "mystery"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="Unknown parameterization"):
        op._inverse_matrix()


def test_observables_and_losses_gaps() -> None:
    with pytest.raises(ValueError, match="data.x is required"):
        graph_laplacian_features(Data(edge_index=_edge_index()))
    with pytest.raises(ValueError, match="Unknown physics_preset"):
        resolve_physics_lifting_fn(physics_preset="nope")
    with pytest.raises(ValueError, match="share num_nodes"):
        concatenate_observables(torch.randn(2, 3), torch.randn(3, 1))
    with pytest.raises(ValueError, match="physics lifting output must have shape"):
        validate_physics_output(torch.randn(2, 3), physics_dim=1, num_nodes=2)

    pred = torch.randn(3, 2)
    target = torch.randn(3, 2)
    empty = masked_mse_loss(pred, target, torch.zeros(3, dtype=torch.bool))
    assert float(empty.item()) == 0.0
    # Force the dtype-cast branch with a bool mask already matching float path twice.
    _ = masked_mse_loss(pred, target, torch.ones(3, dtype=torch.float64))

    koopman = KoopmanOperator(3, init_mode="identity")
    z = torch.randn(2, 3)
    assert (
        ForwardConsistencyLoss()(
            z, z, koopman, mask=torch.zeros(2, dtype=torch.bool)
        ).item()
        == 0.0
    )
    assert (
        BackwardConsistencyLoss()(
            z, z, koopman, mask=torch.zeros(2, dtype=torch.bool)
        ).item()
        == 0.0
    )
    with pytest.raises(ValueError, match="dynamics_mode must be"):
        EigenvalueRegularizationLoss()(koopman, dynamics_mode="bogus")  # type: ignore[arg-type]

    model = _tiny_model()
    sequence = _sequence(with_masks=True)
    loss = rollout_sequence_loss(model, sequence, horizon=2, start=0)
    assert torch.isfinite(loss)


def test_analysis_metrics_protocol_and_training_gaps() -> None:
    with pytest.raises(ValueError, match="square matrix"):
        compute_generator_spectrum(torch.randn(2, 3))
    with pytest.raises(ValueError, match="non-empty"):
        compute_generator_spectrum(torch.empty(0, 0))
    with pytest.raises(TypeError, match="floating-point or complex"):
        compute_generator_spectrum(torch.ones(2, 2, dtype=torch.int64))
    with pytest.raises(ValueError, match="delta_t must be positive"):
        discrete_spectrum_at_delta_t(torch.eye(2), 0.0)

    spectrum = compute_generator_spectrum(torch.eye(2) * -0.5)
    spectrum_b = compute_generator_spectrum(torch.eye(2) * -0.4)
    threshold = calibrate_anomaly_threshold(
        [spectrum, spectrum_b],
        method="mean_std",
        k=1.0,
    )
    result = detect_anomaly([spectrum], spectrum, threshold=threshold)
    assert isinstance(result, AnomalyDetectionResult)

    class NoSig:
        def spectrum(self, *args: Any, **kwargs: Any) -> KoopmanSpectrum:
            return spectrum

    import inspect

    monkey_sig = pytest.MonkeyPatch()
    monkey_sig.setattr(
        inspect,
        "signature",
        lambda *_a, **_k: (_ for _ in ()).throw(TypeError("no signature")),
    )
    try:
        assert resolve_spectrum(NoSig(), delta_t=0.1).eigenvalues.numel() == 2
    finally:
        monkey_sig.undo()
    with pytest.raises(ValueError, match="num_modes must be in"):
        _resolve_num_modes(0, 2, 2)
    assert (
        float(_wasserstein_magnitude_distance(torch.tensor([]), torch.tensor([])))
        == 0.0
    )
    assert float(_subspace_angle_distance(torch.empty(2, 0), torch.empty(2, 0))) == 0.0
    plot_spectrum(torch.tensor([0.1 + 0.2j, 0.3 - 0.1j]), limits="data")
    mismatched = KoopmanSpectrum(
        eigenvalues=torch.tensor([0.1 + 0.0j, 0.2 + 0.0j]),
        eigenvectors=torch.eye(2, dtype=torch.complex64),
        magnitudes=torch.tensor([0.1]),
        growth_rates=torch.zeros(2),
        frequencies=torch.zeros(2),
        time_step=1.0,
    )
    plot_spectrum(mismatched, limits="unit_disk")

    values = torch.randn(3)
    assert masked_mae(values, values, torch.zeros(3, dtype=torch.bool)).item() == 0.0
    assert (
        masked_rmse(
            torch.randn(3, 2),
            torch.randn(3, 2),
            torch.zeros(3, dtype=torch.bool),
        ).item()
        == 0.0
    )
    assert (
        masked_mape(
            torch.randn(3, 2),
            torch.randn(3, 2),
            torch.zeros(3, dtype=torch.bool),
        ).item()
        == 0.0
    )

    model = _tiny_model()
    sequence = _sequence(with_masks=True, num_timesteps=5)
    metrics = evaluate_forecast(model, sequence, horizons=(1, 2))
    assert [item.horizon for item in metrics.horizons] == [1, 2]

    payload = TopologyPayload(edge_index=_edge_index(), num_nodes=2)
    assert list(payload) == ["edge_index", "num_nodes"]
    assert len(payload) == 2
    assert TopologyPayload.from_mapping(payload) is payload

    class BadPredict:
        control_dim = 0

        def predict(self, data: Data, steps: int, controls: Any) -> list[Data]:
            return []

        def fit(self, *args: Any, **kwargs: Any) -> Any:
            return self

    assert accepts_uncontrolled_data_predict(BadPredict()) is False

    class BrokenPredict:
        control_dim = 0
        predict = 123

        def fit(self, *args: Any, **kwargs: Any) -> Any:
            return self

    assert accepts_uncontrolled_data_predict(BrokenPredict()) is False

    baseline = DMDcBaseline()
    with pytest.raises(RuntimeError):
        baseline._require_control_matrix()
    with pytest.raises(RuntimeError):
        baseline._require_control_dim()

    seq = _sequence()
    with pytest.raises(ValueError, match="cannot mix"):
        _classify_trajectory_items(
            [seq, Data(x=torch.randn(2, 2), edge_index=_edge_index())],
            empty_message="empty",
        )
    with pytest.raises(TypeError, match="must be GraphSnapshotSequence or Data"):
        _classify_trajectory_items([seq, "bad"], empty_message="empty")  # type: ignore[list-item]
    with pytest.raises(ValueError, match="validation_sequence list length"):
        resolve_validation_sequences(
            MultiTrajectory([seq]),
            num_training_sequences=2,
        )
