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
from koopman_graph.datasets import (
    EpidemicNetworkBenchmark,
    KuramotoSivashinskyBenchmark,
    Lorenz96GraphBenchmark,
)
from koopman_graph.datasets.topology import TopologyPayload
from koopman_graph.env import GraphKoopmanEnv, unflatten_latent
from koopman_graph.hierarchical import HierarchicalGraphKoopmanModel
from koopman_graph.hierarchical.pooling import (
    ScatterUnpool,
    pool_control,
    pool_control_sequence,
)
from koopman_graph.losses import (
    BackwardConsistencyLoss,
    EigenvalueRegularizationLoss,
    ForwardConsistencyLoss,
    masked_mse_loss,
    rollout_sequence_loss,
)
from koopman_graph.losses.physics import PDEResidualLoss
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
from koopman_graph.operators import (
    matrix_log,
    normalize_auxiliary_hidden_dims,
    van_loan_generator_from_discrete,
)
from koopman_graph.operators.auxiliary_spectral import (
    advance_controlled as aux_advance_controlled,
)
from koopman_graph.operators.auxiliary_spectral import (
    assemble_block_diagonal_generator,
    split_auxiliary_spectrum,
)
from koopman_graph.operators.auxiliary_spectral import (
    inverse_advance_bilinear as aux_inverse_advance_bilinear,
)
from koopman_graph.operators.continuous_parameterizations import (
    max_real_part_of_generator,
    reset_dissipative_generator,
    reset_lyapunov_generator,
    reset_odo_generator,
    reset_schur_generator,
)
from koopman_graph.operators.continuous_propagation import (
    advance_controlled as cont_advance_controlled,
)
from koopman_graph.operators.continuous_propagation import (
    advance_interval,
    inverse_advance_interval,
)
from koopman_graph.operators.continuous_propagation import (
    inverse_advance_bilinear as cont_inverse_advance_bilinear,
)
from koopman_graph.operators.discrete_propagation import (
    advance_step as disc_advance_step,
)
from koopman_graph.operators.discrete_propagation import (
    inverse_bilinear as disc_inverse_bilinear,
)
from koopman_graph.operators.discrete_propagation import (
    inverse_step as disc_inverse_step,
)
from koopman_graph.protocols import accepts_uncontrolled_data_predict
from koopman_graph.spectrum_types import KoopmanSpectrum
from koopman_graph.training.extra_objectives import (
    compute_lie_consistency_loss,
    compute_pde_residual_loss,
)
from koopman_graph.training.history import ExtraLosses
from koopman_graph.training.inputs import (
    _classify_trajectory_items,
    resolve_validation_sequences,
)
from koopman_graph.training.objectives import (
    _graph_eigenvalue_regularization_over_sequence,
    _topologies_equal,
    compute_eigenvalue_regularization_loss,
)
from koopman_graph.uq import (
    EnsembleGraphKoopmanModel,
    LatentGaussianKoopmanUQ,
    empirical_coverage,
)
from koopman_graph.uq.common import PredictionInterval

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
        controlled.advance(z, 0.1, control=torch.ones(2, 2, 2))

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
# model/estimator.py
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
    from koopman_graph.model.timing import resolve_time_increments

    with pytest.raises(ValueError, match="step_deltas must be non-empty"):
        resolve_time_increments(query_times=None, step_deltas=[])
    with pytest.raises(ValueError, match="query_times must be non-empty"):
        resolve_time_increments(query_times=[], step_deltas=None)
    with pytest.raises(ValueError, match="strictly increasing"):
        resolve_time_increments(
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

    from koopman_graph.model.validation import as_data

    snapshot = as_data(torch.randn(2, 2), _edge_index(), torch.ones(2))
    assert snapshot.edge_weight is not None

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
    # Force float64 mask cast path (mask is converted to prediction dtype).
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
    with pytest.raises(TypeError, match="MultiTrajectory"):
        _classify_trajectory_items([seq, seq], empty_message="empty")
    with pytest.raises(TypeError, match="must be GraphSnapshotSequence or Data"):
        _classify_trajectory_items([seq, "bad"], empty_message="empty")  # type: ignore[list-item]
    with pytest.raises(ValueError, match="validation_sequence list length"):
        resolve_validation_sequences(
            MultiTrajectory([seq]),
            num_training_sequences=2,
        )


# ---------------------------------------------------------------------------
# One-time ≥95% branch-coverage fillers (v0.5.0 surface)
# ---------------------------------------------------------------------------


def test_auxiliary_spectral_validation_and_assembly() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        normalize_auxiliary_hidden_dims(())
    with pytest.raises(ValueError, match="positive integers"):
        normalize_auxiliary_hidden_dims((8, 0))
    assert normalize_auxiliary_hidden_dims(None)

    mu = torch.zeros(2)
    omega = torch.zeros(3)
    with pytest.raises(ValueError, match="share shape"):
        assemble_block_diagonal_generator(mu, omega, None)
    with pytest.raises(ValueError, match="trailing dimension must be 1"):
        assemble_block_diagonal_generator(
            torch.zeros(2), torch.zeros(2), torch.zeros(2)
        )
    with pytest.raises(ValueError, match="leading shape"):
        assemble_block_diagonal_generator(
            torch.zeros(2, 2),
            torch.zeros(2, 2),
            torch.zeros(3, 1),
        )
    even = assemble_block_diagonal_generator(torch.zeros(2), torch.ones(2), None)
    assert even.shape == (4, 4)
    odd = assemble_block_diagonal_generator(
        torch.zeros(1),
        torch.ones(1),
        torch.tensor([-0.5]),
    )
    assert odd.shape == (3, 3)
    mu_s, om_s, real_s = split_auxiliary_spectrum(
        torch.tensor([0.1, 0.2, 0.3, 0.4]),
        latent_dim=4,
    )
    assert real_s is None and mu_s.shape == (2,)
    _, _, real_odd = split_auxiliary_spectrum(
        torch.tensor([0.1, 0.2, -0.3]),
        latent_dim=3,
    )
    assert real_odd is not None and real_odd.shape == (1,)


def _noop_van_loan(
    state: torch.Tensor,
    dt: torch.Tensor,
    control: torch.Tensor,
    *,
    generator: torch.Tensor,
) -> torch.Tensor:
    return state


def test_auxiliary_and_continuous_controlled_propagation_branches() -> None:
    latent_dim = 4
    control_dim = 2
    z = torch.randn(latent_dim)
    z_nodes = torch.randn(3, latent_dim)
    z2 = torch.randn(2, latent_dim)
    generator = -0.1 * torch.eye(latent_dim)
    control_matrix = torch.randn(control_dim, latent_dim) * 0.05
    coupling = torch.randn(control_dim, latent_dim, latent_dim) * 0.01
    u = torch.randn(control_dim)
    u_nodes = torch.randn(3, control_dim)
    dt = torch.tensor(0.1)

    out = cont_advance_controlled(
        z,
        dt,
        u,
        generator=generator,
        control_matrix=control_matrix,
        control_mode="bilinear",
        latent_dim=latent_dim,
        coupling=coupling,
        parameterization="dense",
    )
    assert out.shape == z.shape
    out_n = cont_advance_controlled(
        z_nodes,
        dt,
        u_nodes,
        generator=generator,
        control_matrix=control_matrix,
        control_mode="bilinear",
        latent_dim=latent_dim,
        coupling=coupling,
        parameterization="dense",
    )
    assert out_n.shape == z_nodes.shape
    with pytest.raises(ValueError, match="bilinear coupling"):
        cont_advance_controlled(
            z,
            dt,
            u,
            generator=generator,
            control_matrix=control_matrix,
            control_mode="bilinear",
            latent_dim=latent_dim,
            coupling=None,
            parameterization="dense",
        )
    with pytest.raises(ValueError, match="matching node axis"):
        cont_advance_controlled(
            z,
            dt,
            u_nodes,
            generator=generator,
            control_matrix=control_matrix,
            control_mode="bilinear",
            latent_dim=latent_dim,
            coupling=coupling,
            parameterization="dense",
        )
    with pytest.raises(ValueError, match="control input must have shape"):
        cont_advance_controlled(
            z,
            dt,
            torch.randn(2, 2, control_dim),
            generator=generator,
            control_matrix=control_matrix,
            control_mode="bilinear",
            latent_dim=latent_dim,
            coupling=coupling,
            parameterization="dense",
        )

    with pytest.raises(ValueError, match="delta_t is required"):
        advance_interval(
            z,
            None,
            None,
            latent_dim=latent_dim,
            control_dim=0,
            control_mode="additive",
            parameterization="dense",
            generator=generator,
        )
    assert torch.allclose(
        advance_interval(
            z,
            0.0,
            None,
            latent_dim=latent_dim,
            control_dim=0,
            control_mode="additive",
            parameterization="dense",
            generator=generator,
        ),
        z,
    )
    with pytest.raises(ValueError, match="ignored when delta_t is zero"):
        advance_interval(
            z,
            0.0,
            u,
            latent_dim=latent_dim,
            control_dim=control_dim,
            control_mode="additive",
            parameterization="dense",
            generator=generator,
            control_matrix=control_matrix,
        )
    with pytest.raises(ValueError, match="control_matrix is required"):
        advance_interval(
            z,
            0.1,
            u,
            latent_dim=latent_dim,
            control_dim=control_dim,
            control_mode="additive",
            parameterization="dense",
            generator=generator,
            control_matrix=None,
        )

    z_adv = cont_advance_controlled(
        z,
        dt,
        u,
        generator=generator,
        control_matrix=control_matrix,
        control_mode="bilinear",
        latent_dim=latent_dim,
        coupling=coupling,
        parameterization="dense",
    )
    assert (
        cont_inverse_advance_bilinear(
            z_adv,
            dt,
            u,
            generator=generator,
            control_matrix=control_matrix,
            coupling=coupling,
            latent_dim=latent_dim,
        ).shape
        == z.shape
    )
    z_adv_n = cont_advance_controlled(
        z_nodes,
        dt,
        u_nodes,
        generator=generator,
        control_matrix=control_matrix,
        control_mode="bilinear",
        latent_dim=latent_dim,
        coupling=coupling,
        parameterization="dense",
    )
    assert (
        cont_inverse_advance_bilinear(
            z_adv_n,
            dt,
            u_nodes,
            generator=generator,
            control_matrix=control_matrix,
            coupling=coupling,
            latent_dim=latent_dim,
        ).shape
        == z_nodes.shape
    )
    with pytest.raises(ValueError, match="matching"):
        cont_inverse_advance_bilinear(
            z,
            dt,
            u_nodes,
            generator=generator,
            control_matrix=control_matrix,
            coupling=coupling,
            latent_dim=latent_dim,
        )
    with pytest.raises(ValueError, match="control input must have shape"):
        cont_inverse_advance_bilinear(
            z,
            dt,
            torch.randn(2, 2, control_dim),
            generator=generator,
            control_matrix=control_matrix,
            coupling=coupling,
            latent_dim=latent_dim,
        )

    with pytest.raises(ValueError, match="delta_t is required"):
        inverse_advance_interval(
            z,
            None,
            None,
            latent_dim=latent_dim,
            control_dim=0,
            control_mode="additive",
            parameterization="dense",
            generator=generator,
        )
    with pytest.raises(ValueError, match="control_matrix is required"):
        inverse_advance_interval(
            z,
            0.1,
            u,
            latent_dim=latent_dim,
            control_dim=control_dim,
            control_mode="additive",
            parameterization="dense",
            generator=generator,
            control_matrix=None,
        )
    with pytest.raises(ValueError, match="bilinear coupling"):
        inverse_advance_interval(
            z,
            0.1,
            u,
            latent_dim=latent_dim,
            control_dim=control_dim,
            control_mode="bilinear",
            parameterization="dense",
            generator=generator,
            control_matrix=control_matrix,
            coupling=None,
        )

    op = ContinuousKoopmanOperator(
        latent_dim=latent_dim,
        control_dim=control_dim,
        parameterization="auxiliary_spectral",
        control_mode="additive",
    )
    advanced = op.advance(z2, delta_t=0.1, control=u)
    assert op.inverse_advance(advanced, delta_t=0.1, control=u).shape == z2.shape

    op_b = ContinuousKoopmanOperator(
        latent_dim=latent_dim,
        control_dim=control_dim,
        parameterization="auxiliary_spectral",
        control_mode="bilinear",
    )
    advanced_b = op_b.advance(z, delta_t=0.1, control=u)
    assert op_b.inverse_advance(advanced_b, delta_t=0.1, control=u).shape == z.shape
    advanced_bn = op_b.advance(z_nodes, delta_t=0.1, control=u_nodes)
    assert (
        op_b.inverse_advance(advanced_bn, delta_t=0.1, control=u_nodes).shape
        == z_nodes.shape
    )

    gen_batch = generator.expand(2, latent_dim, latent_dim).clone()
    assert (
        aux_advance_controlled(
            z2,
            dt,
            u,
            generator=gen_batch,
            control_mode="additive",
            latent_dim=latent_dim,
            coupling=None,
            advance_van_loan=_noop_van_loan,
        ).shape
        == z2.shape
    )
    with pytest.raises(ValueError, match="matching node"):
        aux_advance_controlled(
            z,
            dt,
            u_nodes,
            generator=generator,
            control_mode="additive",
            latent_dim=latent_dim,
            coupling=None,
            advance_van_loan=_noop_van_loan,
        )
    with pytest.raises(ValueError, match="control input must have shape"):
        aux_advance_controlled(
            z,
            dt,
            torch.randn(2, 2, control_dim),
            generator=generator,
            control_mode="additive",
            latent_dim=latent_dim,
            coupling=None,
            advance_van_loan=_noop_van_loan,
        )
    with pytest.raises(ValueError, match="bilinear coupling"):
        aux_advance_controlled(
            z,
            dt,
            u,
            generator=generator,
            control_mode="bilinear",
            latent_dim=latent_dim,
            coupling=None,
            advance_van_loan=_noop_van_loan,
        )
    gen_nodes = generator.expand(3, latent_dim, latent_dim).clone()
    assert (
        aux_advance_controlled(
            z_nodes,
            dt,
            u_nodes,
            generator=gen_nodes,
            control_mode="additive",
            latent_dim=latent_dim,
            coupling=None,
            advance_van_loan=_noop_van_loan,
        ).shape
        == z_nodes.shape
    )
    assert (
        aux_advance_controlled(
            z2,
            dt,
            u,
            generator=gen_batch,
            control_mode="bilinear",
            latent_dim=latent_dim,
            coupling=coupling,
            advance_van_loan=_noop_van_loan,
        ).shape
        == z2.shape
    )
    assert (
        aux_advance_controlled(
            z_nodes,
            dt,
            u_nodes,
            generator=gen_nodes,
            control_mode="bilinear",
            latent_dim=latent_dim,
            coupling=coupling,
            advance_van_loan=_noop_van_loan,
        ).shape
        == z_nodes.shape
    )
    with pytest.raises(ValueError, match="matching node"):
        aux_advance_controlled(
            z,
            dt,
            u_nodes,
            generator=generator,
            control_mode="bilinear",
            latent_dim=latent_dim,
            coupling=coupling,
            advance_van_loan=_noop_van_loan,
        )
    with pytest.raises(ValueError, match="control input must have shape"):
        aux_advance_controlled(
            z,
            dt,
            torch.randn(2, 2, control_dim),
            generator=generator,
            control_mode="bilinear",
            latent_dim=latent_dim,
            coupling=coupling,
            advance_van_loan=_noop_van_loan,
        )

    assert (
        aux_inverse_advance_bilinear(
            z,
            dt,
            u,
            generator=generator,
            control_matrix=control_matrix,
            coupling=coupling,
            latent_dim=latent_dim,
        ).shape
        == z.shape
    )
    assert (
        aux_inverse_advance_bilinear(
            z2,
            dt,
            u,
            generator=gen_batch,
            control_matrix=control_matrix,
            coupling=coupling,
            latent_dim=latent_dim,
        ).shape
        == z2.shape
    )
    assert (
        aux_inverse_advance_bilinear(
            z_nodes,
            dt,
            u_nodes,
            generator=gen_nodes,
            control_matrix=control_matrix,
            coupling=coupling,
            latent_dim=latent_dim,
        ).shape
        == z_nodes.shape
    )


def test_discrete_propagation_bilinear_inverse_branches() -> None:
    latent_dim = 4
    control_dim = 2
    matrix = 0.5 * torch.eye(latent_dim)
    control_matrix = torch.randn(control_dim, latent_dim) * 0.05
    coupling = torch.randn(control_dim, latent_dim, latent_dim) * 0.01
    z = torch.randn(latent_dim)
    z_nodes = torch.randn(3, latent_dim)
    u = torch.randn(control_dim)
    u_nodes = torch.randn(3, control_dim)

    advanced = disc_advance_step(
        z,
        u,
        matrix=matrix,
        control_matrix=control_matrix,
        control_mode="bilinear",
        control_dim=control_dim,
        latent_dim=latent_dim,
        coupling=coupling,
    )
    assert advanced.shape == z.shape
    with pytest.raises(ValueError, match="bilinear coupling"):
        disc_advance_step(
            z,
            u,
            matrix=matrix,
            control_matrix=control_matrix,
            control_mode="bilinear",
            control_dim=control_dim,
            latent_dim=latent_dim,
            coupling=None,
        )

    assert (
        disc_inverse_bilinear(z, u, matrix=matrix, coupling=coupling).shape == z.shape
    )
    assert (
        disc_inverse_bilinear(z_nodes, u_nodes, matrix=matrix, coupling=coupling).shape
        == z_nodes.shape
    )
    with pytest.raises(ValueError, match="matching"):
        disc_inverse_bilinear(z, u_nodes, matrix=matrix, coupling=coupling)
    with pytest.raises(ValueError, match="control input must have shape"):
        disc_inverse_bilinear(
            z,
            torch.randn(2, 2, control_dim),
            matrix=matrix,
            coupling=coupling,
        )
    singular_k = torch.zeros(latent_dim, latent_dim)
    singular_coupling = torch.zeros(control_dim, latent_dim, latent_dim)
    _ = disc_inverse_bilinear(
        z,
        torch.zeros(control_dim),
        matrix=singular_k,
        coupling=singular_coupling,
    )
    _ = disc_inverse_bilinear(
        z_nodes,
        torch.zeros(3, control_dim),
        matrix=singular_k,
        coupling=singular_coupling,
    )
    inverse_matrix = torch.linalg.inv(matrix)
    assert (
        disc_inverse_step(
            advanced,
            control=u,
            matrix=matrix,
            inverse_matrix=inverse_matrix,
            control_matrix=control_matrix,
            control_mode="bilinear",
            control_dim=control_dim,
            latent_dim=latent_dim,
            coupling=coupling,
        ).shape
        == z.shape
    )


def test_continuous_parameterization_reset_and_certificate_guards() -> None:
    d = 4
    for init_mode in ("identity", "identity_noise", "xavier"):
        reset_odo_generator(
            torch.empty(d, d),
            torch.empty(d, d),
            torch.empty(d),
            init_mode=init_mode,  # type: ignore[arg-type]
            init_scale=0.01,
            max_real_eigenvalue=-0.1,
        )
        reset_schur_generator(
            torch.empty(d, d),
            torch.empty(d),
            torch.empty(d, d),
            init_mode=init_mode,  # type: ignore[arg-type]
            init_scale=0.01,
            max_real_eigenvalue=-0.1,
        )
        reset_dissipative_generator(
            torch.empty(d, d),
            init_mode=init_mode,  # type: ignore[arg-type]
            init_scale=0.01,
        )
        reset_lyapunov_generator(
            torch.empty(d, d),
            torch.empty(d),
            torch.empty(d),
            init_mode=init_mode,  # type: ignore[arg-type]
            init_scale=0.01,
            max_real_eigenvalue=-0.1,
        )
    with pytest.raises(ValueError, match="Unknown init_mode"):
        reset_odo_generator(
            torch.empty(d, d),
            torch.empty(d, d),
            torch.empty(d),
            init_mode="nope",  # type: ignore[arg-type]
            init_scale=0.01,
            max_real_eigenvalue=-0.1,
        )
    with pytest.raises(ValueError, match="Unknown init_mode"):
        reset_dissipative_generator(
            torch.empty(d, d),
            init_mode="nope",  # type: ignore[arg-type]
            init_scale=0.01,
        )
    assert float(max_real_part_of_generator(-torch.eye(d))) <= 0.0
    assert ContinuousKoopmanOperator(
        latent_dim=d,
        parameterization="lyapunov",
        max_real_eigenvalue=0.05,
    ).L.shape == (d, d)
    assert ContinuousKoopmanOperator(
        latent_dim=d,
        parameterization="dissipative",
    ).L.shape == (d, d)


def test_training_extra_objectives_and_graph_eigenvalue_paths() -> None:
    model = _tiny_model(dynamics_mode="discrete")
    seq = _sequence(num_timesteps=4)
    with pytest.raises(ValueError, match="continuous"):
        compute_lie_consistency_loss(
            model,
            seq,
            weight=1.0,
            extra_losses=ExtraLosses(lie_dynamics_fn=lambda snap: snap.x),
        )
    cont = _tiny_model(dynamics_mode="continuous")
    with pytest.raises(ValueError, match="lie_dynamics_fn"):
        compute_lie_consistency_loss(cont, seq, weight=1.0, extra_losses=None)
    with pytest.raises(ValueError, match="lie_dynamics_fn"):
        compute_lie_consistency_loss(cont, seq, weight=1.0, extra_losses=ExtraLosses())

    bad_seq = GraphSnapshotSequence(
        [
            Data(x=torch.randn(2, 2), edge_index=_edge_index()),
            Data(x=torch.randn(2, 2), edge_index=_edge_index()),
        ]
    )
    bad_seq[0].x = None
    with pytest.raises(ValueError, match="snapshot.x"):
        compute_lie_consistency_loss(
            cont,
            bad_seq,
            weight=1.0,
            extra_losses=ExtraLosses(
                lie_dynamics_fn=lambda snap: torch.zeros(2, 2),
            ),
        )
    value = compute_lie_consistency_loss(
        cont,
        seq,
        weight=1.0,
        extra_losses=ExtraLosses(lie_dynamics_fn=lambda snap: torch.zeros_like(snap.x)),
    )
    assert value.ndim == 0

    with pytest.raises(ValueError, match="pde_residual_fn"):
        compute_pde_residual_loss(cont, seq, weight=1.0, extra_losses=None)
    with pytest.raises(ValueError, match="pde_residual_fn"):
        compute_pde_residual_loss(cont, seq, weight=1.0, extra_losses=ExtraLosses())

    pde_loss = PDEResidualLoss()
    with pytest.raises(TypeError, match="Tensor"):
        pde_loss(seq[0].x, seq[0], pde_fn=lambda d, s: 1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        pde_loss(seq[0].x, seq[0], pde_fn=lambda d, s: torch.tensor([]))
    with pytest.raises(ValueError, match="first dimension"):
        pde_loss(
            seq[0].x,
            seq[0],
            pde_fn=lambda d, s: torch.tensor(1.0),
            mask=torch.tensor([True, False]),
        )
    empty_mask = torch.zeros(seq[0].x.shape[0], dtype=torch.bool)
    assert (
        float(
            pde_loss(
                seq[0].x,
                seq[0],
                pde_fn=lambda d, s: torch.ones(d.shape[0]),
                mask=empty_mask,
            )
        )
        == 0.0
    )
    assert (
        float(
            pde_loss(
                seq[0].x,
                seq[0],
                pde_fn=lambda d, s: torch.ones(d.shape[0]),
                mask=torch.tensor([True, False]),
            )
        )
        > 0.0
    )

    g_model = GraphKoopmanModel(
        encoder=GNNEncoder(2, 8, 4),
        decoder=GNNDecoder(4, 8, 2),
        latent_dim=4,
        time_step=0.1,
        koopman="graph",
        koopman_parameterization="dense",
    )
    short = GraphSnapshotSequence([Data(x=torch.randn(2, 2), edge_index=_edge_index())])
    with pytest.raises(ValueError, match="at least 2 snapshots"):
        _graph_eigenvalue_regularization_over_sequence(g_model, short)
    pen = compute_eigenvalue_regularization_loss(g_model, _sequence(num_timesteps=4))
    assert pen.ndim == 0

    ei = _edge_index()
    ew = torch.ones(ei.shape[1])
    snaps = [
        Data(x=torch.randn(2, 2), edge_index=ei.clone(), edge_weight=ew.clone())
        for _ in range(4)
    ]
    snaps[2] = Data(
        x=torch.randn(2, 2),
        edge_index=torch.tensor([[0, 1, 1], [1, 0, 0]], dtype=torch.long),
        edge_weight=torch.ones(3),
    )
    snaps[3] = Data(
        x=torch.randn(2, 2),
        edge_index=torch.tensor([[0, 1, 1], [1, 0, 0]], dtype=torch.long),
        edge_weight=torch.ones(3),
    )
    dynamic = GraphSnapshotSequence(snaps, allow_dynamic_topology=True)
    assert dynamic.is_dynamic_topology
    assert _graph_eigenvalue_regularization_over_sequence(g_model, dynamic).ndim == 0

    assert _topologies_equal(ei, ew, ei.clone(), ew.clone())
    assert not _topologies_equal(ei, ew, ei, None)
    assert not _topologies_equal(ei, None, ei, ew)
    assert _topologies_equal(ei, None, ei.clone(), None)
    assert not _topologies_equal(ei, ew, ei.flip(0), ew)


def test_uq_ensemble_and_latent_gaussian_gap_branches() -> None:
    members = [_tiny_model(latent_dim=4), _tiny_model(latent_dim=4)]
    for member in members:
        member.fit(_sequence(num_timesteps=6), epochs=1, lr=1e-2)
    ensemble = EnsembleGraphKoopmanModel(members)
    initial = _sequence(num_timesteps=3)[0]
    with pytest.raises(ValueError, match="steps must be"):
        ensemble.predict_interval(initial, steps=0)
    preds = ensemble.predict(initial, steps=2)
    assert len(preds) == 2
    future = [
        Data(x=torch.randn(2, 2), edge_index=_edge_index()),
        Data(x=torch.randn(2, 2), edge_index=_edge_index()),
    ]
    history = list(_sequence(num_timesteps=2))
    assert (
        len(
            ensemble.predict(
                initial,
                steps=2,
                future_topologies=future,
                history=history,
            )
        )
        == 2
    )
    interval = ensemble.predict_interval(initial, steps=2)
    assert isinstance(interval, PredictionInterval)
    with pytest.raises(ValueError, match="same number of steps"):
        empirical_coverage(preds[:1], interval)
    bad_interval = PredictionInterval(
        mean=interval.mean,
        lower=[Data(x=torch.randn(2, 3), edge_index=_edge_index()) for _ in range(2)],
        upper=interval.upper,
        level=interval.level,
        n_members=interval.n_members,
    )
    with pytest.raises(ValueError, match="feature shapes"):
        empirical_coverage(preds, bad_interval)

    g_model = GraphKoopmanModel(
        encoder=GNNEncoder(2, 8, 4),
        decoder=GNNDecoder(4, 8, 2),
        latent_dim=4,
        time_step=0.1,
        koopman="graph",
    )
    g_model.fit(_sequence(num_timesteps=6), epochs=1, lr=1e-2)
    uq = LatentGaussianKoopmanUQ(
        g_model,
        process_noise=1e-3,
        observation_noise=1e-2,
        n_samples=8,
    )
    obs = [
        Data(x=torch.randn(2, 2), edge_index=_edge_index()),
        Data(x=torch.randn(2, 2), edge_index=_edge_index()),
    ]
    forecast = uq.predict(
        initial,
        steps=2,
        observations=obs,
        future_topologies=future,
    )
    assert len(forecast) == 2
    with pytest.raises(ValueError, match="process_noise"):
        LatentGaussianKoopmanUQ(g_model, process_noise=0.0)


def test_hierarchical_pooling_and_resolution_gaps() -> None:
    base = GraphKoopmanModel(
        encoder=GNNEncoder(1, 4, 2),
        decoder=GNNDecoder(2, 4, 1),
        latent_dim=2,
        time_step=0.1,
    )
    model = HierarchicalGraphKoopmanModel(base, pool_ratios=(0.5,))
    from koopman_graph.datasets import SyntheticDynamicGraphBenchmark

    seq = SyntheticDynamicGraphBenchmark.generate(
        num_nodes=8,
        num_timesteps=6,
        in_channels=1,
        noise_std=0.0,
        seed=0,
    )
    model.fit(seq, epochs=1, lr=1e-2, unpool_epochs=0)
    with pytest.raises(ValueError, match="resolution"):
        model.predict(seq[0], steps=1, resolution="nope")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="resolution int"):
        model.predict(seq[0], steps=1, resolution=99)
    assert len(model.predict(seq[0], steps=1, resolution=0)) == 1
    assert len(model.predict(seq[0], steps=1, resolution="coarse")) == 1

    with pytest.raises(ValueError, match="control must have shape"):
        pool_control(torch.randn(2, 2, 2), [torch.tensor([0, 1])])
    with pytest.raises(ValueError, match="control_inputs must have shape"):
        pool_control_sequence(torch.randn(2), [[torch.tensor([0, 1])]])
    with pytest.raises(ValueError, match="perm chains"):
        pool_control_sequence(
            torch.randn(2, 8, 1),
            [[torch.arange(4)]],
        )
    pooled = pool_control(torch.randn(8, 1), [torch.arange(4)])
    assert pooled.shape == (4, 1)
    pooled_seq = pool_control_sequence(
        torch.randn(2, 8, 1),
        [[torch.arange(4)], [torch.arange(4)]],
    )
    assert pooled_seq.shape == (2, 4, 1)

    unpool = ScatterUnpool(channels=2, refine=True)
    unpool.train()
    out = unpool(torch.randn(1, 2), torch.tensor([0]), num_fine=2)
    assert out.shape[0] == 2


def test_dataset_validation_gap_branches() -> None:
    with pytest.raises(ValueError, match="num_nodes"):
        Lorenz96GraphBenchmark.generate(num_nodes=3, num_timesteps=5, burn_in=0)
    with pytest.raises(ValueError, match="dt"):
        Lorenz96GraphBenchmark.generate(num_nodes=4, num_timesteps=5, dt=0.0, burn_in=0)
    with pytest.raises(ValueError, match="burn_in"):
        Lorenz96GraphBenchmark.generate(num_nodes=4, num_timesteps=5, burn_in=-1)
    with pytest.raises(ValueError, match="num_nodes"):
        KuramotoSivashinskyBenchmark.generate(
            num_nodes=3,
            num_timesteps=5,
            burn_in=0,
        )
    with pytest.raises(ValueError, match="beta"):
        EpidemicNetworkBenchmark.generate(beta=-0.1)


def test_residual_coverage_to_95() -> None:
    """Hit remaining high-mass validation / inverse / factory branches."""
    from koopman_graph.operators.auxiliary_spectral import (
        AuxiliarySpectralNetwork,
        reset_auxiliary_network,
        split_auxiliary_spectrum,
    )
    from koopman_graph.operators.auxiliary_spectral import (
        inverse_advance as aux_inverse_advance,
    )
    from koopman_graph.operators.discrete_propagation import (
        inverse_matrix_for_parameterization,
    )
    from koopman_graph.uq.latent_gaussian import propagate_gaussian_covariance

    # auxiliary: bad spectral output dim, network ctor, inverse additive per-node,
    # reset modes
    with pytest.raises(ValueError, match="Expected auxiliary output"):
        split_auxiliary_spectrum(torch.zeros(3), latent_dim=4)
    with pytest.raises(ValueError, match="latent_dim must be positive"):
        AuxiliarySpectralNetwork(0)
    net = AuxiliarySpectralNetwork(4, hidden_dims=(8,))
    reset_auxiliary_network(net, init_mode="identity", init_scale=0.01)
    reset_auxiliary_network(net, init_mode="identity_noise", init_scale=0.01)
    reset_auxiliary_network(net, init_mode="xavier", init_scale=0.01)
    with pytest.raises(ValueError, match="Unknown init_mode"):
        reset_auxiliary_network(net, init_mode="nope", init_scale=0.01)  # type: ignore[arg-type]

    latent_dim = 4
    control_dim = 2
    z_nodes = torch.randn(3, latent_dim)
    u_nodes = torch.randn(3, control_dim)
    generator = (-0.1 * torch.eye(latent_dim)).expand(3, latent_dim, latent_dim).clone()
    control_matrix = torch.randn(control_dim, latent_dim) * 0.05
    coupling = torch.randn(control_dim, latent_dim, latent_dim) * 0.01
    dt = torch.tensor(0.1)

    with pytest.raises(ValueError, match="matching node"):
        aux_inverse_advance_bilinear(
            torch.randn(latent_dim),
            dt,
            u_nodes,
            generator=generator[0],
            control_matrix=control_matrix,
            coupling=coupling,
            latent_dim=latent_dim,
        )
    with pytest.raises(ValueError, match="control input must have shape"):
        aux_inverse_advance_bilinear(
            z_nodes,
            dt,
            torch.randn(2, 2, control_dim),
            generator=generator,
            control_matrix=control_matrix,
            coupling=coupling,
            latent_dim=latent_dim,
        )
    # additive inverse via inverse_advance (uncontrolled + per-node + bad ndim)
    recovered = aux_inverse_advance(
        z_nodes,
        dt,
        None,
        generator=generator,
        control_dim=0,
        control_mode="additive",
        latent_dim=latent_dim,
        control_matrix=None,
        coupling=None,
    )
    assert recovered.shape == z_nodes.shape
    recovered = aux_inverse_advance(
        z_nodes,
        dt,
        u_nodes,
        generator=generator,
        control_dim=control_dim,
        control_mode="additive",
        latent_dim=latent_dim,
        control_matrix=control_matrix,
        coupling=None,
    )
    assert recovered.shape == z_nodes.shape
    with pytest.raises(ValueError, match="matching node"):
        aux_inverse_advance(
            torch.randn(latent_dim),
            dt,
            u_nodes,
            generator=generator[0],
            control_dim=control_dim,
            control_mode="additive",
            latent_dim=latent_dim,
            control_matrix=control_matrix,
            coupling=None,
        )
    with pytest.raises(ValueError, match="control input must have shape"):
        aux_inverse_advance(
            z_nodes,
            dt,
            torch.randn(2, 2, control_dim),
            generator=generator,
            control_dim=control_dim,
            control_mode="additive",
            latent_dim=latent_dim,
            control_matrix=control_matrix,
            coupling=None,
        )
    with pytest.raises(ValueError, match="bilinear coupling"):
        aux_inverse_advance(
            z_nodes,
            dt,
            u_nodes,
            generator=generator,
            control_dim=control_dim,
            control_mode="bilinear",
            latent_dim=latent_dim,
            control_matrix=control_matrix,
            coupling=None,
        )

    # continuous parameterization bound / certificate helpers
    from koopman_graph.operators.continuous_parameterizations import (
        continuous_bound_metric,
        continuous_stability_certificate,
    )

    gen = -0.2 * torch.eye(4)
    _ = continuous_bound_metric(
        "dense",
        max_real_eigenvalue=0.1,
        assembled_generator=gen,
    )
    with pytest.raises(ValueError, match="auxiliary_spectral"):
        continuous_bound_metric(
            "auxiliary_spectral",
            max_real_eigenvalue=0.1,
        )
    with pytest.raises(ValueError, match="requires dissipative_generator"):
        continuous_bound_metric(
            "dissipative",
            max_real_eigenvalue=0.1,
        )
    _ = continuous_bound_metric(
        "dissipative",
        max_real_eigenvalue=0.1,
        dissipative_generator=torch.eye(4),
    )
    with pytest.raises(ValueError, match="diagonal raw"):
        continuous_bound_metric(
            "odo",
            max_real_eigenvalue=0.1,
        )
    _ = continuous_bound_metric(
        "odo",
        max_real_eigenvalue=0.1,
        diag_raw=torch.zeros(4),
    )
    _ = continuous_stability_certificate(
        "lyapunov",
        lyapunov_diagonal=-torch.ones(4),
        lyapunov_matrix=torch.eye(4),
    )
    with pytest.raises(ValueError, match="lyapunov_diagonal"):
        continuous_stability_certificate("lyapunov")
    _ = continuous_stability_certificate(
        "schur",
        bound_metric=torch.tensor(-0.1),
    )
    with pytest.raises(ValueError, match="bound_metric"):
        continuous_stability_certificate("dissipative")
    assert continuous_stability_certificate("dense") is None

    # discrete inverse_matrix_for_parameterization branches
    eye = torch.eye(4)
    skew = torch.zeros(4, 4)
    _ = inverse_matrix_for_parameterization(
        "odo",
        odo_left=eye,
        odo_right=eye,
        odo_diagonal=torch.ones(4) * 0.5,
    )
    _ = inverse_matrix_for_parameterization(
        "schur",
        schur_cayley_q=skew,
        schur_triangular=torch.triu(eye * 0.5),
    )
    _ = inverse_matrix_for_parameterization(
        "dissipative",
        dissipative_generator=eye,
    )
    _ = inverse_matrix_for_parameterization(
        "lyapunov",
        lyapunov_cayley_q=skew,
        lyapunov_diagonal=torch.ones(4) * 0.5,
    )
    with pytest.raises(ValueError, match="Unknown parameterization"):
        inverse_matrix_for_parameterization("nope")  # type: ignore[arg-type]

    # latent gaussian helpers / ctor guards
    a = 0.5 * torch.eye(4)
    p = torch.eye(4)
    _ = propagate_gaussian_covariance(a, p, 1e-3)
    _ = propagate_gaussian_covariance(a, p, torch.eye(4) * 1e-3)
    with pytest.raises(ValueError, match="share shape"):
        propagate_gaussian_covariance(a, torch.eye(3), 1e-3)
    with pytest.raises(ValueError, match="process_noise tensor must match"):
        propagate_gaussian_covariance(a, p, torch.eye(3))
    model = _tiny_model()
    with pytest.raises(ValueError, match="observation_noise"):
        LatentGaussianKoopmanUQ(model, observation_noise=0.0)
    with pytest.raises(ValueError, match="initial_covariance"):
        LatentGaussianKoopmanUQ(model, initial_covariance=0.0)
    with pytest.raises(ValueError, match="n_samples"):
        LatentGaussianKoopmanUQ(model, n_samples=0)

    # epidemic validation tails
    with pytest.raises(ValueError, match="num_nodes"):
        EpidemicNetworkBenchmark.generate(num_nodes=0, num_timesteps=5)
    with pytest.raises(ValueError, match="patient_zero"):
        EpidemicNetworkBenchmark.generate(patient_zero=1.5)
    with pytest.raises(ValueError, match="patient_zero \\+ neighbor_seed"):
        EpidemicNetworkBenchmark.generate(patient_zero=0.8, neighbor_seed=0.8)
    with pytest.raises(ValueError, match="small_world"):
        EpidemicNetworkBenchmark.generate(
            num_nodes=2,
            topology="small_world",
        )
    with pytest.raises(ValueError, match="even integer"):
        EpidemicNetworkBenchmark.generate(
            num_nodes=8,
            topology="small_world",
            small_world_k=3,
        )
    with pytest.raises(ValueError, match="rewire_prob"):
        EpidemicNetworkBenchmark.generate(
            num_nodes=8,
            topology="small_world",
            rewire_prob=1.5,
        )
    EpidemicNetworkBenchmark.generate(
        num_nodes=8,
        num_timesteps=5,
        topology="small_world",
        small_world_k=2,
        rewire_prob=0.5,
        seed=0,
    )
    with pytest.raises(ValueError, match="edge_index"):
        EpidemicNetworkBenchmark.generate(topology="custom", edge_index=None)
    edges = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    with pytest.raises(ValueError, match="references node"):
        EpidemicNetworkBenchmark.generate(
            num_nodes=1,
            topology="custom",
            edge_index=edges,
        )

    # KS burn_in / dt
    with pytest.raises(ValueError, match="burn_in"):
        KuramotoSivashinskyBenchmark.generate(
            num_nodes=8,
            num_timesteps=5,
            burn_in=-1,
        )
    with pytest.raises(ValueError, match="dt"):
        KuramotoSivashinskyBenchmark.generate(
            num_nodes=8,
            num_timesteps=5,
            dt=0.0,
            burn_in=0,
        )

    # ensemble empty / empirical_coverage empty / missing x
    with pytest.raises(ValueError, match="at least one member"):
        EnsembleGraphKoopmanModel([])
    with pytest.raises(ValueError, match="at least one snapshot"):
        empirical_coverage(
            [], PredictionInterval(mean=(), lower=(), upper=(), level=0.9, n_members=1)
        )
    interval = PredictionInterval(
        mean=(Data(x=torch.randn(2, 2), edge_index=_edge_index()),),
        lower=(Data(x=torch.randn(2, 2), edge_index=_edge_index()),),
        upper=(Data(x=torch.randn(2, 2), edge_index=_edge_index()),),
        level=0.9,
        n_members=1,
    )
    with pytest.raises(ValueError, match="node features"):
        empirical_coverage(
            [Data(edge_index=_edge_index())],
            interval,
        )

    # observables validation tails
    with pytest.raises(ValueError, match="physics lifting output"):
        validate_physics_output(torch.randn(2, 3), physics_dim=2, num_nodes=2)
    with pytest.raises(ValueError, match="Unknown physics_preset"):
        resolve_physics_lifting_fn(physics_preset="not_a_preset")

    # factory / model injection residual via ContinuousKoopmanOperator odd dim
    op_odd = ContinuousKoopmanOperator(
        latent_dim=3,
        parameterization="auxiliary_spectral",
    )
    z_odd = torch.randn(3)
    assert op_odd.advance(z_odd, delta_t=0.05).shape == (3,)


def test_additional_coverage_gap_branches() -> None:
    """Hit remaining high-value error/happy branches across modules."""
    from koopman_graph.data.construction import build_windowed_snapshots
    from koopman_graph.losses import LieConsistencyLoss, WorstCaseReconstructionLoss
    from koopman_graph.observables import (
        graph_gradient_features,
        make_polynomial_features,
        polynomial_features,
    )
    from koopman_graph.operators.control import (
        map_control_term,
        write_dense_operator_parameters,
    )
    from koopman_graph.operators.discrete_propagation import (
        inverse_matrix_for_parameterization,
    )

    # --- observables guards ---
    edge = _edge_index()
    with pytest.raises(ValueError, match="data.x is required for graph_gradient"):
        graph_gradient_features(Data(edge_index=edge))
    with pytest.raises(ValueError, match="data.x must be 2D"):
        graph_gradient_features(Data(x=torch.randn(2), edge_index=edge))
    with pytest.raises(ValueError, match="degree must be"):
        polynomial_features(Data(x=torch.randn(2, 2), edge_index=edge), degree=0)
    with pytest.raises(ValueError, match="data.x is required for polynomial"):
        polynomial_features(Data(edge_index=edge), degree=2)
    with pytest.raises(ValueError, match="data.x must be 2D"):
        polynomial_features(Data(x=torch.randn(2), edge_index=edge), degree=2)
    with pytest.raises(ValueError, match="degree must be"):
        make_polynomial_features(0)

    # --- worst-case reconstruction ---
    pred = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    target = torch.zeros_like(pred)
    with pytest.raises(ValueError, match="mask length"):
        WorstCaseReconstructionLoss()(
            pred, target, mask=torch.ones(3, dtype=torch.bool)
        )
    assert (
        WorstCaseReconstructionLoss()(
            pred, target, mask=torch.zeros(2, dtype=torch.bool)
        ).item()
        == 0.0
    )

    # --- LieConsistencyLoss generator_at failures ---
    class _BadShape(nn.Module):
        control_dim = 0

        def generator_at(self, z: torch.Tensor) -> torch.Tensor:
            return torch.ones(z.shape[-1] + 1, z.shape[-1] + 1)

    class _NotCallable(nn.Module):
        control_dim = 0
        generator_at = 3

    state = torch.randn(2, requires_grad=True)

    def obs(z: torch.Tensor) -> torch.Tensor:
        return z

    with pytest.raises(TypeError, match="generator_at"):
        LieConsistencyLoss()(
            state,
            observable_fn=obs,
            dynamics_fn=lambda value: value,
            koopman=_NotCallable(),
        )
    with pytest.raises(ValueError, match="generator_at output"):
        LieConsistencyLoss()(
            state,
            observable_fn=obs,
            dynamics_fn=lambda value: value,
            koopman=_BadShape(),
        )

    # --- windowed construction validation + metadata ---
    snaps = [
        Data(x=torch.randn(2, 2), edge_index=edge, edge_weight=torch.ones(2))
        for _ in range(4)
    ]
    seq = GraphSnapshotSequence(
        snaps,
        control_inputs=torch.randn(4, 1),
        timestamps=torch.arange(4, dtype=torch.float32),
        observation_masks=torch.ones(4, 2, dtype=torch.bool),
    )
    with pytest.raises(ValueError, match="n_delays must be"):
        build_windowed_snapshots(seq, n_delays=0)
    with pytest.raises(ValueError, match="stride must be"):
        build_windowed_snapshots(seq, n_delays=2, stride=0)
    short = GraphSnapshotSequence(snaps[:1])
    with pytest.raises(ValueError, match="sequence too short"):
        build_windowed_snapshots(short, n_delays=3, pad=False)
    built = build_windowed_snapshots(seq, n_delays=2, pad=True)
    assert built.control_inputs is not None
    assert built.timestamps is not None
    assert built.observation_masks is not None
    assert built.snapshots[0].edge_weight is not None

    # --- factory reject paths ---
    enc = GNNEncoder(2, 4, 2)
    dec = GNNDecoder(2, 4, 2)
    with pytest.raises(ValueError, match="n_delays must be"):
        GraphKoopmanModel(enc, dec, latent_dim=2, time_step=0.1, n_delays=0)
    with pytest.raises(ValueError, match="koopman string kind"):
        GraphKoopmanModel(enc, dec, latent_dim=2, time_step=0.1, koopman="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="koopman_auxiliary_hidden_dims"):
        GraphKoopmanModel(
            enc,
            dec,
            latent_dim=2,
            time_step=0.1,
            koopman_auxiliary_hidden_dims=(8,),
        )
    with pytest.raises(ValueError, match="koopman='graph' requires"):
        GraphKoopmanModel(
            enc,
            dec,
            latent_dim=2,
            time_step=0.1,
            koopman="graph",
            dynamics_mode="continuous",
        )
    with pytest.raises(ValueError, match="koopman_auxiliary_hidden_dims requires"):
        GraphKoopmanModel(
            enc,
            dec,
            latent_dim=2,
            time_step=0.1,
            dynamics_mode="continuous",
            koopman_auxiliary_hidden_dims=(8,),
            koopman_parameterization="dense",
        )
    with pytest.raises(ValueError, match="koopman_auxiliary_hidden_dims requires"):
        GraphKoopmanModel(
            enc,
            dec,
            latent_dim=2,
            time_step=0.1,
            koopman="graph",
            koopman_auxiliary_hidden_dims=(8,),
        )

    injected = KoopmanOperator(2)
    with pytest.raises(ValueError, match="mutually exclusive"):
        GraphKoopmanModel(
            enc,
            dec,
            latent_dim=2,
            time_step=0.1,
            koopman=injected,
            koopman_init_mode="identity",
        )
    with pytest.raises(ValueError, match="control_mode"):
        GraphKoopmanModel(
            GNNEncoder(2, 4, 2),
            GNNDecoder(2, 4, 2),
            latent_dim=2,
            time_step=0.1,
            control_dim=1,
            control_mode="bilinear",
            koopman=KoopmanOperator(2, control_dim=1, control_mode="additive"),
        )
    with pytest.raises(ValueError, match="bilinear_rank"):
        GraphKoopmanModel(
            GNNEncoder(2, 4, 2),
            GNNDecoder(2, 4, 2),
            latent_dim=2,
            time_step=0.1,
            control_dim=1,
            control_mode="bilinear",
            bilinear_rank=1,
            koopman=KoopmanOperator(
                2, control_dim=1, control_mode="bilinear", bilinear_rank=None
            ),
        )
    with pytest.raises(ValueError, match="dynamics_mode='discrete'"):
        GraphKoopmanModel(
            enc,
            dec,
            latent_dim=2,
            time_step=0.1,
            dynamics_mode="continuous",
            koopman=KoopmanOperator(2),
        )

    # --- map_control_term / write_dense_operator_parameters guards ---
    with pytest.raises(ValueError, match="control_dim > 0"):
        map_control_term(torch.ones(1), control_matrix=torch.ones(1, 2), control_dim=0)
    with pytest.raises(ValueError, match="control matrix B"):
        map_control_term(torch.ones(1), control_matrix=None, control_dim=1)

    dense = nn.Parameter(torch.eye(2))
    with pytest.raises(ValueError, match="control_parameter is required"):
        write_dense_operator_parameters(
            dense,
            torch.eye(2),
            control_dim=1,
            control_parameter=None,
            control_matrix=torch.ones(1, 2),
            control_mode="additive",
            bilinear_rank=None,
            bilinear_parameter=None,
            bilinear_matrices=None,
            latent_dim=2,
            matrix_label="K",
        )
    with pytest.raises(ValueError, match="uncontrolled operator"):
        write_dense_operator_parameters(
            dense,
            torch.eye(2),
            control_dim=0,
            control_parameter=None,
            control_matrix=torch.ones(1, 2),
            control_mode="additive",
            bilinear_rank=None,
            bilinear_parameter=None,
            bilinear_matrices=None,
            latent_dim=2,
            matrix_label="K",
        )

    # --- discrete inverse construction missing factors ---
    with pytest.raises(ValueError, match="dense_matrix is required"):
        inverse_matrix_for_parameterization(parameterization="dense")
    with pytest.raises(ValueError, match="ODO factors"):
        inverse_matrix_for_parameterization(parameterization="odo")
    with pytest.raises(ValueError, match="Schur factors"):
        inverse_matrix_for_parameterization(parameterization="schur")
    with pytest.raises(ValueError, match="dissipative generator"):
        inverse_matrix_for_parameterization(parameterization="dissipative")
    with pytest.raises(ValueError, match="Lyapunov factors"):
        inverse_matrix_for_parameterization(parameterization="lyapunov")
    with pytest.raises(ValueError, match="Unknown parameterization"):
        inverse_matrix_for_parameterization(parameterization="nope")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="control input is required"):
        disc_inverse_step(
            torch.randn(2),
            control=None,
            matrix=torch.eye(2),
            inverse_matrix=torch.eye(2),
            control_matrix=torch.ones(1, 2),
            coupling=None,
            control_dim=1,
            control_mode="additive",
            latent_dim=2,
        )
    with pytest.raises(ValueError, match="bilinear coupling"):
        disc_inverse_step(
            torch.randn(2),
            control=torch.ones(1),
            matrix=torch.eye(2),
            inverse_matrix=torch.eye(2),
            control_matrix=torch.ones(1, 2),
            coupling=None,
            control_dim=1,
            control_mode="bilinear",
            latent_dim=2,
        )
    with pytest.raises(ValueError, match="inverse_matrix is required"):
        disc_inverse_step(
            torch.randn(2),
            control=None,
            matrix=torch.eye(2),
            inverse_matrix=None,
            control_matrix=None,
            coupling=None,
            control_dim=0,
            control_mode="additive",
            latent_dim=2,
        )

    # --- latent Gaussian n_samples=1 quantile branch ---
    model = _tiny_model(latent_dim=2)
    seq2 = _sequence(num_timesteps=4)
    model.fit(seq2, epochs=1)
    uq = LatentGaussianKoopmanUQ(model, n_samples=1)
    interval = uq.predict_interval(seq2[0], steps=1, level=0.9)
    assert len(interval.mean) == 1
    assert torch.allclose(interval.lower[0].x, interval.mean[0].x)
