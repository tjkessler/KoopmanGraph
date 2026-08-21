"""Coverage and error-path tests for :mod:`koopman_graph.observables`."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn
from torch_geometric.data import Data

from koopman_graph import (
    GNNDecoder,
    GNNEncoder,
    GraphKoopmanModel,
    KoopmanOperator,
)
from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.operators.discrete_propagation import (
    inverse_step as disc_inverse_step,
)
from koopman_graph.uq import (
    LatentGaussianKoopmanUQ,
)


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
    parametric = GraphKoopmanModel(
        enc,
        dec,
        latent_dim=2,
        time_step=0.1,
        koopman="parametric",
    )
    assert parametric.koopman_kind == "parametric"
    with pytest.raises(ValueError, match="koopman_auxiliary_hidden_dims"):
        GraphKoopmanModel(
            enc,
            dec,
            latent_dim=2,
            time_step=0.1,
            koopman_auxiliary_hidden_dims=(8,),
        )
    # graph + continuous is the supported continuous_graph alias (0.6.0).
    cg = GraphKoopmanModel(
        enc,
        dec,
        latent_dim=2,
        time_step=0.1,
        koopman="graph",
        dynamics_mode="continuous",
    )
    assert cg.koopman_kind == "continuous_graph"
    with pytest.raises(ValueError, match="continuous_graph"):
        GraphKoopmanModel(
            enc,
            dec,
            latent_dim=2,
            time_step=0.1,
            koopman="continuous_graph",
            dynamics_mode="discrete",
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
