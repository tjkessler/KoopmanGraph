"""Coverage and error-path tests for :mod:`koopman_graph.model`."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch_geometric.data import Data

from koopman_graph import (
    GNNDecoder,
    GNNEncoder,
    GraphKoopmanModel,
    KoopmanOperator,
)
from koopman_graph.data import GraphSnapshotSequence


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
