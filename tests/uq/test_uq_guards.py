"""Coverage and error-path tests for :mod:`koopman_graph.uq`."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch_geometric.data import Data

from koopman_graph import (
    GNNDecoder,
    GNNEncoder,
    GraphKoopmanModel,
)
from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.uq import (
    EnsembleGraphKoopmanModel,
    LatentGaussianKoopmanUQ,
    empirical_coverage,
)
from koopman_graph.uq.common import PredictionInterval


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
