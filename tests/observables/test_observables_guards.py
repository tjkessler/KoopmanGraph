"""Coverage and error-path tests for :mod:`koopman_graph.observables`."""

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
from koopman_graph.losses import (
    BackwardConsistencyLoss,
    EigenvalueRegularizationLoss,
    ForwardConsistencyLoss,
    masked_mse_loss,
    rollout_sequence_loss,
)
from koopman_graph.observables import (
    concatenate_observables,
    graph_laplacian_features,
    resolve_physics_lifting_fn,
    validate_physics_output,
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
