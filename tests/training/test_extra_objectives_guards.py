"""Coverage and error-path tests for :mod:`koopman_graph.training`."""

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
from koopman_graph.losses.physics import PDEResidualLoss
from koopman_graph.training.extra_objectives import (
    compute_lie_consistency_loss,
    compute_pde_residual_loss,
)
from koopman_graph.training.history import ExtraLosses
from koopman_graph.training.objectives import (
    _graph_eigenvalue_regularization_over_sequence,
    _topologies_equal,
    compute_eigenvalue_regularization_loss,
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
