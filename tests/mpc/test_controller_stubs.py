"""Coverage and error-path tests for :mod:`koopman_graph.mpc`."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from torch import Tensor, nn
from torch_geometric.data import Data

from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
from koopman_graph.mpc import KoopmanMPC
from koopman_graph.mpc.controller import (
    _as_numpy,
    _mean_latent,
    _resolve_reference,
    _validate_mpc_model,
)


def _path_edges(num_nodes: int = 4) -> Tensor:
    """Undirected path ``edge_index``."""
    src = list(range(num_nodes - 1)) + list(range(1, num_nodes))
    dst = list(range(1, num_nodes)) + list(range(num_nodes - 1))
    return torch.tensor([src, dst], dtype=torch.long)


def _tiny_model(**kwargs: Any) -> GraphKoopmanModel:
    """Build a tiny homogeneous GCN model."""
    return GraphKoopmanModel(
        encoder=GNNEncoder(2, 4, 4, num_layers=1),
        decoder=GNNDecoder(4, 4, 2, num_layers=1),
        latent_dim=4,
        time_step=0.1,
        **kwargs,
    )


class _IdentityDecoder(nn.Module):
    """Pass-through decoder so the MPC Jacobian is approximately identity."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.latent_dim = channels
        self.out_channels = channels

    def forward(
        self,
        z: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        del edge_index, edge_weight
        return z


def _controlled_identity_model(*, latent_dim: int = 2) -> GraphKoopmanModel:
    """Controlled per-node model with identity encode/decode."""
    model = GraphKoopmanModel(
        encoder=GNNEncoder(latent_dim, 4, latent_dim, num_layers=1),
        decoder=_IdentityDecoder(latent_dim),
        latent_dim=latent_dim,
        time_step=0.1,
        control_dim=1,
        control_mode="additive",
    )
    with torch.no_grad():
        model.koopman.K.copy_(0.85 * torch.eye(latent_dim))
        if model.koopman.B is not None:
            b = torch.zeros(1, latent_dim)
            b[0, 0] = 1.0
            model.koopman.B.copy_(b)

    def _encode(x_or_data, edge_index=None, edge_weight=None):
        del edge_index, edge_weight
        if isinstance(x_or_data, Data):
            assert x_or_data.x is not None
            return x_or_data.x.clone()
        return x_or_data.clone()

    model.encode = _encode  # type: ignore[method-assign]
    return model


def test_mpc_controller_validation_solve_and_rollout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """KoopmanMPC validates plants and solves via a stubbed dense QP."""
    discrete = _tiny_model()
    with pytest.raises(ValueError, match="control_dim"):
        _validate_mpc_model(discrete)
    graph_model = _tiny_model(koopman="graph")
    with pytest.raises(ValueError, match="networked"):
        _validate_mpc_model(graph_model)
    switched = _tiny_model(koopman="switched")
    with pytest.raises(TypeError, match="discrete KoopmanOperator"):
        _validate_mpc_model(switched)
    continuous = _tiny_model()
    continuous.dynamics_mode = "continuous"
    with pytest.raises(ValueError, match="discrete"):
        _validate_mpc_model(continuous)
    with pytest.raises(ValueError, match="shape"):
        _mean_latent(torch.randn(3))
    refs = _resolve_reference(torch.zeros(2), horizon=2, out_dim=2)
    assert refs.shape == (3, 2)
    stacked = _resolve_reference(torch.zeros(1, 2), horizon=2, out_dim=2)
    assert stacked.shape == (3, 2)
    listed = _resolve_reference(
        [torch.zeros(2) for _ in range(3)],
        horizon=2,
        out_dim=2,
    )
    assert listed.shape == (3, 2)
    with pytest.raises(ValueError, match="broadcast"):
        _resolve_reference(torch.zeros(4), horizon=2, out_dim=2)
    assert _as_numpy(torch.ones(2)).dtype == np.float64
    assert _as_numpy(np.ones(2)).dtype == np.float64

    model = _controlled_identity_model()
    with pytest.raises(ValueError, match="horizon"):
        KoopmanMPC(model, horizon=0, Q=torch.eye(2), R=torch.eye(1))
    with pytest.raises(ValueError, match="bilinear_qp_iters"):
        KoopmanMPC(
            model,
            horizon=2,
            Q=torch.eye(2),
            R=torch.eye(1),
            bilinear_qp_iters=0,
        )
    with pytest.raises(TypeError, match="ConformalKoopmanUQ"):
        KoopmanMPC(
            model,
            horizon=2,
            Q=torch.eye(2),
            R=torch.eye(1),
            constraint_tightening=object(),  # type: ignore[arg-type]
        )
    controller = KoopmanMPC(
        model,
        horizon=2,
        Q=torch.eye(2),
        R=np.eye(1),
        Qf=torch.eye(2),
        u_min=torch.tensor([-2.0]),
        u_max=np.array([2.0]),
        y_min=np.array([-5.0, -5.0]),
        y_max=np.array([5.0, 5.0]),
    )
    monkeypatch.setattr(
        "koopman_graph.mpc.controller.solve_dense_qp",
        lambda *_args, **_kwargs: np.array([0.1, 0.0]),
    )
    graph = Data(x=torch.zeros(2, 2), edge_index=_path_edges(2))
    action = controller.solve(graph, torch.zeros(2))
    assert action.shape == (1,)
    with pytest.raises(ValueError, match="steps"):
        controller.rollout(graph, torch.zeros(2), steps=0)
    rolled = controller.rollout(graph, torch.zeros(2), steps=1)
    assert len(rolled) == 1

    from koopman_graph.operators import ContinuousKoopmanOperator
    from koopman_graph.uq import ConformalKoopmanUQ

    continuous = _tiny_model()
    continuous.koopman = ContinuousKoopmanOperator(4, init_mode="identity")
    with pytest.raises(ValueError, match="continuous operators"):
        _validate_mpc_model(continuous)
    controlled = _controlled_identity_model()
    controlled.koopman.control_mode = "bogus"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="control_mode"):
        _validate_mpc_model(controlled)
    controlled = _controlled_identity_model()
    controlled.koopman.B = None
    with pytest.raises(ValueError, match="control matrix B"):
        _validate_mpc_model(controlled)

    tightening = MagicMock(spec=ConformalKoopmanUQ)
    tightening.is_calibrated = False
    with pytest.raises(RuntimeError, match="not calibrated"):
        KoopmanMPC(
            model,
            horizon=2,
            Q=torch.eye(2),
            R=torch.eye(1),
            constraint_tightening=tightening,
            y_min=np.array([-1.0, -1.0]),
        )
    tightening.is_calibrated = True
    tightening.model = _controlled_identity_model()
    with pytest.raises(ValueError, match="same GraphKoopmanModel"):
        KoopmanMPC(
            model,
            horizon=2,
            Q=torch.eye(2),
            R=torch.eye(1),
            constraint_tightening=tightening,
            y_min=np.array([-1.0, -1.0]),
        )
    tightening.model = model
    tightening.calibrated_steps = 1
    with pytest.raises(ValueError, match="shorter than MPC horizon"):
        KoopmanMPC(
            model,
            horizon=2,
            Q=torch.eye(2),
            R=torch.eye(1),
            constraint_tightening=tightening,
            y_min=np.array([-1.0, -1.0]),
        )
    tightening.calibrated_steps = 4
    with pytest.raises(ValueError, match="y_min and/or y_max"):
        KoopmanMPC(
            model,
            horizon=2,
            Q=torch.eye(2),
            R=torch.eye(1),
            constraint_tightening=tightening,
        )
    tightening.quantiles = torch.tensor([0.1, 0.2, 0.3])
    tightened = KoopmanMPC(
        model,
        horizon=2,
        Q=torch.eye(2),
        R=torch.eye(1),
        constraint_tightening=tightening,
        y_min=np.array([-5.0, -5.0]),
        y_max=np.array([5.0, 5.0]),
    )
    action = tightened.solve(graph, torch.zeros(2))
    assert action.shape == (1,)

    bilinear = GraphKoopmanModel(
        encoder=GNNEncoder(2, 4, 2, num_layers=1),
        decoder=_IdentityDecoder(2),
        latent_dim=2,
        time_step=0.1,
        control_dim=1,
        control_mode="bilinear",
    )
    bilinear.encode = model.encode  # type: ignore[method-assign]
    bi_ctrl = KoopmanMPC(bilinear, horizon=2, Q=torch.eye(2), R=torch.eye(1))
    a_mat, b_mat = bi_ctrl._plant_matrices(linearization=np.array([0.05]))
    assert a_mat.shape == (2, 2)
    assert b_mat.shape[0] == 2
    unbounded = KoopmanMPC(model, horizon=2, Q=torch.eye(2), R=torch.eye(1))
    action = unbounded.solve(graph, torch.zeros(2))
    assert action.shape == (1,)
