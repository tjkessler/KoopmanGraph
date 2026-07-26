"""Tests for Koopman-MPC and conformal tightening (TASK-1314 / TASK-1315)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import Tensor, nn
from torch_geometric.data import Data

from koopman_graph import GNNEncoder, GraphKoopmanModel
from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.mpc import KoopmanMPC
from koopman_graph.mpc.controller import _conformal_stage_margins
from koopman_graph.mpc.qp import assemble_condensed_mpc, require_osqp
from koopman_graph.uq import ConformalKoopmanUQ
from koopman_graph.uq.common import snapshot_with_features

pytest.importorskip("osqp")


class _IdentityDecoder(nn.Module):
    """Pass-through decoder so Jacobian C ≈ I for plant tests."""

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


def _two_node_edge_index() -> torch.Tensor:
    return torch.tensor([[0, 1], [1, 0]], dtype=torch.long)


def _identity_plant_model(
    *,
    latent_dim: int = 2,
    control_dim: int = 1,
    control_mode: str = "additive",
) -> GraphKoopmanModel:
    """Build a controlled model with identity encode/decode and fixed K, B."""
    model = GraphKoopmanModel(
        encoder=GNNEncoder(latent_dim, 4, latent_dim, num_layers=1),
        decoder=_IdentityDecoder(latent_dim),
        latent_dim=latent_dim,
        time_step=0.1,
        control_dim=control_dim,
        control_mode=control_mode,  # type: ignore[arg-type]
    )
    with torch.no_grad():
        model.koopman.K.copy_(0.85 * torch.eye(latent_dim))
        if control_dim > 0 and model.koopman.B is not None:
            b = torch.zeros(control_dim, latent_dim)
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


def _origin(num_nodes: int = 2, latent_dim: int = 2) -> Data:
    return Data(
        x=torch.zeros(num_nodes, latent_dim),
        edge_index=_two_node_edge_index(),
    )


def test_mpc_tracks_constant_reference() -> None:
    """MPC drives identity-plant outputs toward a constant reference."""
    model = _identity_plant_model()
    controller = KoopmanMPC(
        model,
        horizon=8,
        Q=torch.eye(2),
        R=0.05 * torch.eye(1),
        u_min=torch.tensor([-2.0]),
        u_max=torch.tensor([2.0]),
    )
    reference = torch.tensor([1.0, 0.0])
    snapshots = controller.rollout(_origin(), reference, steps=20)
    final = snapshots[-1].x.mean(dim=0)
    assert torch.allclose(final, reference, atol=0.15)


def test_mpc_honors_box_input_constraints() -> None:
    """Closed-loop actions stay within declared u bounds."""
    model = _identity_plant_model()
    u_min = torch.tensor([-0.2])
    u_max = torch.tensor([0.2])
    controller = KoopmanMPC(
        model,
        horizon=6,
        Q=torch.eye(2),
        R=1e-3 * torch.eye(1),
        u_min=u_min,
        u_max=u_max,
    )
    reference = torch.tensor([2.0, 0.0])
    current = _origin()
    for _ in range(12):
        action = controller.solve(current, reference)
        assert float(action.min()) >= float(u_min.min()) - 1e-6
        assert float(action.max()) <= float(u_max.max()) + 1e-6
        with torch.no_grad():
            z = model.encode(current)
            z_next = model.koopman.advance(z, control=action)
            current = Data(
                x=model.decoder(z_next, current.edge_index),
                edge_index=current.edge_index,
            )


def test_mpc_infeasible_qp_raises() -> None:
    """Infeasible output bounds raise RuntimeError (documented fallback)."""
    model = _identity_plant_model()
    controller = KoopmanMPC(
        model,
        horizon=3,
        Q=torch.eye(2),
        R=torch.eye(1),
        u_min=torch.tensor([-0.01]),
        u_max=torch.tensor([0.01]),
        # Current state is 0; force an impossible stage-0 output band.
        y_min=torch.tensor([5.0, 5.0]),
        y_max=torch.tensor([6.0, 6.0]),
    )
    with pytest.raises(RuntimeError, match="QP solve failed"):
        controller.solve(_origin(), torch.tensor([0.0, 0.0]))


def test_mpc_rejects_bilinear_control() -> None:
    """Bilinear operators are rejected at construction."""
    model = _identity_plant_model(control_mode="bilinear")
    with pytest.raises(ValueError, match="additive"):
        KoopmanMPC(model, horizon=2, Q=torch.eye(2), R=torch.eye(1))


def _patch_zero_predict(model: GraphKoopmanModel) -> None:
    """Force predict to return zero features (for conformal calibration)."""

    def _predict(
        initial_graph,
        steps,
        edge_index=None,
        edge_weight=None,
        controls=None,
        future_topologies=None,
        history=None,
    ):
        del edge_index, edge_weight, controls, future_topologies, history
        if isinstance(initial_graph, Data):
            template = initial_graph
            assert template.x is not None
            zeros = torch.zeros_like(template.x)
        else:
            template = Data(x=initial_graph, edge_index=_two_node_edge_index())
            zeros = torch.zeros_like(initial_graph)
        return [snapshot_with_features(template, zeros) for _ in range(steps)]

    model.predict = _predict  # type: ignore[method-assign]


def _calibrated_uq(
    model: GraphKoopmanModel,
    *,
    steps: int = 4,
    residual: float = 0.25,
) -> ConformalKoopmanUQ:
    """Calibrate conformal UQ with a constant positive residual."""
    _patch_zero_predict(model)
    edge_index = _two_node_edge_index()
    sequences = []
    for _ in range(8):
        snaps = [Data(x=torch.zeros(2, 2), edge_index=edge_index)]
        for _horizon in range(steps):
            snaps.append(
                Data(
                    x=torch.full((2, 2), residual),
                    edge_index=edge_index,
                )
            )
        sequences.append(GraphSnapshotSequence(snaps))
    uq = ConformalKoopmanUQ(model, method="split", score="aggregate")
    return uq.calibrate(sequences, steps=steps, alpha=0.1)


def test_mpc_rejects_non_conformal_tightening() -> None:
    """Non-ConformalKoopmanUQ tightening objects raise TypeError."""
    model = _identity_plant_model()
    with pytest.raises(TypeError, match="ConformalKoopmanUQ"):
        KoopmanMPC(
            model,
            horizon=2,
            Q=torch.eye(2),
            R=torch.eye(1),
            y_max=torch.tensor([1.0, 1.0]),
            constraint_tightening=object(),
        )


def test_mpc_rejects_uncalibrated_tightening() -> None:
    """Uncalibrated conformal UQ is rejected at construction."""
    model = _identity_plant_model()
    uq = ConformalKoopmanUQ(model)
    with pytest.raises(RuntimeError, match="not calibrated"):
        KoopmanMPC(
            model,
            horizon=2,
            Q=torch.eye(2),
            R=torch.eye(1),
            y_max=torch.tensor([1.0, 1.0]),
            constraint_tightening=uq,
        )


def test_mpc_rejects_short_calibrated_horizon() -> None:
    """Tightening calibrated for fewer steps than the MPC horizon fails."""
    model = _identity_plant_model()
    uq = _calibrated_uq(model, steps=2)
    with pytest.raises(ValueError, match="calibrated_steps"):
        KoopmanMPC(
            model,
            horizon=4,
            Q=torch.eye(2),
            R=torch.eye(1),
            y_max=torch.tensor([1.0, 1.0]),
            constraint_tightening=uq,
        )


def test_mpc_rejects_tightening_without_output_bounds() -> None:
    """Tightening requires y_min and/or y_max."""
    model = _identity_plant_model()
    uq = _calibrated_uq(model, steps=4)
    with pytest.raises(ValueError, match="y_min and/or y_max"):
        KoopmanMPC(
            model,
            horizon=3,
            Q=torch.eye(2),
            R=torch.eye(1),
            constraint_tightening=uq,
        )


def test_mpc_stage_margins_map_quantiles() -> None:
    """Stage 0 is unshrunk; stages 1..H use quantiles[h-1]."""
    model = _identity_plant_model()
    uq = _calibrated_uq(model, steps=4, residual=0.3)
    margins = _conformal_stage_margins(uq, horizon=3)
    assert margins.shape == (4,)
    assert margins[0] == pytest.approx(0.0)
    quantiles = uq.quantiles.detach().cpu().numpy()
    np.testing.assert_allclose(margins[1:], quantiles[:3])
    assert float(margins[1:].min()) > 0.0


def test_mpc_tightened_bounds_strictly_inside() -> None:
    """Positive stage margins shrink the assembled output boxes."""
    y_max = np.array([1.0, 2.0])
    margins = np.array([0.0, 0.1, 0.2])
    a = np.eye(2)
    b = np.array([[1.0], [0.0]])
    c = np.eye(2)
    x0 = np.zeros(2)
    refs = np.zeros((3, 2))
    _p, _q, a_tight, l_tight, u_tight = assemble_condensed_mpc(
        a_mat=a,
        b_mat=b,
        c_mat=c,
        x0=x0,
        references=refs,
        q_cost=np.eye(2),
        r_cost=np.eye(1),
        qf_cost=np.eye(2),
        u_min=None,
        u_max=None,
        y_min=None,
        y_max=y_max,
        stage_margins=margins,
    )
    _p2, _q2, a_plain, l_plain, u_plain = assemble_condensed_mpc(
        a_mat=a,
        b_mat=b,
        c_mat=c,
        x0=x0,
        references=refs,
        q_cost=np.eye(2),
        r_cost=np.eye(1),
        qf_cost=np.eye(2),
        u_min=None,
        u_max=None,
        y_min=None,
        y_max=y_max,
        stage_margins=None,
    )
    assert a_tight.shape == a_plain.shape
    # Upper bounds for stages with positive margin are strictly smaller.
    # Layout: H+1 stages × F features of upper bounds only (no u rows).
    assert u_tight[2] < u_plain[2]  # stage 1, feature 0
    assert u_tight[4] < u_plain[4]  # stage 2, feature 0
    assert u_tight[0] == pytest.approx(u_plain[0])  # stage 0 unshrunk


def test_mpc_tightened_more_conservative_closed_loop() -> None:
    """Tightened MPC applies smaller first action and lower peak under y_max."""
    model = _identity_plant_model()
    uq = _calibrated_uq(model, steps=6, residual=0.25)
    y_max = torch.tensor([0.9, 10.0])
    common = {
        "horizon": 4,
        "Q": torch.eye(2),
        "R": 1e-3 * torch.eye(1),
        "u_min": torch.tensor([-2.0]),
        "u_max": torch.tensor([2.0]),
        "y_max": y_max,
    }
    plain = KoopmanMPC(model, **common)
    tight = KoopmanMPC(model, constraint_tightening=uq, **common)
    reference = torch.tensor([2.0, 0.0])
    origin = _origin()
    u_plain = plain.solve(origin, reference)
    u_tight = tight.solve(origin, reference)
    assert float(u_tight.abs().sum()) < float(u_plain.abs().sum())

    def _safe_rollout(controller: KoopmanMPC) -> list[Data]:
        current = origin
        traj: list[Data] = []
        for _ in range(8):
            try:
                action = controller.solve(current, reference)
            except RuntimeError:
                break
            with torch.no_grad():
                z = model.encode(current)
                z_next = model.koopman.advance(z, control=action)
                decoded = model.decoder(z_next, current.edge_index)
            current = Data(x=decoded, edge_index=current.edge_index)
            traj.append(current)
        return traj

    plain_traj = _safe_rollout(plain)
    tight_traj = _safe_rollout(tight)
    assert plain_traj and tight_traj
    plain_peak = max(float(snap.x.mean(dim=0)[0]) for snap in plain_traj)
    tight_peak = max(float(snap.x.mean(dim=0)[0]) for snap in tight_traj)
    assert tight_peak < plain_peak


def test_mpc_tightened_reduces_noisy_violations() -> None:
    """On a seeded noisy plant, tightened MPC violates y_max no more often."""
    model = _identity_plant_model()
    uq = _calibrated_uq(model, steps=6, residual=0.2)
    y_max_val = 0.85
    y_max = torch.tensor([y_max_val, 10.0])
    common = {
        "horizon": 4,
        "Q": torch.eye(2),
        "R": 1e-3 * torch.eye(1),
        "u_min": torch.tensor([-1.5]),
        "u_max": torch.tensor([1.5]),
        "y_max": y_max,
    }
    plain = KoopmanMPC(model, **common)
    tight = KoopmanMPC(model, constraint_tightening=uq, **common)
    reference = torch.tensor([1.5, 0.0])

    def _noisy_rollout(controller: KoopmanMPC, seed: int) -> int:
        current = _origin()
        generator = torch.Generator().manual_seed(seed)
        violations = 0
        for _ in range(12):
            try:
                action = controller.solve(current, reference)
            except RuntimeError:
                # Stage-0 infeasibility after a bound breach counts as a violation.
                violations += 1
                break
            with torch.no_grad():
                z = model.encode(current)
                z_next = model.koopman.advance(z, control=action)
                noise = 0.12 * torch.randn(z_next.shape, generator=generator)
                decoded = model.decoder(z_next + noise, current.edge_index)
            # Keep the QP plant feasible for the next solve while still
            # scoring noisy outputs against the declared bound.
            mean_y = decoded.mean(dim=0)
            if float(mean_y[0]) > y_max_val + 1e-6:
                violations += 1
            clipped = decoded.clone()
            clipped[:, 0] = torch.clamp(clipped[:, 0], max=y_max_val)
            current = Data(x=clipped, edge_index=current.edge_index)
        return violations

    plain_violations = _noisy_rollout(plain, seed=11)
    tight_violations = _noisy_rollout(tight, seed=11)
    assert plain_violations > 0
    assert tight_violations <= plain_violations


def test_mpc_missing_osqp_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guided ImportError when the [mpc] extra is absent."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "osqp":
            raise ImportError("no osqp")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"koopman-graph\[mpc\]"):
        require_osqp()


def test_mpc_env_closed_loop_smoke() -> None:
    """MPC can act as a controller inside GraphKoopmanEnv."""
    pytest.importorskip("gymnasium")
    from koopman_graph.env import GraphKoopmanEnv

    model = _identity_plant_model()
    edge_index = _two_node_edge_index()
    snapshots = [Data(x=torch.zeros(2, 2), edge_index=edge_index) for _ in range(8)]
    controls = torch.zeros(8, 1)
    sequence = GraphSnapshotSequence(snapshots, control_inputs=controls)
    env = GraphKoopmanEnv(
        model,
        sequence,
        reward_fn=lambda _s, _i: 0.0,
        random_start=False,
        start_index=0,
        max_episode_steps=5,
    )
    controller = KoopmanMPC(
        model,
        horizon=4,
        Q=torch.eye(2),
        R=0.1 * torch.eye(1),
        u_min=torch.tensor([-1.0]),
        u_max=torch.tensor([1.0]),
    )
    obs, _info = env.reset(seed=0)
    assert obs.shape == (4,)
    current = sequence[0]
    for _ in range(3):
        action = controller.solve(current, torch.tensor([0.5, 0.0]))
        obs, _reward, terminated, truncated, _info = env.step(
            action.detach().cpu().numpy()
        )
        assert obs.shape == (4,)
        assert not terminated
        with torch.no_grad():
            z = model.encode(current)
            z_next = model.koopman.advance(z, control=action)
            current = Data(
                x=model.decoder(z_next, current.edge_index),
                edge_index=current.edge_index,
            )
        if truncated:
            break


def test_mpc_package_import_without_solve() -> None:
    """Core mpc package import exposes KoopmanMPC without solving."""
    from koopman_graph import mpc as mpc_pkg

    assert mpc_pkg.KoopmanMPC is KoopmanMPC
