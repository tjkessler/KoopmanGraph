"""Guard-removal regression suite for multiplex/typed hetero parity (TASK-1810).

Locks supported accept paths (conformal / env / hierarchical / evaluate) and
keeps clear raises for unsupported combinations, including typed hierarchical
pooling (TASK-1808).
"""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data, HeteroData

from koopman_graph.data import (
    GraphSnapshotSequence,
    HeteroGraphSnapshotSequence,
    NeighborWindowSampler,
)
from koopman_graph.hierarchical import HierarchicalGraphKoopmanModel
from koopman_graph.metrics import evaluate_forecast
from koopman_graph.model import GraphKoopmanModel
from koopman_graph.nn import GNNDecoder, GNNEncoder, RelGraphDecoder, RelGraphEncoder
from koopman_graph.uq import ConformalKoopmanUQ

_EDGE_TYPES = (
    ("node", "r1", "node"),
    ("node", "r2", "node"),
)


def _multiplex_snapshot(*, seed: int = 0, num_nodes: int = 4) -> HeteroData:
    generator = torch.Generator().manual_seed(seed)
    data = HeteroData()
    data["node"].x = torch.randn(num_nodes, 3, generator=generator)
    data["node", "r1", "node"].edge_index = torch.tensor(
        [[0, 1, 2], [1, 2, 0]],
        dtype=torch.long,
    )
    data["node", "r2", "node"].edge_index = torch.tensor(
        [[0, 2], [2, 3]],
        dtype=torch.long,
    )
    return data


def _multiplex_sequence(
    *,
    timesteps: int = 4,
    seed: int = 0,
    num_nodes: int = 4,
) -> HeteroGraphSnapshotSequence:
    return HeteroGraphSnapshotSequence(
        [
            _multiplex_snapshot(seed=seed + t, num_nodes=num_nodes)
            for t in range(timesteps)
        ]
    )


def _multiplex_model(*, control_dim: int = 0, seed: int = 0) -> GraphKoopmanModel:
    torch.manual_seed(seed)
    return GraphKoopmanModel(
        encoder=RelGraphEncoder(
            3,
            hidden_channels=8,
            latent_dim=4,
            num_relations=2,
            num_layers=1,
        ),
        decoder=RelGraphDecoder(
            latent_dim=4,
            hidden_channels=8,
            out_channels=3,
            num_relations=2,
            num_layers=1,
        ),
        latent_dim=4,
        time_step=1.0,
        koopman="hetero_graph",
        koopman_edge_types=_EDGE_TYPES,
        control_dim=control_dim,
    )


def _homo_model(*, control_dim: int = 0) -> GraphKoopmanModel:
    return GraphKoopmanModel(
        encoder=GNNEncoder(in_channels=3, hidden_channels=8, latent_dim=4),
        decoder=GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=3),
        latent_dim=4,
        time_step=1.0,
        control_dim=control_dim,
    )


def _homo_sequence(*, timesteps: int = 3) -> GraphSnapshotSequence:
    snaps = [
        Data(
            x=torch.randn(4, 3),
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
        )
        for _ in range(timesteps)
    ]
    return GraphSnapshotSequence(snaps)


def _typed_model() -> GraphKoopmanModel:
    return GraphKoopmanModel(
        encoder=RelGraphEncoder(
            {"a": 2, "b": 3},
            hidden_channels=8,
            latent_dim=4,
            num_relations=3,
            num_layers=1,
            node_types=("a", "b"),
            edge_types=(
                ("a", "r0", "b"),
                ("b", "r1", "a"),
                ("a", "r2", "a"),
            ),
        ),
        decoder=RelGraphDecoder(
            latent_dim=4,
            hidden_channels=8,
            out_channels={"a": 2, "b": 3},
            num_relations=3,
            num_layers=1,
            node_types=("a", "b"),
            edge_types=(
                ("a", "r0", "b"),
                ("b", "r1", "a"),
                ("a", "r2", "a"),
            ),
        ),
        latent_dim=4,
        time_step=1.0,
        koopman="hetero_graph",
        koopman_node_types=("a", "b"),
        koopman_edge_types=(
            ("a", "r0", "b"),
            ("b", "r1", "a"),
            ("a", "r2", "a"),
        ),
    )


# ---------------------------------------------------------------------------
# Accept paths (supported multiplex)
# ---------------------------------------------------------------------------


def test_accept_conformal_calibrate_and_predict_interval() -> None:
    """Multiplex conformal calibrate + predict_interval return HeteroData."""
    torch.manual_seed(0)
    model = _multiplex_model(seed=0)
    cal = [_multiplex_sequence(timesteps=3, seed=10 + i) for i in range(3)]
    uq = ConformalKoopmanUQ(model, method="split", score="aggregate")
    uq.calibrate(cal, steps=1, alpha=0.1)
    assert uq.is_calibrated
    interval = uq.predict_interval(cal[0][0], steps=1, level=0.9)
    assert isinstance(interval.mean[0], HeteroData)
    assert isinstance(interval.lower[0], HeteroData)
    assert torch.isfinite(interval.lower[0]["node"].x).all()


def test_accept_env_reset_and_step() -> None:
    """Multiplex GraphKoopmanEnv reset/step smoke."""
    pytest.importorskip("gymnasium")
    from koopman_graph.env import GraphKoopmanEnv

    model = _multiplex_model(control_dim=1, seed=1)
    ref = _multiplex_sequence(timesteps=3, seed=20)
    env = GraphKoopmanEnv(
        model,
        ref,
        reward_fn=lambda data, _t: float(data["node"].x.sum().item()),
        control_low=-1.0,
        control_high=1.0,
        max_episode_steps=2,
        start_index=0,
        random_start=False,
    )
    obs, _info = env.reset(seed=0)
    assert obs.shape == (model.latent_dim * 4,)
    obs2, _reward, terminated, truncated, info = env.step([0.0])
    assert obs2.shape == obs.shape
    assert "decoded_x" in info
    assert info["decoded_x"].shape == (4, 3)
    assert not (terminated and truncated)


def test_accept_hierarchical_pool_predict_and_hold_perm_fit() -> None:
    """Multiplex hierarchical pool/predict and hold_perm fit smoke."""
    torch.manual_seed(2)
    model = _multiplex_model(seed=2)
    hier = HierarchicalGraphKoopmanModel(
        model,
        pool_ratios=(0.5,),
        pool_schedule="hold_perm",
    )
    fine = _multiplex_snapshot(seed=30, num_nodes=8)
    coarse, steps = hier.pool_down(fine)
    assert isinstance(coarse, HeteroData)
    assert coarse["node"].x.size(0) == 4
    assert steps[0].relation_edge_indices is not None

    preds = hier.predict(fine, steps=1, resolution="fine")
    assert isinstance(preds[0], HeteroData)
    assert preds[0]["node"].x.shape == (8, 3)

    history = hier.fit(
        _multiplex_sequence(timesteps=5, seed=31, num_nodes=8),
        epochs=1,
        lr=1e-2,
        unpool_epochs=0,
    )
    assert len(history.loss) == 1
    assert torch.isfinite(torch.tensor(history.loss[0]))


def test_accept_predict_at_and_evaluate_forecast() -> None:
    """Multiplex predict_at / evaluate_forecast return finite stacked metrics."""
    torch.manual_seed(3)
    model = _multiplex_model(seed=3)
    origin = _multiplex_snapshot(seed=40)
    preds = model.predict_at(origin, step_deltas=[1.0, 1.0])
    assert len(preds) == 2
    assert isinstance(preds[0], HeteroData)

    sequence = _multiplex_sequence(timesteps=4, seed=41)
    result = evaluate_forecast(model, sequence, horizons=(1, 2), start_indices=[0])
    assert result.num_origins == 1
    assert torch.isfinite(torch.tensor(result.aggregate_mae))
    via_model = model.evaluate(sequence, horizons=(1,), start_indices=[0])
    assert via_model.num_origins == 1


# ---------------------------------------------------------------------------
# Reject paths (unsupported combinations stay clear)
# ---------------------------------------------------------------------------


def test_accept_typed_hierarchical_pool_and_hold_perm_fit() -> None:
    """Typed multi-node hierarchical pool/predict and hold_perm fit smoke."""
    torch.manual_seed(5)
    model = _typed_model()
    hier = HierarchicalGraphKoopmanModel(
        model,
        pool_ratios=(0.5,),
        pool_schedule="hold_perm",
    )
    fine = HeteroData()
    fine["a"].x = torch.randn(8, 2)
    fine["b"].x = torch.randn(6, 3)
    src_a = torch.arange(8, dtype=torch.long)
    fine["a", "r0", "b"].edge_index = torch.stack([src_a, src_a % 6], dim=0)
    src_b = torch.arange(6, dtype=torch.long)
    fine["b", "r1", "a"].edge_index = torch.stack([src_b, src_b % 8], dim=0)
    fine["a", "r2", "a"].edge_index = torch.stack([src_a, (src_a + 1) % 8], dim=0)

    coarse, steps = hier.pool_down(fine)
    assert isinstance(coarse, HeteroData)
    assert coarse["a"].x.size(0) == 4
    assert coarse["b"].x.size(0) == 3
    assert steps[0].typed_perms is not None

    preds = hier.predict(fine, steps=1, resolution="fine")
    assert isinstance(preds[0], HeteroData)
    assert preds[0]["a"].x.shape == (8, 2)

    sequence = HeteroGraphSnapshotSequence([fine, fine.clone(), fine.clone()])
    history = hier.fit(sequence, epochs=1, lr=1e-2, unpool_epochs=0)
    assert len(history.loss) == 1
    assert torch.isfinite(torch.tensor(history.loss[0]))


def test_reject_neighbor_window_sampler_on_hetero() -> None:
    """NeighborWindowSampler construction names the homo-only restriction."""
    sequence = _multiplex_sequence(timesteps=3, seed=50)
    with pytest.raises(
        ValueError,
        match="does not support HeteroGraphSnapshotSequence",
    ):
        NeighborWindowSampler(
            sequence,  # type: ignore[arg-type]
            window_length=2,
            num_nodes=2,
            num_hops=1,
            batch_size=1,
            shuffle=False,
        )


def test_reject_homo_conformal_with_hetero_sequence() -> None:
    """Homogeneous conformal calibrate rejects hetero sequences."""
    uq = ConformalKoopmanUQ(_homo_model())
    with pytest.raises(TypeError, match="homogeneous ConformalKoopmanUQ"):
        uq.calibrate([_multiplex_sequence(timesteps=3, seed=60)], steps=1)


def test_reject_hetero_conformal_with_homo_sequence() -> None:
    """Hetero conformal calibrate rejects homogeneous sequences."""
    uq = ConformalKoopmanUQ(_multiplex_model(seed=4))
    with pytest.raises(TypeError, match="hetero ConformalKoopmanUQ"):
        uq.calibrate([_homo_sequence()], steps=1)


def test_reject_homo_env_with_hetero_sequence() -> None:
    """Homogeneous env rejects hetero reference sequences."""
    pytest.importorskip("gymnasium")
    from koopman_graph.env import GraphKoopmanEnv

    with pytest.raises(TypeError, match="homogeneous GraphKoopmanEnv"):
        GraphKoopmanEnv(
            _homo_model(control_dim=1),
            _multiplex_sequence(timesteps=2, seed=70),
            reward_fn=lambda _d, _t: 0.0,
            control_low=-1.0,
            control_high=1.0,
        )


def test_reject_hetero_env_with_homo_sequence() -> None:
    """Hetero env rejects homogeneous reference sequences."""
    pytest.importorskip("gymnasium")
    from koopman_graph.env import GraphKoopmanEnv

    with pytest.raises(TypeError, match="hetero GraphKoopmanEnv"):
        GraphKoopmanEnv(
            _multiplex_model(control_dim=1, seed=5),
            _homo_sequence(timesteps=2),
            reward_fn=lambda _d, _t: 0.0,
            control_low=-1.0,
            control_high=1.0,
        )


def test_reject_hetero_evaluate_with_controls() -> None:
    """Controlled hetero sequences are rejected by evaluate_forecast."""
    snaps = [_multiplex_snapshot(seed=80 + t) for t in range(3)]
    controls = torch.zeros(3, 1)
    sequence = HeteroGraphSnapshotSequence(snaps, control_inputs=controls)
    model = _multiplex_model(control_dim=1, seed=6)
    with pytest.raises(ValueError, match="controlled HeteroGraphSnapshotSequence"):
        evaluate_forecast(model, sequence, horizons=(1,))


def test_reject_multiplex_hierarchical_per_snapshot_fit() -> None:
    """Multiplex hierarchical per_snapshot fit points at hold_perm."""
    hier = HierarchicalGraphKoopmanModel(
        _multiplex_model(seed=7),
        pool_ratios=(0.5,),
        pool_schedule="per_snapshot",
    )
    with pytest.raises(ValueError, match="hold_perm"):
        hier.fit(
            _multiplex_sequence(timesteps=4, seed=90, num_nodes=8),
            epochs=1,
            unpool_epochs=0,
        )


def test_reject_hetero_predict_history() -> None:
    """Delay history remains unsupported on hetero predict."""
    model = _multiplex_model(seed=8)
    with pytest.raises(ValueError, match="history / delay embedding"):
        model.predict(
            _multiplex_snapshot(seed=100),
            steps=1,
            history=[_multiplex_snapshot(seed=99)],
        )
