"""Hetero parity tests for ``GraphKoopmanEnv``."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch_geometric.data import Data, HeteroData

from koopman_graph.data import GraphSnapshotSequence, HeteroGraphSnapshotSequence
from koopman_graph.env import GraphKoopmanEnv, flatten_latent, unflatten_latent
from koopman_graph.model import GraphKoopmanModel
from koopman_graph.nn import GNNDecoder, GNNEncoder, RelGraphDecoder, RelGraphEncoder


def _multiplex_snapshot(*, seed: int = 0) -> HeteroData:
    generator = torch.Generator().manual_seed(seed)
    data = HeteroData()
    data["node"].x = torch.randn(4, 3, generator=generator)
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
    timesteps: int = 3,
    seed: int = 0,
) -> HeteroGraphSnapshotSequence:
    return HeteroGraphSnapshotSequence(
        [_multiplex_snapshot(seed=seed + t) for t in range(timesteps)]
    )


def _multiplex_model(*, control_dim: int = 1, seed: int = 0) -> GraphKoopmanModel:
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
        koopman_edge_types=(
            ("node", "r1", "node"),
            ("node", "r2", "node"),
        ),
        control_dim=control_dim,
    )


NODE_TYPES = ("a", "b")
EDGE_TYPES = (("a", "r0", "b"), ("b", "r1", "a"), ("a", "r2", "a"))
FEATURE_DIMS = {"a": 2, "b": 3}
NUM_NODES = {"a": 4, "b": 3}


def _typed_snapshot(*, seed: int = 0) -> HeteroData:
    generator = torch.Generator().manual_seed(seed)
    snapshot = HeteroData()
    snapshot["a"].x = torch.randn(
        NUM_NODES["a"],
        FEATURE_DIMS["a"],
        generator=generator,
    )
    snapshot["b"].x = torch.randn(
        NUM_NODES["b"],
        FEATURE_DIMS["b"],
        generator=generator,
    )
    snapshot["a", "r0", "b"].edge_index = torch.tensor(
        [[0, 1, 2], [0, 1, 2]],
        dtype=torch.long,
    )
    snapshot["b", "r1", "a"].edge_index = torch.tensor(
        [[0, 1], [1, 3]],
        dtype=torch.long,
    )
    snapshot["a", "r2", "a"].edge_index = torch.tensor(
        [[0, 1, 2], [1, 2, 3]],
        dtype=torch.long,
    )
    return snapshot


def _typed_sequence(
    *,
    timesteps: int = 3,
    seed: int = 0,
) -> HeteroGraphSnapshotSequence:
    return HeteroGraphSnapshotSequence(
        [_typed_snapshot(seed=seed + t) for t in range(timesteps)]
    )


def _typed_model(*, control_dim: int = 1, seed: int = 0) -> GraphKoopmanModel:
    torch.manual_seed(seed)
    return GraphKoopmanModel(
        encoder=RelGraphEncoder(
            FEATURE_DIMS,
            hidden_channels=8,
            latent_dim=4,
            num_relations=len(EDGE_TYPES),
            num_layers=1,
            node_types=NODE_TYPES,
            edge_types=EDGE_TYPES,
        ),
        decoder=RelGraphDecoder(
            latent_dim=4,
            hidden_channels=8,
            out_channels=FEATURE_DIMS,
            num_relations=len(EDGE_TYPES),
            num_layers=1,
            node_types=NODE_TYPES,
            edge_types=EDGE_TYPES,
        ),
        latent_dim=4,
        time_step=1.0,
        koopman="hetero_graph",
        koopman_node_types=NODE_TYPES,
        koopman_edge_types=EDGE_TYPES,
        control_dim=control_dim,
    )


def test_multiplex_env_reset_step_smoke() -> None:
    """Multiplex hetero env reset/step returns stacked latent observations."""
    pytest.importorskip("gymnasium")
    model = _multiplex_model()
    sequence = _multiplex_sequence()
    env = GraphKoopmanEnv(
        model,
        sequence,
        reward_fn=lambda snap, _t: float(snap["node"].x.sum().item()),
        start_index=0,
        random_start=False,
        max_episode_steps=3,
    )
    assert env.num_nodes == 4
    assert env.observation_space.shape == (16,)
    obs, info = env.reset(seed=0)
    assert obs.shape == (16,)
    assert isinstance(info["decoded_x"], np.ndarray)
    assert info["decoded_x"].shape == (4, 3)
    assert "decoded_x_by_type" in info
    assert set(info["decoded_x_by_type"]) == {"node"}

    next_obs, reward, terminated, truncated, step_info = env.step(
        np.zeros(1, dtype=np.float32)
    )
    assert next_obs.shape == (16,)
    assert isinstance(reward, float)
    assert terminated is False
    assert truncated is False
    assert step_info["step_index"] == 1
    reshaped = env.reshape_observation(next_obs)
    assert reshaped.shape == (4, 4)


def test_typed_flatten_order_matches_node_types() -> None:
    """Typed stacked latents flatten in operator node_types order.

    Controlled typed hetero operators are not yet supported
    (``control_dim > 0`` is rejected). The env observation contract still
    matches encode stacking: rows ``[0, N_a)`` then ``[N_a, N_a+N_b)``.
    """
    model = _typed_model(control_dim=0)
    snapshot = _typed_snapshot(seed=0)
    with torch.no_grad():
        z = model.encode(snapshot)
    n_total = NUM_NODES["a"] + NUM_NODES["b"]
    assert z.shape == (n_total, model.latent_dim)
    flat = flatten_latent(z)
    assert flat.shape == (n_total * model.latent_dim,)
    recovered = unflatten_latent(
        flat,
        num_nodes=n_total,
        latent_dim=model.latent_dim,
    )
    torch.testing.assert_close(recovered, z.cpu())
    n_a = NUM_NODES["a"]
    assert recovered[:n_a].shape == (n_a, model.latent_dim)
    assert recovered[n_a:].shape == (NUM_NODES["b"], model.latent_dim)
    # Stacking order matches NODE_TYPES = ("a", "b").
    assert model.koopman.node_types == NODE_TYPES


def test_typed_controlled_env_rejected_until_control_lands() -> None:
    """Typed + control_dim>0 fails at model construction (operator limit)."""
    with pytest.raises(ValueError, match="control is unsupported for typed"):
        _typed_model(control_dim=1)


def test_multiplex_env_reward_receives_heterodata() -> None:
    """Reward callback sees HeteroData, not a silent Data cast."""
    pytest.importorskip("gymnasium")
    seen: list[type] = []

    def reward_fn(snap: HeteroData | Data, _t: int) -> float:
        seen.append(type(snap))
        assert isinstance(snap, HeteroData)
        return 0.0

    env = GraphKoopmanEnv(
        _multiplex_model(),
        _multiplex_sequence(),
        reward_fn=reward_fn,
        start_index=0,
        random_start=False,
        max_episode_steps=2,
    )
    env.reset(seed=0)
    env.step(np.zeros(1, dtype=np.float32))
    assert seen and seen[0] is HeteroData


def test_homo_env_rejects_hetero_sequence() -> None:
    """Homogeneous model cannot take a hetero reference sequence."""
    pytest.importorskip("gymnasium")
    model = GraphKoopmanModel(
        encoder=GNNEncoder(in_channels=3, hidden_channels=8, latent_dim=4),
        decoder=GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=3),
        latent_dim=4,
        time_step=1.0,
        control_dim=1,
    )
    with pytest.raises(TypeError, match="homogeneous GraphKoopmanEnv"):
        GraphKoopmanEnv(
            model,
            _multiplex_sequence(),
            reward_fn=lambda _s, _t: 0.0,
        )


def test_hetero_env_rejects_homo_sequence() -> None:
    """Hetero model requires HeteroGraphSnapshotSequence."""
    pytest.importorskip("gymnasium")
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    seq = GraphSnapshotSequence(
        [Data(x=torch.randn(3, 3), edge_index=edge_index) for _ in range(2)]
    )
    with pytest.raises(TypeError, match="hetero GraphKoopmanEnv"):
        GraphKoopmanEnv(
            _multiplex_model(),
            seq,
            reward_fn=lambda _s, _t: 0.0,
        )


def test_flatten_helpers_still_round_trip() -> None:
    """Shared flatten helpers remain inverse for stacked N."""
    z = torch.randn(7, 4)
    flat = flatten_latent(z)
    recovered = unflatten_latent(flat, num_nodes=7, latent_dim=4)
    torch.testing.assert_close(recovered, z)
