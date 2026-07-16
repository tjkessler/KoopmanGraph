"""Tests for the Gymnasium RL environment wrapper."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanEnv, GraphKoopmanModel
from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.env import flatten_latent, unflatten_latent

gymnasium = pytest.importorskip("gymnasium")


def _two_node_edge_index() -> torch.Tensor:
    return torch.tensor([[0, 1], [1, 0]], dtype=torch.long)


def _controlled_sequence(
    edge_index: torch.Tensor,
    *,
    num_nodes: int = 2,
    in_channels: int = 2,
    num_timesteps: int = 6,
) -> GraphSnapshotSequence:
    snapshots = [
        Data(
            x=torch.randn(num_nodes, in_channels),
            edge_index=edge_index,
        )
        for _ in range(num_timesteps)
    ]
    controls = torch.randn(num_timesteps, 1)
    return GraphSnapshotSequence(snapshots, control_inputs=controls)


def _controlled_model(
    *,
    in_channels: int = 2,
    latent_dim: int = 4,
    hidden_channels: int = 8,
) -> GraphKoopmanModel:
    encoder = GNNEncoder(in_channels, hidden_channels, latent_dim)
    decoder = GNNDecoder(latent_dim, hidden_channels, in_channels)
    return GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=latent_dim,
        time_step=0.1,
        control_dim=1,
    )


def test_flatten_unflatten_round_trip() -> None:
    """Verify latent flatten helpers are inverse operations."""
    z = torch.randn(3, 5)
    flat = flatten_latent(z)
    recovered = unflatten_latent(flat, num_nodes=3, latent_dim=5)
    assert flat.shape == (15,)
    assert torch.allclose(recovered, z)


def test_env_requires_control_dim() -> None:
    """Verify uncontrolled models cannot build an environment."""
    encoder = GNNEncoder(2, 4, 4)
    decoder = GNNDecoder(4, 4, 2)
    model = GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=4,
        time_step=0.1,
    )
    sequence = _controlled_sequence(_two_node_edge_index())

    with pytest.raises(ValueError, match="control_dim > 0"):
        GraphKoopmanEnv(model, sequence, lambda _s, _i: 0.0)


def test_to_latent_env_factory() -> None:
    """Verify model factory returns a configured environment."""
    model = _controlled_model()
    sequence = _controlled_sequence(_two_node_edge_index())
    env = model.to_latent_env(
        sequence,
        lambda _s, _i: 1.0,
        random_start=False,
        start_index=0,
        max_episode_steps=3,
    )
    assert isinstance(env, GraphKoopmanEnv)
    assert env.observation_space.shape == (2 * 4,)
    assert env.action_space.shape == (1,)


def test_observation_and_action_space_shapes() -> None:
    """Verify Gymnasium space contracts."""
    model = _controlled_model(latent_dim=6)
    sequence = _controlled_sequence(_two_node_edge_index(), in_channels=2)
    env = GraphKoopmanEnv(
        model,
        sequence,
        lambda _s, _i: 0.0,
        random_start=False,
        start_index=0,
    )
    assert env.observation_space.shape == (12,)
    assert env.action_space.low.shape == (1,)
    assert env.action_space.high.shape == (1,)
    np.testing.assert_allclose(env.action_space.low, [-1.0])
    np.testing.assert_allclose(env.action_space.high, [1.0])


def test_step_matches_manual_koopman_decode() -> None:
    """Verify env transitions match encode → Koopman → decode."""
    torch.manual_seed(0)
    model = _controlled_model()
    sequence = _controlled_sequence(_two_node_edge_index())
    rewards: list[float] = []

    def reward_fn(snapshot: Data, step_idx: int) -> float:
        rewards.append(float(snapshot.x.mean()) + step_idx)
        return float(snapshot.x.sum())

    env = GraphKoopmanEnv(
        model,
        sequence,
        reward_fn,
        random_start=False,
        start_index=1,
        max_episode_steps=2,
    )
    obs, _info = env.reset(seed=42)
    assert obs.shape == (8,)

    action = np.array([0.35], dtype=np.float32)
    with torch.no_grad():
        snapshot = sequence[1]
        z = model.encode(snapshot)
        z_next = model.koopman(z, control=torch.tensor([0.35]))
        expected_x = model.decoder(z_next, snapshot.edge_index)
        expected_obs = flatten_latent(z_next)

    next_obs, reward, terminated, truncated, info = env.step(action)
    np.testing.assert_allclose(next_obs, expected_obs, rtol=1e-5, atol=1e-5)
    assert reward == pytest.approx(float(expected_x.sum()))
    assert info["step_index"] == 1
    assert not terminated
    assert not truncated


def test_action_clipping_and_episode_truncation() -> None:
    """Verify out-of-bounds actions are clipped and episodes truncate."""
    model = _controlled_model()
    sequence = _controlled_sequence(_two_node_edge_index())
    env = GraphKoopmanEnv(
        model,
        sequence,
        lambda _s, _i: 0.0,
        random_start=False,
        start_index=0,
        max_episode_steps=2,
    )
    env.reset(seed=0)

    _obs, _reward, _terminated, truncated, _info = env.step(np.array([5.0]))
    assert not truncated

    _obs, _reward, _terminated, truncated, info = env.step(np.array([-5.0]))
    assert truncated
    assert info["step_index"] == 2


def test_encoder_weights_unchanged_across_steps() -> None:
    """Verify environment interaction does not mutate encoder parameters."""
    model = _controlled_model()
    sequence = _controlled_sequence(_two_node_edge_index())
    before = [p.detach().clone() for p in model.encoder.parameters()]
    assert all(not p.requires_grad for p in model.encoder.parameters()) is False

    env = model.to_latent_env(
        sequence,
        lambda _s, _i: 0.0,
        random_start=False,
        start_index=0,
        max_episode_steps=3,
    )
    assert all(not p.requires_grad for p in model.encoder.parameters())

    env.reset(seed=0)
    for value in (-0.5, 0.2, 0.9):
        env.step(np.array([value], dtype=np.float32))

    after = list(model.encoder.parameters())
    for initial, final in zip(before, after, strict=True):
        assert torch.equal(initial, final.detach().cpu())


def test_reset_before_step_raises() -> None:
    """Verify step without reset raises a clear error."""
    model = _controlled_model()
    sequence = _controlled_sequence(_two_node_edge_index())
    env = GraphKoopmanEnv(
        model,
        sequence,
        lambda _s, _i: 0.0,
        random_start=False,
        start_index=0,
    )
    with pytest.raises(RuntimeError, match="reset\\(\\) must be called"):
        env.step(np.array([0.0], dtype=np.float32))


def _controlled_continuous_model(
    *,
    in_channels: int = 2,
    latent_dim: int = 4,
    hidden_channels: int = 8,
    time_step: float = 0.1,
) -> GraphKoopmanModel:
    encoder = GNNEncoder(in_channels, hidden_channels, latent_dim)
    decoder = GNNDecoder(latent_dim, hidden_channels, in_channels)
    return GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=latent_dim,
        time_step=time_step,
        control_dim=1,
        dynamics_mode="continuous",
        koopman_parameterization="dissipative",
    )


def test_continuous_env_custom_delta_t_matches_manual_advance() -> None:
    """Verify continuous env step with custom delta_t matches advance + decode."""
    torch.manual_seed(1)
    model = _controlled_continuous_model()
    sequence = _controlled_sequence(_two_node_edge_index())
    custom_delta = 0.35

    env = GraphKoopmanEnv(
        model,
        sequence,
        lambda snapshot, _i: float(snapshot.x.sum()),
        random_start=False,
        start_index=0,
        delta_t=custom_delta,
        max_episode_steps=2,
    )
    obs, _info = env.reset(seed=0)
    assert obs.shape == (8,)

    action = np.array([0.4], dtype=np.float32)
    with torch.no_grad():
        snapshot = sequence[0]
        z = model.encode(snapshot)
        control = torch.tensor([0.4])
        z_next = model.koopman.advance(z, custom_delta, control=control)
        expected_x = model.decoder(z_next, snapshot.edge_index)
        expected_obs = flatten_latent(z_next)

    next_obs, reward, terminated, truncated, info = env.step(action)
    np.testing.assert_allclose(next_obs, expected_obs, rtol=1e-5, atol=1e-5)
    assert reward == pytest.approx(float(expected_x.sum()))
    assert info["step_index"] == 1
    assert not terminated
    assert not truncated


def test_to_latent_env_forwards_delta_t() -> None:
    """Verify to_latent_env forwards delta_t to the environment."""
    model = _controlled_continuous_model()
    sequence = _controlled_sequence(_two_node_edge_index())
    env = model.to_latent_env(
        sequence,
        lambda _s, _i: 0.0,
        delta_t=0.2,
        random_start=False,
        start_index=0,
    )
    assert env._delta_t == pytest.approx(0.2)


def test_discrete_env_rejects_mismatched_delta_t() -> None:
    """Verify discrete models require delta_t is None or equal to time_step."""
    model = _controlled_model()
    sequence = _controlled_sequence(_two_node_edge_index())

    with pytest.raises(ValueError, match="discrete models require delta_t"):
        GraphKoopmanEnv(
            model,
            sequence,
            lambda _s, _i: 0.0,
            delta_t=0.5,
        )

    # Matching time_step is allowed and equivalent to the default.
    env = GraphKoopmanEnv(
        model,
        sequence,
        lambda _s, _i: 0.0,
        delta_t=model.time_step,
        random_start=False,
        start_index=0,
    )
    assert env._delta_t == pytest.approx(model.time_step)


def test_continuous_env_default_delta_t_uses_time_step() -> None:
    """Verify omitted delta_t falls back to model.time_step via _advance_latent."""
    torch.manual_seed(2)
    model = _controlled_continuous_model(time_step=0.1)
    sequence = _controlled_sequence(_two_node_edge_index())
    env = GraphKoopmanEnv(
        model,
        sequence,
        lambda _s, _i: 0.0,
        random_start=False,
        start_index=0,
    )
    assert env._delta_t is None

    action = np.array([0.1], dtype=np.float32)
    env.reset(seed=0)
    with torch.no_grad():
        snapshot = sequence[0]
        z = model.encode(snapshot)
        z_next = model.koopman.advance(z, model.time_step, control=torch.tensor([0.1]))
        expected_obs = flatten_latent(z_next)

    next_obs, *_ = env.step(action)
    np.testing.assert_allclose(next_obs, expected_obs, rtol=1e-5, atol=1e-5)
