"""Tests for shared Laplacian diffusion dynamics utilities."""

from __future__ import annotations

import pytest
import torch

from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.datasets.dynamics import (
    add_gaussian_noise,
    apply_laplacian_diffusion_step,
    diffusion_sequence_from_features,
    initial_node_features,
    laplacian_diffusion_rollout,
    make_generator,
    normalized_step_operator,
    validate_diffusion_generation_params,
)


def test_make_generator_returns_none_without_seed() -> None:
    """Verify unseeded generation returns no torch generator."""
    assert make_generator(None) is None


def test_make_generator_returns_seeded_generator() -> None:
    """Verify seeded generation returns a manual-seed generator."""
    generator = make_generator(42)
    assert generator is not None
    first = torch.randn(3, generator=generator)
    generator = make_generator(42)
    second = torch.randn(3, generator=generator)
    assert torch.equal(first, second)


def test_validate_diffusion_generation_params_rejects_invalid_values() -> None:
    """Verify shared diffusion parameter validation raises on bad input."""
    with pytest.raises(ValueError, match="diffusion_rate"):
        validate_diffusion_generation_params(
            diffusion_rate=1.5,
            decay_rate=0.9,
            noise_std=0.0,
        )
    with pytest.raises(ValueError, match="decay_rate"):
        validate_diffusion_generation_params(decay_rate=0.0, noise_std=0.0)
    with pytest.raises(ValueError, match="noise_std"):
        validate_diffusion_generation_params(decay_rate=0.9, noise_std=-0.1)
    with pytest.raises(ValueError, match="initial_state"):
        validate_diffusion_generation_params(
            decay_rate=0.9,
            noise_std=0.0,
            initial_state="bad",  # type: ignore[arg-type]
        )


def test_add_gaussian_noise_skips_when_std_is_zero() -> None:
    """Verify zero noise std returns the input unchanged."""
    state = torch.ones(4, 2)
    result = add_gaussian_noise(state, 0.0, generator=None, dtype=torch.float32)
    assert torch.equal(result, state)


def test_laplacian_diffusion_rollout_with_initial_features() -> None:
    """Verify rollout accepts explicit initial node features."""
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    initial = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    features = laplacian_diffusion_rollout(
        edge_index=edge_index,
        num_nodes=2,
        num_timesteps=3,
        in_channels=2,
        diffusion_rate=0.5,
        decay_rate=0.9,
        noise_std=0.0,
        initial_state="ones",
        dtype=torch.float32,
        generator=None,
        initial_features=initial,
    )
    assert features.shape == (3, 2, 2)
    assert torch.allclose(features[0], initial)


def test_diffusion_sequence_from_features_wraps_sequence() -> None:
    """Verify feature tensors are wrapped in a validated sequence."""
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    features = torch.ones(2, 2, 1)
    sequence = diffusion_sequence_from_features(features, edge_index)
    assert isinstance(sequence, GraphSnapshotSequence)
    assert sequence.num_timesteps == 2


def test_normalized_step_operator_and_diffusion_step() -> None:
    """Verify operator construction and one diffusion update."""
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    operator = normalized_step_operator(edge_index, 2, 0.5, dtype=torch.float32)
    state = torch.ones(2, 1)
    updated = apply_laplacian_diffusion_step(state, operator, decay_rate=0.8)
    assert updated.shape == state.shape


def test_initial_node_features_random_and_ones() -> None:
    """Verify initial feature helpers for both supported states."""
    ones = initial_node_features(
        3,
        2,
        "ones",
        generator=None,
        dtype=torch.float32,
    )
    assert torch.allclose(ones, torch.ones(3, 2))
    random_a = initial_node_features(
        3,
        2,
        "random",
        generator=make_generator(0),
        dtype=torch.float32,
    )
    random_b = initial_node_features(
        3,
        2,
        "random",
        generator=make_generator(0),
        dtype=torch.float32,
    )
    assert torch.equal(random_a, random_b)
