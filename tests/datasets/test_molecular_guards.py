"""Coverage and error-path tests for :mod:`koopman_graph.datasets`."""

from __future__ import annotations

import pytest
import torch

from koopman_graph.datasets.molecular import contact_edge_index


def test_contact_edge_index_rejects_nonfinite() -> None:
    """Non-finite coordinates fail position validation."""
    positions = torch.tensor([[0.0, 0.0, 0.0], [float("nan"), 0.0, 0.0]])
    with pytest.raises(ValueError, match="positions_nm must be finite"):
        contact_edge_index(positions, cutoff_nm=0.5)


def test_synthetic_two_state_generation_guards() -> None:
    """Length / noise / lag / transition-prob generation validators."""
    from koopman_graph.datasets.molecular import (
        generate_synthetic_two_state,
        synthetic_two_state_oracle_timescale,
    )

    with pytest.raises(ValueError, match="num_timesteps must be >= 2"):
        generate_synthetic_two_state(num_timesteps=1)
    with pytest.raises(ValueError, match="noise_std must be finite"):
        generate_synthetic_two_state(num_timesteps=4, noise_std=-0.1)
    with pytest.raises(ValueError, match="noise_std must be finite"):
        generate_synthetic_two_state(num_timesteps=4, noise_std=float("nan"))
    with pytest.raises(ValueError, match="transition_prob must be in"):
        generate_synthetic_two_state(num_timesteps=4, transition_prob=0.5)
    with pytest.raises(ValueError, match="lag_steps must be >= 1"):
        synthetic_two_state_oracle_timescale(0.05, lag_steps=0)


def test_contact_edge_index_cutoff_shape_and_residue_guards() -> None:
    """Cutoff / position / granularity / residue_ids validation branches."""
    positions = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="must be a real number"):
        contact_edge_index("0.5", cutoff_nm="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="positions_nm must be a Tensor"):
        contact_edge_index([[0.0, 0.0, 0.0]], cutoff_nm=0.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least one atom"):
        contact_edge_index(torch.zeros(0, 3), cutoff_nm=0.5)
    with pytest.raises(ValueError, match="granularity must be"):
        contact_edge_index(positions, cutoff_nm=0.5, granularity="chain")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="residue_ids must be a Tensor"):
        contact_edge_index(
            positions,
            cutoff_nm=0.5,
            granularity="residue",
            residue_ids=[0, 1],  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match=r"residue_ids must have shape"):
        contact_edge_index(
            positions,
            cutoff_nm=0.5,
            granularity="residue",
            residue_ids=torch.zeros(3, dtype=torch.long),
        )
    with pytest.raises(ValueError, match="integer tensor"):
        contact_edge_index(
            positions,
            cutoff_nm=0.5,
            granularity="residue",
            residue_ids=torch.zeros(2),
        )
    # Empty contacts (atoms farther than cutoff).
    far = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    empty = contact_edge_index(far, cutoff_nm=0.1)
    assert empty.shape == (2, 0)
