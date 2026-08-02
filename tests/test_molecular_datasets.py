"""Tests for synthetic molecular teaching datasets and FAIR metadata."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from koopman_graph.datasets.molecular import (
    generate_synthetic_two_state,
    load_synthetic_two_state_metadata,
    synthetic_two_state_oracle_timescale,
    synthetic_two_state_slow_eigenvalue,
)


def test_metadata_loads_via_package_resources() -> None:
    """Oracle JSON is reachable without hard-coded absolute paths."""
    meta = load_synthetic_two_state_metadata()
    assert meta["dataset_id"] == "synthetic_two_state_v1"
    assert meta["version"] == "v1"
    assert meta["transition_prob"] == pytest.approx(0.05)
    assert meta["slow_eigenvalue"] == pytest.approx(0.9)
    assert "src/koopman_graph" not in str(meta)
    # Resource lives under the installed / editable package tree.
    package_data = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "koopman_graph"
        / "datasets"
        / "molecular"
        / "data"
        / "synthetic_two_state_v1.json"
    )
    assert package_data.is_file()


def test_oracle_timescale_matches_closed_form_and_metadata() -> None:
    """Hand-checked ``-1 / ln(0.9)`` matches helper and JSON metadata."""
    expected = -1.0 / math.log(0.9)
    assert synthetic_two_state_slow_eigenvalue(0.05) == pytest.approx(0.9)
    assert synthetic_two_state_oracle_timescale(0.05, lag_steps=1) == pytest.approx(
        expected
    )
    meta = load_synthetic_two_state_metadata()
    assert meta["oracle_slow_timescale_steps"] == pytest.approx(expected)


def test_generator_is_seed_deterministic() -> None:
    """Identical seeds reproduce features and state labels."""
    a = generate_synthetic_two_state(num_timesteps=32, seed=7)
    b = generate_synthetic_two_state(num_timesteps=32, seed=7)
    c = generate_synthetic_two_state(num_timesteps=32, seed=8)
    assert torch.equal(a.state_labels, b.state_labels)
    assert torch.equal(a.sequence[0].x, b.sequence[0].x)
    assert not torch.equal(a.state_labels, c.state_labels)
    assert a.dataset_id == "synthetic_two_state_v1"
    assert a.oracle_slow_timescale_steps == pytest.approx(
        synthetic_two_state_oracle_timescale(a.transition_prob)
    )


def test_generator_builds_static_contact_topology() -> None:
    """Contact edges are shared across snapshots and match positions."""
    traj = generate_synthetic_two_state(num_timesteps=16, seed=0)
    assert traj.sequence.num_timesteps == 16
    assert traj.edge_index.shape[0] == 2
    assert traj.edge_index.numel() > 0
    assert traj.positions_nm.shape == (4, 3)
    for t in range(traj.sequence.num_timesteps):
        assert torch.equal(traj.sequence[t].edge_index, traj.edge_index)
        assert traj.sequence[t].x.shape == (4, 2)


def test_transition_prob_validation() -> None:
    """Invalid flip probabilities are rejected."""
    with pytest.raises(ValueError, match="\\(0, 0.5\\)"):
        generate_synthetic_two_state(transition_prob=0.0)
    with pytest.raises(ValueError, match="\\(0, 0.5\\)"):
        generate_synthetic_two_state(transition_prob=0.5)
    with pytest.raises(ValueError, match="num_timesteps must be >= 2"):
        generate_synthetic_two_state(num_timesteps=1)


def test_exported_from_molecular_package() -> None:
    """Synthetic helpers are exported from ``datasets.molecular``."""
    import koopman_graph.datasets.molecular as molecular

    assert "generate_synthetic_two_state" in molecular.__all__
    assert "load_synthetic_two_state_metadata" in molecular.__all__
    assert "SyntheticTwoStateTrajectory" in molecular.__all__
