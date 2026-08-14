"""Coverage and error-path tests for :mod:`koopman_graph.model`."""

from __future__ import annotations

import pytest
import torch

from koopman_graph.analysis.spectrum import attribute_mode_energy
from koopman_graph.data.validation import (
    validate_hetero_control_inputs,
)


def test_attribute_mode_energy_and_control_input_guards() -> None:
    """Spectrum attribution and hetero control validation reject bad inputs."""
    k_eff = torch.eye(6)
    eigenvectors = torch.eye(6)
    with pytest.raises(ValueError, match="eigenvectors must have shape"):
        attribute_mode_energy(k_eff, torch.ones(6), latent_dim=2)
    with pytest.raises(ValueError, match="rows must match"):
        attribute_mode_energy(k_eff, torch.eye(4, 6), latent_dim=2)
    with pytest.raises(ValueError, match="latent_dim must be positive"):
        attribute_mode_energy(k_eff, eigenvectors, latent_dim=0)
    with pytest.raises(ValueError, match="not divisible by"):
        attribute_mode_energy(k_eff, eigenvectors, latent_dim=5)
    with pytest.raises(ValueError, match="latent slice"):
        attribute_mode_energy(
            k_eff,
            eigenvectors,
            latent_dim=2,
            node_type_slices={"a": slice(0, 4)},
            mode_indices=(0,),
        )
    report = attribute_mode_energy(
        k_eff,
        eigenvectors,
        latent_dim=2,
        node_type_slices={"a": slice(0, 2), "b": slice(2, 3)},
        mode_indices=(0,),
    )
    assert report.type_fractions["a"].shape == (1,)

    with pytest.raises(ValueError, match="non-empty"):
        validate_hetero_control_inputs({}, num_timesteps=2, num_nodes={"node": 4})
    with pytest.raises(ValueError, match="unknown node types"):
        validate_hetero_control_inputs(
            {"ghost": torch.randn(2, 1)},
            num_timesteps=2,
            num_nodes={"node": 4},
        )
    with pytest.raises(ValueError, match="control_inputs\\['node'\\]"):
        validate_hetero_control_inputs(
            {"node": torch.randn(2, 5, 1)},
            num_timesteps=2,
            num_nodes={"node": 4},
        )
    with pytest.raises(ValueError, match="share a common control_dim"):
        validate_hetero_control_inputs(
            {
                "gen": torch.randn(2, 1),
                "load": torch.randn(2, 2),
            },
            num_timesteps=2,
            num_nodes={"gen": 2, "load": 3},
        )
    with pytest.raises(ValueError, match=r"\(T, N, C\) require a single node type"):
        validate_hetero_control_inputs(
            torch.randn(2, 5, 1),
            num_timesteps=2,
            num_nodes={"gen": 2, "load": 3},
        )
