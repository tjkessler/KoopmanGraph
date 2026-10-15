"""Tests for :mod:`koopman_graph.datasets.reaction.random_network`."""

from __future__ import annotations

import json
import math

import pytest
import torch

from koopman_graph.datasets.reaction import (
    ReactionSystem,
    check_element_balance,
    integrate_rk4,
    sample_random_network,
)


@pytest.mark.parametrize("seed", range(20))
def test_sampled_networks_are_element_balanced(seed: int) -> None:
    network = sample_random_network(seed=seed)
    system = network.system
    assert check_element_balance(system).balanced
    assert system.num_species == 6
    assert system.num_reactions == 8
    assert int(system.reaction_orders.max()) <= 2
    assert int(system.product_stoichiometry.sum(dim=1).max()) <= 2
    # First species are the bare elements; compositions are unique.
    assert system.element_matrix is not None
    composition = system.element_matrix.T
    assert torch.equal(composition[:2], torch.eye(2, dtype=torch.float64))
    assert len({tuple(row) for row in composition.tolist()}) == 6
    assert int(composition.sum(dim=1).max()) <= 4


def test_seed_reproduces_network_and_provenance() -> None:
    a = sample_random_network(seed=7)
    b = sample_random_network(seed=7)
    assert a.system.to_dict() == b.system.to_dict()
    assert a.seed == 7
    assert a.parameters["seed"] == 7
    assert a.system.name == "random_network_seed7"
    payload = json.loads(json.dumps(a.to_dict()))
    assert ReactionSystem.from_dict(payload["system"]).to_dict() == a.system.to_dict()
    assert payload["parameters"]["rate_log10_range"] == [-1.0, 1.0]
    assert payload["num_candidates"] >= 8
    assert sample_random_network(seed=8).system.to_dict() != a.system.to_dict()


def test_rates_are_log_uniform_in_range() -> None:
    network = sample_random_network(
        num_species=8, num_reactions=20, rate_log10_range=(-2.0, 0.5), seed=3
    )
    log_k = torch.log10(network.system.rate_constants)
    assert float(log_k.min()) >= -2.0
    assert float(log_k.max()) <= 0.5
    fixed = sample_random_network(rate_log10_range=(0.0, 0.0), seed=1)
    assert torch.allclose(
        fixed.system.rate_constants, torch.ones(8, dtype=torch.float64)
    )


def test_max_order_one_gives_linear_dissociation_network() -> None:
    system = sample_random_network(num_reactions=4, max_order=1, seed=2).system
    assert system.is_linear
    assert check_element_balance(system).balanced
    assert bool((system.reaction_orders == 1).all())
    # Unique compositions rule out 1 -> 1 isomerizations, so every reaction
    # dissociates one species into two.
    assert bool((system.product_stoichiometry.sum(dim=1) == 2).all())


def test_sampled_network_integrates_and_conserves_elements() -> None:
    system = sample_random_network(seed=4).system
    c0 = torch.full((6,), 0.5, dtype=torch.float64)
    states = integrate_rk4(system, c0, torch.linspace(0.0, 2.0, 11), substeps=200)
    assert system.element_matrix is not None
    drift = (states - c0) @ system.element_matrix.T
    assert float(drift.abs().max()) < 1e-10
    assert bool(torch.isfinite(states).all())


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"num_elements": 0}, "num_elements must be an int"),
        ({"num_species": 2, "num_elements": 2}, "num_species must be an int"),
        ({"num_species": 40}, "num_species must be an int"),
        ({"num_reactions": 0}, "num_reactions must be an int"),
        ({"max_atoms": 1}, "max_atoms must be an int"),
        ({"max_order": 3}, "max_order must be an int"),
        ({"seed": -1}, "seed must be an int"),
        ({"seed": True}, "seed must be an int"),
        ({"rate_log10_range": (1.0, 0.0)}, "rate_log10_range"),
        ({"rate_log10_range": (0.0, math.inf)}, "rate_log10_range"),
        ({"num_species": 10, "max_atoms": 2}, "could not build 10 distinct species"),
        ({"num_reactions": 500}, "balanced candidate reactions"),
    ],
)
def test_sampler_validation(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        sample_random_network(**kwargs)  # type: ignore[arg-type]
