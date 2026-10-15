"""Tests for :mod:`koopman_graph.datasets.reaction.catalog`."""

from __future__ import annotations

import math

import pytest
import torch

from koopman_graph.datasets.reaction import (
    DEFAULT_BRUSSELATOR_PARAMS,
    DEFAULT_DIMERIZATION_RATES,
    DEFAULT_LOTKA_VOLTERRA_RATES,
    DEFAULT_MICHAELIS_MENTEN_RATES,
    ReactionSystem,
    brusselator,
    check_element_balance,
    first_order_chain,
    integrate_rk4,
    lotka_volterra,
    michaelis_menten,
    reversible_dimerization,
)

_CLOSED = [first_order_chain, reversible_dimerization, michaelis_menten]
_OPEN = [brusselator, lotka_volterra]


@pytest.mark.parametrize("factory", _CLOSED + _OPEN)
def test_catalog_returns_named_validated_systems(factory: object) -> None:
    system = factory()  # type: ignore[operator]
    assert isinstance(system, ReactionSystem)
    assert system.reaction_names is not None
    assert len(system.reaction_names) == system.num_reactions
    assert system.rate_constants.dtype == torch.float64
    # Frozen defaults: two calls give the same system.
    assert system.to_dict() == factory().to_dict()  # type: ignore[operator]


@pytest.mark.parametrize("factory", _CLOSED)
def test_closed_systems_are_element_balanced(factory: object) -> None:
    system = factory()  # type: ignore[operator]
    report = check_element_balance(system)
    assert report.balanced, report.unbalanced_reactions


@pytest.mark.parametrize("factory", _OPEN)
def test_open_systems_have_no_element_matrix(factory: object) -> None:
    assert factory().element_matrix is None  # type: ignore[operator]


def test_first_order_chain_defaults_and_options() -> None:
    system = first_order_chain()
    assert system.species == ("S0", "S1", "S2")
    assert system.rate_constants.tolist() == [1.0, 0.5]
    assert system.is_linear
    assert system.reaction_names == ("S0 -> S1", "S1 -> S2")
    long = first_order_chain(5, rate_constants=(4.0, 3.0, 2.0, 1.0))
    assert long.num_species == 5
    assert long.rate_constants.tolist() == [4.0, 3.0, 2.0, 1.0]
    assert long.name == "first_order_chain_5"
    for bad in (1, 65, 2.5, True):
        with pytest.raises(ValueError, match="num_species must be an int"):
            first_order_chain(bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must have 2 entries"):
        first_order_chain(3, rate_constants=(1.0,))


def test_default_parameters_are_used() -> None:
    assert reversible_dimerization().rate_constants.tolist() == list(
        DEFAULT_DIMERIZATION_RATES
    )
    assert michaelis_menten().rate_constants.tolist() == list(
        DEFAULT_MICHAELIS_MENTEN_RATES
    )
    a, b = DEFAULT_BRUSSELATOR_PARAMS
    assert brusselator().rate_constants.tolist() == [a, b, 1.0, 1.0]
    assert b > 1.0 + a * a  # default Brusselator is in the oscillatory regime
    assert lotka_volterra().rate_constants.tolist() == list(
        DEFAULT_LOTKA_VOLTERRA_RATES
    )
    assert not michaelis_menten().is_linear


@pytest.mark.parametrize(("a", "b"), [(0.0, 1.0), (1.0, -2.0), (math.inf, 1.0)])
def test_brusselator_rejects_bad_params(a: float, b: float) -> None:
    with pytest.raises(ValueError, match="must be finite and > 0"):
        brusselator(a, b)


@pytest.mark.parametrize(
    ("factory", "kwargs"),
    [
        (first_order_chain, {"rate_constants": (1.0, 0.0)}),
        (reversible_dimerization, {"k_reverse": 0.0}),
        (michaelis_menten, {"k_cat": math.nan}),
        (lotka_volterra, {"gamma": -1.0}),
    ],
)
def test_catalog_rejects_non_positive_rates(factory: object, kwargs: dict) -> None:
    with pytest.raises(ValueError, match="must be finite and > 0"):
        factory(**kwargs)  # type: ignore[operator]


def test_lotka_volterra_orbit_is_closed() -> None:
    """Conserved quantity ``V = βx - γ ln x + βy - α ln y`` along RK4 orbits."""
    system = lotka_volterra()
    c0 = torch.tensor([2.0, 1.0], dtype=torch.float64)
    states = integrate_rk4(system, c0, torch.linspace(0.0, 10.0, 101), substeps=20)
    x, y = states[:, 0], states[:, 1]
    invariant = x - torch.log(x) + y - torch.log(y)
    assert float((invariant - invariant[0]).abs().max()) < 1e-6


def test_michaelis_menten_converts_all_substrate() -> None:
    system = michaelis_menten()
    c0 = torch.tensor([0.1, 1.0, 0.0, 0.0], dtype=torch.float64)
    states = integrate_rk4(system, c0, torch.tensor([0.0, 80.0]), substeps=4000)
    final = states[-1]
    assert torch.allclose(
        final, torch.tensor([0.1, 0.0, 0.0, 1.0], dtype=torch.float64), atol=1e-6
    )
