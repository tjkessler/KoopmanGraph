"""Tests for :mod:`koopman_graph.datasets.reaction.cstr`."""

from __future__ import annotations

import math

import pytest
import torch

from koopman_graph.datasets.reaction import (
    VAN_DE_VUSSE_FEED_A,
    VAN_DE_VUSSE_NOMINAL_DILUTION,
    VAN_DE_VUSSE_NOMINAL_TEMPERATURE_K,
    VanDeVusseCSTR,
    mass_action_jacobian,
    van_de_vusse_cstr,
)

_NOMINAL = {
    "temperature_k": VAN_DE_VUSSE_NOMINAL_TEMPERATURE_K,
    "dilution_rate": VAN_DE_VUSSE_NOMINAL_DILUTION,
}


def test_nominal_steady_state_matches_benchmark_operating_point() -> None:
    reactor = van_de_vusse_cstr()
    steady = reactor.steady_state(**_NOMINAL)
    # Benchmark operating point: c_A ≈ 2.14, c_B ≈ 1.09 mol/L.
    assert math.isclose(float(steady[0]), 2.14, abs_tol=5e-3)
    assert math.isclose(float(steady[1]), 1.09, abs_tol=5e-3)
    assert float(reactor.rhs(steady, **_NOMINAL).abs().max()) < 1e-9


def test_literature_balance_for_a_is_reproduced() -> None:
    """``dc_A/dt = -k1 c_A - k3 c_A² + D (c_A0 - c_A)`` with literature ``k3``."""
    reactor = van_de_vusse_cstr()
    temperature = VAN_DE_VUSSE_NOMINAL_TEMPERATURE_K
    k1 = 1.287e12 * math.exp(-9758.3 / temperature)
    k3 = 9.043e9 * math.exp(-8560.0 / temperature)
    c = torch.tensor([2.0, 1.0, 0.5, 0.25], dtype=torch.float64)
    d = 10.0
    rhs = reactor.rhs(c, temperature_k=temperature, dilution_rate=d)
    expected_a = -k1 * 2.0 - k3 * 4.0 + d * (VAN_DE_VUSSE_FEED_A - 2.0)
    assert math.isclose(float(rhs[0]), expected_a, rel_tol=1e-6)
    # dc_D/dt = k3 c_A² / 2 - D c_D in the literature.
    assert math.isclose(float(rhs[3]), 0.5 * k3 * 4.0 - d * 0.25, rel_tol=1e-6)


def test_simulation_converges_to_steady_state() -> None:
    reactor = van_de_vusse_cstr()
    steady = reactor.steady_state(**_NOMINAL)
    states = reactor.simulate(
        torch.zeros(4, dtype=torch.float64),
        torch.linspace(0.0, 1.0, 11, dtype=torch.float64),
        **_NOMINAL,
    )
    assert states.shape == (11, 4)
    assert torch.allclose(states[-1], steady, atol=1e-6)
    # Linearized controlled field is stable at the operating point.
    jac = mass_action_jacobian(
        reactor.system.at_temperature(VAN_DE_VUSSE_NOMINAL_TEMPERATURE_K), steady
    ) - VAN_DE_VUSSE_NOMINAL_DILUTION * torch.eye(4, dtype=torch.float64)
    assert bool((torch.linalg.eigvals(jac).real < 0).all())


def test_piecewise_constant_inputs_switch_operating_point() -> None:
    reactor = van_de_vusse_cstr()
    times = torch.linspace(0.0, 2.0, 5, dtype=torch.float64)
    temps = torch.tensor([387.35, 387.35, 400.0, 400.0], dtype=torch.float64)
    dilutions = torch.tensor([14.19, 14.19, 20.0, 20.0], dtype=torch.float64)
    states = reactor.simulate(
        torch.zeros(4, dtype=torch.float64),
        times,
        temperature_k=temps,
        dilution_rate=dilutions,
        substeps=200,
    )
    first = reactor.steady_state(temperature_k=387.35, dilution_rate=14.19)
    second = reactor.steady_state(temperature_k=400.0, dilution_rate=20.0)
    assert torch.allclose(states[2], first, atol=1e-6)
    assert torch.allclose(states[4], second, atol=1e-6)
    assert not torch.allclose(first, second, atol=1e-2)


def test_batched_simulation_and_rhs_shape() -> None:
    reactor = van_de_vusse_cstr(feed_a=3.0)
    assert reactor.feed.tolist() == [3.0, 0.0, 0.0, 0.0]
    c0 = torch.zeros(3, 4, dtype=torch.float64)
    states = reactor.simulate(c0, torch.tensor([0.0, 0.1]), **_NOMINAL, substeps=10)
    assert states.shape == (2, 3, 4)
    assert torch.allclose(states[-1, 0], states[-1, 2])


def test_reactor_type_and_arrhenius_data() -> None:
    reactor = van_de_vusse_cstr()
    assert isinstance(reactor, VanDeVusseCSTR)
    assert reactor.system.arrhenius is not None
    assert reactor.system.reaction_names == ("A -> B", "B -> C", "2 A -> D")
    assert torch.allclose(
        reactor.system.rate_constants,
        reactor.system.rate_constants_at(VAN_DE_VUSSE_NOMINAL_TEMPERATURE_K),
    )


def test_cstr_validation() -> None:
    reactor = van_de_vusse_cstr()
    c = torch.zeros(4, dtype=torch.float64)
    for feed in (0.0, -1.0, math.nan):
        with pytest.raises(ValueError, match="feed_a must be"):
            van_de_vusse_cstr(feed)
    with pytest.raises(ValueError, match="dilution_rate must be"):
        reactor.rhs(c, temperature_k=400.0, dilution_rate=-1.0)
    with pytest.raises(ValueError, match="temperature_k must be"):
        reactor.rhs(c, temperature_k=0.0, dilution_rate=1.0)
    times = torch.tensor([0.0, 1.0, 2.0])
    with pytest.raises(ValueError, match=r"schedule must have shape \(2,\)"):
        reactor.simulate(c, times, temperature_k=torch.ones(3) * 400, dilution_rate=1.0)
    with pytest.raises(ValueError, match="strictly increasing"):
        reactor.simulate(c, torch.tensor([0.0, 0.0]), **_NOMINAL)
    with pytest.raises(ValueError, match="non-empty 1-D"):
        reactor.simulate(c, torch.zeros(0), **_NOMINAL)
    with pytest.raises(ValueError, match="substeps must be"):
        reactor.simulate(c, times, **_NOMINAL, substeps=0)
    with pytest.raises(RuntimeError, match="did not converge"):
        reactor.steady_state(**_NOMINAL, max_iter=1)
