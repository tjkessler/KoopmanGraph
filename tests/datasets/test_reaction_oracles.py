"""Tests for :mod:`koopman_graph.datasets.reaction.oracles`."""

from __future__ import annotations

import math

import pytest
import torch

from koopman_graph.datasets.reaction import (
    ReactionSystem,
    brusselator,
    conservation_laws,
    equilibrium,
    first_order_chain,
    integrate_rk4,
    lotka_volterra,
    michaelis_menten,
    oracle_spectrum,
    oracle_timescales,
    reversible_dimerization,
    stoichiometric_rank,
)

_F64 = torch.float64


def _fed_chain(*, outflow: bool) -> ReactionSystem:
    """``∅ → A → B`` (feed 0.5), optionally with ``B → ∅`` (rate 0.25)."""
    alpha = [[0.0, 0.0], [1.0, 0.0]]
    beta = [[1.0, 0.0], [0.0, 1.0]]
    rates = [0.5, 1.0]
    if outflow:
        alpha.append([0.0, 1.0])
        beta.append([0.0, 0.0])
        rates.append(0.25)
    return ReactionSystem(
        species=("A", "B"),
        reactant_stoichiometry=torch.tensor(alpha, dtype=_F64),
        product_stoichiometry=torch.tensor(beta, dtype=_F64),
        rate_constants=torch.tensor(rates, dtype=_F64),
    )


def test_conservation_laws_span_left_nullspace() -> None:
    system = michaelis_menten()
    w = conservation_laws(system)
    assert stoichiometric_rank(system) == 2
    assert w.shape == (2, 4)
    assert torch.allclose(
        w @ system.stoichiometric_matrix, torch.zeros(2, 3, dtype=_F64), atol=1e-12
    )
    assert torch.allclose(w @ w.T, torch.eye(2, dtype=_F64), atol=1e-12)
    # The element matrix rows lie in the same space.
    assert system.element_matrix is not None
    projected = system.element_matrix @ w.T @ w
    assert torch.allclose(projected, system.element_matrix, atol=1e-12)
    assert conservation_laws(brusselator()).shape == (0, 2)


def test_chain_spectrum_is_closed_form() -> None:
    system = first_order_chain(4, rate_constants=(2.0, 1.0, 0.25))
    c0 = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=_F64)
    c_eq = equilibrium(system, c0)
    assert torch.allclose(
        c_eq, torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=_F64), atol=1e-12
    )
    spectrum = oracle_spectrum(system, c_eq)
    assert spectrum.num_conservation_laws == 1
    assert torch.allclose(
        spectrum.dynamic_eigenvalues.real, torch.tensor([-0.25, -1.0, -2.0], dtype=_F64)
    )
    assert torch.allclose(spectrum.dynamic_eigenvalues.imag, torch.zeros(3, dtype=_F64))
    assert spectrum.conservation_eigenvalues.abs().max() < 1e-12
    assert torch.allclose(
        spectrum.timescales, torch.tensor([4.0, 1.0, 0.5], dtype=_F64)
    )
    assert spectrum.is_stable
    discrete = spectrum.discrete_eigenvalues(0.1)
    assert torch.allclose(
        discrete.real, torch.exp(torch.tensor([-0.025, -0.1, -0.2], dtype=_F64))
    )
    assert torch.allclose(oracle_timescales(system, c_eq), spectrum.timescales)


def test_linear_open_equilibrium_and_missing_steady_state() -> None:
    system = _fed_chain(outflow=True)
    c_eq = equilibrium(system, torch.zeros(2, dtype=_F64))
    # A* = feed / k1 = 0.5, B* = k1 A* / k_out = 2.0.
    assert torch.allclose(c_eq, torch.tensor([0.5, 2.0], dtype=_F64))
    with pytest.raises(ValueError, match="no steady state"):
        equilibrium(_fed_chain(outflow=False), torch.zeros(2, dtype=_F64))


def test_dimerization_equilibrium_matches_closed_form_and_rk4() -> None:
    system = reversible_dimerization()
    c0 = torch.tensor([1.0, 0.0], dtype=_F64)
    c_eq = equilibrium(system, c0)
    # A + 2 A2 = 1 and 3 A^2 = 0.5 A2  =>  A = 1/4, A2 = 3/8.
    assert torch.allclose(c_eq, torch.tensor([0.25, 0.375], dtype=_F64), atol=1e-12)
    assert float(system.time_derivative(c_eq).abs().max()) < 1e-12
    spectrum = oracle_spectrum(system, c_eq)
    # Linearized along the reaction extent: -(4 k_f A* + k_r) = -3.5.
    assert torch.allclose(
        spectrum.dynamic_eigenvalues, torch.tensor([-3.5 + 0j], dtype=torch.complex128)
    )
    late = integrate_rk4(system, c0, torch.tensor([0.0, 20.0]), substeps=400)[-1]
    assert torch.allclose(late, c_eq, atol=1e-10)


def test_michaelis_menten_boundary_equilibrium_spectrum() -> None:
    system = michaelis_menten()
    c0 = torch.tensor([0.1, 1.0, 0.0, 0.0], dtype=_F64)
    c_eq = equilibrium(system, c0)
    assert torch.allclose(
        c_eq, torch.tensor([0.1, 0.0, 0.0, 1.0], dtype=_F64), atol=1e-12
    )
    spectrum = oracle_spectrum(system, c_eq)
    assert spectrum.num_conservation_laws == 2
    # Free-substrate block [[-k_on E0, k_off], [k_on E0, -(k_off + k_cat)]].
    trace, det = -2.5, 0.5
    root = math.sqrt(trace * trace - 4 * det)
    expected = torch.tensor([(trace + root) / 2, (trace - root) / 2], dtype=_F64)
    assert torch.allclose(spectrum.dynamic_eigenvalues.real, expected)
    late = integrate_rk4(system, c0, torch.tensor([0.0, 80.0]), substeps=4000)[-1]
    assert torch.allclose(late, c_eq, atol=1e-6)


def test_brusselator_unstable_focus() -> None:
    system = brusselator()
    c_eq = equilibrium(system, torch.tensor([0.5, 0.5], dtype=_F64))
    assert torch.allclose(c_eq, torch.tensor([1.0, 3.0], dtype=_F64), atol=1e-12)
    spectrum = oracle_spectrum(system, c_eq)
    assert spectrum.num_conservation_laws == 0
    assert spectrum.conservation_eigenvalues.numel() == 0
    assert torch.allclose(
        spectrum.dynamic_eigenvalues.real, torch.full((2,), 0.5, dtype=_F64)
    )
    assert torch.allclose(
        spectrum.dynamic_eigenvalues.imag.abs(),
        torch.full((2,), math.sqrt(3) / 2, dtype=_F64),
    )
    assert not spectrum.is_stable
    assert torch.allclose(spectrum.timescales, torch.full((2,), -2.0, dtype=_F64))


def test_lotka_volterra_centre_has_infinite_timescale() -> None:
    system = lotka_volterra()
    c_eq = equilibrium(
        system,
        torch.tensor([2.0, 2.0], dtype=_F64),
        guess=torch.tensor([1.5, 0.8], dtype=_F64),
    )
    assert torch.allclose(c_eq, torch.ones(2, dtype=_F64), atol=1e-12)
    spectrum = oracle_spectrum(system, c_eq)
    assert torch.allclose(spectrum.dynamic_eigenvalues.real, torch.zeros(2, dtype=_F64))
    assert torch.all(torch.isinf(spectrum.timescales))
    assert not spectrum.is_stable
    discrete = spectrum.discrete_eigenvalues(0.5)
    assert torch.allclose(discrete.abs(), torch.ones(2, dtype=_F64))


def test_equilibrium_validation() -> None:
    system = reversible_dimerization()
    c0 = torch.tensor([1.0, 0.0], dtype=_F64)
    with pytest.raises(TypeError, match="floating-point Tensor"):
        equilibrium(system, [1.0, 0.0])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"shape \(2,\)"):
        equilibrium(system, torch.ones(3, dtype=_F64))
    with pytest.raises(ValueError, match="must be finite"):
        equilibrium(system, torch.tensor([math.nan, 0.0]))
    with pytest.raises(ValueError, match=r"guess must have shape"):
        equilibrium(system, c0, guess=torch.ones(1, dtype=_F64))
    for tol in (0.0, math.nan):
        with pytest.raises(ValueError, match="tol must be"):
            equilibrium(system, c0, tol=tol)
    for max_iter in (0, 1.5, True):
        with pytest.raises(ValueError, match="max_iter must be"):
            equilibrium(system, c0, max_iter=max_iter)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="did not converge"):
        equilibrium(system, c0, max_iter=1)
    with pytest.raises(TypeError, match="must be a ReactionSystem"):
        oracle_spectrum("dimer", c0)  # type: ignore[arg-type]
    spectrum = oracle_spectrum(system, c0)
    for dt in (0.0, -1.0, math.inf):
        with pytest.raises(ValueError, match="time_step must be"):
            spectrum.discrete_eigenvalues(dt)


def test_equilibrium_rejects_one_way_boundary_point() -> None:
    """A clamped Newton stall where flux runs one way is not a steady state."""
    # A -> B with a slow B -> A: from (1, 0) the only steady state is interior.
    system = ReactionSystem(
        species=("A", "B", "C"),
        reactant_stoichiometry=torch.tensor(
            [[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=_F64
        ),
        product_stoichiometry=torch.tensor(
            [[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]], dtype=_F64
        ),
        rate_constants=torch.tensor([1.0, 1e-3], dtype=_F64),
    )
    c_eq = equilibrium(system, torch.tensor([1.0, 1.0, 0.0], dtype=_F64))
    assert float(system.time_derivative(c_eq).abs().max()) < 1e-12
    from koopman_graph.datasets.reaction.oracles import _require_stationary

    with pytest.raises(RuntimeError, match="non-stationary point"):
        _require_stationary(system, torch.tensor([1.0, 0.0, 1e-9], dtype=_F64))
