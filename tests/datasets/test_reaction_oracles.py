"""Tests for :mod:`koopman_graph.datasets.reaction.oracles`."""

from __future__ import annotations

import math

import pytest
import torch

from koopman_graph.datasets.reaction import (
    OracleSpectrum,
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


def _system(
    reactants: list[list[float]], products: list[list[float]], rates: list[float]
) -> ReactionSystem:
    """Build a small system from nested lists."""
    return ReactionSystem(
        species=tuple(f"S{i}" for i in range(len(reactants[0]))),
        reactant_stoichiometry=torch.tensor(reactants, dtype=_F64),
        product_stoichiometry=torch.tensor(products, dtype=_F64),
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


def test_linear_equilibrium_with_non_unique_steady_state_is_the_limit() -> None:
    # A -> B (1), A -> C (3): every (0, B, 1 - B) is steady; the limit
    # splits A in the ratio of the rates.
    system = _system([[1.0, 0.0, 0.0]] * 2, [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [1, 3])
    c0 = torch.tensor([1.0, 0.0, 0.0], dtype=_F64)
    c_eq = equilibrium(system, c0)
    assert c_eq[0] == 0.0  # rounding noise is snapped to zero, not negative
    assert torch.allclose(c_eq, torch.tensor([0.0, 0.25, 0.75], dtype=_F64))
    late = integrate_rk4(system, c0, torch.tensor([0.0, 40.0]), substeps=4000)[-1]
    assert torch.allclose(late, c_eq, atol=1e-10)


def test_linear_equilibrium_rejects_defective_zero_mode() -> None:
    # S1 -> S0 + S1: S0 grows linearly while S1 > 0 and never settles.
    system = _system([[0.0, 1.0]], [[1.0, 1.0]], [1.0])
    with pytest.raises(ValueError, match="no limiting steady state"):
        equilibrium(system, torch.tensor([0.0, 1.0], dtype=_F64))


def test_newton_stall_raises_instead_of_returning_non_root() -> None:
    # -> X (1), 2 X -> 3 X (1): dX/dt = 1 + X^2 > 0 has no root. From X = 0
    # the Jacobian vanishes, Newton cannot move, and it must not report
    # convergence.
    system = _system([[0.0], [2.0]], [[1.0], [3.0]], [1.0, 1.0])
    with pytest.raises(RuntimeError, match="stalled"):
        equilibrium(system, torch.zeros(1, dtype=_F64))


def test_near_zero_real_parts_are_neutral() -> None:
    eigenvalues = torch.tensor([1e-17 + 1j, -1e-17 - 1j], dtype=torch.complex128)
    fields = {
        "state": torch.zeros(2, dtype=_F64),
        "jacobian": torch.zeros(2, 2, dtype=_F64),
        "dynamic_eigenvalues": eigenvalues,
        "conservation_eigenvalues": torch.zeros(0, dtype=torch.complex128),
        "num_conservation_laws": 0,
    }
    exact = OracleSpectrum(**fields)
    assert torch.isfinite(exact.timescales).all()
    tolerant = OracleSpectrum(**fields, zero_tolerance=1e-12)
    assert torch.isinf(tolerant.timescales).all()
    assert (tolerant.timescales > 0).all()
    assert not tolerant.is_stable
    spectrum = oracle_spectrum(lotka_volterra(), torch.ones(2, dtype=_F64))
    assert spectrum.zero_tolerance > 0.0


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


def test_equilibrium_rejects_one_way_boundary_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A clamped Newton stall where flux runs one way is not a steady state."""
    # A + B -> C with a slow C -> A + B: the only steady state is interior.
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

    # Force Newton to stall where it starts (zero Jacobian -> zero step): the
    # step test passes immediately, and equilibrium must still refuse the point.
    from koopman_graph.datasets.reaction import oracles

    monkeypatch.setattr(
        oracles,
        "mass_action_jacobian",
        lambda system, c: torch.zeros(3, 3, dtype=_F64),
    )
    # Either guard may fire first, the line-search stall check or the
    # stationarity check; both refuse the point.
    with pytest.raises(RuntimeError, match="non-stationary point|line search stalled"):
        equilibrium(system, torch.tensor([1.0, 1.0, 0.0], dtype=_F64))


def test_eigenvalue_timescales_keep_slow_modes_in_stiff_spectra() -> None:
    """Only round-off-level real parts become infinite, not slow stiff modes."""
    from koopman_graph.datasets.reaction import eigenvalue_timescales

    # max|λ| = 1e10, so the round-off cutoff is about 2.2e-4.
    eigenvalues = torch.tensor([-1e10, -1e-2, -1e-3, 1e-6], dtype=_F64)
    timescales = eigenvalue_timescales(eigenvalues)
    assert torch.allclose(
        timescales[:3], torch.tensor([1e-10, 1e2, 1e3], dtype=_F64), rtol=1e-12
    )
    assert math.isinf(float(timescales[3]))
    assert eigenvalue_timescales(torch.zeros(0, dtype=_F64)).shape == (0,)


def test_equilibrium_accepts_vanishing_leftovers_on_boundary() -> None:
    """Draining onto a zero-flux boundary is stationary despite tiny leftovers."""
    from koopman_graph.datasets.reaction import sample_random_network

    system = sample_random_network(seed=0).system
    c0 = torch.full((6,), 0.55, dtype=_F64)
    relaxed = integrate_rk4(system, c0, torch.tensor([0.0, 200.0]), substeps=4000)[-1]
    c_eq = equilibrium(system, c0, guess=relaxed)
    assert float(system.time_derivative(c_eq).abs().max()) < 1e-12
    assert system.element_matrix is not None
    drift = system.element_matrix @ (c_eq - c0)
    assert float(drift.abs().max()) < 1e-10


def test_conservation_laws_are_canonical_across_bases() -> None:
    """The basis does not depend on which orthonormal basis an SVD returns."""
    from koopman_graph.datasets.reaction.oracles import _canonical_row_basis

    system = michaelis_menten()
    w = conservation_laws(system)
    theta = torch.tensor(0.7, dtype=_F64)
    rotation = torch.stack(
        [
            torch.stack([torch.cos(theta), -torch.sin(theta)]),
            torch.stack([torch.sin(theta), torch.cos(theta)]),
        ]
    )
    rotated = -(rotation @ w)  # another valid orthonormal basis, sign flipped
    assert torch.allclose(_canonical_row_basis(rotated), w, atol=1e-12)
    # Substrate backbone S + ES + P is the second canonical row.
    expected = torch.tensor([0.0, 1.0, 1.0, 1.0], dtype=_F64) / math.sqrt(3.0)
    assert torch.allclose(w[1], expected, atol=1e-12)
    assert _canonical_row_basis(torch.zeros(0, 4, dtype=_F64)).shape == (0, 4)
    # Round-off zeros are exact zeros, so stored bases match across platforms.
    assert float(w[1, 0]) == 0.0
    noisy = w + 1e-16 * torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    assert float(_canonical_row_basis(noisy.to(_F64))[1, 0]) == 0.0
