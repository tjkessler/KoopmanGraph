"""Tests for :mod:`koopman_graph.datasets.reaction.integrate`."""

from __future__ import annotations

import math
import sys

import pytest
import torch

from koopman_graph.datasets.reaction import (
    ReactionSystem,
    element_balance_violation,
    integrate_expm,
    integrate_reaction_system,
    integrate_rk4,
    integrate_stiff,
    linear_rate_equations,
    mass_action_jacobian,
    require_scipy,
)


def _chain(*, feed: float = 0.0) -> ReactionSystem:
    """``A -> B -> C`` (rates 1.0, 0.3) with optional zeroth-order ``∅ -> A``."""
    return ReactionSystem(
        species=("A", "B", "C"),
        reactant_stoichiometry=torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
        ),
        product_stoichiometry=torch.tensor(
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
        ),
        rate_constants=torch.tensor([1.0, 0.3, feed], dtype=torch.float64),
        element_matrix=torch.tensor([[1.0, 1.0, 1.0]]),
        elements=("M",),
    )


def _dimerization() -> ReactionSystem:
    """Closed ``2 A <-> A2``; element ``X`` with ``A: 1, A2: 2``."""
    return ReactionSystem(
        species=("A", "A2"),
        reactant_stoichiometry=torch.tensor([[2.0, 0.0], [0.0, 1.0]]),
        product_stoichiometry=torch.tensor([[0.0, 1.0], [2.0, 0.0]]),
        rate_constants=torch.tensor([3.0, 0.5]),
        element_matrix=torch.tensor([[1.0, 2.0]]),
        elements=("X",),
    )


def _robertson() -> ReactionSystem:
    """Robertson's stiff problem (rates 0.04, 3e7, 1e4)."""
    return ReactionSystem(
        species=("A", "B", "C"),
        reactant_stoichiometry=torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 1.0, 1.0]]
        ),
        product_stoichiometry=torch.tensor(
            [[0.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]]
        ),
        rate_constants=torch.tensor([0.04, 3.0e7, 1.0e4]),
        element_matrix=torch.tensor([[1.0, 1.0, 1.0]]),
        elements=("M",),
    )


def test_linear_rate_equations_for_chain_with_feed() -> None:
    a, b = linear_rate_equations(_chain(feed=0.2))
    expected_a = torch.tensor(
        [[-1.0, 0.0, 0.0], [1.0, -0.3, 0.0], [0.0, 0.3, 0.0]], dtype=torch.float64
    )
    assert torch.allclose(a, expected_a)
    assert torch.allclose(b, torch.tensor([0.2, 0.0, 0.0], dtype=torch.float64))
    with pytest.raises(ValueError, match="is not linear"):
        linear_rate_equations(_dimerization())
    with pytest.raises(TypeError, match="must be a ReactionSystem"):
        linear_rate_equations("chain")  # type: ignore[arg-type]


def test_expm_matches_closed_form_chain() -> None:
    system = _chain()
    times = torch.tensor([0.0, 0.5, 2.0, 7.0], dtype=torch.float64)
    c0 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    states = integrate_expm(system, c0, times)
    assert states.shape == (4, 3)
    assert torch.equal(states[0], c0)
    k1, k2 = 1.0, 0.3
    for t, state in zip(times.tolist(), states, strict=True):
        a = math.exp(-k1 * t)
        b = k1 / (k2 - k1) * (math.exp(-k1 * t) - math.exp(-k2 * t))
        assert math.isclose(float(state[0]), a, rel_tol=1e-10, abs_tol=1e-12)
        assert math.isclose(float(state[1]), b, rel_tol=1e-10, abs_tol=1e-12)
        assert math.isclose(float(state.sum()), 1.0, rel_tol=1e-12)


def test_expm_feed_is_exact_and_rk4_agrees() -> None:
    system = _chain(feed=0.2)
    times = torch.linspace(1.0, 6.0, 6, dtype=torch.float64)
    c0 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.5, 0.5]], dtype=torch.float64)
    exact = integrate_expm(system, c0, times)
    assert exact.shape == (6, 2, 3)
    # With feed f, total mass grows linearly: M(t) = M(t0) + f (t - t0).
    totals = exact.sum(dim=-1)
    expected = c0.sum(dim=-1) + 0.2 * (times - times[0]).unsqueeze(-1)
    assert torch.allclose(totals, expected, atol=1e-12)
    approx = integrate_rk4(system, c0, times, substeps=20)
    assert torch.allclose(approx, exact, atol=1e-7)


def test_auto_picks_expm_for_linear_and_rk4_otherwise() -> None:
    times = torch.tensor([0.0, 1.0], dtype=torch.float64)
    linear = integrate_reaction_system(
        _chain(), torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64), times
    )
    assert torch.allclose(
        linear, integrate_expm(_chain(), torch.tensor([1.0, 0.0, 0.0]), times)
    )
    nonlinear = integrate_reaction_system(
        _dimerization(), torch.tensor([1.0, 0.0], dtype=torch.float64), times
    )
    assert torch.allclose(
        nonlinear,
        integrate_rk4(
            _dimerization(), torch.tensor([1.0, 0.0], dtype=torch.float64), times
        ),
    )


def test_auto_accepts_substeps_for_linear_and_nonlinear() -> None:
    times = torch.tensor([0.0, 1.0], dtype=torch.float64)
    c0 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    linear = integrate_reaction_system(_chain(), c0, times, substeps=20)
    assert torch.allclose(linear, integrate_expm(_chain(), c0, times))
    explicit = integrate_reaction_system(_chain(), c0, times, method="expm")
    assert torch.allclose(explicit, linear)
    dimer0 = torch.tensor([1.0, 0.0], dtype=torch.float64)
    nonlinear = integrate_reaction_system(_dimerization(), dimer0, times, substeps=20)
    assert torch.allclose(
        nonlinear, integrate_rk4(_dimerization(), dimer0, times, substeps=20)
    )
    with pytest.raises(ValueError, match="only accepts 'substeps'"):
        integrate_reaction_system(_chain(), c0, times, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_expm_keeps_input_device() -> None:
    c0 = torch.tensor([1.0, 0.0, 0.0], device="cuda")
    states = integrate_expm(_chain(), c0, torch.tensor([0.0, 1.0]))
    assert states.device == c0.device


def test_rk4_conserves_elements_and_keeps_dtype() -> None:
    system = _dimerization()
    c0 = torch.tensor([1.0, 0.2], dtype=torch.float32)
    times = torch.linspace(0.0, 3.0, 13)
    states = integrate_rk4(system, c0, times, substeps=8)
    assert states.dtype == torch.float32
    violation = element_balance_violation(states, c0, system)
    assert float(violation.abs().max()) < 1e-5
    # Reaches the dimerization equilibrium k_f A^2 = k_r A2.
    a, a2 = states[-1].tolist()
    assert math.isclose(3.0 * a * a, 0.5 * a2, rel_tol=1e-3)


def test_rk4_is_differentiable_in_initial_state() -> None:
    c0 = torch.tensor([1.0, 0.0], dtype=torch.float64, requires_grad=True)
    states = integrate_rk4(_dimerization(), c0, torch.tensor([0.0, 0.5]))
    states[-1, 1].backward()
    assert c0.grad is not None
    assert bool(torch.isfinite(c0.grad).all())


def test_mass_action_jacobian_matches_autograd() -> None:
    system = _robertson()
    for c in (
        torch.tensor([0.7, 2e-5, 0.3], dtype=torch.float64),
        torch.zeros(3, dtype=torch.float64),
    ):
        expected = torch.autograd.functional.jacobian(system.time_derivative, c)
        jac = mass_action_jacobian(system, c)
        assert torch.allclose(jac, expected)
        assert bool(torch.isfinite(jac).all())
    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        mass_action_jacobian(system, torch.zeros(2, 3, dtype=torch.float64))
    with pytest.raises(ValueError, match="concentrations must be finite"):
        mass_action_jacobian(system, torch.tensor([math.nan, 0.0, 0.0]))


def test_mass_action_jacobian_with_wide_reactant_sets() -> None:
    """Three-reactant and zeroth-order reactions over unused species."""
    system = ReactionSystem(
        species=("A", "B", "C", "D", "E"),
        reactant_stoichiometry=torch.tensor(
            [
                [2.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 3.0, 0.0, 0.0],
            ]
        ),
        product_stoichiometry=torch.tensor(
            [
                [0.0, 0.0, 1.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 1.0],
            ]
        ),
        rate_constants=torch.tensor([1.5, 0.2, 0.7]),
    )
    for c in (
        torch.tensor([0.4, 1.3, 0.8, 0.6, 2.0], dtype=torch.float64),
        torch.tensor([0.0, 1.3, 0.0, 0.6, 0.0], dtype=torch.float64),
    ):
        expected = torch.autograd.functional.jacobian(system.time_derivative, c)
        assert torch.allclose(mass_action_jacobian(system, c), expected)


def test_stiff_robertson_conserves_mass() -> None:
    pytest.importorskip("scipy")
    system = _robertson()
    times = torch.tensor([0.0, 1.0, 10.0, 100.0], dtype=torch.float64)
    c0 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    for method in ("bdf", "radau", "lsoda"):
        states = integrate_reaction_system(system, c0, times, method=method)
        assert states.shape == (4, 3)
        assert states.dtype == torch.float64
        assert torch.allclose(states[0], c0)
        violation = element_balance_violation(states, c0, system)
        assert float(violation.abs().max()) < 1e-6
    # Reference value at t = 1 (Hairer & Wanner): A ≈ 0.9664597.
    bdf = integrate_stiff(system, c0, times)
    assert math.isclose(float(bdf[1, 0]), 0.9664597, rel_tol=1e-5)


def test_stiff_solver_failure_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    scipy_integrate = pytest.importorskip("scipy.integrate")

    class _Failed:
        success = False
        message = "step size too small"

    monkeypatch.setattr(scipy_integrate, "solve_ivp", lambda *a, **k: _Failed())
    with pytest.raises(RuntimeError, match="step size too small"):
        integrate_stiff(
            _robertson(), torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 1.0])
        )


def test_require_scipy_names_reaction_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "scipy", None)
    monkeypatch.setitem(sys.modules, "scipy.integrate", None)
    with pytest.raises(ImportError, match=r"koopman-graph\[reaction\]"):
        require_scipy()
    with pytest.raises(ImportError, match=r"koopman-graph\[reaction\]"):
        integrate_reaction_system(
            _robertson(),
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor([0.0, 1.0]),
            method="bdf",
        )


def test_single_time_returns_initial_state() -> None:
    c0 = torch.tensor([0.4, 0.6, 0.0], dtype=torch.float64)
    times = torch.tensor([2.0], dtype=torch.float64)
    assert torch.equal(integrate_expm(_chain(), c0, times), c0.unsqueeze(0))
    assert torch.equal(integrate_rk4(_chain(), c0, times), c0.unsqueeze(0))


@pytest.mark.parametrize(
    ("times", "error", "match"),
    [
        ([0.0, 1.0], TypeError, "times must be a Tensor"),
        (torch.zeros(0), ValueError, "non-empty 1-D"),
        (torch.zeros(2, 2), ValueError, "non-empty 1-D"),
        (torch.tensor([0.0, math.inf]), ValueError, "must be finite"),
        (torch.tensor([0.0, 1.0, 1.0]), ValueError, "strictly increasing"),
    ],
)
def test_time_validation(times: object, error: type[Exception], match: str) -> None:
    with pytest.raises(error, match=match):
        integrate_rk4(_chain(), torch.ones(3), times)  # type: ignore[arg-type]


def test_integer_times_are_accepted() -> None:
    states = integrate_rk4(_chain(), torch.ones(3), torch.tensor([0, 1, 2]))
    assert states.shape == (3, 3)


@pytest.mark.parametrize(
    ("initial", "error", "match"),
    [
        ([1.0, 0.0, 0.0], TypeError, "must be a Tensor"),
        (torch.ones(3, dtype=torch.long), TypeError, "must be floating point"),
        (torch.ones(2), ValueError, r"shape \(\.\.\., 3\)"),
        (torch.tensor(1.0), ValueError, r"shape \(\.\.\., 3\)"),
        (torch.tensor([1.0, math.nan, 0.0]), ValueError, "initial must be finite"),
    ],
)
def test_initial_validation(
    initial: object, error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        integrate_expm(_chain(), initial, torch.tensor([0.0, 1.0]))  # type: ignore[arg-type]


def test_option_validation() -> None:
    c0 = torch.ones(3)
    times = torch.tensor([0.0, 1.0])
    for substeps in (0, 1.5, True):
        with pytest.raises(ValueError, match="substeps must be a positive int"):
            integrate_rk4(_chain(), c0, times, substeps=substeps)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="takes no options"):
        integrate_reaction_system(_chain(), c0, times, method="expm", substeps=4)
    with pytest.raises(ValueError, match="method must be one of"):
        integrate_reaction_system(_chain(), c0, times, method="euler")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="method must be one of"):
        integrate_stiff(_chain(), c0, times, method="rk45")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="single initial state"):
        integrate_stiff(_chain(), torch.ones(2, 3), times)
    with pytest.raises(TypeError, match="must be a ReactionSystem"):
        integrate_rk4("chain", c0, times)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must be a ReactionSystem"):
        integrate_reaction_system(None, c0, times)  # type: ignore[arg-type]
