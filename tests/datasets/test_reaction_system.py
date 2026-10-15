"""Tests for :mod:`koopman_graph.datasets.reaction.system`."""

from __future__ import annotations

import math

import pytest
import torch

from koopman_graph.datasets.reaction import (
    GAS_CONSTANT_J_PER_MOL_K,
    ArrheniusParams,
    ReactionSystem,
    check_element_balance,
)


def _dimerization() -> ReactionSystem:
    """``2 A <-> A2`` with composition ``A: 1 X, A2: 2 X``."""
    return ReactionSystem(
        species=("A", "A2"),
        reactant_stoichiometry=torch.tensor([[2.0, 0.0], [0.0, 1.0]]),
        product_stoichiometry=torch.tensor([[0.0, 1.0], [2.0, 0.0]]),
        rate_constants=torch.tensor([3.0, 0.5]),
        element_matrix=torch.tensor([[1.0, 2.0]]),
        elements=("X",),
        reaction_names=("forward", "reverse"),
        arrhenius=(
            ArrheniusParams(1.0e6, 4.0e4),
            ArrheniusParams(2.0e3, 1.0e4, temperature_exponent=0.5),
        ),
        name="dimerization",
    )


def _chain() -> ReactionSystem:
    """Linear chain ``A -> B -> C`` with a zeroth-order feed ``∅ -> A``."""
    return ReactionSystem(
        species=("A", "B", "C"),
        reactant_stoichiometry=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        ),
        product_stoichiometry=torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ),
        rate_constants=torch.tensor([0.2, 1.0, 0.1]),
        element_matrix=torch.tensor([[1.0, 1.0, 1.0]]),
        elements=("M",),
    )


def test_stoichiometric_matrix_orders_and_linearity() -> None:
    system = _dimerization()
    assert system.num_species == 2
    assert system.num_reactions == 2
    expected = torch.tensor([[-2.0, 2.0], [1.0, -1.0]], dtype=torch.float64)
    assert torch.equal(system.stoichiometric_matrix, expected)
    assert system.reaction_orders.tolist() == [2.0, 1.0]
    assert not system.is_linear
    assert _chain().is_linear


def test_mass_action_rates_and_rhs_match_hand_calculation() -> None:
    system = _dimerization()
    c = torch.tensor([0.4, 0.3], dtype=torch.float64)
    rates = system.reaction_rates(c)
    expected = torch.tensor([3.0 * 0.4**2, 0.5 * 0.3], dtype=torch.float64)
    assert torch.allclose(rates, expected)
    rhs = system.time_derivative(c)
    r_f, r_r = 3.0 * 0.16, 0.5 * 0.3
    assert torch.allclose(
        rhs, torch.tensor([-2.0 * r_f + 2.0 * r_r, r_f - r_r], dtype=torch.float64)
    )
    # Closed system: element total is conserved by the vector field.
    assert torch.allclose(system.element_matrix @ rhs, torch.zeros(1, dtype=rhs.dtype))


def test_rates_batch_dtype_override_and_zero_concentration() -> None:
    system = _chain()
    c = torch.zeros(4, 5, 3, dtype=torch.float32)
    rates = system.reaction_rates(c)
    assert rates.shape == (4, 5, 3)
    assert rates.dtype == torch.float32
    # Zeroth-order feed keeps its rate at c = 0 (0^0 = 1).
    assert torch.allclose(rates[..., 0], torch.full((4, 5), 0.2))
    assert torch.all(rates[..., 1:] == 0)

    override = system.reaction_rates(
        torch.ones(3, dtype=torch.float64), rate_constants=torch.tensor([1.0, 2.0, 3.0])
    )
    assert override.tolist() == [1.0, 2.0, 3.0]


def test_time_derivative_gradient_is_finite_at_zero() -> None:
    system = _chain()
    c = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    system.time_derivative(c).sum().backward()
    assert c.grad is not None
    assert bool(torch.isfinite(c.grad).all())


def test_arrhenius_rate_constant_and_temperature_copy() -> None:
    params = ArrheniusParams(2.0e3, 1.0e4, temperature_exponent=0.5)
    temperature = 350.0
    expected = (
        2.0e3
        * temperature**0.5
        * math.exp(-1.0e4 / (GAS_CONSTANT_J_PER_MOL_K * temperature))
    )
    assert math.isclose(float(params.rate_constant(temperature)), expected)
    batched = params.rate_constant(torch.tensor([300.0, 400.0]))
    assert batched.shape == (2,)
    assert float(batched[1]) > float(batched[0])
    assert params.rate_constant(torch.tensor([300])).dtype == torch.float64

    system = _dimerization()
    hot = system.at_temperature(temperature)
    assert math.isclose(float(hot.rate_constants[1]), expected)
    assert hot.species == system.species
    assert hot is not system
    # Original stays unchanged (immutable value semantics).
    assert system.rate_constants.tolist() == [3.0, 0.5]


def test_element_balance_report() -> None:
    closed = check_element_balance(_dimerization())
    assert closed.balanced
    assert closed.max_abs_residual == 0.0
    assert closed.residual.shape == (1, 2)
    assert closed.elements == ("X",)

    open_system = check_element_balance(_chain())
    assert not open_system.balanced
    assert open_system.unbalanced_reactions == (0,)
    assert open_system.max_abs_residual == 1.0

    with pytest.raises(ValueError, match="atol must be >= 0"):
        check_element_balance(_chain(), atol=-1.0)
    no_elements = ReactionSystem(
        species=("A", "B"),
        reactant_stoichiometry=torch.tensor([[1.0, 0.0]]),
        product_stoichiometry=torch.tensor([[0.0, 1.0]]),
        rate_constants=torch.tensor([1.0]),
    )
    with pytest.raises(ValueError, match="no element_matrix"):
        check_element_balance(no_elements)
    with pytest.raises(ValueError, match="no Arrhenius parameters"):
        no_elements.at_temperature(300.0)


def test_round_trip_through_dict() -> None:
    system = _dimerization()
    payload = system.to_dict()
    rebuilt = ReactionSystem.from_dict(payload)
    assert rebuilt.to_dict() == payload
    assert rebuilt.arrhenius == system.arrhenius
    minimal = ReactionSystem.from_dict(
        {
            "species": ["A", "B"],
            "reactant_stoichiometry": [[1, 0]],
            "product_stoichiometry": [[0, 1]],
            "rate_constants": [1.0],
        }
    )
    assert minimal.element_matrix is None
    assert minimal.name == "reaction_system"
    assert set(minimal.to_dict()) == {
        "name",
        "species",
        "reactant_stoichiometry",
        "product_stoichiometry",
        "rate_constants",
    }


def test_inputs_are_copied_not_aliased() -> None:
    alpha = torch.tensor([[1.0, 0.0]])
    k = torch.tensor([1.0])
    system = ReactionSystem(
        species=("A", "B"),
        reactant_stoichiometry=alpha,
        product_stoichiometry=torch.tensor([[0.0, 1.0]]),
        rate_constants=k,
    )
    alpha[0, 0] = 5.0
    k[0] = 9.0
    assert system.reactant_stoichiometry[0, 0] == 1.0
    assert system.rate_constants[0] == 1.0
    assert system.rate_constants.dtype == torch.float64


def _kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "species": ("A", "B"),
        "reactant_stoichiometry": torch.tensor([[1.0, 0.0]]),
        "product_stoichiometry": torch.tensor([[0.0, 1.0]]),
        "rate_constants": torch.tensor([1.0]),
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize(
    ("overrides", "error", "match"),
    [
        ({"species": ()}, ValueError, "at least one name"),
        ({"species": "AB"}, TypeError, "not a single string"),
        ({"species": ("A", "A")}, ValueError, "species must be unique"),
        ({"species": ("A", "")}, ValueError, "non-empty strings"),
        ({"reactant_stoichiometry": [[1, 0]]}, TypeError, "must be a Tensor"),
        ({"reactant_stoichiometry": torch.ones(2)}, ValueError, r"shape \(rows, 2\)"),
        (
            {"reactant_stoichiometry": torch.tensor([[0.5, 0.0]])},
            ValueError,
            "non-negative integers",
        ),
        (
            {"reactant_stoichiometry": torch.tensor([[-1.0, 0.0]])},
            ValueError,
            "non-negative integers",
        ),
        (
            {"reactant_stoichiometry": torch.tensor([[float("inf"), 0.0]])},
            ValueError,
            "must be finite",
        ),
        (
            {"product_stoichiometry": torch.tensor([[0.0, 1.0], [1.0, 0.0]])},
            ValueError,
            "same shape",
        ),
        (
            {
                "reactant_stoichiometry": torch.zeros(0, 2),
                "product_stoichiometry": torch.zeros(0, 2),
                "rate_constants": torch.zeros(0),
            },
            ValueError,
            "at least one reaction",
        ),
        (
            {"product_stoichiometry": torch.tensor([[1.0, 0.0]])},
            ValueError,
            "identical reactants and products",
        ),
        ({"rate_constants": [1.0]}, TypeError, "rate_constants must be a Tensor"),
        ({"rate_constants": torch.ones(2)}, ValueError, r"shape \(1,\)"),
        ({"rate_constants": torch.tensor([-1.0])}, ValueError, ">= 0"),
        ({"rate_constants": torch.tensor([float("nan")])}, ValueError, ">= 0"),
        ({"element_matrix": torch.ones(1, 2)}, ValueError, "given together"),
        (
            {"element_matrix": torch.ones(2, 2), "elements": ("X",)},
            ValueError,
            "2 rows but 1 elements",
        ),
        ({"reaction_names": ("a", "b")}, ValueError, "2 entries for 1 reactions"),
        ({"arrhenius": ()}, ValueError, "0 entries for 1 reactions"),
        ({"arrhenius": (1.0,)}, TypeError, "must be ArrheniusParams"),
        ({"name": ""}, ValueError, "non-empty string"),
    ],
)
def test_reaction_system_validation(
    overrides: dict[str, object], error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        ReactionSystem(**_kwargs(**overrides))  # type: ignore[arg-type]


def test_reaction_names_may_repeat() -> None:
    system = ReactionSystem(
        **_kwargs(  # type: ignore[arg-type]
            reactant_stoichiometry=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
            product_stoichiometry=torch.tensor([[0.0, 1.0], [0.0, 2.0]]),
            rate_constants=torch.tensor([1.0, 2.0]),
            reaction_names=("decay", "decay"),
        )
    )
    assert system.reaction_names == ("decay", "decay")


def test_concentration_validation() -> None:
    system = _chain()
    with pytest.raises(TypeError, match="must be a Tensor"):
        system.reaction_rates([1.0, 2.0, 3.0])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="floating point"):
        system.reaction_rates(torch.ones(3, dtype=torch.long))
    with pytest.raises(ValueError, match=r"shape \(\.\.\., 3\)"):
        system.reaction_rates(torch.ones(2))
    with pytest.raises(ValueError, match=r"shape \(\.\.\., 3\)"):
        system.reaction_rates(torch.tensor(1.0))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"pre_exponential": 0.0, "activation_energy": 1.0}, "pre_exponential must"),
        ({"pre_exponential": "x", "activation_energy": 1.0}, "real number"),
        ({"pre_exponential": 1.0, "activation_energy": math.inf}, "must be finite"),
        (
            {
                "pre_exponential": 1.0,
                "activation_energy": 1.0,
                "temperature_exponent": math.nan,
            },
            "must be finite",
        ),
    ],
)
def test_arrhenius_validation(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ArrheniusParams(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "temperature",
    [0.0, -10.0, math.inf, torch.tensor([300.0, 0.0]), torch.tensor([math.nan])],
)
def test_arrhenius_rejects_bad_temperature(temperature: object) -> None:
    with pytest.raises(ValueError, match="temperature_k must be"):
        ArrheniusParams(1.0, 1.0).rate_constant(temperature)  # type: ignore[arg-type]
