"""Tests for :mod:`koopman_graph.datasets.reaction.transform`."""

from __future__ import annotations

import json
import math

import pytest
import torch

from koopman_graph.datasets.reaction import (
    DEFAULT_LOG_FLOOR,
    LogConcentrationTransform,
    ReactionSystem,
    element_balance_violation,
)


def _closed_system() -> ReactionSystem:
    """``A + B <-> AB`` with elements ``{a, b}``."""
    return ReactionSystem(
        species=("A", "B", "AB"),
        reactant_stoichiometry=torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        product_stoichiometry=torch.tensor([[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]]),
        rate_constants=torch.tensor([2.0, 0.1]),
        element_matrix=torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]),
        elements=("a", "b"),
    )


def test_log_transform_round_trip_above_floor() -> None:
    transform = LogConcentrationTransform()
    assert transform.floor == DEFAULT_LOG_FLOOR
    c = torch.tensor([[1e-6, 0.5, 3.0]], dtype=torch.float64)
    z = transform.forward(c)
    assert torch.allclose(z, torch.log(c))
    assert torch.allclose(transform.inverse(z), c)
    assert z.dtype == torch.float64


def test_log_transform_floors_zero_and_negative_values() -> None:
    transform = LogConcentrationTransform(floor=1e-8)
    c = torch.tensor([0.0, -1e-3, 1e-9, 1e-2], dtype=torch.float32)
    z = transform.forward(c)
    assert bool(torch.isfinite(z).all())
    assert torch.allclose(z[:3], torch.full((3,), math.log(1e-8)))
    assert transform.floored_mask(c).tolist() == [True, True, True, False]
    restored = transform.inverse(z)
    assert torch.allclose(restored, torch.clamp(c, min=1e-8))
    # NaN is not silently floored.
    assert bool(torch.isnan(transform.forward(torch.tensor([math.nan]))).all())


def test_log_transform_serialization() -> None:
    transform = LogConcentrationTransform(floor=1e-10)
    payload = json.loads(json.dumps(transform.to_dict()))
    assert payload == {"kind": "log_concentration", "floor": 1e-10}
    assert LogConcentrationTransform.from_dict(payload) == transform
    with pytest.raises(ValueError, match="expected kind"):
        LogConcentrationTransform.from_dict({"kind": "identity", "floor": 1.0})
    with pytest.raises(KeyError):
        LogConcentrationTransform.from_dict({"kind": "log_concentration"})


@pytest.mark.parametrize("floor", [0.0, -1.0, math.inf, math.nan, "x"])
def test_log_transform_rejects_bad_floor(floor: object) -> None:
    with pytest.raises(ValueError, match="floor must be"):
        LogConcentrationTransform(floor=floor)  # type: ignore[arg-type]


def test_log_transform_input_validation() -> None:
    transform = LogConcentrationTransform()
    with pytest.raises(TypeError, match="must be a Tensor"):
        transform.forward([1.0])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="floating point"):
        transform.inverse(torch.ones(2, dtype=torch.long))
    with pytest.raises(TypeError, match="floating point"):
        transform.floored_mask(torch.ones(2, dtype=torch.int32))


def test_element_balance_violation_zero_along_reaction_extent() -> None:
    system = _closed_system()
    c0 = torch.tensor([1.0, 0.8, 0.0], dtype=torch.float64)
    # Moving along the reaction coordinate conserves both elements.
    extents = torch.linspace(0.0, 0.7, 5, dtype=torch.float64).unsqueeze(-1)
    trajectory = c0 + extents * system.stoichiometric_matrix[:, 0]
    violation = element_balance_violation(trajectory, c0, system)
    assert violation.shape == (5, 2)
    assert torch.allclose(violation, torch.zeros_like(violation))


def test_element_balance_violation_signed_drift_and_matrix_input() -> None:
    system = _closed_system()
    c0 = torch.tensor([1.0, 0.8, 0.0])
    drifted = torch.tensor([1.1, 0.8, 0.05])
    expected = torch.tensor([0.15, 0.05])
    assert torch.allclose(element_balance_violation(drifted, c0, system), expected)
    assert system.element_matrix is not None
    from_matrix = element_balance_violation(drifted, c0, system.element_matrix)
    assert torch.allclose(from_matrix, expected)
    assert from_matrix.dtype == torch.float32


def test_element_balance_violation_validation() -> None:
    system = _closed_system()
    c = torch.ones(3)
    with pytest.raises(TypeError, match="ReactionSystem or Tensor"):
        element_balance_violation(c, c, [[1.0, 0.0, 1.0]])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="reference must be a Tensor"):
        element_balance_violation(c, [1.0, 1.0, 1.0], system)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="trailing dimension 3"):
        element_balance_violation(torch.ones(2), c, system)
    with pytest.raises(ValueError, match="trailing dimension 3"):
        element_balance_violation(c, torch.tensor(1.0), system)
    with pytest.raises(ValueError, match=r"\(num_elements, num_species\)"):
        element_balance_violation(c, c, torch.ones(3))
    no_elements = ReactionSystem(
        species=("A", "B"),
        reactant_stoichiometry=torch.tensor([[1.0, 0.0]]),
        product_stoichiometry=torch.tensor([[0.0, 1.0]]),
        rate_constants=torch.tensor([1.0]),
    )
    with pytest.raises(ValueError, match="no element_matrix"):
        element_balance_violation(torch.ones(2), torch.ones(2), no_elements)
