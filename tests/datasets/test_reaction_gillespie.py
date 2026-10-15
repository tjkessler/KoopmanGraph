"""Tests for :mod:`koopman_graph.datasets.reaction.gillespie`."""

from __future__ import annotations

import math

import pytest
import torch

from koopman_graph.datasets.reaction import (
    ReactionSystem,
    bin_to_grid,
    first_order_chain,
    gillespie_ssa,
    integrate_rk4,
    propensities,
    reversible_dimerization,
)


def test_propensities_use_falling_factorials_and_volume() -> None:
    system = reversible_dimerization()  # 2 A -> A2 (3.0), A2 -> 2 A (0.5)
    a = propensities(system, torch.tensor([10, 4]), volume=2.0)
    # 2 A: k V^{-1} n (n - 1); A2: k n.
    assert torch.allclose(
        a, torch.tensor([3.0 / 2.0 * 10 * 9, 0.5 * 4], dtype=torch.float64)
    )
    one = propensities(system, torch.tensor([1, 0]), volume=1.0)
    assert one.tolist() == [0.0, 0.0]  # one A molecule cannot dimerize
    feed = ReactionSystem(
        species=("A",),
        reactant_stoichiometry=torch.zeros(1, 1),
        product_stoichiometry=torch.ones(1, 1),
        rate_constants=torch.tensor([2.0]),
    )
    # Zeroth order scales with V.
    assert propensities(feed, torch.tensor([0]), volume=5.0).tolist() == [10.0]


def test_seed_reproduces_path_exactly() -> None:
    system = reversible_dimerization()
    kwargs = {"volume": 100.0, "seed": 11}
    a = gillespie_ssa(system, torch.tensor([100, 0]), 3.0, **kwargs)
    b = gillespie_ssa(system, torch.tensor([100, 0]), 3.0, **kwargs)
    assert torch.equal(a.event_times, b.event_times)
    assert torch.equal(a.counts, b.counts)
    c = gillespie_ssa(system, torch.tensor([100, 0]), 3.0, volume=100.0, seed=12)
    assert not torch.equal(a.counts, c.counts) or not torch.equal(
        a.event_times, c.event_times
    )
    assert a.seed == 11
    assert a.counts.dtype == torch.long
    assert float(a.event_times[0]) == 0.0
    assert bool((a.event_times[1:] > a.event_times[:-1]).all())
    assert float(a.event_times[-1]) <= 3.0
    assert not a.truncated


def test_path_conserves_monomer_units() -> None:
    system = reversible_dimerization()
    path = gillespie_ssa(system, torch.tensor([60, 20]), 5.0, volume=50.0, seed=0)
    units = path.counts[:, 0] + 2 * path.counts[:, 1]
    assert bool((units == 100).all())
    assert bool((path.counts >= 0).all())


def test_large_volume_approaches_ode() -> None:
    system = reversible_dimerization()
    grid = torch.linspace(0.0, 4.0, 9, dtype=torch.float64)
    ode = integrate_rk4(
        system, torch.tensor([1.0, 0.0], dtype=torch.float64), grid, substeps=50
    )

    def mean_error(volume: int) -> float:
        """Mean max-abs SSA-vs-ODE error over three fixed seeds.

        Parameters
        ----------
        volume
            See signature.

        Returns
        -------
            See signature."""
        errors = []
        for seed in range(3):
            path = gillespie_ssa(
                system, torch.tensor([volume, 0]), 4.0, volume=float(volume), seed=seed
            )
            errors.append(float((path.concentrations_on_grid(grid) - ode).abs().max()))
        return sum(errors) / len(errors)

    small, large = mean_error(40), mean_error(2000)
    assert large < 0.05
    assert large < small / 2


def test_absorbing_state_stops_early() -> None:
    system = first_order_chain(2)  # S0 -> S1 only
    path = gillespie_ssa(system, torch.tensor([5, 0]), 1.0e6, seed=0)
    assert path.num_events == 5
    assert path.counts[-1].tolist() == [0, 5]
    held = path.counts_on_grid(torch.tensor([0.0, 1.0e6]))
    assert held.tolist() == [[5, 0], [0, 5]]


def test_max_events_truncates() -> None:
    system = reversible_dimerization()
    path = gillespie_ssa(
        system, torch.tensor([200, 0]), 100.0, volume=200.0, seed=0, max_events=10
    )
    assert path.truncated
    assert path.num_events == 10
    with pytest.raises(ValueError, match="truncated"):
        path.counts_on_grid(torch.tensor([0.0, 100.0]))
    assert path.counts_on_grid(torch.tensor([0.0])).tolist() == [[200, 0]]


def test_bin_to_grid_holds_last_value() -> None:
    times = torch.tensor([0.0, 1.0, 2.5], dtype=torch.float64)
    counts = torch.tensor([[3, 0], [2, 1], [1, 2]])
    grid = torch.tensor([0.0, 0.5, 1.0, 2.0, 2.5, 3.0])
    held = bin_to_grid(times, counts, grid)
    assert held.tolist() == [[3, 0], [3, 0], [2, 1], [2, 1], [1, 2], [1, 2]]
    assert bin_to_grid(times, counts, torch.zeros(0)).shape == (0, 2)
    with pytest.raises(ValueError, match="past t_end"):
        bin_to_grid(times, counts, grid, t_end=2.9)
    with pytest.raises(ValueError, match="before the first event"):
        bin_to_grid(times, counts, torch.tensor([-1.0]))
    with pytest.raises(ValueError, match="non-decreasing"):
        bin_to_grid(times, counts, torch.tensor([1.0, 0.5]))
    with pytest.raises(ValueError, match="1-D and counts 2-D"):
        bin_to_grid(times, counts[:, 0], grid)
    with pytest.raises(ValueError, match="same non-zero length"):
        bin_to_grid(times[:2], counts, grid)


@pytest.mark.parametrize(
    ("initial", "kwargs", "error", "match"),
    [
        ([5, 0], {}, TypeError, "must be a Tensor"),
        (torch.tensor([5]), {}, ValueError, r"shape \(2,\)"),
        (torch.tensor([5.5, 0.0]), {}, ValueError, "non-negative integers"),
        (torch.tensor([-1, 0]), {}, ValueError, "non-negative integers"),
        (torch.tensor([5, 0]), {"t_end": 0.0}, ValueError, "t_end must be"),
        (torch.tensor([5, 0]), {"volume": math.inf}, ValueError, "volume must be"),
        (torch.tensor([5, 0]), {"seed": -1}, ValueError, "seed must be"),
        (torch.tensor([5, 0]), {"max_events": 0}, ValueError, "max_events must be"),
    ],
)
def test_ssa_validation(
    initial: object, kwargs: dict[str, object], error: type[Exception], match: str
) -> None:
    t_end = kwargs.pop("t_end", 1.0)
    with pytest.raises(error, match=match):
        gillespie_ssa(reversible_dimerization(), initial, t_end, **kwargs)  # type: ignore[arg-type]


def test_ssa_requires_reaction_system() -> None:
    with pytest.raises(TypeError, match="must be a ReactionSystem"):
        gillespie_ssa("dimer", torch.tensor([1, 0]), 1.0)  # type: ignore[arg-type]
