"""Seeded stochastic simulation (Gillespie direct method) and grid binning.

Molecule counts ``n`` live in a volume ``V`` (concentration ``c = n / V``).
The propensity of reaction ``j`` matches the deterministic mass-action rate
``r_j = k_j ∏ c_s^{α_js}`` used by
:meth:`~koopman_graph.datasets.reaction.ReactionSystem.reaction_rates`:

.. math::

    a_j(n) = k_j V^{1 - o_j} \\prod_s n_s (n_s - 1) \\cdots (n_s - α_{js} + 1),

where ``o_j = Σ_s α_js`` is the reaction order. The ``n(n-1)`` falling
factorial makes ``2 A → …`` need two distinct molecules. As ``V`` grows
with ``n / V`` fixed, ``n(t) / V`` approaches the ODE solution.

:func:`gillespie_ssa` uses one seeded ``torch.Generator``, so a seed
reproduces the whole event path. :func:`bin_to_grid` samples the
piecewise-constant path on a regular grid, holding the last value.

Pure PyTorch; importing this module does not import scipy or cantera.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from koopman_graph.datasets.reaction.system import ReactionSystem

DEFAULT_MAX_EVENTS = 1_000_000
"""Default cap on SSA events per trajectory."""


@dataclass(frozen=True)
class SSATrajectory:
    """Event path of one stochastic simulation.

    Attributes
    ----------
    event_times : Tensor
        ``float64`` times ``(num_events + 1,)``; entry 0 is ``0.0``.
    counts : Tensor
        ``long`` molecule counts ``(num_events + 1, num_species)`` right
        after each event (row 0 is the initial state).
    t_end : float
        Simulated horizon; the state after the last event holds to it.
    volume : float
        System volume ``V`` relating counts to concentrations.
    seed : int
        Generator seed.
    truncated : bool
        ``True`` when ``max_events`` stopped the run before ``t_end``.
    """

    event_times: Tensor
    counts: Tensor
    t_end: float
    volume: float
    seed: int
    truncated: bool

    @property
    def num_events(self) -> int:
        """Return the number of reaction events.

        Returns
        -------
        int
            ``len(event_times) - 1``.
        """
        return int(self.event_times.shape[0]) - 1

    def counts_on_grid(self, grid: Tensor) -> Tensor:
        """Return counts held at each grid time; see :func:`bin_to_grid`.

        Parameters
        ----------
        grid : Tensor
            Non-decreasing 1-D times in ``[0, t_end]``.

        Returns
        -------
        Tensor
            ``long`` counts ``(len(grid), num_species)``.
        """
        last_event = float(self.event_times[-1])
        if self.truncated and grid.numel() and float(grid.max()) > last_event:
            msg = "grid extends past the last event of a truncated trajectory"
            raise ValueError(msg)
        return bin_to_grid(self.event_times, self.counts, grid, t_end=self.t_end)

    def concentrations_on_grid(self, grid: Tensor) -> Tensor:
        """Return concentrations ``counts / volume`` on ``grid``.

        Parameters
        ----------
        grid : Tensor
            Non-decreasing 1-D times in ``[0, t_end]``.

        Returns
        -------
        Tensor
            ``float64`` concentrations ``(len(grid), num_species)``.
        """
        return self.counts_on_grid(grid).to(torch.float64) / self.volume


def propensities(system: ReactionSystem, counts: Tensor, volume: float) -> Tensor:
    """Return mass-action SSA propensities ``a_j(n)``.

    Parameters
    ----------
    system : ReactionSystem
        Reaction network.
    counts : Tensor
        Non-negative molecule counts ``(num_species,)``.
    volume : float
        System volume ``V`` (finite, > 0).

    Returns
    -------
    Tensor
        ``float64`` propensities ``(num_reactions,)``. Zero whenever a
        reactant has fewer molecules than its coefficient.
    """
    n = counts.to(torch.float64)
    alpha = system.reactant_stoichiometry
    falling = torch.ones_like(alpha)
    for m in range(int(alpha.max().item()) if alpha.numel() else 0):
        # Multiply in (n - m) wherever the coefficient exceeds m.
        falling = falling * torch.where(alpha > m, (n - m).clamp(min=0), 1.0)
    scale = volume ** (1.0 - system.reaction_orders)
    return system.rate_constants * scale * falling.prod(dim=-1)


def gillespie_ssa(
    system: ReactionSystem,
    initial_counts: Tensor,
    t_end: float,
    *,
    volume: float = 1.0,
    seed: int = 0,
    max_events: int = DEFAULT_MAX_EVENTS,
) -> SSATrajectory:
    """Simulate one exact stochastic trajectory with Gillespie's direct method.

    Parameters
    ----------
    system : ReactionSystem
        Reaction network.
    initial_counts : Tensor
        Non-negative integer counts ``(num_species,)`` (any integer or
        integer-valued floating dtype).
    t_end : float
        Horizon (finite, > 0).
    volume : float, optional
        System volume ``V``. Default is ``1.0``.
    seed : int, optional
        Generator seed. Default is ``0``.
    max_events : int, optional
        Event cap; the run stops early with ``truncated=True`` when hit.
        Default is :data:`DEFAULT_MAX_EVENTS`.

    Returns
    -------
    SSATrajectory
        Event times and counts. The run ends at ``t_end``, when all
        propensities vanish (absorbing state), or at ``max_events``.

    Raises
    ------
    ValueError
        If arguments are invalid.
    """
    if not isinstance(system, ReactionSystem):
        msg = f"system must be a ReactionSystem, got {type(system).__name__}"
        raise TypeError(msg)
    n = _initial_counts(system, initial_counts)
    horizon = _positive("t_end", t_end)
    v = _positive("volume", volume)
    for name, value, low in (("seed", seed, 0), ("max_events", max_events, 1)):
        if isinstance(value, bool) or not isinstance(value, int) or value < low:
            msg = f"{name} must be an int >= {low}, got {value!r}"
            raise ValueError(msg)

    generator = torch.Generator().manual_seed(seed)
    net_change = system.stoichiometric_matrix.T.to(torch.long)  # (R, S)
    times = [0.0]
    states = [n.clone()]
    t = 0.0
    truncated = False
    while True:
        a = propensities(system, n, v)
        total = float(a.sum())
        if total <= 0.0:
            break
        u = torch.rand(2, generator=generator, dtype=torch.float64)
        t += -math.log(1.0 - float(u[0])) / total
        if t > horizon:
            break
        if len(times) - 1 >= max_events:
            truncated = True
            break
        # First reaction whose cumulative propensity exceeds u * a0; right=True
        # never selects a zero-propensity reaction on a tie.
        threshold = torch.tensor(float(u[1]) * total, dtype=torch.float64)
        j = int(torch.searchsorted(torch.cumsum(a, 0), threshold, right=True))
        j = min(j, system.num_reactions - 1)
        n = n + net_change[j]
        times.append(t)
        states.append(n.clone())
    return SSATrajectory(
        event_times=torch.tensor(times, dtype=torch.float64),
        counts=torch.stack(states),
        t_end=horizon,
        volume=v,
        seed=seed,
        truncated=truncated,
    )


def bin_to_grid(
    event_times: Tensor,
    counts: Tensor,
    grid: Tensor,
    *,
    t_end: float | None = None,
) -> Tensor:
    """Sample a piecewise-constant event path at grid times (last-value hold).

    Parameters
    ----------
    event_times : Tensor
        Non-decreasing event times ``(K,)`` starting at the path's origin.
    counts : Tensor
        State after each event ``(K, num_species)``.
    grid : Tensor
        Non-decreasing 1-D query times ``>= event_times[0]``.
    t_end : float or None, optional
        Upper bound for ``grid`` (the path's horizon). Default allows any
        time after the last event.

    Returns
    -------
    Tensor
        ``counts[i]`` for the last event ``i`` with ``event_times[i] <= t``,
        one row per grid time, in the dtype of ``counts``.

    Raises
    ------
    ValueError
        If shapes disagree or the grid leaves ``[event_times[0], t_end]``.
    """
    if event_times.ndim != 1 or counts.ndim != 2:
        msg = "event_times must be 1-D and counts 2-D"
        raise ValueError(msg)
    if int(event_times.shape[0]) != int(counts.shape[0]) or event_times.numel() == 0:
        msg = (
            "event_times and counts must have the same non-zero length, got "
            f"{int(event_times.shape[0])} and {int(counts.shape[0])}"
        )
        raise ValueError(msg)
    if grid.ndim != 1 or (grid.numel() > 1 and bool((grid[1:] < grid[:-1]).any())):
        msg = "grid must be a non-decreasing 1-D Tensor"
        raise ValueError(msg)
    g = grid.to(torch.float64)
    if g.numel() and float(g.min()) < float(event_times[0]):
        msg = "grid starts before the first event time"
        raise ValueError(msg)
    if t_end is not None and g.numel() and float(g.max()) > float(t_end):
        msg = f"grid extends past t_end={t_end}"
        raise ValueError(msg)
    index = torch.searchsorted(event_times.to(torch.float64), g, right=True) - 1
    return counts[index]


def _initial_counts(system: ReactionSystem, initial_counts: Tensor) -> Tensor:
    """Validate non-negative integer counts and return them as ``long``.

    Parameters
    ----------
    system
        See signature.
    initial_counts
        See signature.

    Returns
    -------
        See signature."""
    if not isinstance(initial_counts, Tensor):
        msg = f"initial_counts must be a Tensor, got {type(initial_counts).__name__}"
        raise TypeError(msg)
    if initial_counts.shape != (system.num_species,):
        msg = (
            f"initial_counts must have shape ({system.num_species},), got "
            f"{tuple(initial_counts.shape)}"
        )
        raise ValueError(msg)
    as_float = initial_counts.to(torch.float64)
    if (
        not bool(torch.isfinite(as_float).all())
        or bool((as_float < 0).any())
        or not bool((as_float == as_float.round()).all())
    ):
        msg = "initial_counts must be non-negative integers"
        raise ValueError(msg)
    return as_float.to(torch.long)


def _positive(name: str, value: float) -> float:
    """Return ``value`` as a finite positive float or raise ``ValueError``.

    Parameters
    ----------
    name
        See signature.
    value
        See signature.

    Returns
    -------
        See signature."""
    x = float(value)
    if not math.isfinite(x) or x <= 0.0:
        msg = f"{name} must be finite and > 0, got {value!r}"
        raise ValueError(msg)
    return x
