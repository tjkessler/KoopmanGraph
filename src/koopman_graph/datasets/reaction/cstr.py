"""Van de Vusse continuous stirred-tank reactor with feed and temperature inputs.

The Van de Vusse scheme ``A → B → C`` and ``2 A → D`` is a standard
benchmark for nonlinear process control: the intermediate ``B`` is the
product, and its yield depends non-monotonically on dilution rate and
temperature. This module uses the kinetics of Klatt and Engell (1998),
with time in hours and concentrations in mol/L:

=============  ==================  ===================
Reaction       ``k0``              ``E_a / R`` (K)
=============  ==================  ===================
``A → B``      1.287e12 1/h        9758.3
``B → C``      1.287e12 1/h        9758.3
``2 A → D``    9.043e9 L/(mol h)   8560.0
=============  ==================  ===================

The literature writes the third rate as ``r3 = k3 c_A²`` with
``dc_A/dt ⊃ -k3 c_A²``. Mass action on ``2 A → D`` consumes two ``A`` per
event, so the mass-action pre-exponential here is ``k0 / 2``, which
reproduces the literature balance exactly.

The reactor temperature is treated as a direct input (no energy balance).
The controlled field is

.. math::

    \\frac{dc}{dt} = N r(c; T) + D (c_\\text{feed} - c),

with dilution rate ``D = F / V`` (1/h) and feed ``c_feed = (c_A0, 0, 0,
0)``. At ``D = 14.19``, ``T = 387.35 K`` (114.2 °C) and ``c_A0 = 5.10`` the
steady state is ``c_A ≈ 2.14``, ``c_B ≈ 1.09``, the nominal operating point
of the CSTR control benchmark.

References: K.-U. Klatt and S. Engell, "Gain-scheduling trajectory control
of a continuous stirred tank reactor", *Computers & Chemical Engineering*
22(4-5), 491-502, 1998 (kinetics); H. Chen, A. Kremling and F. Allgöwer,
"Nonlinear predictive control of a benchmark CSTR", *Proc. 3rd European
Control Conference*, 1995 (operating point).

Pure PyTorch; importing this module does not import scipy or cantera.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from koopman_graph.datasets.reaction.integrate import mass_action_jacobian
from koopman_graph.datasets.reaction.system import (
    GAS_CONSTANT_J_PER_MOL_K,
    ArrheniusParams,
    ReactionSystem,
)

VAN_DE_VUSSE_FEED_A = 5.10
"""Nominal feed concentration of ``A`` in mol/L (Klatt and Engell, 1998)."""

VAN_DE_VUSSE_NOMINAL_DILUTION = 14.19
"""Nominal dilution rate ``F / V`` in 1/h."""

VAN_DE_VUSSE_NOMINAL_TEMPERATURE_K = 387.35
"""Nominal reactor temperature in K (114.2 °C)."""

_K0 = (1.287e12, 1.287e12, 9.043e9 / 2.0)
_EA_OVER_R = (9758.3, 9758.3, 8560.0)


@dataclass(frozen=True, eq=False)
class VanDeVusseCSTR:
    """Van de Vusse CSTR: reaction system plus feed composition.

    Attributes
    ----------
    system : ReactionSystem
        ``A → B``, ``B → C``, ``2 A → D`` over species ``(A, B, C, D)`` with
        Arrhenius parameters; its ``rate_constants`` are those at the
        nominal temperature.
    feed : Tensor
        ``float64`` feed concentrations ``(4,)`` in mol/L.
    """

    system: ReactionSystem
    feed: Tensor

    def rhs(
        self,
        concentrations: Tensor,
        *,
        temperature_k: float,
        dilution_rate: float,
    ) -> Tensor:
        """Evaluate the controlled field ``N r(c; T) + D (c_feed - c)``.

        Parameters
        ----------
        concentrations : Tensor
            Floating tensor ``(..., 4)`` in mol/L.
        temperature_k : float
            Reactor temperature in K (finite, > 0).
        dilution_rate : float
            ``F / V`` in 1/h (finite, >= 0).

        Returns
        -------
        Tensor
            ``dc/dt`` in mol/(L h) with the shape and dtype of
            ``concentrations``.
        """
        k = self.system.rate_constants_at(temperature_k)
        d = _dilution(dilution_rate)
        reaction = self.system.time_derivative(concentrations, rate_constants=k)
        feed = self.feed.to(dtype=reaction.dtype, device=reaction.device)
        return reaction + d * (feed - concentrations)

    def simulate(
        self,
        initial: Tensor,
        times: Tensor,
        *,
        temperature_k: float | Tensor,
        dilution_rate: float | Tensor,
        substeps: int = 50,
    ) -> Tensor:
        """Integrate with RK4 under piecewise-constant inputs.

        Parameters
        ----------
        initial : Tensor
            Floating initial concentrations ``(..., 4)``.
        times : Tensor
            Strictly increasing 1-D output times in hours.
        temperature_k : float or Tensor
            Constant temperature, or one value per interval with shape
            ``(len(times) - 1,)`` held over ``[times[i], times[i + 1])``.
        dilution_rate : float or Tensor
            Constant dilution rate, or one value per interval.
        substeps : int, optional
            RK4 steps per output interval. Default is ``50``.

        Returns
        -------
        Tensor
            Concentrations ``(len(times), ..., 4)``.

        Raises
        ------
        ValueError
            If schedules have the wrong length or inputs are invalid.
        """
        if isinstance(substeps, bool) or not isinstance(substeps, int) or substeps < 1:
            msg = f"substeps must be a positive int, got {substeps!r}"
            raise ValueError(msg)
        if not isinstance(times, Tensor) or times.ndim != 1 or times.numel() < 1:
            msg = "times must be a non-empty 1-D Tensor"
            raise ValueError(msg)
        t = times.to(torch.float64)
        if times.numel() > 1 and not bool((t[1:] > t[:-1]).all()):
            msg = "times must be strictly increasing"
            raise ValueError(msg)
        num_intervals = int(times.numel()) - 1
        temps = _schedule(temperature_k, num_intervals, "temperature_k")
        dilutions = _schedule(dilution_rate, num_intervals, "dilution_rate")
        c = initial
        states = [c]
        for i in range(num_intervals):
            h = float(t[i + 1] - t[i]) / substeps
            kwargs = {"temperature_k": temps[i], "dilution_rate": dilutions[i]}
            for _ in range(substeps):
                k1 = self.rhs(c, **kwargs)
                k2 = self.rhs(c + 0.5 * h * k1, **kwargs)
                k3 = self.rhs(c + 0.5 * h * k2, **kwargs)
                k4 = self.rhs(c + h * k3, **kwargs)
                c = c + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            states.append(c)
        return torch.stack(states)

    def steady_state(
        self,
        *,
        temperature_k: float,
        dilution_rate: float,
        guess: Tensor | None = None,
        tol: float = 1e-12,
        max_iter: int = 100,
    ) -> Tensor:
        """Solve ``rhs(c) = 0`` with damped Newton (non-negative iterates).

        Parameters
        ----------
        temperature_k : float
            Reactor temperature in K.
        dilution_rate : float
            ``F / V`` in 1/h (> 0 for a unique steady state).
        guess : Tensor or None, optional
            Starting point ``(4,)``. Default is the feed.
        tol : float, optional
            Relative step tolerance. Default is ``1e-12``.
        max_iter : int, optional
            Iteration cap. Default is ``100``.

        Returns
        -------
        Tensor
            ``float64`` steady-state concentrations ``(4,)``.

        Raises
        ------
        RuntimeError
            If Newton does not converge.
        """
        d = _dilution(dilution_rate)
        at_t = self.system.at_temperature(temperature_k)
        c = (self.feed if guess is None else guess).to(torch.float64).clone()
        eye = torch.eye(4, dtype=torch.float64)

        def residual(state: Tensor) -> Tensor:
            """Controlled field at ``temperature_k`` / ``dilution_rate``.

            Parameters
            ----------
            state
                See signature.

            Returns
            -------
                See signature."""
            return at_t.time_derivative(state) + d * (self.feed - state)

        f = residual(c)
        for _ in range(max_iter):
            jac = mass_action_jacobian(at_t, c) - d * eye
            step = torch.linalg.lstsq(jac, -f.unsqueeze(-1)).solution.squeeze(-1)
            alpha, norm = 1.0, float(f.norm())
            for _ in range(40):
                candidate = (c + alpha * step).clamp(min=0)
                f_candidate = residual(candidate)
                if float(f_candidate.norm()) < norm or norm == 0.0:
                    break
                alpha *= 0.5
            moved = float((candidate - c).abs().max())
            c, f = candidate, f_candidate
            if moved <= tol * (1.0 + float(c.abs().max())):
                return c
        msg = f"CSTR steady state did not converge in {max_iter} iterations"
        raise RuntimeError(msg)


def van_de_vusse_cstr(feed_a: float = VAN_DE_VUSSE_FEED_A) -> VanDeVusseCSTR:
    """Return the Van de Vusse CSTR with Klatt and Engell (1998) kinetics.

    Parameters
    ----------
    feed_a : float, optional
        Feed concentration of ``A`` in mol/L (finite, > 0). Default is
        :data:`VAN_DE_VUSSE_FEED_A`.

    Returns
    -------
    VanDeVusseCSTR
        Reactor with feed ``(feed_a, 0, 0, 0)``.

    Raises
    ------
    ValueError
        If ``feed_a`` is not finite and positive.
    """
    if not math.isfinite(float(feed_a)) or float(feed_a) <= 0.0:
        msg = f"feed_a must be finite and > 0, got {feed_a!r}"
        raise ValueError(msg)
    arrhenius = tuple(
        ArrheniusParams(k0, ea_over_r * GAS_CONSTANT_J_PER_MOL_K)
        for k0, ea_over_r in zip(_K0, _EA_OVER_R, strict=True)
    )
    nominal = tuple(
        float(p.rate_constant(VAN_DE_VUSSE_NOMINAL_TEMPERATURE_K)) for p in arrhenius
    )
    system = ReactionSystem(
        species=("A", "B", "C", "D"),
        reactant_stoichiometry=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]],
            dtype=torch.float64,
        ),
        product_stoichiometry=torch.tensor(
            [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=torch.float64,
        ),
        rate_constants=torch.tensor(nominal, dtype=torch.float64),
        reaction_names=("A -> B", "B -> C", "2 A -> D"),
        arrhenius=arrhenius,
        name="van_de_vusse_cstr",
    )
    feed = torch.tensor([float(feed_a), 0.0, 0.0, 0.0], dtype=torch.float64)
    return VanDeVusseCSTR(system=system, feed=feed)


def _dilution(value: float) -> float:
    """Validate a finite, non-negative dilution rate.

    Parameters
    ----------
    value
        See signature.

    Returns
    -------
        See signature."""
    d = float(value)
    if not math.isfinite(d) or d < 0.0:
        msg = f"dilution_rate must be finite and >= 0, got {value!r}"
        raise ValueError(msg)
    return d


def _schedule(value: float | Tensor, num_intervals: int, name: str) -> list[float]:
    """Expand a scalar or per-interval tensor into a list of floats.

    Parameters
    ----------
    value
        See signature.
    num_intervals
        See signature.
    name
        See signature.

    Returns
    -------
        See signature."""
    if isinstance(value, Tensor):
        if value.shape != (num_intervals,):
            msg = (
                f"{name} schedule must have shape ({num_intervals},), got "
                f"{tuple(value.shape)}"
            )
            raise ValueError(msg)
        return [float(v) for v in value.tolist()]
    return [float(value)] * num_intervals
