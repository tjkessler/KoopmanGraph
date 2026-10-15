"""Exact references for reaction systems: equilibria, spectra, timescales.

These are the oracle quantities that a learned Koopman operator can be
checked against:

* :func:`conservation_laws` returns an orthonormal basis of the left
  nullspace of the stoichiometric matrix: every ``w`` with ``wᵀ N = 0``
  gives a conserved total ``wᵀ c``. This covers element balance and any
  extra stoichiometric conservation.
* :func:`equilibrium` finds a steady state in the stoichiometric
  compatibility class of an initial state, ``{c ≥ 0 : W c = W c0}``. It is
  exact for linear systems and uses damped Newton otherwise.
* :func:`oracle_spectrum` gives the Jacobian at a state, its eigenvalues,
  and relaxation timescales ``-1 / Re(λ)``. It drops the
  ``num_species - rank(N)`` zero modes that conservation laws force.

Pure PyTorch; importing this module does not import scipy or cantera.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from koopman_graph.datasets.reaction.integrate import (
    linear_rate_equations,
    mass_action_jacobian,
)
from koopman_graph.datasets.reaction.system import ReactionSystem


@dataclass(frozen=True)
class OracleSpectrum:
    """Linearization of a reaction system at one state.

    Attributes
    ----------
    state : Tensor
        ``float64`` state ``(num_species,)`` the Jacobian is evaluated at.
    jacobian : Tensor
        ``float64`` Jacobian ``∂(dc/dt)/∂c`` with shape
        ``(num_species, num_species)``.
    dynamic_eigenvalues : Tensor
        ``complex128`` eigenvalues after removing conservation zero modes,
        sorted by real part, largest (slowest-decaying) first.
    conservation_eigenvalues : Tensor
        ``complex128`` eigenvalues removed as conservation zero modes (the
        ``num_conservation_laws`` eigenvalues of smallest modulus).
    num_conservation_laws : int
        ``num_species - rank(N)``.
    """

    state: Tensor
    jacobian: Tensor
    dynamic_eigenvalues: Tensor
    conservation_eigenvalues: Tensor
    num_conservation_laws: int

    @property
    def timescales(self) -> Tensor:
        """Return ``-1 / Re(λ)`` for each dynamic eigenvalue.

        Returns
        -------
        Tensor
            ``float64`` tensor aligned with :attr:`dynamic_eigenvalues`:
            positive for decaying modes, ``inf`` when ``Re(λ) = 0`` (e.g. a
            neutral centre), and negative for growing modes.
        """
        real = self.dynamic_eigenvalues.real
        return torch.where(
            real == 0,
            torch.full_like(real, math.inf),
            -1.0 / real,
        )

    @property
    def is_stable(self) -> bool:
        """Return whether every dynamic mode decays.

        Returns
        -------
        bool
            ``True`` when all dynamic eigenvalues have ``Re(λ) < 0``.
        """
        return bool((self.dynamic_eigenvalues.real < 0).all())

    def discrete_eigenvalues(self, time_step: float) -> Tensor:
        """Return the eigenvalues of the flow map over ``time_step``.

        Parameters
        ----------
        time_step : float
            Sampling interval ``Δt`` (finite, > 0).

        Returns
        -------
        Tensor
            ``complex128`` values ``exp(λ Δt)`` aligned with
            :attr:`dynamic_eigenvalues`, the ones to compare with a
            discrete-time learned Koopman operator.

        Raises
        ------
        ValueError
            If ``time_step`` is not finite and positive.
        """
        dt = float(time_step)
        if not math.isfinite(dt) or dt <= 0.0:
            msg = f"time_step must be finite and > 0, got {time_step!r}"
            raise ValueError(msg)
        return torch.exp(self.dynamic_eigenvalues * dt)


def stoichiometric_rank(system: ReactionSystem) -> int:
    """Return the rank of the stoichiometric matrix ``N``.

    Parameters
    ----------
    system : ReactionSystem
        Reaction network.

    Returns
    -------
    int
        Numerical rank of ``N`` (``num_species × num_reactions``).
    """
    _require_system(system)
    return int(torch.linalg.matrix_rank(system.stoichiometric_matrix).item())


def conservation_laws(system: ReactionSystem) -> Tensor:
    """Return an orthonormal basis of conserved linear combinations.

    Parameters
    ----------
    system : ReactionSystem
        Reaction network.

    Returns
    -------
    Tensor
        ``float64`` matrix ``W`` with shape ``(num_species - rank(N),
        num_species)`` and orthonormal rows, ``W N = 0``. It has zero rows
        when nothing is conserved (e.g. an open system with independent
        sources).
    """
    rank = stoichiometric_rank(system)
    u, _s, _vh = torch.linalg.svd(system.stoichiometric_matrix, full_matrices=True)
    return u[:, rank:].T.contiguous()


def equilibrium(
    system: ReactionSystem,
    initial: Tensor,
    *,
    guess: Tensor | None = None,
    tol: float = 1e-12,
    max_iter: int = 100,
) -> Tensor:
    """Return a steady state in the compatibility class of ``initial``.

    Parameters
    ----------
    system : ReactionSystem
        Reaction network.
    initial : Tensor
        Floating state ``(num_species,)`` that fixes the conserved totals
        ``W c = W initial``.
    guess : Tensor or None, optional
        Newton starting point for nonlinear systems. Default is
        ``initial``. Systems with several steady states (e.g. Lotka-Volterra,
        whose origin is also a fixed point) return the one Newton reaches
        from here.
    tol : float, optional
        Relative step tolerance for Newton convergence. Default is
        ``1e-12``.
    max_iter : int, optional
        Newton iteration cap. Default is ``100``.

    Returns
    -------
    Tensor
        ``float64`` steady state ``c*`` with ``dc/dt(c*) ≈ 0`` and
        ``W c* = W initial``. Nonlinear solves are kept non-negative.

    Raises
    ------
    ValueError
        If a linear system has no steady state in the class (e.g. a feed
        accumulating into a species with no outflow), or arguments are
        invalid.
    RuntimeError
        If Newton does not converge within ``max_iter`` iterations.
    """
    c0 = _state(system, initial, "initial")
    if not math.isfinite(float(tol)) or float(tol) <= 0.0:
        msg = f"tol must be finite and > 0, got {tol!r}"
        raise ValueError(msg)
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
        msg = f"max_iter must be a positive int, got {max_iter!r}"
        raise ValueError(msg)
    w = conservation_laws(system)
    if system.is_linear:
        return _linear_equilibrium(system, c0, w)

    rank = int(system.num_species - w.shape[0])
    u, _s, _vh = torch.linalg.svd(system.stoichiometric_matrix, full_matrices=True)
    q = u[:, :rank]  # orthonormal basis of range(N); dc/dt always lies in it
    targets = w @ c0

    def residual(c: Tensor) -> Tensor:
        """Stack the projected rate equations and conservation constraints.

        Parameters
        ----------
        c
            See signature.

        Returns
        -------
            See signature."""
        return torch.cat([q.T @ system.time_derivative(c), w @ c - targets])

    start = c0 if guess is None else _state(system, guess, "guess")
    c = start.clamp(min=0)
    f = residual(c)
    for _ in range(max_iter):
        jac = torch.cat([q.T @ mass_action_jacobian(system, c), w])
        step = torch.linalg.lstsq(jac, -f.unsqueeze(-1)).solution.squeeze(-1)
        alpha = 1.0
        norm = float(f.norm())
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
    msg = (
        f"Newton did not converge for {system.name!r} in {max_iter} iterations "
        f"(residual norm {float(f.norm()):.3e})"
    )
    raise RuntimeError(msg)


def oracle_spectrum(system: ReactionSystem, state: Tensor) -> OracleSpectrum:
    """Linearize the mass-action field at ``state`` and split its spectrum.

    Parameters
    ----------
    system : ReactionSystem
        Reaction network.
    state : Tensor
        Floating state ``(num_species,)``, typically an :func:`equilibrium`.

    Returns
    -------
    OracleSpectrum
        Jacobian, dynamic and conservation eigenvalues, and timescales.

    Notes
    -----
    ``J = N ∂r/∂c`` has rank at most ``rank(N)``, so at least
    ``num_species - rank(N)`` eigenvalues are exactly zero. These are the
    ones removed. Extra zeros (e.g. at a boundary state) stay among the
    dynamic eigenvalues and show up as ``inf`` timescales.
    """
    c = _state(system, state, "state")
    jac = mass_action_jacobian(system, c)
    eigenvalues = torch.linalg.eigvals(jac)
    num_laws = system.num_species - stoichiometric_rank(system)
    by_modulus = torch.argsort(eigenvalues.abs(), stable=True)
    conservation = eigenvalues[by_modulus[:num_laws]]
    dynamic = eigenvalues[by_modulus[num_laws:]]
    dynamic = dynamic[torch.argsort(dynamic.real, descending=True, stable=True)]
    return OracleSpectrum(
        state=c,
        jacobian=jac,
        dynamic_eigenvalues=dynamic,
        conservation_eigenvalues=conservation,
        num_conservation_laws=num_laws,
    )


def oracle_timescales(system: ReactionSystem, state: Tensor) -> Tensor:
    """Return relaxation timescales at ``state``, slowest first.

    Parameters
    ----------
    system : ReactionSystem
        Reaction network.
    state : Tensor
        Floating state ``(num_species,)``.

    Returns
    -------
    Tensor
        ``OracleSpectrum.timescales`` of :func:`oracle_spectrum`.
    """
    return oracle_spectrum(system, state).timescales


def _linear_equilibrium(system: ReactionSystem, c0: Tensor, w: Tensor) -> Tensor:
    """Solve ``A c + b = 0`` with ``W c = W c0`` for a linear system.

    Parameters
    ----------
    system
        See signature.
    c0
        See signature.
    w
        See signature.

    Returns
    -------
        See signature."""
    a, b = linear_rate_equations(system)
    lhs = torch.cat([a, w])
    rhs = torch.cat([-b, w @ c0])
    solution = torch.linalg.lstsq(lhs, rhs.unsqueeze(-1)).solution.squeeze(-1)
    scale = 1.0 + float(rhs.abs().max())
    if float((lhs @ solution - rhs).abs().max()) > 1e-9 * scale:
        msg = (
            f"{system.name!r} has no steady state for this initial state "
            "(a source feeds a species with no outflow)"
        )
        raise ValueError(msg)
    return solution


def _state(system: ReactionSystem, value: Tensor, name: str) -> Tensor:
    """Validate one finite state ``(num_species,)`` and return it as float64.

    Parameters
    ----------
    system
        See signature.
    value
        See signature.
    name
        See signature.

    Returns
    -------
        See signature."""
    _require_system(system)
    if not isinstance(value, Tensor) or not value.is_floating_point():
        msg = f"{name} must be a floating-point Tensor"
        raise TypeError(msg)
    if value.shape != (system.num_species,):
        msg = (
            f"{name} must have shape ({system.num_species},), got {tuple(value.shape)}"
        )
        raise ValueError(msg)
    if not bool(torch.isfinite(value).all()):
        msg = f"{name} must be finite"
        raise ValueError(msg)
    return value.detach().to(dtype=torch.float64, device="cpu")


def _require_system(system: object) -> None:
    """Raise ``TypeError`` unless ``system`` is a :class:`ReactionSystem`.

    Parameters
    ----------
    system
        See signature."""
    if not isinstance(system, ReactionSystem):
        msg = f"system must be a ReactionSystem, got {type(system).__name__}"
        raise TypeError(msg)
