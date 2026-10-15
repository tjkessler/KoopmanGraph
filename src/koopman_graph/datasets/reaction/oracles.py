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

_ZERO_REAL_RTOL = 1e-10
"""Relative tolerance, in units of ``‖J‖``, below which ``Re(λ)`` counts as 0."""

_MAX_BACKTRACKS = 40


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
    zero_tolerance : float
        Absolute tolerance below which ``|Re(λ)|`` is treated as zero, so
        rounding in ``eigvals`` cannot turn a neutral mode into a growing or
        decaying one. Default is ``0.0`` (exact comparison);
        :func:`oracle_spectrum` sets it relative to ``‖J‖``.
    """

    state: Tensor
    jacobian: Tensor
    dynamic_eigenvalues: Tensor
    conservation_eigenvalues: Tensor
    num_conservation_laws: int
    zero_tolerance: float = 0.0

    @property
    def timescales(self) -> Tensor:
        """Return ``-1 / Re(λ)`` for each dynamic eigenvalue.

        Returns
        -------
        Tensor
            ``float64`` tensor aligned with :attr:`dynamic_eigenvalues`:
            positive for decaying modes, ``inf`` when ``|Re(λ)|`` is within
            :attr:`zero_tolerance` (a neutral centre, or a species drained
            onto a zero-flux boundary), and negative for growing modes.
            ``zero_tolerance = 0`` compares exactly.
        """
        return eigenvalue_timescales(
            self.dynamic_eigenvalues, zero_tolerance=self.zero_tolerance
        )

    @property
    def is_stable(self) -> bool:
        """Return whether every dynamic mode decays.

        Returns
        -------
        bool
            ``True`` when all dynamic eigenvalues have
            ``Re(λ) < -zero_tolerance``.
        """
        return bool((self.dynamic_eigenvalues.real < -self.zero_tolerance).all())

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


# Eigenvalues are only resolved to about ``eps · ‖J‖``; real parts below a
# small multiple of that are numerical zeros. Keeping the default cutoff at
# round-off level preserves genuinely slow modes in stiff mechanisms, whose
# eigenvalue magnitudes can span ten or more decades.
_ZERO_EIGENVALUE_RTOL = 100.0 * torch.finfo(torch.float64).eps


def eigenvalue_timescales(
    eigenvalues: Tensor, *, zero_tolerance: float | None = None
) -> Tensor:
    """Return relaxation timescales ``-1 / Re(λ)`` with numerical zeros as ``inf``.

    Parameters
    ----------
    eigenvalues : Tensor
        Complex (or real) eigenvalues ``(k,)``.
    zero_tolerance : float or None, optional
        Absolute cutoff: ``|Re(λ)| <= zero_tolerance`` maps to ``inf``.
        Default ``None`` uses round-off level, ``100 · eps · max|λ|``
        (about ``2.2e-14 · max|λ|``). :func:`oracle_spectrum` passes
        ``1e-10 ‖J‖_F`` through :attr:`OracleSpectrum.zero_tolerance`.

    Returns
    -------
    Tensor
        ``float64`` timescales ``(k,)``.
    """
    real = eigenvalues.real.to(torch.float64)
    if zero_tolerance is None:
        scale = float(eigenvalues.abs().max()) if eigenvalues.numel() else 0.0
        zero_tolerance = _ZERO_EIGENVALUE_RTOL * scale
    zero = real.abs() <= zero_tolerance
    safe = torch.where(zero, torch.ones_like(real), real)
    return torch.where(zero, torch.full_like(real, math.inf), -1.0 / safe)


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
    return _stoichiometric_basis(system)[0]


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
    return _stoichiometric_basis(system)[2]


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
        ``W c* = W initial``. For a linear system this is
        ``c_p + P0 (initial - c_p)``, where ``c_p`` is any steady state and
        ``P0`` the spectral projector onto the kernel of ``A``. That is the
        ``t → ∞`` limit from ``initial`` whenever the nonzero modes decay,
        also when the steady state in the class is not unique (e.g.
        ``A → B`` and ``A → C`` split ``A`` in the ratio of their rates).
        Entries within rounding of zero are set to zero.

    Raises
    ------
    ValueError
        If a linear system has no steady state (e.g. a feed accumulating
        into a species with no outflow) or no limiting one (a defective
        zero eigenvalue), or arguments are invalid.
    RuntimeError
        If Newton does not reach a steady state within ``max_iter``
        iterations, its line search stalls away from one, or it stops at a
        non-negative point that is not stationary. Clamping can stop Newton
        on the boundary where every flux is tiny but flows one way; pass a
        ``guess`` closer to the interior steady state, or use a
        thermochemical equilibrium (e.g. Cantera's ``equilibrate``) as the
        reference.
    """
    c0 = _state(system, initial, "initial")
    if not math.isfinite(float(tol)) or float(tol) <= 0.0:
        msg = f"tol must be finite and > 0, got {tol!r}"
        raise ValueError(msg)
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
        msg = f"max_iter must be a positive int, got {max_iter!r}"
        raise ValueError(msg)
    if system.is_linear:
        return _linear_equilibrium(system, c0)

    # q is an orthonormal basis of range(N); dc/dt always lies in it.
    _rank, q, w = _stoichiometric_basis(system)
    targets = w @ c0
    # Step convergence alone also fires when the line search or the c >= 0
    # clamp stalls, so the residual must be small as well.
    residual_tol = math.sqrt(float(tol)) * (1.0 + float(c0.abs().max()))

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
    # Turnover at the initial state sets the absolute stationarity scale.
    flux_scale = float(_gross_turnover(system, c0.clamp(min=0)).max())
    f = residual(c)
    for _ in range(max_iter):
        jac = torch.cat([q.T @ mass_action_jacobian(system, c), w])
        step = torch.linalg.lstsq(jac, -f.unsqueeze(-1)).solution.squeeze(-1)
        alpha = 1.0
        norm = float(f.norm())
        for _ in range(_MAX_BACKTRACKS):
            candidate = (c + alpha * step).clamp(min=0)
            f_candidate = residual(candidate)
            if float(f_candidate.norm()) < norm:
                break
            alpha *= 0.5
        else:
            if norm <= residual_tol:
                _require_stationary(system, c, flux_scale=flux_scale)
                return c
            msg = (
                f"Newton line search stalled for {system.name!r} "
                f"(residual norm {norm:.3e})"
            )
            raise RuntimeError(msg)
        moved = float((candidate - c).abs().max())
        c, f = candidate, f_candidate
        if (
            moved <= tol * (1.0 + float(c.abs().max()))
            and float(f.norm()) <= residual_tol
        ):
            _require_stationary(system, c, flux_scale=flux_scale)
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
    dynamic eigenvalues and show up as ``inf`` timescales. Real parts within
    ``1e-10 ‖J‖_F`` of zero count as zero.
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
        zero_tolerance=_ZERO_REAL_RTOL * float(torch.linalg.matrix_norm(jac)),
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


_STATIONARY_RTOL = 1e-6


_STATIONARY_ATOL = 1e-12


def _gross_turnover(system: ReactionSystem, c: Tensor) -> Tensor:
    """Return per-species gross turnover ``Σ_j r_j (α_js + β_js)``.

    Parameters
    ----------
    system
        See signature.
    c
        See signature.

    Returns
    -------
        See signature."""
    rates = system.reaction_rates(c)
    return rates @ (system.reactant_stoichiometry + system.product_stoichiometry)


def _require_stationary(
    system: ReactionSystem, c: Tensor, *, flux_scale: float = 0.0
) -> None:
    """Raise unless net rates are negligible against species turnover.

    A species passes when ``|dc_s/dt| <= rtol * gross_s + atol * flux_scale``:
    the relative term rejects one-way flux at a clamped boundary point, and
    the absolute term (scaled by the peak gross turnover at ``initial``)
    accepts vanishing leftovers on the way to a boundary steady state.

    Parameters
    ----------
    system
        See signature.
    c
        See signature.
    flux_scale
        See signature."""
    net = system.time_derivative(c).abs()
    gross = _gross_turnover(system, c)
    allowed = _STATIONARY_RTOL * gross + _STATIONARY_ATOL * flux_scale
    if bool((net > allowed).any()):
        tiny = torch.finfo(torch.float64).tiny
        ratio = float((net / gross.clamp(min=tiny)).max())
        msg = (
            f"Newton stalled at a non-stationary point for {system.name!r} "
            f"(max net/gross species turnover {ratio:.3e}); pass a guess "
            "closer to the steady state"
        )
        raise RuntimeError(msg)


def _stoichiometric_basis(system: ReactionSystem) -> tuple[int, Tensor, Tensor]:
    """Return ``rank(N)`` and orthonormal bases of ``range(N)`` and its complement.

    Parameters
    ----------
    system
        See signature.

    Returns
    -------
        ``(rank, q, w)`` with ``q`` of shape ``(num_species, rank)`` and the
        conservation laws ``w`` of shape ``(num_species - rank, num_species)``,
        from one SVD."""
    _require_system(system)
    u, _vh, rank = _svd_rank(system.stoichiometric_matrix)
    return rank, u[:, :rank], u[:, rank:].T.contiguous()


def _svd_rank(matrix: Tensor) -> tuple[Tensor, Tensor, int]:
    """Return the full singular vectors and numerical rank of ``matrix``.

    Parameters
    ----------
    matrix
        See signature.

    Returns
    -------
        ``(u, vh, rank)``, with the ``torch.linalg.matrix_rank`` default
        tolerance ``max(m, n) eps σ_max``."""
    u, singular, vh = torch.linalg.svd(matrix, full_matrices=True)
    eps = torch.finfo(matrix.dtype).eps
    cutoff = max(matrix.shape) * eps * float(singular.max())
    return u, vh, int((singular > cutoff).sum())


def _linear_equilibrium(system: ReactionSystem, c0: Tensor) -> Tensor:
    """Return ``c_p + P0 (c0 - c_p)`` for the linear field ``A c + b``.

    Parameters
    ----------
    system
        See signature.
    c0
        See signature.

    Returns
    -------
        See signature."""
    a, b = linear_rate_equations(system)
    particular = torch.linalg.lstsq(a, -b.unsqueeze(-1)).solution.squeeze(-1)
    scale = 1.0 + max(float(b.abs().max()), float(c0.abs().max()))
    if float((a @ particular + b).abs().max()) > 1e-9 * scale:
        msg = (
            f"{system.name!r} has no steady state for this initial state "
            "(a source feeds a species with no outflow)"
        )
        raise ValueError(msg)
    # P0 = K (L K)^-1 L projects onto ker(A) along range(A), where K and L
    # are orthonormal right and left kernel bases. L K is singular exactly
    # when the zero eigenvalue is defective and the trajectory never settles.
    u, vh, rank = _svd_rank(a)
    solution = particular
    if rank < system.num_species:
        right = vh[rank:].T
        left = u[:, rank:].T
        coupling = left @ right
        if float(torch.linalg.svdvals(coupling).min()) < 1e-8:
            msg = (
                f"{system.name!r} has no limiting steady state for this "
                "initial state (the zero eigenvalue of its rate matrix is "
                "defective)"
            )
            raise ValueError(msg)
        projector = right @ torch.linalg.solve(coupling, left)
        solution = particular + projector @ (c0 - particular)
    return torch.where(solution.abs() <= 1e-12 * scale, 0.0, solution)


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
