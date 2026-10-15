"""Time integration of mass-action reaction systems.

Three paths, picked by :func:`integrate_reaction_system` (``method="auto"``
uses the exact path for linear systems and RK4 otherwise):

``"expm"``
    Exact propagation for linear systems (every reaction of order at most
    one). The affine field ``dc/dt = A c + b`` is folded into one augmented
    matrix exponential, so zeroth-order sources are exact too.
``"rk4"``
    Classical fixed-step fourth-order Runge-Kutta in PyTorch for non-stiff
    systems. It supports batched initial conditions and autograd.
``"bdf"`` / ``"radau"`` / ``"lsoda"``
    SciPy :func:`scipy.integrate.solve_ivp` stiff solvers with an exact
    mass-action Jacobian. They need the ``[reaction]`` extra; SciPy is
    imported only when one of these methods runs.

All paths take output ``times`` (strictly increasing, with ``times[0]`` the
time of ``initial``) and return concentrations of shape
``(len(times), ..., num_species)`` whose first entry is ``initial``.
Importing this module does not import scipy or cantera.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
from torch import Tensor

from koopman_graph.datasets.reaction.system import ReactionSystem, _require_system

IntegrationMethod = Literal["auto", "expm", "rk4", "bdf", "radau", "lsoda"]
"""Accepted ``method`` names for :func:`integrate_reaction_system`."""

_STIFF_METHODS = {"bdf": "BDF", "radau": "Radau", "lsoda": "LSODA"}

_SCIPY_IMPORT_ERROR = (
    "scipy is required for stiff reaction integration (method='bdf', "
    "'radau', or 'lsoda'). Install with: pip install koopman-graph[reaction]"
)


def require_scipy() -> Any:
    """Import and return :mod:`scipy.integrate`, or explain how to install it.

    Returns
    -------
    module
        The :mod:`scipy.integrate` module.

    Raises
    ------
    ImportError
        If SciPy is not installed; the message names the ``[reaction]``
        extra.
    """
    try:
        import scipy.integrate as scipy_integrate
    except ImportError as exc:
        raise ImportError(_SCIPY_IMPORT_ERROR) from exc
    return scipy_integrate


def linear_rate_equations(system: ReactionSystem) -> tuple[Tensor, Tensor]:
    """Return ``(A, b)`` with ``dc/dt = A c + b`` for a linear system.

    Parameters
    ----------
    system : ReactionSystem
        System with :attr:`~ReactionSystem.is_linear` true.

    Returns
    -------
    A : Tensor
        ``float64`` rate matrix with shape ``(num_species, num_species)``.
    b : Tensor
        ``float64`` constant source term with shape ``(num_species,)``.

    Raises
    ------
    ValueError
        If any reaction has total order greater than one.
    """
    _require_linear(system)
    n = system.stoichiometric_matrix  # (S, R)
    # For order-0 / order-1 reactions r_j = k_j α_j · c + k_j [order_j == 0].
    weighted_alpha = system.rate_constants.unsqueeze(-1) * (
        system.reactant_stoichiometry
    )  # (R, S)
    a = n @ weighted_alpha
    zeroth = (system.reaction_orders == 0).to(torch.float64)
    b = n @ (system.rate_constants * zeroth)
    return a, b


def integrate_expm(
    system: ReactionSystem,
    initial: Tensor,
    times: Tensor,
) -> Tensor:
    """Propagate a linear system exactly with an augmented matrix exponential.

    Parameters
    ----------
    system : ReactionSystem
        System with :attr:`~ReactionSystem.is_linear` true.
    initial : Tensor
        Floating initial concentrations with shape ``(..., num_species)``.
    times : Tensor
        Strictly increasing 1-D output times; ``times[0]`` is the time of
        ``initial``.

    Returns
    -------
    Tensor
        Concentrations ``(len(times), ..., num_species)`` in ``float64`` on
        the device of ``initial``.

    Raises
    ------
    ValueError
        If the system is not linear or inputs are malformed.
    """
    a, b = linear_rate_equations(system)
    c0 = _validate_state(system, initial, "initial").to(torch.float64)
    t = _validate_times(times)
    num_species = system.num_species
    device = c0.device
    generator = torch.zeros(
        num_species + 1, num_species + 1, dtype=torch.float64, device=device
    )
    generator[:num_species, :num_species] = a.to(device)
    generator[:num_species, num_species] = b.to(device)
    t = t.to(dtype=torch.float64, device=device)
    elapsed = t - t[0]
    propagators = torch.linalg.matrix_exp(elapsed[:, None, None] * generator)
    augmented = torch.cat([c0, torch.ones_like(c0[..., :1])], dim=-1)
    # result[t, ..., i] = Σ_j P[t, i, j] y[..., j]
    states = torch.einsum("tij,...j->t...i", propagators, augmented)
    return states[..., :num_species]


def integrate_rk4(
    system: ReactionSystem,
    initial: Tensor,
    times: Tensor,
    *,
    substeps: int = 10,
) -> Tensor:
    """Integrate with classical fixed-step RK4 (non-stiff systems).

    Parameters
    ----------
    system : ReactionSystem
        Reaction network.
    initial : Tensor
        Floating initial concentrations with shape ``(..., num_species)``.
        The output keeps its dtype and device; gradients flow through.
    times : Tensor
        Strictly increasing 1-D output times; ``times[0]`` is the time of
        ``initial``.
    substeps : int, optional
        Equal RK4 steps per output interval. Default is ``10``.

    Returns
    -------
    Tensor
        Concentrations ``(len(times), ..., num_species)``.

    Raises
    ------
    ValueError
        If ``substeps < 1`` or inputs are malformed.
    """
    if isinstance(substeps, bool) or not isinstance(substeps, int) or substeps < 1:
        msg = f"substeps must be a positive int, got {substeps!r}"
        raise ValueError(msg)
    c = _validate_state(system, initial, "initial")
    t = _validate_times(times).to(dtype=c.dtype, device=c.device)
    states = [c]
    for index in range(1, int(t.shape[0])):
        h = (t[index] - t[index - 1]) / substeps
        for _ in range(substeps):
            k1 = system.time_derivative(c)
            k2 = system.time_derivative(c + 0.5 * h * k1)
            k3 = system.time_derivative(c + 0.5 * h * k2)
            k4 = system.time_derivative(c + h * k3)
            c = c + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        states.append(c)
    return torch.stack(states)


def integrate_stiff(
    system: ReactionSystem,
    initial: Tensor,
    times: Tensor,
    *,
    method: Literal["bdf", "radau", "lsoda"] = "bdf",
    rtol: float = 1e-8,
    atol: float = 1e-12,
) -> Tensor:
    """Integrate a stiff system with SciPy ``solve_ivp`` (``[reaction]`` extra).

    Parameters
    ----------
    system : ReactionSystem
        Reaction network.
    initial : Tensor
        Floating initial concentrations with shape ``(num_species,)``.
    times : Tensor
        Strictly increasing 1-D output times; ``times[0]`` is the time of
        ``initial``.
    method : {"bdf", "radau", "lsoda"}, optional
        SciPy implicit method. Default is ``"bdf"``.
    rtol : float, optional
        Relative tolerance. Default is ``1e-8``.
    atol : float, optional
        Absolute tolerance. Default is ``1e-12``.

    Returns
    -------
    Tensor
        ``float64`` concentrations with shape ``(len(times), num_species)``.

    Raises
    ------
    ImportError
        If SciPy is not installed.
    ValueError
        If ``method`` is unknown, ``initial`` is batched, or inputs are
        malformed.
    RuntimeError
        If the solver reports failure.
    """
    if method not in _STIFF_METHODS:
        msg = f"method must be one of {sorted(_STIFF_METHODS)}, got {method!r}"
        raise ValueError(msg)
    c0 = _validate_state(system, initial, "initial")
    if c0.ndim != 1:
        msg = (
            "stiff integration takes a single initial state of shape "
            f"({system.num_species},), got {tuple(c0.shape)}"
        )
        raise ValueError(msg)
    t = _validate_times(times).to(dtype=torch.float64, device="cpu")
    scipy_integrate = require_scipy()
    # Built once; solve_ivp calls rhs / jac many times per solve.
    alpha = system.reactant_stoichiometry
    k = system.rate_constants
    n = system.stoichiometric_matrix
    terms = _jacobian_terms(system, dtype=torch.float64, device=torch.device("cpu"))

    def rhs(_time: float, y: Any) -> Any:
        """Evaluate ``dc/dt`` on a NumPy state for ``solve_ivp``.

        Parameters
        ----------
        _time
            See signature.
        y
            See signature.

        Returns
        -------
            See signature."""
        rates = k * torch.pow(torch.from_numpy(y), alpha).prod(dim=-1)
        return (n @ rates).numpy()

    def jac(_time: float, y: Any) -> Any:
        """Evaluate the mass-action Jacobian on a NumPy state for ``solve_ivp``.

        Parameters
        ----------
        _time
            See signature.
        y
            See signature.

        Returns
        -------
            See signature."""
        return _jacobian_from_terms(torch.from_numpy(y), *terms).numpy()

    t_values = t.tolist()
    solution = scipy_integrate.solve_ivp(
        rhs,
        (t_values[0], t_values[-1]),
        c0.detach().to(dtype=torch.float64, device="cpu").numpy(),
        method=_STIFF_METHODS[method],
        t_eval=t_values,
        jac=jac,
        rtol=float(rtol),
        atol=float(atol),
    )
    if not solution.success:
        msg = f"solve_ivp ({_STIFF_METHODS[method]}) failed: {solution.message}"
        raise RuntimeError(msg)
    return torch.from_numpy(solution.y.T.copy())


def integrate_reaction_system(
    system: ReactionSystem,
    initial: Tensor,
    times: Tensor,
    *,
    method: IntegrationMethod = "auto",
    **kwargs: Any,
) -> Tensor:
    """Integrate a reaction system with the requested (or automatic) method.

    Parameters
    ----------
    system : ReactionSystem
        Reaction network.
    initial : Tensor
        Floating initial concentrations with shape ``(..., num_species)``
        (unbatched for stiff methods).
    times : Tensor
        Strictly increasing 1-D output times; ``times[0]`` is the time of
        ``initial``.
    method : {"auto", "expm", "rk4", "bdf", "radau", "lsoda"}, optional
        ``"auto"`` (default) uses ``"expm"`` when the system is linear and
        ``"rk4"`` otherwise. Stiff systems should ask for a SciPy method
        explicitly.
    **kwargs
        Forwarded to :func:`integrate_rk4` (``substeps``) or
        :func:`integrate_stiff` (``rtol``, ``atol``). With ``"auto"`` only
        ``substeps`` is accepted; it is ignored when the exact path is used.

    Returns
    -------
    Tensor
        Concentrations ``(len(times), ..., num_species)``.

    Raises
    ------
    TypeError
        If ``system`` is not a :class:`ReactionSystem`.
    ValueError
        If ``method`` is unknown or ``kwargs`` do not apply to it.
    """
    _require_system(system)
    if method == "auto":
        unknown = sorted(set(kwargs) - {"substeps"})
        if unknown:
            msg = f"method='auto' only accepts 'substeps', got {unknown}"
            raise ValueError(msg)
        if system.is_linear:
            return integrate_expm(system, initial, times)
        method = "rk4"
    if method == "expm":
        if kwargs:
            msg = f"method='expm' takes no options, got {sorted(kwargs)}"
            raise ValueError(msg)
        return integrate_expm(system, initial, times)
    if method == "rk4":
        return integrate_rk4(system, initial, times, **kwargs)
    if method in _STIFF_METHODS:
        return integrate_stiff(system, initial, times, method=method, **kwargs)
    msg = (
        "method must be one of ['auto', 'expm', 'rk4', 'bdf', 'radau', "
        f"'lsoda'], got {method!r}"
    )
    raise ValueError(msg)


def mass_action_jacobian(system: ReactionSystem, concentrations: Tensor) -> Tensor:
    """Return the exact Jacobian ``∂(dc/dt)/∂c`` of the mass-action field.

    Parameters
    ----------
    system : ReactionSystem
        Reaction network.
    concentrations : Tensor
        Floating state with shape ``(num_species,)``.

    Returns
    -------
    Tensor
        Jacobian ``J[t, s] = Σ_j N_tj ∂r_j/∂c_s`` with shape
        ``(num_species, num_species)`` in the dtype of ``concentrations``.
        Finite at ``c = 0``.

    Raises
    ------
    ValueError
        If ``concentrations`` is not a single state.
    """
    c = _validate_state(system, concentrations, "concentrations")
    if c.ndim != 1:
        msg = (
            f"concentrations must have shape ({system.num_species},), got "
            f"{tuple(c.shape)}"
        )
        raise ValueError(msg)
    terms = _jacobian_terms(system, dtype=c.dtype, device=c.device)
    return _jacobian_from_terms(c, *terms)


def _jacobian_terms(
    system: ReactionSystem,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Pack each reaction's reactants into padded slots for the Jacobian.

    Parameters
    ----------
    system
        See signature.
    dtype
        See signature.
    device
        See signature.

    Returns
    -------
        ``(slot_species, slot_alpha, k, n)``: reactant species indices and
        coefficients with shape ``(num_reactions, width)`` (padding has
        ``α = 0``), rate constants, and the stoichiometric matrix."""
    alpha = system.reactant_stoichiometry
    is_reactant = alpha > 0
    width = max(int(is_reactant.sum(dim=1).max()), 1)
    # A stable sort moves each reaction's reactant columns to the front.
    order = torch.sort((~is_reactant).to(torch.uint8), dim=1, stable=True).indices
    slot_species = order[:, :width]
    slot_alpha = alpha.gather(1, slot_species)
    return (
        slot_species.to(device=device),
        slot_alpha.to(dtype=dtype, device=device),
        system.rate_constants.to(dtype=dtype, device=device),
        system.stoichiometric_matrix.to(dtype=dtype, device=device),
    )


def _jacobian_from_terms(
    c: Tensor,
    slot_species: Tensor,
    slot_alpha: Tensor,
    k: Tensor,
    n: Tensor,
) -> Tensor:
    """Evaluate the mass-action Jacobian from :func:`_jacobian_terms` slots.

    Parameters
    ----------
    c
        See signature.
    slot_species
        See signature.
    slot_alpha
        See signature.
    k
        See signature.
    n
        See signature.

    Returns
    -------
        See signature."""
    # ∂r_j/∂c_s = k_j α_js c_s^(α_js - 1) ∏_{s'≠s} c_s'^α_js', taken over the
    # reactant slots only (memory O(R·width²), not O(R·S²)). Lowering the
    # exponent by one instead of dividing by c keeps c = 0 finite.
    values = c[slot_species].unsqueeze(1)  # (R, 1, width)
    eye = torch.eye(slot_alpha.shape[1], dtype=c.dtype, device=c.device)
    shifted = slot_alpha.unsqueeze(1) - eye  # (R, width_s, width_s')
    powered = torch.where(
        shifted < 0,
        torch.ones((), dtype=c.dtype, device=c.device),
        torch.pow(values, shifted.clamp(min=0)),
    )
    slot_grad = k[:, None] * slot_alpha * powered.prod(dim=-1)  # (R, width)
    d_rates = torch.zeros(
        k.shape[0], n.shape[0], dtype=c.dtype, device=c.device
    ).scatter_add(1, slot_species, slot_grad)  # (R, S)
    return n @ d_rates


def _require_linear(system: ReactionSystem) -> None:
    """Raise ``ValueError`` unless every reaction has order at most one.

    Parameters
    ----------
    system
        See signature."""
    _require_system(system)
    if not system.is_linear:
        orders = system.reaction_orders.tolist()
        msg = (
            f"{system.name!r} is not linear (reaction orders {orders}); "
            "use method='rk4' or a stiff SciPy method"
        )
        raise ValueError(msg)


def _validate_state(system: ReactionSystem, state: Tensor, name: str) -> Tensor:
    """Validate a finite floating state ``(..., num_species)``.

    Parameters
    ----------
    system
        See signature.
    state
        See signature.
    name
        Argument name used in the finiteness error.

    Returns
    -------
        See signature."""
    _require_system(system)
    state = system._validate_concentrations(state)
    if not bool(torch.isfinite(state).all()):
        msg = f"{name} must be finite"
        raise ValueError(msg)
    return state


def _validate_times(times: Tensor) -> Tensor:
    """Validate strictly increasing, finite 1-D output times.

    Parameters
    ----------
    times
        See signature.

    Returns
    -------
        See signature."""
    if not isinstance(times, Tensor):
        msg = f"times must be a Tensor, got {type(times).__name__}"
        raise TypeError(msg)
    t = times if times.is_floating_point() else times.to(torch.float64)
    if t.ndim != 1 or int(t.shape[0]) < 1:
        msg = f"times must be a non-empty 1-D Tensor, got {tuple(times.shape)}"
        raise ValueError(msg)
    if not bool(torch.isfinite(t).all()):
        msg = "times must be finite"
        raise ValueError(msg)
    if int(t.shape[0]) > 1 and not bool((t[1:] > t[:-1]).all()):
        msg = "times must be strictly increasing"
        raise ValueError(msg)
    return t
