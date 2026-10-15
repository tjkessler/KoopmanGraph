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

from koopman_graph.datasets.reaction.system import ReactionSystem

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
        Concentrations ``(len(times), ..., num_species)`` in ``float64``.

    Raises
    ------
    ValueError
        If the system is not linear or inputs are malformed.
    """
    a, b = linear_rate_equations(system)
    c0 = _validate_initial(system, initial).to(torch.float64)
    t = _validate_times(times)
    num_species = system.num_species
    generator = torch.zeros(num_species + 1, num_species + 1, dtype=torch.float64)
    generator[:num_species, :num_species] = a
    generator[:num_species, num_species] = b
    elapsed = (t - t[0]).to(dtype=torch.float64, device="cpu")
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
    c = _validate_initial(system, initial)
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
    c0 = _validate_initial(system, initial)
    if c0.ndim != 1:
        msg = (
            "stiff integration takes a single initial state of shape "
            f"({system.num_species},), got {tuple(c0.shape)}"
        )
        raise ValueError(msg)
    t = _validate_times(times).to(dtype=torch.float64, device="cpu")
    scipy_integrate = require_scipy()

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
        state = torch.from_numpy(y)
        return system.time_derivative(state).numpy()

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
        return mass_action_jacobian(system, torch.from_numpy(y)).numpy()

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
        :func:`integrate_stiff` (``rtol``, ``atol``).

    Returns
    -------
    Tensor
        Concentrations ``(len(times), ..., num_species)``.

    Raises
    ------
    ValueError
        If ``method`` is unknown or ``kwargs`` do not apply to it.
    """
    if method == "auto":
        method = "expm" if system.is_linear else "rk4"
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
    c = _validate_initial(system, concentrations)
    if c.ndim != 1:
        msg = (
            f"concentrations must have shape ({system.num_species},), got "
            f"{tuple(c.shape)}"
        )
        raise ValueError(msg)
    alpha = system.reactant_stoichiometry.to(dtype=c.dtype, device=c.device)
    k = system.rate_constants.to(dtype=c.dtype, device=c.device)
    num_species = system.num_species
    # ∂r_j/∂c_s = k_j α_js c_s^(α_js - 1) ∏_{s'≠s} c_s'^α_js'; building the
    # product with exponent α - e_s keeps c = 0 finite (no division by c).
    eye = torch.eye(num_species, dtype=c.dtype, device=c.device)
    shifted = alpha.unsqueeze(1) - eye.unsqueeze(0)  # (R, S_s, S_s')
    powered = torch.where(
        shifted < 0,
        torch.ones((), dtype=c.dtype, device=c.device),
        torch.pow(c, shifted.clamp(min=0)),
    )
    d_rates = k[:, None] * alpha * powered.prod(dim=-1)  # (R, S)
    n = system.stoichiometric_matrix.to(dtype=c.dtype, device=c.device)
    return n @ d_rates


def _require_linear(system: ReactionSystem) -> None:
    """Raise ``ValueError`` unless every reaction has order at most one.

    Parameters
    ----------
    system
        See signature."""
    if not isinstance(system, ReactionSystem):
        msg = f"system must be a ReactionSystem, got {type(system).__name__}"
        raise TypeError(msg)
    if not system.is_linear:
        orders = system.reaction_orders.tolist()
        msg = (
            f"{system.name!r} is not linear (reaction orders {orders}); "
            "use method='rk4' or a stiff SciPy method"
        )
        raise ValueError(msg)


def _validate_initial(system: ReactionSystem, initial: Tensor) -> Tensor:
    """Validate an initial state ``(..., num_species)`` of finite floats.

    Parameters
    ----------
    system
        See signature.
    initial
        See signature.

    Returns
    -------
        See signature."""
    if not isinstance(system, ReactionSystem):
        msg = f"system must be a ReactionSystem, got {type(system).__name__}"
        raise TypeError(msg)
    if not isinstance(initial, Tensor) or not initial.is_floating_point():
        msg = "initial concentrations must be a floating-point Tensor"
        raise TypeError(msg)
    if initial.ndim == 0 or int(initial.shape[-1]) != system.num_species:
        msg = (
            f"initial concentrations must have shape (..., {system.num_species}), "
            f"got {tuple(initial.shape)}"
        )
        raise ValueError(msg)
    if not bool(torch.isfinite(initial).all()):
        msg = "initial concentrations must be finite"
        raise ValueError(msg)
    return initial


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
