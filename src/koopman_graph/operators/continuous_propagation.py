"""Continuous-time advance / inverse execution helpers.

Shallow sibling of :mod:`koopman_graph.operators.continuous`. Holds cohesive
controlled / uncontrolled Van Loan application, interval advance / inverse
orchestration, and bilinear inverse mechanics used by
:class:`~koopman_graph.operators.continuous.ContinuousKoopmanOperator`.
Van Loan **factor construction** remains in
:mod:`~koopman_graph.operators.continuous_van_loan`; this module applies those
factors. Prefer importing via ``continuous`` re-exports when writing against
the continuous capability surface.
"""

from __future__ import annotations

import torch
from torch import Tensor

from koopman_graph.operators.auxiliary_spectral import (
    advance_controlled as advance_auxiliary_controlled,
)
from koopman_graph.operators.auxiliary_spectral import (
    advance_uncontrolled as advance_auxiliary_uncontrolled,
)
from koopman_graph.operators.auxiliary_spectral import (
    inverse_advance as inverse_advance_auxiliary,
)
from koopman_graph.operators.continuous_van_loan import van_loan_factors
from koopman_graph.operators.contract import Parameterization
from koopman_graph.operators.control import (
    ControlMode,
    broadcast_control_term,
    effective_bilinear_matrix,
)


def advance_van_loan(
    z: Tensor,
    delta_t: Tensor,
    control: Tensor,
    *,
    generator: Tensor,
    control_matrix: Tensor,
    latent_dim: int,
) -> Tensor:
    """Apply Van Loan factors for a fixed generator over ``Δt``.

    Parameters
    ----------
    z : Tensor
        Latent states with shape ``(..., latent_dim)``.
    delta_t : Tensor
        Integration interval.
    control : Tensor
        Control with shape ``(control_dim,)`` or ``(num_nodes, control_dim)``.
    generator : Tensor
        Fixed generator ``L`` (or effective ``L_eff``) for the interval.
    control_matrix : Tensor
        Control matrix ``B`` with shape ``(control_dim, latent_dim)``.
    latent_dim : int
        Trailing latent dimension (for global-control broadcasting).

    Returns
    -------
    Tensor
        Advanced latents.
    """
    phi11, phi12 = van_loan_factors(generator, control_matrix, delta_t)
    if control.ndim == 1:
        offset = control @ phi12.T
        if z.ndim > 1:
            offset = broadcast_control_term(z, offset, latent_dim=latent_dim)
        return z @ phi11.T + offset
    if control.ndim == 2:
        return z @ phi11.T + control @ phi12.T
    msg = (
        "control input must have shape (control_dim,) or "
        f"(num_nodes, control_dim), got {tuple(control.shape)}"
    )
    raise ValueError(msg)


def advance_uncontrolled_fixed(
    z: Tensor,
    generator: Tensor,
    delta_t: Tensor,
) -> Tensor:
    """Uncontrolled advance ``z @ exp(L · Δt).T`` for a fixed generator.

    Parameters
    ----------
    z : Tensor
        Latent states.
    generator : Tensor
        Fixed generator ``L``.
    delta_t : Tensor
        Integration interval.

    Returns
    -------
    Tensor
        Advanced latents.
    """
    transition = torch.linalg.matrix_exp(generator * delta_t)
    return z @ transition.T


def advance_controlled(
    z: Tensor,
    delta_t: Tensor,
    control: Tensor,
    *,
    generator: Tensor,
    control_matrix: Tensor,
    control_mode: ControlMode,
    latent_dim: int,
    coupling: Tensor | None,
    parameterization: Parameterization,
) -> Tensor:
    """Advance with Van Loan block-matrix exponential integration.

    For bilinear mode, uses ``L_eff = L + sum_i u_i N_i`` (global) or a
    per-node ``L_eff`` when ``control`` is per-node. For
    ``auxiliary_spectral``, freezes ``L(z)`` at the step start (locally
    linear) via :func:`~koopman_graph.operators.auxiliary_spectral.advance_controlled`.

    Parameters
    ----------
    z : Tensor
        Latent states.
    delta_t : Tensor
        Integration interval.
    control : Tensor
        Piecewise-constant control over ``[0, Δt]``.
    generator : Tensor
        Fixed ``L`` or frozen ``L(z)`` for auxiliary spectral.
    control_matrix : Tensor
        Control matrix ``B``.
    control_mode : {"additive", "bilinear"}
        Control coupling mode.
    latent_dim : int
        Trailing latent dimension.
    coupling : Tensor or None
        Bilinear coupling stack when ``control_mode="bilinear"``.
    parameterization : Parameterization
        String-mode generator parameterization.

    Returns
    -------
    Tensor
        Controlled advanced latent states.
    """

    def _van_loan_step(
        state: Tensor,
        dt: Tensor,
        u: Tensor,
        *,
        generator: Tensor,
    ) -> Tensor:
        """Advance one controlled Van Loan step for a fixed generator.

        Returns
        -------
        Tensor
            Latent state after the controlled interval.
        """
        return advance_van_loan(
            state,
            dt,
            u,
            generator=generator,
            control_matrix=control_matrix,
            latent_dim=latent_dim,
        )

    if parameterization == "auxiliary_spectral":
        return advance_auxiliary_controlled(
            z,
            delta_t,
            control,
            generator=generator,
            control_mode=control_mode,
            latent_dim=latent_dim,
            coupling=coupling,
            advance_van_loan=_van_loan_step,
        )

    if control_mode == "additive":
        return _van_loan_step(z, delta_t, control, generator=generator)

    if coupling is None:
        msg = "bilinear coupling tensor is required for bilinear advance"
        raise ValueError(msg)

    if control.ndim == 1:
        l_eff = effective_bilinear_matrix(generator, control, coupling)
        return _van_loan_step(z, delta_t, control, generator=l_eff)

    if control.ndim == 2:
        if z.ndim < 2 or z.shape[-2] != control.shape[0]:
            msg = (
                "per-node bilinear control requires z with a matching "
                f"node axis, got z={tuple(z.shape)}, u={tuple(control.shape)}"
            )
            raise ValueError(msg)
        advanced = torch.empty_like(z)
        for node_idx in range(control.shape[0]):
            l_eff = effective_bilinear_matrix(
                generator,
                control[node_idx],
                coupling,
            )
            node_z = z[..., node_idx : node_idx + 1, :]
            node_u = control[node_idx]
            node_next = _van_loan_step(
                node_z,
                delta_t,
                node_u,
                generator=l_eff,
            )
            advanced[..., node_idx : node_idx + 1, :] = node_next
        return advanced

    msg = (
        "control input must have shape (control_dim,) or "
        f"(num_nodes, control_dim), got {tuple(control.shape)}"
    )
    raise ValueError(msg)


def advance_interval(
    z: Tensor,
    delta_t: float | Tensor | None,
    control: Tensor | None,
    *,
    latent_dim: int,
    control_dim: int,
    control_mode: ControlMode,
    parameterization: Parameterization,
    generator: Tensor,
    control_matrix: Tensor | None = None,
    coupling: Tensor | None = None,
) -> Tensor:
    """Advance latent states over a continuous-time interval ``Δt``.

    Owns validation and uncontrolled / controlled branching used by
    :meth:`~koopman_graph.operators.continuous.ContinuousKoopmanOperator.advance`.
    The orchestrator resolves ``generator`` (fixed ``L`` or frozen
    ``L(z)``) before calling this helper.

    Parameters
    ----------
    z : Tensor
        Latent states with shape ``(..., latent_dim)``.
    delta_t : float, Tensor, or None
        Integration interval. ``0`` returns ``z`` unchanged. Must not be
        ``None``.
    control : Tensor or None
        Piecewise-constant control over ``[0, Δt]``.
    latent_dim : int
        Trailing latent dimension.
    control_dim : int
        Control input dimension (``0`` disables control).
    control_mode : {"additive", "bilinear"}
        Control coupling mode.
    parameterization : Parameterization
        Active generator parameterization.
    generator : Tensor
        Fixed ``L`` or state-dependent ``L(z)`` for the interval.
    control_matrix : Tensor or None, optional
        Control matrix ``B`` when ``control_dim > 0``.
    coupling : Tensor or None, optional
        Bilinear coupling stack when ``control_mode="bilinear"``.

    Returns
    -------
    Tensor
        Advanced latent states with the same shape as ``z``.

    Raises
    ------
    ValueError
        If ``delta_t`` is ``None``, the trailing dimension of ``z`` does not
        match ``latent_dim``, or controls are invalid.
    """
    if delta_t is None:
        msg = "delta_t is required for ContinuousKoopmanOperator.advance"
        raise ValueError(msg)
    if z.shape[-1] != latent_dim:
        msg = f"Expected trailing dimension {latent_dim}, got shape {tuple(z.shape)}"
        raise ValueError(msg)

    delta = torch.as_tensor(delta_t, dtype=z.dtype, device=z.device)
    if torch.isclose(delta, torch.zeros((), device=z.device, dtype=z.dtype)).item():
        if control is not None and control_dim > 0:
            msg = "control input is ignored when delta_t is zero"
            raise ValueError(msg)
        return z

    if control_dim == 0:
        if control is not None:
            msg = "control input provided to an uncontrolled operator"
            raise ValueError(msg)
        if parameterization == "auxiliary_spectral":
            return advance_auxiliary_uncontrolled(z, generator, delta)
        return advance_uncontrolled_fixed(z, generator, delta)

    if control is None:
        msg = "control input is required when control_dim > 0"
        raise ValueError(msg)
    if control_matrix is None:
        msg = "control_matrix is required when control_dim > 0"
        raise ValueError(msg)

    return advance_controlled(
        z,
        delta,
        control,
        generator=generator,
        control_matrix=control_matrix,
        control_mode=control_mode,
        latent_dim=latent_dim,
        coupling=coupling,
        parameterization=parameterization,
    )


def subtract_additive_control_offset(
    z: Tensor,
    delta_t: Tensor,
    control: Tensor,
    *,
    generator: Tensor,
    control_matrix: Tensor,
    latent_dim: int,
) -> Tensor:
    """Remove the additive Van Loan control offset before inverse free advance.

    Parameters
    ----------
    z : Tensor
        Latents after the forward controlled interval.
    delta_t : Tensor
        Forward integration interval.
    control : Tensor
        Control applied during the forward interval.
    generator : Tensor
        Fixed generator ``L`` used for Van Loan factors.
    control_matrix : Tensor
        Control matrix ``B``.
    latent_dim : int
        Trailing latent dimension.

    Returns
    -------
    Tensor
        Latents with the control offset subtracted.
    """
    _, phi12 = van_loan_factors(generator, control_matrix, delta_t)
    if control.ndim == 1:
        offset = control @ phi12.T
        if z.ndim > 1:
            offset = broadcast_control_term(z, offset, latent_dim=latent_dim)
        return z - offset
    return z - control @ phi12.T


def inverse_advance_bilinear(
    z: Tensor,
    delta_t: Tensor,
    control: Tensor,
    *,
    generator: Tensor,
    control_matrix: Tensor,
    coupling: Tensor,
    latent_dim: int,
) -> Tensor:
    """Invert a bilinear continuous step under piecewise-constant ``u``.

    Parameters
    ----------
    z : Tensor
        Latents after the forward interval.
    delta_t : Tensor
        Forward integration interval.
    control : Tensor
        Control applied during the interval.
    generator : Tensor
        Nominal generator ``L`` (before bilinear effective update).
    control_matrix : Tensor
        Control matrix ``B``.
    coupling : Tensor
        Bilinear coupling stack ``N``.
    latent_dim : int
        Trailing latent dimension.

    Returns
    -------
    Tensor
        Recovered latents before the interval.
    """

    def _invert_one(state: Tensor, u: Tensor, effective: Tensor) -> Tensor:
        """Invert one bilinear Van Loan step for a fixed effective generator.

        Returns
        -------
        Tensor
            Latent state before the controlled interval.
        """
        phi11, phi12 = van_loan_factors(effective, control_matrix, delta_t)
        if u.ndim == 1:
            offset = u @ phi12.T
            if state.ndim > 1:
                offset = broadcast_control_term(state, offset, latent_dim=latent_dim)
            adjusted = state - offset
        else:
            adjusted = state - u @ phi12.T
        try:
            inverse_phi = torch.linalg.inv(phi11)
        except RuntimeError:
            inverse_phi = torch.linalg.pinv(phi11)
        return adjusted @ inverse_phi.T

    if control.ndim == 1:
        l_eff = effective_bilinear_matrix(generator, control, coupling)
        return _invert_one(z, control, l_eff)

    if control.ndim == 2:
        if z.ndim < 2 or z.shape[-2] != control.shape[0]:
            msg = (
                "per-node bilinear inverse requires matching node axes, "
                f"got z={tuple(z.shape)}, u={tuple(control.shape)}"
            )
            raise ValueError(msg)
        recovered = torch.empty_like(z)
        for node_idx in range(control.shape[0]):
            l_eff = effective_bilinear_matrix(
                generator,
                control[node_idx],
                coupling,
            )
            recovered[..., node_idx : node_idx + 1, :] = _invert_one(
                z[..., node_idx : node_idx + 1, :],
                control[node_idx],
                l_eff,
            )
        return recovered

    msg = (
        "control input must have shape (control_dim,) or "
        f"(num_nodes, control_dim), got {tuple(control.shape)}"
    )
    raise ValueError(msg)


def inverse_advance_interval(
    z: Tensor,
    delta_t: float | Tensor | None,
    control: Tensor | None,
    *,
    latent_dim: int,
    control_dim: int,
    control_mode: ControlMode,
    parameterization: Parameterization,
    generator: Tensor,
    control_matrix: Tensor | None = None,
    coupling: Tensor | None = None,
) -> Tensor:
    """Recover the previous latent state before advancing over ``Δt``.

    Owns validation and uncontrolled / controlled inverse branching used by
    :meth:`~koopman_graph.operators.continuous.ContinuousKoopmanOperator.inverse_advance`.
    The orchestrator resolves ``generator`` before calling this helper.

    Parameters
    ----------
    z : Tensor
        Latent states after advancing over ``Δt``.
    delta_t : float, Tensor, or None
        Integration interval. Must not be ``None``.
    control : Tensor or None
        Control applied during the forward interval.
    latent_dim : int
        Trailing latent dimension.
    control_dim : int
        Control input dimension.
    control_mode : {"additive", "bilinear"}
        Control coupling mode.
    parameterization : Parameterization
        Active generator parameterization.
    generator : Tensor
        Fixed ``L`` or state-dependent ``L(z)`` for the inverse step.
    control_matrix : Tensor or None, optional
        Control matrix ``B`` when ``control_dim > 0``.
    coupling : Tensor or None, optional
        Bilinear coupling stack when ``control_mode="bilinear"``.

    Returns
    -------
    Tensor
        Recovered latent states.

    Raises
    ------
    ValueError
        If ``delta_t`` is ``None`` or controls are missing when required.
    """
    if delta_t is None:
        msg = "delta_t is required for ContinuousKoopmanOperator.inverse_advance"
        raise ValueError(msg)
    if control_dim > 0 and control is None:
        msg = "control input is required when control_dim > 0"
        raise ValueError(msg)

    delta = torch.as_tensor(delta_t, dtype=z.dtype, device=z.device)
    if parameterization == "auxiliary_spectral":
        return inverse_advance_auxiliary(
            z,
            delta,
            control,
            generator=generator,
            control_dim=control_dim,
            control_mode=control_mode,
            latent_dim=latent_dim,
            control_matrix=control_matrix,
            coupling=coupling,
        )

    adjusted = z
    if control_dim > 0:
        assert control is not None
        if control_matrix is None:
            msg = "control_matrix is required when control_dim > 0"
            raise ValueError(msg)
        if control_mode == "bilinear":
            if coupling is None:
                msg = "bilinear coupling tensor is required for bilinear inverse"
                raise ValueError(msg)
            return inverse_advance_bilinear(
                z,
                delta,
                control,
                generator=generator,
                control_matrix=control_matrix,
                coupling=coupling,
                latent_dim=latent_dim,
            )

        adjusted = subtract_additive_control_offset(
            z,
            delta,
            control,
            generator=generator,
            control_matrix=control_matrix,
            latent_dim=latent_dim,
        )

    return advance_uncontrolled_fixed(adjusted, generator, -delta)
