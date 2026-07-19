"""Discrete-time advance / inverse execution helpers.

Shallow sibling of :mod:`koopman_graph.operators.discrete`. Holds cohesive
controlled / uncontrolled forward steps, additive-control offset removal,
bilinear inverse, and inverse-matrix execution used by
:class:`~koopman_graph.operators.discrete.KoopmanOperator`.
Structural assembly / reset remains in
:mod:`~koopman_graph.operators.discrete_parameterizations`. Prefer importing
via ``discrete`` when writing against the discrete capability surface.
"""

from __future__ import annotations

import torch
from torch import Tensor

from koopman_graph.operators.contract import (
    Parameterization,
    cayley_orthogonal,
    safe_diagonal_inverse,
)
from koopman_graph.operators.control import (
    ControlMode,
    bilinear_state_control_term,
    broadcast_control_term,
    effective_bilinear_matrix,
    map_control_term,
)


def advance_step(
    z: Tensor,
    control: Tensor | None,
    *,
    matrix: Tensor,
    control_matrix: Tensor | None,
    control_dim: int,
    control_mode: ControlMode,
    latent_dim: int,
    coupling: Tensor | None = None,
) -> Tensor:
    """Advance latent states by one linear Koopman step.

    When ``control_dim > 0`` and ``control_mode="additive"``::

        z_next = z @ K.T + u @ B

    When ``control_mode="bilinear"``::

        z_next = z @ K.T + u @ B + sum_i u[..., i] * (z @ N_i.T)

    Parameters
    ----------
    z : Tensor
        Latent states with shape ``(..., latent_dim)``.
    control : Tensor or None
        Exogenous control input. Required when ``control_dim > 0``.
    matrix : Tensor
        Discrete operator ``K`` with shape ``(latent_dim, latent_dim)``.
    control_matrix : Tensor or None
        Control matrix ``B`` when ``control_dim > 0``.
    control_dim : int
        Control input dimension.
    control_mode : {"additive", "bilinear"}
        Control coupling mode.
    latent_dim : int
        Trailing latent dimension.
    coupling : Tensor or None, optional
        Bilinear coupling stack when ``control_mode="bilinear"``.

    Returns
    -------
    Tensor
        Advanced latent states with the same shape as ``z``.

    Raises
    ------
    ValueError
        If the trailing dimension of ``z`` does not match ``latent_dim``,
        controls are missing for a controlled operator, or ``control`` has
        an invalid shape.
    """
    if z.shape[-1] != latent_dim:
        msg = (
            f"Expected trailing dimension {latent_dim}, "
            f"got shape {tuple(z.shape)}"
        )
        raise ValueError(msg)
    z_next = z @ matrix.T
    if control_dim == 0:
        if control is not None:
            msg = "control input provided to an uncontrolled operator"
            raise ValueError(msg)
        return z_next
    if control is None:
        msg = "control input is required when control_dim > 0"
        raise ValueError(msg)
    offset = map_control_term(
        control,
        control_matrix,
        control_dim=control_dim,
        num_nodes=z.shape[-2] if z.ndim >= 2 else None,
    )
    if control.ndim == 1:
        offset = broadcast_control_term(z, offset, latent_dim=latent_dim)
    z_next = z_next + offset
    if control_mode == "bilinear":
        if coupling is None:
            msg = "bilinear coupling tensor is required for bilinear advance"
            raise ValueError(msg)
        z_next = z_next + bilinear_state_control_term(z, control, coupling)
    return z_next


def subtract_additive_control_offset(
    z: Tensor,
    control: Tensor,
    *,
    control_matrix: Tensor | None,
    control_dim: int,
    latent_dim: int,
) -> Tensor:
    """Remove the additive ``u @ B`` offset before inverse free advance.

    Parameters
    ----------
    z : Tensor
        Latents after the forward controlled step.
    control : Tensor
        Control applied during the forward step.
    control_matrix : Tensor or None
        Control matrix ``B``.
    control_dim : int
        Control input dimension.
    latent_dim : int
        Trailing latent dimension.

    Returns
    -------
    Tensor
        Latents with the additive control offset subtracted.
    """
    offset = map_control_term(
        control,
        control_matrix,
        control_dim=control_dim,
        num_nodes=z.shape[-2] if z.ndim >= 2 else None,
    )
    if control.ndim == 1:
        offset = broadcast_control_term(z, offset, latent_dim=latent_dim)
    return z - offset


def inverse_bilinear(
    adjusted: Tensor,
    control: Tensor,
    *,
    matrix: Tensor,
    coupling: Tensor,
) -> Tensor:
    """Invert a bilinear step after subtracting the additive ``u @ B`` term.

    Parameters
    ----------
    adjusted : Tensor
        ``z_next - u @ B`` with shape ``(..., latent_dim)``.
    control : Tensor
        Control that drove the forward step.
    matrix : Tensor
        Nominal discrete map ``K``.
    coupling : Tensor
        Bilinear coupling stack ``N``.

    Returns
    -------
    Tensor
        Recovered ``z_t``.
    """
    if control.ndim == 1:
        k_eff = effective_bilinear_matrix(matrix, control, coupling)
        try:
            inverse_k = torch.linalg.inv(k_eff)
        except RuntimeError:
            inverse_k = torch.linalg.pinv(k_eff)
        return adjusted @ inverse_k.T

    if control.ndim == 2:
        if adjusted.ndim < 2 or adjusted.shape[-2] != control.shape[0]:
            msg = (
                "per-node bilinear inverse requires adjusted latents with "
                f"node axis matching control rows, got {tuple(adjusted.shape)}"
            )
            raise ValueError(msg)
        recovered = torch.empty_like(adjusted)
        for node_idx in range(control.shape[0]):
            k_eff = effective_bilinear_matrix(
                matrix,
                control[node_idx],
                coupling,
            )
            try:
                inverse_k = torch.linalg.inv(k_eff)
            except RuntimeError:
                inverse_k = torch.linalg.pinv(k_eff)
            recovered[..., node_idx, :] = adjusted[..., node_idx, :] @ inverse_k.T
        return recovered

    msg = (
        "control input must have shape (control_dim,) or "
        f"(num_nodes, control_dim), got {tuple(control.shape)}"
    )
    raise ValueError(msg)


def inverse_step(
    z: Tensor,
    *,
    control: Tensor | None,
    matrix: Tensor,
    control_matrix: Tensor | None,
    control_dim: int,
    control_mode: ControlMode,
    latent_dim: int,
    coupling: Tensor | None,
    inverse_matrix: Tensor | None,
) -> Tensor:
    """Apply one inverse Koopman step to recover the previous latent state.

    Parameters
    ----------
    z : Tensor
        Latent states at time ``t+1`` with shape ``(..., latent_dim)``.
    control : Tensor or None
        Control input that drove the forward transition. Required when
        ``control_dim > 0``.
    matrix : Tensor
        Nominal discrete map ``K`` (used for bilinear effective inverses).
    control_matrix : Tensor or None
        Control matrix ``B`` when ``control_dim > 0``.
    control_dim : int
        Control input dimension.
    control_mode : {"additive", "bilinear"}
        Control coupling mode.
    latent_dim : int
        Trailing latent dimension.
    coupling : Tensor or None
        Bilinear coupling stack when ``control_mode="bilinear"``.
    inverse_matrix : Tensor or None
        Precomputed ``K^{-1}`` for additive / uncontrolled inversion.
        Unused for bilinear mode.

    Returns
    -------
    Tensor
        Recovered latent states at time ``t``, same shape as ``z``.
    """
    adjusted = z
    if control_dim > 0:
        if control is None:
            msg = "control input is required when control_dim > 0"
            raise ValueError(msg)
        adjusted = subtract_additive_control_offset(
            z,
            control,
            control_matrix=control_matrix,
            control_dim=control_dim,
            latent_dim=latent_dim,
        )
        if control_mode == "bilinear":
            if coupling is None:
                msg = "bilinear coupling tensor is required for bilinear inverse"
                raise ValueError(msg)
            return inverse_bilinear(
                adjusted,
                control,
                matrix=matrix,
                coupling=coupling,
            )

    if inverse_matrix is None:
        msg = "inverse_matrix is required for additive / uncontrolled inverse"
        raise ValueError(msg)
    return adjusted @ inverse_matrix.T


def dense_inverse_or_pinv(matrix: Tensor) -> Tensor:
    """Return ``K^{-1}`` or the pseudo-inverse when ``K`` is singular.

    Parameters
    ----------
    matrix : Tensor
        Dense operator ``K``.

    Returns
    -------
    Tensor
        Inverse (or pseudo-inverse) with the same shape as ``matrix``.
    """
    try:
        return torch.linalg.inv(matrix)
    except RuntimeError:
        return torch.linalg.pinv(matrix)


def inverse_odo_matrix(
    orthogonal_left: Tensor,
    orthogonal_right: Tensor,
    diagonal: Tensor,
) -> Tensor:
    """Return the ODO inverse ``O2 @ Λ^{-1} @ O1.T``.

    Parameters
    ----------
    orthogonal_left : Tensor
        Left orthogonal factor ``O1``.
    orthogonal_right : Tensor
        Right orthogonal factor ``O2``.
    diagonal : Tensor
        ODO diagonal factor as a diagonal matrix (or 1-D entries).

    Returns
    -------
    Tensor
        Inverse operator matrix.
    """
    # ``safe_diagonal_inverse`` expects 1-D entries; ODO assembly exposes a
    # diagonal matrix via ``bounded_diagonal``.
    diag_values = torch.diag(diagonal) if diagonal.ndim == 2 else diagonal
    return orthogonal_right @ safe_diagonal_inverse(diag_values) @ orthogonal_left.T


def inverse_schur_matrix(cayley_q: Tensor, triangular: Tensor) -> Tensor:
    """Return the Schur inverse ``Q @ T^{-1} @ Q.T``.

    Parameters
    ----------
    cayley_q : Tensor
        Cayley parameter for the orthogonal factor.
    triangular : Tensor
        Upper-triangular Schur factor.

    Returns
    -------
    Tensor
        Inverse operator matrix.
    """
    q = cayley_orthogonal(cayley_q)
    triangular_inv = torch.linalg.inv(triangular)
    return q @ triangular_inv @ q.T


def inverse_dissipative_matrix(generator: Tensor) -> Tensor:
    """Return the dissipative inverse ``exp(L)`` for ``K = exp(-L)``.

    Parameters
    ----------
    generator : Tensor
        Dissipative generator used by the forward assembly.

    Returns
    -------
    Tensor
        Inverse operator matrix.
    """
    return torch.linalg.matrix_exp(generator)


def inverse_lyapunov_matrix(cayley_q: Tensor, diagonal: Tensor) -> Tensor:
    """Return the Lyapunov inverse ``Q @ Λ^{-1} @ Q.T``.

    Parameters
    ----------
    cayley_q : Tensor
        Cayley parameter for the orthogonal factor.
    diagonal : Tensor
        Bounded Lyapunov diagonal.

    Returns
    -------
    Tensor
        Inverse operator matrix.
    """
    q = cayley_orthogonal(cayley_q)
    return q @ safe_diagonal_inverse(diagonal) @ q.T


def inverse_matrix_for_parameterization(
    parameterization: Parameterization,
    *,
    dense_matrix: Tensor | None = None,
    inverse_matrix: Tensor | None = None,
    odo_left: Tensor | None = None,
    odo_right: Tensor | None = None,
    odo_diagonal: Tensor | None = None,
    schur_cayley_q: Tensor | None = None,
    schur_triangular: Tensor | None = None,
    dissipative_generator: Tensor | None = None,
    lyapunov_cayley_q: Tensor | None = None,
    lyapunov_diagonal: Tensor | None = None,
) -> Tensor:
    """Build ``K^{-1}`` for the active discrete parameterization.

    Parameters
    ----------
    parameterization : Parameterization
        String-mode discrete parameterization.
    dense_matrix : Tensor or None, optional
        Dense ``K`` when ``parameterization="dense"``.
    inverse_matrix : Tensor or None, optional
        Precomputed dense inverse override.
    odo_left, odo_right, odo_diagonal : Tensor or None, optional
        ODO factors.
    schur_cayley_q, schur_triangular : Tensor or None, optional
        Schur factors.
    dissipative_generator : Tensor or None, optional
        Dissipative generator.
    lyapunov_cayley_q, lyapunov_diagonal : Tensor or None, optional
        Lyapunov factors.

    Returns
    -------
    Tensor
        Inverse operator matrix with shape ``(latent_dim, latent_dim)``.
    """
    if parameterization == "dense":
        if inverse_matrix is not None:
            return inverse_matrix
        if dense_matrix is None:
            msg = "dense_matrix is required for dense inverse construction"
            raise ValueError(msg)
        return dense_inverse_or_pinv(dense_matrix)
    if parameterization == "odo":
        if odo_left is None or odo_right is None or odo_diagonal is None:
            msg = "ODO factors are required for ODO inverse construction"
            raise ValueError(msg)
        return inverse_odo_matrix(odo_left, odo_right, odo_diagonal)
    if parameterization == "schur":
        if schur_cayley_q is None or schur_triangular is None:
            msg = "Schur factors are required for Schur inverse construction"
            raise ValueError(msg)
        return inverse_schur_matrix(schur_cayley_q, schur_triangular)
    if parameterization == "dissipative":
        if dissipative_generator is None:
            msg = "dissipative generator is required for dissipative inverse"
            raise ValueError(msg)
        return inverse_dissipative_matrix(dissipative_generator)
    if parameterization == "lyapunov":
        if lyapunov_cayley_q is None or lyapunov_diagonal is None:
            msg = "Lyapunov factors are required for Lyapunov inverse"
            raise ValueError(msg)
        return inverse_lyapunov_matrix(lyapunov_cayley_q, lyapunov_diagonal)
    msg = f"Unknown parameterization: {parameterization!r}"
    raise ValueError(msg)
