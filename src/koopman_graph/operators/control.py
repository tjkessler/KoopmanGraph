"""Shared additive / bilinear control helpers for Koopman operators."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

ControlMode = Literal["additive", "bilinear"]


def validate_control_mode(
    *,
    control_dim: int,
    control_mode: ControlMode,
    bilinear_rank: int | None,
    latent_dim: int,
) -> None:
    """Validate control-mode settings for operator construction.

    Parameters
    ----------
    control_dim : int
        Exogenous control dimension (``0`` disables control).
    control_mode : {"additive", "bilinear"}
        Control coupling mode.
    bilinear_rank : int or None
        Optional low-rank factor size for bilinear ``N_i``. ``None`` means
        full-rank ``N_i``.
    latent_dim : int
        Latent state dimension.

    Raises
    ------
    ValueError
        If the mode / rank combination is invalid.
    """
    if control_mode not in {"additive", "bilinear"}:
        msg = f"control_mode must be 'additive' or 'bilinear', got {control_mode!r}"
        raise ValueError(msg)
    if control_mode == "bilinear" and control_dim <= 0:
        msg = "control_mode='bilinear' requires control_dim > 0"
        raise ValueError(msg)
    if control_mode == "additive" and bilinear_rank is not None:
        msg = "bilinear_rank requires control_mode='bilinear'"
        raise ValueError(msg)
    if bilinear_rank is not None:
        if bilinear_rank < 1:
            msg = f"bilinear_rank must be >= 1, got {bilinear_rank}"
            raise ValueError(msg)
        if bilinear_rank > latent_dim:
            msg = (
                f"bilinear_rank ({bilinear_rank}) cannot exceed "
                f"latent_dim ({latent_dim})"
            )
            raise ValueError(msg)


def allocate_bilinear_parameters(
    module: nn.Module,
    *,
    control_dim: int,
    latent_dim: int,
    bilinear_rank: int | None,
) -> None:
    """Allocate learnable bilinear factors on ``module``.

    Full-rank mode registers ``N`` with shape
    ``(control_dim, latent_dim, latent_dim)``. Low-rank mode registers
    ``P`` / ``Q`` with shape ``(control_dim, latent_dim, bilinear_rank)``
    such that ``N_i = P_i @ Q_i.T``.

    Parameters
    ----------
    module : nn.Module
        Operator module that will own the parameters.
    control_dim : int
        Number of control channels.
    latent_dim : int
        Latent dimension.
    bilinear_rank : int or None
        Low-rank size, or ``None`` for full-rank ``N``.
    """
    if bilinear_rank is None:
        module.register_parameter(
            "N",
            nn.Parameter(torch.empty(control_dim, latent_dim, latent_dim)),
        )
    else:
        module.register_parameter(
            "P",
            nn.Parameter(torch.empty(control_dim, latent_dim, bilinear_rank)),
        )
        module.register_parameter(
            "Q",
            nn.Parameter(torch.empty(control_dim, latent_dim, bilinear_rank)),
        )


def reset_bilinear_parameters(module: nn.Module) -> None:
    """Zero-initialize bilinear factors on ``module`` when present.

    Parameters
    ----------
    module : nn.Module
        Operator that may own ``N`` or ``P`` / ``Q``.
    """
    bilinear_n = getattr(module, "N", None)
    if isinstance(bilinear_n, nn.Parameter):
        nn.init.zeros_(bilinear_n)
        return
    bilinear_p = getattr(module, "P", None)
    bilinear_q = getattr(module, "Q", None)
    if isinstance(bilinear_p, nn.Parameter) and isinstance(bilinear_q, nn.Parameter):
        nn.init.zeros_(bilinear_p)
        nn.init.zeros_(bilinear_q)


def bilinear_coupling_tensor(module: nn.Module) -> Tensor:
    """Assemble full bilinear couplings ``N`` with shape ``(C, D, D)``.

    Parameters
    ----------
    module : nn.Module
        Operator owning full-rank ``N`` or low-rank ``P`` / ``Q``.

    Returns
    -------
    Tensor
        Couplings with shape ``(control_dim, latent_dim, latent_dim)``.

    Raises
    ------
    AttributeError
        If neither full-rank nor low-rank factors are present.
    """
    bilinear_n = getattr(module, "N", None)
    if isinstance(bilinear_n, Tensor):
        return bilinear_n
    bilinear_p = getattr(module, "P", None)
    bilinear_q = getattr(module, "Q", None)
    if isinstance(bilinear_p, Tensor) and isinstance(bilinear_q, Tensor):
        # N_i = P_i @ Q_i.T  →  (D, R) @ (R, D) = (D, D)
        return torch.einsum("cdr,cer->cde", bilinear_p, bilinear_q)
    msg = f"{type(module).__name__} has no bilinear factors (N or P/Q)"
    raise AttributeError(msg)


def bilinear_state_control_term(
    z: Tensor,
    control: Tensor,
    coupling: Tensor,
) -> Tensor:
    """Compute ``sum_i u[..., i] * (z @ N_i.T)``.

    Parameters
    ----------
    z : Tensor
        Latent states with shape ``(..., latent_dim)``.
    control : Tensor
        Global control ``(control_dim,)`` or per-node
        ``(num_nodes, control_dim)`` aligned with ``z``'s node axis.
    coupling : Tensor
        Bilinear matrices ``N`` with shape
        ``(control_dim, latent_dim, latent_dim)``.

    Returns
    -------
    Tensor
        Bilinear contribution with the same shape as ``z``.

    Raises
    ------
    ValueError
        If ``control`` rank/shape is incompatible with ``z``.
    """
    if control.ndim == 1:
        # sum_c u_c * (z @ N_c.T) = einsum('c,...d,ced->...e', u, z, N)
        return torch.einsum("c,...d,ced->...e", control, z, coupling)

    if control.ndim == 2:
        if z.ndim < 2:
            msg = (
                "per-node bilinear control requires z with a node axis, "
                f"got shape {tuple(z.shape)}"
            )
            raise ValueError(msg)
        if control.shape[0] != z.shape[-2]:
            msg = (
                f"Per-node control has {control.shape[0]} rows, expected {z.shape[-2]}"
            )
            raise ValueError(msg)
        # Nodes along the last batch axis of z.
        return torch.einsum("nc,...nd,ced->...ne", control, z, coupling)

    msg = (
        "control input must have shape (control_dim,) for global control "
        f"or (num_nodes, control_dim) for per-node control, got {tuple(control.shape)}"
    )
    raise ValueError(msg)


def effective_bilinear_matrix(
    base: Tensor,
    control: Tensor,
    coupling: Tensor,
) -> Tensor:
    """Return ``base + sum_i u_i N_i`` for a **global** control vector.

    Parameters
    ----------
    base : Tensor
        Base operator ``K`` or generator ``L`` with shape ``(D, D)``.
    control : Tensor
        Global control with shape ``(control_dim,)``.
    coupling : Tensor
        Bilinear couplings with shape ``(control_dim, D, D)``.

    Returns
    -------
    Tensor
        Effective matrix with shape ``(D, D)``.

    Raises
    ------
    ValueError
        If ``control`` is not a 1-D global vector.
    """
    if control.ndim != 1:
        msg = (
            "effective_bilinear_matrix requires global control with shape "
            f"(control_dim,), got {tuple(control.shape)}"
        )
        raise ValueError(msg)
    return base + torch.einsum("c,cde->de", control, coupling)
