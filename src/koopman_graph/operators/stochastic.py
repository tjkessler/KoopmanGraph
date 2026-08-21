"""Diagonal process-noise helpers for ``dynamics_mode='stochastic'``.

Opt-in discrete maps use :math:`z \\mapsto Kz + \\varepsilon` with a learned
diagonal covariance on stacked latents
:math:`Q = I_N \\otimes \\mathrm{diag}(\\sigma^2)`, where
:math:`\\sigma = \\mathrm{softplus}(\\texttt{process_log_std})`. Noise is applied
in :meth:`~koopman_graph.operators.contract.KoopmanOperatorContract.advance`
after the linear map; :meth:`forward` stays deterministic. Distinct from
the opt-in :class:`~koopman_graph.operators.DriftDiffusionKoopman`
Euler–Maruyama / Yosida stepper.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def softplus_inverse(value: float) -> float:
    """Inverse of ``softplus`` for positive ``value``.

    Parameters
    ----------
    value : float
        Positive softplus output.

    Returns
    -------
    float
        Pre-image ``x`` with ``softplus(x) == value``.
    """
    if value <= 0.0:
        msg = f"softplus_inverse requires a positive value, got {value}"
        raise ValueError(msg)
    return float(math.log(math.expm1(value)))


def attach_process_noise(
    module: nn.Module,
    *,
    latent_dim: int,
    init_std: float = 1e-3,
) -> nn.Module:
    """Register learned diagonal process-noise parameters on ``module``.

    Parameters
    ----------
    module : nn.Module
        Discrete Koopman operator that will apply noise in ``advance``.
    latent_dim : int
        Latent width ``d`` (shared across nodes).
    init_std : float, optional
        Initial ``softplus(process_log_std)`` scale. Default ``1e-3``.

    Returns
    -------
    nn.Module
        The same ``module`` (mutated in place) for fluent factory use.

    Raises
    ------
    ValueError
        If ``latent_dim`` or ``init_std`` is invalid.
    """
    if latent_dim < 1:
        msg = f"latent_dim must be positive, got {latent_dim}"
        raise ValueError(msg)
    if init_std <= 0.0:
        msg = f"init_std must be positive, got {init_std}"
        raise ValueError(msg)

    log_std = softplus_inverse(init_std)
    module.process_log_std = nn.Parameter(torch.full((latent_dim,), log_std))
    module.process_noise_scale = 1.0
    module.stochastic = True
    return module


def process_noise_std(module: nn.Module) -> Tensor:
    """Return per-latent standard deviations ``softplus(process_log_std)``.

    Parameters
    ----------
    module : nn.Module
        Operator with ``process_log_std`` attached.

    Returns
    -------
    Tensor
        Positive std vector with shape ``(latent_dim,)``.
    """
    log_std = getattr(module, "process_log_std", None)
    if log_std is None:
        msg = f"{type(module).__name__} has no process_log_std parameter"
        raise AttributeError(msg)
    return torch.nn.functional.softplus(log_std)


def apply_process_noise(
    z: Tensor,
    module: nn.Module,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Add diagonal Gaussian process noise to advanced latents.

    Parameters
    ----------
    z : Tensor
        Post-linear latents with trailing dimension ``latent_dim``.
    module : nn.Module
        Operator providing ``process_log_std`` and ``process_noise_scale``.
    generator : torch.Generator or None, optional
        Optional RNG for reproducible draws.

    Returns
    -------
    Tensor
        Noisy latents with the same shape as ``z``. When
        ``process_noise_scale == 0``, returns ``z`` unchanged.
    """
    scale = float(getattr(module, "process_noise_scale", 1.0))
    if scale == 0.0:
        return z
    std = process_noise_std(module).to(device=z.device, dtype=z.dtype)
    if z.shape[-1] != std.numel():
        msg = (
            "process_log_std width must match latent trailing dimension; "
            f"got std={tuple(std.shape)} vs z={tuple(z.shape)}"
        )
        raise ValueError(msg)
    eps = torch.randn(z.shape, dtype=z.dtype, device=z.device, generator=generator)
    return z + scale * eps * std


def maybe_apply_process_noise(
    z: Tensor,
    module: nn.Module,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Apply process noise when ``module.stochastic`` is true.

    Parameters
    ----------
    z : Tensor
        Post-linear latents.
    module : nn.Module
        Operator that may have stochastic process noise attached.
    generator : torch.Generator or None, optional
        Optional RNG.

    Returns
    -------
    Tensor
        Possibly noisy latents.
    """
    if not getattr(module, "stochastic", False):
        return z
    return apply_process_noise(z, module, generator=generator)


def diagonal_process_covariance(log_std: Tensor, num_nodes: int) -> Tensor:
    """Build stacked-state process covariance ``Q = I_N ⊗ diag(σ²)``.

    Matches the node-major stacking used by
    :func:`~koopman_graph.uq.dense_nodewise_transition`.

    Parameters
    ----------
    log_std : Tensor
        Log-std vector with shape ``(d,)`` (pre-softplus).
    num_nodes : int
        Number of nodes ``N``.

    Returns
    -------
    Tensor
        Dense covariance with shape ``(N·d, N·d)``.

    Raises
    ------
    ValueError
        If ``num_nodes`` is invalid or ``log_std`` is not 1-D.
    """
    if num_nodes < 1:
        msg = f"num_nodes must be positive, got {num_nodes}"
        raise ValueError(msg)
    if log_std.ndim != 1:
        msg = f"log_std must be 1-D with shape (d,), got {tuple(log_std.shape)}"
        raise ValueError(msg)
    sigma = torch.nn.functional.softplus(log_std)
    diag = (sigma * sigma).repeat(num_nodes)
    return torch.diag(diag)
