"""Finite-sample identification report helpers.

Populate one-step latent mean squared error (MSE), a short latent
rollout MSE, and the spectral radius of identified ``K``. When
``gate_resdmd=True``, also fill the finite-dictionary ResDMD
``spectral`` block. Invariance, reconstruction, and closure stay
unset. These scalars are not Haseli–Cortés, certified ResDMD, or
stability certificates.

This module must not import :mod:`koopman_graph.training` or
:mod:`koopman_graph.model`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor

from koopman_graph.identification.protocol import LatentPairs, OperatorSnapshot
from koopman_graph.identification.report import (
    IdentificationReport,
    MetricBlock,
    SpectralReliabilityBlock,
    StabilityBlock,
)

__all__ = [
    "DEFAULT_IDENTIFICATION_ROLLOUT_HORIZON",
    "build_identification_report",
]

DEFAULT_IDENTIFICATION_ROLLOUT_HORIZON = 5


def _flatten_trailing(z_t: Tensor, z_next: Tensor) -> tuple[Tensor, Tensor]:
    """Flatten encodings to ``(n_samples, d)``.

    Parameters
    ----------
    z_t : Tensor
        Source encodings.
    z_next : Tensor
        Target encodings, same shape as ``z_t``.

    Returns
    -------
    tuple of Tensor
        Flattened source and target.
    """
    width = z_t.shape[-1]
    return z_t.reshape(-1, width), z_next.reshape(-1, width)


def _one_step_mse(encodings: LatentPairs, matrix: Tensor) -> MetricBlock:
    """Latent one-step mean squared error under ``z @ K.T``.

    Parameters
    ----------
    encodings : LatentPairs
        Consecutive encodings.
    matrix : Tensor
        Identified ``K`` with shape ``(d, d)``.

    Returns
    -------
    MetricBlock
        MSE over tensor entries and the entry count.
    """
    left, right = _flatten_trailing(encodings.z_t, encodings.z_next)
    residual = left @ matrix.T - right
    squared = residual.square()
    n_samples = int(squared.numel())
    mse = float(squared.mean().item()) if n_samples else 0.0
    return MetricBlock(mse=mse, n_samples=n_samples)


def _rollout_mse(
    encodings: LatentPairs,
    matrix: Tensor,
    horizon: int,
) -> MetricBlock:
    """Latent multi-step MSE along the leading time axis.

    Restores a length-``T`` trajectory as ``[z_t[0], z_next[0], …]`` and
    rolls ``z_t[0]`` for ``min(horizon, T-1)`` steps. For 2-D pairs the
    sample axis is treated as time (one vector per row); prefer 3-D
    ``(T-1, N, d)`` so nodes stay aligned.

    Parameters
    ----------
    encodings : LatentPairs
        Consecutive encodings.
    matrix : Tensor
        Identified ``K``.
    horizon : int
        Maximum rollout steps.

    Returns
    -------
    MetricBlock
        Mean squared error over rollout entries, or empty when ``T < 2``.
    """
    stacked = torch.cat([encodings.z_t[:1], encodings.z_next], dim=0)
    n_times = stacked.shape[0]
    steps = min(horizon, n_times - 1)
    if steps < 1:
        return MetricBlock()
    predicted = stacked[0]
    total = stacked.new_zeros(())
    n_samples = 0
    for step in range(1, steps + 1):
        predicted = predicted @ matrix.T
        err = (predicted - stacked[step]).square()
        total = total + err.sum()
        n_samples += int(err.numel())
    mse = float((total / n_samples).item()) if n_samples else 0.0
    return MetricBlock(mse=mse, n_samples=n_samples)


def _mean_metric(blocks: Sequence[MetricBlock]) -> MetricBlock:
    """Weighted mean of MSE blocks by ``n_samples``.

    Parameters
    ----------
    blocks : sequence of MetricBlock
        Per-trajectory summaries. Empty or unset blocks are skipped.

    Returns
    -------
    MetricBlock
        Pooled MSE, or an empty block when nothing was scored.
    """
    weighted = 0.0
    n_total = 0
    for block in blocks:
        if block.mse is None or block.n_samples is None or block.n_samples == 0:
            continue
        weighted += block.mse * block.n_samples
        n_total += block.n_samples
    if n_total == 0:
        return MetricBlock()
    return MetricBlock(mse=weighted / n_total, n_samples=n_total)


def _spectral_reliability(
    blocks: Sequence[LatentPairs],
    *,
    tolerance: float,
) -> SpectralReliabilityBlock:
    """Finite-dictionary ResDMD max residual on flattened latent pairs.

    Parameters
    ----------
    blocks : sequence of LatentPairs
        Consecutive encodings (pooled along the sample axis).
    tolerance : float
        Pollution cutoff (same units as ResDMD residuals).

    Returns
    -------
    SpectralReliabilityBlock
        ``residual_max`` and ``polluted`` (``residual_max > tolerance``).
    """
    from koopman_graph.analysis.resdmd import resdmd

    left_parts = []
    right_parts = []
    for block in blocks:
        left, right = _flatten_trailing(block.z_t, block.z_next)
        left_parts.append(left)
        right_parts.append(right)
    psi0 = torch.cat(left_parts, dim=0)
    psi1 = torch.cat(right_parts, dim=0)
    report = resdmd(psi0, psi1, tolerance=tolerance)
    residual_max = float(report.residuals.max().real.item())
    if not math.isfinite(residual_max) or residual_max < 0.0:
        msg = (
            "ResDMD residual_max must be a finite non-negative float, "
            f"got {residual_max!r}"
        )
        raise ValueError(msg)
    return SpectralReliabilityBlock(
        residual_max=residual_max,
        polluted=residual_max > tolerance,
    )


def build_identification_report(
    encodings: LatentPairs | Sequence[LatentPairs],
    snapshot: OperatorSnapshot,
    *,
    rollout_horizon: int = DEFAULT_IDENTIFICATION_ROLLOUT_HORIZON,
    gate_resdmd: bool = False,
    residual_tolerance: float = 1e-2,
) -> IdentificationReport:
    """Fill one-step, short rollout, and spectral-radius slots.

    Reconstruction, closure, and invariance stay empty. When
    ``gate_resdmd`` is True, ``spectral`` records the max
    finite-dictionary ResDMD residual on the flattened encodings.
    ``rollout_horizon`` is a step count, not a physical duration.

    Parameters
    ----------
    encodings : LatentPairs or sequence of LatentPairs
        Encodings used to score the snapshot (typically the same pairs
        used to fit, or a re-encode after the last encoder step).
    snapshot : OperatorSnapshot
        Identified dense ``matrix``.
    rollout_horizon : int, optional
        Maximum latent rollout steps. Default is ``5``.
    gate_resdmd : bool, optional
        Fill :class:`~koopman_graph.identification.SpectralReliabilityBlock`.
        Default is ``False``.
    residual_tolerance : float, optional
        Pollution cutoff when ``gate_resdmd`` is True. Default ``1e-2``.

    Returns
    -------
    IdentificationReport
        Frozen report with populated ``one_step``, ``rollout``, and
        ``stability`` when ``matrix`` is set.

    Raises
    ------
    ValueError
        If ``snapshot.matrix`` is missing, ``rollout_horizon < 1``, or
        ``residual_tolerance`` is invalid.
    TypeError
        If ``gate_resdmd`` is not a ``bool``.
    """
    if type(gate_resdmd) is not bool:
        msg = f"gate_resdmd must be a bool, got {type(gate_resdmd).__name__}"
        raise TypeError(msg)
    if rollout_horizon < 1:
        msg = f"rollout_horizon must be >= 1, got {rollout_horizon}"
        raise ValueError(msg)
    if isinstance(residual_tolerance, bool) or not isinstance(
        residual_tolerance, (int, float)
    ):
        msg = (
            "residual_tolerance must be a finite float, got "
            f"{type(residual_tolerance).__name__}"
        )
        raise ValueError(msg)
    if not math.isfinite(float(residual_tolerance)) or float(residual_tolerance) < 0.0:
        msg = (
            "residual_tolerance must be a finite non-negative float, "
            f"got {residual_tolerance!r}"
        )
        raise ValueError(msg)
    matrix = snapshot.matrix
    if matrix is None:
        msg = "build_identification_report requires snapshot.matrix"
        raise ValueError(msg)
    radius = float(torch.linalg.eigvals(matrix).abs().max().real.item())
    if not math.isfinite(radius) or radius < 0.0:
        msg = f"spectral radius must be a finite non-negative float, got {radius!r}"
        raise ValueError(msg)
    blocks: tuple[LatentPairs, ...]
    if isinstance(encodings, LatentPairs):
        blocks = (encodings,)
    else:
        blocks = tuple(encodings)
        if not blocks:
            msg = "build_identification_report requires at least one LatentPairs"
            raise ValueError(msg)
    spectral = (
        _spectral_reliability(blocks, tolerance=float(residual_tolerance))
        if gate_resdmd
        else SpectralReliabilityBlock()
    )
    return IdentificationReport(
        one_step=_mean_metric([_one_step_mse(block, matrix) for block in blocks]),
        rollout=_mean_metric(
            [_rollout_mse(block, matrix, rollout_horizon) for block in blocks]
        ),
        spectral=spectral,
        stability=StabilityBlock(spectral_radius=radius),
    )
