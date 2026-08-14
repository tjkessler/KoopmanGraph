"""Pairwise Granger-style influence scores on latent time series.

Assumption-laden: linear residual-MSE reduction, not interventional causal
discovery and not :func:`~koopman_graph.analysis.explain_representation`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class CausalInfluenceReport:
    """Pairwise influence matrix.

    Attributes
    ----------
    scores : Tensor
        Shape ``(N, N)``. Entry ``(i, j)`` is the relative MSE drop when
        predicting node ``j`` from ``i`` versus an intercept-only model.
    """

    scores: Tensor


def granger_latent_influence(latents: Tensor) -> CausalInfluenceReport:
    """Score directed influence among node latent trajectories.

    Parameters
    ----------
    latents : Tensor
        Time series ``(T, N, d)`` with ``T >= 3``.

    Returns
    -------
    CausalInfluenceReport
        Pairwise residual-MSE reduction scores.

    Raises
    ------
    ValueError
        If the shape or length is invalid.
    """
    if latents.ndim != 3:
        raise ValueError(
            f"latents must have shape (T, N, d), got {tuple(latents.shape)}"
        )
    time, num_nodes, dim = latents.shape
    if time < 3:
        raise ValueError("latents require T >= 3")
    past = latents[:-1].reshape(time - 1, num_nodes * dim)
    future = latents[1:].reshape(time - 1, num_nodes, dim)
    scores = torch.zeros(num_nodes, num_nodes, dtype=latents.dtype)
    ones = torch.ones(time - 1, 1, dtype=latents.dtype, device=latents.device)
    for target in range(num_nodes):
        y = future[:, target, :].reshape(time - 1, dim)
        intercept_pred = ones @ torch.linalg.lstsq(ones, y).solution
        base = (y - intercept_pred).square().mean()
        for source in range(num_nodes):
            x = torch.cat(
                [ones, past[:, source * dim : (source + 1) * dim]],
                dim=-1,
            )
            fitted = x @ torch.linalg.lstsq(x, y).solution
            residual = (y - fitted).square().mean()
            scores[source, target] = (base - residual) / base.clamp_min(1e-12)
    return CausalInfluenceReport(scores=scores.clamp(min=0.0))
