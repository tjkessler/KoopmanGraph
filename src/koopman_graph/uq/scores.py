"""Proper scoring rules for Gaussian and ensemble forecasts.

Closed-form univariate Gaussian CRPS / NLL and the energy score on
iid samples. Scores are not coverage certificates and do not change
``dynamics_mode="stochastic"``.

This module must not import :mod:`koopman_graph.model` or
:mod:`koopman_graph.operators`.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor

ScoreReduction = Literal["mean", "sum", "none"]
_VARIANCE_FLOOR = 1e-12

__all__ = [
    "energy_score",
    "gaussian_crps",
    "gaussian_nll",
]


def _reduce(values: Tensor, reduction: ScoreReduction) -> Tensor:
    """Reduce a score tensor.

    Parameters
    ----------
    values : Tensor
        Per-event scores.
    reduction : {"mean", "sum", "none"}
        Reduction over all elements.

    Returns
    -------
    Tensor
        Reduced scalar, or ``values`` when ``reduction="none"``.
    """
    if reduction == "none":
        return values
    if reduction == "sum":
        return values.sum()
    if reduction == "mean":
        return values.mean()
    msg = f"reduction must be 'mean', 'sum', or 'none', got {reduction!r}"
    raise ValueError(msg)


def _positive_variance(variance: Tensor) -> Tensor:
    """Refuse non-finite or non-positive variances.

    Parameters
    ----------
    variance : Tensor
        Predictive variances (same unit squared as the observation).

    Returns
    -------
    Tensor
        Clamped positive variance.

    Raises
    ------
    ValueError
        If any variance is non-finite or non-positive before the floor.
    """
    if not bool(torch.all(torch.isfinite(variance))):
        msg = "variance must be finite"
        raise ValueError(msg)
    if bool(torch.any(variance <= 0.0)):
        msg = "variance must be strictly positive"
        raise ValueError(msg)
    return torch.clamp(variance, min=_VARIANCE_FLOOR)


def gaussian_nll(
    observation: Tensor,
    mean: Tensor,
    variance: Tensor,
    *,
    reduction: ScoreReduction = "mean",
) -> Tensor:
    """Univariate Gaussian negative log-likelihood.

    :math:`\\frac{1}{2}\\bigl(\\log(2\\pi\\sigma^2) + (y-\\mu)^2/\\sigma^2\\bigr)`
    per entry. Proper for a Gaussian predictive density.

    Parameters
    ----------
    observation, mean : Tensor
        Observation and predictive mean (same unit; broadcastable).
    variance : Tensor
        Predictive variance (unit squared). Must be positive and finite.
    reduction : {"mean", "sum", "none"}, optional
        Reduction over broadcast elements. Default ``mean``.

    Returns
    -------
    Tensor
        NLL in nats.
    """
    sigma2 = _positive_variance(variance)
    residual = observation - mean
    nll = 0.5 * (torch.log(2.0 * math.pi * sigma2) + (residual * residual) / sigma2)
    return _reduce(nll, reduction)


def gaussian_crps(
    observation: Tensor,
    mean: Tensor,
    variance: Tensor,
    *,
    reduction: ScoreReduction = "mean",
) -> Tensor:
    """Closed-form univariate Gaussian CRPS.

    For :math:`Z\\sim\\mathcal{N}(\\mu,\\sigma^2)` and residual
    :math:`z=(y-\\mu)/\\sigma`,
    :math:`\\mathrm{CRPS}=\\sigma\\bigl(z(2\\Phi(z)-1)+2\\varphi(z)-1/\\sqrt{\\pi}\\bigr)`.
    Proper for a Gaussian predictive distribution.

    Parameters
    ----------
    observation, mean : Tensor
        Observation and predictive mean (same unit; broadcastable).
    variance : Tensor
        Predictive variance (unit squared). Must be positive and finite.
    reduction : {"mean", "sum", "none"}, optional
        Reduction over broadcast elements. Default ``mean``.

    Returns
    -------
    Tensor
        CRPS in the observation unit.
    """
    sigma2 = _positive_variance(variance)
    sigma = torch.sqrt(sigma2)
    z = (observation - mean) / sigma
    cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    pdf = torch.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    crps = sigma * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / math.sqrt(math.pi))
    return _reduce(crps, reduction)


def energy_score(
    observation: Tensor,
    samples: Tensor,
    *,
    reduction: ScoreReduction = "mean",
) -> Tensor:
    """Energy score of an iid ensemble against an observation.

    :math:`\\mathbb{E}\\|y-x\\| - \\tfrac{1}{2}\\mathbb{E}\\|x-x'\\|` with
    an unbiased pairwise term. Proper for a multivariate forecast.
    ``samples`` lead with the ensemble axis.

    Parameters
    ----------
    observation : Tensor
        Observation with shape ``(..., d)``.
    samples : Tensor
        Ensemble with shape ``(S, ..., d)`` and ``S >= 2``.
    reduction : {"mean", "sum", "none"}, optional
        Reduction over event axes (not the ensemble). Default ``mean``.

    Returns
    -------
    Tensor
        Energy score in the observation unit.

    Raises
    ------
    ValueError
        If the ensemble axis has fewer than two draws or shapes disagree.
    """
    if samples.ndim < 2:
        msg = "samples must have shape (S, ..., d) with S >= 2"
        raise ValueError(msg)
    ensemble = samples.shape[0]
    if ensemble < 2:
        msg = f"energy_score requires at least 2 samples, got {ensemble}"
        raise ValueError(msg)
    if observation.shape != samples.shape[1:]:
        msg = (
            "observation shape must match samples[1:], "
            f"got {tuple(observation.shape)} vs {tuple(samples.shape[1:])}"
        )
        raise ValueError(msg)
    residual = samples - observation.unsqueeze(0)
    term_obs = torch.linalg.vector_norm(residual, dim=-1).mean(dim=0)
    delta = samples.unsqueeze(1) - samples.unsqueeze(0)
    pairwise = torch.linalg.vector_norm(delta, dim=-1)
    off_diag = 1.0 - torch.eye(ensemble, dtype=pairwise.dtype, device=pairwise.device)
    while off_diag.ndim < pairwise.ndim:
        off_diag = off_diag.unsqueeze(-1)
    term_pair = (pairwise * off_diag).sum(dim=(0, 1)) / float(ensemble * (ensemble - 1))
    score = term_obs - 0.5 * term_pair
    return _reduce(score, reduction)
