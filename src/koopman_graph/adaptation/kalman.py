"""Reference linear-Gaussian Kalman filter and RTS smoother numerics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class FilterResult:
    """Per-timestep filtered or smoothed latent means and covariances.

    Public result types in this package are frozen dataclasses with attribute
    access (not mapping/dict styles).

    Attributes
    ----------
    latents : Tensor
        Filtered or smoothed latent states with shape
        ``(num_timesteps, num_nodes, latent_dim)``.
    covariances : Tensor
        Flattened-state covariances with shape
        ``(num_timesteps, num_nodes * latent_dim, num_nodes * latent_dim)``.
        Node blocks are stacked in row-major order matching ``latents.reshape
        (..., -1)``.
    """

    latents: Tensor
    covariances: Tensor


def reference_kalman_filter(
    *,
    transition: Tensor,
    process_cov: Tensor,
    observation: Tensor,
    observation_cov: Tensor,
    measurements: Tensor,
    x0: Tensor,
    p0: Tensor,
    control_bias: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Run a textbook linear-Gaussian Kalman filter (predict then update).

    Parameters use column-vector state convention.

    Parameters
    ----------
    transition : Tensor
        State transition matrix ``A`` with shape ``(D, D)``.
    process_cov : Tensor
        Process noise covariance ``Q`` with shape ``(D, D)``.
    observation : Tensor
        Observation matrix ``H`` with shape ``(M, D)``.
    observation_cov : Tensor
        Observation noise covariance ``R`` with shape ``(M, M)``.
    measurements : Tensor
        Measurement sequence with shape ``(T, M)``.
    x0 : Tensor
        Prior mean with shape ``(D,)``.
    p0 : Tensor
        Prior covariance with shape ``(D, D)``.
    control_bias : Tensor or None, optional
        Optional per-interval additive bias with shape ``(T-1, D)``.

    Returns
    -------
    tuple of Tensor
        Filtered means, filtered covariances, one-step predicted means, and
        one-step predicted covariances.
    """
    t_steps = measurements.shape[0]
    dim = x0.shape[0]
    means = torch.empty(t_steps, dim, dtype=x0.dtype, device=x0.device)
    covs = torch.empty(t_steps, dim, dim, dtype=x0.dtype, device=x0.device)
    pred_means = torch.empty(t_steps, dim, dtype=x0.dtype, device=x0.device)
    pred_covs = torch.empty(t_steps, dim, dim, dtype=x0.dtype, device=x0.device)

    x_prev = x0
    p_prev = p0
    eye = torch.eye(dim, dtype=x0.dtype, device=x0.device)
    for t in range(t_steps):
        if t == 0:
            x_pred = x_prev
            p_pred = p_prev
        else:
            bias = (
                torch.zeros(dim, dtype=x0.dtype, device=x0.device)
                if control_bias is None
                else control_bias[t - 1]
            )
            x_pred = transition @ x_prev + bias
            p_pred = transition @ p_prev @ transition.T + process_cov
        pred_means[t] = x_pred
        pred_covs[t] = p_pred

        innov_cov = observation @ p_pred @ observation.T + observation_cov
        gain = torch.linalg.solve(innov_cov, observation @ p_pred).T
        innov = measurements[t] - observation @ x_pred
        x_filt = x_pred + gain @ innov
        p_filt = (eye - gain @ observation) @ p_pred
        # Symmetrize for numerical hygiene.
        p_filt = 0.5 * (p_filt + p_filt.T)
        means[t] = x_filt
        covs[t] = p_filt
        x_prev = x_filt
        p_prev = p_filt
    return means, covs, pred_means, pred_covs


def rts_smooth(
    *,
    transition: Tensor,
    filtered_means: Tensor,
    filtered_covs: Tensor,
    pred_means: Tensor,
    pred_covs: Tensor,
) -> tuple[Tensor, Tensor]:
    """Rauch–Tung–Striebel smoother given forward-filter caches.

    Parameters
    ----------
    transition : Tensor
        Constant transition matrix used for smoother gains.
    filtered_means : Tensor
        Filtered means with shape ``(T, D)``.
    filtered_covs : Tensor
        Filtered covariances with shape ``(T, D, D)``.
    pred_means : Tensor
        One-step predicted means with shape ``(T, D)``.
    pred_covs : Tensor
        One-step predicted covariances with shape ``(T, D, D)``.

    Returns
    -------
    tuple of Tensor
        Smoothed means and covariances.
    """
    t_steps = filtered_means.shape[0]
    means = filtered_means.clone()
    covs = filtered_covs.clone()
    for t in range(t_steps - 2, -1, -1):
        # G = P_t F^T (P_{t+1}^-)^{-1}
        gain = torch.linalg.solve(pred_covs[t + 1], transition @ filtered_covs[t].T).T
        means[t] = filtered_means[t] + gain @ (means[t + 1] - pred_means[t + 1])
        covs[t] = filtered_covs[t] + gain @ (covs[t + 1] - pred_covs[t + 1]) @ gain.T
        covs[t] = 0.5 * (covs[t] + covs[t].T)
    return means, covs
