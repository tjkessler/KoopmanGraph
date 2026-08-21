"""Opt-in drift–diffusion stepper (Euler–Maruyama / Yosida MVP).

The SDE :math:`dz = L z\\,dt + \\Sigma\\,dW` is a discrete-semigroup
approximation, not certified Itô theory and not stochastic gEDMD / SDMD
identification (``Xu2025StochasticSemigroup``, ``Zhou2025Yosida``).
``dynamics_mode="stochastic"`` remains diagonal process noise after a
discrete :math:`K` (:mod:`koopman_graph.operators.stochastic`).

This module must not import :mod:`koopman_graph.model`,
:mod:`koopman_graph.data`, or :mod:`koopman_graph.uq`.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor, nn

StepKind = Literal["euler_maruyama", "yosida"]
DEFAULT_DIFFUSION_RANK = 1
DEFAULT_STEP_KIND: StepKind = "euler_maruyama"
_DIFFUSION_INIT = 0.1

__all__ = [
    "DEFAULT_DIFFUSION_RANK",
    "DEFAULT_STEP_KIND",
    "DriftDiffusionKoopman",
]


def _require_positive_interval(delta_t: float | Tensor) -> float:
    """Validate a shared integration interval.

    Parameters
    ----------
    delta_t : float or Tensor
        Step length in the caller time unit.

    Returns
    -------
    float
        Accepted interval.

    Raises
    ------
    ValueError
        If ``delta_t`` is not a finite non-negative scalar.
    """
    value = float(torch.as_tensor(delta_t).reshape(-1)[0].item())
    if not math.isfinite(value) or value < 0.0:
        msg = f"delta_t must be a finite non-negative interval, got {delta_t!r}"
        raise ValueError(msg)
    return value


class DriftDiffusionKoopman(nn.Module):
    """Latent drift–diffusion :math:`dz = Lz\\,dt + F dW` with a discrete step.

    Row-vector convention matches continuous operators
    (:math:`z \\mapsto z L^{\\top}`). Diffusion is low-rank
    :math:`\\Sigma\\Sigma^{\\top} = F F^{\\top}` with independent Wiener
    increments per leading batch / node row. ``forward`` is the
    conditional-expectation semigroup step (no noise). ``advance`` draws
    one path increment.

    Euler–Maruyama uses :math:`z + \\Delta t\\, z L^{\\top}`. Yosida uses
    the implicit resolvent :math:`z (I - \\Delta t L^{\\top})^{-1}`.
    Neither is a certified SDE semigroup or SDMD estimator.

    Parameters
    ----------
    latent_dim : int
        Latent width :math:`d`.
    diffusion_rank : int, optional
        Rank of :math:`F` with shape ``(d, rank)``. Default is 1.
    step_kind : {"euler_maruyama", "yosida"}, optional
        Discrete increment. Default is Euler–Maruyama.

    Notes
    -----
    :math:`L` has units of inverse time. :math:`F` has units of latent
    per square-root time. ``delta_t`` shares the time unit.
    ``dynamics_mode="stochastic"`` is a different API (diagonal
    :math:`\\varepsilon` after discrete :math:`K`).

    References
    ----------
    Xu, Y., Shao, K., Ishikawa, I., Hashimoto, Y., Logothetis, N. and
    Shen, Z. (2025). A data-driven framework for Koopman semigroup
    estimation in stochastic dynamical systems. *Chaos* 35:103123.
    doi:10.1063/5.0283640 (``Xu2025StochasticSemigroup``).
    Zhou, J., Meng, Y. and Liu, J. (2025). Koopman spectral analysis
    and system identification for stochastic dynamical systems via
    Yosida approximation of generators. arXiv:2504.13912
    (``Zhou2025Yosida``).
    """

    def __init__(
        self,
        latent_dim: int,
        *,
        diffusion_rank: int = DEFAULT_DIFFUSION_RANK,
        step_kind: StepKind = DEFAULT_STEP_KIND,
    ) -> None:
        """Initialize a dense drift and low-rank diffusion factor.

        Parameters
        ----------
        latent_dim : int
            Latent width.
        diffusion_rank : int, optional
            Rank of the diffusion factor. Default is 1.
        step_kind : {"euler_maruyama", "yosida"}, optional
            Discrete increment kind.
        """
        super().__init__()
        if type(latent_dim) is not int or latent_dim < 1:
            msg = f"latent_dim must be a positive int, got {latent_dim!r}"
            raise ValueError(msg)
        if type(diffusion_rank) is not int or not 1 <= diffusion_rank <= latent_dim:
            msg = (
                "diffusion_rank must be an int in "
                f"[1, {latent_dim}], got {diffusion_rank!r}"
            )
            raise ValueError(msg)
        if step_kind not in {"euler_maruyama", "yosida"}:
            msg = f"step_kind must be 'euler_maruyama' or 'yosida', got {step_kind!r}"
            raise ValueError(msg)
        self.latent_dim = latent_dim
        self.diffusion_rank = diffusion_rank
        self.step_kind: StepKind = step_kind
        self.L = nn.Parameter(-torch.eye(latent_dim))
        factor = torch.zeros(latent_dim, diffusion_rank)
        for index in range(diffusion_rank):
            factor[index, index] = _DIFFUSION_INIT
        self.diffusion = nn.Parameter(factor)

    def set_drift(self, drift: Tensor) -> None:
        """Write the dense generator :math:`L`.

        Parameters
        ----------
        drift : Tensor
            Square matrix with shape ``(d, d)`` in inverse-time units.

        Raises
        ------
        ValueError
            If the shape is not ``(latent_dim, latent_dim)``.
        """
        if drift.shape != (self.latent_dim, self.latent_dim):
            msg = (
                "drift must have shape "
                f"({self.latent_dim}, {self.latent_dim}), got {tuple(drift.shape)}"
            )
            raise ValueError(msg)
        with torch.no_grad():
            self.L.copy_(drift.to(dtype=self.L.dtype, device=self.L.device))

    def set_diffusion(self, factor: Tensor) -> None:
        """Write the low-rank factor :math:`F`.

        Parameters
        ----------
        factor : Tensor
            Matrix with shape ``(d, rank)`` in latent per square-root time.

        Raises
        ------
        ValueError
            If the shape is not ``(latent_dim, diffusion_rank)``.
        """
        expected = (self.latent_dim, self.diffusion_rank)
        if factor.shape != expected:
            msg = (
                "diffusion factor must have shape "
                f"{expected}, got {tuple(factor.shape)}"
            )
            raise ValueError(msg)
        with torch.no_grad():
            self.diffusion.copy_(
                factor.to(dtype=self.diffusion.dtype, device=self.diffusion.device)
            )

    def diffusion_covariance(self) -> Tensor:
        """Return :math:`F F^{\\top}` with shape ``(d, d)``.

        Returns
        -------
        Tensor
            Instantaneous diffusion covariance (latent² / time).
        """
        return self.diffusion @ self.diffusion.transpose(-1, -2)

    def mean_advance(self, z: Tensor, delta_t: float | Tensor) -> Tensor:
        """Conditional-expectation semigroup step (no diffusion).

        Parameters
        ----------
        z : Tensor
            Latents with trailing dimension ``latent_dim``.
        delta_t : float or Tensor
            Interval in the caller time unit. ``0`` returns ``z``.

        Returns
        -------
        Tensor
            Mean increment with the same shape as ``z``.
        """
        self._check_latent(z)
        interval = _require_positive_interval(delta_t)
        if interval == 0.0:
            return z
        if self.step_kind == "euler_maruyama":
            return z + interval * (z @ self.L.transpose(-1, -2))
        eye = torch.eye(self.latent_dim, dtype=z.dtype, device=z.device)
        left = eye - interval * self.L
        return torch.linalg.solve(left, z.transpose(-1, -2)).transpose(-1, -2)

    def advance(
        self,
        z: Tensor,
        delta_t: float | Tensor | None = None,
        *,
        control: Tensor | None = None,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Draw one path increment (sampled diffusion).

        Parameters
        ----------
        z : Tensor
            Latents with trailing dimension ``latent_dim``.
        delta_t : float, Tensor, or None
            Required interval in the caller time unit.
        control : Tensor or None, optional
            Must be ``None``. The MVP is uncontrolled.
        edge_index, edge_weight : Tensor or None, optional
            Ignored (per-node stepper).
        generator : torch.Generator or None, optional
            Optional RNG for the Wiener increment.

        Returns
        -------
        Tensor
            Sampled latents with the same shape as ``z``.

        Raises
        ------
        ValueError
            If ``delta_t`` is missing or ``control`` is set.
        """
        del edge_index, edge_weight
        if control is not None:
            msg = "DriftDiffusionKoopman is uncontrolled; pass control=None"
            raise ValueError(msg)
        if delta_t is None:
            msg = "DriftDiffusionKoopman.advance requires delta_t"
            raise ValueError(msg)
        mean = self.mean_advance(z, delta_t)
        interval = _require_positive_interval(delta_t)
        if interval == 0.0:
            return mean
        noise_shape = (*z.shape[:-1], self.diffusion_rank)
        xi = torch.randn(
            noise_shape,
            dtype=z.dtype,
            device=z.device,
            generator=generator,
        )
        return mean + math.sqrt(interval) * (xi @ self.diffusion.transpose(-1, -2))

    def forward(self, z: Tensor, delta_t: float | Tensor) -> Tensor:
        """Module forward: conditional-expectation step.

        Parameters
        ----------
        z : Tensor
            Latents with trailing dimension ``latent_dim``.
        delta_t : float or Tensor
            Interval in the caller time unit.

        Returns
        -------
        Tensor
            Mean increment (no diffusion).
        """
        return self.mean_advance(z, delta_t)

    def _check_latent(self, z: Tensor) -> None:
        """Refuse a trailing width that is not ``latent_dim``.

        Parameters
        ----------
        z : Tensor
            Candidate latents.

        Raises
        ------
        ValueError
            If the trailing dimension does not match.
        """
        if z.shape[-1] != self.latent_dim:
            msg = (
                "z trailing dimension must equal latent_dim "
                f"{self.latent_dim}, got {tuple(z.shape)}"
            )
            raise ValueError(msg)
