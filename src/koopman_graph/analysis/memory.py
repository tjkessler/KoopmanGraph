"""Innovation-whiteness diagnostic and a finite-memory convolution MVP.

:func:`markov_closure_report` scores residual autocorrelation with a
Ljung–Box-style portmanteau (``Ljung1978Box``). It is a diagnostic on
already-formed innovations, not Mori–Zwanzig projection-operator
identification (``Lin2021MoriZwanzig``).

:class:`FiniteMemoryKoopman` applies the convolution

.. math::

    z_{t+1} = z_t \\Omega^{\\top} + \\sum_{s=1}^{M} z_{t-s} K_s^{\\top}.

That map is **not** a factory kind, **not**
:class:`~koopman_graph.nn.delay.DelayEmbeddingEncoder` (Takens channel
stacking), and **not** :class:`~koopman_graph.baselines.HAVOKBaseline`
(delay-plus-forcing; ``Brunton2017HAVOK``). Recovered memory length is
an oracle test, not a general theorem.

This module must not import :mod:`koopman_graph.model`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from scipy.stats import chi2
from torch import Tensor, nn

DEFAULT_MAX_LAG = 10
DEFAULT_ALPHA = 0.05
DEFAULT_MEMORY_ORDER = 1
LAG_UNIT = "timestep"

__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_MAX_LAG",
    "DEFAULT_MEMORY_ORDER",
    "FiniteMemoryKoopman",
    "LAG_UNIT",
    "MarkovClosureReport",
    "markov_closure_report",
]


@dataclass(frozen=True)
class MarkovClosureReport:
    """Ljung–Box-style whiteness flags on a residual-energy series.

    Attributes
    ----------
    autocorrelation : Tensor
        Sample autocorrelation at lags ``1, …, max_lag``. Dimensionless.
        Shape ``(max_lag,)``.
    ljung_box_statistic : float
        Portmanteau statistic :math:`Q` (dimensionless).
    ljung_box_pvalue : float
        Survival function of :math:`\\chi^2` with
        ``max_lag - n_fit_parameters`` degrees of freedom.
    max_abs_autocorrelation : float
        Maximum absolute sample autocorrelation over those lags.
    max_lag : int
        Highest lag included in :math:`Q`.
    n_timesteps : int
        Length of the residual-energy series.
    n_fit_parameters : int
        Degrees of freedom subtracted for a previously fitted Markov
        map. ``0`` treats the series as given.
    alpha : float
        Whiteness test size in ``(0, 1)``.
    white : bool
        ``True`` when ``ljung_box_pvalue >= alpha`` (fail to reject
        residual whiteness).
    lag_unit : str
        Always ``"timestep"``. Lags are steps in the residual series,
        not a physical time unit.
    """

    autocorrelation: Tensor
    ljung_box_statistic: float
    ljung_box_pvalue: float
    max_abs_autocorrelation: float
    max_lag: int
    n_timesteps: int
    n_fit_parameters: int
    alpha: float
    white: bool
    lag_unit: str = LAG_UNIT

    def __post_init__(self) -> None:
        """Validate report fields.

        Raises
        ------
        ValueError
            If shapes, ranges, or the lag unit are inconsistent.
        """
        acf = self.autocorrelation
        if acf.ndim != 1 or int(acf.numel()) != int(self.max_lag):
            msg = (
                "autocorrelation must have shape (max_lag,), "
                f"got {tuple(acf.shape)} for max_lag={self.max_lag}"
            )
            raise ValueError(msg)
        if int(self.max_lag) < 1:
            raise ValueError(f"max_lag must be >= 1, got {self.max_lag}")
        if int(self.n_timesteps) <= int(self.max_lag):
            msg = (
                "n_timesteps must exceed max_lag, "
                f"got n={self.n_timesteps} and max_lag={self.max_lag}"
            )
            raise ValueError(msg)
        if int(self.n_fit_parameters) < 0:
            msg = f"n_fit_parameters must be >= 0, got {self.n_fit_parameters}"
            raise ValueError(msg)
        degrees = int(self.max_lag) - int(self.n_fit_parameters)
        if degrees < 1:
            msg = (
                "max_lag - n_fit_parameters must be >= 1, "
                f"got {self.max_lag} - {self.n_fit_parameters}"
            )
            raise ValueError(msg)
        if not 0.0 < float(self.alpha) < 1.0:
            raise ValueError(f"alpha must lie in (0, 1), got {self.alpha!r}")
        if self.lag_unit != LAG_UNIT:
            msg = f"lag_unit must be {LAG_UNIT!r}, got {self.lag_unit!r}"
            raise ValueError(msg)
        if not math.isfinite(self.ljung_box_statistic):
            raise ValueError("ljung_box_statistic must be finite")
        if not (0.0 <= float(self.ljung_box_pvalue) <= 1.0):
            msg = f"ljung_box_pvalue must lie in [0, 1], got {self.ljung_box_pvalue!r}"
            raise ValueError(msg)
        if not math.isfinite(self.max_abs_autocorrelation):
            raise ValueError("max_abs_autocorrelation must be finite")


def _residual_energy_series(innovations: Tensor) -> Tensor:
    """Reduce innovations to a real 1-D residual-energy series.

    Parameters
    ----------
    innovations : Tensor
        Finite real residual tensor. Leading axis is time.

    Returns
    -------
    Tensor
        Shape ``(T,)`` Euclidean norms over trailing axes. A 1-D input
        is returned as a contiguous copy.

    Raises
    ------
    TypeError
        If ``innovations`` is not a tensor.
    ValueError
        If the tensor is empty, non-floating, non-finite, or complex.
    """
    if not isinstance(innovations, Tensor):
        msg = f"innovations must be a Tensor, got {type(innovations).__name__}"
        raise TypeError(msg)
    if innovations.is_complex():
        raise ValueError("innovations must be real")
    if not innovations.is_floating_point():
        raise ValueError("innovations must be a floating-point tensor")
    if int(innovations.ndim) < 1 or int(innovations.shape[0]) == 0:
        raise ValueError("innovations must be a nonempty time series")
    if not bool(torch.isfinite(innovations).all().item()):
        raise ValueError("innovations must be finite")
    if int(innovations.ndim) == 1:
        return innovations.detach().to(dtype=torch.float64).contiguous()
    flat = innovations.detach().reshape(int(innovations.shape[0]), -1)
    return flat.to(dtype=torch.float64).norm(dim=-1)


def _sample_autocorrelation(series: Tensor, max_lag: int) -> Tensor:
    """Sample autocorrelation at lags ``1, …, max_lag``.

    Parameters
    ----------
    series : Tensor
        1-D finite series of length ``n > max_lag``.
    max_lag : int
        Highest lag.

    Returns
    -------
    Tensor
        Shape ``(max_lag,)``, dimensionless.

    Raises
    ------
    ValueError
        If the series has zero variance.
    """
    centered = series - series.mean()
    denom = torch.dot(centered, centered)
    if float(denom.item()) <= 0.0 or not math.isfinite(float(denom.item())):
        raise ValueError("innovations must have positive residual energy")
    rows = [
        torch.dot(centered[lag:], centered[:-lag]) / denom
        for lag in range(1, max_lag + 1)
    ]
    return torch.stack(rows)


def _resolve_max_lag(n_timesteps: int, max_lag: int | None) -> int:
    """Choose a portmanteau lag that leaves degrees of freedom.

    Parameters
    ----------
    n_timesteps : int
        Residual-series length.
    max_lag : int or None
        Caller lag, or ``None`` for ``min(10, (n-1)//5)``.

    Returns
    -------
    int
        Accepted ``max_lag >= 1``.

    Raises
    ------
    ValueError
        If the series is too short or ``max_lag`` is invalid.
    """
    if max_lag is None:
        resolved = min(DEFAULT_MAX_LAG, (int(n_timesteps) - 1) // 5)
    else:
        if isinstance(max_lag, bool) or not isinstance(max_lag, int):
            raise ValueError(f"max_lag must be a positive int, got {max_lag!r}")
        resolved = int(max_lag)
    if resolved < 1:
        msg = (
            "need enough timesteps for at least one lag "
            f"(n={n_timesteps}, max_lag={max_lag!r})"
        )
        raise ValueError(msg)
    if int(n_timesteps) <= resolved:
        msg = (
            "n_timesteps must exceed max_lag, "
            f"got n={n_timesteps} and max_lag={resolved}"
        )
        raise ValueError(msg)
    return resolved


def markov_closure_report(
    innovations: Tensor,
    *,
    max_lag: int | None = None,
    alpha: float = DEFAULT_ALPHA,
    n_fit_parameters: int = 0,
) -> MarkovClosureReport:
    """Score residual autocorrelation with a Ljung–Box-style flag.

    The input is reduced to residual energy :math:`\\|\\eta_t\\|_2` when
    it has trailing axes. That is a **scalar** portmanteau on pooled
    energy, not a matrix-variate (Hosking) test and not Mori–Zwanzig
    identification (``Lin2021MoriZwanzig``).

    Parameters
    ----------
    innovations : Tensor
        Finite real residuals. Leading axis is time (timesteps, not a
        physical unit unless the caller sampled uniformly).
    max_lag : int or None, optional
        Highest lag. Default is ``min(10, (T-1)//5)``.
    alpha : float, optional
        Test size in ``(0, 1)``. Default is ``0.05``.
    n_fit_parameters : int, optional
        Degrees of freedom already spent on a fitted Markov map.
        Default ``0`` treats ``innovations`` as given.

    Returns
    -------
    MarkovClosureReport
        Autocorrelation, :math:`Q`, :math:`p`-value, and ``white``.

    Raises
    ------
    TypeError
        If ``innovations`` is not a tensor.
    ValueError
        If the series is empty, non-finite, degenerate, or the lag /
        alpha / degrees-of-freedom arguments are invalid.

    Notes
    -----
    :math:`Q = n(n+2)\\sum_{k=1}^{h}\\rho_k^2/(n-k)` is compared to
    :math:`\\chi^2(h-p)` (``Ljung1978Box``). ``white`` is a hypothesis
    flag, not a certificate that the latent map is Markov.

    References
    ----------
    Ljung, G. M. and Box, G. E. P. (1978). On a measure of lack of fit
    in time series models. *Biometrika* 65:297–303.
    doi:10.1093/biomet/65.2.297 (``Ljung1978Box``).
    Lin, Y. T., Tian, Y., Anghel, M. and Livescu, D. (2021).
    Data-driven learning for the Mori–Zwanzig formalism: a
    generalization of the Koopman learning framework. *SIAM Journal on
    Applied Dynamical Systems* 20:2558–2601. doi:10.1137/21M1401759
    (``Lin2021MoriZwanzig``).
    """
    series = _residual_energy_series(innovations)
    n_timesteps = int(series.shape[0])
    resolved_lag = _resolve_max_lag(n_timesteps, max_lag)
    if isinstance(n_fit_parameters, bool) or not isinstance(n_fit_parameters, int):
        msg = f"n_fit_parameters must be a non-negative int, got {n_fit_parameters!r}"
        raise ValueError(msg)
    if int(n_fit_parameters) < 0:
        msg = f"n_fit_parameters must be >= 0, got {n_fit_parameters}"
        raise ValueError(msg)
    degrees = resolved_lag - int(n_fit_parameters)
    if degrees < 1:
        msg = (
            "max_lag - n_fit_parameters must be >= 1, "
            f"got {resolved_lag} - {n_fit_parameters}"
        )
        raise ValueError(msg)
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha!r}")
    autocorrelation = _sample_autocorrelation(series, resolved_lag)
    lags = torch.arange(
        1,
        resolved_lag + 1,
        dtype=autocorrelation.dtype,
        device=autocorrelation.device,
    )
    lag_terms = autocorrelation.square() / (float(n_timesteps) - lags)
    statistic = (
        float(n_timesteps) * (float(n_timesteps) + 2.0) * float(lag_terms.sum().item())
    )
    pvalue = float(chi2.sf(statistic, degrees))
    max_abs = float(autocorrelation.abs().max().item())
    return MarkovClosureReport(
        autocorrelation=autocorrelation,
        ljung_box_statistic=statistic,
        ljung_box_pvalue=pvalue,
        max_abs_autocorrelation=max_abs,
        max_lag=resolved_lag,
        n_timesteps=n_timesteps,
        n_fit_parameters=int(n_fit_parameters),
        alpha=float(alpha),
        white=pvalue >= float(alpha),
        lag_unit=LAG_UNIT,
    )


class FiniteMemoryKoopman(nn.Module):
    """Convolution :math:`z_{t+1}=z_t\\Omega^{\\top}+\\sum_s z_{t-s}K_s^{\\top}`.

    Row-vector convention matches discrete operators
    (:math:`z \\mapsto z K^{\\top}`). Default ``omega`` is
    :math:`\\tfrac12 I` and default kernels are zero, so the map is
    Markov until the caller writes memory taps. This is an MVP
    parameterization, **not** Mori–Zwanzig identification
    (``Lin2021MoriZwanzig``), **not** a factory kind, **not**
    :class:`~koopman_graph.nn.delay.DelayEmbeddingEncoder`, and **not**
    :class:`~koopman_graph.baselines.HAVOKBaseline`
    (``Brunton2017HAVOK``). Delay stacking changes encoder
    ``in_channels``; this class keeps latent width ``d`` and stores
    ``M`` dense taps.

    Parameters
    ----------
    latent_dim : int
        Latent width :math:`d`.
    memory_order : int, optional
        Number of memory taps :math:`M \\ge 1`. Default is 1.

    Notes
    -----
    ``omega`` and ``kernels`` are dimensionless maps on latent
    coordinates, the same convention as discrete ``K``. There is no
    checkpoint key and no ``koopman="finite_memory"`` factory string.
    Memory-order selection by held-out rollout is not implemented;
    recovered length is an oracle test, not a theorem.

    References
    ----------
    Lin, Y. T., Tian, Y., Anghel, M. and Livescu, D. (2021).
    Data-driven learning for the Mori–Zwanzig formalism: a
    generalization of the Koopman learning framework. *SIAM Journal on
    Applied Dynamical Systems* 20:2558–2601. doi:10.1137/21M1401759
    (``Lin2021MoriZwanzig``).
    Brunton, S. L., Brunton, B. W., Proctor, J. L., Kaiser, E. and
    Kutz, J. N. (2017). Chaos as an intermittently forced linear
    system. *Nature Communications* 8:19. doi:10.1038/s41467-017-00030-8
    (``Brunton2017HAVOK``).
    """

    def __init__(
        self,
        latent_dim: int,
        memory_order: int = DEFAULT_MEMORY_ORDER,
    ) -> None:
        """Allocate the Markov map and memory taps.

        Parameters
        ----------
        latent_dim : int
            Latent width :math:`d`.
        memory_order : int, optional
            Number of taps :math:`M \\ge 1`. Default is 1.

        Raises
        ------
        ValueError
            If ``latent_dim`` or ``memory_order`` is invalid.
        """
        super().__init__()
        if isinstance(latent_dim, bool) or not isinstance(latent_dim, int):
            raise ValueError(f"latent_dim must be a positive int, got {latent_dim!r}")
        if int(latent_dim) < 1:
            raise ValueError(f"latent_dim must be >= 1, got {latent_dim}")
        if isinstance(memory_order, bool) or not isinstance(memory_order, int):
            msg = f"memory_order must be a positive int, got {memory_order!r}"
            raise ValueError(msg)
        if int(memory_order) < 1:
            raise ValueError(f"memory_order must be >= 1, got {memory_order}")
        width = int(latent_dim)
        order = int(memory_order)
        self.latent_dim = width
        self.memory_order = order
        self.omega = nn.Parameter(0.5 * torch.eye(width))
        self.kernels = nn.Parameter(torch.zeros(order, width, width))

    def advance(self, z: Tensor, history: Tensor) -> Tensor:
        """Apply one convolution step.

        Parameters
        ----------
        z : Tensor
            Current latent with trailing width ``latent_dim``.
        history : Tensor
            Past latents with shape ``(memory_order, *z.shape)``.
            ``history[0]`` is :math:`z_{t-1}` and ``history[s]`` is
            :math:`z_{t-s-1}` (equivalently :math:`z_{t-s}` at
            ``s`` in ``1..M`` uses ``history[s-1]``).

        Returns
        -------
        Tensor
            Next latent, same shape as ``z``.

        Raises
        ------
        ValueError
            If shapes do not match the allocated width and order.
        """
        if z.shape[-1] != self.latent_dim:
            msg = (
                "z trailing dimension must equal latent_dim "
                f"{self.latent_dim}, got {tuple(z.shape)}"
            )
            raise ValueError(msg)
        expected = (self.memory_order, *z.shape)
        if tuple(history.shape) != expected:
            msg = (
                "history must have shape "
                f"(memory_order, *z.shape)={expected}, got {tuple(history.shape)}"
            )
            raise ValueError(msg)
        markov = z @ self.omega.transpose(-1, -2)
        memory = torch.einsum("m...d,med->...e", history, self.kernels)
        return markov + memory

    def forward(self, z: Tensor, history: Tensor) -> Tensor:
        """Alias of :meth:`advance`.

        Parameters
        ----------
        z : Tensor
            Current latent.
        history : Tensor
            Past latents ``(memory_order, *z.shape)``.

        Returns
        -------
        Tensor
            Next latent.
        """
        return self.advance(z, history)
