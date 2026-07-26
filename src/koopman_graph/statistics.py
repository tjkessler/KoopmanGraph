"""Long-horizon distributional statistics for chaotic / multi-step forecasts.

Neutral leaf (peer to :mod:`koopman_graph.metrics` and
:mod:`koopman_graph.observables`): pure ``torch`` tensor→tensor helpers with
no dependency on ``data``, ``model``, ``protocols``, or ``analysis``. Prefer
``from koopman_graph.statistics import …``; this module is intentionally off
root ``__all__``.

Series layout
-------------
Public helpers accept time series with shape ``(T, …)`` and treat axis
``dim`` (default ``0``) as time. Trailing axes are nodes and optional
features ``(T, N)`` or ``(T, N, F)``. Non-finite inputs raise ``ValueError``.

NaN / Inf policy
----------------
Inputs must be finite. Outputs are finite when inputs are finite (PSD uses an
``eps`` floor only inside log-domain distances).

References
----------
Welch, P. D. (1967). The use of fast Fourier transform for the estimation of
power spectra: a method based on time averaging over short, modified
periodograms. *IEEE Transactions on Audio and Electroacoustics*, 15(2),
70–73. https://doi.org/10.1109/TAU.1967.1161901
(``Welch1967``)

Rosenstein, M. T., Collins, J. J., & De Luca, C. J. (1993). A practical
method for calculating largest Lyapunov exponents from small data sets.
*Physica D: Nonlinear Phenomena*, 65(1–2), 117–134.
https://doi.org/10.1016/0167-2789(93)90009-P
(``Rosenstein1993Lyapunov``)

Notes
-----
:func:`largest_lyapunov_exponent` implements the Rosenstein et al. (1993)
small-data-set estimator. Estimates are sensitive to delay, embedding
dimension, Theiler window, and the linear fit window; defaults follow the
usual autocorrelation / mean-period heuristics. For construction oracles,
pass the documented hyperparameters from the unit tests rather than relying
on every auto setting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

_EPS = 1e-12
WindowName = Literal["hann"]


def _require_finite(name: str, tensor: Tensor) -> None:
    """Raise ``ValueError`` if ``tensor`` contains non-finite values.

    Parameters
    ----------
    name : str
        Argument name for the error message.
    tensor : Tensor
        Candidate input tensor.
    """
    if not torch.isfinite(tensor).all():
        msg = f"{name} must be finite (no NaN/Inf)"
        raise ValueError(msg)


def _move_time_first(series: Tensor, dim: int) -> Tensor:
    """Permute ``series`` so time is axis 0.

    Parameters
    ----------
    series : Tensor
        Input series.
    dim : int
        Time axis in ``series``.

    Returns
    -------
    Tensor
        Series with time on axis 0 and remaining axes flattened as
        ``(T, num_channels)``.
    """
    if series.ndim < 1:
        msg = f"series must have at least 1 dimension, got shape {tuple(series.shape)}"
        raise ValueError(msg)
    time_dim = dim if dim >= 0 else series.ndim + dim
    if time_dim < 0 or time_dim >= series.ndim:
        msg = f"dim={dim} is out of range for series shape {tuple(series.shape)}"
        raise ValueError(msg)
    moved = series.transpose(0, time_dim).contiguous()
    time_len = moved.shape[0]
    return moved.reshape(time_len, -1)


def _hann_window(length: int, *, dtype: torch.dtype, device: torch.device) -> Tensor:
    """Return a periodic Hann window of the given length.

    Parameters
    ----------
    length : int
        Window length ``L >= 2``.
    dtype : torch.dtype
        Floating dtype.
    device : torch.device
        Device for the window tensor.

    Returns
    -------
    Tensor
        Window with shape ``(L,)``.
    """
    if length < 2:
        msg = f"segment_length must be >= 2, got {length}"
        raise ValueError(msg)
    n = torch.arange(length, dtype=dtype, device=device)
    return 0.5 - 0.5 * torch.cos(2.0 * torch.pi * n / length)


def power_spectral_density(
    series: Tensor,
    *,
    dim: int = 0,
    segment_length: int | None = None,
    overlap: float = 0.5,
    window: WindowName = "hann",
) -> Tensor:
    """Estimate one-sided power spectral density via Welch's method.

    Splits the time axis into overlapping segments, applies a Hann window,
    computes ``torch.fft.rfft`` periodograms, and averages. No SciPy
    dependency. Channels (nodes × features) are estimated independently.

    Parameters
    ----------
    series : Tensor
        Real-valued series with time on ``dim``. Typical shapes ``(T,)``,
        ``(T, N)``, or ``(T, N, F)``.
    dim : int, optional
        Time axis. Default ``0``.
    segment_length : int or None, optional
        Welch segment length. Default ``min(256, T)`` (at least ``2``).
    overlap : float, optional
        Fractional overlap in ``[0, 1)``. Default ``0.5``.
    window : {"hann"}, optional
        Segment window. Only ``"hann"`` is supported.

    Returns
    -------
    Tensor
        One-sided PSD with shape ``(n_freqs, num_channels)`` where
        ``n_freqs = segment_length // 2 + 1`` and ``num_channels`` is the
        product of non-time dimensions (``1`` for a 1-D series).

    Raises
    ------
    ValueError
        If shapes / overlap / window are invalid or the series is non-finite.

    Notes
    -----
    Scaling follows the usual Welch normalization
    ``2 / (U · fs_equiv)`` with unit sample rate ``fs = 1``, where ``U`` is the
    window power. DC and Nyquist bins are not doubled.

    References
    ----------
    Welch (1967), IEEE Trans. Audio Electroacoust. 15(2):70–73.
    https://doi.org/10.1109/TAU.1967.1161901 (``Welch1967``). Content match:
    section the record, form windowed (modified) periodograms, and average.
    """
    _require_finite("series", series)
    if window != "hann":
        msg = f'window must be "hann", got {window!r}'
        raise ValueError(msg)
    if not 0.0 <= overlap < 1.0:
        msg = f"overlap must be in [0, 1), got {overlap}"
        raise ValueError(msg)

    flat = _move_time_first(series, dim)
    time_len, num_channels = flat.shape
    if time_len < 2:
        msg = f"series time length must be >= 2, got {time_len}"
        raise ValueError(msg)

    seg_len = min(256, time_len) if segment_length is None else int(segment_length)
    if seg_len < 2:
        msg = f"segment_length must be >= 2, got {seg_len}"
        raise ValueError(msg)
    if seg_len > time_len:
        msg = f"segment_length ({seg_len}) cannot exceed time length ({time_len})"
        raise ValueError(msg)

    hop = max(1, int(round(seg_len * (1.0 - overlap))))
    starts = list(range(0, time_len - seg_len + 1, hop))
    if not starts:
        starts = [0]

    win = _hann_window(seg_len, dtype=flat.dtype, device=flat.device)
    window_power = torch.sum(win * win)
    # Unit sample rate; one-sided scaling with non-doubled endpoints.
    scale = 2.0 / (window_power * 1.0)

    accum = torch.zeros(
        seg_len // 2 + 1,
        num_channels,
        dtype=flat.dtype,
        device=flat.device,
    )
    for start in starts:
        segment = flat[start : start + seg_len] * win.unsqueeze(1)
        spectrum = torch.fft.rfft(segment, dim=0)
        periodogram = (spectrum.real.square() + spectrum.imag.square()) * scale
        periodogram[0].mul_(0.5)
        if seg_len % 2 == 0:
            periodogram[-1].mul_(0.5)
        accum = accum + periodogram
    return accum / len(starts)


def spectral_distance(
    prediction: Tensor,
    target: Tensor,
    *,
    dim: int = 0,
    eps: float = _EPS,
    segment_length: int | None = None,
    overlap: float = 0.5,
) -> Tensor:
    """Mean absolute log10-PSD distance between two series.

    Computes Welch PSDs for ``prediction`` and ``target``, then

    .. math::

        \\frac{1}{F C}\\sum_{f,c}
        \\bigl|\\log_{10}(P_{f,c}+\\varepsilon)
        - \\log_{10}(Q_{f,c}+\\varepsilon)\\bigr|

    where ``F`` is the number of frequency bins and ``C`` the number of
    channels. The log domain is used because PSD values often span many
    orders of magnitude; ``eps`` floors both spectra before the logarithm.

    Parameters
    ----------
    prediction : Tensor
        Predicted series (same layout as ``target``).
    target : Tensor
        Reference series.
    dim : int, optional
        Time axis. Default ``0``.
    eps : float, optional
        Positive floor inside the log. Default ``1e-12``.
    segment_length : int or None, optional
        Forwarded to :func:`power_spectral_density`.
    overlap : float, optional
        Forwarded to :func:`power_spectral_density`.

    Returns
    -------
    Tensor
        Non-negative scalar distance.

    Raises
    ------
    ValueError
        If shapes disagree, ``eps <= 0``, or inputs are non-finite.
    """
    if prediction.shape != target.shape:
        msg = (
            "prediction and target must have the same shape, got "
            f"{tuple(prediction.shape)} vs {tuple(target.shape)}"
        )
        raise ValueError(msg)
    if eps <= 0:
        msg = f"eps must be positive, got {eps}"
        raise ValueError(msg)
    pred_psd = power_spectral_density(
        prediction,
        dim=dim,
        segment_length=segment_length,
        overlap=overlap,
    )
    target_psd = power_spectral_density(
        target,
        dim=dim,
        segment_length=segment_length,
        overlap=overlap,
    )
    log_pred = torch.log10(pred_psd + eps)
    log_target = torch.log10(target_psd + eps)
    return torch.mean(torch.abs(log_pred - log_target))


def invariant_measure_distance(
    prediction: Tensor,
    target: Tensor,
    *,
    dim: int = 0,
    per_node: bool = False,
) -> Tensor:
    """1-D Wasserstein-1 distance between empirical marginals.

    For equal-length samples the exact 1-D Wasserstein-1 distance is

    .. math::

        W_1(\\mu,\\nu) = \\frac{1}{n}\\sum_{i=1}^{n}
        \\lvert x_{(i)} - y_{(i)}\\rvert

    (mean absolute difference of sorted samples). Time is taken along ``dim``;
    other axes are channels. When ``per_node=False`` (default), channels are
    pooled into one scalar (mean of per-channel :math:`W_1`). When
    ``per_node=True``, returns one distance per channel.

    Parameters
    ----------
    prediction : Tensor
        Predicted series.
    target : Tensor
        Reference series with the same shape as ``prediction``.
    dim : int, optional
        Time axis. Default ``0``.
    per_node : bool, optional
        If ``True``, return per-channel distances with shape
        ``(num_channels,)``. If ``False``, return a scalar mean.

    Returns
    -------
    Tensor
        Non-negative Wasserstein-1 distance(s).

    Raises
    ------
    ValueError
        If shapes disagree or inputs are non-finite.
    """
    if prediction.shape != target.shape:
        msg = (
            "prediction and target must have the same shape, got "
            f"{tuple(prediction.shape)} vs {tuple(target.shape)}"
        )
        raise ValueError(msg)
    _require_finite("prediction", prediction)
    _require_finite("target", target)

    pred = _move_time_first(prediction, dim)
    tgt = _move_time_first(target, dim)
    pred_sorted, _ = torch.sort(pred, dim=0)
    tgt_sorted, _ = torch.sort(tgt, dim=0)
    per_channel = torch.mean(torch.abs(pred_sorted - tgt_sorted), dim=0)
    if per_node:
        return per_channel
    return torch.mean(per_channel)


def _delay_embedding(series_1d: Tensor, embedding_dim: int, delay: int) -> Tensor:
    """Build a delay embedding with shape ``(n_vectors, embedding_dim)``.

    Parameters
    ----------
    series_1d : Tensor
        Scalar series with shape ``(T,)``.
    embedding_dim : int
        Embedding dimension ``m >= 2``.
    delay : int
        Lag between embedding coordinates in samples.

    Returns
    -------
    Tensor
        Embedded orbit with shape ``(T - (m - 1) * delay, m)``.
    """
    time_len = int(series_1d.numel())
    n_vectors = time_len - (embedding_dim - 1) * delay
    if n_vectors < 2:
        msg = (
            "series too short for delay embedding with "
            f"embedding_dim={embedding_dim}, delay={delay} "
            f"(length={time_len})"
        )
        raise ValueError(msg)
    return torch.stack(
        [series_1d[i * delay : i * delay + n_vectors] for i in range(embedding_dim)],
        dim=1,
    )


def _rosenstein_delay_theiler(
    series_1d: Tensor,
    *,
    embedding_dim: int,
    trajectory_len: int,
    min_neighbors: int,
    delay: int | None,
    theiler: int | None,
) -> tuple[int, int]:
    """Choose Rosenstein lag and Theiler window (autocorr / mean period).

    Parameters
    ----------
    series_1d : Tensor
        Scalar series with shape ``(T,)``.
    embedding_dim : int
        Delay-embedding dimension.
    trajectory_len : int
        Divergence trajectory length used when bounding automatic lag search.
    min_neighbors : int
        Minimum remaining orbit vectors required while searching for ``delay``.
    delay : int or None
        Explicit lag, or ``None`` to select from the autocorrelation drop to
        ``(1 - 1/e)`` of its maximum.
    theiler : int or None
        Explicit Theiler window, or ``None`` to use the mean FFT period.

    Returns
    -------
    tuple of int
        ``(delay, theiler)`` in samples.
    """
    n = int(series_1d.numel())
    max_tsep_factor = 0.25
    # Match the common FFT length used for joint lag / mean-period heuristics.
    n_fft = n * 2 - 1
    spectrum = torch.fft.rfft(series_1d, n=n_fft)
    if theiler is None:
        freqs = torch.fft.rfftfreq(n_fft, d=1.0, device=series_1d.device)
        psd = spectrum.real.square() + spectrum.imag.square()
        power = psd[1:].sum()
        if power <= 0:
            theiler = 1
        else:
            mean_freq = float((freqs[1:] * psd[1:]).sum() / power)
            theiler = int(math.ceil(1.0 / mean_freq)) if mean_freq > 0 else 1
        theiler = max(1, min(theiler, int(max_tsep_factor * n)))
    else:
        theiler = max(1, int(theiler))

    if delay is None:
        # Wiener–Khinchin autocorrelation; lag where ACF drops to (1-1/e) max.
        acorr = torch.fft.irfft(spectrum * torch.conj(spectrum), n=n_fft)
        acorr = torch.roll(acorr, shifts=n - 1)
        threshold = acorr[n - 1] * (1.0 - 1.0 / math.e)

        def _neighbors_remaining(lag: int) -> int:
            min_len = (embedding_dim - 1) * lag + trajectory_len + theiler * 2
            return max(0, n - min_len)

        delay = 1
        for lag in range(1, n):
            delay = lag
            if acorr[n - 1 + lag] < threshold or acorr[n - 1 - lag] < threshold:
                break
            if _neighbors_remaining(lag) < min_neighbors:
                break
    else:
        delay = int(delay)
        if delay < 1:
            msg = f"delay must be >= 1, got {delay}"
            raise ValueError(msg)
    return delay, theiler


def largest_lyapunov_exponent(
    series: Tensor,
    *,
    dim: int = 0,
    delay: int | None = None,
    embedding_dim: int = 10,
    fit_range: tuple[int, int] | None = None,
    dt: float = 1.0,
    theiler: int | None = None,
    trajectory_len: int = 20,
    fit_offset: int = 0,
    channel: int = 0,
    min_neighbors: int = 20,
) -> Tensor:
    """Estimate the largest Lyapunov exponent (Rosenstein et al., 1993).

    Reconstructs a delay embedding, finds nearest neighbors with a Theiler
    (temporal) exclusion window, averages ``log`` divergence of neighbor
    pairs over a short trajectory, and returns the least-squares slope
    converted to inverse-time units via ``dt``.

    Parameters
    ----------
    series : Tensor
        Real-valued series with time on ``dim``. If more than one channel is
        present after flattening non-time axes, only ``channel`` is used
        (scalar Rosenstein reconstruction).
    dim : int, optional
        Time axis. Default ``0``.
    delay : int or None, optional
        Embedding lag in samples. ``None`` (default) uses the lag where the
        autocorrelation drops to ``(1 - 1/e)`` of its maximum (Rosenstein).
    embedding_dim : int, optional
        Delay-embedding dimension. Default ``10``.
    fit_range : tuple of int or None, optional
        Inclusive ``(i_start, i_end)`` sample indices for the linear fit on
        the mean-log divergence curve. When ``None``, fit
        ``[fit_offset, trajectory_len)``.
    dt : float, optional
        Sampling period in the same time unit as the desired exponent.
        Default ``1.0`` (exponent per sample).
    theiler : int or None, optional
        Minimum temporal separation of neighbors in samples. ``None``
        (default) uses the mean period from a weighted FFT frequency.
    trajectory_len : int, optional
        Number of divergence samples ``k = 0, …, trajectory_len-1``.
        Default ``20``. Ignored as an upper bound when ``fit_range`` needs a
        longer curve (the curve is extended to ``fit_range[1] + 1``).
    fit_offset : int, optional
        Leading samples skipped when ``fit_range is None``. Default ``0``.
    channel : int, optional
        Channel index when ``series`` has multiple non-time axes.
        Default ``0``.
    min_neighbors : int, optional
        Minimum orbit vectors required while searching for an automatic lag.
        Default ``20``.

    Returns
    -------
    Tensor
        Scalar estimate of the largest Lyapunov exponent (nats per unit time
        matching ``dt``).

    Raises
    ------
    ValueError
        If inputs are non-finite, shapes / parameters are invalid, or too few
        finite divergence points remain for a fit.

    Notes
    -----
    Complexity is dominated by nearest-neighbor search over orbit vectors
    (``O(T^2)`` for a naive distance matrix). Prefer moderate series lengths
    for interactive use.

    References
    ----------
    Rosenstein, Collins & De Luca (1993), Physica D 65(1–2):117–134.
    https://doi.org/10.1016/0167-2789(93)90009-P
    (``Rosenstein1993Lyapunov``)
    """
    _require_finite("series", series)
    if embedding_dim < 2:
        msg = f"embedding_dim must be >= 2, got {embedding_dim}"
        raise ValueError(msg)
    if dt <= 0:
        msg = f"dt must be positive, got {dt}"
        raise ValueError(msg)
    if trajectory_len < 2:
        msg = f"trajectory_len must be >= 2, got {trajectory_len}"
        raise ValueError(msg)
    if fit_offset < 0:
        msg = f"fit_offset must be >= 0, got {fit_offset}"
        raise ValueError(msg)

    flat = _move_time_first(series, dim)
    if channel < 0 or channel >= flat.shape[1]:
        msg = (
            f"channel={channel} is out of range for series with "
            f"{flat.shape[1]} channel(s)"
        )
        raise ValueError(msg)
    series_1d = flat[:, channel].contiguous()

    if fit_range is not None:
        i_start, i_end = int(fit_range[0]), int(fit_range[1])
        if i_start < 0 or i_end < i_start:
            msg = f"fit_range must satisfy 0 <= i_start <= i_end, got {fit_range}"
            raise ValueError(msg)
        trajectory_len = max(trajectory_len, i_end + 1)

    delay_v, theiler_v = _rosenstein_delay_theiler(
        series_1d,
        embedding_dim=embedding_dim,
        trajectory_len=trajectory_len,
        min_neighbors=min_neighbors,
        delay=delay,
        theiler=theiler,
    )
    orbit = _delay_embedding(series_1d, embedding_dim, delay_v)
    n_orbit = orbit.shape[0]
    ntraj = n_orbit - trajectory_len + 1
    if ntraj <= theiler_v * 2 + 2:
        msg = (
            "series too short for Rosenstein search with "
            f"delay={delay_v}, theiler={theiler_v}, "
            f"trajectory_len={trajectory_len}, embedding_dim={embedding_dim}"
        )
        raise ValueError(msg)

    # Nearest neighbors among followable orbit vectors (Theiler exclusion).
    dists = torch.cdist(orbit[:ntraj], orbit[:ntraj])
    for i in range(ntraj):
        lo = max(0, i - theiler_v)
        hi = min(ntraj, i + theiler_v + 1)
        dists[i, lo:hi] = float("inf")
    if not torch.isfinite(dists).any(dim=1).all():
        msg = "Theiler window leaves no valid neighbor for at least one orbit vector"
        raise ValueError(msg)
    neighbor_index = torch.argmin(dists, dim=1)

    div_traj = torch.empty(trajectory_len, dtype=orbit.dtype, device=orbit.device)
    base = torch.arange(ntraj, device=orbit.device)
    for step in range(trajectory_len):
        delta = orbit[base + step] - orbit[neighbor_index + step]
        dist = torch.linalg.vector_norm(delta, dim=1)
        positive = dist > 0
        if not bool(positive.any()):
            div_traj[step] = float("-inf")
        else:
            div_traj[step] = torch.log(dist[positive]).mean()

    steps = torch.arange(trajectory_len, dtype=orbit.dtype, device=orbit.device)
    finite = torch.isfinite(div_traj)
    steps = steps[finite]
    curve = div_traj[finite]
    if fit_range is not None:
        i_start, i_end = int(fit_range[0]), int(fit_range[1])
        mask = (steps >= i_start) & (steps <= i_end)
        steps = steps[mask]
        curve = curve[mask]
    else:
        steps = steps[fit_offset:]
        curve = curve[fit_offset:]
    if steps.numel() < 2:
        msg = "not enough finite divergence points to fit a Lyapunov slope"
        raise ValueError(msg)

    steps_c = steps - steps.mean()
    curve_c = curve - curve.mean()
    denom = torch.sum(steps_c * steps_c)
    if float(denom.item()) <= 0:
        msg = "degenerate fit window for Lyapunov slope"
        raise ValueError(msg)
    slope_per_sample = torch.sum(steps_c * curve_c) / denom
    return slope_per_sample / dt


@dataclass(frozen=True)
class LongHorizonReport:
    """Summary of long-horizon distributional fidelity.

    Attributes
    ----------
    spectral_distance : float
        Log10-PSD distance from :func:`spectral_distance`.
    invariant_measure_distance : float
        Pooled 1-D Wasserstein-1 from :func:`invariant_measure_distance`.
    largest_lyapunov_exponent : float or None
        Rosenstein estimate on ``target`` when requested; otherwise ``None``.
    num_steps : int
        Number of time samples along the compared series.
    """

    spectral_distance: float
    invariant_measure_distance: float
    largest_lyapunov_exponent: float | None
    num_steps: int


def compute_long_horizon_report(
    prediction: Tensor,
    target: Tensor,
    *,
    dim: int = 0,
    eps: float = _EPS,
    lyapunov: bool = False,
    lyapunov_kwargs: dict[str, object] | None = None,
) -> LongHorizonReport:
    """Build a :class:`LongHorizonReport` for a prediction / target pair.

    Parameters
    ----------
    prediction : Tensor
        Predicted series.
    target : Tensor
        Reference series with the same shape.
    dim : int, optional
        Time axis. Default ``0``.
    eps : float, optional
        Floor for :func:`spectral_distance`.
    lyapunov : bool, optional
        If ``True``, estimate :func:`largest_lyapunov_exponent` on ``target``.
        Default ``False`` (nearest-neighbor search is ``O(T^2)``).
    lyapunov_kwargs : dict or None, optional
        Forwarded to :func:`largest_lyapunov_exponent` when ``lyapunov=True``.

    Returns
    -------
    LongHorizonReport
        Frozen summary.
    """
    if prediction.shape != target.shape:
        msg = (
            "prediction and target must have the same shape, got "
            f"{tuple(prediction.shape)} vs {tuple(target.shape)}"
        )
        raise ValueError(msg)
    time_dim = dim if dim >= 0 else prediction.ndim + dim
    num_steps = int(prediction.shape[time_dim])
    spec = float(spectral_distance(prediction, target, dim=dim, eps=eps).detach().cpu())
    w1 = float(invariant_measure_distance(prediction, target, dim=dim).detach().cpu())
    lle: float | None = None
    if lyapunov:
        kwargs = dict(lyapunov_kwargs or {})
        kwargs.setdefault("dim", dim)
        lle = float(largest_lyapunov_exponent(target, **kwargs).detach().cpu())
    return LongHorizonReport(
        spectral_distance=spec,
        invariant_measure_distance=w1,
        largest_lyapunov_exponent=lle,
        num_steps=num_steps,
    )


__all__ = [
    "LongHorizonReport",
    "compute_long_horizon_report",
    "invariant_measure_distance",
    "largest_lyapunov_exponent",
    "power_spectral_density",
    "spectral_distance",
]
