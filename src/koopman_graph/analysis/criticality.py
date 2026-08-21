"""Sliding-window spectral-gap and near-defectivity monitor.

Honesty contract
----------------
:func:`monitor_critical_transition` scores a *sequence* of already-computed
:class:`~koopman_graph.spectrum_types.KoopmanSpectrum` objects. The gap

.. math::

    \\gamma_t = \\min_{i\\neq j}\\lvert\\lambda_i(t)-\\lambda_j(t)\\rvert

and the windowed rate :math:`(\\gamma_{t-w+1}-\\gamma_t)/(\\tau_t-
\\tau_{t-w+1})` are heuristics. Positive rate means the gap shrank.
They are **not** a topology-criticality certificate, not an
early-warning score for infrastructure forecasting, and not a
replacement for
:meth:`~koopman_graph.operators.KoopmanOperator.stability_certificate`
or :func:`~koopman_graph.analysis.resolvent_norm_grid`. Near-defectivity
is a flag on :math:`\\kappa(V)`, not a Schur form and not
:class:`~koopman_graph.spectrum_types.DefectiveSpectrumError`.

This module does not implement Ghosh, *Intelligent Systems with
Applications*, 2025 (``Ghosh2025``). That paper is cited as related
literature and as the honesty ceiling for this helper.

See Also
--------
:class:`~koopman_graph.spectrum_types.SpectralDiagnostics`
    Per-spectrum :math:`\\kappa(V)` used for the near-defectivity flag.
:func:`~koopman_graph.operators.continuous_van_loan.matrix_log`
    Structured error on Jordan / singular eigenbases, distinct from
    this sequence monitor.

References
----------
Ghosh, R. (2025). Neural Koopman forecasting for critical transitions
in infrastructure networks. *Intelligent Systems with Applications*,
27, 200575. https://doi.org/10.1016/j.iswa.2025.200575
(``Ghosh2025``; related literature / non-goal — this helper does not
implement that forecasting method.)
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from koopman_graph.spectrum_types import (
    CONDITION_WARN,
    KoopmanSpectrum,
    _column_normalized_condition,
)


@dataclass(frozen=True)
class CriticalityReport:
    """Sliding-window spectral-gap trajectory and near-defectivity flags.

    Attributes
    ----------
    spectral_gap : Tensor
        Per-spectrum gap :math:`\\gamma_t` with shape ``(T,)``. Same
        units as the stored eigenvalues (dimensionless for discrete
        ``K``; 1 / time unit for generator spectra). Non-negative.
    gap_closure_rate : Tensor
        Windowed :math:`(\\gamma_{t-w+1}-\\gamma_t)/(\\tau_t-
        \\tau_{t-w+1})` with shape ``(T - window + 1,)``. Positive
        values mean the gap shrank. Units: eigenvalue units / sample
        time unit (1 / step when ``sample_times`` is omitted).
    near_defective : Tensor
        Boolean mask with shape ``(T,)``. ``True`` when
        :math:`\\kappa(V)` is non-finite or exceeds ``CONDITION_WARN``.
    max_gap_closure_rate : float
        Maximum of ``gap_closure_rate`` (the scalar “score”).
    """

    spectral_gap: Tensor
    gap_closure_rate: Tensor
    near_defective: Tensor
    max_gap_closure_rate: float

    def __post_init__(self) -> None:
        """Validate trajectory shapes and non-negativity of the gap.

        Raises
        ------
        ValueError
            If tensors are empty or misaligned, the gap is negative, or
            ``max_gap_closure_rate`` is NaN.
        """
        gap = self.spectral_gap
        rate = self.gap_closure_rate
        flags = self.near_defective
        if gap.ndim != 1 or int(gap.numel()) == 0:
            raise ValueError("spectral_gap must be a nonempty 1-D tensor")
        n_times = int(gap.numel())
        if flags.ndim != 1 or int(flags.numel()) != n_times:
            msg = (
                "near_defective must have shape (T,), "
                f"got {tuple(flags.shape)} for T={n_times}"
            )
            raise ValueError(msg)
        if flags.dtype != torch.bool:
            raise ValueError("near_defective must be a boolean tensor")
        if rate.ndim != 1 or int(rate.numel()) == 0 or int(rate.numel()) > n_times:
            msg = (
                "gap_closure_rate must be a nonempty 1-D tensor with "
                f"length <= T={n_times}, got {tuple(rate.shape)}"
            )
            raise ValueError(msg)
        if bool((gap < 0).any().item()):
            raise ValueError("spectral_gap must be non-negative")
        if math.isnan(self.max_gap_closure_rate):
            raise ValueError("max_gap_closure_rate must not be NaN")


def _pairwise_spectral_gap(eigenvalues: Tensor) -> float:
    """Return :math:`\\min_{i\\neq j}|\\lambda_i-\\lambda_j|`.

    Parameters
    ----------
    eigenvalues : Tensor
        1-D eigenvalues, real or complex, length ``n >= 2``.

    Returns
    -------
    float
        Non-negative gap in the same units as ``eigenvalues``.

    Raises
    ------
    ValueError
        If there are fewer than two eigenvalues or any value is
        non-finite.
    """
    values = eigenvalues.reshape(-1)
    n_modes = int(values.numel())
    if n_modes < 2:
        msg = (
            f"spectral gap requires at least two eigenvalues, got latent_dim={n_modes}"
        )
        raise ValueError(msg)
    if values.is_complex():
        finite = torch.isfinite(values.real) & torch.isfinite(values.imag)
    else:
        finite = torch.isfinite(values)
    if not bool(finite.all().item()):
        raise ValueError("eigenvalues must be finite")
    delta = values.unsqueeze(0) - values.unsqueeze(1)
    distance = delta.abs()
    eye = torch.eye(n_modes, dtype=torch.bool, device=values.device)
    off_diagonal = distance.masked_fill(eye, math.inf)
    gap = off_diagonal.min()
    return float(gap.item())


def _resolve_sample_times(
    n_times: int,
    sample_times: Tensor | Sequence[float] | None,
) -> Tensor:
    """Return strictly increasing sample times of length ``n_times``.

    Parameters
    ----------
    n_times : int
        Number of spectra ``T``.
    sample_times : Tensor or sequence of float or None
        Caller times, or ``None`` for ``0, 1, ..., T-1``.

    Returns
    -------
    Tensor
        1-D float64 times with shape ``(T,)``.

    Raises
    ------
    ValueError
        If the length differs from ``T``, any time is non-finite, or
        the sequence is not strictly increasing.
    """
    if sample_times is None:
        return torch.arange(n_times, dtype=torch.float64)
    times = torch.as_tensor(sample_times, dtype=torch.float64).reshape(-1)
    if int(times.numel()) != n_times:
        msg = f"sample_times must have length T={n_times}, got {int(times.numel())}"
        raise ValueError(msg)
    if not bool(torch.isfinite(times).all().item()):
        raise ValueError("sample_times must be finite")
    if n_times >= 2 and bool((times[1:] - times[:-1] <= 0).any().item()):
        raise ValueError("sample_times must be strictly increasing")
    return times


def _near_defective_flag(spectrum: KoopmanSpectrum) -> bool:
    """Return True when column-normalized :math:`\\kappa(V)` is unusable.

    Parameters
    ----------
    spectrum : KoopmanSpectrum
        Spectrum whose eigenbasis is tested.

    Returns
    -------
    bool
        ``True`` when :math:`\\kappa(V)` is non-finite or exceeds
        ``CONDITION_WARN``.
    """
    if spectrum.diagnostics is not None:
        kappa = spectrum.diagnostics.eigenvector_condition
    else:
        kappa = _column_normalized_condition(spectrum.eigenvectors)
    return (not math.isfinite(kappa)) or kappa > CONDITION_WARN


def monitor_critical_transition(
    spectra: Sequence[KoopmanSpectrum],
    *,
    window: int,
    sample_times: Tensor | Sequence[float] | None = None,
) -> CriticalityReport:
    """Score spectral-gap closure and near-defectivity on a spectrum sequence.

    The pairwise gap :math:`\\gamma_t` uses every eigenpair, not
    consecutive magnitude-sorted neighbors. The windowed rate is

    .. math::

        s_t = \\frac{\\gamma_{t-w+1} - \\gamma_t}{\\tau_t - \\tau_{t-w+1}}

    so a **positive** rate means the gap closed. Default :math:`\\tau`
    is unit spacing (one step per spectrum). This is a heuristic
    diagnostic, not a Ghosh-grade critical-transition certificate.

    Parameters
    ----------
    spectra : sequence of KoopmanSpectrum
        Nonempty sequence of length ``T``. Every spectrum must have the
        same ``latent_dim >= 2``.
    window : int
        Sliding-window length ``w``. Must satisfy ``2 <= w <= T``.
    sample_times : Tensor or sequence of float or None, optional
        Strictly increasing times of length ``T`` in the caller time
        unit. ``None`` (default) uses ``0, 1, ..., T-1``.

    Returns
    -------
    CriticalityReport
        Gap trajectory, windowed closure rates, near-defectivity flags,
        and ``max_gap_closure_rate``.

    Raises
    ------
    ValueError
        If ``spectra`` is empty, widths differ, ``latent_dim < 2``,
        ``window`` is invalid, eigenvalues are non-finite, or
        ``sample_times`` is not strictly increasing of length ``T``.
    """
    n_times = len(spectra)
    if n_times == 0:
        raise ValueError("spectra must be a nonempty sequence")
    if isinstance(window, bool) or not isinstance(window, int) or window < 2:
        msg = f"window must be an int >= 2, got {window!r}"
        raise ValueError(msg)
    if window > n_times:
        msg = f"window must be <= len(spectra) ({n_times}), got {window}"
        raise ValueError(msg)

    latent_dim = int(spectra[0].eigenvalues.numel())
    gaps: list[float] = []
    flags: list[bool] = []
    for index, spectrum in enumerate(spectra):
        width = int(spectrum.eigenvalues.numel())
        if width != latent_dim:
            msg = (
                "all spectra must share latent_dim="
                f"{latent_dim}, got {width} at index {index}"
            )
            raise ValueError(msg)
        gaps.append(_pairwise_spectral_gap(spectrum.eigenvalues))
        flags.append(_near_defective_flag(spectrum))

    times = _resolve_sample_times(n_times, sample_times)
    gap_tensor = torch.tensor(gaps, dtype=torch.float64)
    n_windows = n_times - window + 1
    gap_start = gap_tensor[:n_windows]
    gap_end = gap_tensor[window - 1 :]
    delta_tau = times[window - 1 :] - times[:n_windows]
    rates = (gap_start - gap_end) / delta_tau
    max_rate = float(rates.max().item())
    return CriticalityReport(
        spectral_gap=gap_tensor,
        gap_closure_rate=rates,
        near_defective=torch.tensor(flags, dtype=torch.bool),
        max_gap_closure_rate=max_rate,
    )
