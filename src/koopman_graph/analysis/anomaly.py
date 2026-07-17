"""Anomaly detection from Koopman spectral distances."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch

from koopman_graph.analysis.similarity import (
    SpectrumDistanceMethod,
    koopman_std,
    spectrum_distance,
)
from koopman_graph.spectrum_types import KoopmanSpectrum

if TYPE_CHECKING:
    from collections.abc import Sequence

AnomalyThresholdMethod = Literal["percentile", "mean_std"]


@dataclass(frozen=True)
class AnomalyDetectionResult:
    """Outcome of comparing a test spectrum against reference spectra.

    Attributes
    ----------
    is_anomaly : bool
        ``True`` when ``distance`` exceeds ``threshold``.
    distance : float
        Mean spectral distance from ``test_spectrum`` to each reference.
    reference_mean_distance : float
        Mean pairwise distance among reference spectra (baseline spread).
        Useful for calibrating ``threshold`` from in-distribution variation.
    """

    is_anomaly: bool
    distance: float
    reference_mean_distance: float


def calibrate_anomaly_threshold(
    reference_spectra: Sequence[KoopmanSpectrum],
    method: AnomalyThresholdMethod = "percentile",
    *,
    distance_method: SpectrumDistanceMethod = "wasserstein",
    num_modes: int | None = None,
    q: float = 95.0,
    k: float = 2.0,
) -> float:
    """Derive an anomaly threshold from pairwise reference-spectrum distances.

    Builds the upper triangle of :func:`~koopman_graph.analysis.koopman_std`
    among ``reference_spectra`` and summarizes those in-distribution distances.
    Pass the returned scalar to :func:`detect_anomaly` as ``threshold``.

    Calibration uses only reference replicates. It does not validate operating
    limits: few replicates, metric choice, and trajectory length all affect the
    scale. Prefer held-out nominal checks before deploying a threshold.

    Parameters
    ----------
    reference_spectra : sequence of KoopmanSpectrum
        At least two in-distribution reference spectra.
    method : {"percentile", "mean_std"}, optional
        Summary rule. ``"percentile"`` returns the ``q``-th percentile of
        pairwise distances. ``"mean_std"`` returns ``mean + k * std`` using the
        population standard deviation (``unbiased=False``) over the pairwise
        distances. Default is ``"percentile"``.
    distance_method : {"wasserstein", "subspace_angle"}, optional
        Spectral distance forwarded to :func:`~koopman_graph.analysis.koopman_std`.
        Default is ``"wasserstein"``.
    num_modes : int or None, optional
        Leading modes for subspace-angle comparisons.
    q : float, optional
        Percentile in ``[0, 100]`` when ``method="percentile"``. Default is
        ``95``.
    k : float, optional
        Non-negative multiplier for the standard-deviation term when
        ``method="mean_std"``. Uses the population standard deviation
        (``unbiased=False``). Default is ``2``.

    Returns
    -------
    float
        Scalar threshold in the same units as
        :func:`~koopman_graph.analysis.spectrum_distance`.

    Raises
    ------
    ValueError
        If fewer than two references are provided, ``method`` is unknown, or
        ``q`` / ``k`` are out of range.
    """
    if len(reference_spectra) < 2:
        raise ValueError("reference_spectra must contain at least two spectra")

    matrix = koopman_std(reference_spectra, distance_method, num_modes=num_modes)
    mask = torch.triu(torch.ones_like(matrix, dtype=torch.bool), diagonal=1)
    pairwise = matrix[mask]

    if method == "percentile":
        if not 0.0 <= q <= 100.0:
            msg = f"q must be in [0, 100], got {q}"
            raise ValueError(msg)
        return torch.quantile(pairwise, q / 100.0).item()

    if method == "mean_std":
        if k < 0.0:
            msg = f"k must be >= 0, got {k}"
            raise ValueError(msg)
        mean = pairwise.mean()
        if pairwise.numel() > 1:
            std = pairwise.std(unbiased=False)
        else:
            std = mean.new_zeros(())
        return (mean + k * std).item()

    msg = f"method must be 'percentile' or 'mean_std', got {method!r}"
    raise ValueError(msg)


def detect_anomaly(
    reference_spectra: Sequence[KoopmanSpectrum],
    test_spectrum: KoopmanSpectrum,
    threshold: float,
    method: SpectrumDistanceMethod = "wasserstein",
    *,
    num_modes: int | None = None,
) -> AnomalyDetectionResult:
    """Flag a test spectrum as anomalous when it is far from reference dynamics.

    ``threshold`` is required. Derive it from in-distribution reference
    spectra with :func:`calibrate_anomaly_threshold` (percentile or
    mean-plus-``k``-std of pairwise
    :func:`~koopman_graph.analysis.koopman_std` distances), or supply a
    domain-specific operating limit.

    Parameters
    ----------
    reference_spectra : sequence of KoopmanSpectrum
        In-distribution reference spectra (healthy baselines or known regimes).
    test_spectrum : KoopmanSpectrum
        Spectrum to evaluate.
    threshold : float
        Maximum acceptable mean distance from ``test_spectrum`` to the
        references. Distances above this value are flagged as anomalies.
    method : {"wasserstein", "subspace_angle"}, optional
        Distance definition. Default is ``"wasserstein"``.
    num_modes : int or None, optional
        Leading modes for subspace-angle comparisons.

    Returns
    -------
    AnomalyDetectionResult
        ``is_anomaly`` is ``True`` when the mean reference distance exceeds
        ``threshold``.

    Raises
    ------
    ValueError
        If ``reference_spectra`` is empty or ``threshold`` is negative.
    """
    if not reference_spectra:
        raise ValueError("reference_spectra must be non-empty")
    if threshold < 0.0:
        msg = f"threshold must be >= 0, got {threshold}"
        raise ValueError(msg)

    test_distances = [
        spectrum_distance(test_spectrum, reference, method, num_modes=num_modes)
        for reference in reference_spectra
    ]
    distance = torch.stack(test_distances).mean().item()

    if len(reference_spectra) == 1:
        reference_mean_distance = 0.0
    else:
        reference_matrix = koopman_std(reference_spectra, method, num_modes=num_modes)
        pair_mask = torch.triu(
            torch.ones_like(reference_matrix, dtype=torch.bool),
            diagonal=1,
        )
        reference_mean_distance = reference_matrix[pair_mask].mean().item()

    return AnomalyDetectionResult(
        is_anomaly=distance > threshold,
        distance=distance,
        reference_mean_distance=reference_mean_distance,
    )
