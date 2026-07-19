"""Spectral analysis capability package.

Capability layout
-----------------
``spectrum``
    Discrete/continuous spectrum helpers and
    :func:`~koopman_graph.analysis.decode_mode_shapes`. Discrete
    :func:`~koopman_graph.analysis.compute_spectrum` is re-exported from
    the neutral :mod:`koopman_graph.spectrum_types` leaf.
``similarity``
    Spectral distances, KoopSTD,
    :func:`~koopman_graph.analysis.resolve_spectrum`, and
    :func:`~koopman_graph.analysis.dynamical_similarity`.
``anomaly``
    Threshold calibration and anomaly detection over reference spectra.
``plotting``
    Discrete complex-plane spectrum figures
    (:func:`~koopman_graph.analysis.plot_spectrum`).

:class:`~koopman_graph.spectrum_types.KoopmanSpectrum` stays in the neutral
top-level :mod:`koopman_graph.spectrum_types` module to preserve dependency
direction. Discrete ``compute_spectrum`` also lives there, and both are
re-exported here for the public analysis surface.
"""

from koopman_graph.analysis.anomaly import (
    AnomalyDetectionResult,
    AnomalyThresholdMethod,
    calibrate_anomaly_threshold,
    detect_anomaly,
)
from koopman_graph.analysis.plotting import SpectrumLimits, plot_spectrum
from koopman_graph.analysis.similarity import (
    SpectrumDistanceMethod,
    SpectrumSource,
    dynamical_similarity,
    koopman_std,
    resolve_spectrum,
    spectrum_distance,
)
from koopman_graph.analysis.spectrum import (
    compute_generator_spectrum,
    compute_spectrum,
    decode_mode_shapes,
    discrete_spectrum_at_delta_t,
)
from koopman_graph.spectrum_types import KoopmanSpectrum

__all__ = [
    "AnomalyDetectionResult",
    "AnomalyThresholdMethod",
    "KoopmanSpectrum",
    "SpectrumDistanceMethod",
    "SpectrumLimits",
    "SpectrumSource",
    "calibrate_anomaly_threshold",
    "compute_generator_spectrum",
    "compute_spectrum",
    "decode_mode_shapes",
    "detect_anomaly",
    "discrete_spectrum_at_delta_t",
    "dynamical_similarity",
    "koopman_std",
    "plot_spectrum",
    "resolve_spectrum",
    "spectrum_distance",
]
