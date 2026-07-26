"""Spectral analysis capability package.

Capability layout
-----------------
``spectrum``
    Analysis-owned :func:`~koopman_graph.analysis.decode_mode_shapes` plus
    re-exports of neutral-leaf
    :func:`~koopman_graph.spectrum_types.compute_spectrum`,
    :func:`~koopman_graph.spectrum_types.compute_generator_spectrum`, and
    :func:`~koopman_graph.spectrum_types.discrete_spectrum_at_delta_t`.
``similarity``
    Spectral distances, KoopSTD,
    :func:`~koopman_graph.analysis.resolve_spectrum`, and
    :func:`~koopman_graph.analysis.dynamical_similarity`.
``anomaly``
    Threshold calibration and anomaly detection over reference spectra.
``plotting``
    Discrete complex-plane spectrum figures
    (:func:`~koopman_graph.analysis.plot_spectrum`).
``topology_estimation``
    DMD-estimated inter-node coupling
    (:func:`~koopman_graph.analysis.estimate_coupling_from_snapshots`,
    :class:`~koopman_graph.analysis.CouplingEstimate`).
``sindy``
    Sparse identification of nonlinear dynamics on learned latents
    (:func:`~koopman_graph.analysis.identify_sparse_dynamics`,
    :class:`~koopman_graph.analysis.SINDyReport`). Interprets the model's
    latent map — not physical ground-truth ODEs.
``clustering``
    Koopman spectral clustering
    (:func:`~koopman_graph.analysis.koopman_spectral_clustering`,
    :class:`~koopman_graph.analysis.ClusteringResult`). Node embeddings from
    leading eigenmodes; quality inherits the operator / spectrum.
``residuals``
    Data-driven spectral residuals
    (:func:`~koopman_graph.analysis.spectral_residuals`,
    :class:`~koopman_graph.analysis.SpectralResidualReport`). A posteriori
    diagnostic in the learned observable norm — **not** a certified ResDMD
    bound.

:class:`~koopman_graph.spectrum_types.KoopmanSpectrum` and the spectrum
assembly helpers stay in the neutral top-level
:mod:`koopman_graph.spectrum_types` module so operators and the model
façade never import this package for eigendecomposition. They are
re-exported here for the public analysis surface.
"""

from koopman_graph.analysis.anomaly import (
    AnomalyDetectionResult,
    AnomalyThresholdMethod,
    calibrate_anomaly_threshold,
    detect_anomaly,
)
from koopman_graph.analysis.clustering import (
    ClusteringResult,
    koopman_spectral_clustering,
)
from koopman_graph.analysis.plotting import SpectrumLimits, plot_spectrum
from koopman_graph.analysis.residuals import (
    SpectralResidualReport,
    spectral_residuals,
)
from koopman_graph.analysis.similarity import (
    SpectrumDistanceMethod,
    SpectrumSource,
    dynamical_similarity,
    koopman_std,
    resolve_spectrum,
    spectrum_distance,
)
from koopman_graph.analysis.sindy import SINDyReport, identify_sparse_dynamics
from koopman_graph.analysis.spectrum import (
    compute_generator_spectrum,
    compute_spectrum,
    decode_mode_shapes,
    discrete_spectrum_at_delta_t,
)
from koopman_graph.analysis.topology_estimation import (
    CouplingEstimate,
    estimate_coupling_from_snapshots,
)
from koopman_graph.spectrum_types import KoopmanSpectrum

__all__ = [
    "AnomalyDetectionResult",
    "AnomalyThresholdMethod",
    "ClusteringResult",
    "CouplingEstimate",
    "KoopmanSpectrum",
    "SINDyReport",
    "SpectralResidualReport",
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
    "estimate_coupling_from_snapshots",
    "identify_sparse_dynamics",
    "koopman_spectral_clustering",
    "koopman_std",
    "plot_spectrum",
    "resolve_spectrum",
    "spectral_residuals",
    "spectrum_distance",
]
