"""Spectral analysis capability package.

Capability layout
-----------------
``spectrum``
    Analysis-owned :func:`~koopman_graph.analysis.decode_mode_shapes` and
    :func:`~koopman_graph.analysis.attribute_mode_energy` (interpretive
    type / relation energy fractions on assembled ``K_eff``) plus
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
``resdmd``
    Colbrook–Townsend ResDMD on a **fixed** dictionary
    (:func:`~koopman_graph.analysis.resdmd`,
    :class:`~koopman_graph.analysis.ResDMDReport`). Finite-dictionary
    Galerkin residual bounds — **not**
    :func:`~koopman_graph.analysis.spectral_residuals` and not
    infinite-dimensional pseudospectra / spectral measures.
``pseudospectra``
    Finite-matrix resolvent-norm grids and empirical eigenvalue measures
    (:func:`~koopman_graph.analysis.resolvent_norm_grid`,
    :func:`~koopman_graph.analysis.empirical_spectral_measure`,
    :class:`~koopman_graph.analysis.ResolventNormGrid`,
    :class:`~koopman_graph.analysis.EmpiricalSpectralMeasure`). MVP on
    assembled ``K`` / ``K_eff`` — **not** certified infinite-dimensional
    ResDMD pseudospectra.
``joint_stability``
    Joint bounds / certificates on assembled ``ρ(K_eff)``
    (:func:`~koopman_graph.analysis.gershgorin_radius_bound`,
    :func:`~koopman_graph.analysis.schur_radius_bound`,
    :func:`~koopman_graph.analysis.lyapunov_joint_bound`,
    :class:`~koopman_graph.analysis.JointStabilityCertificate`). Default
    Gershgorin is sufficient, not tight; opt-in Schur / Lyapunov kinds are
    post-hoc on assembled ``K_eff`` under size ceilings — distinct from
    factor ``bound_metric`` /
    :class:`~koopman_graph.operators.StabilityCertificate` and from soft
    assembled eigenvalue regularization.

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
from koopman_graph.analysis.joint_stability import (
    MAX_JOINT_LYAPUNOV_SIZE,
    MAX_JOINT_SCHUR_SIZE,
    JointStabilityCertificate,
    build_joint_stability_certificate,
    gershgorin_radius_bound,
    lyapunov_joint_bound,
    schur_radius_bound,
)
from koopman_graph.analysis.plotting import SpectrumLimits, plot_spectrum
from koopman_graph.analysis.pseudospectra import (
    EmpiricalSpectralMeasure,
    ResolventNormGrid,
    empirical_spectral_measure,
    resolvent_norm_grid,
)
from koopman_graph.analysis.resdmd import ResDMDReport, resdmd
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
    ModeEnergyAttribution,
    attribute_mode_energy,
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
    "EmpiricalSpectralMeasure",
    "JointStabilityCertificate",
    "KoopmanSpectrum",
    "MAX_JOINT_LYAPUNOV_SIZE",
    "MAX_JOINT_SCHUR_SIZE",
    "ModeEnergyAttribution",
    "ResDMDReport",
    "ResolventNormGrid",
    "SINDyReport",
    "SpectralResidualReport",
    "SpectrumDistanceMethod",
    "SpectrumLimits",
    "SpectrumSource",
    "attribute_mode_energy",
    "build_joint_stability_certificate",
    "calibrate_anomaly_threshold",
    "compute_generator_spectrum",
    "compute_spectrum",
    "decode_mode_shapes",
    "detect_anomaly",
    "discrete_spectrum_at_delta_t",
    "dynamical_similarity",
    "empirical_spectral_measure",
    "estimate_coupling_from_snapshots",
    "gershgorin_radius_bound",
    "identify_sparse_dynamics",
    "koopman_spectral_clustering",
    "koopman_std",
    "lyapunov_joint_bound",
    "plot_spectrum",
    "resdmd",
    "resolvent_norm_grid",
    "resolve_spectrum",
    "schur_radius_bound",
    "spectral_residuals",
    "spectrum_distance",
]
