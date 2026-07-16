"""KoopmanGraph: topology-aware Koopman dynamics on graphs.

Public API
----------
``GraphKoopmanModel``
    End-to-end encode → Koopman advance → decode model.
``GNNEncoder``, ``GATEncoder``
    GNN encoders for latent lifting.
``GNNDecoder``
    GNN decoder for physical reconstruction.
``KoopmanOperator``
    Learnable finite-dimensional Koopman matrix.
``ContinuousKoopmanOperator``
    Continuous-time Koopman generator integrated via matrix exponentials.
``KoopmanSpectrum``
    Eigendecomposition and continuous-time mode characteristics.
``compute_spectrum``, ``decode_mode_shapes``
    Spectral analysis and latent-to-spatial mode decoding helpers.
``spectrum_distance``, ``koopman_std``, ``dynamical_similarity``
    Dynamical similarity metrics for comparing Koopman spectra.
``detect_anomaly``, ``calibrate_anomaly_threshold``
    Anomaly detection and reference-based threshold calibration.
``DMDBaseline``, ``EDMDBaseline``, ``DMDcBaseline``
    Classical topology-agnostic Koopman baselines.
``GraphSnapshotSequence``, ``WindowSampler``
    Container and fixed-length mini-batch sampler for graph snapshots.
``TemporalSplit``, ``temporal_split``
    Temporal train/validation/test splitting utilities.
``EvaluationResult``, ``evaluate_forecast``, ``mae``, ``rmse``, ``mape``
    Multi-horizon forecast evaluation metrics.
``ForwardConsistencyLoss``
    Latent-space linear evolution consistency loss.
``BackwardConsistencyLoss``
    Latent-space inverse linear evolution consistency loss.
``EigenvalueRegularizationLoss``
    Unit-circle eigenvalue hinge penalty for operator stability.
``FitHistory``
    Training history returned by :meth:`~koopman_graph.model.GraphKoopmanModel.fit`.
``RecursiveKoopmanAdapter``
    Recursive least-squares online Koopman operator adaptation.
``graph_laplacian_features``
    Normalized Laplacian physics features for hybrid Koopman latents.
``GraphKoopmanEnv``
    Gymnasium environment for latent-space closed-loop control.
``LossWeights``
    Weights for reconstruction and consistency loss terms.
``__version__``
    Package version string.
"""

from koopman_graph.adaptation import AdaptationStepResult, RecursiveKoopmanAdapter
from koopman_graph.analysis import (
    AnomalyDetectionResult,
    KoopmanSpectrum,
    calibrate_anomaly_threshold,
    compute_generator_spectrum,
    compute_spectrum,
    decode_mode_shapes,
    detect_anomaly,
    discrete_spectrum_at_delta_t,
    dynamical_similarity,
    koopman_std,
    spectrum_distance,
)
from koopman_graph.baselines import DMDBaseline, DMDcBaseline, EDMDBaseline
from koopman_graph.continuous import ContinuousKoopmanOperator
from koopman_graph.data import (
    GraphSnapshotSequence,
    TemporalSplit,
    WindowSampler,
    temporal_split,
)
from koopman_graph.decoder import GNNDecoder
from koopman_graph.encoder import GATEncoder, GNNEncoder
from koopman_graph.env import GraphKoopmanEnv
from koopman_graph.losses import (
    BackwardConsistencyLoss,
    EigenvalueRegularizationLoss,
    ForwardConsistencyLoss,
)
from koopman_graph.metrics import (
    EvaluationResult,
    HorizonMetrics,
    evaluate_forecast,
    mae,
    mape,
    rmse,
)
from koopman_graph.model import GraphKoopmanModel
from koopman_graph.observables import graph_laplacian_features
from koopman_graph.operator import KoopmanOperator
from koopman_graph.training import FitHistory, LossWeights

__all__ = [
    "AdaptationStepResult",
    "AnomalyDetectionResult",
    "BackwardConsistencyLoss",
    "EigenvalueRegularizationLoss",
    "EvaluationResult",
    "DMDBaseline",
    "DMDcBaseline",
    "EDMDBaseline",
    "FitHistory",
    "ForwardConsistencyLoss",
    "GATEncoder",
    "GNNDecoder",
    "GNNEncoder",
    "GraphKoopmanEnv",
    "GraphKoopmanModel",
    "GraphSnapshotSequence",
    "HorizonMetrics",
    "KoopmanOperator",
    "KoopmanSpectrum",
    "RecursiveKoopmanAdapter",
    "LossWeights",
    "TemporalSplit",
    "WindowSampler",
    "__version__",
    "ContinuousKoopmanOperator",
    "calibrate_anomaly_threshold",
    "compute_generator_spectrum",
    "compute_spectrum",
    "detect_anomaly",
    "discrete_spectrum_at_delta_t",
    "decode_mode_shapes",
    "dynamical_similarity",
    "evaluate_forecast",
    "graph_laplacian_features",
    "koopman_std",
    "mae",
    "mape",
    "rmse",
    "spectrum_distance",
    "temporal_split",
]
__version__ = "0.3.0"
