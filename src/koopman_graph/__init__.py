"""KoopmanGraph: topology-aware Koopman dynamics on graphs.

Public API
----------
``GraphKoopmanModel``
    End-to-end encode → Koopman advance → decode model.
``GNNEncoder``, ``GATEncoder``
    GNN encoders for latent lifting.
``DelayEmbeddingEncoder``
    Hankel / delay-coordinate wrapper around a sized base encoder.
``GNNDecoder``, ``GATDecoder``
    GNN decoders for physical reconstruction.
``KoopmanOperator``
    Learnable finite-dimensional Koopman matrix.
``ContinuousKoopmanOperator``
    Continuous-time Koopman generator integrated via matrix exponentials.
``GraphKoopmanOperator``
    Discrete networked Koopman step with self/neighbor coupling via ``edge_index``.
``KoopmanSpectrum``, ``compute_spectrum``
    Primary spectral analysis entrypoints. Specialized helpers
    (``compute_generator_spectrum``, ``discrete_spectrum_at_delta_t``,
    ``decode_mode_shapes``, ``spectrum_distance``, ``koopman_std``,
    ``dynamical_similarity``, ``detect_anomaly``,
    ``calibrate_anomaly_threshold``, ``AnomalyDetectionResult``) live in
    :mod:`koopman_graph.analysis`.
``DMDBaseline``, ``EDMDBaseline``, ``DMDcBaseline``
    Classical topology-agnostic Koopman baselines.
``GraphSnapshotSequence``, ``MultiTrajectory``, ``WindowSampler``
    Container, explicit multi-trajectory wrapper, and fixed-length mini-batch
    sampler for graph snapshots. ``as_multi_trajectory`` lives in
    :mod:`koopman_graph.data`.
``TemporalSplit``, ``temporal_split``
    Temporal train/validation/test splitting utilities.
``EvaluationResult``, ``evaluate_forecast``
    Multi-horizon forecast evaluation entrypoints. Low-level metrics
    (``mae``, ``rmse``, ``mape``, ``HorizonMetrics``) live in
    :mod:`koopman_graph.metrics`.
``ForwardConsistencyLoss``
    Latent-space linear evolution consistency loss.
``BackwardConsistencyLoss``
    Latent-space inverse linear evolution consistency loss.
``EigenvalueRegularizationLoss``
    Unit-circle eigenvalue hinge penalty for operator stability.
``FitHistory``, ``LossWeights``
    Training history and loss-weight schedule for ``fit``.
``RecursiveKoopmanAdapter``
    Recursive least-squares online Koopman operator adaptation.
    ``AdaptationStepResult``, ``KoopmanObserver``, and ``FilterResult`` live in
    :mod:`koopman_graph.adaptation`.
``GraphKoopmanEnv``
    Gymnasium environment for latent-space closed-loop control.
``__version__``
    Package version string.

Physics-informed helpers such as ``graph_laplacian_features`` live in
:mod:`koopman_graph.observables`. Power-user modules such as
:mod:`koopman_graph.graph_utils`, :mod:`koopman_graph.nn`,
:mod:`koopman_graph.protocols`, and :mod:`koopman_graph.spectrum_types` are
importable but intentionally omitted from ``__all__`` (encoder/decoder
*classes* remain public). See the architecture docs for the public vs
power-user contract.
"""

from koopman_graph.adaptation import RecursiveKoopmanAdapter
from koopman_graph.analysis import KoopmanSpectrum, compute_spectrum
from koopman_graph.baselines import DMDBaseline, DMDcBaseline, EDMDBaseline
from koopman_graph.data import (
    GraphSnapshotSequence,
    MultiTrajectory,
    TemporalSplit,
    WindowSampler,
    temporal_split,
)
from koopman_graph.env import GraphKoopmanEnv
from koopman_graph.losses import (
    BackwardConsistencyLoss,
    EigenvalueRegularizationLoss,
    ForwardConsistencyLoss,
)
from koopman_graph.metrics import EvaluationResult, evaluate_forecast
from koopman_graph.model import GraphKoopmanModel
from koopman_graph.nn import (
    DelayEmbeddingEncoder,
    GATDecoder,
    GATEncoder,
    GNNDecoder,
    GNNEncoder,
)
from koopman_graph.operators import (
    ContinuousKoopmanOperator,
    GraphKoopmanOperator,
    KoopmanOperator,
)
from koopman_graph.training import FitHistory, LossWeights

__all__ = [
    "BackwardConsistencyLoss",
    "ContinuousKoopmanOperator",
    "DelayEmbeddingEncoder",
    "DMDBaseline",
    "DMDcBaseline",
    "EDMDBaseline",
    "EigenvalueRegularizationLoss",
    "EvaluationResult",
    "FitHistory",
    "ForwardConsistencyLoss",
    "GATDecoder",
    "GATEncoder",
    "GNNDecoder",
    "GNNEncoder",
    "GraphKoopmanEnv",
    "GraphKoopmanModel",
    "GraphKoopmanOperator",
    "GraphSnapshotSequence",
    "KoopmanOperator",
    "KoopmanSpectrum",
    "LossWeights",
    "MultiTrajectory",
    "RecursiveKoopmanAdapter",
    "TemporalSplit",
    "WindowSampler",
    "__version__",
    "compute_spectrum",
    "evaluate_forecast",
    "temporal_split",
]
__version__ = "0.4.0"
