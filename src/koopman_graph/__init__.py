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
``KoopmanSpectrum``
    Eigendecomposition and continuous-time mode characteristics.
``compute_spectrum``, ``decode_mode_shapes``
    Spectral analysis and latent-to-spatial mode decoding helpers.
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
``LossWeights``
    Weights for reconstruction and consistency loss terms.
``__version__``
    Package version string.
"""

from koopman_graph.analysis import (
    KoopmanSpectrum,
    compute_spectrum,
    decode_mode_shapes,
)
from koopman_graph.baselines import DMDBaseline, DMDcBaseline, EDMDBaseline
from koopman_graph.data import (
    GraphSnapshotSequence,
    TemporalSplit,
    WindowSampler,
    temporal_split,
)
from koopman_graph.decoder import GNNDecoder
from koopman_graph.encoder import GATEncoder, GNNEncoder
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
from koopman_graph.operator import KoopmanOperator
from koopman_graph.training import FitHistory, LossWeights

__all__ = [
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
    "GraphKoopmanModel",
    "GraphSnapshotSequence",
    "HorizonMetrics",
    "KoopmanOperator",
    "KoopmanSpectrum",
    "LossWeights",
    "TemporalSplit",
    "WindowSampler",
    "__version__",
    "compute_spectrum",
    "decode_mode_shapes",
    "evaluate_forecast",
    "mae",
    "mape",
    "rmse",
    "temporal_split",
]
__version__ = "0.2.0"
