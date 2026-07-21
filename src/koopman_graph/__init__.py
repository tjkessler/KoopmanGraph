"""KoopmanGraph: topology-aware Koopman dynamics on graphs.

Public API
----------
``GraphKoopmanModel``
    End-to-end encode → Koopman advance → decode model.
``GNNEncoder``, ``GATEncoder``, ``SAGEEncoder``, ``DiffConvEncoder``
    Topology-aware GNN encoders (GCN, GAT, GraphSAGE, DiffConv).
``GraphTransformerEncoder``
    Graph Transformer encoder (PyG ``TransformerConv``; edges×heads cost).
``DelayEmbeddingEncoder``
    Hankel / delay-coordinate wrapper around a sized base encoder.
``GNNDecoder``, ``GATDecoder``, ``SAGEDecoder``, ``DiffConvDecoder``
    Matching GNN decoders for physical reconstruction.
``GraphTransformerDecoder``
    Graph Transformer decoder peer.
``KoopmanOperator``
    Learnable finite-dimensional Koopman matrix.
``ContinuousKoopmanOperator``
    Continuous-time Koopman generator integrated via matrix exponentials.
``GraphKoopmanOperator``
    Discrete networked Koopman step with self/neighbor coupling via ``edge_index``.
``GraphSnapshotSequence``, ``MultiTrajectory``
    Container and explicit multi-trajectory wrapper for graph snapshots.
    ``TemporalSplit``, ``temporal_split``, ``WindowSampler``, and
    ``as_multi_trajectory`` live in :mod:`koopman_graph.data`.
``KoopmanSpectrum``, ``compute_spectrum``
    Primary spectral analysis entrypoints. Specialized helpers
    (``compute_generator_spectrum``, ``discrete_spectrum_at_delta_t``,
    ``decode_mode_shapes``, ``spectrum_distance``, ``koopman_std``,
    ``dynamical_similarity``, ``detect_anomaly``,
    ``calibrate_anomaly_threshold``, ``AnomalyDetectionResult``) live in
    :mod:`koopman_graph.analysis`.
``__version__``
    Package version string.

Training customization, evaluation utilities, baselines, adaptation, and
control live in capability modules: :mod:`koopman_graph.losses` (consistency /
eigenvalue / physics residuals), :mod:`koopman_graph.training`
(``FitHistory``, ``LossWeights``), :mod:`koopman_graph.baselines`,
:mod:`koopman_graph.adaptation` (``RecursiveKoopmanAdapter`` and observers),
:mod:`koopman_graph.env` (``GraphKoopmanEnv``), and :mod:`koopman_graph.metrics`
(``evaluate_forecast``, ``EvaluationResult``, and low-level metrics).

Physics-informed helpers such as ``graph_laplacian_features`` live in
:mod:`koopman_graph.observables`. Power-user modules such as
:mod:`koopman_graph.graph_utils`, :mod:`koopman_graph.nn`,
:mod:`koopman_graph.protocols`, :mod:`koopman_graph.spectrum_types`, and
:mod:`koopman_graph.uq` (deep ensembles / optional ``predict_interval``), and
:mod:`koopman_graph.hierarchical` (TopK / SAG pool → coarse Koopman → unpool)
are importable but intentionally omitted from ``__all__`` (encoder/decoder/delay
and operator *classes*, including ``DelayEmbeddingEncoder`` and
``GraphKoopmanOperator``, remain root-stable). See the architecture docs for
the public vs power-user contract.
"""

from koopman_graph.analysis import KoopmanSpectrum, compute_spectrum
from koopman_graph.data import GraphSnapshotSequence, MultiTrajectory
from koopman_graph.model import GraphKoopmanModel
from koopman_graph.nn import (
    DelayEmbeddingEncoder,
    DiffConvDecoder,
    DiffConvEncoder,
    GATDecoder,
    GATEncoder,
    GNNDecoder,
    GNNEncoder,
    GraphTransformerDecoder,
    GraphTransformerEncoder,
    SAGEDecoder,
    SAGEEncoder,
)
from koopman_graph.operators import (
    ContinuousKoopmanOperator,
    GraphKoopmanOperator,
    KoopmanOperator,
)

__all__ = [
    "ContinuousKoopmanOperator",
    "DelayEmbeddingEncoder",
    "DiffConvDecoder",
    "DiffConvEncoder",
    "GATDecoder",
    "GATEncoder",
    "GNNDecoder",
    "GNNEncoder",
    "GraphKoopmanModel",
    "GraphKoopmanOperator",
    "GraphSnapshotSequence",
    "GraphTransformerDecoder",
    "GraphTransformerEncoder",
    "KoopmanOperator",
    "KoopmanSpectrum",
    "MultiTrajectory",
    "SAGEDecoder",
    "SAGEEncoder",
    "__version__",
    "compute_spectrum",
]
__version__ = "0.5.0"
