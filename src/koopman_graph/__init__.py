"""KoopmanGraph: topology-aware Koopman dynamics on graphs.

Public API
----------
``GraphKoopmanModel``
    End-to-end encode → Koopman advance → decode model.
``GNNEncoder``, ``GATEncoder``, ``SAGEEncoder``, ``DiffConvEncoder``
    Topology-aware GNN encoders (GCN, GAT, GraphSAGE, DiffConv).
``GraphTransformerEncoder``
    Graph Transformer encoder (PyG ``TransformerConv``; edges×heads cost).
``HypergraphEncoder``
    Hypergraph encoder (PyG ``HypergraphConv``; static hyperedge incidence).
``SimplicialEncoder``
    Simplicial-1 encoder (combinatorial ``L_1`` mixing; not sheaf/cell).
``InvariantGeometryEncoder``
    Tier A invariant distance/angle features from ``Data.pos`` (not
    equivariant ``K``).
``E3EquivariantEncoder``
    Optional Tier B steerable encode via ``e3nn`` (``[equivariance]``);
    default projects to invariant latents for linear ``K``. Set
    ``project_invariants=False`` to keep vector channels for an
    equivariant latent operator.
``RelGraphEncoder``
    Multiplex relational encoder (R-GCN-lite; per-relation message banks).
``DelayEmbeddingEncoder``
    Hankel / delay-coordinate wrapper around a sized base encoder.
``GNNDecoder``, ``GATDecoder``, ``SAGEDecoder``, ``DiffConvDecoder``
    Matching GNN decoders for physical reconstruction.
``GraphTransformerDecoder``
    Graph Transformer decoder peer.
``HypergraphDecoder``
    Hypergraph decoder peer.
``SimplicialDecoder``
    Simplicial-1 decoder peer.
``RelGraphDecoder``
    Multiplex relational decoder peer.
``KoopmanOperator``
    Learnable finite-dimensional Koopman matrix.
``ContinuousKoopmanOperator``
    Continuous-time Koopman generator integrated via matrix exponentials.
``GraphKoopmanOperator``
    Discrete networked Koopman step with self/neighbor coupling via ``edge_index``.
``HypergraphKoopmanOperator``
    Discrete hyperedge-coupled Koopman step via ``hyperedge_index``.
``GlobalLocalKoopmanOperator``
    Discrete global backbone plus low-rank local window correction
    (``koopman="global_local"``).
``ContinuousGraphKoopmanOperator``
    Continuous networked generator via ``koopman="graph"`` + continuous
    (or ``koopman="continuous_graph"``).
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
(``FitHistory``, ``LossWeights``, ``FitCallback``), :mod:`koopman_graph.baselines`,
:mod:`koopman_graph.adaptation` (``RecursiveKoopmanAdapter`` and observers),
:mod:`koopman_graph.env` (``GraphKoopmanEnv``), and :mod:`koopman_graph.metrics`
(``evaluate_forecast``, ``EvaluationResult``, and low-level metrics).
``FitCallback`` / ``NoOpFitCallback`` are also re-exported at the package
root for ``fit(..., callbacks=)``.

Physics-informed helpers such as ``graph_laplacian_features`` live in
:mod:`koopman_graph.observables`. Power-user modules such as
:mod:`koopman_graph.graph_utils`, :mod:`koopman_graph.nn`,
:mod:`koopman_graph.protocols`, :mod:`koopman_graph.spectrum_types`,
:mod:`koopman_graph.uq` (deep ensembles / optional ``predict_interval``),
:mod:`koopman_graph.hierarchical` (TopK / SAG pool → coarse Koopman → unpool),
:mod:`koopman_graph.mpc` (receding-horizon Koopman-MPC and residual-tube
tightening; ``[mpc]`` extra),
:mod:`koopman_graph.tracking` (CSV / TensorBoard fit adapters),
:mod:`koopman_graph.cli` (``koopman-graph`` console script), and
:mod:`koopman_graph.tuning` (HPO metric helpers; not AutoML)
are importable but intentionally omitted from ``__all__`` (encoder/decoder/delay
and operator *classes*, including ``DelayEmbeddingEncoder`` and
``GraphKoopmanOperator``, remain root-stable). Frozen identification
config / report types and closed-form solvers live in
:mod:`koopman_graph.identification` (opt-in ``fit(..., identification=)``;
not on root ``__all__``). See the architecture docs for
the public vs power-user contract.
"""

from koopman_graph.analysis import KoopmanSpectrum, compute_spectrum
from koopman_graph.data import GraphSnapshotSequence, MultiTrajectory
from koopman_graph.model import GraphKoopmanModel
from koopman_graph.nn import (
    CellComplexGNNDecoder,
    CellComplexGNNEncoder,
    DelayEmbeddingEncoder,
    DiffConvDecoder,
    DiffConvEncoder,
    E3EquivariantEncoder,
    GATDecoder,
    GATEncoder,
    GNNDecoder,
    GNNEncoder,
    GraphTransformerDecoder,
    GraphTransformerEncoder,
    HypergraphDecoder,
    HypergraphEncoder,
    InvariantGeometryEncoder,
    RelGraphDecoder,
    RelGraphEncoder,
    SAGEDecoder,
    SAGEEncoder,
    SheafGNNDecoder,
    SheafGNNEncoder,
    SimplicialDecoder,
    SimplicialEncoder,
)
from koopman_graph.operators import (
    ContinuousGraphKoopmanOperator,
    ContinuousKoopmanOperator,
    GlobalLocalKoopmanOperator,
    GraphKoopmanOperator,
    HypergraphKoopmanOperator,
    KoopmanOperator,
)
from koopman_graph.training import FitCallback, NoOpFitCallback

__all__ = [
    "CellComplexGNNDecoder",
    "CellComplexGNNEncoder",
    "ContinuousGraphKoopmanOperator",
    "ContinuousKoopmanOperator",
    "DelayEmbeddingEncoder",
    "DiffConvDecoder",
    "DiffConvEncoder",
    "E3EquivariantEncoder",
    "FitCallback",
    "GATDecoder",
    "GATEncoder",
    "GNNDecoder",
    "GNNEncoder",
    "GlobalLocalKoopmanOperator",
    "GraphKoopmanModel",
    "GraphKoopmanOperator",
    "GraphSnapshotSequence",
    "GraphTransformerDecoder",
    "GraphTransformerEncoder",
    "HypergraphDecoder",
    "HypergraphEncoder",
    "HypergraphKoopmanOperator",
    "InvariantGeometryEncoder",
    "KoopmanOperator",
    "KoopmanSpectrum",
    "MultiTrajectory",
    "NoOpFitCallback",
    "RelGraphDecoder",
    "RelGraphEncoder",
    "SAGEDecoder",
    "SAGEEncoder",
    "SheafGNNDecoder",
    "SheafGNNEncoder",
    "SimplicialDecoder",
    "SimplicialEncoder",
    "__version__",
    "compute_spectrum",
]
__version__ = "0.15.0"
