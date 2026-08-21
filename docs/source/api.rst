API Reference
=============

Public classes and functions are documented by module below. Core workflow
symbols are re-exported from :mod:`koopman_graph` (see ``__all__``); specialized
helpers live in capability modules such as :mod:`koopman_graph.metrics`,
:mod:`koopman_graph.analysis`, :mod:`koopman_graph.data`,
:mod:`koopman_graph.adaptation`, and :mod:`koopman_graph.observables`.

For the public vs power-user vs private layering contract, device conventions,
and design philosophy, see :doc:`architecture`. Narrative guides for
new surfaces: :doc:`benchmarks`, :doc:`identification`,
:doc:`spectral_diagnostics`, :doc:`graph_dynamics`, :doc:`matrix_free`,
:doc:`criticality`, and :doc:`time_conditioning`.

Package
-------

.. automodule:: koopman_graph
   :members: __version__
   :no-index:

Model
-----

Same-named capability package (:mod:`koopman_graph.model`) with peer modules
``estimator``, ``factory``, ``validation``, ``timing``, ``encoding``,
``inference``, and ``online_adaptation``. Prefer
``from koopman_graph import GraphKoopmanModel`` or
``from koopman_graph.model import GraphKoopmanModel``.

.. automodule:: koopman_graph.model
   :members:
   :imported-members:
   :exclude-members: DiffConvDecoder, DiffConvEncoder, EvaluationResult, GATDecoder, GATEncoder, GNNDecoder, GNNEncoder, GraphSnapshotSequence, GraphKoopmanOperator, GraphTransformerDecoder, GraphTransformerEncoder, KoopmanOperator, SAGEDecoder, SAGEEncoder, encoding, estimator, factory, inference, online_adaptation, timing, validation
   :show-inheritance:

Encoders
--------

Built-in encoders live in :mod:`koopman_graph.nn` (``encoder``, ``decoder``,
``gnn``). Prefer ``from koopman_graph import GNNEncoder, GATEncoder,
SAGEEncoder, DiffConvEncoder, GraphTransformerEncoder`` for application code,
or ``from koopman_graph.nn import …`` for power-user imports.
Former deep imports ``koopman_graph.encoder`` / ``decoder`` / ``gnn`` were
removed in v0.3.0.

.. automodule:: koopman_graph.nn.encoder
   :members:
   :show-inheritance:

Delay Embeddings
----------------

Hankel / delay-coordinate lifting wraps a sized base encoder.
:class:`~koopman_graph.nn.delay.DelayEmbeddingEncoder` is a root-stable
``__all__`` member (also importable from :mod:`koopman_graph.nn`). Prefer
``from koopman_graph import DelayEmbeddingEncoder`` or pass ``n_delays`` to
:class:`~koopman_graph.model.GraphKoopmanModel`. This is Takens-style channel
stacking, not
:class:`~koopman_graph.baselines.HankelDMDBaseline` or
:class:`~koopman_graph.baselines.HAVOKBaseline`. Pure snapshot/tensor window
helpers are owned by :mod:`koopman_graph.data.delay_windows` and
thin-re-exported from this module for compatibility.

.. automodule:: koopman_graph.nn.delay
   :members:
   :imported-members:
   :show-inheritance:

Hypergraph Encoders / Decoders
--------------------------------

Higher-order incidence encoders and decoders
(:class:`~koopman_graph.nn.HypergraphEncoder`,
:class:`~koopman_graph.nn.HypergraphDecoder`) are root-stable ``__all__``
members. Prefer ``from koopman_graph import HypergraphEncoder, HypergraphDecoder``
with ``koopman="hypergraph"`` on the model. Pairwise GCN/GAT/SAGE peers remain
under :mod:`koopman_graph.nn.encoder` / ``decoder``.

.. automodule:: koopman_graph.nn.hypergraph
   :members:
   :show-inheritance:

Heterogeneous / RelGraph Encoders / Decoders
---------------------------------------------

Multiplex and typed RelGraph peers
(:class:`~koopman_graph.nn.RelGraphEncoder`,
:class:`~koopman_graph.nn.RelGraphDecoder`) are root-stable ``__all__``
members. Prefer ``from koopman_graph import RelGraphEncoder, RelGraphDecoder``
with factory ``koopman="hetero_graph"`` on
:class:`~koopman_graph.data.HeteroGraphSnapshotSequence`. Optional HGT peers
(:class:`~koopman_graph.nn.HGTEncoder`,
:class:`~koopman_graph.nn.HGTDecoder`) are imported from
:mod:`koopman_graph.nn` only (not factory defaults; not on the root façade).

.. automodule:: koopman_graph.nn.heterogeneous
   :members:
   :show-inheritance:

Adaptive Topology (power-user)
------------------------------

Self-adaptive pairwise adjacency
(:class:`~koopman_graph.nn.AdaptiveAdjacency`) is enabled via
``learn_topology="self_adaptive"`` on :class:`~koopman_graph.model.GraphKoopmanModel`.
Import from :mod:`koopman_graph.nn` (not on the root façade).

.. automodule:: koopman_graph.nn.adaptive_topology
   :members:
   :show-inheritance:

Simplicial Encoders / Decoders
--------------------------------

Combinatorial simplicial-1 / Hodge peers
(:class:`~koopman_graph.nn.SimplicialEncoder`,
:class:`~koopman_graph.nn.SimplicialDecoder`) are root-stable ``__all__``
members. Prefer ``from koopman_graph import SimplicialEncoder,
SimplicialDecoder``. Oriented ``edge_index`` and optional ``face_index``.
For sheaf / cell-complex MVPs see the sections below (not full
TopologicX parity — :doc:`limitations`).

.. automodule:: koopman_graph.nn.simplicial
   :members:
   :show-inheritance:

Sheaf Encoders / Decoders
---------------------------

Diagonal-restriction sheaf peers
(:class:`~koopman_graph.nn.SheafGNNEncoder`,
:class:`~koopman_graph.nn.SheafGNNDecoder`) are root-stable ``__all__``
members. Prefer ``from koopman_graph import SheafGNNEncoder,
SheafGNNDecoder`` or factory ``encoder="sheaf"``. In-repo MVP with the
same linear Koopman head — **not** full sheaf / TopologicX feature
parity.

.. automodule:: koopman_graph.nn.sheaf
   :members:
   :show-inheritance:

Cell-complex Encoders / Decoders
----------------------------------

Cell-complex peers
(:class:`~koopman_graph.nn.CellComplexGNNEncoder`,
:class:`~koopman_graph.nn.CellComplexGNNDecoder`) are root-stable
``__all__`` members; :class:`~koopman_graph.nn.CellComplex` helpers are
importable from :mod:`koopman_graph.nn`. Prefer
``from koopman_graph import CellComplexGNNEncoder, CellComplexGNNDecoder``
or factory ``encoder="cell_complex"`` (``Data.face_index`` required).
:func:`~koopman_graph.nn.order2_cochain_teaching` binds
:class:`~koopman_graph.operators.CochainKoopmanOperator` to a filled
triangle (order-2 teaching; :math:`k=2` not evolved).
:data:`~koopman_graph.nn.MAX_CELL_COMPLEX_DEGREE` (3) is the
teaching ceiling — **not** a full cellular TDL stack.

.. automodule:: koopman_graph.nn.cell_complex
   :members:
   :show-inheritance:

Geometry / steerable encoders
-----------------------------

Tier A :class:`~koopman_graph.nn.InvariantGeometryEncoder` and optional
Tier B :class:`~koopman_graph.nn.E3EquivariantEncoder` (``e3nn``,
``[equivariance]``) are root-stable ``__all__`` members. Prefer
``from koopman_graph import InvariantGeometryEncoder`` /
``E3EquivariantEncoder``. Default encode still projects to invariant
scalars. Pass ``project_invariants=False`` with
:class:`~koopman_graph.operators.EquivariantKoopmanOperator` for a block
:math:`K` (vector channels are multiples of :math:`I_3`; optional
:math:`l=2` channels are multiples of :math:`I_5`).

.. automodule:: koopman_graph.nn.equivariant
   :members:
   :show-inheritance:

Predicted topology and neural-operator lifts
--------------------------------------------

Default graph-state scoring is
:class:`~koopman_graph.nn.SparseCandidateTopologyHead` (at most
``candidate_k`` destinations per node). Dense
:class:`~koopman_graph.nn.PredictedTopologyHead` is the ``dense_mlp``
power-user path with an :math:`N` ceiling. Both are distinct from
:class:`~koopman_graph.nn.AdaptiveAdjacency`. When a head is attached and
``recursive_training`` is true, ``fit`` / ``predict`` / ``evaluate`` /
UQ consume predicted :math:`\hat A_{t+1}` as sigmoid weights unless
``future_topologies`` or ``topology_policy="hold_last"`` is set.
:class:`~koopman_graph.nn.PresenceHead` is a linear per-node logit used
when presence masks exist.

.. automodule:: koopman_graph.nn.predicted_topology
   :members:
   :show-inheritance:

.. automodule:: koopman_graph.nn.neural_operator
   :members:
   :show-inheritance:

Decoder
-------

.. automodule:: koopman_graph.nn.decoder
   :members:
   :show-inheritance:

Constraint-preserving decoder heads
-----------------------------------

Opt-in wrappers
(:class:`~koopman_graph.nn.MassConservingDecoder`,
:class:`~koopman_graph.nn.PositivityDecoder`,
:class:`~koopman_graph.nn.LinearConservingDecoder`) project named
decoded channels onto a simplex, a linear conservation law, or a
positivity map. They are not a factory kind and have no checkpoint
type string. Latent symplectic :math:`K` alone does not conserve
decoded mass.

.. automodule:: koopman_graph.nn.constraint_decoders
   :members:
   :show-inheritance:

Shared GNN Primitives (power-user)
----------------------------------

Neutral message-passing helpers shared by peer encoder and decoder modules
(:class:`~koopman_graph.nn.gnn.BaseGNNModule`, activation typing,
GCN/GAT/SAGE/DiffConv/Transformer builders). Importable via
:mod:`koopman_graph.nn.gnn`,
but **not** part of the stable public façade (not in ``koopman_graph.__all__``).
Prefer the public encoder/decoder classes for application code. See
:doc:`architecture`.

.. automodule:: koopman_graph.nn.gnn
   :members:
   :show-inheritance:

Encoder / operator receptive field (power-user)
-----------------------------------------------

Warn-only hop comparison for discrete ``koopman="graph"``. Prefer
:meth:`~koopman_graph.model.GraphKoopmanModel.fit`, which calls the
helper at fit start. Not in root ``__all__``.

.. automodule:: koopman_graph.nn.receptive_field
   :members:
   :show-inheritance:

Physics-Informed Observables
----------------------------

Hybrid physics helpers are imported from :mod:`koopman_graph.observables` only.
Built-in names are ``"graph_laplacian"``, ``"graph_gradient"``,
``"graph_curvature"``, and ``"polynomial(degree)"``. Pass a custom
``physics_lifting_fn`` when needed.

.. automodule:: koopman_graph.observables
   :members:
   :show-inheritance:

Koopman Operator
----------------

Built-in operators live in :mod:`koopman_graph.operators` (``contract``,
``control``, ``discrete``, ``discrete_parameterizations``,
``discrete_propagation``, ``continuous``, ``continuous_van_loan``,
``continuous_parameterizations``, ``continuous_propagation``,
``auxiliary_spectral``, ``graph``, ``polynomial_graph``, ``linear``, ``hypergraph``, ``heterogeneous``,
``global_local``, ``continuous_graph``, ``switched``, ``mixture``,
``parametric``,
``hodge``, ``cochain``, ``equivariant``, ``graphon``, ``stochastic``,
``stochastic_sde``). Prefer
``from koopman_graph import KoopmanOperator, ContinuousKoopmanOperator,
GraphKoopmanOperator, HypergraphKoopmanOperator, GlobalLocalKoopmanOperator,
ContinuousGraphKoopmanOperator`` (root-stable ``__all__`` members) or
``from koopman_graph.operators import …`` (includes
:class:`~koopman_graph.operators.HeteroGraphKoopmanOperator` for
``koopman="hetero_graph"`` and
:func:`~koopman_graph.operators.estimate_graphon` /
:class:`~koopman_graph.operators.GraphonEstimate` for dense teaching
kernels). Former deep imports
``koopman_graph.operator`` / ``koopman_graph.continuous`` were removed in
v0.3.0.
:class:`~koopman_graph.operators.CochainKoopmanOperator` advances
node / edge latents on a static signed :math:`B_1` — **not** a factory
kind, **not** ``koopman="hodge"``, and **not** TopologicX parity.
:class:`~koopman_graph.operators.EquivariantKoopmanOperator` is a
block MVP (scalars, ``scale * I_3`` vectors, optional
:math:`l=2` ``scale * I_5`` tensors) — **not** a factory kind
and **not** a molecular MD stack.

.. automodule:: koopman_graph.operators
   :members:
   :exclude-members: CochainState, CochainKoopmanOperator, BoundaryNilpotencyReport, boundary_nilpotency, DEFAULT_NILPOTENCY_ATOL, LinearOperatorProtocol, PolynomialGraphLinearOperator, MatrixFreeGraphLinearOperator, MemoryEstimate, EigResult, MAX_DENSE_LINEAR_OPERATOR_SIZE
   :show-inheritance:

.. automodule:: koopman_graph.operators.cochain
   :members:
   :show-inheritance:

Matrix-free linear operators
----------------------------

:class:`~koopman_graph.operators.LinearOperatorProtocol` is the common
``matvec`` / ``rmatvec`` / ``solve`` / ``expm_action`` / leading-eigpair
surface. Wrappers cover a monomial graph polynomial and the existing
one-tap ``matrix_free`` path. Dense assembly is refused above
:data:`~koopman_graph.operators.MAX_DENSE_LINEAR_OPERATOR_SIZE`.
Trainer DDP does not shrink that representation. Not a factory kind.
See :doc:`matrix_free`.

.. automodule:: koopman_graph.operators.linear
   :members:
   :show-inheritance:

Spectral Analysis
-----------------

:class:`~koopman_graph.spectrum_types.KoopmanSpectrum` is defined in the
neutral :mod:`koopman_graph.spectrum_types` module and re-exported from
:mod:`koopman_graph.analysis` (and the package root) for the public API.
Primary entrypoints (``KoopmanSpectrum``, ``compute_spectrum``) remain on
the root façade. Specialized helpers (``compute_generator_spectrum``,
``discrete_spectrum_at_delta_t``, ``decode_mode_shapes``,
``spectrum_distance``, ``koopman_std``, ``dynamical_similarity``,
``detect_anomaly``, ``calibrate_anomaly_threshold``,
``AnomalyDetectionResult``, ``plot_spectrum``,
``spectral_residuals``, ``SpectralResidualReport``,
``identify_sparse_dynamics``, ``SINDyReport``,
``koopman_spectral_clustering``, ``ClusteringResult``,
``estimate_coupling_from_snapshots``, ``CouplingEstimate``,
``attribute_mode_energy``, ``ModeEnergyAttribution``,
``evaluate_topology_transfer``, ``TopologyTransferReport``,
``explain_representation``, ``RepresentationExplanation``,
``SpectralDiagnostics``, ``compute_spectral_diagnostics``,
``DefectiveSpectrumError``, ``CONDITION_WARN``,
``monitor_critical_transition``, ``CriticalityReport``) are imported
from :mod:`koopman_graph.analysis` only. The helpers live in the
``spectrum`` / ``conditioning`` / ``criticality`` / ``similarity`` / ``anomaly`` /
``plotting`` / ``residuals`` / ``sindy`` / ``clustering`` /
``topology_estimation`` / ``transfer`` / ``explain`` submodules.
:func:`~koopman_graph.analysis.graph_dispersion`,
:func:`~koopman_graph.analysis.granger_latent_influence`,
:func:`~koopman_graph.analysis.recover_synthetic_interventional_edges`,
:func:`~koopman_graph.analysis.persistence_diagram_0d`, and
:func:`~koopman_graph.analysis.discrete_lyapunov_lmi` are additional
package helpers (finite ResDMD may attach to ``evaluate``).
:func:`~koopman_graph.analysis.granger_latent_influence` is
**non-interventional**. The synthetic SCM helper recovers a known
do-edge on a teaching fixture only.
``attribute_mode_energy`` is an interpretive diagnostic on assembled
:math:`K_{\mathrm{eff}}` (not a causal claim; not a ResDMD residual).
:func:`~koopman_graph.analysis.explain_representation` /
:class:`~koopman_graph.analysis.RepresentationExplanation` provide
homogeneous MVP node / edge / feature masks (GNNExplainer or optional
Captum integrated gradients) — **interpretive** / **non-causal**; not
``ModeEnergyAttribution`` and not ResDMD. Finite-dictionary ResDMD
(:func:`~koopman_graph.analysis.resdmd`,
:class:`~koopman_graph.analysis.ResDMDReport`) and the finite-matrix
resolvent-norm grid
(:func:`~koopman_graph.analysis.resolvent_norm_grid`) are package exports;
they are distinct from ``spectral_residuals`` and are **not**
infinite-dimensional certified pseudospectra.
:func:`~koopman_graph.analysis.evaluate_topology_transfer` **measures**
cross-topology transfer (mandatory ``pernode`` control; negative
advantage allowed) — see :doc:`limitations`.
``plot_spectrum`` requires Matplotlib (``pip install matplotlib`` or the
``[dev]`` extra).
:func:`~koopman_graph.analysis.monitor_critical_transition` is a
heuristic sliding-window spectral-gap / near-defectivity score — **not**
a topology-criticality certificate (see :doc:`limitations` and
:doc:`criticality`). Conditioning and Nyquist fields are documented
in :doc:`spectral_diagnostics`.
:func:`~koopman_graph.analysis.markov_closure_report` is a Ljung–Box-style
residual-energy flag; :class:`~koopman_graph.analysis.FiniteMemoryKoopman`
is a convolution MVP — **not** Mori–Zwanzig identification, **not**
HAVOK, and **not** :class:`~koopman_graph.nn.delay.DelayEmbeddingEncoder`.
:func:`~koopman_graph.analysis.hodge_decompose_modes` is a
combinatorial Hodge split of stored eigenvector columns — **not**
physical circulation, **not** ``koopman="hodge"``, and **not**
TopologicX / sheaf parity.

.. automodule:: koopman_graph.spectrum_types
   :members:
   :show-inheritance:

.. automodule:: koopman_graph.analysis
   :members:
   :exclude-members: KoopmanSpectrum, SpectralDiagnostics, compute_spectral_diagnostics, DefectiveSpectrumError, CONDITION_WARN, CriticalityReport, monitor_critical_transition, SpectralResidualReport, spectral_residuals, ResDMDReport, resdmd, ResolventNormGrid, resolvent_norm_grid, TopologyTransferReport, evaluate_topology_transfer, RepresentationExplanation, explain_representation, PersistenceDiagram, persistence_diagram_0d, betti_curve, DispersionRelation, graph_dispersion, CausalInfluenceReport, granger_latent_influence, SyntheticSCM, SyntheticInterventionReport, recover_synthetic_interventional_edges, sample_synthetic_intervention, sample_synthetic_observational, teaching_three_node_scm, LyapunovLMIResult, discrete_lyapunov_lmi, FiniteMemoryKoopman, MarkovClosureReport, markov_closure_report, HodgeModeComponents, hodge_decompose_modes
   :show-inheritance:

.. automodule:: koopman_graph.analysis.residuals
   :members:
   :show-inheritance:

.. automodule:: koopman_graph.analysis.resdmd
   :members:
   :show-inheritance:

.. automodule:: koopman_graph.analysis.pseudospectra
   :members:
   :show-inheritance:

.. automodule:: koopman_graph.analysis.transfer
   :members:
   :show-inheritance:

.. automodule:: koopman_graph.analysis.explain
   :members:
   :show-inheritance:

.. automodule:: koopman_graph.analysis.tda
   :members:
   :show-inheritance:

.. automodule:: koopman_graph.analysis.dispersion
   :members:
   :show-inheritance:

.. automodule:: koopman_graph.analysis.causal
   :members:
   :show-inheritance:

Synthetic SCM interventions
---------------------------

Labeled linear Gaussian teaching SCM and paired do-interventions.
:func:`~koopman_graph.analysis.granger_latent_influence` remains
**non-interventional**. Not a factory kind; off root ``__all__``.

.. automodule:: koopman_graph.analysis.causal_intervention
   :members:
   :show-inheritance:

.. automodule:: koopman_graph.analysis.lmi
   :members:
   :show-inheritance:

.. automodule:: koopman_graph.analysis.criticality
   :members:
   :show-inheritance:

.. automodule:: koopman_graph.analysis.memory
   :members:
   :show-inheritance:

.. automodule:: koopman_graph.analysis.hodge_modes
   :members:
   :show-inheritance:

Identification (opt-in solvers)
-------------------------------

Frozen config / report types, a
:class:`~koopman_graph.identification.IdentificationBackend` protocol,
and closed-form :func:`~koopman_graph.identification.identify_operator`
(ridge, TLS, constrained least squares) live under
:mod:`koopman_graph.identification` (not on root ``__all__``). Prefer
``from koopman_graph.identification import IdentificationConfig,
IdentificationReport, identify_operator``. Pass
``fit(..., identification=IdentificationConfig(...))`` to bind solvers;
default ``identification=None`` is the Adam path. Discrete dense
per-node operators only. Finite-sample subspace leakage
(:func:`~koopman_graph.identification.subspace_invariance_report`)
attaches to ``evaluate(..., include_invariance=True)``; it is not a
Haseli–Cortés certificate.
:func:`~koopman_graph.identification.select_resdmd_gated` is residual-aware
dictionary selection on a finite dictionary (not a ResDMD certificate).
``gate_resdmd=True`` fills ``IdentificationReport.spectral`` on the
final identification ``fit``; it does not abort training.
:func:`~koopman_graph.identification.identify_sparse_graph_factors`
identifies sparse graph factors on frozen encodings (STLSQ or
teaching group-lasso with unpenalized refit). It is **not**
:func:`~koopman_graph.analysis.identify_sparse_dynamics` and **not**
:class:`~koopman_graph.losses.KoopmanSparsityLoss`; those paths still
ship. Dual random-walk and polynomial :math:`P>1` maps are out of
scope. See :doc:`architecture`.
:func:`~koopman_graph.identification.select_latent_rank` selects a
numerical rank of frozen encodings over a candidate grid (in-tree
VAMP-2, ResDMD elbow, or stability-penalized held-out MSE). It is
**not** Ray Tune AutoML for encoder ``latent_dim``.
See :doc:`identification`.

.. automodule:: koopman_graph.identification
   :members:
   :imported-members:
   :show-inheritance:

Benchmark manifests and identity-bound summaries
------------------------------------------------

Frozen :class:`~koopman_graph.benchmark.ExperimentManifest` records
(schema ``benchmark_manifest_v1``) live under
:mod:`koopman_graph.benchmark` (not on root ``__all__``). Prefer
``from koopman_graph.benchmark import ExperimentManifest, load_manifest,
run_manifest, verify_summary``. JSON always; YAML requires PyYAML
(``pip install "koopman-graph[cli]"``). Teaching GNN methods require
non-empty ``deviations``. Dataset SHA-256 mismatch is rejected.
``run_manifest`` writes a canonical ``summary.json``
(``benchmark_summary_v1``) with ``executed=False``; it does **not**
train a model or invent forecast metrics. This package is **not** a
LibCity / BasicTS leaderboard host. The console script is
``koopman-graph benchmark run`` / ``verify`` (see :doc:`cli`).

.. automodule:: koopman_graph.benchmark
   :members:
   :imported-members:
   :show-inheritance:

Long-horizon statistics (power-user)
------------------------------------

Welch PSD, spectral / invariant-measure distances, and Rosenstein largest-
Lyapunov estimation live in the neutral leaf :mod:`koopman_graph.statistics`
(torch-only; peer to :mod:`koopman_graph.metrics`). Prefer
``from koopman_graph.statistics import …``. Symbols are **not** on root
``__all__``. See :doc:`limitations` for embedding sensitivity and
:math:`O(T^{2})` cost of the Lyapunov helper.

.. automodule:: koopman_graph.statistics
   :members:
   :show-inheritance:

Baselines
---------

Classical DMD-family baselines live in the ``koopman_graph.baselines``
capability package (``base`` / ``dmd`` / ``dmdc`` / ``edmd`` and related peer
modules). They share :class:`~koopman_graph.baselines.ClassicalBaseline`
scaffolding and satisfy :class:`~koopman_graph.protocols.ForecastModel`
(``fit`` / ``predict`` / ``spectrum``). Call sites are **not** fully
interchangeable — see the ``ForecastModel`` call-site matrix in
:doc:`architecture`.
:class:`~koopman_graph.baselines.EDMDBaseline` exposes
``reconstruction_matrix`` for observable-to-state least squares (not a GNN
decoder) and supports ``dictionary`` in ``{"polynomial", "rbf", "kernel"}``
(Williams2015 polynomial / RBF EDMD; kernel sections following
Williams2015KernelDMD / Klus2018TransferOperator).
Additional classical peers include
:class:`~koopman_graph.baselines.MpEDMDBaseline` (measure-preserving EDMD;
Gram-weighted Procrustes polar factor, ``Colbrook2023mpEDMD`` — not a
Euclidean :math:`K_{\\mathrm{eff}}` certificate),
:class:`~koopman_graph.baselines.GEDMDBaseline` (generator EDMD from
supplied dictionary derivatives, ``Klus2020gEDMD`` — not derivative-mode
SINDy; irregular timestamps do not create :math:`L`),
:class:`~koopman_graph.baselines.HankelDMDBaseline` (Hankel-DMD on
delay-embedded flattened states, ``Arbabi2017HankelDMD`` — not
:class:`~koopman_graph.nn.delay.DelayEmbeddingEncoder`),
:class:`~koopman_graph.baselines.HAVOKBaseline` (teaching HAVOK on the
same delay rows, ``Brunton2017HAVOK``; autonomous ``predict`` uses
forcing :math:`u=0`),
:class:`~koopman_graph.baselines.FBDMDBaseline`,
:class:`~koopman_graph.baselines.TLSDMDBaseline`,
:class:`~koopman_graph.baselines.OptDMDBaseline`,
:class:`~koopman_graph.baselines.StreamingDMDBaseline`,
:class:`~koopman_graph.baselines.MRDMDBaseline`, and
:class:`~koopman_graph.baselines.UlamTransferOperatorBaseline`.
Topology-blind VAMP-2 precursor helpers
(:func:`~koopman_graph.baselines.vamp2_score`,
:func:`~koopman_graph.baselines.vamp2_loss`) and the contact-graph
teaching baseline
(:class:`~koopman_graph.baselines.GraphVAMPBaseline`) are package
exports — teaching / diagnostic, not GraphVAMPnets production.
Truncated-SVD ``rank`` accepts ``None`` (full least squares), a positive
integer, or ``"auto"`` (Gavish–Donoho median threshold;
``koopman_graph.baselines.base.optimal_hard_threshold_rank``); fitted
``selected_rank`` is stored on the baseline instance.
The full ``kernel`` path with one center per training snapshot is
:math:`O(T^2)` in feature dimension and is intended for small/medium ``T``
only; ``kernel="linear"`` reduces to DMD. Prefer
``from koopman_graph.baselines import …``.

Spatiotemporal GNN forecaster baselines
(:class:`~koopman_graph.baselines.gnn.STGCNBaseline`,
:class:`~koopman_graph.baselines.gnn.DCRNNBaseline`,
:class:`~koopman_graph.baselines.gnn.GraphWaveNetBaseline`,
:class:`~koopman_graph.baselines.gnn.AGCRNBaseline`,
:class:`~koopman_graph.baselines.gnn.MTGNNBaseline`,
:class:`~koopman_graph.baselines.gnn.STGODEBaseline`,
:class:`~koopman_graph.baselines.gnn.GraphCastBaseline`) live under
``koopman_graph.baselines.gnn``. They are lightweight ``nn.Module``
**teaching** references with ``ForecasterProtocol`` deviation tables for
comparisons with :class:`~koopman_graph.model.GraphKoopmanModel`
(including :func:`~koopman_graph.metrics.evaluate_forecast`) — **not**
leaderboard-matched reproductions. Their ``spectrum`` method raises
``RuntimeError`` (no linear Koopman operator). Prefer
``from koopman_graph.baselines.gnn import …``.

.. automodule:: koopman_graph.baselines
   :members:
   :exclude-members: STGCNBaseline, DCRNNBaseline, GraphWaveNetBaseline, AGCRNBaseline, MTGNNBaseline, STGODEBaseline, GraphCastBaseline
   :show-inheritance:

.. automodule:: koopman_graph.baselines.vamp2
   :members:
   :show-inheritance:

.. automodule:: koopman_graph.baselines.gnn
   :members:
   :show-inheritance:

Typing Protocols (power-user)
-----------------------------

Structural contracts for forecasting and trainable façades:

* :class:`~koopman_graph.protocols.ForecastModel` — loose ``fit`` / ``predict`` /
  ``spectrum`` façade (method presence; not drop-in call-site compatible)
* :class:`~koopman_graph.protocols.UncontrolledForecastModel` — autonomous
  ``predict(Data, steps)`` peer set; use
  :func:`~koopman_graph.protocols.accepts_uncontrolled_data_predict` at runtime
* :class:`~koopman_graph.protocols.TrainableKoopmanModel` — training and
  ``evaluate_forecast`` duck-typing contract (extends ``ForecastModel``)

Importable, but **not** part of the stable public façade (not in
``koopman_graph.__all__``). See :doc:`architecture` for the call-site matrix and
duck-typed training vs hard-typed env / serialization.

.. automodule:: koopman_graph.protocols
   :members:
   :show-inheritance:

Data Utilities
--------------

Containers for ``fit`` (``GraphSnapshotSequence``, ``MultiTrajectory``)
remain on the root façade. Split / sampling helpers
(``TemporalSplit``, ``temporal_split``, ``WindowSampler``) and
``as_multi_trajectory`` are imported from :mod:`koopman_graph.data`
only. The package peers are ``containers`` / ``construction`` /
``validation`` / ``trajectories`` / ``batching`` / ``graph_state`` /
``conditioning`` / ``calendar`` /
``delay_windows`` /
``sampling`` /
``splits`` / ``rollout``; prefer ``from koopman_graph.data import …``.
Array→snapshot builders are power-user imports from
:mod:`koopman_graph.data.construction`. Delay-window stack/flatten helpers
are power-user imports from :mod:`koopman_graph.data.delay_windows` (also
re-exported by :mod:`koopman_graph.nn.delay`). Opt-in multi-graph collate
(``BatchedGraphTrajectory``, ``collate_graph_trajectories``) is a
package export used by ``fit(..., batch_graphs=True)``. Opt-in graph-state
closure uses :class:`~koopman_graph.data.GraphDynamicsConfig` and
:class:`~koopman_graph.data.GraphStateSnapshot` (not a replacement for
:class:`~koopman_graph.data.GraphSnapshotSequence`; not on the root façade).
:class:`~koopman_graph.data.EntityRemap` is the injective union-placement
hook (finite :math:`N_{\max}`; not entity resolution).
:class:`~koopman_graph.data.ConditioningContext` records per-snapshot
:math:`(\\mu, t, u)` from optional homogeneous ``parameter_trajectory``
(not itself :math:`K(\\mu)`; not on the root façade).
``koopman="parametric"`` evaluates the interpolant from those
coordinates. :func:`~koopman_graph.data.diurnal_control_features` and
:func:`~koopman_graph.data.diurnal_phase_index` are time-of-day
recipes for existing control / switched ``phase_index``, not a
calendar serializer.

.. automodule:: koopman_graph.data
   :members:
   :show-inheritance:

.. automodule:: koopman_graph.data.batching
   :members:
   :show-inheritance:

.. automodule:: koopman_graph.data.graph_state
   :members:
   :show-inheritance:

.. automodule:: koopman_graph.data.conditioning
   :members:
   :show-inheritance:

.. automodule:: koopman_graph.data.calendar
   :members:
   :show-inheritance:

.. automodule:: koopman_graph.data.construction
   :members:
   :show-inheritance:

.. automodule:: koopman_graph.data.delay_windows
   :members:
   :show-inheritance:

Shared Graph Utilities (power-user)
-----------------------------------

Documented internal helpers for graph-input resolution, Laplacian mathematics,
latent propagation, symmetry orbits, and exact-automorphism isotypic
projectors. :mod:`koopman_graph.graph_utils` is a shallow capability package
(``topology`` / ``propagation`` / ``symmetry`` / ``representation`` peers)
whose ``__init__`` re-exports the documented surface. Importable, but
**not** part of the stable public façade (not in ``koopman_graph.__all__``).
Use :meth:`~koopman_graph.model.GraphKoopmanModel.encode` when lifting
snapshots. See :doc:`architecture`.

.. automodule:: koopman_graph.graph_utils
   :members:
   :imported-members:
   :show-inheritance:

.. automodule:: koopman_graph.graph_utils.topology
   :members:
   :show-inheritance:

.. automodule:: koopman_graph.graph_utils.propagation
   :members:
   :show-inheritance:

.. automodule:: koopman_graph.graph_utils.symmetry
   :members:
   :show-inheritance:

.. automodule:: koopman_graph.graph_utils.representation
   :members:
   :show-inheritance:

Losses
------

Same-named capability package (:mod:`koopman_graph.losses`) with peer
modules ``consistency``, ``regularization``, ``reconstruction``,
``physics``, and ``rollout``. Prefer ``from koopman_graph.losses import …``.

PIKN-style :class:`~koopman_graph.losses.LieConsistencyLoss`, PINN-style
:class:`~koopman_graph.losses.PDEResidualLoss`,
:class:`~koopman_graph.losses.KoopmanSparsityLoss`, and
:class:`~koopman_graph.losses.WorstCaseReconstructionLoss` are power-user
module imports; they are intentionally omitted from root
``koopman_graph.__all__``. Training composition uses
:class:`~koopman_graph.training.ExtraLosses` (physics residuals) plus
:class:`~koopman_graph.training.LossWeights` (including ``sparsity`` /
``worst_case``).

.. automodule:: koopman_graph.losses
   :members:
   :exclude-members: GraphSnapshotSequence
   :show-inheritance:

Training
--------

Capability peers under :mod:`koopman_graph.training` include
``pair_objectives`` (reconstruction / consistency composition),
``extra_objectives`` (Lie / PDE / sparsity / worst-case),
``objectives`` (``compute_training_loss`` orchestrator plus eigenvalue /
rollout), ``batched_objectives`` (opt-in collated multi-graph
reconstruction / forward), ``epochs`` (single-epoch train / eval helpers), ``inputs``
(multi-trajectory resolve), and ``loop`` (``run_fit_loop`` plus
early-stop / scheduler helpers). Prefer
``from koopman_graph.training import …``. The frozen internal
``TrainingLossBreakdown`` snapshot lives under
:mod:`koopman_graph.training.history` and is not a training-package
export.

.. automodule:: koopman_graph.training
   :members:
   :exclude-members: GraphSnapshotSequence
   :show-inheritance:

Experiment tracking adapters (power-user)
-----------------------------------------

Optional :class:`~koopman_graph.FitCallback` adapters live under
:mod:`koopman_graph.tracking` (not on root ``__all__``). Prefer
``from koopman_graph.tracking import CsvFitLogger, TensorBoardFitLogger``.
``TensorBoardFitLogger`` needs a peer ``tensorboard`` install. Cloud SDKs
(W&B, MLflow) are not pinned; see :doc:`faq` and
``examples/tracking/wandb_mlflow_callback.py``.

.. automodule:: koopman_graph.tracking
   :members:
   :imported-members:
   :show-inheritance:

Hyperparameter search helpers (power-user)
------------------------------------------

Optional HPO helpers live under :mod:`koopman_graph.tuning` (not on root
``__all__``). Prefer ``from koopman_graph.tuning import fit_history_metrics,
run_ray_tune``. This package is **not** an AutoML product: search spaces
remain caller-owned; ``example_*`` scaffolds are smoke-only.
``run_ray_tune`` and the example spaces require
``pip install "koopman-graph[ray]"`` (lazy import). Optuna is
examples-only (no library Optuna API). See
``examples/scripts/ray_tune_koopman_example.py``.

.. automodule:: koopman_graph.tuning
   :members:
   :imported-members:
   :show-inheritance:

Distributed trainers and Dask prep (power-user)
------------------------------------------------

Optional multi-process / multi-GPU *trainer orchestration* and offline
Dask materialize helpers live under :mod:`koopman_graph.distributed`
(not on root ``__all__``). This is **not** the operator flag
``sparsity="distributed"``. Prefer
``from koopman_graph.distributed import …``. Lazy symbols
(``KoopmanLightningModule``, ``run_ray_train_fit_loop``,
``fit_ensemble_with_ray``, ``materialize_sequences``,
``materialize_window_index_list``) require the matching extras
(``[lightning]`` / ``[ray]`` / ``[dask]``).
``run_ray_train_fit_loop`` is model DDP under Ray Train;
``fit_ensemble_with_ray`` parallelizes ensemble *members* — do not
conflate them (see :doc:`faq`). ``dask_prep`` is offline prep only —
not a Dask training loop.

.. automodule:: koopman_graph.distributed
   :members:
   :imported-members:
   :show-inheritance:

.. automodule:: koopman_graph.distributed.dask_prep
   :members:
   :show-inheritance:

Metrics
-------

Forecast evaluation entrypoints (``evaluate_forecast``,
``EvaluationResult``) and low-level helpers (``mae``, ``rmse``, ``mape``,
``HorizonMetrics``) are imported from :mod:`koopman_graph.metrics` only.

.. automodule:: koopman_graph.metrics
   :members:
   :show-inheritance:

Online Adaptation
-----------------

``RecursiveKoopmanAdapter``, ``AdaptationStepResult``, ``KoopmanObserver``,
``FilterResult``, ``JointStateTopologyObserver``, and
``JointObserverResult`` are imported from :mod:`koopman_graph.adaptation`
only. Shallow peers
``kalman`` (reference filter / RTS) and ``impute``
(``graph_diffuse_impute``) are power-user deep imports under the same
package. ``joint_observer`` lazily loads
:mod:`koopman_graph.identification`.
:class:`~koopman_graph.nn.SeparableDictionaryEncoder` /
:class:`~koopman_graph.nn.SeparableDictionaryDecoder` live under
:mod:`koopman_graph.nn` (not on the root façade).

.. automodule:: koopman_graph.adaptation
   :members:
   :show-inheritance:

Uncertainty Quantification (power-user)
---------------------------------------

Deep ensembles, latent-Gaussian forecast UQ, conformal intervals, and
Bayesian Laplace UQ over operator factors live under
:mod:`koopman_graph.uq` and are **not** on the root façade.
:class:`~koopman_graph.uq.EnsembleGraphKoopmanModel` composes independently
seeded :class:`~koopman_graph.model.GraphKoopmanModel` members
(Lakshminarayanan et al., NeurIPS 2017).
:class:`~koopman_graph.uq.LatentGaussianKoopmanUQ` propagates a Gaussian
latent under the linear Koopman map with optional Kalman refinement
(related to the Kalman half of K²VAE-style pipelines).
:class:`~koopman_graph.uq.ConformalKoopmanUQ` provides split and adaptive
(ACI) conformal prediction intervals with the shared
:class:`~koopman_graph.uq.PredictionInterval` type; marginal coverage
``≥ 1 − α`` assumes exchangeability and is approximate under temporal
dependence (prefer ACI under drift).
:attr:`~koopman_graph.uq.ConformalKoopmanUQ.coverage` names
:class:`~koopman_graph.uq.JointCoverageSpec`
``target="per_node_marginal"``; simultaneous and event targets raise.
Proper scores :func:`~koopman_graph.uq.gaussian_crps`,
:func:`~koopman_graph.uq.gaussian_nll`, and
:func:`~koopman_graph.uq.energy_score` evaluate forecasts and do not
certify coverage. Nonconformity ``score`` modes are
``"aggregate"``, legacy ``"per_node"`` (max-pool over nodes into a scalar),
and ``"node_wise"`` (per-node marginal half-widths; optional
``neighbor_smoothing``). Calibration persistence uses kind
``ConformalKoopmanUQ.calibration.v2``.
:class:`~koopman_graph.uq.BayesianKoopmanUQ` fits a diagonal Laplace
posterior over linear Koopman factors (seeded ``sample_forecast``); it is
**not** a Bayesian neural net over encoder weights.
Neither ensemble nor Gaussian path is Deep Probabilistic Koopman
(time-varying distribution parameters), and the Gaussian peer is **not**
a full VAE + KalmanNet reimplementation.

.. automodule:: koopman_graph.uq
   :members:
   :show-inheritance:

Hierarchical forecasting (power-user)
-------------------------------------

Multi-resolution pool → coarse Koopman → unpool lives under
:mod:`koopman_graph.hierarchical` and is **not** on the root façade.
:class:`~koopman_graph.hierarchical.HierarchicalGraphKoopmanModel` composes
:class:`~koopman_graph.model.GraphKoopmanModel` on a TopK-pooled (optional SAG)
graph. This is coarse-level forecasting with learned unpooling — **not** a
P-K-GCN-style physics-augmented spatiotemporal super-resolution pipeline
(Zhang et al., 2026). Graph-operator spectra use the **pooled** topology.
Global controls pass through; per-node controls follow the pooling ``perm``
chain.

.. automodule:: koopman_graph.hierarchical
   :members:
   :show-inheritance:

RL Environment
--------------

``GraphKoopmanEnv`` is imported from :mod:`koopman_graph.env` (not on the
root façade). Soft-imports Gymnasium so the module loads without the
``[rl]`` extra; construction fails at call time with install guidance.

.. automodule:: koopman_graph.env
   :members:
   :show-inheritance:

Koopman-MPC
-----------

``KoopmanMPC`` is imported from :mod:`koopman_graph.mpc` (not on the root
façade). OSQP is an optional ``[mpc]`` extra imported at solve time with
install guidance when missing. Additive discrete control solves one QP;
bilinear control uses sequential linearization (iterated QP). Optional
``constraint_tightening=`` accepts a calibrated
``ConformalKoopmanUQ`` to shrink output boxes by per-horizon half-widths.

.. automodule:: koopman_graph.mpc
   :members:
   :exclude-members: TubeKoopmanMPC, TubeMPCReport, ensemble_residual_radii
   :show-inheritance:

Tube Koopman-MPC
----------------

:class:`~koopman_graph.mpc.TubeKoopmanMPC` erodes nominal output boxes
by conformal quantiles or ensemble residual radii on **additive
discrete** plants. :class:`~koopman_graph.mpc.TubeMPCReport` records
constraint-violation rate, feasibility rate, and quadratic stage cost.
Local decoder linearization remains. This is not a chance-constraint
solver and not a recursive-feasibility or Lyapunov closed-loop
certificate (``Zhang2022TubeMPC``). Not a factory kind; off root
``__all__``.

.. automodule:: koopman_graph.mpc.tube
   :members:
   :show-inheritance:

Serialization
-------------

.. automodule:: koopman_graph.serialization
   :members:
   :show-inheritance:

Export, federated averaging, robustness, probabilistic
------------------------------------------------------

Power-user modules, off root ``__all__``. Export is a fixed-topology
discrete homogeneous MVP. Federated averaging is not DP-SGD.

.. automodule:: koopman_graph.export
   :members:
   :show-inheritance:

.. automodule:: koopman_graph.federated
   :members:
   :show-inheritance:

.. automodule:: koopman_graph.robustness
   :members:
   :show-inheritance:

.. automodule:: koopman_graph.probabilistic
   :members:
   :show-inheritance:

Datasets
--------

.. automodule:: koopman_graph.datasets
   :members:
   :show-inheritance:

.. automodule:: koopman_graph.datasets.dynamics
   :members:
   :show-inheritance:

.. automodule:: koopman_graph.datasets.topology
   :members:
   :show-inheritance:

.. automodule:: koopman_graph.datasets.download
   :members:
   :show-inheritance:

.. automodule:: koopman_graph.datasets.cache_cli
   :members:
   :show-inheritance:

.. automodule:: koopman_graph.datasets.synthetic
   :members:
   :show-inheritance:

.. automodule:: koopman_graph.datasets.grid
   :members:
   :show-inheritance:

.. automodule:: koopman_graph.datasets.ieee118
   :members:
   :show-inheritance:

.. automodule:: koopman_graph.datasets.metr_la
   :members:
   :show-inheritance:

.. automodule:: koopman_graph.datasets.pems
   :members:
   :show-inheritance:

.. automodule:: koopman_graph.datasets.contact_epidemic
   :members:
   :show-inheritance:

.. automodule:: koopman_graph.datasets.epidemic
   :members:
   :show-inheritance:

.. automodule:: koopman_graph.datasets.lorenz96
   :members:
   :show-inheritance:

.. automodule:: koopman_graph.datasets.kuramoto_sivashinsky
   :members:
   :show-inheritance:

.. automodule:: koopman_graph.datasets.cylinder_wake
   :members:
   :show-inheritance:

.. automodule:: koopman_graph.datasets.molecular
   :members:
   :show-inheritance:

.. automodule:: koopman_graph.datasets.era5
   :members:
   :show-inheritance:

External toolchain interop (power-user)
---------------------------------------

Optional deeptime trajectory-feature bridges under
:mod:`koopman_graph.interop` (lazy ``[msm]``). Off root ``__all__``.
Other ``koopman_graph`` modules must **not** import ``interop`` (acyclic
layer boundary). Teaching / diagnostic — not a PyEMMA replacement.
:mod:`koopman_graph.interop.topologicx` is the optional ``[tdl]``
incidence-tensor bridge (not TopologicX feature parity).

.. automodule:: koopman_graph.interop
   :members:
   :show-inheritance:
