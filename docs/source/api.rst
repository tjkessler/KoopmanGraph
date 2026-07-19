API Reference
=============

Public classes and functions are documented by module below. Core workflow
symbols are re-exported from :mod:`koopman_graph` (see ``__all__``); specialized
helpers live in capability modules such as :mod:`koopman_graph.metrics`,
:mod:`koopman_graph.analysis`, :mod:`koopman_graph.data`,
:mod:`koopman_graph.adaptation`, and :mod:`koopman_graph.observables`.

For the public vs power-user vs private layering contract, device conventions,
and design philosophy, see :doc:`architecture`.

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
stacking, not a full HAVOK / Hankel-DMD solver. Pure snapshot/tensor window
helpers are owned by :mod:`koopman_graph.data.delay_windows` and
thin-re-exported from this module for compatibility.

.. automodule:: koopman_graph.nn.delay
   :members:
   :imported-members:
   :show-inheritance:

Decoder
-------

.. automodule:: koopman_graph.nn.decoder
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
``auxiliary_spectral``, ``graph``). Prefer
``from koopman_graph import KoopmanOperator, ContinuousKoopmanOperator,
GraphKoopmanOperator`` (all three are root-stable ``__all__`` members) or
``from koopman_graph.operators import …``. Former deep imports
``koopman_graph.operator`` / ``koopman_graph.continuous`` were removed in
v0.3.0.

.. automodule:: koopman_graph.operators
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
``AnomalyDetectionResult``, ``plot_spectrum``) are imported from
:mod:`koopman_graph.analysis` only. The helpers live in
the ``spectrum`` / ``similarity`` / ``anomaly`` / ``plotting`` submodules.
``plot_spectrum`` requires Matplotlib (``pip install matplotlib`` or the
``[dev]`` extra).

.. automodule:: koopman_graph.spectrum_types
   :members:
   :show-inheritance:

.. automodule:: koopman_graph.analysis
   :members:
   :exclude-members: KoopmanSpectrum
   :show-inheritance:

Baselines
---------

Classical DMD-family baselines live in the ``koopman_graph.baselines``
capability package (``base`` / ``dmd`` / ``dmdc`` / ``edmd`` submodules). They
share :class:`~koopman_graph.baselines.ClassicalBaseline` scaffolding and
satisfy :class:`~koopman_graph.protocols.ForecastModel` (``fit`` / ``predict`` /
``spectrum``). Call sites are **not** fully interchangeable — see the
``ForecastModel`` call-site matrix in :doc:`architecture`.
:class:`~koopman_graph.baselines.EDMDBaseline` exposes
``reconstruction_matrix`` for observable-to-state least squares (not a GNN
decoder) and supports ``dictionary`` in ``{"polynomial", "rbf", "kernel"}``
(Williams2015 polynomial / RBF EDMD; kernel sections following
Williams2015KernelDMD / Klus2018TransferOperator).
The full ``kernel`` path with one center per training snapshot is
:math:`O(T^2)` in feature dimension and is intended for small/medium ``T``
only; ``kernel="linear"`` reduces to DMD. Prefer
``from koopman_graph.baselines import …``.

Spatiotemporal GNN forecaster baselines
(:class:`~koopman_graph.baselines.gnn.STGCNBaseline`,
:class:`~koopman_graph.baselines.gnn.DCRNNBaseline`,
:class:`~koopman_graph.baselines.gnn.GraphWaveNetBaseline`) live under
``koopman_graph.baselines.gnn``. They are lightweight ``nn.Module`` reference
implementations for protocol-matched comparisons with
:class:`~koopman_graph.model.GraphKoopmanModel` (including
:func:`~koopman_graph.metrics.evaluate_forecast`). Their ``spectrum`` method
raises ``RuntimeError`` (no linear Koopman operator). Prefer
``from koopman_graph.baselines.gnn import …``.

.. automodule:: koopman_graph.baselines
   :members:
   :exclude-members: STGCNBaseline, DCRNNBaseline, GraphWaveNetBaseline
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
``validation`` / ``trajectories`` / ``delay_windows`` / ``sampling`` /
``splits`` / ``rollout``; prefer ``from koopman_graph.data import …``.
Array→snapshot builders are power-user imports from
:mod:`koopman_graph.data.construction`. Delay-window stack/flatten helpers
are power-user imports from :mod:`koopman_graph.data.delay_windows` (also
re-exported by :mod:`koopman_graph.nn.delay`).

.. automodule:: koopman_graph.data
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
and latent propagation. :mod:`koopman_graph.graph_utils` is a shallow
capability package (``topology`` / ``propagation`` peers) whose ``__init__``
re-exports the documented surface. Importable, but **not** part of the stable
public façade (not in ``koopman_graph.__all__``). Use
:meth:`~koopman_graph.model.GraphKoopmanModel.encode` when lifting snapshots.
See :doc:`architecture`.

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
rollout), ``epochs`` (single-epoch train / eval helpers), ``inputs``
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
and ``FilterResult`` are imported from :mod:`koopman_graph.adaptation`
only. Shallow peers
``kalman`` (reference filter / RTS) and ``impute``
(``graph_diffuse_impute``) are power-user deep imports under the same
package.

.. automodule:: koopman_graph.adaptation
   :members:
   :show-inheritance:

Uncertainty Quantification (power-user)
---------------------------------------

Deep ensembles and latent-Gaussian forecast UQ live under
:mod:`koopman_graph.uq` and are **not** on the root façade.
:class:`~koopman_graph.uq.EnsembleGraphKoopmanModel` composes independently
seeded :class:`~koopman_graph.model.GraphKoopmanModel` members
(Lakshminarayanan et al., NeurIPS 2017).
:class:`~koopman_graph.uq.LatentGaussianKoopmanUQ` propagates a Gaussian
latent under the linear Koopman map with optional Kalman refinement
(related to the Kalman half of K²VAE-style pipelines). Neither path is Deep
Probabilistic Koopman (which predicts time-varying distribution parameters),
and the Gaussian peer is **not** a full VAE + KalmanNet reimplementation.

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

Serialization
-------------

.. automodule:: koopman_graph.serialization
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
