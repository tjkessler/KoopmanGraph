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

.. automodule:: koopman_graph.model
   :members:
   :exclude-members: EvaluationResult, GATDecoder, GATEncoder, GNNDecoder, GNNEncoder, GraphSnapshotSequence, GraphKoopmanOperator, KoopmanOperator
   :show-inheritance:

Encoders
--------

Built-in encoders live in :mod:`koopman_graph.nn` (``encoder``, ``decoder``,
``gnn``). Prefer ``from koopman_graph import GNNEncoder, GATEncoder`` for
application code, or ``from koopman_graph.nn import …`` for power-user imports.
Former deep imports ``koopman_graph.encoder`` / ``decoder`` / ``gnn`` were
removed in v0.3.0.

.. automodule:: koopman_graph.nn.encoder
   :members:
   :show-inheritance:

Delay Embeddings
----------------

Hankel / delay-coordinate lifting wraps a sized base encoder. Prefer
``from koopman_graph import DelayEmbeddingEncoder`` or pass
``n_delays`` to :class:`~koopman_graph.model.GraphKoopmanModel`.
This is Takens-style channel stacking, not a full HAVOK / Hankel-DMD solver.

.. automodule:: koopman_graph.nn.delay
   :members:
   :show-inheritance:

Decoder
-------

.. automodule:: koopman_graph.nn.decoder
   :members:
   :show-inheritance:

Shared GNN Primitives (power-user)
----------------------------------

Neutral message-passing helpers shared by peer encoder and decoder modules
(:class:`~koopman_graph.nn.gnn.BaseGNNModule`, activation typing, GCN/GAT
builders). Importable via :mod:`koopman_graph.nn.gnn`, but **not** part of the
stable public façade (not in ``koopman_graph.__all__``). Prefer the public
encoder/decoder classes for application code. See :doc:`architecture`.

.. automodule:: koopman_graph.nn.gnn
   :members:
   :show-inheritance:

Physics-Informed Observables
----------------------------

Hybrid physics helpers such as ``graph_laplacian_features`` are imported from
:mod:`koopman_graph.observables` only. Prefer
``physics_preset="graph_laplacian"`` on :class:`~koopman_graph.model.GraphKoopmanModel`
for the built-in Laplacian path; pass a custom ``physics_lifting_fn`` when needed.

.. automodule:: koopman_graph.observables
   :members:
   :show-inheritance:

Koopman Operator
----------------

Built-in operators live in :mod:`koopman_graph.operators` (``contract``,
``control``, ``discrete``, ``continuous``, ``graph``). Prefer ``from koopman_graph import
KoopmanOperator, ContinuousKoopmanOperator, GraphKoopmanOperator`` or
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
decoder). Prefer ``from koopman_graph.baselines import …`` or the root façade.

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

Containers and split helpers (``GraphSnapshotSequence``, ``MultiTrajectory``,
``TemporalSplit``, ``temporal_split``, ``WindowSampler``) remain on the root
façade. ``as_multi_trajectory`` is imported from :mod:`koopman_graph.data`
only.

.. automodule:: koopman_graph.data
   :members:
   :show-inheritance:

Shared Graph Utilities (power-user)
-----------------------------------

Documented internal helpers for graph-input resolution and latent propagation.
Importable, but **not** part of the stable public façade (not in
``koopman_graph.__all__``). Use :meth:`~koopman_graph.model.GraphKoopmanModel.encode`
when lifting snapshots. See :doc:`architecture`.

.. automodule:: koopman_graph.graph_utils
   :members:
   :show-inheritance:

Losses
------

.. automodule:: koopman_graph.losses
   :members:
   :exclude-members: GraphSnapshotSequence
   :show-inheritance:

Training
--------

.. automodule:: koopman_graph.training
   :members:
   :exclude-members: GraphSnapshotSequence
   :show-inheritance:

Metrics
-------

Primary forecast evaluation entrypoints (``evaluate_forecast``,
``EvaluationResult``) remain on the root façade. Low-level helpers
(``mae``, ``rmse``, ``mape``, ``HorizonMetrics``) are imported from
:mod:`koopman_graph.metrics` only.

.. automodule:: koopman_graph.metrics
   :members:
   :show-inheritance:

Online Adaptation
-----------------

``RecursiveKoopmanAdapter`` remains on the root façade.
``AdaptationStepResult``, ``KoopmanObserver``, and ``FilterResult`` are
imported from :mod:`koopman_graph.adaptation` only.

.. automodule:: koopman_graph.adaptation
   :members:
   :show-inheritance:

RL Environment
--------------

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

.. automodule:: koopman_graph.datasets.nonlinear
   :members:
   :show-inheritance:
