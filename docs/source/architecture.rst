Architecture and API layers
===========================

This page is the maintainer-facing contract for how KoopmanGraph is structured.
It exists so contributors and agents do not reintroduce style drift across
releases. Read it when changing package layout, exports, device handling, or
shared helpers.

Design philosophy
-----------------

KoopmanGraph combines four complementary styles:

1. **Composition** — users assemble an encoder, decoder, and Koopman operator
   (discrete or continuous) into :class:`~koopman_graph.model.GraphKoopmanModel`.
   Tutorials typically omit ``koopman=`` and let the model factory-build a
   built-in operator from ``dynamics_mode`` / ``koopman_parameterization``
   strings. Advanced callers may inject a custom
   :class:`~koopman_graph.operators.KoopmanOperatorContract` ``nn.Module`` via
   ``koopman=...`` (mutually exclusive with non-default factory kwargs).
   Custom operators are not checkpoint-round-trippable; see serialization
   notes below.
2. **Sklearn-like façade** — the model exposes ``fit`` / ``predict`` /
   ``evaluate`` / ``save`` / ``load`` as the primary workflow.
3. **Functional training and evaluation** — lower-level helpers in
   :mod:`koopman_graph.training` and :mod:`koopman_graph.metrics` implement
   losses, epoch loops, and forecast metrics without requiring subclassing.
   The multi-epoch fit loop lives in
   :func:`~koopman_graph.training.run_fit_loop`;
   :meth:`~koopman_graph.model.GraphKoopmanModel.fit` validates sequences /
   controls and delegates to that helper.
4. **String-mode configuration** — behavioral modes are selected with explicit
   strings (for example ``dynamics_mode="discrete"|"continuous"`` and
   ``koopman_parameterization="dense"|"structural"|"odo"``) rather than parallel
   class hierarchies.

Prefer extending these patterns over introducing a fifth style (global device
managers, deep inheritance trees, or underscore helpers used as a second public
API).

Package layout
--------------

Target style is **capability packages at one level** (similar to
``torch_geometric.nn`` or ``sklearn.linear_model``): group related peers under
a single shallow package when growth warrants it. Do **not** over-nest “because
PyG does,” invent one folder per tiny module, or introduce a fifth architectural
style beyond composition / sklearn façade / functional helpers / string-mode
configuration.

When to nest
~~~~~~~~~~~~

Convert a flat module into a capability package when **any** of these hold:

* **Multi-concern grab-bag** — one file mixes several cohesive jobs
  (orchestration + schedules + history types, containers + splits + sampling,
  spectrum + anomaly + similarity) and is already ~1k+ lines of maintainable
  surface.
* **Peer implementations** — two or more sibling implementations share a
  contract (discrete/continuous operators; encoder/decoder/GNN primitives;
  classical DMD-family baselines).
* **Room to grow peers** — a new classical method or GNN variant should land
  as a sibling module, not another flat mega-file.

Phase 8 capability packages: ``training/``, ``data/``, ``operators/``,
``nn/``, ``analysis/``, and ``baselines/`` (all landed).

``koopman_graph.training`` package layout:

* ``history`` — ``LossWeights``, ``FitHistory``, ``TrainingLossBreakdown``,
  fit input type aliases
* ``schedules`` — constant / ramp loss-weight schedules
* ``device`` — ``resolve_device``, ``sequence_to_device``
* ``objectives`` — training-side composition over top-level
  :mod:`koopman_graph.losses` (left flat; no cycle benefit from nesting)
* ``loop`` — epoch helpers, input resolution, ``run_fit_loop``

``koopman_graph.data`` package layout (kept separate from
:mod:`koopman_graph.datasets`):

* ``containers`` — ``GraphSnapshotSequence``, ``MultiTrajectory``,
  ``as_multi_trajectory``, ``resolve_sequence``, ``resolve_pair_delta_t``
* ``sampling`` — ``WindowSampler``
* ``splits`` — ``TemporalSplit``, ``temporal_split``
* ``rollout`` — ``resolve_rollout_start_indices`` and related type aliases

``koopman_graph.operators`` package layout (peer discrete/continuous pair):

* ``contract`` — ``KoopmanOperatorContract``, ``Parameterization``,
  ``DynamicsMode``, ``StabilityCertificate``, shared structural helpers
* ``discrete`` — :class:`~koopman_graph.operators.discrete.KoopmanOperator`
* ``continuous`` — :class:`~koopman_graph.operators.continuous.ContinuousKoopmanOperator`
  and Van Loan helpers

Prefer ``from koopman_graph.operators import …`` (or the root façade for public
operator classes). Former root modules ``koopman_graph.operator`` and
``koopman_graph.continuous`` were removed in v0.3.0.

``koopman_graph.nn`` package layout (peer encoder / decoder / GNN primitives;
PyG-style ``nn`` capability package, no ``conv/`` subtree):

* ``gnn`` — :class:`~koopman_graph.nn.gnn.BaseGNNModule`, activation typing,
  validators, GCN/GAT convolution builders (power-user)
* ``encoder`` — :class:`~koopman_graph.nn.encoder.GNNEncoder` /
  :class:`~koopman_graph.nn.encoder.GATEncoder`
* ``decoder`` — :class:`~koopman_graph.nn.decoder.GNNDecoder` /
  :class:`~koopman_graph.nn.decoder.GATDecoder`

Prefer ``from koopman_graph.nn import …`` (or the root façade for public
classes). Former root modules ``koopman_graph.encoder``,
``koopman_graph.decoder``, and ``koopman_graph.gnn`` were removed in v0.3.0.
Encoder and decoder remain peers: both import from ``nn.gnn``;
neither imports the other.

``koopman_graph.analysis`` package layout (spectrum / similarity / anomaly /
plotting):

* ``spectrum`` — ``compute_spectrum``, ``compute_generator_spectrum``,
  ``discrete_spectrum_at_delta_t``, ``decode_mode_shapes``
* ``similarity`` — ``spectrum_distance``, ``koopman_std``, ``resolve_spectrum``,
  ``dynamical_similarity``
* ``anomaly`` — ``AnomalyDetectionResult``, ``calibrate_anomaly_threshold``,
  ``detect_anomaly``
* ``plotting`` — ``plot_spectrum`` (discrete unit-disk / data-zoom complex-plane
  figures; Matplotlib call-site import)

:mod:`koopman_graph.spectrum_types` remains a **top-level neutral leaf** so
:mod:`koopman_graph.protocols` never imports heavy analysis code.
``KoopmanSpectrum`` is re-exported from :mod:`koopman_graph.analysis` and the
package root for the public API.

``koopman_graph.baselines`` package layout (peer DMD-family methods behind
:class:`~koopman_graph.baselines.ClassicalBaseline`):

* ``base`` — :class:`~koopman_graph.baselines.ClassicalBaseline` and shared
  flattening / least-squares / topology-control guard helpers
* ``dmd`` — :class:`~koopman_graph.baselines.DMDBaseline`
* ``dmdc`` — :class:`~koopman_graph.baselines.DMDcBaseline`
* ``edmd`` — :class:`~koopman_graph.baselines.EDMDBaseline`

Deep imports continue to use ``from koopman_graph.training import …``,
``from koopman_graph.data import …``, ``from koopman_graph.analysis import …``,
and ``from koopman_graph.baselines import …`` via package ``__init__``
re-exports (same-named package compatibility).

When to stay flat
~~~~~~~~~~~~~~~~~

Keep a single module (or leave an existing small package alone) when:

* The module is **small and single-purpose** (for example ``metrics``,
  ``losses``, ``env``, ``adaptation``, ``observables``, ``serialization``,
  ``protocols``, ``graph_utils``, ``spectrum_types``).
* :mod:`koopman_graph.datasets` is already the correct benchmark/load
  subpackage — do **not** merge it into ``data/`` (data structures vs
  datasets, same distinction as sklearn/PyG).
* Nesting would only rename a path without clarifying capability boundaries.

Explicit non-goals unless a later audit shows multi-peer growth: nesting
``env``, ``adaptation``, ``observables``, ``metrics``, ``serialization``,
``protocols``, or ``graph_utils``.

``model.py`` stays at the package root
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~koopman_graph.model.GraphKoopmanModel` and :mod:`koopman_graph.model`
remain top-level. Do **not** split the façade across packages or bury the
primary workflow class under ``nn/``, ``training/``, or similar.

No deep trees
~~~~~~~~~~~~~

One nesting level is enough (``koopman_graph.nn``, ``koopman_graph.operators``).
Do **not** add deep trees such as ``nn/conv/...`` or ``operators/discrete/...``
unless a future audit shows a genuine multi-layer library surface. Prefer peer
modules inside the capability package.

Three-layer API regardless of folders
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Moving files must **not** change the layering contract above:

1. **Public façade** — symbols in package ``__all__`` / ``from koopman_graph
   import …``
2. **Power-user modules** — importable packages or modules documented here but
   not newly promoted into ``__all__``
3. **Private helpers** — leading-``_`` names, same-module only

Folders are an organization tool; they do not create a fourth API tier.

Compatibility contract
~~~~~~~~~~~~~~~~~~~~~~

When splitting or nesting modules:

* **Root façade is stable** — ``from koopman_graph import …`` symbols and
  package ``__all__`` must keep the same public names and semantics.
* **Power-user deep imports** — for each move, choose **one** and document it
  in this page (and a short migration note if paths change):

  * **Same-named package** — ``koopman_graph.training``, ``data``,
    ``analysis``, ``baselines`` keep working via package ``__init__``
    re-exports when the old flat module is replaced by a package of the same
    name; or
  * **Migrate in-repo** — update every docs/tests/notebook deep import to the
    new path and record the rename here. **Do not** add long-lived root shim
    modules for renamed paths (``operator`` → ``operators``, ``encoder`` →
    ``nn.encoder``, …). v0.3.0 removed those temporary root shims.

Do not leave half-migrated import paths. Serialization type strings and
checkpoint reconstruct rules stay intact unless a task explicitly bumps
``FORMAT_VERSION``.

API layers
----------

The package has three intentional layers.

Public façade
~~~~~~~~~~~~~

Stable, supported entry points re-exported from :mod:`koopman_graph` (see
``__all__``). Prefer these in tutorials, notebooks, and application code.

v0.3.0 uses a **thin** root façade: keep the core encode → evolve → decode /
fit / evaluate workflow at the package root; import specialized helpers from
capability modules. Demotions are a **hard cut** (no root deprecation
aliases), consistent with the shim-removal policy above.

**Keep in** ``koopman_graph.__all__`` (core workflow):

* :class:`~koopman_graph.model.GraphKoopmanModel`
* :class:`~koopman_graph.nn.encoder.GNNEncoder`,
  :class:`~koopman_graph.nn.encoder.GATEncoder`,
  :class:`~koopman_graph.nn.decoder.GNNDecoder`,
  :class:`~koopman_graph.nn.decoder.GATDecoder`
  (also via :mod:`koopman_graph.nn`)
* :class:`~koopman_graph.operators.KoopmanOperator`,
  :class:`~koopman_graph.operators.ContinuousKoopmanOperator`
* Classical baselines (:class:`~koopman_graph.baselines.DMDBaseline`,
  :class:`~koopman_graph.baselines.EDMDBaseline`,
  :class:`~koopman_graph.baselines.DMDcBaseline`)
* Data I/O for ``fit``: :class:`~koopman_graph.data.GraphSnapshotSequence`,
  :class:`~koopman_graph.data.MultiTrajectory`,
  :class:`~koopman_graph.data.TemporalSplit`,
  :func:`~koopman_graph.data.temporal_split`,
  :class:`~koopman_graph.data.WindowSampler`
* Training knobs: :class:`~koopman_graph.losses.ForwardConsistencyLoss`,
  :class:`~koopman_graph.losses.BackwardConsistencyLoss`,
  :class:`~koopman_graph.losses.EigenvalueRegularizationLoss`,
  :class:`~koopman_graph.training.FitHistory`,
  :class:`~koopman_graph.training.LossWeights`
* Featured Phase 6 surfaces:
  :class:`~koopman_graph.adaptation.RecursiveKoopmanAdapter`,
  :class:`~koopman_graph.env.GraphKoopmanEnv`
* Primary eval / spectrum entrypoints:
  :func:`~koopman_graph.metrics.evaluate_forecast`,
  :class:`~koopman_graph.metrics.EvaluationResult`,
  :class:`~koopman_graph.spectrum_types.KoopmanSpectrum`,
  :func:`~koopman_graph.analysis.compute_spectrum`
* ``__version__``

Dataset generators remain via :mod:`koopman_graph.datasets` (not root
``__all__`` members; use ``from koopman_graph.datasets import …``).

**Demote to module imports** (still public and documented; not in root
``__all__``):

* :mod:`koopman_graph.metrics` — ``mae``, ``rmse``, ``mape``,
  ``HorizonMetrics``
* :mod:`koopman_graph.analysis` — ``compute_generator_spectrum``,
  ``discrete_spectrum_at_delta_t``, ``decode_mode_shapes``,
  ``spectrum_distance``, ``koopman_std``, ``dynamical_similarity``,
  ``detect_anomaly``, ``calibrate_anomaly_threshold``,
  ``AnomalyDetectionResult``
* :mod:`koopman_graph.data` — ``as_multi_trajectory``
* :mod:`koopman_graph.adaptation` — ``AdaptationStepResult``
* :mod:`koopman_graph.observables` — ``graph_laplacian_features``

.. code-block:: python

   from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
   from koopman_graph.analysis import dynamical_similarity
   from koopman_graph.data import as_multi_trajectory
   from koopman_graph.metrics import rmse
   from koopman_graph.observables import graph_laplacian_features

Power-user modules
~~~~~~~~~~~~~~~~~~

Importable modules that are documented and useful for advanced workflows, but
**not** advertised as a stable public contract. They may change without a major
version bump. Do not re-export new power-user symbols from
:mod:`koopman_graph` ``__all__`` unless a blueprint task explicitly promotes
them.

Examples:

* :mod:`koopman_graph.training` — loss schedules, ``train_one_epoch``,
  ``run_fit_loop``, ``resolve_device``, and related helpers
* :mod:`koopman_graph.serialization` — checkpoint build/load internals behind
  ``GraphKoopmanModel.save`` / ``load``
* :mod:`koopman_graph.datasets.dynamics` — Laplacian diffusion primitives and
  shared generation validators used by benchmark generators (dense step
  operators; shares ``L_sym`` weights with :mod:`koopman_graph.graph_utils`)
* :mod:`koopman_graph.graph_utils` — shared graph-input resolution, latent
  propagation, and symmetric-normalized adjacency helpers (not in ``__all__``)
* :mod:`koopman_graph.nn` — encoder / decoder / shared GNN primitives
  capability package (:class:`~koopman_graph.nn.gnn.BaseGNNModule`,
  activation typing, GCN/GAT builders). The package is power-user; public
  encoder/decoder *classes* remain in ``__all__``. Peers import from
  ``nn.gnn`` only (no encoder↔decoder inversion).
* :mod:`koopman_graph.spectrum_types` — neutral
  :class:`~koopman_graph.spectrum_types.KoopmanSpectrum` value type
  (re-exported from :mod:`koopman_graph.analysis` and the package root)
* :class:`~koopman_graph.operators.KoopmanOperatorContract` — shared Protocol for
  discrete and continuous operators (``matrix``, ``advance``,
  ``inverse_advance``, ``bound_metric``); importable from
  :mod:`koopman_graph.operators` / used as ``KoopmanPropagator`` in
  :mod:`koopman_graph.graph_utils`
* :class:`~koopman_graph.protocols.ForecastModel` — loose Protocol for
  ``fit`` / ``predict`` / ``spectrum`` on classical baselines and
  :class:`~koopman_graph.model.GraphKoopmanModel` (method presence only; **not**
  drop-in interchangeable at call sites)
* :class:`~koopman_graph.protocols.SpectrumProvider` — spectrum-only surface
  used by :func:`~koopman_graph.analysis.dynamical_similarity` (baselines and
  neural models; optional ``delta_t`` only when the implementer accepts it)
* :class:`~koopman_graph.protocols.UncontrolledForecastModel` — narrower peer
  for autonomous ``predict(initial_graph: Data, steps: int)``; use
  :func:`~koopman_graph.protocols.accepts_uncontrolled_data_predict` to reject
  controlled-only implementers at runtime
* :class:`~koopman_graph.protocols.TrainableKoopmanModel` — extends
  :class:`~koopman_graph.protocols.ForecastModel` with the encode / operator /
  Module members that :mod:`koopman_graph.training` and
  :func:`~koopman_graph.metrics.evaluate_forecast` duck-type

.. code-block:: python

   from koopman_graph.training import resolve_device, train_one_epoch
   from koopman_graph.datasets.dynamics import laplacian_diffusion_rollout
   from koopman_graph.protocols import (
       ForecastModel,
       SpectrumProvider,
       TrainableKoopmanModel,
       UncontrolledForecastModel,
   )

Phase 8 note: shared utilities live in :mod:`koopman_graph.graph_utils` —
documented, importable as ``koopman_graph.graph_utils``, and **not** listed in
``__all__``. Prefer this module over importing leading-underscore helpers from
peer packages. Typing Protocols
(:class:`~koopman_graph.protocols.ForecastModel`,
:class:`~koopman_graph.protocols.SpectrumProvider`,
:class:`~koopman_graph.protocols.UncontrolledForecastModel`,
:class:`~koopman_graph.protocols.TrainableKoopmanModel`,
:class:`~koopman_graph.operators.KoopmanOperatorContract`) and neutral value
types (:mod:`koopman_graph.spectrum_types`) stay power-user and are likewise
omitted from ``__all__``.

**Dependency direction.** Prefer
``types/protocols → data / graph_utils → feature modules → façade``:

* :mod:`koopman_graph.protocols` imports
  :class:`~koopman_graph.spectrum_types.KoopmanSpectrum` from
  :mod:`koopman_graph.spectrum_types`, not from :mod:`koopman_graph.analysis`.
* :func:`~koopman_graph.data.resolve_rollout_start_indices` lives next to
  :class:`~koopman_graph.data.GraphSnapshotSequence` and is shared by
  :mod:`koopman_graph.training` and :mod:`koopman_graph.metrics`.
* Forecast metrics reuse :func:`~koopman_graph.losses.masked_mse_loss` for
  masked RMSE so training and evaluation share one masked-MSE reduction.

Encode API
~~~~~~~~~~

Prefer :meth:`~koopman_graph.model.GraphKoopmanModel.encode` for latent lifting.
It accepts a ``Data`` snapshot or explicit ``(x, edge_index, edge_weight)``
tensors and applies hybrid physics observables when configured.
Training helpers require ``encode`` via
:class:`~koopman_graph.protocols.TrainableKoopmanModel` (no encoder-only
fallback).

Private helpers
~~~~~~~~~~~~~~~

Names that begin with a leading underscore (``_helper``, ``_validate_…``) are
**not a public contract**. They are implementation details of their enclosing
module.

Rules:

* Call private helpers only from the same module (or tightly coupled tests).
* Do **not** import leading-``_`` symbols across modules. Cross-module reuse
  must go through a documented internal (power-user) module without a leading
  underscore (see :mod:`koopman_graph.graph_utils`).
* Autodoc may still render some private members for maintainers; that does not
  make them supported API.

Duck-typed training vs hard-typed env / checkpoints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two coupling styles coexist on purpose:

* **Duck-typed training and evaluation** —
  :func:`~koopman_graph.training.train_one_epoch`,
  :func:`~koopman_graph.losses.rollout_sequence_loss` /
  :func:`~koopman_graph.losses.rollout_multi_start_loss`, other loss helpers,
  and :func:`~koopman_graph.metrics.evaluate_forecast` annotate against
  :class:`~koopman_graph.protocols.TrainableKoopmanModel`. They require
  ``encode``, ``predict``, ``encoder`` / ``koopman`` / ``decoder``,
  ``time_step``, ``dynamics_mode``, ``control_dim``, ``resolve_delta_t``, and
  the ``nn.Module`` train/eval/parameters façade (no encoder-only fallback).
  Classical baselines satisfy only
  :class:`~koopman_graph.protocols.ForecastModel` and are **not** training
  targets for these helpers. ``TrainableKoopmanModel`` disables runtime
  ``isinstance`` (inherited ``@runtime_checkable`` would be unreliable because
  submodule attributes live in ``nn.Module._modules``). Use static typing or
  structural smoke tests instead.
* **Hard-typed Gymnasium and serialization** —
  :class:`~koopman_graph.env.GraphKoopmanEnv` and checkpoint reconstruct in
  :mod:`koopman_graph.serialization` require
  :class:`~koopman_graph.model.GraphKoopmanModel` specifically. They freeze or
  rebuild concrete encoder/decoder classes and architecture config fields that
  a structural Protocol cannot safely express. Env is **not** required to accept
  arbitrary Protocol implementers.

Shared graph utilities
~~~~~~~~~~~~~~~~~~~~~~

:mod:`koopman_graph.graph_utils` consolidates graph-input resolution, snapshot
device transfer, discrete/continuous latent propagation, and the shared
autoregressive rollout loop:

* ``snapshot_edge_weight``, ``resolve_graph_inputs``, ``resolve_edge_index``,
  ``resolve_edge_weight``, ``snapshot_to_device``
* ``propagate_latent`` / ``inverse_propagate_latent``
* ``advance_and_decode``, ``autoregressive_latent_rollout``,
  ``hold_last_topology_at``, ``snapshot_topology_at``

Continuous ``delta_t`` policy: pass an explicit ``delta_t`` when known. When
``delta_t is None``, :func:`~koopman_graph.graph_utils.resolve_delta_t` (used by
``propagate_latent`` / ``inverse_propagate_latent``) applies
``default_delta_t``. Model-backed callers — including
:meth:`~koopman_graph.model.GraphKoopmanModel.resolve_delta_t`, training/loss
pair helpers, and :class:`~koopman_graph.env.GraphKoopmanEnv` — pass
``time_step``. Bare helpers and
:meth:`~koopman_graph.operators.ContinuousKoopmanOperator.forward` soft-default
to ``1.0`` outside a model context. Functional pair losses still prefer
timestamp intervals from :func:`~koopman_graph.data.resolve_pair_delta_t` when
present.

Autoregressive rollouts share one encode → advance → decode primitive
(:func:`~koopman_graph.graph_utils.autoregressive_latent_rollout`). Topology
policy is intentional and documented at the call site:

* **Hold-last (inference)** — ``predict`` / ``_rollout`` via
  ``hold_last_topology_at`` (optional ``future_topologies`` schedule).
* **Teacher targets (training)** — ``rollout_sequence_loss`` via
  ``snapshot_topology_at`` on observed target snapshots.

``GraphKoopmanEnv.step`` uses the same one-step ``advance_and_decode`` helper.

Topology capability matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~

Dynamic topology is first-class in the data container and neural model path,
but not every consumer supports it. Callers must not assume silent freeze/flatten
behavior:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Surface
     - Dynamic topology (``is_dynamic_topology=True``)
   * - :class:`~koopman_graph.data.GraphSnapshotSequence`
     - Supported via ``allow_dynamic_topology=True`` /
       :meth:`~koopman_graph.data.GraphSnapshotSequence.from_dynamic_arrays`
   * - Model ``predict`` / inference rollout
     - Supported: hold-last topology (optional ``future_topologies`` schedule)
   * - Training losses (``rollout_sequence_loss``)
     - Supported: teacher-target topology from observed snapshots
   * - :class:`~koopman_graph.env.GraphKoopmanEnv`
     - **Rejected** at construction (episode topology frozen from reset)
   * - Classical baselines (DMD / EDMD / DMDc)
     - **Rejected** at ``fit`` (flatten states; freeze initial edges on
       ``predict``)

Control layout capability matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~koopman_graph.data.GraphSnapshotSequence` stores either global
``(T, control_dim)`` or per-node ``(T, N, control_dim)`` controls. Consumers
must not assume the same physics for both layouts:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Surface
     - Supported control layouts
   * - :class:`~koopman_graph.data.GraphSnapshotSequence`
     - Global ``(T, C)`` and per-node ``(T, N, C)``
   * - :class:`~koopman_graph.model.GraphKoopmanModel` / operators
     - Global broadcast and **per-node row matching**
       (``u`` shape ``(C,)`` or ``(N, C)`` → latent ``u @ B``)
   * - :class:`~koopman_graph.adaptation.RecursiveKoopmanAdapter`
     - Same as neural: global or per-node rows aligned to latent samples
   * - :class:`~koopman_graph.env.GraphKoopmanEnv`
     - **Global only** ``(C,)`` actions; per-node action spaces are not
       supported
   * - :class:`~koopman_graph.baselines.DMDcBaseline`
     - **Global only**; per-node (3-D) ``control_inputs`` are **rejected**
       at ``fit`` (no silent flatten into a joint control vector)

Shared operator contract
~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~koopman_graph.operators.KoopmanOperator` and
:class:`~koopman_graph.operators.ContinuousKoopmanOperator` share one
:class:`~koopman_graph.operators.Parameterization` Literal and implement
:class:`~koopman_graph.operators.KoopmanOperatorContract`:

* ``matrix`` — assembled ``K`` (discrete) or ``L`` (continuous)
* ``advance`` / ``inverse_advance`` — forward and inverse latent steps
* ``bound_metric`` — cheap soft/structural monitoring bound (diagonal-factor
  bound for ``"odo"``; closed-form certified bound for structural modes;
  equals the true spectrum for ``"dense"``)

Domain-specific names (``K`` / ``L``, ``forward`` / ``inverse_step``,
``spectral_radius`` / ``max_real_part``) remain as thin aliases.
``spectral_radius`` and ``max_real_part`` always report the **true** spectrum
of assembled ``K`` / ``L`` via ``eigvals``. Prefer ``bound_metric`` for ODO
factor monitoring and structural certificates. Discrete ODO still satisfies
``ρ(K) ≤ bound_metric`` via the operator 2-norm; continuous ODO does **not**
guarantee Hurwitz stability from the factor bound alone. See the quickstart
stability section.
``KoopmanPropagator`` in :mod:`koopman_graph.graph_utils` is an alias of the
Protocol and is the single typing surface for losses and adaptation.

Dynamics mode and stability-bound vocabulary
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:data:`~koopman_graph.protocols.DynamicsMode` is the single
``Literal["discrete", "continuous"]`` type, defined next to
:data:`~koopman_graph.operators.Parameterization` in
:mod:`koopman_graph.operators` and re-exported from
:mod:`koopman_graph.protocols`. ``GraphKoopmanModel`` imports it from
protocols; :data:`~koopman_graph.adaptation.AdaptationMode` is an alias of
the same type for the RLS adapter API. Do not redefine the literals in call
sites.

Factory / checkpoint construction uses one neutral stability knob. Domain
operator attributes keep their physics-accurate names; use
:func:`~koopman_graph.operators.resolve_factory_stability_bound` when reading
the bound for serialization or factory reconstruction:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Surface
     - Discrete
     - Continuous
   * - Factory / checkpoint key
     - ``koopman_max_spectral_radius``
     - ``koopman_max_spectral_radius``
   * - Built-in operator attribute
     - ``max_spectral_radius``
     - ``max_real_eigenvalue``

Optional operator injection
~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~koopman_graph.model.GraphKoopmanModel` accepts an optional
``koopman=`` argument satisfying
:class:`~koopman_graph.operators.KoopmanOperatorContract` (must be an
``nn.Module``). Use this to compose a pre-built or custom propagator without
editing the model class. String-mode factory construction remains the default
when ``koopman`` is omitted.

**Enforced capability tier**

* **Protocol-complete** (any contract ``nn.Module``): latent propagation
  (:func:`~koopman_graph.graph_utils.propagate_latent` /
  :func:`~koopman_graph.graph_utils.inverse_propagate_latent` always call
  unified ``advance`` / ``inverse_advance`` with a resolved ``delta_t``;
  discrete operators ignore the interval), encode/decode training and
  ``predict``, :meth:`~koopman_graph.model.GraphKoopmanModel.spectrum` via
  ``koopman.matrix``, and eigenvalue regularization via ``eigvals(matrix)``
  for ``dense`` / ``odo`` and ``bound_metric`` for structural modes, plus the
  model's ``dynamics_mode``. Discrete dense
  backward-consistency training optionally reuses a built-in
  ``dense_inverse_matrix()`` when present; operators without that helper
  still invert via ``inverse_advance`` (``inverse_matrix=None``).
* **Built-in-only**: checkpoint ``save`` / ``load`` round-trips
  :class:`~koopman_graph.operators.KoopmanOperator` and
  :class:`~koopman_graph.operators.ContinuousKoopmanOperator` only; RLS
  :class:`~koopman_graph.adaptation.RecursiveKoopmanAdapter` seed and
  write-back likewise. Custom injected operators raise on save; reconstruct
  with ``koopman=...`` after loading encoder/decoder state separately if
  needed.

Factory and dimension rules:

* Factory kwargs (``koopman_init_mode``, ``koopman_init_scale``,
  ``koopman_parameterization``, ``koopman_max_spectral_radius``) must stay at
  their defaults when injecting.
* ``latent_dim`` / ``control_dim`` must match the injected operator.
* Built-in :class:`~koopman_graph.operators.KoopmanOperator` requires
  ``dynamics_mode="discrete"``;
  :class:`~koopman_graph.operators.ContinuousKoopmanOperator` requires
  ``"continuous"``. For custom operators, set ``dynamics_mode`` to match
  the operator's semantics (spectrum and eigenvalue hinge follow the model
  flag; propagation always supplies ``delta_t``).

.. code-block:: python

   from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
   from koopman_graph.operators import KoopmanOperator

   operator = KoopmanOperator(latent_dim=64, parameterization="odo")
   model = GraphKoopmanModel(
       encoder=encoder,
       decoder=decoder,
       latent_dim=64,
       time_step=0.1,
       koopman=operator,
   )

.. code-block:: python

   from koopman_graph.graph_utils import (
       autoregressive_latent_rollout,
       propagate_latent,
       resolve_graph_inputs,
   )

Optional dependencies and result types
--------------------------------------

Optional extras
~~~~~~~~~~~~~~~

Optional dependencies must **not** break importing ``koopman_graph``. Prefer
**fail-at-call** with an ``ImportError`` that names the install extra or
package:

* **Soft import + call guard (Gymnasium / ``[rl]``)** —
  :mod:`koopman_graph.env` soft-imports ``gymnasium`` so
  :class:`~koopman_graph.env.GraphKoopmanEnv` can subclass ``gym.Env`` when
  present. Construction calls ``_require_gymnasium()`` and raises with
  ``pip install koopman-graph[rl]`` guidance when the extra is missing.
* **Call-site import (e.g. ``h5py`` for METR-LA)** —
  import inside the function that needs the dependency and re-raise
  ``ImportError`` with install guidance (see
  :mod:`koopman_graph.datasets.metr_la`).

Do not fail at import of core modules for optional extras.

Public result types
~~~~~~~~~~~~~~~~~~~

**Value / result objects** (snapshots returned to callers or passed between
pure helpers) should be ``@dataclass(frozen=True)`` with **attribute access**
(examples:
:class:`~koopman_graph.metrics.EvaluationResult`,
:class:`~koopman_graph.operators.StabilityCertificate`,
:class:`~koopman_graph.training.FitHistory`,
:class:`~koopman_graph.adaptation.AdaptationStepResult`,
and the module-local
:class:`~koopman_graph.training.TrainingLossBreakdown`). Prefer optional fields
with ``None`` defaults over mapping/``TypedDict`` styles for new APIs.
Collection-valued fields should be immutable sequences (``tuple``) so callers
cannot mutate nested contents in place.

**Stateful workflow objects** (samplers, optimizers, live training loops,
accumulators that update in place) remain plain / mutable classes. Do not
freeze types whose primary job is mutation across steps.

:class:`~koopman_graph.data.GraphSnapshotSequence` follows the same collection
rule: :attr:`~koopman_graph.data.GraphSnapshotSequence.snapshots` is a
``tuple[Data, ...]`` after construction (no ``.append`` / in-place replace).
Individual ``Data`` objects are **borrowed**, not cloned — in-place mutation of
node features or topology remains possible. Clone explicitly when isolation is
required. :class:`~koopman_graph.data.MultiTrajectory` stores its trajectories
as a frozen tuple of sequences.

``TrainingLossBreakdown`` is a frozen internal snapshot (batch/epoch loss
terms), not a package ``__all__`` export. Epoch aggregation uses a local dict
accumulator, then constructs a new breakdown.

Multi-trajectory ``fit`` input
------------------------------

:meth:`~koopman_graph.model.GraphKoopmanModel.fit` accepts one trajectory or
several trajectories of the same system. Prefer the explicit wrapper:

.. code-block:: python

   from koopman_graph import MultiTrajectory
   from koopman_graph.data import as_multi_trajectory

   model.fit(MultiTrajectory((trajectory_a, trajectory_b)), epochs=50)
   # equivalent helper
   model.fit(as_multi_trajectory(trajectory_a, trajectory_b), epochs=50)

Discrimination rules (used by
:func:`~koopman_graph.training.resolve_training_sequences`):

* :class:`~koopman_graph.data.MultiTrajectory` — multi-trajectory (required)
* :class:`~koopman_graph.data.GraphSnapshotSequence` — single trajectory
* non-empty ``list`` / ``tuple`` of only ``Data`` — single trajectory of
  snapshots
* bare ``list`` / ``tuple`` of :class:`~koopman_graph.data.GraphSnapshotSequence`
  — ``TypeError`` (wrap in ``MultiTrajectory``)
* empty list or mixed ``GraphSnapshotSequence`` / ``Data`` — ``ValueError``

Validation input follows the same rules; a multi-trajectory validation
container must match the training trajectory count, while a single validation
sequence is reused for every training trajectory.

Device and tensor conventions
-----------------------------

There is **no global device manager**. Device placement is local and explicit.

Who moves tensors
~~~~~~~~~~~~~~~~~

* **``GraphKoopmanModel.fit``** validates inputs then delegates to
  :func:`~koopman_graph.training.run_fit_loop`, which resolves the training
  device with :func:`~koopman_graph.training.resolve_device` (explicit
  ``device=`` argument, else the model's parameter device, else CPU), moves
  the model with ``to(device)``, and copies training / validation sequences
  via :func:`~koopman_graph.training.sequence_to_device`.
* **``GraphKoopmanEnv``** takes an explicit ``device`` (or falls back to the
  wrapped model's parameter device) and keeps graph / latent state there for
  stepping.
* **Callers of ``predict`` / ``evaluate``** are responsible for placing the
  model and inputs consistently. These APIs do not silently relocate the model
  the way ``fit`` does; keep inputs on the same device as the model parameters.

Façade vs functional training ownership
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Model façade** — :meth:`~koopman_graph.model.GraphKoopmanModel.fit` owns
  user-facing validation (epoch bounds, control layouts via
  ``_validate_sequence_controls``, multi-trajectory resolve, snapshot-length
  checks) and keeps the public signature / ``FitHistory`` return type.
* **Functional loop** — :func:`~koopman_graph.training.run_fit_loop` owns
  device placement, optimizer / scheduler setup, epoch iteration
  (``train_one_epoch`` / ``train_windowed_epoch`` / ``eval_one_epoch``), early
  stopping, best-weight tracking, optional checkpoint writes, and history
  assembly. Prefer extending that helper over growing the model class.

Online adaptation
~~~~~~~~~~~~~~~~~

:class:`~koopman_graph.adaptation.RecursiveKoopmanAdapter` keeps RLS regression
state on **CPU**. Operator matrices are detached and copied to CPU when the
adapter is constructed from a live operator. :meth:`~koopman_graph.adaptation.RecursiveKoopmanAdapter.apply_to`
copies adapted weights back onto the target operator's device and dtype.

Classical baselines and datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DMD-family baselines (:class:`~koopman_graph.baselines.DMDBaseline`,
:class:`~koopman_graph.baselines.EDMDBaseline`,
:class:`~koopman_graph.baselines.DMDcBaseline`) share
:class:`~koopman_graph.baselines.ClassicalBaseline` for ``time_step`` /
``rank``, fitted metadata, and ``_check_fitted`` scaffolding. They remain
siblings under that ABC (DMDc does not subclass DMD) and satisfy
:class:`~koopman_graph.protocols.ForecastModel` structurally.
:class:`~koopman_graph.baselines.EDMDBaseline` stores a least-squares
``reconstruction_matrix`` (not a GNN decoder).

Dataset / benchmark generators and these baselines default to **CPU**
``float32`` tensors unless a docstring or parameter explicitly says otherwise.
Move results to a training device only when feeding a GPU model.
Sequences with ``is_dynamic_topology=True`` are rejected at ``fit`` (see
Topology capability matrix). Per-node (3-D) controls are rejected by
:class:`~koopman_graph.baselines.DMDcBaseline` (see Control layout capability
matrix); neural and adaptation paths keep per-node row matching.

``ForecastModel`` call-site matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~koopman_graph.protocols.ForecastModel` is a **loose** façade
(``*args`` / ``**kwargs``). Peers share method names, not interchangeable
signatures. Prefer
:class:`~koopman_graph.protocols.UncontrolledForecastModel` (and
:func:`~koopman_graph.protocols.accepts_uncontrolled_data_predict`) when code
must call ``predict(data, steps)`` portably.

.. list-table::
   :header-rows: 1
   :widths: 20 16 24 20 20

   * - Implementer
     - ``predict`` initial
     - ``predict`` extras
     - ``spectrum`` kwargs
     - ``fit`` returns
   * - :class:`~koopman_graph.model.GraphKoopmanModel`
     - ``Tensor`` or ``Data``
     - Optional ``edge_*``, ``controls``, ``future_topologies``; ``controls``
       required when ``control_dim > 0``
     - Optional ``delta_t`` in continuous mode
     - :class:`~koopman_graph.training.FitHistory`
   * - :class:`~koopman_graph.baselines.DMDBaseline`
     - ``Data`` only
     - None (uncontrolled peer)
     - None
     - ``self`` (sklearn chaining)
   * - :class:`~koopman_graph.baselines.EDMDBaseline`
     - ``Data`` only
     - None (uncontrolled peer)
     - None
     - ``self`` (sklearn chaining)
   * - :class:`~koopman_graph.baselines.DMDcBaseline`
     - ``Data`` only
     - **Required** ``controls`` (global ``(C,)`` only; not an uncontrolled
       peer)
     - None
     - ``self`` (sklearn chaining)

**``fit`` return divergence.** Classical baselines return ``self`` for chaining
(``baseline.fit(seq).predict(...)``).
:meth:`~koopman_graph.model.GraphKoopmanModel.fit` returns a frozen
:class:`~koopman_graph.training.FitHistory` with per-epoch losses and early-stop
metadata. Do not assume a portable ``fit`` return type across
:class:`~koopman_graph.protocols.ForecastModel` peers; this is intentional for
v0.3.0 (document rather than unify).

Spectral similarity vs mode shapes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`~koopman_graph.analysis.dynamical_similarity` compares **spectra**, not
concrete model classes. Documented call patterns:

* Precomputed spectra — ``dynamical_similarity(spectrum_a, spectrum_b)``
* Classical baselines — ``dynamical_similarity(dmd_a, dmd_b)``
* Mixed peers — ``dynamical_similarity(dmd, neural_model)``
* Continuous neural horizon — ``dynamical_similarity(model_a, model_b, delta_t=0.1)``
  (``delta_t`` is forwarded only when ``spectrum`` accepts it; ignored for
  precomputed :class:`~koopman_graph.spectrum_types.KoopmanSpectrum` values and
  classical baselines)

:func:`~koopman_graph.analysis.resolve_spectrum` is the shared resolver.
:func:`~koopman_graph.analysis.decode_mode_shapes` stays hard-typed to
:class:`~koopman_graph.model.GraphKoopmanModel` because it needs encode /
decode and a GNN decoder; do not widen it to :class:`~koopman_graph.protocols.SpectrumProvider`.

Interchangeable under the uncontrolled peer contract:
:class:`~koopman_graph.baselines.DMDBaseline`,
:class:`~koopman_graph.baselines.EDMDBaseline`, and
:class:`~koopman_graph.model.GraphKoopmanModel` when called as
``predict(data, steps)``.

Dataset factory idioms
~~~~~~~~~~~~~~~~~~~~~~

:mod:`koopman_graph.datasets` uses two complementary factory styles:

* **Simulated dynamics** — ``Benchmark.generate(...)`` synthesizes a
  :class:`~koopman_graph.data.GraphSnapshotSequence` (synthetic path/ring,
  grid, IEEE 118 voltage/load diffusion). Prefer these classmethods in
  tutorials and application code.
* **Real telemetry** — ``Benchmark.load_topology`` / ``load_sequence`` (METR-LA)
  read cached downloaded artifacts. There is no ``generate`` because the
  time series is observed, not simulated.

**Seed defaults.** All simulated ``generate`` methods default ``seed=None``
(unseeded RNG). Pass an explicit integer for reproducible runs; tutorial
notebooks and quickstart examples use ``seed=42``.

**Topology payloads.** IEEE 118 and METR-LA ``load_topology`` return a frozen
:class:`~koopman_graph.datasets.TopologyPayload` (attribute access preferred).
Mapping-style access (``payload["edge_index"]``) remains supported for existing
notebooks. On-disk caches stay plain ``dict``; the typed payload is the public
Python return type.

IEEE 118 also exposes a module-level ``load_topology`` free function used by
download scripts; prefer ``IEEE118DynamicBenchmark.load_topology``.

**Generation validators.** Shared physical parameters for Laplacian-diffusion
benchmarks (synthetic, grid, IEEE 118) route through
:func:`~koopman_graph.datasets.dynamics.validate_diffusion_generation_params`
(``diffusion_rate ∈ [0, 1]``, ``decay_rate > 0``, ``noise_std ≥ 0``, optional
``initial_state``). Anisotropic advection uses the same helper for noise /
initial state, but its self-retention factor is validated separately by
:func:`~koopman_graph.datasets.dynamics.validate_advection_decay_rate`
(``decay_rate ∈ (0, 1)``) — same parameter name, stricter open interval.
Benchmark-specific knobs (load ramp, grid size, neighbor weights) stay local.

Laplacian helpers: sparse vs dense
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The symmetric normalized Laplacian ``L_sym = I - D^{-1/2} A D^{-1/2}`` has one
shared weight core in :mod:`koopman_graph.graph_utils`
(:func:`~koopman_graph.graph_utils.symmetric_normalized_adjacency_edge_weights`):

* **Sparse** — :func:`~koopman_graph.observables.graph_laplacian_features`
  applies ``L_sym @ x`` via matvec for hybrid physics lifting during training.
* **Dense** — :mod:`koopman_graph.datasets.dynamics` builds
  ``I - alpha * L_sym`` step operators for offline benchmark rollouts.

Keep both surfaces: they are numerically aligned on the same ``L_sym`` but
serve different performance and API roles.

Hybrid physics checkpoint schema
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When hybrid observables are enabled, checkpoint ``config.physics`` owns
``dim``, ``preset``, and ``position``. :func:`~koopman_graph.serialization.build_model_config`
writes ``position`` from ``model.physics_position``;
:func:`~koopman_graph.serialization.reconstruct_model` reads and validates it via
:func:`~koopman_graph.observables.resolve_physics_position` (missing ``position``
defaults to ``"prepend"``). Only ``"prepend"`` is supported today — unsupported
values raise on load so a future ``"append"`` mode cannot silently mis-restore.
No ``FORMAT_VERSION`` bump was required: the field was already emitted in v2.

v0.3.0 architectural consistency outcomes
-----------------------------------------

Phase 8 standardized style and release quality without bumping the package
beyond ``0.3.0`` or ``FORMAT_VERSION`` 2. Outcomes folded into the first public
v0.3.0 cut:

**Highest-impact API remediations (second style audit)**

1. **Shared rollout** — ``predict`` / training losses / the RL env share
   :func:`~koopman_graph.graph_utils.autoregressive_latent_rollout` (and
   one-step ``advance_and_decode``), with hold-last vs teacher-target topology
   policies documented above.
2. **``ForecastModel`` call sites** — loose Protocol for method presence;
   :class:`~koopman_graph.protocols.UncontrolledForecastModel` for portable
   ``predict(data, steps)`` among autonomous peers (see call-site matrix).
3. **Optional operator injection** — ``koopman=`` composes a custom
   :class:`~koopman_graph.operators.KoopmanOperatorContract` ``nn.Module``
   with Protocol-complete propagate / spectrum / eigenvalue paths; RLS and
   checkpoints remain built-in-only (see Optional operator injection).
4. **Classical baseline scaffolding** —
   :class:`~koopman_graph.baselines.ClassicalBaseline` ABC; EDMD stores
   ``reconstruction_matrix`` (not a GNN ``decoder``).
5. **Frozen public result types** — ``FitHistory``, ``AdaptationStepResult``,
   and related dataclasses use ``frozen=True`` with tuple series;
   ``TrainingLossBreakdown`` is a frozen internal snapshot (not package-exported).

**Contract and boundary follow-ups**

* Honest continuous/discrete dispatch for injected operators; sequence
  immutability; topology and control-layout capability guards; shared GNN
  primitives; thin ``fit`` orchestration; spectrum-based dynamical similarity;
  shared dataset validators; centralized dynamics/stability vocabulary;
  hybrid ``physics.position`` round-trip; loss-breakdown hygiene.

**Tutorial claim↔result integrity**

* Notebooks 02 / 03 / 06 / 14 / 15 remediations so scientific claims match
  stored outputs (history-constrained Vm scope, METR-LA baseline protocol,
  epidemic spectrum wording, physics-informed evidence scope, RL takeaway
  metrics).

**Capability packages and thin façade**

* Layout policy plus ``training/``, ``data/``, ``operators/``, ``nn/``,
  ``analysis/``, and ``baselines/`` capability packages; deep-import shim
  hard cut; thin root ``__all__`` with metrics / analysis / data /
  adaptation / observables secondaries demoted to capability imports.

**Analysis UX and release quality**

* :func:`~koopman_graph.analysis.plot_spectrum` with ``limits="unit_disk"`` /
  ``"data"`` (capability-module import only).
* Enforced pytest coverage gate of **90%** (``fail_under`` / CI
  ``--cov-fail-under``); branch-aware suite remains well above the floor.

Related documentation
---------------------

* :doc:`api` — module-level API reference
* :doc:`quickstart` — end-user training and prediction walkthrough
* Repository ``CONTRIBUTING.md`` — development setup and agent workflow links
