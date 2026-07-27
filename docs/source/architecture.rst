Architecture and API layers
===========================

This page is the maintainer-facing contract for how KoopmanGraph is structured
(``ARCH-060``). It exists so contributors do not reintroduce style drift across
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
   ``koopman_parameterization="dense"|"structural"|"odo"|"auxiliary_spectral"``) rather than parallel
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

**Not nesting triggers by themselves**:

* **Public API docstring bulk** — numpy-style docstrings on sklearn-façade
  methods and operator-contract surfaces can push a module past ~1k lines
  without adding a second cohesive concern. Do **not** nest (or strip
  user-facing docs) solely to reduce line count.
* **Thin orchestrator dispatch wrappers** — same-module
  ``_assemble_*`` / ``_reset_*`` (and similar) one-liner forwards to peer
  helpers are acceptable string-mode orchestrator glue. They are **not**
  alone sufficient reason to invent another nesting wave.

Nest only when the bullets above under “When to nest” hold for *maintainable
surface* (multi-concern grab-bag, peer implementations, or room to grow
peers). Raw LOC and docstring volume are diagnostic signals, not contracts.

Capability packages include ``training/``, ``data/``, ``operators/``,
``nn/``, ``analysis/``, and ``baselines/``.
``adaptation/`` is also a capability package (RLS + Kalman observer).
``graph_utils/`` is a capability package (topology/Laplacian peers +
latent propagation/rollout peers).
``losses/`` is a capability package (consistency / regularization /
reconstruction / physics / rollout peers).
``model/`` is a same-named capability package (estimator / factory /
validation / timing / encoding / inference / online_adaptation peers).

``koopman_graph.adaptation`` package layout:

* ``rls`` — ``RecursiveKoopmanAdapter``, ``AdaptationStepResult``
* ``kalman`` — ``FilterResult``, ``reference_kalman_filter``, ``rts_smooth``
* ``impute`` — ``graph_diffuse_impute`` heuristic warm-start
* ``observer`` — ``KoopmanObserver`` façade (filter / smooth / impute)

``koopman_graph.graph_utils`` package layout (topology vs propagation peers):

* ``topology`` — degree / adjacency / Laplacian helpers
  (``node_degrees``, ``degree_support_mask``,
  ``symmetric_normalized_adjacency_edge_weights``, dense and sparse
  ``Â`` / ``L_sym`` surfaces, plus row-walk
  ``dense_random_walk_normalized_adjacency`` /
  ``random_walk_normalized_adjacency_matvec`` for directed
  ``adjacency="random_walk"|"dual_random_walk"``), plus graph-input
  resolution (``snapshot_edge_weight``, ``resolve_edge_index``,
  ``resolve_edge_weight``, ``resolve_graph_inputs``,
  ``snapshot_to_device``). Row-walk helpers live here (neutral leaf), not
  in ``nn.gnn``.
* ``propagation`` — ``KoopmanPropagator`` alias, rollout callback types,
  ``resolve_delta_t``, ``propagate_latent`` /
  ``inverse_propagate_latent``, ``advance_and_decode``,
  ``hold_last_topology_at``, ``snapshot_topology_at``,
  ``autoregressive_latent_rollout``, ``pack_rollout_snapshots``
* ``symmetry`` — node-orbit partitions for symmetry-adapted operators
  (``node_orbit_partition``, ``validate_orbit_partition``,
  ``identity_orbit_partition``; requires the ``[symmetry]`` extra for
  non-identity auto-orbits)

Prefer ``from koopman_graph.graph_utils import …``. Do not import
leading-underscore helpers across peer modules.

``koopman_graph.training`` package layout:

* ``history`` — ``LossWeights``, ``FitHistory``, fit input type aliases;
  internal ``TrainingLossBreakdown`` (not package-exported)
* ``schedules`` — constant / ramp loss-weight schedules
* ``device`` — ``resolve_device``, ``sequence_to_device``
* ``pair_objectives`` — one-step / pair reconstruction and
  forward/backward consistency composition over
  :mod:`koopman_graph.losses` (re-exports ``encode_at_timestep`` for
  stable deep imports)
* ``timestep_encode`` — private ``encode_at_timestep`` leaf shared by
  pair losses and the latent cache (not package-exported; keeps
  ``latent_cache`` and ``pair_objectives`` acyclic)
* ``latent_cache`` — private ``SequenceLatentCache`` /
  ``encode_sequence_latents`` helpers (not package-exported)
* ``extra_objectives`` — Lie / PDE / sparsity / worst-case composition
  helpers over ``ExtraLosses`` / ``LossWeights``
* ``objectives`` — thin ``compute_training_loss`` orchestrator plus
  eigenvalue / rollout composition (re-exports pair helpers for stable
  deep imports; prefer package ``__init__`` re-exports)
* ``epochs`` — ``train_one_epoch``, ``train_windowed_epoch``,
  ``eval_one_epoch``
* ``inputs`` — multi-trajectory classification plus
  ``resolve_training_sequences`` / ``resolve_validation_sequences``
* ``loop`` — ``run_fit_loop`` plus early-stopping / LR-scheduler helpers
  (re-exports epoch helpers so fit-loop monkeypatches of
  ``training.loop.train_one_epoch`` remain valid). Also hosts
  ``bind_pending_orbit_ties``, a deep-import fit-start hook that binds
  ``koopman_auto_orbits`` from the first training snapshot **before** the
  optimizer is constructed so orbit ``K_self`` parameters are trained.
  Prefer calling it only via ``run_fit_loop``; it is not a
  :mod:`koopman_graph.training` package ``__all__`` export.

Do not invent a ``training/objectives/`` or ``training/loop/`` subtree.
Prefer ``from koopman_graph.training import …``. Do not import
leading-underscore helpers across training peers.

Training-path contracts (performance)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These contracts describe evaluation-scoped reuse. They do not change
public scientific defaults or checkpoint payloads.

* **Shared latents.** When ``compute_training_loss`` evaluates multiple
  pair terms on one sequence, a ``SequenceLatentCache`` encodes each
  timestep once per evaluation. Stochastic encoders (e.g. dropout)
  therefore sample once per timestep rather than once per pair endpoint
  call. Autoregressive rollout still advances from the origin; it does
  not replace future latents with teacher-forced cache entries.
* **Ephemeral operator / support caches.** Dense networked inverses,
  DiffConv diffusion supports, hypergraph Zhou :math:`\hat{H}`,
  continuous :math:`\Phi` / :math:`L_{\mathrm{eff}}`, and structural /
  low-rank bilinear assemblies may be reused when topology / incidence /
  factor keys match. Caches are module- or evaluation-scoped, keyed
  (often by storage pointers), and **not** written to ``state_dict``.
  In-place mutations that keep the same ``data_ptr`` require the matching
  clear API (``clear_support_cache``, ``clear_hyperedge_cache``, or the
  next ``compute_training_loss`` evaluation for continuous
  :math:`\Phi`).
* **Shared one-step predictions.** Reconstruction, PDE residual, and
  worst-case terms that need the same one-step decode share a single
  prediction pass when composed through ``compute_training_loss``.
* **AMP.** ``GraphKoopmanModel.fit`` and ``run_fit_loop`` accept
  ``use_amp`` / ``amp_dtype``. Mixed precision is CUDA-only; non-CUDA
  devices warn once and continue in FP32. Optimizer steps use
  ``zero_grad(set_to_none=True)``.
* **Hierarchical pool schedule.**
  ``HierarchicalGraphKoopmanModel(pool_schedule=...)`` defaults to
  ``"per_snapshot"`` (feature-dependent scores every timestep).
  ``"hold_perm"`` pools once from the first snapshot and reuses that
  permutation for the sequence.
* **Learned topology / multi-start.** Self-adaptive
  ``materialize_learned_topology`` runs at most once per top-level
  ``forward`` or ``encode_rollout_origin`` (``encode`` accepts
  ``resolve_learned_topology=False`` when edges are already resolved).
  ``rollout_multi_start_loss`` encodes each distinct origin once when no
  sequence cache is supplied.

``koopman_graph.losses`` package layout (peer training objectives):

* ``consistency`` — ``ForwardConsistencyLoss``, ``BackwardConsistencyLoss``,
  ``LieConsistencyLoss``
* ``regularization`` — ``EigenvalueRegularizationLoss``,
  ``KoopmanSparsityLoss``
* ``reconstruction`` — ``masked_mse_loss``, ``WorstCaseReconstructionLoss``
* ``physics`` — ``PDEResidualLoss``
* ``rollout`` — ``rollout_sequence_loss``, ``rollout_multi_start_loss``

Prefer ``from koopman_graph.losses import …``. Do not import
leading-underscore helpers across peer modules. Root façade exports remain
the three consistency / eigenvalue symbols listed under Public façade;
power-user losses stay package imports outside root ``__all__``.

``koopman_graph.data`` package layout (kept separate from
:mod:`koopman_graph.datasets`):

* ``containers`` — ``GraphSnapshotSequence``
* ``construction`` — array / dynamic-array / Hankel-window builders
  (``build_snapshots_from_arrays``, ``build_snapshots_from_dynamic_arrays``,
  ``build_windowed_snapshots``, ``ConstructedSnapshots``) used by container
  classmethods. Prefer constructing sequences via the container APIs; these
  are power-user helpers for callers that build snapshots outside that path.
  Not promoted on the package or root ``__all__``.
* ``validation`` — coercion and validation helpers for snapshot construction
  (``as_tensor``, timestamp / mask / control / topology validators). Prefer
  constructing sequences via the container APIs; these are power-user helpers
  for callers that validate outside that path. Do not import leading-``_``
  symbols across ``data/`` peers.
* ``trajectories`` — ``MultiTrajectory``, ``as_multi_trajectory``,
  ``resolve_sequence``, ``resolve_pair_delta_t``
* ``delay_windows`` — non-private Hankel-style delay helpers
  (``stack_delay_features``, ``flatten_delay_window``,
  ``history_from_snapshots``, ``apply_observation_mask_to_features``).
  Owned here so feature modules (``nn.delay``, model encoding, adaptation)
  import **down** into ``data``. :mod:`koopman_graph.data` must **not**
  import :mod:`koopman_graph.nn` (eager or lazy). Power-user submodule; not
  promoted on the package or root ``__all__``.
* ``sampling`` — ``WindowSampler`` (full-graph temporal windows);
  ``NeighborWindowSampler`` (k-hop subgraph windows for large static graphs;
  training approximation only — package export, not root ``__all__``)
* ``splits`` — ``TemporalSplit``, ``temporal_split``
* ``rollout`` — ``resolve_rollout_start_indices`` and related type aliases
* ``sequence_types`` — internal ``SnapshotSequenceLike`` Protocol used by
  ``construction`` / ``delay_windows`` so those peers stay acyclic with
  ``containers``. Not package-exported; do not promote to
  :mod:`koopman_graph.data` ``__all__``.

``koopman_graph.operators`` package layout (peer discrete/continuous/networked):

* ``contract`` — ``KoopmanOperatorContract``, ``Parameterization``,
  ``DynamicsMode``, ``KoopmanKind``, ``StabilityCertificate``, and the
  documented **non-private** structural helpers
  ``bounded_diagonal``, ``strict_diagonal_values``, and
  ``safe_diagonal_inverse`` (plus ``cayley_orthogonal`` /
  ``strict_spectral_bound`` / ``build_stability_certificate``). Discrete /
  continuous / graph peers must use
  these names rather than importing leading-``_`` symbols from ``contract``.
* ``control`` — shared additive / bilinear control helpers
  (``ControlMode``, bilinear factor allocation/assembly,
  ``broadcast_control_term`` for global control offsets,
  ``map_control_term`` for ``u @ B`` latent offsets,
  ``write_dense_operator_parameters`` for dense ``K``/``L`` + ``B``/``N``
  write-back, ``effective_bilinear_matrix`` /
  ``per_node_effective_bilinear_matrices`` for inverse self-term
  assembly). Discrete, continuous,
  and graph operators share this non-private surface; do not reach into peer
  private members for control broadcasting or related initialization.
* ``discrete`` — :class:`~koopman_graph.operators.discrete.KoopmanOperator`
  thin string-mode orchestrator for dense / ODO / Schur / dissipative /
  Lyapunov / controlled advance. Re-exports discrete identity-init helpers
  from ``discrete_parameterizations``.
* ``discrete_parameterizations`` — non-private ODO / Schur / dissipative /
  Lyapunov / dense assembly and reset helpers used by the discrete
  orchestrator. Preserves string-mode ``parameterization=...``; no
  parallel parameterization class hierarchy.
* ``discrete_propagation`` — non-private controlled / uncontrolled advance,
  additive-control offset removal, bilinear inverse, and inverse-matrix
  execution helpers used by the discrete orchestrator. Assembly / reset
  stay in ``discrete_parameterizations``; do not nest under
  ``operators/discrete/``.
* ``continuous`` — :class:`~koopman_graph.operators.continuous.ContinuousKoopmanOperator`
  thin string-mode orchestrator for dense / ODO / Schur / dissipative /
  Lyapunov / controlled advance, plus re-exports of continuous-surface Van
  Loan helpers. Thin dispatcher for
  ``parameterization="auxiliary_spectral"`` (delegates construction,
  advance, inverse, reset, and Hurwitz bound / certificate computation to
  ``auxiliary_spectral`` / ``continuous_propagation`` /
  ``continuous_parameterizations``).
* ``continuous_van_loan`` — cohesive matrix-log / Van Loan factor
  construction (``matrix_log``, ``van_loan_factors``,
  ``van_loan_generator_from_discrete``). Continuous-surface owned via
  ``continuous`` / package re-exports; do not nest under
  ``operators/continuous/``.
* ``continuous_parameterizations`` — non-private ODO / Schur / dissipative /
  Lyapunov / dense assembly and reset helpers, plus continuous Hurwitz
  ``continuous_bound_metric`` / ``max_real_part_of_generator`` /
  ``continuous_stability_certificate`` helpers used by the continuous
  orchestrator. Preserves string-mode ``parameterization=...``; no
  parallel parameterization class hierarchy.
* ``continuous_propagation`` — non-private controlled / uncontrolled
  advance, Van Loan integral application, interval advance / inverse
  orchestration (``advance_interval`` / ``inverse_advance_interval``), and
  bilinear inverse execution helpers used by the continuous orchestrator.
  Factor construction stays in ``continuous_van_loan``; do not nest under
  ``operators/continuous/``.
* ``auxiliary_spectral`` — Lusch-style state-dependent block-diagonal
  generator MLP (``AuxiliarySpectralNetwork``) plus cohesive
  uncontrolled/controlled advance, approximate inverse, and reset helpers
  for continuous ``parameterization="auxiliary_spectral"``. Does not host
  Van Loan factor construction (that lives in ``continuous_van_loan`` and
  remains continuous-surface owned).
* ``graph`` — :class:`~koopman_graph.operators.graph.GraphKoopmanOperator`
  (discrete networked self/neighbor coupling; select via ``koopman="graph"``)
* ``graph_inverse`` — block-diagonal / Jacobi approximate inverse helpers for
  ``GraphKoopmanOperator`` / ``HypergraphKoopmanOperator`` with
  ``sparsity="block_diagonal"`` (no ``operators/graph/`` subtree)
* ``continuous_graph`` —
  :class:`~koopman_graph.operators.continuous_graph.ContinuousGraphKoopmanOperator`
  (continuous networked generator ``I⊗L_self + Â⊗L_nbr``; select via
  ``koopman="graph"`` + ``dynamics_mode="continuous"`` or
  ``koopman="continuous_graph"``). Dense advance uses an ``N·d`` matrix
  exponential; ``sparsity="block_diagonal"`` is the self-dominated
  approximate path
* ``hypergraph`` —
  :class:`~koopman_graph.operators.hypergraph.HypergraphKoopmanOperator`
  (discrete hyperedge-coupled advance; select via ``koopman="hypergraph"``;
  ``sparsity="block_diagonal"`` shares the Jacobi approximate inverse path)
* ``global_local`` —
  :class:`~koopman_graph.operators.global_local.GlobalLocalKoopmanOperator`
  (discrete global backbone ``K_g`` plus low-rank local window correction
  ``K_ℓ``; select via ``koopman="global_local"``)
* ``orbit_ties`` — shared ``OrbitTiedSelfMixin`` and
  ``build_orbit_self_bank`` helpers for discrete ``koopman="graph"`` /
  ``"hypergraph"`` orbit-tied ``K_self`` factors (two hosts only; not a
  deep inheritance tree). Symmetry discovery stays in
  :mod:`koopman_graph.graph_utils.symmetry`; fit-start auto-orbit binding
  is orchestrated by ``training.loop.bind_pending_orbit_ties``.

Prefer ``from koopman_graph.operators import …`` (or the root façade for public
operator classes). Former root modules ``koopman_graph.operator`` and
``koopman_graph.continuous`` were removed in v0.3.0.

``koopman_graph.nn`` package layout (peer encoder / decoder / GNN primitives;
PyG-style ``nn`` capability package, no ``conv/`` subtree):

* ``gnn`` — :class:`~koopman_graph.nn.gnn.BaseGNNModule`, activation typing,
  validators, GCN/GAT/SAGE/DiffConv/Transformer/Hypergraph convolution
  builders (power-user)
* ``encoder`` — :class:`~koopman_graph.nn.encoder.GNNEncoder` /
  :class:`~koopman_graph.nn.encoder.GATEncoder` /
  :class:`~koopman_graph.nn.encoder.SAGEEncoder` /
  :class:`~koopman_graph.nn.encoder.DiffConvEncoder` /
  :class:`~koopman_graph.nn.encoder.GraphTransformerEncoder`
* ``decoder`` — :class:`~koopman_graph.nn.decoder.GNNDecoder` /
  :class:`~koopman_graph.nn.decoder.GATDecoder` /
  :class:`~koopman_graph.nn.decoder.SAGEDecoder` /
  :class:`~koopman_graph.nn.decoder.DiffConvDecoder` /
  :class:`~koopman_graph.nn.decoder.GraphTransformerDecoder`
* ``hypergraph`` — :class:`~koopman_graph.nn.hypergraph.HypergraphEncoder` /
  :class:`~koopman_graph.nn.hypergraph.HypergraphDecoder`
  (PyG ``HypergraphConv``; static ``hyperedge_index`` / ``hyperedge_weight``)
* ``delay`` — :class:`~koopman_graph.nn.delay.DelayEmbeddingEncoder`
  delay-coordinate wrapper (compose with a base encoder sized as
  ``in_channels = n_delays * feature_dim``; optional
  ``GraphKoopmanModel(n_delays=...)`` wraps without rebuilding layers).
  Pure snapshot/tensor window helpers
  (``stack_delay_features``, ``flatten_delay_window``,
  ``history_from_snapshots``, ``apply_observation_mask_to_features``) are
  owned by :mod:`koopman_graph.data.delay_windows` and thin-re-exported
  here for stable ``nn.delay`` import paths.
* ``adaptive_topology`` —
  :class:`~koopman_graph.nn.adaptive_topology.AdaptiveAdjacency`
  (Graph WaveNet self-adaptive adjacency ``softmax(ReLU(E₁E₂ᵀ))``; power-user
  ``koopman_graph.nn`` export). Enable via
  ``GraphKoopmanModel(learn_topology="self_adaptive",
  topology_embedding_dim=k)`` (default ``k=8``). When enabled, the learned
  dense COO adjacency **replaces** pairwise ``edge_index`` / ``edge_weight``
  for encode, decode, networked graph / continuous-graph advance, and
  spectrum (materialize the same ``Â``; ``N²`` edges — prefer modest ``N``).
  Hyperedge incidence for hypergraph encode / operators stays exogenous.
  Default ``learn_topology=None`` is a numerical no-op. Static per fit (node
  count binds on first use). Framing: forecasting inductive bias, **not**
  causal structure discovery.

Prefer ``from koopman_graph.nn import …`` (or the root façade for public
classes). Former root modules ``koopman_graph.encoder``,
``koopman_graph.decoder``, and ``koopman_graph.gnn`` were removed in v0.3.0.
Encoder and decoder remain peers: both import from ``nn.gnn``;
neither imports the other.

``koopman_graph.analysis`` package layout (spectrum / similarity / anomaly /
plotting / residuals / topology estimation / SINDy / clustering):

* ``spectrum`` — re-exports neutral-leaf ``compute_spectrum``,
  ``compute_generator_spectrum``, and ``discrete_spectrum_at_delta_t``, plus
  analysis-owned ``decode_mode_shapes``
* ``similarity`` — ``spectrum_distance``, ``koopman_std``, ``resolve_spectrum``,
  ``dynamical_similarity``
* ``anomaly`` — ``AnomalyDetectionResult``, ``calibrate_anomaly_threshold``,
  ``detect_anomaly``
* ``plotting`` — ``plot_spectrum`` (discrete unit-disk / data-zoom complex-plane
  figures; Matplotlib call-site import; optional untrustworthy-mode overlay)
* ``residuals`` —
  :func:`~koopman_graph.analysis.spectral_residuals` and
  :class:`~koopman_graph.analysis.SpectralResidualReport` (held-out
  data-driven residuals / ``trustworthy_mask``; diagnostic, not ResDMD)
* ``topology_estimation`` —
  :func:`~koopman_graph.analysis.estimate_coupling_from_snapshots` and
  :class:`~koopman_graph.analysis.CouplingEstimate` (flattened DMD → ``N×N``
  coupling magnitudes; diagnostics / warm-start aid — **not** causal
  structure discovery). Rejects dynamic-topology and hyperedge sequences
  like classical baselines.
* ``sindy`` —
  :func:`~koopman_graph.analysis.identify_sparse_dynamics` and frozen
  :class:`~koopman_graph.analysis.SINDyReport` (STLSQ on polynomial /
  graph-neighbor libraries over encoded per-node latents; discrete or
  finite-difference derivative mode). Pure analysis — identifies the
  **learned latent** dynamics, not physical ground-truth ODEs.
* ``clustering`` —
  :func:`~koopman_graph.analysis.koopman_spectral_clustering` and frozen
  :class:`~koopman_graph.analysis.ClusteringResult` (seeded k-means on
  leading Koopman mode embeddings; sign/ordering canonicalized by
  magnitude-sorted ``|λ|`` and largest-magnitude positive pivot). Cluster
  quality inherits operator / spectrum quality.

:mod:`koopman_graph.spectrum_types` remains a **top-level neutral leaf** so
:mod:`koopman_graph.protocols`, :mod:`koopman_graph.operators`, and
:mod:`koopman_graph.model` never import heavy analysis code for spectrum
assembly. It hosts :class:`~koopman_graph.spectrum_types.KoopmanSpectrum`,
discrete :func:`~koopman_graph.spectrum_types.compute_spectrum`, continuous
:func:`~koopman_graph.spectrum_types.compute_generator_spectrum`, and
:func:`~koopman_graph.spectrum_types.discrete_spectrum_at_delta_t`.
These are re-exported from :mod:`koopman_graph.analysis` (and
``KoopmanSpectrum`` / ``compute_spectrum`` from the package root) for the
public API.

``koopman_graph.baselines`` package layout (peer DMD-family methods behind
:class:`~koopman_graph.baselines.ClassicalBaseline`, plus GNN forecasters):

* ``base`` — :class:`~koopman_graph.baselines.ClassicalBaseline` and shared
  **non-private** power-user helpers used by classical and GNN peers:
  ``require_static_topology``, ``flatten_snapshots``, ``fit_row_operator``,
  ``fit_controlled_row_operator``, ``require_global_controls``,
  ``transition_controls``, ``copy_topology``, ``check_initial_graph``,
  ``optimal_hard_threshold_rank`` / ``gavish_donoho_omega`` /
  ``resolve_fit_rank`` (``rank="auto"``). Peers must import these names
  (not leading-``_`` aliases). Helpers stay off package and root ``__all__``.
* ``dmd`` — :class:`~koopman_graph.baselines.DMDBaseline`
* ``dmdc`` — :class:`~koopman_graph.baselines.DMDcBaseline`
* ``edmd`` — :class:`~koopman_graph.baselines.EDMDBaseline`
* ``gnn`` — spatiotemporal GNN forecaster baselines
  (:class:`~koopman_graph.baselines.gnn.STGCNBaseline`,
  :class:`~koopman_graph.baselines.gnn.DCRNNBaseline`,
  :class:`~koopman_graph.baselines.gnn.GraphWaveNetBaseline`);
  lightweight in-repo references for ``evaluate_forecast`` comparisons;
  ``spectrum()`` raises ``RuntimeError``

Deep imports continue to use ``from koopman_graph.training import …``,
``from koopman_graph.data import …``, ``from koopman_graph.analysis import …``,
and ``from koopman_graph.baselines import …`` via package ``__init__``
re-exports (same-named package compatibility).

When to stay flat
~~~~~~~~~~~~~~~~~

Keep a single module (or leave an existing small package alone) when:

* The module is **small and single-purpose** (for example ``metrics``,
  ``statistics``, ``env``, ``observables``, ``serialization``,
  ``protocols``, ``spectrum_types``).
* :mod:`koopman_graph.datasets` is already the correct benchmark/load
  subpackage — do **not** merge it into ``data/`` (data structures vs
  datasets, same distinction as sklearn/PyG).
* Nesting would only rename a path without clarifying capability boundaries.
* The file is large mainly because of **documented public API** or **thin
  same-module dispatch wrappers** (see “Not nesting triggers by themselves”
  under When to nest) rather than mixed cohesive concerns.

Explicit non-goals unless a later audit shows multi-peer growth: nesting
``env``, ``observables``, ``metrics``, ``serialization``, or
``protocols``.

:mod:`koopman_graph.adaptation` is a capability package because RLS adaptation
and the Kalman observer are peer concerns
(``adaptation/rls.py``, ``adaptation/observer.py``). Generic Kalman / RTS
numerics and graph-diffusion imputation live in shallow
peers (``adaptation/kalman.py``, ``adaptation/impute.py``) so the observer
façade remains a thin Koopman-specific orchestrator.

:mod:`koopman_graph.graph_utils` is a capability package because
topology/Laplacian mathematics and latent propagation/rollout are cohesive
peer concerns (``graph_utils/topology.py``, ``graph_utils/propagation.py``).

``model`` stays at the package root
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~koopman_graph.model.GraphKoopmanModel` and :mod:`koopman_graph.model`
remain top-level. Do **not** bury the primary workflow class under ``nn/``,
``training/``, or similar. The same-named ``koopman_graph.model`` *package*
preserves ``koopman_graph.model.GraphKoopmanModel`` while housing cohesive
peer helpers:

* ``estimator`` — :class:`~koopman_graph.model.GraphKoopmanModel`
* ``factory`` — operator construction / injection validation and model
  component assembly (``resolve_model_components``,
  ``apply_resolved_components``)
* ``validation`` — control and sequence validation helpers, including
  ``prepare_fit_inputs`` for the ``fit`` preamble
* ``timing`` — timestamp / increment policy helpers
* ``encoding`` — physics / delay / encode-origin helpers
* ``inference`` — spectrum / predict / evaluate orchestration helpers
* ``online_adaptation`` — online RLS façade-bridge helpers (does **not**
  relocate :class:`~koopman_graph.adaptation.RecursiveKoopmanAdapter` /
  :class:`~koopman_graph.adaptation.KoopmanObserver`)

Prefer ``from koopman_graph.model import GraphKoopmanModel`` (or the root
façade). Do not import leading-underscore helpers across peer modules.

No deep trees
~~~~~~~~~~~~~

One nesting level is enough (``koopman_graph.nn``, ``koopman_graph.operators``,
``koopman_graph.graph_utils``). Do **not** add deep trees such as
``nn/conv/...`` or ``operators/discrete/...`` unless a future audit shows a
genuine multi-layer library surface. Prefer peer modules inside the
capability package.

**Deliberate exception:** ``koopman_graph.baselines.gnn`` is a second-level
subtree that separates classical DMD-family baselines from spatiotemporal GNN
forecaster references. Treat it as a classical-vs-neural baseline boundary,
not as precedent for arbitrary deep trees elsewhere. Base nesting decisions
on cohesive peer growth rather than line count alone; public docstring bulk
and thin orchestrator dispatch wrappers are not alone sufficient nesting
triggers (see When to nest).

Three-layer API regardless of folders
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Moving files must **not** change the layering contract above
(``ARCH-020`` — public subpackages expose a curated ``__all__``):

1. **Public façade** — symbols in package ``__all__`` / ``from koopman_graph
   import …``
2. **Power-user modules** — importable packages or modules documented here but
   omitted from ``__all__``
3. **Private helpers** — leading-``_`` names, same-module only

Folders are an organization tool; they do not create a fourth API tier.
Internal peer modules may deep-import leaf modules (for example
``operators.contract`` or ``graph_utils.topology``) to keep the import graph
acyclic; external callers should still prefer package ``__init__`` re-exports.

Compatibility contract
~~~~~~~~~~~~~~~~~~~~~~

When splitting or nesting modules:

* **Root façade is stable by default** — ``from koopman_graph import …``
  symbols and package ``__all__`` keep the same public names and semantics
  across layout refactors. Explicit versioned breaking migrations may rewrite
  the root export inventory; those changes must update this page and migrate
  in-repo imports with **no** root deprecation aliases.
* **Power-user deep imports** — for each move, choose **one** and document it
  in this page (and a short migration note if paths change):

  * **Same-named package** — ``koopman_graph.training``, ``data``,
    ``analysis``, ``baselines``, ``losses`` keep working via package
    ``__init__`` re-exports when the old flat module is replaced by a package
    of the same name; or
  * **Migrate in-repo** — update every docs/tests/notebook deep import to the
    new path and record the rename here. **Do not** add long-lived root shim
    modules for renamed paths (``operator`` → ``operators``, ``encoder`` →
    ``nn.encoder``, …). v0.3.0 removed those temporary root shims.

Do not leave half-migrated import paths. Serialization type strings and
checkpoint reconstruct rules stay intact unless a release explicitly bumps
``FORMAT_VERSION``. The current checkpoint baseline is ``FORMAT_VERSION`` 1
(full schema) through the 0.x beta; incomplete older payloads are deprecated
and rejected rather than migrated. Formal multi-version checkpoint tracking
begins at 1.0.

API layers
----------

The package has three intentional layers.

Public façade
~~~~~~~~~~~~~

Stable, supported entry points re-exported from :mod:`koopman_graph` (see
``__all__``). Prefer these in tutorials, notebooks, and application code.

The package uses a **thin** root façade: root exports construct and operate
the primary encode → evolve → decode / fit / predict / spectrum workflow;
capability modules provide training customization, evaluation utilities,
baselines, adaptation, control, and specialized helpers. Demotions remove
symbols from root ``__all__`` with no root deprecation aliases, consistent
with the shim-removal policy above.

v0.4.0 shipped two factory-composable surfaces at the package root —
:class:`~koopman_graph.nn.delay.DelayEmbeddingEncoder` and
:class:`~koopman_graph.operators.GraphKoopmanOperator`. Both remain in
``__all__`` (not capability-only) so released ``from koopman_graph import …``
code stays valid; capability packages
(:mod:`koopman_graph.nn`, :mod:`koopman_graph.operators`) stay supported as
alternate import paths. Do not silently demote either without a separately
versioned breaking migration.

**Keep in** ``koopman_graph.__all__`` (core workflow; exactly these 25):

* :class:`~koopman_graph.model.GraphKoopmanModel`
* :class:`~koopman_graph.nn.encoder.GNNEncoder`,
  :class:`~koopman_graph.nn.encoder.GATEncoder`,
  :class:`~koopman_graph.nn.encoder.SAGEEncoder`,
  :class:`~koopman_graph.nn.encoder.DiffConvEncoder`,
  :class:`~koopman_graph.nn.encoder.GraphTransformerEncoder`,
  :class:`~koopman_graph.nn.hypergraph.HypergraphEncoder`,
  :class:`~koopman_graph.nn.delay.DelayEmbeddingEncoder`,
  :class:`~koopman_graph.nn.decoder.GNNDecoder`,
  :class:`~koopman_graph.nn.decoder.GATDecoder`,
  :class:`~koopman_graph.nn.decoder.SAGEDecoder`,
  :class:`~koopman_graph.nn.decoder.DiffConvDecoder`,
  :class:`~koopman_graph.nn.decoder.GraphTransformerDecoder`,
  :class:`~koopman_graph.nn.hypergraph.HypergraphDecoder`
  (also via :mod:`koopman_graph.nn`)
* :class:`~koopman_graph.operators.KoopmanOperator`,
  :class:`~koopman_graph.operators.ContinuousKoopmanOperator`,
  :class:`~koopman_graph.operators.GraphKoopmanOperator`,
  :class:`~koopman_graph.operators.ContinuousGraphKoopmanOperator`,
  :class:`~koopman_graph.operators.HypergraphKoopmanOperator`,
  :class:`~koopman_graph.operators.GlobalLocalKoopmanOperator`
* Data I/O for ``fit``: :class:`~koopman_graph.data.GraphSnapshotSequence`,
  :class:`~koopman_graph.data.MultiTrajectory`
* Primary spectrum entrypoints:
  :class:`~koopman_graph.spectrum_types.KoopmanSpectrum`,
  :func:`~koopman_graph.analysis.compute_spectrum`
* ``__version__``

Dataset generators remain via :mod:`koopman_graph.datasets` (not root
``__all__`` members; use ``from koopman_graph.datasets import …``).

**Demote to module imports** (still public and documented; not in root
``__all__``):

* :mod:`koopman_graph.baselines` — ``DMDBaseline``, ``EDMDBaseline``,
  ``DMDcBaseline``
* :mod:`koopman_graph.losses` — ``ForwardConsistencyLoss``,
  ``BackwardConsistencyLoss``, ``EigenvalueRegularizationLoss``,
  ``KoopmanSparsityLoss``, ``WorstCaseReconstructionLoss``,
  ``LieConsistencyLoss``, ``PDEResidualLoss``
* :mod:`koopman_graph.training` — ``FitHistory``, ``LossWeights``
* :mod:`koopman_graph.adaptation` — ``RecursiveKoopmanAdapter``,
  ``AdaptationStepResult``, ``KoopmanObserver``, ``FilterResult``
* :mod:`koopman_graph.env` — ``GraphKoopmanEnv``
* :mod:`koopman_graph.data` — ``TemporalSplit``, ``temporal_split``,
  ``WindowSampler``, ``as_multi_trajectory``
* :mod:`koopman_graph.metrics` — ``evaluate_forecast``,
  ``EvaluationResult``, ``mae``, ``rmse``, ``mape``, ``HorizonMetrics``
* :mod:`koopman_graph.analysis` — ``compute_generator_spectrum``,
  ``discrete_spectrum_at_delta_t``, ``decode_mode_shapes``,
  ``spectrum_distance``, ``koopman_std``, ``dynamical_similarity``,
  ``detect_anomaly``, ``calibrate_anomaly_threshold``,
  ``AnomalyDetectionResult``, ``estimate_coupling_from_snapshots``,
  ``CouplingEstimate``, ``identify_sparse_dynamics``, ``SINDyReport``,
  ``koopman_spectral_clustering``, ``ClusteringResult``
* :mod:`koopman_graph.observables` — ``graph_laplacian_features``
* :mod:`koopman_graph.uq` — ``EnsembleGraphKoopmanModel``,
  ``LatentGaussianKoopmanUQ``, ``ConformalKoopmanUQ``
* :mod:`koopman_graph.hierarchical` —
  ``HierarchicalGraphKoopmanModel``

.. code-block:: python

   from koopman_graph import (
       DelayEmbeddingEncoder,
       GNNDecoder,
       GNNEncoder,
       GraphKoopmanModel,
       GraphKoopmanOperator,
   )
   from koopman_graph.analysis import dynamical_similarity
   from koopman_graph.baselines import DMDBaseline
   from koopman_graph.data import as_multi_trajectory, temporal_split
   from koopman_graph.metrics import evaluate_forecast, rmse
   from koopman_graph.observables import graph_laplacian_features
   from koopman_graph.training import LossWeights

Power-user modules
~~~~~~~~~~~~~~~~~~

Importable modules that are documented and useful for advanced workflows, but
**not** advertised as a stable public contract. They may change without a major
version bump. Do not re-export new power-user symbols from
:mod:`koopman_graph` ``__all__`` without an intentional, documented API
change.

Examples:

* :mod:`koopman_graph.training` — loss schedules, ``train_one_epoch``,
  ``run_fit_loop``, ``resolve_device``, and related helpers
* :mod:`koopman_graph.serialization` — checkpoint build/load internals behind
  ``GraphKoopmanModel.save`` / ``load``
* :mod:`koopman_graph.datasets.dynamics` — Laplacian diffusion primitives and
  shared generation validators used by benchmark generators (dense
  ``I - alpha * L_sym`` step operators; shares pseudoinverse-normalized
  ``L_sym = P - Â`` with :mod:`koopman_graph.graph_utils`)
* :mod:`koopman_graph.datasets.topology` — shared path/ring ``edge_index``
  builders (:func:`~koopman_graph.datasets.topology.path_edge_index`,
  :func:`~koopman_graph.datasets.topology.ring_edge_index`) and
  :class:`~koopman_graph.datasets.TopologyPayload` for cached load APIs
  (not in root ``__all__``; import from the module or package as needed)
* :mod:`koopman_graph.datasets.download` — shared HTTP fetch and integrity
  helpers
  (:func:`~koopman_graph.datasets.download.download_url_bytes`,
  :func:`~koopman_graph.datasets.download.download_url_text`,
  :func:`~koopman_graph.datasets.download.download_url_to_path`,
  :func:`~koopman_graph.datasets.download.verify_sha256`,
  :func:`~koopman_graph.datasets.download.verify_sha256_bytes`,
  :func:`~koopman_graph.datasets.download.resolve_cache_path`) used by
  cached telemetry loaders and ``scripts/download_*.py`` (power-user; not in
  root ``__all__``)
* :mod:`koopman_graph.datasets.cache_cli` — shared argparse helpers for
  teaching-cache acquisition scripts (``--cache-dir``, ``--force``,
  ``--fetch``, ``--print-acquisition``; power-user; not in root ``__all__``)
* :mod:`koopman_graph.graph_utils` — capability package for shared
  graph-input resolution, latent propagation, and symmetric-normalized
  adjacency / Laplacian helpers (``topology`` / ``propagation`` peers; not
  in ``__all__``)
* :mod:`koopman_graph.nn` — encoder / decoder / delay / shared GNN primitives
  capability package (:class:`~koopman_graph.nn.gnn.BaseGNNModule`,
  activation typing, GCN/GAT builders). The package is power-user; public
  encoder/decoder/delay *classes* remain in ``__all__``. Peers import from
  ``nn.gnn`` only (no encoder↔decoder inversion).
* :mod:`koopman_graph.spectrum_types` — neutral
  :class:`~koopman_graph.spectrum_types.KoopmanSpectrum` value type plus
  discrete / generator / ``Δt`` spectrum assembly helpers
  (``compute_spectrum``, ``compute_generator_spectrum``,
  ``discrete_spectrum_at_delta_t``; re-exported from
  :mod:`koopman_graph.analysis` and, for the type / discrete helper, the
  package root)
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

Shared utilities live in :mod:`koopman_graph.graph_utils` —
documented, importable as ``koopman_graph.graph_utils``, and **not** listed in
``__all__``. Prefer this package over importing leading-underscore helpers from
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
* :mod:`koopman_graph.operators` (including
  :meth:`~koopman_graph.operators.graph.GraphKoopmanOperator.spectrum`)
  imports discrete spectrum assembly from
  :mod:`koopman_graph.spectrum_types` and must **not** import
  :mod:`koopman_graph.analysis` (eager or lazy).
* :mod:`koopman_graph.model` (spectrum helpers in ``model.inference``)
  imports :func:`~koopman_graph.spectrum_types.discrete_spectrum_at_delta_t`
  from the same neutral leaf and must **not** import
  :mod:`koopman_graph.analysis` for spectrum assembly (eager or lazy).
  Mode-shape decoding stays analysis-owned and hard-typed separately.
* :mod:`koopman_graph.data` must **not** import :mod:`koopman_graph.nn`
  (eager or lazy). Delay-window stacking / flattening / observation-mask
  helpers live in :mod:`koopman_graph.data.delay_windows`;
  :mod:`koopman_graph.nn.delay` and model encode paths import that surface
  (with thin re-exports on ``nn.delay`` for compatibility).
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

:mod:`koopman_graph.graph_utils` is a shallow capability package that
consolidates graph-input resolution, snapshot device transfer,
discrete/continuous latent propagation, and the shared autoregressive
rollout loop across two peer modules:

* ``topology`` — ``snapshot_edge_weight``, ``resolve_graph_inputs``,
  ``resolve_edge_index``, ``resolve_edge_weight``, ``snapshot_to_device``,
  and ``Â`` / ``L_sym`` helpers
* ``propagation`` — ``resolve_delta_t``, ``propagate_latent`` /
  ``inverse_propagate_latent``, ``advance_and_decode``,
  ``autoregressive_latent_rollout``, ``hold_last_topology_at``,
  ``snapshot_topology_at``, ``pack_rollout_snapshots``, plus
  ``KoopmanPropagator`` and rollout callback type aliases

Package ``__init__`` re-exports the documented power-user surface so
``from koopman_graph.graph_utils import …`` keeps working. Do not add
alternate compatibility shims or promote these symbols into root
``__all__``.

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

Decoded rollout tuples are packed into PyG ``Data`` snapshots by
:func:`~koopman_graph.graph_utils.pack_rollout_snapshots` (model predict /
evaluate and UQ ensemble paths). Peers outside :mod:`koopman_graph.model`
must import that helper from :mod:`koopman_graph.graph_utils`, not from
``model.inference``.

``GraphKoopmanEnv.step`` uses the same one-step ``advance_and_decode`` helper.

Topology capability matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~

Dynamic topology is first-class in the data container and neural model path,
but not every consumer supports it. Callers must not assume silent freeze/flatten
behavior. Optional static hyperedge incidence (``hyperedge_index`` /
``hyperedge_weight`` on each ``Data`` snapshot; PyG bipartite format) is
accepted by :class:`~koopman_graph.data.GraphSnapshotSequence`. Hypergraph
encoder/decoder peers and ``koopman="hypergraph"`` consume the incidence;
other consumers reject it:

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

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Surface
     - Hyperedge incidence (``has_hyperedges=True``)
   * - :class:`~koopman_graph.data.GraphSnapshotSequence`
     - Supported as optional static fields on each snapshot (time-varying
       hyperedges **rejected**); convenience accessors
       :attr:`~koopman_graph.data.GraphSnapshotSequence.hyperedge_index` /
       :attr:`~koopman_graph.data.GraphSnapshotSequence.hyperedge_weight`
   * - Shared normalization
     - :func:`~koopman_graph.graph_utils.dense_hyperedge_normalized_adjacency`
       (Zhou ``Ĥ``; isolated nodes → zero rows)
   * - :class:`~koopman_graph.nn.hypergraph.HypergraphEncoder` /
       :class:`~koopman_graph.nn.hypergraph.HypergraphDecoder`
     - Supported: model ``fit`` / ``predict`` with a matched hypergraph
       encoder/decoder pair
   * - :class:`~koopman_graph.operators.HypergraphKoopmanOperator`
       (``koopman="hypergraph"``)
     - Supported: discrete hyperedge-coupled advance; ``spectrum`` /
       eigenvalue regularization require ``hyperedge_index`` (never
       silently substitute ``K_self``)
   * - Classical baselines / :class:`~koopman_graph.env.GraphKoopmanEnv` /
       non-hypergraph encoder/decoder models without a hypergraph operator
     - **Rejected** (hyperedge incidence would be silently ignored)

Control layout capability matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~koopman_graph.data.GraphSnapshotSequence` stores either global
``(T, control_dim)`` or per-node ``(T, N, control_dim)`` controls. Consumers
must not assume the same physics for both layouts. Operator construction
defaults to additive control ``z @ K.T + u @ B``; set
``control_mode="bilinear"`` (optional ``bilinear_rank``) for control-affine
state–control couplings ``sum_i u_i N_i z`` on discrete and continuous
operators (and networked self-terms).

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

ODO and structural modes assemble their diagonal factors via the non-private
contract helpers ``bounded_diagonal``, ``strict_diagonal_values``, and
``safe_diagonal_inverse`` (re-exported from :mod:`koopman_graph.operators`).
Continuous Hurwitz diagonals keep a same-module private helper; do not import
leading-``_`` symbols across operator modules.

Continuous-only ``"auxiliary_spectral"`` extends the shared Literal but is
**rejected** on discrete / graph operators. In that mode there is no fixed
global ``matrix`` / ``L``: use state-dependent ``generator_at(z)`` and
``instantaneous_spectrum(z)`` (parametric / locally linear spectrum; not a
global spectral-radius certificate). Configure the auxiliary MLP via
``auxiliary_hidden_dims`` /
``koopman_auxiliary_hidden_dims`` (default ``(64, 64)``).

**Global/local (discrete) state-dependent caveat.**
:class:`~koopman_graph.operators.global_local.GlobalLocalKoopmanOperator`
(``koopman="global_local"``) uses a shared backbone ``K_g`` plus a per-step
low-rank correction ``K_ℓ(z_{t-w:t})``. ``matrix`` / ``spectral_radius`` /
model ``spectrum`` report ``K_g`` only; the effective step operator is
state-dependent and is **not** a single global certificate (same vocabulary
precedent as continuous ``auxiliary_spectral``). Configure
``koopman_local_window`` / ``koopman_local_rank`` /
``koopman_local_hidden_dims`` (defaults ``4`` / ``2`` / ``(32,)``).
``inverse_advance`` approximates with ``K_g`` only. Online RLS adaptation
rejects this operator.

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

**Networked graph / continuous-graph / hypergraph extension.**
:class:`~koopman_graph.operators.GraphKoopmanOperator`,
:class:`~koopman_graph.operators.ContinuousGraphKoopmanOperator`, and
:class:`~koopman_graph.operators.HypergraphKoopmanOperator` also implement the
contract, but ``matrix`` / ``K`` / ``L`` / ``spectral_radius`` /
``max_real_part`` refer to the **per-node self-term** only. Topology-coupled
spectral analysis uses the explicit helpers ``effective_matrix`` /
``effective_generator`` / ``spectrum``:

* Graph — assemble a topology-coupled map on a supplied ``edge_index`` /
  ``edge_weight`` (package row-state latent layout). The adjacency factor
  depends on ``adjacency``:

  - ``"symmetric"`` (default):
    :math:`I_N \otimes K_{\mathrm{self}} + \widehat{A}_{\mathrm{sym}} \otimes K_{\mathrm{nbr}}`
    with symmetric normalization
  - ``"random_walk"``:
    :math:`I_N \otimes K_{\mathrm{self}} + \widehat{A}_f \otimes K_{\mathrm{nbr}}`
  - ``"dual_random_walk"``:
    :math:`I_N \otimes K_{\mathrm{self}} + \widehat{A}_f \otimes K_{\mathrm{fwd}} + \widehat{A}_b \otimes K_{\mathrm{bwd}}`
    (``K_fwd`` aliases ``K_nbr``)
* Continuous graph — assemble ``L_eff`` with the same ``adjacency`` modes as
  discrete ``GraphKoopmanOperator`` (``symmetric`` / ``random_walk`` /
  ``dual_random_walk``); advance applies ``exp(L_eff Δt)``. Dense realization
  forms an ``N·d`` matrix exponential (prefer modest ``N`` or
  ``sparsity="block_diagonal"`` for the self-dominated approximation, which
  ignores neighbor coupling). Checkpoint kind is always ``continuous_graph``
  (including when selected via ``koopman="graph"`` +
  ``dynamics_mode="continuous"``).
* Hypergraph — assemble ``I_N ⊗ K_self + Ĥ ⊗ K_hedge`` on a supplied
  ``hyperedge_index`` / ``hyperedge_weight`` (Zhou ``Ĥ`` semantics as advance).

:meth:`~koopman_graph.model.GraphKoopmanModel.spectrum` requires those
topology arguments when ``koopman="graph"``, ``koopman="continuous_graph"``,
or ``koopman="hypergraph"`` and never silently substitutes the self-term.
:class:`~koopman_graph.losses.EigenvalueRegularizationLoss` follows the same
split: ``dense`` / ``odo`` hinge on ``eigvals`` of the topology-coupled
effective map / generator with required topology (training averages
pair-target topologies for dynamic sequences); structural modes keep
factor-level ``bound_metric`` and must not be described as whole-network
stability guarantees.

Bilinear ``inverse_advance`` matches the control capability matrix: global
``(C,)`` folds into a shared ``K_self`` override, while per-node ``(N, C)``
uses node-specific bilinear self blocks plus the same ``Â ⊗ K_nbr`` neighbor
coupling as forward advance.

**Graph / hypergraph operator sparsity (``sparsity``).** Forward advance is
always the sparse message-passing matvec (pairwise ``Â`` or Zhou ``Ĥ``).
``inverse_advance`` and serialization differ:

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - ``sparsity``
     - ``inverse_advance``
     - Notes
   * - ``"dense"`` (default)
     - Exact dense ``N·d`` inverse (pseudoinverse fallback)
     - Suitable for modest ``N``; ``effective_matrix`` / ``spectrum`` still
       assemble the dense map
   * - ``"block_diagonal"``
     - Approximate per-node ``d×d`` Jacobi inverse
       (exact when the coupling factor is zero or the topology is empty —
       ``K_nbr=0`` / no edges for graph; ``K_hedge=0`` / no hyperedges for
       hypergraph)
     - For backward-consistency / imputation at scale; **not** an exact
       whole-network inverse. ``inverse_matrix=`` is rejected. Shared by
       :class:`~koopman_graph.operators.GraphKoopmanOperator` and
       :class:`~koopman_graph.operators.HypergraphKoopmanOperator`
   * - ``"distributed"``
     - — (rejected)
     - Planned; not in 0.6.0

The stored checkpoint field is ``config.sparsity`` (format 1). Continuous-graph
``block_diagonal`` remains the self-dominated advance / inverse shortcut
documented under that peer (not the Jacobi path above).

**Graph / continuous-graph adjacency (``adjacency``).** Neighbor-coupling
normalization for pairwise networked operators:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - ``adjacency``
     - Coupling
   * - ``"symmetric"`` (default)
     - ``D^{-1/2} A D^{-1/2}``
   * - ``"random_walk"``
     - ``D_{\mathrm{out}}^{-1} A``
   * - ``"dual_random_walk"``
     - Forward plus ``D_{\mathrm{in}}^{-1} A^{\top}`` (``K_bwd`` / ``L_bwd``
       in the state dict only for this mode)

Factory keyword ``koopman_adjacency`` selects the mode for
``koopman="graph"`` / ``"continuous_graph"`` (rejected for other kinds).
Checkpoint field ``config.adjacency`` is required for those kinds and is
``null`` otherwise (format 1; incomplete payloads are rejected, not
migrated). Hypergraph operators stay Zhou-symmetric and do not expose
``adjacency``.

**Symmetry-adapted self factors (orbit ties).** Opt-in inductive bias for
discrete ``koopman="graph"`` / ``"hypergraph"``: nodes in the same orbit
share a ``K_self`` factor. Neighbor / hyperedge factors
(``K_nbr`` / ``K_fwd`` / ``K_bwd`` / ``K_hedge``) stay globally shared and
are **not** orbit-tied.
Default (symmetry off) keeps today's single shared ``K_self``. Enable via
factory kwargs ``koopman_orbit_partition=...`` (explicit) or
``koopman_auto_orbits=True`` (bind from topology on first advance, and
again at the start of ``fit`` before the optimizer is built so orbit
``K_self`` parameters are trained; ``koopman_orbit_method="auto"|"exact"``).
Explicit partitions always win over auto-detection. Hypergraph auto-orbits use the pairwise 2-section of
``hyperedge_index``. Orbit finding uses ``networkx`` (Weisfeiler–Lehman
approximate) or optional ``pynauty`` (exact) behind the ``[symmetry]``
extra; when neither is available, ``node_orbit_partition`` warns and
returns the identity partition (one orbit per node). Approximate
automorphisms can mis-tie blocks — treat as an inductive bias, not a
guarantee of the true automorphism group. Checkpoint field
``config.symmetry`` is ``None`` or
``{auto_orbits, orbit_partition, method}`` (format 1).

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
  ``koopman.matrix`` for ordinary / custom per-node operators (and via the
  topology-bound effective matrix for
  :class:`~koopman_graph.operators.GraphKoopmanOperator`, which requires
  ``edge_index`` / ``num_nodes``), and eigenvalue regularization via
  ``eigvals(matrix)`` for ordinary / custom ``dense`` / ``odo`` operators,
  ``eigvals(effective_matrix)`` for graph ``dense`` / ``odo`` (requires the
  same topology / edge-weight semantics as advance; training threads
  pair-target topologies from the sequence / window and never falls back to
  ``K_self`` alone), and ``bound_metric`` for structural modes
  (``max(bound(K_self), bound(K_nbr))`` for graph operators is a
  **factor-level surrogate**, not a whole-network stability certificate),
  plus the model's ``dynamics_mode``. Discrete dense
  backward-consistency training optionally reuses a built-in
  ``dense_inverse_matrix()`` when present; operators without that helper
  still invert via ``inverse_advance`` (``inverse_matrix=None``).
* **Built-in-only**: checkpoint ``save`` / ``load`` round-trips all
  serializable built-in operator kinds —
  :class:`~koopman_graph.operators.KoopmanOperator`,
  :class:`~koopman_graph.operators.ContinuousKoopmanOperator`,
  :class:`~koopman_graph.operators.GraphKoopmanOperator`,
  :class:`~koopman_graph.operators.HypergraphKoopmanOperator`,
  :class:`~koopman_graph.operators.GlobalLocalKoopmanOperator`, and
  :class:`~koopman_graph.operators.ContinuousGraphKoopmanOperator`
  (including ``koopman_kind``, control / bilinear / sparsity / adjacency /
  symmetry / global-local factory metadata, and ``n_delays``). RLS
  :class:`~koopman_graph.adaptation.RecursiveKoopmanAdapter` seed and
  write-back is a **narrower** surface: dense built-in discrete and
  continuous operators only (not networked graph / hypergraph /
  global-local / continuous-graph kinds, and not custom injections).
  Custom injected operators raise on save; reconstruct with
  ``koopman=...`` after loading encoder/decoder state separately if needed.

Factory and dimension rules:

* Factory kwargs (``koopman_init_mode``, ``koopman_init_scale``,
  ``koopman_parameterization``, ``koopman_max_spectral_radius``) must stay at
  their defaults when injecting.
* ``latent_dim`` / ``control_dim`` must match the injected operator.
* When set, ``control_mode`` / ``bilinear_rank`` must match the injected
  operator (defaults remain ``"additive"`` / ``None``).
* Built-in :class:`~koopman_graph.operators.KoopmanOperator` and
  :class:`~koopman_graph.operators.GraphKoopmanOperator` require
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

Optional dependencies must **not** break importing ``koopman_graph``
(``ARCH-050`` — heavy extras stay in declared adapter modules). Prefer
**fail-at-call** with an ``ImportError`` that names the install extra or
package:

* **Soft import + call guard (Gymnasium / ``[rl]``)** —
  :mod:`koopman_graph.env` soft-imports ``gymnasium`` so
  :class:`~koopman_graph.env.GraphKoopmanEnv` can subclass ``gym.Env`` when
  present. Construction calls ``_require_gymnasium()`` and raises with
  ``pip install koopman-graph[rl]`` guidance when the extra is missing.
* **Call-site import (``[symmetry]`` / ``networkx``)** —
  :func:`~koopman_graph.graph_utils.node_orbit_partition` imports
  ``networkx`` (and optionally ``pynauty``) at call time. Missing extras
  fall back to the identity partition with a ``UserWarning`` and install
  guidance (``pip install 'koopman-graph[symmetry]'``). Core imports stay
  clean without the extra.
* **Call-site import (``[mpc]`` / ``osqp``)** —
  :class:`~koopman_graph.mpc.KoopmanMPC` imports ``osqp`` at solve time.
  Missing extras raise ``ImportError`` with
  ``pip install 'koopman-graph[mpc]'`` guidance. Core imports stay clean
  without the extra.
* **Call-site import (e.g. ``h5py`` for METR-LA / PEMS-BAY)** —
  import inside the function that needs the dependency and re-raise
  ``ImportError`` with install guidance (see
  :mod:`koopman_graph.datasets.metr_la` and
  :mod:`koopman_graph.datasets.pems`).

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
:class:`~koopman_graph.training.history.TrainingLossBreakdown`). Prefer optional fields
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
required. :class:`~koopman_graph.uq.PredictionInterval` follows the same
pattern for its ``mean`` / ``lower`` / ``upper`` forecast collections.
:class:`~koopman_graph.data.MultiTrajectory` stores its trajectories
as a frozen tuple of sequences.

``TrainingLossBreakdown`` is a frozen internal snapshot (batch/epoch loss
terms) owned by :mod:`koopman_graph.training.history`, not a
:mod:`koopman_graph.training` package ``__all__`` export. Epoch aggregation
uses a local dict accumulator, then constructs a new breakdown.

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
  the public signature / ``FitHistory`` return type and delegates the
  validation preamble to
  :func:`~koopman_graph.model.validation.prepare_fit_inputs` (epoch bounds,
  control layouts, multi-trajectory resolve, snapshot-length checks,
  early-stopping monitor resolution).
* **Functional loop** — :func:`~koopman_graph.training.run_fit_loop` owns
  device placement, optimizer / scheduler setup, epoch iteration
  (``train_one_epoch`` / ``train_windowed_epoch`` / ``eval_one_epoch``), early
  stopping, best-weight tracking, optional checkpoint writes, and history
  assembly. Prefer extending that helper over growing the model class.
* **Neighbor-subgraph training** — pass
  :class:`~koopman_graph.data.NeighborWindowSampler` via ``fit(..., sampler=...)``
  (mutually exclusive with ``window_length``). Training losses (rollout,
  consistency, graph-operator eigenvalue regularization) use the **sampled**
  induced topology — document as an approximation for graphs too large to
  train whole. ``predict`` / ``evaluate`` remain full-graph. Dynamic topology
  and hyperedge sequences are rejected by the sampler.

Online adaptation and state estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~koopman_graph.adaptation.RecursiveKoopmanAdapter` keeps RLS regression
state on **CPU**. Operator matrices are detached and copied to CPU when the
adapter is constructed from a live operator. :meth:`~koopman_graph.adaptation.RecursiveKoopmanAdapter.apply_to`
copies adapted weights back onto the target operator's device and dtype.

:class:`~koopman_graph.adaptation.KoopmanObserver` runs a latent-space Kalman
filter / RTS smoother for imputation under ``observation_masks``. The process
model is exactly linear in the Koopman latent for fixed dense ``K`` (library
row convention ``z ← z Kᵀ``; flattened ``A = I ⊗ K``). Observation
handling is **not** an exact KF in feature space:

* ``observation_model="latent_encode"`` (default) — heuristic: encode
  mask-prepared features and keep **selected** rows of ``H = I`` for observed
  node blocks (selection matrix ``H_t = S_t``; fast; suitable for CI /
  notebooks).
* ``observation_model="decoder_jacobian"`` — EKF-style ``H = ∂decode/∂z``
  with unobserved rows dropped (design-faithful; costly and only locally
  valid for nonlinear GNN decoders).

Import the observer from :mod:`koopman_graph.adaptation` (power-user; not on
the root façade). See ``examples/25_kalman_koopman_state_estimation.ipynb``.

Classical baselines and datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DMD-family baselines (:class:`~koopman_graph.baselines.DMDBaseline`,
:class:`~koopman_graph.baselines.EDMDBaseline`,
:class:`~koopman_graph.baselines.DMDcBaseline`) share
:class:`~koopman_graph.baselines.ClassicalBaseline` for ``time_step`` /
``rank``, fitted metadata, and ``_check_fitted`` scaffolding, plus the
documented non-private helpers in :mod:`koopman_graph.baselines.base`
(flattening, least-squares fits, static-topology / global-control guards,
topology copy, and initial-graph checks). They remain siblings under that ABC
(DMDc does not subclass DMD) and satisfy
:class:`~koopman_graph.protocols.ForecastModel` structurally.
:class:`~koopman_graph.baselines.EDMDBaseline` stores a least-squares
``reconstruction_matrix`` (not a GNN decoder) and lifts via
``dictionary`` in ``{"polynomial", "rbf", "kernel"}``. Kernel dictionaries
with full-data centers are :math:`O(T^2)` and documented as small/medium-``T``
only; the linear kernel uses the identity feature map (DMD reduction).

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
     - Optional ``delta_t`` in continuous mode; for ``koopman="graph"`` or
       ``koopman="continuous_graph"``, required ``edge_index`` /
       ``num_nodes`` (optional ``edge_weight``); for ``koopman="hypergraph"``,
       required ``hyperedge_index`` / ``num_nodes`` (optional
       ``hyperedge_weight``)
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
* Networked graph / continuous-graph models — pass the same ``edge_index`` /
  ``num_nodes`` / ``edge_weight`` used for advance so
  :func:`~koopman_graph.analysis.resolve_spectrum` builds the effective
  operator / generator (missing topology raises on ``koopman="graph"`` or
  ``koopman="continuous_graph"``)
* Networked hypergraph models — pass the same ``hyperedge_index`` /
  ``num_nodes`` / ``hyperedge_weight`` used for advance (missing topology
  raises on ``koopman="hypergraph"``)

:func:`~koopman_graph.analysis.resolve_spectrum` is the shared resolver.
:func:`~koopman_graph.analysis.decode_mode_shapes` stays hard-typed to
:class:`~koopman_graph.model.GraphKoopmanModel` because it needs encode /
decode and a GNN decoder; do not widen it to :class:`~koopman_graph.protocols.SpectrumProvider`.
For graph models it derives topology from the reference ``Data`` / tensor and
forwards it into :meth:`~koopman_graph.model.GraphKoopmanModel.spectrum`.

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
* **Real telemetry** — ``Benchmark.load_topology`` / ``load_sequence``
  (METR-LA, PEMS-BAY, PEMS03/04/07/08, SocioPatterns primary-school
  contacts) read cached downloaded artifacts. There is no ``generate``
  because the time series is observed, not simulated. PEMS and contact-
  epidemic loaders follow the same cache idiom with fetch-script
  acquisition and SHA256 verification; FAIR dataset cards live in
  :doc:`data`. Contact-epidemic upstream data are CC-BY-NC-SA and must
  not be bundled in the repository.

**Seed defaults.** All simulated ``generate`` methods default ``seed=None``
(unseeded RNG). Pass an explicit integer for reproducible runs; tutorial
notebooks and quickstart examples use ``seed=42``.

**Topology payloads.** IEEE 118, METR-LA, PEMS, and contact-epidemic
``load_topology`` return a frozen
:class:`~koopman_graph.datasets.TopologyPayload` (attribute access
preferred). Mapping-style access (``payload["edge_index"]``) remains supported
for existing notebooks. On-disk caches stay plain ``dict``; the typed payload
is the public Python return type.

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

**Shared path/ring builders.** Synthetic, Lorenz-96, and
Kuramoto–Sivashinsky path/ring topologies use
:func:`~koopman_graph.datasets.topology.path_edge_index` and
:func:`~koopman_graph.datasets.topology.ring_edge_index` (power-user;
not root ``__all__``). Epidemic row-normalized adjacency and Watts–Strogatz
construction remain local to the epidemic module.

**Nonlinear / chaotic dataset peers.** The former grab-bag
``koopman_graph.datasets.nonlinear`` was hard-migrated into flat siblings
(no compatibility shim):

* :mod:`koopman_graph.datasets.epidemic` —
  :class:`~koopman_graph.datasets.EpidemicNetworkBenchmark` plus private
  Watts–Strogatz and row-normalized-adjacency helpers
* :mod:`koopman_graph.datasets.lorenz96` —
  :class:`~koopman_graph.datasets.Lorenz96GraphBenchmark`
* :mod:`koopman_graph.datasets.kuramoto_sivashinsky` —
  :class:`~koopman_graph.datasets.KuramotoSivashinskyBenchmark`
* :mod:`koopman_graph.datasets.cylinder_wake` —
  wake mesh/cache helpers and
  :class:`~koopman_graph.datasets.CylinderWakeBenchmark`

Package-level ``datasets.__all__`` names and semantics are unchanged.
Prefer ``from koopman_graph.datasets import …``; update any deep imports
that previously targeted ``koopman_graph.datasets.nonlinear``.

Laplacian helpers: sparse vs dense
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The symmetrically normalized Laplacian (Chung / Wikipedia) is

.. math::

   L_{\mathrm{sym}}
   = (D^{+})^{1/2} (D - A) (D^{+})^{1/2}
   = P - \widehat{A},

where ``Â = D_0^{-1/2} A D_0^{-1/2}`` uses the entrywise inverse-square-root
degree with zeros on isolated nodes (``(D^+)^{1/2}``), and
``P = diag(1_{d_i > 0})`` is the projector onto non-isolated nodes. When every
node has positive degree, ``P = I`` and this reduces to the familiar ``I - Â``.
Isolated nodes therefore have a **zero** Laplacian diagonal (not ``1``), so
``L_sym x`` maps isolates to zeros and the diffusion step ``I - α L_sym`` leaves
them unchanged. Symmetric normalization / ``L_sym`` assume an **undirected,
symmetrically represented** adjacency and are the contract for
``GraphKoopmanOperator(adjacency="symmetric")``. Directed coupling uses the
separate random-walk helpers
(:func:`~koopman_graph.graph_utils.dense_random_walk_normalized_adjacency` /
:func:`~koopman_graph.graph_utils.random_walk_normalized_adjacency_matvec`)
via ``adjacency="random_walk"`` (``D_{\mathrm{out}}^{-1} A``) or
``adjacency="dual_random_walk"`` (forward plus
``D_{\mathrm{in}}^{-1} A^{\top}``), matching DCRNN-style bidirectional
diffusion.

One shared weight core lives in :mod:`koopman_graph.graph_utils`
(:func:`~koopman_graph.graph_utils.symmetric_normalized_adjacency_edge_weights`
for ``Â``, plus
:func:`~koopman_graph.graph_utils.degree_support_mask` /
:func:`~koopman_graph.graph_utils.symmetric_normalized_laplacian_matvec` for
``P`` / ``L_sym``):

* **Sparse** — :func:`~koopman_graph.observables.graph_laplacian_features`
  applies ``L_sym @ x`` via matvec for hybrid physics lifting during training.
* **Dense** — :mod:`koopman_graph.datasets.dynamics` builds
  ``I - alpha * L_sym`` step operators for offline benchmark rollouts.

Keep both surfaces: they are numerically aligned on the same ``L_sym`` but
serve different performance and API roles. Networked operator advance / spectrum
paths that consume ``Â`` only should not be documented as ``I - Â`` under the
name ``L_sym``.

Hybrid physics checkpoint schema
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When hybrid observables are enabled, checkpoint ``config.physics`` owns
``dim``, ``preset``, and ``position``. :func:`~koopman_graph.serialization.build_model_config`
writes ``position`` from ``model.physics_position``;
:func:`~koopman_graph.serialization.reconstruct_model` reads and validates it via
:func:`~koopman_graph.observables.resolve_physics_position` (missing ``position``
defaults to ``"prepend"``). Only ``"prepend"`` is supported today — unsupported
values raise on load so a future ``"append"`` mode cannot silently mis-restore.
These fields are part of the current ``FORMAT_VERSION`` 1 baseline schema.

v0.3.0 architecture
-------------------

The v0.3.0 release standardized style and release quality. Checkpoint schema
versioning for that release has since been reset: the current source baseline
is ``FORMAT_VERSION`` 1 (see below).
Outcomes folded into the first public v0.3.0 cut:

**Core API changes**

1. **Shared rollout** — ``predict`` / training losses / the RL env share
   :func:`~koopman_graph.graph_utils.autoregressive_latent_rollout` (and
   one-step ``advance_and_decode``), with hold-last vs teacher-target topology
   policies documented above.
2. **``ForecastModel`` call sites** — loose Protocol for method presence;
   :class:`~koopman_graph.protocols.UncontrolledForecastModel` for portable
   ``predict(data, steps)`` among autonomous peers (see call-site matrix).
3. **Optional operator injection** — ``koopman=`` composes a custom
   :class:`~koopman_graph.operators.KoopmanOperatorContract` ``nn.Module``
   with Protocol-complete propagate / spectrum / eigenvalue paths;
   checkpoints round-trip all serializable built-in operator kinds while
   RLS seed / write-back stays on the narrower discrete/continuous surface
   (see Optional operator injection).
4. **Classical baseline scaffolding** —
   :class:`~koopman_graph.baselines.ClassicalBaseline` ABC; EDMD stores
   ``reconstruction_matrix`` (not a GNN ``decoder``).
5. **Frozen public result types** — ``FitHistory``, ``AdaptationStepResult``,
   and related dataclasses use ``frozen=True`` with tuple series;
   ``TrainingLossBreakdown`` is a frozen internal snapshot (not package-exported).

**Contract and boundary changes**

* Honest continuous/discrete dispatch for injected operators; sequence
  immutability; topology and control-layout capability guards; shared GNN
  primitives; thin ``fit`` orchestration; spectrum-based dynamical similarity;
  shared dataset validators; centralized dynamics/stability vocabulary;
  hybrid ``physics.position`` round-trip; loss-breakdown hygiene.

**Tutorial evidence**

* Notebooks 02 / 03 / 06 / 14 / 15 document scoped findings alongside stored
  outputs (history-constrained Vm scope, METR-LA baseline protocol, epidemic
  spectrum wording, physics-informed evidence scope, RL takeaway metrics).

**Capability packages and thin façade**

* Layout policy plus ``training/``, ``data/``, ``operators/``, ``nn/``,
  ``analysis/``, ``baselines/``, ``losses/``, and ``model/`` capability
  packages; former deep-import shims removed; thin root ``__all__`` retaining the
  twenty-five core-workflow names while baselines, primary losses, training
  knobs, adaptation/env, temporal-split helpers, forecast-eval entrypoints,
  and metrics / analysis / data / adaptation / observables secondaries
  import from capability modules.

**Analysis UX and release quality**

* :func:`~koopman_graph.analysis.plot_spectrum` with ``limits="unit_disk"`` /
  ``"data"`` (capability-module import only).
* Enforced pytest coverage gate of **90%** (``fail_under`` / CI
  ``--cov-fail-under`` / Codecov project and patch ``target``); branch-aware
  suite remains well above the floor.

v0.4.0 expanded capabilities
----------------------------

v0.4.0 added the following capabilities. The current checkpoint baseline is
``FORMAT_VERSION`` 1 (full schema including these fields); previously published
format-2 checkpoints are not loadable:

* Networked :class:`~koopman_graph.operators.GraphKoopmanOperator`
  (``koopman="graph"``) so dynamic topology participates in the linear step;
  model-level spectrum analyzes the topology-bound effective operator (not
  ``K_self`` alone)
* :class:`~koopman_graph.nn.DelayEmbeddingEncoder` / ``n_delays`` for
  Hankel / Takens-style partial observability
* Bilinear / control-affine terms (``control_mode="bilinear"``)
* :class:`~koopman_graph.adaptation.KoopmanObserver` for latent Kalman
  filtering and imputation under observation masks
* Nonlinear / chaotic graph benchmarks and STGCN / DCRNN / Graph WaveNet
  reference forecasters under ``koopman_graph.baselines.gnn``

Checkpoint schema baseline
--------------------------

``FORMAT_VERSION`` 1 is the current checkpoint schema through the 0.x beta.
New saves write ``format_version: 1`` with the full architecture config
(dynamics mode, control / bilinear metadata, hybrid physics, delays,
built-in operator kinds including ``hypergraph`` / ``global_local`` /
``continuous_graph``, encoder/decoder type strings including ``hyper_enc`` /
``hyper_dec``, keys ``sparsity`` / ``learn_topology`` /
``topology_embedding_dim`` / ``symmetry``, and global/local config keys
``local_window`` / ``local_rank`` / ``local_hidden_dims``). Loaders accept
only ``{1}``. Incomplete or previously published incompatible format-1
payloads are **deprecated** and rejected with a clear re-save error (no
silent migration). Other ``format_version`` values remain unsupported.
Formal multi-version checkpoint tracking and migration branches begin at 1.0.

v0.5.0 capability architecture
------------------------------

The v0.5.0 capabilities preserve the four existing styles (composition,
sklearn-like façade, functional helpers, string-mode configuration). Do
**not** introduce a parallel model hierarchy or silently expand root
``__all__``.

**Capability and API-tier map**

.. list-table::
   :header-rows: 1
   :widths: 24 18 58

   * - Capability home
     - API tier
     - Contracts to preserve
   * - :mod:`koopman_graph.uq` (ensemble peer)
     - Power-user
     - Compose :class:`~koopman_graph.model.GraphKoopmanModel`; reuse
       :func:`~koopman_graph.graph_utils.autoregressive_latent_rollout`;
       topology-required spectrum / regularization for ``koopman="graph"`` /
       ``koopman="continuous_graph"`` / ``koopman="hypergraph"``; format-1
       checkpoints
   * - :mod:`koopman_graph.operators` (``auxiliary_spectral`` on continuous)
     - Power-user (string mode)
     - State-dependent ``generator_at(z)`` vocabulary distinct from fixed
       ``matrix`` / graph-effective spectrum; cohesive helpers live in
       ``auxiliary_spectral`` / ``continuous_van_loan`` /
       ``continuous_parameterizations`` / ``continuous_propagation``
       rather than a continuous grab-bag;
       format-1 serialization
   * - :mod:`koopman_graph.observables` + :mod:`koopman_graph.losses`
     - Power-user / training
     - ``extra_losses`` / ``LossWeights`` keep topology threading; use
       documented operator helpers; ``L_sym = P - Â`` for Laplacian /
       PDE-residual presets; format-1 config fields; losses live in a
       same-named capability package
   * - :mod:`koopman_graph.nn` (DiffConv / SAGE peers)
     - Public root façade (with GCN/GAT)
     - Shallow peers under ``nn/``; no ``nn/conv/``; no encoder↔decoder
       imports; reuse ``nn.gnn`` primitives; format-1 type strings
       ``sage`` / ``diffconv``
   * - :mod:`koopman_graph.nn` (Transformer peers)
     - Public root façade (with GCN/GAT/SAGE/DiffConv)
     - Same ``nn/`` peer rules as the DiffConv/SAGE peers; format-1 type string
       ``transformer``; document compute cost honestly
   * - :mod:`koopman_graph.baselines` (``edmd`` peer)
     - Power-user / baseline façade as today
     - Topology-blind classical contract unchanged; scale limits for kernel
       dictionaries documented
   * - :mod:`koopman_graph.losses`
     - Power-user / training
     - Graph sparsity targets distinguish ``K_self`` / ``K_nbr`` parameters
       from topology-bound effective operators; documented non-private
       helper surfaces only; format-1 ``LossWeights``; same-named capability
       package
   * - :mod:`koopman_graph.uq` (probabilistic peer)
     - Power-user
     - Optional path after ensembles; honest naming vs DPK / K²VAE; compose
       rather than subclass the primary model
   * - :mod:`koopman_graph.uq` (conformal peer)
     - Power-user
     - :class:`~koopman_graph.uq.ConformalKoopmanUQ` composes a fitted
       model; split + adaptive (ACI) calibration; returns shared
       :class:`~koopman_graph.uq.PredictionInterval`; calibration state
       is wrapper-local (not model ``FORMAT_VERSION``; kind
       ``ConformalKoopmanUQ.calibration.v2``); scores
       ``aggregate`` / ``per_node`` / ``node_wise`` (+ optional
       ``neighbor_smoothing``); marginal coverage ``≥ 1 − α`` under
       exchangeability — approximate under temporal dependence; prefer ACI
       under drift
   * - :mod:`koopman_graph.hierarchical` (capability module/package)
     - Power-user
     - Compose the primary model; spectrum / regularization use **pooled**
       topology; global and per-node controls retain documented semantics
       through pool/unpool; format-1 serialization
   * - :mod:`koopman_graph.mpc` (capability package)
     - Power-user
     - :class:`~koopman_graph.mpc.KoopmanMPC` composes a fitted model;
       additive discrete per-node operators only (bilinear / networked /
       continuous rejected); plant state is mean latent; OSQP behind
       ``[mpc]``; output constraints via local decoder Jacobian (local
       guarantee); optional ``constraint_tightening=`` with calibrated
       :class:`~koopman_graph.uq.ConformalKoopmanUQ` shrinks output boxes
       only (inputs exact; calibrated margins, not a closed-loop
       guarantee); off root ``__all__``

**UQ and optional uncertainty**

:mod:`koopman_graph.uq` hosts the landed ensemble, latent-Gaussian, and
conformal peers. Wrappers compose :class:`~koopman_graph.model.GraphKoopmanModel`
and must not duplicate rollout implementation or invent a fifth architectural
style. Keep UQ classes off root ``__all__`` unless an explicit public API
decision promotes them.

Shared interval helpers and the public
:class:`~koopman_graph.uq.PredictionInterval` result type live in
:mod:`koopman_graph.uq.common` and are re-exported from
:mod:`koopman_graph.uq`
(:class:`~koopman_graph.uq.PredictionInterval`,
:func:`~koopman_graph.uq.quantile_levels`,
:func:`~koopman_graph.uq.snapshot_with_features`). Ensemble,
latent-Gaussian, and conformal peers must import those non-private symbols
— never each other's leading-``_`` helpers and never each other's copy of
the shared interval type.
:class:`~koopman_graph.uq.ConformalKoopmanUQ` documents frequentist
marginal coverage under exchangeability and recommends adaptive conformal
inference (ACI) when residuals drift. ``score="node_wise"`` stores
per-node quantiles and returns per-node marginal half-widths (not joint
coverage); legacy ``"per_node"`` still max-pools over nodes.
``PredictionInterval.mean`` / ``.lower`` / ``.upper`` are
immutable ``tuple[Data, ...]`` sequences (borrowed ``Data`` mutability as
for :class:`~koopman_graph.data.GraphSnapshotSequence`). Latent-Gaussian
forecast scheduling reuses
:meth:`~koopman_graph.model.GraphKoopmanModel.encode_rollout_origin`,
:func:`~koopman_graph.graph_utils.hold_last_topology_at`, and
:func:`~koopman_graph.graph_utils.propagate_latent` (same encode / hold-last
topology / advance policy as ``predict``). Ensemble member forecasts reuse
:func:`~koopman_graph.graph_utils.autoregressive_latent_rollout` and pack
decoded steps with
:func:`~koopman_graph.graph_utils.pack_rollout_snapshots` — never import
those packing helpers from ``model.inference``. Closed-form Gaussian covariance
updates may remain a distinct moment loop and need not call
:func:`~koopman_graph.graph_utils.autoregressive_latent_rollout` when
per-step decoding is inappropriate.

**Wrapper-local persistence (UQ / hierarchy)**

Wrapper state is **not** part of model ``FORMAT_VERSION`` checkpoints. Three
intentional patterns coexist:

* **Ensemble** — :meth:`~koopman_graph.uq.EnsembleGraphKoopmanModel.save` /
  ``load`` write a directory with per-member ``.pt`` files plus
  ``ensemble_manifest.json``.
* **Hierarchical** — :meth:`~koopman_graph.hierarchical.HierarchicalGraphKoopmanModel.save`
  / ``load`` write a directory with the wrapped model checkpoint, wrapper
  weights, and ``hierarchical_manifest.json``.
* **Conformal** — :meth:`~koopman_graph.uq.ConformalKoopmanUQ.save_calibration`
  / ``load_calibration`` persist calibration quantiles to a single file
  (wrapper-local; the underlying model is saved separately via
  :meth:`~koopman_graph.model.GraphKoopmanModel.save`).

Do not unify these into model checkpoints without an explicit schema decision.

**Physics residual training**

Physics residual callables are fit-time configuration:
:class:`~koopman_graph.training.ExtraLosses` is passed to
:meth:`~koopman_graph.model.GraphKoopmanModel.fit` alongside
:class:`~koopman_graph.training.LossWeights`. The callables are not constructor
state and are not serialized. :class:`~koopman_graph.losses.LieConsistencyLoss`
and :class:`~koopman_graph.losses.PDEResidualLoss` remain power-user module
imports outside root ``__all__``. The Lie term is continuous-time only; graph
derivative/PDE callables reuse the shared ``L_sym = P - Â`` utilities rather
than defining local Laplacians. The current Lie composition is autonomous and
rejects controlled operators instead of silently dropping additive/bilinear
control terms.

**Sparsity and worst-case reconstruction**

:class:`~koopman_graph.losses.KoopmanSparsityLoss` and
:class:`~koopman_graph.losses.WorstCaseReconstructionLoss` are power-user
training helpers (also outside root ``__all__``), enabled via
``LossWeights.sparsity`` / ``LossWeights.worst_case``. Graph sparsity targets
``K_self`` / ``K_nbr`` parameters only — not the topology-bound effective
operator. The worst-case term is a robust training objective, not a
generalization certificate.

``predict_interval`` is an **optional** capability detected via a dedicated
optional Protocol and/or ``hasattr`` / ``callable`` checks. Do **not** add it
as a required :class:`~koopman_graph.protocols.ForecastModel` method (that
would break existing classical and neural implementers).

**Hierarchy**

Hierarchical forecasting lives under
:mod:`koopman_graph.hierarchical`, composing the primary workflow class.
Keep the wrapper off root ``__all__`` by default. Directory + JSON-manifest
persistence is documented with the other wrapper-local patterns under
**Wrapper-local persistence (UQ / hierarchy)** above.

**Koopman-MPC**

:mod:`koopman_graph.mpc` provides receding-horizon control on additive
latent dynamics (Korda & Mezić, Automatica 2018). The façade composes
:class:`~koopman_graph.model.GraphKoopmanModel` and must not subclass it
or invent a parallel plant API. In 0.6.0 the QP plant is the mean latent
over nodes with a global control; output maps use a finite-difference
decoder Jacobian and therefore give only local constraint guarantees.
Keep ``KoopmanMPC`` off root ``__all__``. Optional
``constraint_tightening=`` accepts a calibrated
:class:`~koopman_graph.uq.ConformalKoopmanUQ` on the same model and
shrinks output boxes by per-horizon half-widths (stage ``0`` unshrunk;
stages ``1..H`` use ``quantiles[h-1]``; input bounds unchanged). Margins
inherit conformal exchangeability caveats plus local linearization — they
are statistically motivated, not a formal closed-loop guarantee.

**Current package boundaries**

:mod:`koopman_graph.losses` is a same-named capability package because peer
consistency / regularization / reconstruction / physics / rollout concerns
satisfy “When to nest”. :mod:`koopman_graph.model` is also a same-named
capability package (estimator + factory / validation / timing peers) while
keeping a single :class:`~koopman_graph.model.GraphKoopmanModel` workflow
surface.
Keep :mod:`koopman_graph.observables` flat unless a later audit shows
multi-concern or peer-growth criteria.

v0.6.0 capability architecture
------------------------------

v0.6.0 extends the four styles (composition, sklearn-like façade, functional
helpers, string-mode configuration) with higher-order topology, adaptive
edges, non-stationary / continuous networked operators, symmetry-adapted
orbit ties, analysis diagnostics (SINDy / spectral clustering / topology
estimation), conformal UQ, and Koopman-MPC. Do **not** introduce a parallel
model hierarchy or silently expand root ``__all__`` beyond the documented
twenty-five core-workflow names (hypergraph encoder/decoder and the three
new operator classes are the 0.6.0 root additions).

**Shipped capability list**

* Hypergraph encode / decode / operator
  (:class:`~koopman_graph.nn.HypergraphEncoder`,
  :class:`~koopman_graph.nn.HypergraphDecoder`,
  :class:`~koopman_graph.operators.HypergraphKoopmanOperator`;
  ``koopman="hypergraph"``)
* Self-adaptive pairwise topology
  (:class:`~koopman_graph.nn.AdaptiveAdjacency`;
  ``learn_topology="self_adaptive"``)
* Global/local discrete operator
  (:class:`~koopman_graph.operators.GlobalLocalKoopmanOperator`;
  ``koopman="global_local"``)
* Continuous networked generator
  (:class:`~koopman_graph.operators.ContinuousGraphKoopmanOperator`;
  ``koopman="graph"`` + ``dynamics_mode="continuous"`` or
  ``koopman="continuous_graph"``; dense ``N·d`` matrix-exp cost caveat)
* Symmetry-adapted orbit-tied ``K_self``
  (``koopman_auto_orbits`` / ``koopman_orbit_partition``;
  :mod:`koopman_graph.graph_utils.symmetry`; ``[symmetry]`` extra)
* Analysis peers: :func:`~koopman_graph.analysis.identify_sparse_dynamics`,
  :func:`~koopman_graph.analysis.koopman_spectral_clustering`,
  :func:`~koopman_graph.analysis.estimate_coupling_from_snapshots`,
  :func:`~koopman_graph.analysis.spectral_residuals`
* Long-horizon statistics leaf :mod:`koopman_graph.statistics` (PSD / W1 /
  Rosenstein LLE; off root ``__all__``)
* Pairwise ``adjacency`` modes on graph / continuous-graph operators
  (``symmetric`` / ``random_walk`` / ``dual_random_walk``;
  ``koopman_adjacency``; format-1 ``config.adjacency``)
* :class:`~koopman_graph.uq.ConformalKoopmanUQ` (split + ACI;
  ``node_wise`` / ``neighbor_smoothing``)
* Classical DMD ``rank="auto"`` (Gavish–Donoho)
* :class:`~koopman_graph.mpc.KoopmanMPC` (``[mpc]`` / OSQP; optional conformal
  tightening)
* Datasets: PEMS variants and SocioPatterns contact-epidemic FAIR cards
* Tutorials ``27``–``38`` covering the surfaces above (including residual /
  transfer / factorization notebooks)
* :doc:`limitations` — consolidated scope boundaries

**Capability and API-tier map (0.6.0 / 0.7.0 peers)**

.. list-table::
   :header-rows: 1
   :widths: 24 18 58

   * - Capability home
     - API tier
     - Contracts to preserve
   * - :mod:`koopman_graph.nn.hypergraph`
     - Public root façade
     - Incidence ``hyperedge_index``; root
       ``HypergraphEncoder`` / ``HypergraphDecoder``; format-1 type strings
       ``hyper_enc`` / ``hyper_dec``
   * - :mod:`koopman_graph.nn.adaptive_topology`
     - Power-user (``nn``)
     - ``learn_topology="self_adaptive"``; off root ``__all__``; pairwise
       only (not hypergraph incidence learning)
   * - :mod:`koopman_graph.operators.hypergraph`
     - Public root façade
     - ``koopman="hypergraph"``; effective spectrum requires incidence;
       ``sparsity="dense"|"block_diagonal"`` (Jacobi approx inverse);
       RLS rejected
   * - :mod:`koopman_graph.operators.global_local`
     - Public root façade
     - Discrete only; ``matrix`` / ``spectrum`` report ``K_g``; RLS rejected;
       window / rank / hidden-dims factory keys
   * - :mod:`koopman_graph.operators.continuous_graph`
     - Public root façade
     - Checkpoint kind always ``continuous_graph``; dense ``N·d`` exp or
       ``sparsity="block_diagonal"`` self-dominated approx
   * - :mod:`koopman_graph.graph_utils.symmetry`
     - Power-user
     - ``[symmetry]`` (networkx) for ``method="auto"``; identity fallback
       warns without backends; approximate orbits are an inductive bias
   * - :mod:`koopman_graph.analysis` (SINDy / clustering / topology estimation
       / residuals)
     - Power-user
     - Learned-latent caveat for SINDy; ARI / operator-quality caveats for
       clustering; coupling estimates are diagnostics;
       ``spectral_residuals`` is a held-out diagnostic (not ResDMD)
   * - :mod:`koopman_graph.statistics`
     - Power-user (neutral leaf)
     - Torch-only; off root ``__all__``; Lyapunov helper is
       embedding-sensitive and :math:`O(T^{2})`
   * - :mod:`koopman_graph.uq` (conformal)
     - Power-user
     - Wrapper-local calibration (``calibration.v2``);
       ``node_wise`` marginal widths; ``per_node`` legacy max-pool retained
   * - :mod:`koopman_graph.baselines` (``rank="auto"``)
     - Power-user
     - Gavish–Donoho ``optimal_hard_threshold_rank``; ``selected_rank`` on
       fit; helpers off package ``__all__``
   * - :mod:`koopman_graph.mpc`
     - Power-user
     - Same contracts as the v0.5.0 MPC row; ``[mpc]`` extra

**Optional extras inventory**

* ``[mpc]`` — OSQP for :class:`~koopman_graph.mpc.KoopmanMPC`
* ``[symmetry]`` — ``networkx`` for approximate node-orbit partitions
* ``[rl]`` — Gymnasium / Stable-Baselines3 (unchanged)
* ``[dev]`` / ``[docs]`` — tests and Sphinx (unchanged)

**Checkpoint note (format-1 baseline)**

0.6.0 continues the format-1 schema (see Checkpoint schema baseline above).
Previously published incompatible format-1 payloads and historical format-2
checkpoints remain rejected with a clear re-save error. New 0.6.0 fields
(``symmetry``, ``learn_topology``, hypergraph / global-local /
continuous-graph kinds, local-window keys) round-trip under format 1; loaders
still accept only ``{1}``.

Related documentation
---------------------

* :doc:`api` — module-level API reference
* :doc:`capabilities` — feature inventory
* :doc:`limitations` — scope boundaries and when not to use
* :doc:`quickstart` — end-user training and prediction walkthrough
* :doc:`faq` — install / import / checkpoint troubleshooting and support routing
* Repository ``CONTRIBUTING.md`` — development setup and contribution workflow
* Repository ``SECURITY.md`` — supported versions and private vulnerability reporting
