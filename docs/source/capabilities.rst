Capabilities and datasets
=========================

This page is the detailed capability inventory for KoopmanGraph. The
repository ``README.md`` stays a short landing page; use this page (and
:doc:`tutorials`) when you need the full surface area.

Capability groups
-----------------

Topology-aware learning
~~~~~~~~~~~~~~~~~~~~~~~

* ``GraphKoopmanModel`` — encode → Koopman advance → decode with
  ``fit``, ``predict``, ``evaluate``, and ``encode``
* ``GNNEncoder`` / ``GATEncoder`` / ``SAGEEncoder`` / ``DiffConvEncoder`` /
  ``GraphTransformerEncoder`` and matching ``GNNDecoder`` / ``GATDecoder`` /
  ``SAGEDecoder`` / ``DiffConvDecoder`` / ``GraphTransformerDecoder``
  (GCN uses Kipf-normalized adjacency
  :math:`\widehat{A} = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}`;
  GraphSAGE: Hamilton et al. 2017; DiffConv: DCRNN-style bidirectional
  diffusion, Li et al. 2018; Transformer: PyG ``TransformerConv`` / Shi et al.
  masked attention on edges — typically denser compute than GCN/GAT/DiffConv
  per edge × heads)
* ``SeparableDictionaryEncoder`` / ``SeparableDictionaryDecoder``
  (node-wise MLP; ``encoder_kind="separable"``; zero graph hops). Pass
  the classes into ``GraphKoopmanModel``; there is no factory
  ``encoder="separable"``. Homomorphism precondition
  (``Peng2026KoopmanGKFA``); not a GNN and not on the root façade.
* ``HypergraphEncoder`` / ``HypergraphDecoder`` with incidence
  ``hyperedge_index`` (``koopman="hypergraph"``); directed incidence via
  ``tail_index`` / ``head_index`` when using non-Zhou ``incidence_mode``
* Heterogeneous / multiplex RelGraph peers
  (``RelGraphEncoder`` / ``RelGraphDecoder``) with
  ``koopman="hetero_graph"`` on ``HeteroGraphSnapshotSequence`` /
  PyG ``HeteroData`` (R-GCN-lite per-relation messages; Schlichtkrull et
  al. 2018 motivation only). Shared latent width :math:`d` is the default
  (stacked layout in :mod:`koopman_graph.data.hetero_layout`); opt-in
  per-type widths :math:`d_\tau` via ``latent_dims`` use rectangular
  relation maps :math:`K_r`. Continuous hetero
  (``dynamics_mode="continuous"``) is supported with documented dense
  :math:`\Phi` cost. Optional HGT peers (``HGTEncoder`` / ``HGTDecoder``
  in :mod:`koopman_graph.nn`) wrap PyG ``HGTConv`` and are **not**
  required for hetero support. Windowed single-process
  ``run_fit_loop`` accepts windowed hetero sequences
* Combinatorial simplicial-1 / Hodge lifts via
  ``SimplicialEncoder`` / ``SimplicialDecoder``
  (:mod:`koopman_graph.nn.simplicial`; oriented ``edge_index``, optional
  ``face_index``)
* Sheaf MVP peers
  (:class:`~koopman_graph.nn.sheaf.SheafGNNEncoder` /
  :class:`~koopman_graph.nn.sheaf.SheafGNNDecoder`; factory
  ``encoder="sheaf"``) — diagonal restriction maps by default; opt-in
  dense maps under a documented channel ceiling. **Not** full TopologicX
  parity (see :doc:`limitations`)
* Cell-complex MVP peers
  (:class:`~koopman_graph.nn.cell_complex.CellComplex`,
  :class:`~koopman_graph.nn.cell_complex.CellComplexGNNEncoder` /
  :class:`~koopman_graph.nn.cell_complex.CellComplexGNNDecoder`; factory
  ``encoder="cell_complex"``; ``Data.face_index`` required) — Hodge
  :math:`L_0` mix with the same linear Koopman head
* ``InvariantGeometryEncoder``
  (:mod:`koopman_graph.nn.equivariant`) — Tier A invariant distance /
  angle features from ``Data.pos`` lifted by a GCN; invariant features
  do **not** make latent :math:`K` E(n)/SE(3) equivariant
* Optional ``E3EquivariantEncoder`` (``e3nn``,
  ``pip install "koopman-graph[equivariance]"``) — Tier B steerable
  encode to invariant scalar latents; still **not** an equivariant
  :math:`K`
* ``DelayEmbeddingEncoder`` / ``n_delays`` for Takens-style channel
  stacking — not :class:`~koopman_graph.baselines.HankelDMDBaseline` or
  :class:`~koopman_graph.baselines.HAVOKBaseline`
* Per-snapshot ``edge_index`` (dynamic topology) and end-to-end
  ``edge_weight`` support — mechanical rewiring only; not
  criticality / spectral-degeneracy analysis (see
  :ref:`limitations-topology-criticality`)
* Opt-in fixed-union node churn: ``allow_node_churn=True`` with
  :math:`N_{\max}` capacity and ``presence_masks`` ``(T, N_{\max})``
  (losses ignore inactive nodes; matvecs stay at capacity — see
  :doc:`limitations`)
* :class:`~koopman_graph.data.EntityRemap` places source nodes into a
  finite :math:`N_{\max}` via a user-supplied injective index (not
  automatic identity resolution; changing :math:`N` without a remap
  raises)
* Homogeneous ``parameter_trajectory`` and
  :class:`~koopman_graph.data.ConditioningContext` record
  :math:`(\\mu, t, u)` (off the root façade). This does not change the
  default operator. ``koopman="switched"`` / ``"mixture"`` are not
  :math:`K(\\mu)`. Opt-in ``koopman="parametric"`` interpolates
  :math:`K(\\mu)=\\sum_j \\alpha_j(\\mu) K_j`.
  :func:`~koopman_graph.data.diurnal_control_features` and
  :func:`~koopman_graph.data.diurnal_phase_index` are time-of-day
  recipes on existing control / switched ``phase_index``, not a
  calendar serializer. Discrete sequences still require uniform
  :math:`\\Delta t`.
* Optional self-adaptive pairwise topology
  (``learn_topology="self_adaptive"``; ``koopman_graph.nn.AdaptiveAdjacency``)
* Optional symmetry-adapted orbit-tied ``K_self`` (and, on discrete
  graph / hypergraph / Hodge operators, per-orbit ``K_nbr`` /
  ``K_hedge``)
  (``koopman_auto_orbits`` / ``koopman_orbit_partition``; requires
  ``pip install "koopman-graph[symmetry]"`` for auto orbits)
* Opt-in isotypic self-block and neighbor-factor ties
  (``koopman_symmetry="isotypic"``; exact automorphism groups;
  :func:`~koopman_graph.graph_utils.compute_isotypic_decomposition`;
  ``examples/45_isotypic_symmetry.ipynb``) — not a guaranteed
  sample-efficiency win
* Predicted next-step topology: default
  :class:`~koopman_graph.nn.SparseCandidateTopologyHead` when
  :class:`~koopman_graph.data.GraphDynamicsConfig` is set
  (``topology_head="sparse_candidate"``; at most ``candidate_k``
  destinations per node). :class:`~koopman_graph.nn.PredictedTopologyHead`
  remains the ``dense_mlp`` power-user path with an :math:`N` ceiling of
  64. Distinct from ``learn_topology="self_adaptive"`` /
  :class:`~koopman_graph.nn.AdaptiveAdjacency` (static Graph WaveNet);
  the two cannot be combined when ``topology_head`` is not ``none``.
  When ``recursive_training`` is true, omitted ``future_topologies``
  uses predicted :math:`\hat A_{t+1}=g_\phi(z_t)` as sigmoid weights on
  the candidate COO (``topology_policy="hold_last"`` restores the 0.14
  control). Evaluate on dynamic sequences injects oracle futures only
  when that recursive path is off. See
  ``examples/50_graph_state_closure.ipynb`` (wiring check versus
  hold-last; not a learned-forecast claim). Narrative guide:
  :doc:`graph_dynamics`.

Dynamics
~~~~~~~~

* Discrete ``KoopmanOperator`` with soft modes (``dense``, ``odo``),
  structural guarantees (``schur``, ``dissipative``, ``lyapunov``), or
  structure-preserving maps (``row_stochastic``, ``doubly_stochastic``,
  ``symplectic``; those maps constrain parameterized :math:`K`, not
  decoded :math:`x`)
* Opt-in decoded-space constraint heads
  (:class:`~koopman_graph.nn.MassConservingDecoder`,
  :class:`~koopman_graph.nn.PositivityDecoder`,
  :class:`~koopman_graph.nn.LinearConservingDecoder`). Softmax /
  affine mass, positivity, and :math:`Cx=c_0` act after decode.
  Latent symplectic :math:`K` alone does not conserve decoded mass
  (``Greydanus2019HNN``). IEEE-118 remains Laplacian diffusion, not
  AC power flow
* Networked ``GraphKoopmanOperator`` (``koopman="graph"``) with
  self/neighbor coupling
  :math:`I_N \otimes K_{\mathrm{self}} + \widehat{A} \otimes K_{\mathrm{nbr}}`
  by default (``koopman_filter_degree=1``). Degree :math:`P>1` adds
  :math:`\sum_{k=2}^{P}\widehat{A}^{k}\otimes K_{k}` (monomial; extra hops
  globally shared; factory ``koopman_filter_degree``). Factorized blocks
  inspired by compositional / networked Koopman constructions;
  cross-topology transfer is measured, not assumed — see
  :doc:`limitations` and ``examples/37_cross_topology_transfer.ipynb``.
  Degree-:math:`P` ablation and Kronecker-versus-dense spectrum live in
  ``examples/38_operator_factorization_ablation.ipynb`` and
  ``examples/49_multi_hop_factorization.ipynb``; they do not attribute
  the path-diffusion joint-LS gap to hop order.
  ``.spectrum`` auto-routes Kronecker-sum reduction when eligible (shared
  self; ``symmetric`` / ``random_walk``; any monomial :math:`P\ge 0`); see
  :doc:`limitations`
  (Scale) for distributed Arnoldi, dense fall-backs, and inverse ceilings.
  ``fit`` warns when encoder hops exceed ``filter_degree``; encoder
  mixing does not compensate for a one-hop :math:`K`
* Matrix-free linear-operator protocol
  (:class:`~koopman_graph.operators.LinearOperatorProtocol`) wrapping
  a degree-:math:`P\\ge 2` graph polynomial and the existing one-tap
  ``matrix_free`` helpers. Dense assembly is refused above
  :data:`~koopman_graph.operators.MAX_DENSE_LINEAR_OPERATOR_SIZE`.
  Trainer DDP does not shrink that representation. Leading eigpairs
  are Arnoldi Ritz values, not Kronecker :math:`\\operatorname{eig}(B(\\lambda))`.
  :math:`10^{5}`-node scaling is not a release gate. See
  :doc:`matrix_free`.
* Pairwise ``adjacency`` modes on graph / continuous-graph operators:
  ``"symmetric"`` (default), ``"random_walk"``
  (:math:`D_{\mathrm{out}}^{-1}A`), and ``"dual_random_walk"`` (forward plus
  reverse); factory keyword ``koopman_adjacency``; format-1
  ``config.adjacency``. Hypergraph operators do not expose ``adjacency``;
  use ``incidence_mode`` instead
* Hypergraph ``HypergraphKoopmanOperator`` (``koopman="hypergraph"``)
  with ``incidence_mode``
  ``"zhou_symmetric"`` (default), ``"forward_random_walk"``, or
  ``"dual_random_walk"`` (factory ``koopman_hypergraph_incidence_mode``;
  dual exposes ``K_hedge`` / ``K_bwd``)
* Relational multiplex / typed
  ``HeteroGraphKoopmanOperator`` (``koopman="hetero_graph"``):
  :math:`K_{\mathrm{eff}} = I\otimes K_{\mathrm{self}} + \sum_r
  \widehat{A}_r \otimes K_r` (typed: block-diagonal per-type self).
  Optional ``relation_tying="basis"``. Dense :math:`N\cdot d` spectrum /
  inverse ceiling unchanged — see :doc:`limitations`
* Global/local non-stationary discrete operator
  (``GlobalLocalKoopmanOperator``; ``koopman="global_local"``)
* Switched and mixture discrete maps (``koopman="switched"`` /
  ``"mixture"``; piecewise-linear or softmax mixture of LTI modes,
  not a parameter interpolant :math:`K(\\mu)`)
* Opt-in parametric interpolant (``koopman="parametric"``;
  :class:`~koopman_graph.operators.ParametricKoopmanOperator`)
  :math:`K(\\mu)=\\sum_j \\alpha_j(\\mu) K_j` with RBF or simplex
  weights. Convex combinations preserve dense / row-stochastic /
  doubly-stochastic factors; symplectic and other structural mixes
  raise. Export refuses the interpolant
  (``Macesic2018Nonautonomous``)
* Hodge / Laplacian-structured networked operator (``koopman="hodge"``)
* Degree-specific cochain maps
  (:class:`~koopman_graph.operators.CochainKoopmanOperator`) on a
  static signed :math:`B_1` for :math:`k\\le 1`. Not a factory kind;
  default ``koopman=None`` stays ``"pernode"``. Face latents are
  stored, not evolved. Not TopologicX parity
* Order-2 TDL teaching path
  (:func:`~koopman_graph.nn.order2_cochain_teaching`) binding that
  cochain operator to a filled triangle. Cell-complex degree
  :data:`~koopman_graph.nn.MAX_CELL_COMPLEX_DEGREE` (3) is the
  ceiling. Sheaf restriction maps stay learned-optional. Not
  ecosystem parity
* Combinatorial Hodge split of stored Koopman mode shapes
  (:func:`~koopman_graph.analysis.hodge_decompose_modes`).
  Analysis-only gradient / curl / harmonic blocks on a static
  signed :math:`B_1` (:math:`k\\in\\{0,1\\}`). Not physical
  circulation, not a factory kind, and not ``koopman="hodge"``.
  Teaching notebook: ``examples/52_cochain_hodge_modes.ipynb``
* Equivariant block operator
  (:class:`~koopman_graph.operators.EquivariantKoopmanOperator`;
  vector channels are multiples of :math:`I_3`; optional
  :math:`l=2` channels are multiples of :math:`I_5`). Not a
  factory kind and not a molecular MD stack. Default encode may
  still project to invariants
* Continuous-time ``ContinuousKoopmanOperator``
  (``dynamics_mode="continuous"``), irregular timestamps, and
  ``predict_at``
* Continuous networked generator ``ContinuousGraphKoopmanOperator``
  (``koopman="graph"`` + ``dynamics_mode="continuous"`` or
  ``koopman="continuous_graph"``; ``adjacency`` modes match discrete
  ``GraphKoopmanOperator``). ``.spectrum`` uses the same Kronecker
  eligibility as discrete graph when applicable; dense ``N·d``
  matrix-exp (:math:`\Phi=\exp(\Delta t\, L_{\mathrm{eff}})`) remains a
  separate ceiling — prefer modest ``N`` or ``sparsity="block_diagonal"``
  self-only shortcut for advances (see :doc:`limitations`)
* Continuous hetero generators on ``koopman="hetero_graph"`` +
  ``dynamics_mode="continuous"`` (dense stacked :math:`\Phi` cost; see
  :doc:`limitations`)
* Opt-in ``dynamics_mode="stochastic"`` — discrete linear map plus learned
  diagonal process noise (not a continuous-time SDE).
  :class:`~koopman_graph.operators.DriftDiffusionKoopman` is a separate
  Euler–Maruyama / Yosida stepper, not that factory string
* :class:`~koopman_graph.uq.JointCoverageSpec` names the conformal
  estimand (default ``per_node_marginal``). Simultaneous / event
  targets are named but not implemented. Proper scores
  :func:`~koopman_graph.uq.gaussian_crps`,
  :func:`~koopman_graph.uq.gaussian_nll`, and
  :func:`~koopman_graph.uq.energy_score` are not coverage certificates
* Innovation-whiteness report
  (:func:`~koopman_graph.analysis.markov_closure_report`) and a
  convolution MVP
  (:class:`~koopman_graph.analysis.FiniteMemoryKoopman`). Not
  Mori–Zwanzig identification, not HAVOK, and not
  :class:`~koopman_graph.nn.delay.DelayEmbeddingEncoder`. Not a
  factory kind
* Continuous ``koopman_parameterization="auxiliary_spectral"`` — state-dependent
  ``generator_at(z)`` / instantaneous spectrum (Lusch-style; locally linear,
  not a fixed global matrix). Prefer delay embeddings first for continuous-
  spectrum phenomenology; see ``examples/20_continuous_spectrum_auxiliary_network.ipynb``
* Additive control and optional bilinear / control-affine terms
  (``control_mode="bilinear"``)

Forecasting and training
~~~~~~~~~~~~~~~~~~~~~~~~

* Multi-step rollout from a single initial state
* Consistency losses (forward / backward), optional eigenvalue regularization,
  fit-time PIKN-style Lie / PINN-style PDE residual terms, optional
  :math:`L_1` / smoothed :math:`L_p` Koopman sparsity, an optional
  worst-case (:math:`L_{\infty}`\-style) reconstruction term (robust training
  only — not a generalization bound), and optional topology-blind VAMP-2
  precursor weight (``LossWeights.vamp2`` /
  :func:`~koopman_graph.baselines.vamp2.vamp2_score`; not GraphVAMPnets)
* LR schedulers, per-term loss history, ``MultiTrajectory`` fit, and
  windowed mini-batching
* Temporal train/val/test splits and per-horizon MAE, RMSE, and MAPE via
  ``koopman_graph.metrics.evaluate_forecast``
* Measured cross-topology transfer via
  :func:`~koopman_graph.analysis.evaluate_topology_transfer`
  (structured report; mandatory ``pernode`` control; negative advantage
  allowed — see :doc:`limitations` and
  ``examples/37_cross_topology_transfer.ipynb``)
* Frozen identification config / report types and closed-form ridge /
  TLS / constrained least-squares solvers under
  :mod:`koopman_graph.identification` (off root ``__all__``).
  ``GraphKoopmanModel.fit(..., identification=IdentificationConfig(...))``
  is opt-in; default ``identification=None`` is the Adam path. Discrete
  dense per-node operators only. The report records latent one-step /
  short-rollout mean squared error (MSE) and :math:`\rho(K)`, not an
  invariance or ResDMD certificate.
  :func:`~koopman_graph.identification.subspace_invariance_report` and
  ``evaluate(..., include_invariance=True)`` report finite-sample
  projection leakage :math:`\eta` on a truncated-SVD encoding span
  (discrete dense per-node :math:`K` only). That ratio is not a
  Haseli–Cortés invariance-proximity certificate.
  :func:`~koopman_graph.identification.select_resdmd_gated` compares
  pre-scored dictionaries by train one-step mean squared error (MSE)
  after optionally dropping candidates whose max finite-dictionary
  ResDMD residual exceeds :math:`10^{-2}`.
  ``IdentificationConfig.gate_resdmd=True`` fills the report
  ``spectral`` block; it does not abort ``fit``.
  :class:`~koopman_graph.training.ResDMDFitCallback` ``mode="gate"``
  raises at fit end when the observed max residual exceeds that
  cutoff. Default callback mode is ``"observe"``. None of these is a
  certified infinite-dimensional residual bound.
  :func:`~koopman_graph.identification.identify_sparse_graph_factors`
  selects sparse :math:`K_{\\mathrm{self}}` / :math:`K_{\\mathrm{nbr}}`
  on frozen encodings (STLSQ or teaching group-lasso, then unpenalized
  refit; related literature ``Pan2021SparseSubspace``, not that paper's
  multi-task EDMD pruning). Existing
  :class:`~koopman_graph.losses.KoopmanSparsityLoss` and
  :func:`~koopman_graph.analysis.identify_sparse_dynamics` still ship.
  :func:`~koopman_graph.identification.select_latent_rank` scores
  truncated-SVD ranks of frozen encodings (in-tree VAMP-2, ResDMD
  elbow, or stability-penalized held-out MSE). It is not Ray Tune /
  AutoML for encoder ``latent_dim``
  (:mod:`koopman_graph.tuning` remains caller-owned HPO).
  Teaching notebooks: ``examples/48_identification_invariance.ipynb``
  and ``examples/53_latent_rank_selection.ipynb``. See
  :doc:`identification`.
* :mod:`koopman_graph.benchmark` holds frozen
  :class:`~koopman_graph.benchmark.ExperimentManifest` records (schema
  ``benchmark_manifest_v1``; JSON always, YAML via PyYAML / ``[cli]``).
  Teaching GNN methods require non-empty ``deviations``. Dataset SHA-256
  mismatch is rejected. The CLI ``benchmark run`` / ``verify`` commands
  write identity-bound summaries (dataset SHA-256 and a canonical
  digest). They do not train models, download telemetry, or host a
  LibCity / BasicTS leaderboard. Tracked smoke YAML under
  ``benchmarks/v0.15/`` is what default CI verifies. See
  :doc:`benchmarks` and ``examples/47_benchmark_manifest.ipynb``.

Training performance
~~~~~~~~~~~~~~~~~~~~

Internal training-path reuse (same scientific defaults; see
:doc:`architecture` and :doc:`limitations`):

* Shared sequence latents (``SequenceLatentCache``) when multiple pair
  losses share a window — each timestep is encoded once per
  ``compute_training_loss`` evaluation
* Networked dense inverse reuse for static topology; DiffConv support
  cache (``clear_support_cache``); hypergraph Zhou :math:`\hat{H}` cache
  (``clear_hyperedge_cache``)
* Continuous dense :math:`\Phi` / :math:`L_{\mathrm{eff}}` reuse within
  one training-loss evaluation
* Shared one-step predictions across reconstruction / PDE / worst-case
  terms when those losses are active together
* Opt-in multi-graph ``Batch`` collate: ``fit(..., batch_graphs=True)``
  vectorizes the existing ``MultiTrajectory`` loop (reconstruction and
  forward consistency). Default per-sequence averaging is unchanged.
  Multi-topology training did not require this flag.
* Opt-in CUDA automatic mixed precision: ``use_amp=True`` on
  ``GraphKoopmanModel.fit`` / ``run_fit_loop`` (FP32 fallback on
  CPU/MPS)
* Hierarchical ``pool_schedule="hold_perm"`` to amortize TopK / SAG
  pooling across a sequence
* Ephemeral structural ``K`` / ``L`` and low-rank bilinear assemblies;
  learned-topology materialize at most once per top-level
  ``forward`` / rollout-origin encode; multi-start rollout encodes each
  distinct origin once

Distributed training (optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Optional multi-process / multi-GPU *trainer orchestration* around the same
scientific fit loop (``run_fit_loop`` / epoch helpers /
``compute_training_loss``). Import from
:mod:`koopman_graph.distributed` (power-user; not on root ``__all__``).
This is **not** the operator flag ``sparsity="distributed"`` (matrix-free
inverse / spectrum; see :doc:`faq` and :doc:`limitations`).

* Native PyTorch DDP via :func:`~koopman_graph.distributed.run_ddp_fit_loop`
  or ``GraphKoopmanModel.fit(..., strategy="ddp")`` (core install;
  typically launched with ``torchrun``). Homogeneous and hetero
  (``HeteroData`` / ``HeteroGraphSnapshotSequence``) models compose on
  this path; ``find_unused_parameters`` defaults to ``True`` for RelGraph
  hetero stacks
* Lightning Fabric via :func:`~koopman_graph.distributed.fit_with_fabric`
  (requires ``pip install "koopman-graph[lightning]"``); accepts hetero
  sequences
* Optional Lightning ``Trainer`` sugar via
  :class:`~koopman_graph.distributed.KoopmanLightningModule` (same
  ``[lightning]`` extra). Prefer Fabric / native DDP when you need full
  loss schedules, :class:`~koopman_graph.distributed.DistributedWindowSampler`,
  or the shared epoch driver. The module composes a
  :class:`~koopman_graph.model.GraphKoopmanModel`, accepts batches of
  :class:`~koopman_graph.data.GraphSnapshotSequence` /
  :class:`~koopman_graph.data.HeteroGraphSnapshotSequence` (or a list
  thereof), and exports format-1 checkpoints with
  ``export_format1_checkpoint``
* Rank-aware window sampling
  (:class:`~koopman_graph.distributed.DistributedWindowSampler`) and
  trajectory sharding helpers (including hetero)
* Optional Ray Train *model* DDP via
  :func:`~koopman_graph.distributed.run_ray_train_fit_loop`
  (requires ``pip install "koopman-graph[ray]"`` or ``[distributed]``;
  wraps Ray Train ``TorchTrainer`` around the same scientific fit loop).
  Prefer native DDP / Fabric unless you already standardize on Ray Train.
  Multi-node Ray Train is opt-in (``KOOPMAN_GRAPH_MULTINODE=1``); default
  CI stays single-process — see :doc:`limitations`
* Optional Ray parallel ensemble member fits via
  :func:`~koopman_graph.distributed.fit_ensemble_with_ray` or
  ``EnsembleGraphKoopmanModel.fit(..., parallel_backend="ray",
  member_factory=...)`` (same ``[ray]`` extra). Sequential ensemble fit
  remains the default. This does **not** change UQ coverage guarantees —
  members stay independent fits. Do **not** confuse ensemble Ray with
  :func:`~koopman_graph.distributed.run_ray_train_fit_loop`
* Example script (outside ``nbmake`` CI)::

    torchrun --standalone --nproc_per_node=2 \\
      examples/scripts/ddp_fit_torchrun.py

  See ``examples/scripts/README.md``. Multi-process smoke tests are
  opt-in (``KOOPMAN_GRAPH_DISTRIBUTED_TESTS=1``); default PR CI does not
  require multi-node hardware.

Ray Tune HPO is supported via power-user helpers under
:mod:`koopman_graph.tuning` (``fit_history_metrics``, ``run_ray_tune``,
optional ``example_*`` smoke scaffolds) and the examples-only script
``examples/scripts/ray_tune_koopman_example.py``. Search spaces remain
**user-defined / script-owned**; the library is not an AutoML product.
Optuna is **examples-only** (no library Optuna API). Optional
``pip install "koopman-graph[dask]"`` activates
:mod:`koopman_graph.distributed.dask_prep` helpers
(``materialize_sequences``, ``materialize_window_index_list``) for offline
prep. The library does **not** import Dask on the training path; trainers
remain native DDP / Fabric / Ray Train / Ray ensemble (see :doc:`faq`).
Distributed data-parallel training does **not** reduce dense
:math:`N\cdot d` operator ceilings — see :doc:`limitations` (Scale).

Analysis
~~~~~~~~

* ``KoopmanSpectrum`` / ``compute_spectrum`` with mode decoding helpers
  and optional :class:`~koopman_graph.spectrum_types.SpectralDiagnostics`
  (:math:`\kappa(V)`, Wilkinson :math:`\kappa_i`, departure from
  normality, discrete Nyquist :math:`1/(2\Delta t)` in cycles per unit
  time, aliasing flags). Not a finite-horizon :math:`\|K^{k}\|` bound.
  See :doc:`spectral_diagnostics` and
  ``examples/51_spectral_diagnostics.ipynb``
* Sliding-window spectral-gap monitor
  (:func:`~koopman_graph.analysis.monitor_critical_transition`) — positive
  rate means the gap shrank; **not** a Ghosh-grade certificate
  (``Ghosh2025``). See :doc:`criticality` and
  ``examples/54_criticality_monitor.ipynb``
* Networked graph / continuous-graph ``.spectrum`` paths: distributed
  Arnoldi (discrete / multiplex hetero / Zhou-symmetric hypergraph /
  continuous-graph generator factors), Kronecker-sum exact reduction
  when eligible, or dense assembled eigendecomposition — see
  :doc:`limitations` (Scale)
* ``attribute_mode_energy`` / ``ModeEnergyAttribution`` — interpretive
  type / relation energy fractions on assembled ``K_eff`` (not causal;
  not a ResDMD residual on relation-attributed modes)
* ``explain_representation`` / ``RepresentationExplanation`` —
  representation-level node / edge / feature masks on a homogeneous
  snapshot for targets ``latent``, ``one_step_forecast``, or
  ``reconstruction``. Default algorithm ``gnn_explainer`` uses PyG
  GNNExplainer (``Ying2019GNNExplainer``); optional
  ``algorithm="integrated_gradients"`` uses Captum via the ``[explain]``
  extra (``Sundararajan2017IntegratedGradients``). Masks are
  **interpretive** and **non-causal**; they are **not**
  ``ModeEnergyAttribution`` and **not** ResDMD residuals. Homogeneous
  graphs including delay embeddings and additive control; hetero /
  hypergraph / adaptive remain rejected — see :doc:`limitations`
* ``spectral_residuals`` / ``SpectralResidualReport`` — held-out data-driven
  residuals and ``trustworthy_mask()``; optional
  ``plot_spectrum(..., annotate_untrustworthy=True)``. Diagnostic in the
  learned observable space, **not** a certified ResDMD bound
  (``ColbrookTownsend2023ResDMD``, ``Colbrook2023ResidualDMD``)
* Finite-dictionary ResDMD MVP
  (:func:`~koopman_graph.analysis.resdmd`,
  :class:`~koopman_graph.analysis.ResDMDReport`) optionally attached to
  ``evaluate(..., include_resdmd=True)`` and
  :class:`~koopman_graph.training.ResDMDFitCallback`, plus finite-matrix
  resolvent-norm grid
  (:func:`~koopman_graph.analysis.resolvent_norm_grid`) — **not**
  infinite-dimensional certified pseudospectra; see
  ``examples/40_resdmd_pseudospectra.ipynb``
* Finite-sample subspace invariance leakage
  (:func:`~koopman_graph.identification.subspace_invariance_report`,
  ``evaluate(..., include_invariance=True)``) — truncated-SVD
  projection residual on encoded samples; **not** a Haseli–Cortés
  certificate (``HaseliCortes2023``); discrete dense per-node
  :math:`K` only
* Residual-aware dictionary selection
  (:func:`~koopman_graph.identification.select_resdmd_gated`) and
  opt-in ``ResDMDFitCallback(mode="gate")`` — drop or reject
  finite-dictionary ResDMD-polluted RMSE-only winners; **not** a
  certified residual bound; default callback mode remains
  ``"observe"``
* Sparse graph-factor identification
  (:func:`~koopman_graph.identification.identify_sparse_graph_factors`)
  — STLSQ or teaching group-lasso on frozen
  :math:`K_{\\mathrm{self}}` / :math:`K_{\\mathrm{nbr}}` with
  unpenalized refit. Distinct from latent SINDy
  (:func:`~koopman_graph.analysis.identify_sparse_dynamics`) and from
  :class:`~koopman_graph.losses.KoopmanSparsityLoss`; those tools still
  ship. Dual random-walk and :math:`P>1` polynomial hops are out of
  scope.
* Latent-rank selection
  (:func:`~koopman_graph.identification.select_latent_rank`) — VAMP-2,
  ResDMD elbow, or stability-penalized held-out MSE on a truncated-SVD
  grid of frozen encodings. Not Ray Tune AutoML for encoder
  ``latent_dim``; ``[msm]`` / deeptime is an optional VAMP-2
  cross-check (``deeptime2021``).
* Protocol-locked experiment manifests
  (:class:`~koopman_graph.benchmark.ExperimentManifest`) — schema
  ``benchmark_manifest_v1`` with dataset SHA-256, ≥3 seeds, mandatory
  controls, and teaching-port ``deviations``. CLI ``run`` / ``verify``
  hash protocol identity; they do not train and are not SOTA. Default
  CI verifies hashed synthetics under ``benchmarks/v0.15/``, not full
  METR-LA.
* Kronecker dispersion
  (:func:`~koopman_graph.analysis.graph_dispersion`) and Granger-style
  causal MVP (:func:`~koopman_graph.analysis.granger_latent_influence`;
  assumption-laden, **non-interventional**)
* Labeled synthetic SCM interventions
  (:func:`~koopman_graph.analysis.teaching_three_node_scm`,
  :func:`~koopman_graph.analysis.recover_synthetic_interventional_edges`).
  Recovers a known do-edge on the teaching fixture only — not Granger
  and not field-data discovery
* 0-dimensional persistence
  (:func:`~koopman_graph.analysis.persistence_diagram_0d`; optional
  ``[tda]``) — not a TDA library
* Long-horizon statistics via ``koopman_graph.statistics`` (power-user):
  Welch PSD (``Welch1967``), ``spectral_distance``,
  ``invariant_measure_distance``, Rosenstein
  ``largest_lyapunov_exponent`` (``Rosenstein1993Lyapunov``;
  embedding-sensitive; :math:`O(T^{2})`), and ``LongHorizonReport``
* Dynamical similarity and anomaly utilities via
  ``koopman_graph.analysis``
* ``identify_sparse_dynamics`` (SINDy / STLSQ on learned latents — not
  physical governing equations; distinct from
  :class:`~koopman_graph.baselines.GEDMDBaseline`)
* ``koopman_spectral_clustering`` (node communities from Koopman modes)
* ``estimate_coupling_from_snapshots`` (topology / coupling diagnostics)
* ``plot_spectrum`` for unit-disk / data-zoom views

Control, adaptation, and observation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``koopman_graph.adaptation.RecursiveKoopmanAdapter`` / ``adapt_step`` for
  online RLS updates
* ``KoopmanObserver`` for latent Kalman filtering / imputation under
  ``observation_masks``
* ``JointStateTopologyObserver`` for Kalman filtering plus group-sparse
  :math:`K_{\mathrm{self}}` / :math:`K_{\mathrm{nbr}}` write-back on the
  observed COO (or dense RLS on per-node :math:`K`). Homomorphism
  claims require :class:`~koopman_graph.nn.SeparableDictionaryEncoder`
  and a one-tap graph operator (``Peng2026KoopmanGKFA``); default GNN
  encoders raise when ``claim_homomorphism=True``. This is not
  Koopman-GKFA, does not infer a new edge set, and is not on the root
  façade.
* ``koopman_graph.env.GraphKoopmanEnv`` / ``to_latent_env`` for Gymnasium
  closed-loop control
* ``koopman_graph.mpc.KoopmanMPC`` for additive-control receding-horizon
  QP and bilinear sequential-linearization (``[mpc]`` / OSQP; local
  decoder-linearization guarantees; optional conformal output-constraint
  tightening)
* ``koopman_graph.mpc.TubeKoopmanMPC`` for additive discrete residual-tube
  tightening from conformal quantiles or ensemble radii
  (``[mpc]`` / OSQP). ``evaluate`` reports constraint-violation rate,
  feasibility rate, and cost — not tracking MSE alone, not recursive
  feasibility, and not a Lyapunov closed-loop certificate
  (``Zhang2022TubeMPC``)
* Hybrid physics observables: Laplacian, nodewise graph-gradient magnitude,
  graph curvature (:math:`L_{\mathrm{sym}}^2 x`), polynomial dictionaries, or
  custom lifting callables. Residual losses are soft penalties, not
  symplectic/Hamiltonian structure guarantees or a PIKE/SPIKE implementation.

Uncertainty quantification (power-user)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``koopman_graph.uq.EnsembleGraphKoopmanModel`` — deep ensembles with
  empirical ``predict_interval`` mean / quantile bounds (not on root
  ``__all__``)
* ``koopman_graph.uq.LatentGaussianKoopmanUQ`` — linear-Gaussian latent
  forecast with closed-form covariance propagation and optional Kalman
  refinement (not DPK)
* ``koopman_graph.probabilistic`` — variational encoder weights plus
  linear latent :math:`K` (distinct from Laplace factor UQ)
* ``koopman_graph.uq.ConformalKoopmanUQ`` — split and adaptive (ACI)
  conformal intervals; marginal coverage under exchangeability (approximate
  under temporal dependence). Scores: ``"aggregate"``, legacy ``"per_node"``
  (max-pool over nodes), and ``"node_wise"`` (per-node marginal widths;
  optional ``neighbor_smoothing`` following DAPS-style diffusion,
  ``Zargarbashi2023ConformalGNN``). Calibration payload kind
  ``ConformalKoopmanUQ.calibration.v2``
* ``koopman_graph.uq.BayesianKoopmanUQ`` — diagonal Laplace posterior over
  linear Koopman factors with seeded ``sample_forecast`` intervals (not a
  BNN over encoder weights; not full DPK / :math:`K^{2}`\ VAE)
* See notebooks ``21_uncertainty_quantification.ipynb`` and
  ``30_conformal_uncertainty.ipynb``

Hierarchical / multi-resolution (power-user)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``koopman_graph.hierarchical.HierarchicalGraphKoopmanModel`` — TopK
  (optional SAG) pool → composed ``GraphKoopmanModel`` on the coarse graph →
  scatter-unpool; ``predict(..., resolution=...)`` for coarse vs fine
* Coarse-level forecasting with unpooling — **not** P-K-GCN spatiotemporal
  super-resolution; graph spectra use the pooled topology
* See notebook ``23_hierarchical_multiresolution.ipynb`` for in-sample RMSE,
  fine-grid snapshots, and a dense-``K`` ``spectrum`` API demo on a grid

Research tooling
~~~~~~~~~~~~~~~~

* Classical baselines via ``koopman_graph.baselines``: ``DMDBaseline``,
  ``EDMDBaseline`` (polynomial / RBF / kernel dictionaries; kernel path is
  :math:`O(T^2)` — small/medium ``T``; Nyström / random-feature
  approximations available), ``MpEDMDBaseline`` (measure-preserving EDMD
  via a Gram-weighted Procrustes polar factor; ``Colbrook2023mpEDMD``;
  not a Euclidean :math:`K_{\\mathrm{eff}}` certificate),
  ``GEDMDBaseline`` (generator EDMD from supplied
  :math:`dx/dt`; ``Klus2020gEDMD``; not
  derivative-mode SINDy; irregular timestamps do not create
  :math:`L`),
  ``HankelDMDBaseline`` (Hankel-DMD on delay-embedded flattened
  states; ``Arbabi2017HankelDMD``; not
  :class:`~koopman_graph.nn.delay.DelayEmbeddingEncoder`),
  ``HAVOKBaseline`` (teaching HAVOK on the same delay rows;
  ``Brunton2017HAVOK``; autonomous ``predict`` uses
  :math:`u=0`),
  ``DMDcBaseline``, ``FBDMDBaseline``,
  ``TLSDMDBaseline``, ``OptDMDBaseline``, ``StreamingDMDBaseline``,
  ``MRDMDBaseline``, and ``UlamTransferOperatorBaseline``. Truncated-SVD
  ``rank`` accepts ``None``, a positive integer, or ``"auto"``
  (Gavish–Donoho median threshold, ``GavishDonoho2014``); fitted
  ``selected_rank`` is recorded
* Topology-blind VAMP-2 precursor helpers
  (``vamp2_score`` / ``vamp2_loss``; optional ``[msm]`` for deeptime
  oracle tests)
* Contact-graph GraphVAMP teaching baseline
  (:class:`~koopman_graph.baselines.GraphVAMPBaseline`) with synthetic
  molecular oracles under :mod:`koopman_graph.datasets.molecular` and
  optional deeptime interop (:mod:`koopman_graph.interop`; ``[msm]`` /
  ``[md]``) — teaching / diagnostic, not GraphVAMPnets production or a
  PyEMMA replacement. Packaged alanine-dipeptide teaching card via
  :func:`~koopman_graph.datasets.molecular.alanine_dipeptide_card`
  (not Folding@home; see :doc:`limitations`)
* Lightweight STGCN / DCRNN / Graph WaveNet references plus teaching
  AGCRN / MTGNN / STGODE / GraphCast / STAEformer-class / spatiotemporal
  SSM ports in ``koopman_graph.baselines.gnn`` (``ForecasterProtocol``
  deviation tables; not dedicated-library SOTA)
* Protocol-matched LibCity / BasicTS adapters
  (:class:`~koopman_graph.baselines.gnn.LeaderboardProtocol`) — not SOTA
* Restricted portable export (:mod:`koopman_graph.export`; fixed-topology
  discrete homogeneous MVP) and in-tree FedAvg
  (:mod:`koopman_graph.federated`; not DP)
* Graphon-sampled adjacency and a dense teaching estimator
  (:func:`~koopman_graph.operators.sample_graphon_adjacency`,
  :func:`~koopman_graph.operators.estimate_graphon`;
  :doc:`graphon`) — constant / product oracle recovery on aligned
  graphs; not sparse-graph graphon theory and not a transferability
  certificate
* Benchmark datasets and Jupyter tutorials under ``examples/``
* Model ``save`` / ``load`` checkpoints (default ``safetensors_v1``
  directory or ``.kgckpt`` / ``.zip`` bundle; ``legacy_pt`` pickle escape
  hatch); ≥90% coverage enforced in CI. See the repository
  ``SECURITY.md`` for load trust boundaries.

Experiment tracking (fit callbacks)
-----------------------------------

Observe-only hooks on the functional fit path (single-process
``GraphKoopmanModel.fit`` / :func:`~koopman_graph.training.run_fit_loop`):

* :class:`~koopman_graph.FitCallback` protocol and
  :class:`~koopman_graph.NoOpFitCallback` (root-exported)
* In-tree adapters under :mod:`koopman_graph.tracking` (power-user; not on
  root ``__all__``): :class:`~koopman_graph.tracking.CsvFitLogger` (stdlib
  CSV) and :class:`~koopman_graph.tracking.TensorBoardFitLogger` (requires
  peer ``pip install tensorboard``)
* Callbacks must not mutate model parameters; default ``callbacks=None``
  preserves prior ``FitHistory`` behavior
* ``fit(..., strategy="ddp")`` does not accept ``callbacks`` yet — use
  single-process fit or Lightning loggers for distributed runs
* Lightning ``Trainer`` users should attach Lightning loggers to
  :class:`~koopman_graph.distributed.KoopmanLightningModule`; this release
  does not add a second FitCallback stack on Trainer
* Weights & Biases / MLflow are intentionally not core dependencies —
  see :doc:`faq` and ``examples/tracking/wandb_mlflow_callback.py``

Hyperparameter search helpers
-----------------------------

Power-user package :mod:`koopman_graph.tuning` (not on root ``__all__``):

* :func:`~koopman_graph.tuning.fit_history_metrics` — flatten a
  :class:`~koopman_graph.training.FitHistory` into scalar floats for
  Tune / Optuna reporters
* :func:`~koopman_graph.tuning.run_ray_tune` — thin Ray Tune façade
  (lazy import; requires ``pip install "koopman-graph[ray]"``)
* :func:`~koopman_graph.tuning.example_lr_loguniform_space` /
  :func:`~koopman_graph.tuning.example_lr_latent_dim_space` — *example*
  smoke scaffolds only (not scientific defaults)

This is **not** an AutoML product. Searchable hyperparameters and ranges
remain caller-owned. Optuna is **examples-only** (no library Optuna API).
See ``examples/scripts/ray_tune_koopman_example.py`` and
``examples/scripts/README.md``.

Stability mode selection
------------------------

Use ``dense`` or ``odo`` when you want a soft prior (``odo`` bounds
``ρ(K)`` via the operator 2-norm but lacks a strict ε-interior
certificate; continuous ``odo`` needs eigenvalue loss on the true
spectrum). Choose ``schur``, ``dissipative``, or ``lyapunov`` when you
need eigenvalues forced inside the unit disk. See
``examples/11_long_horizon_stability.ipynb`` versus
``examples/08_loss_stability.ipynb``, and the stability section in
:doc:`quickstart`.

Built-in datasets
-----------------

.. list-table::
   :header-rows: 1
   :widths: 32 18 50

   * - Benchmark
     - Domain
     - Description
   * - ``SyntheticDynamicGraphBenchmark``
     - Synthetic
     - Laplacian diffusion on path/ring graphs
   * - ``GridDynamicGraphBenchmark``
     - Synthetic
     - Laplacian diffusion on a 4-connected 2D lattice
   * - ``AnisotropicAdvectionGridBenchmark``
     - Synthetic
     - Directional advection with asymmetric edge weights
   * - ``EpidemicNetworkBenchmark``
     - Epidemic
     - Networked SIR on ring / small-world / custom graphs
   * - ``ContactEpidemicBenchmark``
     - Epidemic (cache)
     - SocioPatterns primary-school contacts (fetch-script + SHA256; CC-BY-NC-SA)
   * - ``Lorenz96GraphBenchmark``
     - Chaotic ODE
     - Lorenz-96 on a ring graph
   * - ``KuramotoSivashinskyBenchmark``
     - Chaotic PDE
     - 1D KS on a path/ring discretization
   * - ``CylinderWakeBenchmark``
     - Fluids (cache)
     - Hopf/Stuart–Landau cylinder-wake teaching surrogate
   * - ``IEEE118DynamicBenchmark``
     - Power systems
     - IEEE 118-bus topology with simulated voltage/load dynamics;
       typed helpers (``load_typed_topology`` / ``generate_typed``,
       generator/load/slack) for hetero demos — simulated-dynamics
       disclaimer applies
   * - ``MetrLaTrafficBenchmark``
     - Traffic
     - METR-LA sensor graph with cached speed snapshots
   * - ``PemsBayTrafficBenchmark``
     - Traffic
     - PEMS-BAY speeds (325 sensors; fetch-script + SHA256)
   * - ``PemsTrafficBenchmark``
     - Traffic
     - PEMS03/04/07/08 flows (``variant=``; fetch-script + SHA256)

Related pages
-------------

* :doc:`quickstart` — runnable train/predict walkthrough
* :doc:`limitations` — consolidated scope boundaries and “when not to use”
* :doc:`data` — FAIR dataset cards (PEMS / contact epidemic / acquisition)
* :doc:`tutorials` — notebook gallery
* :doc:`architecture` — public vs power-user API layers
* :doc:`benchmarks` — identity-bound manifests (not trained scores)
* :doc:`identification` — opt-in closed-form :math:`K` versus Adam
* :doc:`spectral_diagnostics` — Nyquist, :math:`\kappa(V)`, and aliasing
* :doc:`graph_dynamics` — predicted topology versus hold-last
* :doc:`matrix_free` — ``LinearOperatorProtocol`` (not trainer DDP)
* :doc:`criticality` — spectral-gap monitor (not a certificate)
* :doc:`time_conditioning` — :math:`(\mu, t, u)` records and
  parametric :math:`K(\mu)`
* :doc:`api` — module reference
* Repository ``README.md`` — short landing page with a Related software niche map
