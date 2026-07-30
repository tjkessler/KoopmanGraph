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
* ``HypergraphEncoder`` / ``HypergraphDecoder`` with incidence
  ``hyperedge_index`` (``koopman="hypergraph"``)
* Heterogeneous / multiplex RelGraph peers
  (``RelGraphEncoder`` / ``RelGraphDecoder``) with
  ``koopman="hetero_graph"`` on ``HeteroGraphSnapshotSequence`` /
  PyG ``HeteroData`` (R-GCN-lite per-relation messages; Schlichtkrull et
  al. 2018 motivation only). Typed multi-node graphs use a shared latent
  width :math:`d` and stacked layout
  (:mod:`koopman_graph.data.hetero_layout`). Optional HGT peers
  (``HGTEncoder`` / ``HGTDecoder`` in :mod:`koopman_graph.nn`) wrap PyG
  ``HGTConv`` and are **not** required for hetero support
* ``DelayEmbeddingEncoder`` / ``n_delays`` for Hankel-style partial
  observability
* Per-snapshot ``edge_index`` (dynamic topology) and end-to-end
  ``edge_weight`` support
* Optional self-adaptive pairwise topology
  (``learn_topology="self_adaptive"``; ``koopman_graph.nn.AdaptiveAdjacency``)
* Optional symmetry-adapted orbit-tied ``K_self``
  (``koopman_auto_orbits`` / ``koopman_orbit_partition``; requires
  ``pip install "koopman-graph[symmetry]"`` for auto orbits)

Dynamics
~~~~~~~~

* Discrete ``KoopmanOperator`` with soft modes (``dense``, ``odo``) or
  structural guarantees (``schur``, ``dissipative``, ``lyapunov``)
* Networked ``GraphKoopmanOperator`` (``koopman="graph"``) with
  self/neighbor coupling
  :math:`I_N \otimes K_{\mathrm{self}} + \widehat{A} \otimes K_{\mathrm{nbr}}`
  (factorized blocks inspired by compositional / networked Koopman
  constructions; cross-topology transfer is measured, not assumed — see
  :doc:`limitations` and ``examples/37_cross_topology_transfer.ipynb``)
* Pairwise ``adjacency`` modes on graph / continuous-graph operators:
  ``"symmetric"`` (default), ``"random_walk"``
  (:math:`D_{\mathrm{out}}^{-1}A`), and ``"dual_random_walk"`` (forward plus
  reverse); factory keyword ``koopman_adjacency``; format-1
  ``config.adjacency``. Hypergraph operators stay Zhou-symmetric and do not
  expose ``adjacency``
* Hypergraph ``HypergraphKoopmanOperator`` (``koopman="hypergraph"``)
* Relational multiplex / typed
  ``HeteroGraphKoopmanOperator`` (``koopman="hetero_graph"``):
  :math:`K_{\mathrm{eff}} = I\otimes K_{\mathrm{self}} + \sum_r
  \widehat{A}_r \otimes K_r` (typed: block-diagonal per-type self).
  Optional ``relation_tying="basis"``. Dense :math:`N\cdot d` spectrum /
  inverse ceiling unchanged — see :doc:`limitations`
* Global/local non-stationary discrete operator
  (``GlobalLocalKoopmanOperator``; ``koopman="global_local"``)
* Continuous-time ``ContinuousKoopmanOperator``
  (``dynamics_mode="continuous"``), irregular timestamps, and
  ``predict_at``
* Continuous networked generator ``ContinuousGraphKoopmanOperator``
  (``koopman="graph"`` + ``dynamics_mode="continuous"`` or
  ``koopman="continuous_graph"``; ``adjacency`` modes match discrete
  ``GraphKoopmanOperator``; dense ``N·d`` matrix-exp cost caveat —
  prefer modest ``N`` or ``sparsity="block_diagonal"`` self-only shortcut)
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
  :math:`L_1` / smoothed :math:`L_p` Koopman sparsity, and an optional
  worst-case (:math:`L_{\infty}`\-style) reconstruction term (robust training
  only — not a generalization bound)
* LR schedulers, per-term loss history, ``MultiTrajectory`` fit, and
  windowed mini-batching
* Temporal train/val/test splits and per-horizon MAE, RMSE, and MAPE via
  ``koopman_graph.metrics.evaluate_forecast``

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
This is **not** the unimplemented operator flag ``sparsity="distributed"``
(see :doc:`faq` and :doc:`limitations`).

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
* Optional Ray parallel ensemble member fits via
  :func:`~koopman_graph.distributed.fit_ensemble_with_ray` or
  ``EnsembleGraphKoopmanModel.fit(..., parallel_backend="ray",
  member_factory=...)`` (requires ``pip install "koopman-graph[ray]"``).
  Sequential ensemble fit remains the default. This does **not** change
  UQ coverage guarantees — members stay independent fits. Prefer native
  DDP / Fabric for multi-GPU *model* training (Ray Train is out of scope)
* Example script (outside ``nbmake`` CI)::

    torchrun --standalone --nproc_per_node=2 \\
      examples/scripts/ddp_fit_torchrun.py

  See ``examples/scripts/README.md``. Multi-process smoke tests are
  opt-in (``KOOPMAN_GRAPH_DISTRIBUTED_TESTS=1``); default PR CI does not
  require multi-node hardware.

Ray Tune is **examples-only**: see
``examples/scripts/ray_tune_koopman_example.py`` (search space stays in the
script; the library does not expose a Tune / AutoML API). Dask is
**docs-only** in 0.8.0: there is no library ``dask_prep`` API and no Dask
training loop. Use Dask in *user* code to materialize trajectories or
window lists offline, then pass in-memory
:class:`~koopman_graph.data.GraphSnapshotSequence` objects into ``fit`` /
:class:`~koopman_graph.distributed.DistributedWindowSampler` (see
:doc:`faq`). The ``[dask]`` extra remains a reserved dependency pin.
Distributed data-parallel training does **not** reduce dense
:math:`N\cdot d` operator ceilings — see :doc:`limitations` (Scale).

Analysis
~~~~~~~~

* ``KoopmanSpectrum`` / ``compute_spectrum`` with mode decoding helpers
* ``attribute_mode_energy`` / ``ModeEnergyAttribution`` — interpretive
  type / relation energy fractions on assembled ``K_eff`` (not causal;
  not a ResDMD residual on relation-attributed modes)
* ``spectral_residuals`` / ``SpectralResidualReport`` — held-out data-driven
  residuals and ``trustworthy_mask()``; optional
  ``plot_spectrum(..., annotate_untrustworthy=True)``. Diagnostic in the
  learned observable space, **not** a certified ResDMD bound
  (``ColbrookTownsend2023ResDMD``, ``Colbrook2023ResidualDMD``)
* Long-horizon statistics via ``koopman_graph.statistics`` (power-user):
  Welch PSD (``Welch1967``), ``spectral_distance``,
  ``invariant_measure_distance``, Rosenstein
  ``largest_lyapunov_exponent`` (``Rosenstein1993Lyapunov``;
  embedding-sensitive; :math:`O(T^{2})`), and ``LongHorizonReport``
* Dynamical similarity and anomaly utilities via
  ``koopman_graph.analysis``
* ``identify_sparse_dynamics`` (SINDy / STLSQ on learned latents — not
  physical governing equations)
* ``koopman_spectral_clustering`` (node communities from Koopman modes)
* ``estimate_coupling_from_snapshots`` (topology / coupling diagnostics)
* ``plot_spectrum`` for unit-disk / data-zoom views

Control, adaptation, and observation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``koopman_graph.adaptation.RecursiveKoopmanAdapter`` / ``adapt_step`` for
  online RLS updates
* ``KoopmanObserver`` for latent Kalman filtering / imputation under
  ``observation_masks``
* ``koopman_graph.env.GraphKoopmanEnv`` / ``to_latent_env`` for Gymnasium
  closed-loop control
* ``koopman_graph.mpc.KoopmanMPC`` for additive-control receding-horizon
  QP control (``[mpc]`` / OSQP; local decoder-linearization guarantees;
  optional conformal output-constraint tightening)
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
  refinement (not DPK; not a full K²VAE)
* ``koopman_graph.uq.ConformalKoopmanUQ`` — split and adaptive (ACI)
  conformal intervals; marginal coverage under exchangeability (approximate
  under temporal dependence). Scores: ``"aggregate"``, legacy ``"per_node"``
  (max-pool over nodes), and ``"node_wise"`` (per-node marginal widths;
  optional ``neighbor_smoothing`` following DAPS-style diffusion,
  ``Zargarbashi2023ConformalGNN``). Calibration payload kind
  ``ConformalKoopmanUQ.calibration.v2``
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
  :math:`O(T^2)` — small/medium ``T``), ``DMDcBaseline``. Truncated-SVD
  ``rank`` accepts ``None``, a positive integer, or ``"auto"``
  (Gavish–Donoho median threshold, ``GavishDonoho2014``); fitted
  ``selected_rank`` is recorded
* Lightweight STGCN / DCRNN / Graph WaveNet references in
  ``koopman_graph.baselines.gnn`` (teaching baselines, not dedicated-library
  SOTA)
* Benchmark datasets and Jupyter tutorials under ``examples/``
* Model ``save`` / ``load`` checkpoints; ≥90% coverage enforced in CI

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
* :doc:`api` — module reference
* Repository ``README.md`` — short landing page with a Related software niche map
