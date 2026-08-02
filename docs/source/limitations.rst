Scope and limitations
=====================

KoopmanGraph is a PyTorch Geometric library for topology-aware Koopman
autoencoders on graphs. This page consolidates scope boundaries that are
otherwise scattered across :doc:`capabilities`, dataset cards in :doc:`data`,
docstrings, and tutorial prose. It is the authoritative public list of
limitations; the repository ``README.md`` links here rather than duplicating
the full inventory.

Every quantitative statement below is tied to in-repo measurements or to
behavior enforced by the current API. Absences listed under
:ref:`limitations-when-else` and :ref:`limitations-011-roadmap` are **scoped
non-goals**, not overlooked gaps.

Graph structure
---------------

* **Default coupling is undirected-symmetric.** Networked operators
  (``koopman="graph"``) default to ``adjacency="symmetric"`` (normalized
  undirected adjacency). Directed coupling is available via
  ``adjacency="random_walk"`` (row-normalized
  :math:`D_{\mathrm{out}}^{-1}A`) or ``adjacency="dual_random_walk"``
  (forward plus reverse walks). Hypergraph operators keep Zhou-style
  symmetric incidence normalization; **directed** hypergraph coupling is
  out of scope (see :ref:`limitations-011-roadmap`).
* **Fixed node cardinality per sequence.** Homogeneous
  ``GraphSnapshotSequence`` locks :math:`N` to the first snapshot.
  Heterogeneous ``HeteroGraphSnapshotSequence`` locks per-type counts
  :math:`N_\tau` and the edge-type set. Unobserved nodes are handled with
  observation masks; **variable node cardinality / unbounded node churn** is
  not supported (see :ref:`limitations-011-roadmap`).
* **Heterogeneous / multiplex graphs are supported (opt-in).** Use
  ``koopman="hetero_graph"`` with
  :class:`~koopman_graph.nn.heterogeneous.RelGraphEncoder` /
  :class:`~koopman_graph.nn.heterogeneous.RelGraphDecoder` on
  :class:`~koopman_graph.data.HeteroGraphSnapshotSequence` (PyG
  ``HeteroData``). Multiplex (one node type, :math:`|R|\ge 1` relations)
  and typed multi-node graphs are in scope. Shared latent width :math:`d`
  remains the default; opt-in per-type widths :math:`d_\tau` use
  ``latent_dims`` with rectangular relation maps :math:`K_r` (see
  :doc:`architecture`). Continuous hetero generators
  (``dynamics_mode="continuous"``) are supported with documented
  :math:`\Phi=\exp(\Delta t\, L_{\mathrm{eff}})` cost. Optional HGT peers
  under :mod:`koopman_graph.nn` are **not** required. See
  :doc:`capabilities`, :doc:`architecture`, and
  ``examples/39_heterogeneous_relational_koopman.ipynb``.
* **Hypergraphs and simplicial-1 lifts.** Incidence-based hypergraph
  encode / decode / operators are supported. Combinatorial simplicial-1 /
  Hodge helpers
  (:mod:`koopman_graph.observables`,
  :class:`~koopman_graph.nn.simplicial.SimplicialEncoder` /
  :class:`~koopman_graph.nn.simplicial.SimplicialDecoder`) use oriented
  ``edge_index`` and optional ``face_index``. **Sheaf Laplacians and full
  cell-complex TDL stacks** remain out of scope (see
  :ref:`limitations-011-roadmap`).
* **Cross-topology transfer is not automatic.** On a seeded path-diffusion
  transfer from :math:`N_1=4` to :math:`N_2=6` nodes, the factorized graph
  operator achieved hold-out mean squared error (MSE) approximately
  :math:`0.26` both in-distribution and after the node-count change; a
  per-node dense control reached approximately :math:`0.21`. That run did
  **not** show a transfer advantage for the graph factorization.
  Self-adaptive topology and orbit-tied ``K_{\mathrm{self}}`` configurations
  raise on node-count changes and are excluded from any transfer claim.
  See ``examples/37_cross_topology_transfer.ipynb``. A public measured
  cross-topology transfer API is reserved for later
  (:ref:`limitations-011-roadmap`).

Operator and theory
-------------------

* **Factorized networked form is a parameter-tying assumption.** The discrete
  graph operator uses
  :math:`I_N \otimes K_{\mathrm{self}} + \widehat{A} \otimes K_{\mathrm{nbr}}`.
  On path diffusion with :math:`N=4`, latent width :math:`d=2`, and 80
  training epochs, the factorized model reached hold-out MSE approximately
  :math:`0.71` with 8 trainable factor parameters, versus approximately
  :math:`0.019` MSE for a post-hoc unconstrained joint least-squares fit on
  the same frozen encodings (64 fitted entries; spectral distance
  approximately :math:`0.59` via ``dynamical_similarity``). The joint arm is
  not a second end-to-end trained model. See
  ``examples/38_operator_factorization_ablation.ipynb``.
* **Relational factorization must earn its keep.** The multiplex operator
  :math:`K_{\mathrm{eff}} = I_N \otimes K_{\mathrm{self}} + \sum_r
  \widehat{A}_r \otimes K_r` (typed: block-diagonal per-type self blocks)
  does **not** guarantee forecasting wins. Ablate against union-adjacency /
  dense-joint controls; negative relational results are allowed. Mode-energy
  attribution in :mod:`koopman_graph.analysis` is an interpretive diagnostic,
  not a causal claim and not a ResDMD residual bound. See
  ``examples/39_heterogeneous_relational_koopman.ipynb``.
* **Per-factor structural modes do not certify joint stability.** Soft /
  structural parameterizations on individual ``K_self`` / ``K_r`` blocks do
  **not** certify :math:`\rho(K_{\mathrm{eff}})`. Topology-aware
  ``stability_certificate`` on graph / hetero operators defaults to a
  **sufficient** Gershgorin joint bound
  (:class:`~koopman_graph.analysis.JointStabilityCertificate`), not a tight
  certificate and not soft assembled eigenvalue regularization
  (:class:`~koopman_graph.losses.EigenvalueRegularizationLoss`). Opt-in
  ``kind="schur"`` / ``"lyapunov"`` post-hoc certificates on assembled
  ``K_{\mathrm{eff}}`` are available when ``N·d`` is modest (ceilings
  :data:`~koopman_graph.analysis.MAX_JOINT_SCHUR_SIZE` and
  :data:`~koopman_graph.analysis.MAX_JOINT_LYAPUNOV_SIZE`); they are
  **not** factor-wise training parameterizations. Prefer assembled
  spectrum / ``spectral_radius`` or ``kind="schur"`` when the true joint
  radius matters.
* **Orbit ties are an inductive bias**, not isotypic / irreducible-
  representation block diagonalization. Shared-:math:`d` typed hetero orbits
  are partitioned independently within each type block; rectangular typed
  orbits are unsupported. A representation-theoretic (isotypic / irrep)
  reduction is reserved for 0.11
  (:ref:`limitations-011-roadmap`).
* **Stochastic mode is process noise, not an SDE.** Opt-in
  ``dynamics_mode="stochastic"`` adds learned diagonal process noise after a
  discrete linear map (:math:`z \mapsto Kz+\varepsilon`); it is not a
  continuous-time stochastic generator.
* **Topology-blind VAMP-2 precursor.** Optional
  :func:`~koopman_graph.baselines.vamp2.vamp2_score` /
  :func:`~koopman_graph.baselines.vamp2.vamp2_loss` and
  ``LossWeights.vamp2`` act on flattened encoder latents. This is **not**
  GraphVAMPnets or an MD / MSM production toolchain
  (:ref:`limitations-011-roadmap`).
* **Bayesian Laplace UQ over operator factors.** 
  :class:`~koopman_graph.uq.BayesianKoopmanUQ` provides a diagonal Laplace
  posterior over linear Koopman factors with seeded sample forecasts. It is
  **not** a Bayesian neural net over nonlinear encoder weights and not a full
  deep probabilistic Koopman / :math:`K^{2}`\ VAE
  (:ref:`limitations-011-roadmap`). Ensemble, latent-Gaussian, and conformal
  surfaces remain available (see below).
* **Invariant geometry encode (Tier A).** 
  :class:`~koopman_graph.nn.equivariant.InvariantGeometryEncoder` builds
  rotation-/translation-invariant distance and angle features from
  ``Data.pos`` and lifts with a standard GCN. **Invariant features do not
  make the latent operator** :math:`K` **E(n)/SE(3) equivariant.** Optional
  Tier B :class:`~koopman_graph.nn.equivariant.E3EquivariantEncoder`
  (``e3nn``, ``[equivariance]``) uses steerable layers but still projects to
  invariant scalar latents for linear :math:`K` — not an equivariant
  operator.

Spectral analysis
-----------------

* **Data-driven residuals are diagnostics, not certified ResDMD bounds.**
  ``koopman_graph.analysis.spectral_residuals`` measures whether claimed
  eigenpairs propagate as :math:`a(t+1) \approx \lambda\, a(t)` in the
  *learned* latent / observable space on held-out trajectories. That is
  distinct from residual dynamic mode decomposition (ResDMD).
* **Finite-dictionary ResDMD MVP and resolvent grid.** 
  :func:`~koopman_graph.analysis.resdmd` /
  :class:`~koopman_graph.analysis.ResDMDReport` and the finite-matrix
  resolvent-norm helper in :mod:`koopman_graph.analysis.pseudospectra`
  operate on a **fixed** dictionary. They are **not** infinite-dimensional
  certified pseudospectra or spectral-measure estimators
  (Colbrook & Townsend, *Commun. Pure Appl. Math.*, 2023/2024;
  ``ColbrookTownsend2023ResDMD``). See
  ``examples/40_resdmd_pseudospectra.ipynb``.
* **Largest Lyapunov exponent (Rosenstein) is available but sensitive.**
  ``koopman_graph.statistics.largest_lyapunov_exponent`` implements the
  Rosenstein et al. (1993) small-data-set method
  (``Rosenstein1993Lyapunov``). Estimates depend on delay,
  embedding dimension, Theiler window, and fit range; nearest-neighbor search
  is :math:`O(T^{2})`. Construction oracles use documented hyperparameters
  (see the statistics unit tests); bare defaults are not a high-precision
  spectral computation.

Scale
-----

Training reuses shared latents and several ephemeral operator / support
caches (see :doc:`capabilities` and :doc:`architecture`), but those
optimizations do **not** remove the dense representation ceilings below.
For large :math:`N`, prefer ``sparsity="block_diagonal"`` where applicable,
modest latent width, and optional CUDA automatic mixed precision
(``use_amp=True`` on ``fit`` / ``run_fit_loop``); see :doc:`faq`.

* **Exact spectrum and exact inverse** of networked operators (including
  relational ``hetero_graph``) assemble an effective dense matrix of size
  :math:`(N \cdot d) \times (N \cdot d)` (typed / rectangular:
  stacked :math:`\sum_\tau N_\tau d_\tau`). Prefer modest :math:`N` or
  ``sparsity="block_diagonal"`` (approximate Jacobi / self-dominated path)
  when that cost dominates. Static topology may reuse a precomputed dense
  inverse within a training-loss evaluation; the assembly size is unchanged.
  Multi-GPU *trainer* orchestration (DDP / Fabric / Lightning / Ray) shards
  data and synchronizes gradients; it does **not** shrink this dense ceiling.
* **Hetero × trainers.** Multiplex / typed models compose with native DDP
  (``strategy="ddp"``), Lightning Fabric, optional Lightning ``Trainer``,
  and Ray ensemble helpers under :mod:`koopman_graph.distributed`.
  ``find_unused_parameters`` defaults to ``True`` for hetero RelGraph stacks.
  Single-process windowed ``run_fit_loop`` accepts windowed hetero sequences
  (parity with world-size-1 DDP window sampling). Do not confuse trainer
  orchestration with the operator flag ``sparsity="distributed"``
  (matrix-free inverse / spectrum; see below).
* **Eigenvalue regularization** with a non-zero
  ``LossWeights.eigenvalue`` on dense or ODO networked operators runs an
  :math:`O((N \cdot d)^3)` eigendecomposition of the effective map unless
  ``sparsity="distributed"`` (Arnoldi surrogate on discrete graph /
  multiplex hetero; zero hinge on hypergraph / continuous distributed).
  Prefer structural parameterizations (``schur`` / ``lyapunov`` /
  ``dissipative``) or modest :math:`N` when the assembled term is enabled.
* **``sparsity="distributed"``** is the *operator* matrix-free path
  (Richardson / Neumann inverse and Arnoldi spectrum for discrete graph and
  multiplex hetero; hypergraph / continuous kinds may still assemble). It is
  **not** multi-GPU training. Optional *trainer* orchestration under
  :mod:`koopman_graph.distributed` (native DDP / Lightning Fabric) shards
  data and synchronizes gradients; it does **not** shrink the dense
  :math:`N\cdot d` representation ceilings above.
* **Multi-process / multi-GPU CI is not guaranteed.** Default test jobs
  stay single-process. Optional gloo smokes are opt-in
  (``KOOPMAN_GRAPH_DISTRIBUTED_TESTS=1`` / ``@pytest.mark.distributed``).
  Multi-node launches and measured speedups are outside the documented
  contract.
* **Dask prep, not a Dask training loop.** Optional
  ``pip install "koopman-graph[dask]"`` activates
  :mod:`koopman_graph.distributed.dask_prep` helpers
  (``materialize_sequences``, ``materialize_window_index_list``). The library
  does **not** import Dask on the training path; training remains PyTorch /
  :mod:`koopman_graph.distributed` (see :doc:`faq`).
* **DiffConv** bidirectional diffusion supports are dense
  :math:`N \times N` tensors for static graphs. Identical
  ``edge_index`` / weight storage may reuse cached supports
  (``clear_support_cache``); caching does not sparsify the representation.
* **Continuous-graph dense advances** form
  :math:`\Phi = \exp(\Delta t\, L_{\mathrm{eff}})` on the same
  :math:`(N \cdot d) \times (N \cdot d)` scale (typed continuous hetero:
  stacked width). Prefer ``sparsity="block_diagonal"`` (self-dominated
  shortcut) for large :math:`N`. Within one training-loss evaluation the
  dense path may reuse :math:`\Phi` (and assembled :math:`L_{\mathrm{eff}}`)
  for repeated topology / :math:`\Delta t` keys; see :doc:`faq`.
* **Hypergraph Zhou** :math:`\hat{H}` is a dense :math:`N \times N`
  matrix on the advance / eigen path. Static incidence may reuse a cached
  :math:`\hat{H}` (see ``clear_hyperedge_cache``); caching does not remove
  the :math:`O(N^2)` representation or the dense networked
  :math:`(N \cdot d)` ceiling above.
* **Hierarchical pooling** (``HierarchicalGraphKoopmanModel``) recomputes
  feature-dependent TopK / SAG scores every snapshot under the default
  ``pool_schedule="per_snapshot"``. ``pool_schedule="hold_perm"`` holds the
  first-snapshot permutation for the sequence (amortized pooling; scores
  no longer track per-timestep features).

Uncertainty
-----------

* **Conformal coverage is marginal under exchangeability** and only
  approximate under temporal dependence
  (``koopman_graph.uq.ConformalKoopmanUQ``), including hetero sequences.
* **``score="node_wise"``** yields per-node marginal half-widths (optional
  neighbor smoothing), not joint coverage across nodes. The legacy
  ``score="per_node"`` mode still max-pools over nodes into a scalar score;
  it is not heteroscedastic per node.
* **Ensemble and latent-Gaussian UQ** are power-user surfaces with their own
  modeling assumptions (empirical quantiles; linear-Gaussian latent
  propagation). They are not deep probabilistic Koopman (DPK) or a full
  :math:`K^{2}` variational autoencoder.
* **Bayesian Laplace UQ** (above) samples linear operator factors only.

Benchmarks and baselines
------------------------

* **Measured telemetry** (observed time series): METR-LA, PEMS-BAY,
  PEMS03/04/07/08, and SocioPatterns contact networks (see :doc:`data` for
  acquisition and license cards).
* **Simulated or surrogate dynamics:** synthetic Laplacian / advection
  graphs, networked SIR, Lorenz-96, Kuramoto–Sivashinsky, IEEE 118 voltage /
  load diffusion (homogeneous and typed generator / load / slack helpers),
  and the Hopf / Stuart–Landau cylinder-wake teaching surrogate. IEEE 118
  uses a real bus topology with *simulated* dynamics, not SCADA telemetry.
* **In-repo GNN forecasters** (STGCN, DCRNN, Graph WaveNet in
  ``koopman_graph.baselines.gnn``) are **teaching baselines**, not
  reproductions of dedicated-library state-of-the-art (SOTA) implementations.
  Adding further SOTA spatiotemporal GNN baselines (for example AGCRN,
  MTGNN, STGODE, GraphCast) is out of scope
  (:ref:`limitations-011-roadmap`).
* **Classical DMD family.** Shipped baselines include DMD, EDMD (including
  Nyström / random-feature kernel approximations), DMDc, forward–backward
  DMD, total-least-squares DMD, optDMD (variable-projection MVP), streaming
  DMD, depth-2 multi-resolution DMD, and the Ulam transfer-operator
  baseline. ``RecursiveKoopmanAdapter`` is not a streaming-DMD substitute; it
  refines a learned latent operator and requires dense parameterization.
* **Long-horizon chaotic evaluation** should report distributional statistics
  (``koopman_graph.statistics``) alongside short-horizon RMSE; pointwise error
  alone saturates past the predictability horizon. See
  ``examples/24_nonlinear_chaotic_benchmarks.ipynb``.

.. _limitations-011-roadmap:

0.11 roadmap
------------

The following remain **explicit non-goals** of the current release and are
tracked for a later 0.11-oriented roadmap (or external toolchains). Public
docs will not silently drop them.

* Traffic-forecasting teaching SOTA GNN baselines (AGCRN, MTGNN, STGODE,
  GraphCast) and leaderboard-matched traffic protocols / GraphCast ERA5
  production.
* Variable node cardinality / unbounded node churn (union :math:`N_{\max}`
  + presence masks).
* Directed hypergraph incidence modes.
* Public measured cross-topology transfer API (notebook 37 remains a
  measurement demo until then).
* Sheaf Laplacian layers and full cell-complex topological deep-learning
  stacks (0.10 ships simplicial-1 / Hodge lifts only).
* GraphVAMPnets and molecular-dynamics / Markov-state-model production
  toolchains (0.10 ships a topology-blind VAMP-2 precursor only; optional
  ``[msm]`` is for deeptime oracle tests).
* Ray Train as a multi-GPU *model* DDP backend (native DDP / Fabric / Ray
  *ensemble* helpers remain).
* Representation-theoretic isotypic / irrep block diagonalization (0.10
  enables hetero *orbit* ties only).
* Infinite-dimensional ResDMD pseudospectra / spectral measures beyond the
  finite-dictionary MVP.
* Full deep probabilistic Koopman / :math:`K^{2}`\ VAE; Bayesian sampling over
  nonlinear encoder weights.
* Equivariant latent operators :math:`K` / isotypic irrep block
  diagonalization beyond orbit ties (0.10 ships Tier A invariant geometry
  and optional Tier B steerable *encode* to invariant latents only).
* Changing homogeneous scientific defaults (AMP off by default; sparsity
  default ``"dense"``; linear time-invariant latent advance).

.. _limitations-when-else:

When to use something else
--------------------------

* **Traffic forecasting leaderboards.** Prefer dedicated traffic-forecasting
  libraries and their maintained SOTA models when the goal is protocol-matched
  leaderboard numbers on METR-LA / PEMS-style benchmarks. KoopmanGraph’s
  traffic demos and in-repo GNN baselines are teaching comparisons around an
  inspectable linear latent operator, not a forecasting competition entry.
* **Sheaf / full cell-complex TDL.** Prefer libraries built for sheaf
  Laplacians and cellular TDL stacks when the domain model requires those
  operators. KoopmanGraph’s simplicial-1 / Hodge lifts are combinatorial
  precursors, not a full TSP stack.
* **Biomolecular / molecular-dynamics Koopman analysis.** Prefer
  GraphVAMPnets / MSM toolchains aimed at molecular trajectory data. The
  topology-blind VAMP-2 precursor here is not that toolchain.
* **Certified infinite-dimensional spectral computation.** Prefer full
  ResDMD / spectral-measure packages when you need residual-based
  certificates beyond the finite-dictionary MVP and held-out
  ``spectral_residuals`` diagnostics.
* **Flat-vector deep Koopman without graph structure.** Packages such as
  PyKoopman and DLKoopman remain appropriate when the state is naturally a
  vector and edge-based coupling is not part of the model.

Related pages
-------------

* :doc:`capabilities` — feature inventory
* :doc:`architecture` — public vs power-user API contract
* :doc:`data` — dataset cards and per-dataset limitations
* :doc:`tutorials` — notebook gallery
* :doc:`faq` — install and runtime troubleshooting
