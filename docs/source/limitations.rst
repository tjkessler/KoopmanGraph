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
:ref:`limitations-remaining` and :ref:`limitations-when-else` are **scoped
non-goals or honesty boundaries**, not overlooked gaps.

Graph structure
---------------

* **Default coupling is undirected-symmetric.** Networked operators
  (``koopman="graph"``) default to ``adjacency="symmetric"`` (normalized
  undirected adjacency). Directed coupling is available via
  ``adjacency="random_walk"`` (row-normalized
  :math:`D_{\mathrm{out}}^{-1}A`) or ``adjacency="dual_random_walk"``
  (forward plus reverse walks). Hypergraph operators support
  ``incidence_mode``
  ``"zhou_symmetric"`` (default), ``"forward_random_walk"``, and
  ``"dual_random_walk"`` (factory ``koopman_hypergraph_incidence_mode``).
  Encode and advance orientations may differ under directed incidence;
  that asymmetry is intentional and is **not** a simplicial / Hodge claim.
* **Fixed node cardinality is the default; opt-in presence-mask churn.**
  Homogeneous ``GraphSnapshotSequence`` locks :math:`N` to the first
  snapshot unless ``allow_node_churn=True`` with fixed-union capacity
  :math:`N_{\max}` and boolean ``presence_masks`` of shape
  ``(T, N_{\max})``. Heterogeneous sequences lock per-type counts
  :math:`N_\tau` (or per-type masks under the same contract). Losses ignore
  inactive nodes; operator matvecs still run at the full
  :math:`N_{\max}` (or typed) capacity. **Unbounded open-world growth**
  and index remapping across unrelated universes remain unsupported
  (see :ref:`limitations-remaining`).
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
* **Hypergraphs, simplicial-1, and TDL MVP peers.** Incidence-based
  hypergraph encode / decode / operators are supported. Combinatorial
  simplicial-1 / Hodge helpers
  (:mod:`koopman_graph.observables`,
  :class:`~koopman_graph.nn.simplicial.SimplicialEncoder` /
  :class:`~koopman_graph.nn.simplicial.SimplicialDecoder`) use oriented
  ``edge_index`` and optional ``face_index``. Sheaf peers
  (:class:`~koopman_graph.nn.sheaf.SheafGNNEncoder` /
  :class:`~koopman_graph.nn.sheaf.SheafGNNDecoder`; factory
  ``encoder="sheaf"``) and cell-complex peers
  (:class:`~koopman_graph.nn.cell_complex.CellComplexGNNEncoder` /
  :class:`~koopman_graph.nn.cell_complex.CellComplexGNNDecoder`;
  ``encoder="cell_complex"``) are in-repo MVPs with the same linear
  Koopman head. They are **not** full TopologicX / cellular TDL feature
  parity (see :ref:`limitations-remaining`).
* **Cross-topology transfer is measured, not assumed.** 
  :func:`~koopman_graph.analysis.evaluate_topology_transfer` returns a
  structured report with a mandatory ``pernode`` control; negative
  transfer advantage is an expected outcome. On a seeded path-diffusion
  transfer from :math:`N_1=4` to :math:`N_2=6` nodes, the factorized graph
  operator achieved hold-out mean squared error (MSE) approximately
  :math:`0.26` both in-distribution and after the node-count change; a
  per-node dense control reached approximately :math:`0.21`. That run did
  **not** show a transfer advantage for the graph factorization.
  Self-adaptive topology, orbit-tied ``K_{\mathrm{self}}``, and isotypic
  configurations bind node cardinality and are excluded from transfer
  claims. See ``examples/37_cross_topology_transfer.ipynb``.

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
* **Orbit ties and isotypic MVP are inductive biases.** Orbit-label ties
  share ``K_{\mathrm{self}}`` blocks; they are not automatic isotypic
  reduction. Opt-in ``koopman_symmetry="isotypic"`` (exact automorphism
  groups on modest :math:`N`; see ``examples/45_isotypic_symmetry.ipynb``)
  ties isotypic projectors for the self block. Neighbor-factor
  (:math:`K_{\mathrm{nbr}}`) isotypic tying is **not** shipped. Shared-
  :math:`d` typed hetero orbits are partitioned independently within each
  type block; rectangular typed orbits are unsupported. Neither path
  guarantees sample-efficiency wins
  (see :ref:`limitations-remaining`).
* **Stochastic mode is process noise, not an SDE.** Opt-in
  ``dynamics_mode="stochastic"`` adds learned diagonal process noise after a
  discrete linear map (:math:`z \mapsto Kz+\varepsilon`); it is not a
  continuous-time stochastic generator.
* **VAMP-2 precursor and GraphVAMP teaching path.** Optional
  :func:`~koopman_graph.baselines.vamp2.vamp2_score` /
  :func:`~koopman_graph.baselines.vamp2.vamp2_loss` and
  ``LossWeights.vamp2`` act on flattened encoder latents (topology-blind).
  :class:`~koopman_graph.baselines.GraphVAMPBaseline` plus
  :mod:`koopman_graph.datasets.molecular` (synthetic contact-graph oracle;
  optional ``[md]`` / ``[msm]``) form a **teaching / diagnostic** MD-adjacent
  toolchain, not GraphVAMPnets production software, not a PyEMMA
  replacement, and not Folding@home-scale MD. A public alanine-dipeptide
  loader is not shipped; CI uses the synthetic oracle
  (see :ref:`limitations-remaining`).
* **Bayesian Laplace UQ over operator factors.** 
  :class:`~koopman_graph.uq.BayesianKoopmanUQ` provides a diagonal Laplace
  posterior over linear Koopman factors with seeded sample forecasts. It is
  **not** a Bayesian neural net over nonlinear encoder weights and not a full
  deep probabilistic Koopman / :math:`K^{2}`\ VAE
  (:ref:`limitations-remaining`). Ensemble, latent-Gaussian, and conformal
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

.. _limitations-topology-criticality:

Topology criticality (non-goal)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Dynamic topology is mechanical, not criticality-aware.** Mid-horizon
  rewiring (``allow_dynamic_topology`` / per-snapshot ``edge_index``) is
  supported in the networked linear step so forecasts can consume changing
  graphs. The library does **not** analyze or mitigate spectral degeneracy,
  spectral-gap closure, defective spectra, or forecast-horizon collapse near
  graph-topology criticality (connectivity / load / epidemic-threshold-style
  regimes discussed for infrastructure Koopman forecasting by Ghosh,
  *Intelligent Systems with Applications*, 2025; ``Ghosh2025``). Held-out
  ``spectral_residuals`` and finite-dictionary ResDMD remain diagnostics,
  not certificates (see Spectral analysis). Continuous matrix-log /
  Van Loan helpers do not handle non-diagonalizable (Jordan) generators
  (:mod:`koopman_graph.operators.continuous_van_loan`). No early-warning
  score or spectral-gap monitor is shipped.

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
* **Representation explain masks are interpretive, not causal.**
  :func:`~koopman_graph.analysis.explain_representation` returns
  :class:`~koopman_graph.analysis.RepresentationExplanation` node / edge /
  feature masks under GNNExplainer (``Ying2019GNNExplainer``) or optional
  Captum integrated gradients (``Sundararajan2017IntegratedGradients``;
  ``[explain]`` extra). Masks are **non-causal**: they are not
  interventional topology recovery, not a ResDMD residual bound, and not
  complementary operator-level
  :class:`~koopman_graph.analysis.ModeEnergyAttribution` on assembled
  ``K_eff``. The MVP is **homogeneous** ``Data`` only and rejects
  ``HeteroData``, RelGraph / ``hetero_graph``, hypergraph encoders /
  operators, adaptive / learned pairwise topology, ``n_delays != 1``, and
  ``control_dim > 0``.
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
optimizations do **not** remove the representation ceilings below.
For large :math:`N`, prefer ``sparsity="block_diagonal"`` where applicable,
modest latent width, and optional CUDA automatic mixed precision
(``use_amp=True`` on ``fit`` / ``run_fit_loop``); see :doc:`faq`.

* **Exact spectrum (graph / continuous-graph).** Discrete
  ``GraphKoopmanOperator.spectrum`` and continuous
  ``ContinuousGraphKoopmanOperator.spectrum`` auto-route (no path-selection
  kwarg):

  1. ``sparsity="distributed"`` — on discrete graph and multiplex hetero,
     matrix-free Arnoldi returns leading-modulus Ritz values (placeholder
     eigenvectors of size ``num_modes``). Continuous graph has **no**
     Arnoldi spectrum path and falls through to dense
     :math:`L_{\mathrm{eff}}` eigendecomposition.
  2. Else if Kronecker-sum eligible — shared self factor,
     ``adjacency`` in ``{"symmetric", "random_walk"}``, and
     ``sparsity`` in ``{"dense", "block_diagonal"}`` — exact reduction of
     :math:`I\otimes M_{\mathrm{self}} + \widehat{A}\otimes M_{\mathrm{nbr}}`
     (discrete :math:`K` or continuous generator :math:`L`). Cost is
     dominated by a dense :math:`N\times N` eigendecomposition of
     :math:`\widehat{A}` plus :math:`N` blocks of size :math:`d\times d`
     (order :math:`O(N^3 + N d^3)`), not a full ambient
     :math:`O((N\cdot d)^3)` factorization when the helper succeeds.
  3. Else — dense :math:`(N\cdot d)\times(N\cdot d)` eigendecomposition
     of the assembled effective map (``dual_random_walk``, discrete
     orbit / isotypic self banks, helper fall-back, and other ineligible
     cases).

  Hetero / hypergraph spectrum remains under the dense assembled
  :math:`(N\cdot d)` (or stacked typed) ceiling. Kronecker spectrum does
  **not** imply a Kronecker inverse or cheaper eig-regularization.

* **Exact inverse** of networked operators (including relational
  ``hetero_graph``) still assembles an effective dense matrix of size
  :math:`(N \cdot d) \times (N \cdot d)` (typed / rectangular:
  stacked :math:`\sum_\tau N_\tau d_\tau`), unless an approximate
  Jacobi / Richardson / Neumann path applies. Prefer modest :math:`N` or
  ``sparsity="block_diagonal"`` (approximate Jacobi / self-dominated path)
  when that cost dominates. Static topology may reuse a precomputed dense
  inverse within a training-loss evaluation; the assembly size is unchanged.
  Multi-GPU *trainer* orchestration (DDP / Fabric / Lightning / Ray Train)
  shards data and synchronizes gradients; it does **not** shrink this dense
  inverse ceiling. Presence-mask churn still matvecs at :math:`N_{\max}`
  capacity.
* **Hetero × trainers.** Multiplex / typed models compose with native DDP
  (``strategy="ddp"``), Lightning Fabric, optional Lightning ``Trainer``,
  and Ray helpers under :mod:`koopman_graph.distributed`.
  ``find_unused_parameters`` defaults to ``True`` for hetero RelGraph stacks.
  Single-process windowed ``run_fit_loop`` accepts windowed hetero sequences
  (parity with world-size-1 DDP window sampling). Do not confuse trainer
  orchestration with the operator flag ``sparsity="distributed"``
  (matrix-free inverse / Arnoldi spectrum on discrete graph and multiplex
  hetero; see below).
* **Eigenvalue regularization** with a non-zero
  ``LossWeights.eigenvalue`` on dense or ODO networked operators runs an
  :math:`O((N \cdot d)^3)` eigendecomposition of the assembled effective map
  unless ``sparsity="distributed"`` (Arnoldi surrogate on discrete graph /
  multiplex hetero; zero hinge on hypergraph / continuous distributed).
  The Kronecker spectrum path does **not** change this training-hinge
  ceiling. Prefer structural parameterizations (``schur`` / ``lyapunov`` /
  ``dissipative``) or modest :math:`N` when the assembled term is enabled.
* **``sparsity="distributed"``** is the *operator* matrix-free path
  (Richardson / Neumann inverse and Arnoldi spectrum for discrete graph and
  multiplex hetero; hypergraph / continuous kinds may still assemble for
  inverse and for continuous spectrum). It is **not** multi-GPU training
  and is **not** the Kronecker exact-spectrum reduction. Optional
  *trainer* orchestration under :mod:`koopman_graph.distributed` shards
  data and synchronizes gradients; it does **not** shrink the dense
  :math:`N\cdot d` representation ceilings above.
* **Trainer taxonomy (DDP / Fabric / Ray).** Native DDP and Lightning Fabric
  remain the **recommended** multi-GPU *model* paths.
  :func:`~koopman_graph.distributed.run_ray_train_fit_loop` (optional
  ``[ray]`` / ``[distributed]``) wraps Ray Train ``TorchTrainer`` around the
  same scientific fit loop (model DDP under Ray). Separately,
  :func:`~koopman_graph.distributed.fit_ensemble_with_ray` parallelizes
  *ensemble members*. Prefer DDP / Fabric unless you already standardize on
  Ray Train. Multi-node / cluster Ray Train production and measured
  multi-node speedups are outside the documented CI contract
  (see :ref:`limitations-remaining`).
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
* **Hypergraph Zhou** :math:`\hat{H}` (and directed incidence assemblies)
  are dense :math:`N \times N` tensors on the advance / eigen path. Static
  incidence may reuse a cached factor (see ``clear_hyperedge_cache``);
  caching does not remove the :math:`O(N^2)` representation or the dense
  networked :math:`(N \cdot d)` ceiling above.
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
  the Hopf / Stuart–Landau cylinder-wake teaching surrogate, and synthetic
  molecular contact-graph oracles under
  :mod:`koopman_graph.datasets.molecular`. IEEE 118 uses a real bus
  topology with *simulated* dynamics, not SCADA telemetry.
* **In-repo GNN forecasters** in ``koopman_graph.baselines.gnn`` include
  STGCN, DCRNN, Graph WaveNet, and teaching ports of AGCRN, MTGNN, STGODE,
  and GraphCast (``ForecasterProtocol`` deviation tables). They are
  **teaching baselines**, not protocol-matched leaderboard reproductions
  or dedicated-library SOTA. GraphCast is a small-mesh weather teaching
  adapter, not a PEMS sensor-graph forecaster and not ERA5-scale
  production training (see :ref:`limitations-remaining`).
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

.. _limitations-remaining:

Remaining limits
----------------

The following remain accurate honesty boundaries for the current release.
They are **not** a deferred roadmap of the same items under a future version
label.

* **Leaderboard-matched** traffic protocols (exact LibCity / BasicTS
  schedules, metrics, and preprocessing). Teaching deviations are
  documented; competition numbers stay upstream.
* **ERA5-scale / production GraphCast** training. The in-repo adapter is a
  documented small-mesh teaching slice only.
* **Unbounded growing graphs / open-world entity discovery** beyond fixed
  union :math:`N_{\max}` + presence masks. Index remapping across unrelated
  universes is out of scope.
* **Multi-node Ray Train / cluster production** support in CI. Documented
  path is single-node multi-GPU manual smoke; native DDP / Fabric remain the
  recommended defaults.
* **Full TopologicX / cellular TDL feature parity.** In-repo sheaf and
  cell-complex MVPs share the linear Koopman head; they do not replace
  dedicated TDL frameworks when the domain model is complex-native
  end-to-end. An optional richer external-complex bridge is not shipped.
* **Folding@home-scale MD**, a full PyEMMA replacement, or a public
  alanine-dipeptide fetch loader. GraphVAMP + deeptime interop and the
  synthetic contact-graph oracle are teaching / diagnostic surfaces.
* **Guaranteed sample-efficiency wins** from isotypic ties or transfer
  APIs. Negative transfer advantage and null ablations are allowed
  outcomes.
* **Neighbor-factor isotypic tying** (:math:`K_{\mathrm{nbr}}` Aut-consistent
  blocks). The isotypic MVP ties the self block only.
* **Infinite-dimensional ResDMD** pseudospectra / spectral measures beyond
  the finite-dictionary MVP.
* **Full deep probabilistic Koopman / :math:`K^{2}`\ VAE**; Bayesian
  sampling over nonlinear encoder weights.
* **Equivariant latent operators** :math:`K` (Tier A / B encode paths still
  feed invariant scalar latents into linear :math:`K`).
* **Changing homogeneous scientific defaults** (AMP off by default; sparsity
  default ``"dense"``; linear time-invariant latent advance).

.. _limitations-when-else:

When to use something else
--------------------------

* **Traffic forecasting leaderboards.** Prefer dedicated traffic-forecasting
  libraries and their maintained SOTA models when the goal is protocol-matched
  leaderboard numbers on METR-LA / PEMS-style benchmarks. KoopmanGraph’s
  traffic demos and in-repo GNN baselines are teaching comparisons around an
  inspectable linear latent operator, not a forecasting competition entry.
* **Full sheaf / cell-complex TDL stacks.** Prefer libraries built for rich
  sheaf Laplacians and cellular TDL when the domain model requires
  complex-native operators end-to-end. KoopmanGraph’s sheaf / cell MVPs and
  simplicial-1 lifts are precursors with a linear Koopman head, not a full
  TopologicX replacement.
* **Production biomolecular Koopman / MSM analysis.** Prefer GraphVAMPnets /
  PyEMMA-scale MSM toolchains aimed at molecular trajectory data when you
  need that production surface. The GraphVAMP teaching path and
  topology-blind VAMP-2 precursor here are not that toolchain.
* **Certified infinite-dimensional spectral computation.** Prefer full
  ResDMD / spectral-measure packages when you need residual-based
  certificates beyond the finite-dictionary MVP and held-out
  ``spectral_residuals`` diagnostics.
* **Portable inference export (TorchScript / ONNX).** 0.13.0 does **not**
  ship a TorchScript / ``torch.export`` / ONNX inference-export API.
  Share weights with the default ``safetensors_v1`` checkpoint path (or
  explicit ``legacy_pt`` when required). ONNX remains deferred.
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
