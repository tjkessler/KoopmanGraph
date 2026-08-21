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

Homogeneous scientific defaults stay as in 0.14:
``koopman=None`` still selects ``"pernode"``, ``sparsity="dense"``,
AMP is off, and discrete advance remains linear time-invariant.
Polynomial graph hops, closed-form identification, graph-state
closure, cochain dynamics, and conditioned interpolants are
**opt-in**. Negative transfer advantage and large one-tap versus
joint least-squares gaps remain allowed outcomes.

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
  :math:`N_{\max}` (or typed) capacity. Opt-in
  :func:`~koopman_graph.data.remap_node_features` grows a larger union
  from a **user-supplied injective**
  :class:`~koopman_graph.data.EntityRemap` (or the tensor helper).
  Silent merge of colliding IDs is refused. Changing :math:`N` across
  snapshots without that remap raises. Graph-state closure
  (:class:`~koopman_graph.data.GraphStateSnapshot`) stays on that
  **fixed-union** :math:`N_{\max}`; it is **not** open-world entity
  resolution across unrelated universes (see :ref:`limitations-remaining`).
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
* **Predicted topology is sparse by default and enters the forecast
  loop only when configured.** :class:`~koopman_graph.data.GraphDynamicsConfig`
  (opt-in; default model argument ``None``) selects
  ``sparse_candidate`` scoring at most ``candidate_k`` destinations per
  node (default 8). ``dense_mlp`` uses
  :class:`~koopman_graph.nn.PredictedTopologyHead` and refuses
  :math:`N>64` unless ``max_nodes`` is raised. This is distinct from
  ``learn_topology="self_adaptive"`` /
  :class:`~koopman_graph.nn.AdaptiveAdjacency`. Combining
  ``self_adaptive`` with a non-``none`` topology head raises.
  Predicted edges enter :math:`K_{\mathrm{eff}}` as sigmoid weights on
  the candidate COO (not a hard threshold). Homogeneous models only;
  RelGraph / hypergraph layouts raise. ``batch_graphs=True`` refuses
  recursive graph-state training. On a seeded six-node directed
  :math:`G(n,p)` event, a dense stub that scored the post-event graph
  beat hold-last when the comparison target was the model's own oracle
  one-step forecast; that is a wiring check, not a learned-forecast
  claim. See ``examples/50_graph_state_closure.ipynb``.

Operator and theory
-------------------

* **Factorized networked form is a parameter-tying assumption.** The discrete
  graph operator uses
  :math:`I_N \otimes K_{\mathrm{self}} + \widehat{A} \otimes K_{\mathrm{nbr}}`
  by default (``koopman_filter_degree=1``).
  On path diffusion with :math:`N=4`, latent width :math:`d=2`, and 80
  training epochs, the factorized model reached hold-out MSE approximately
  :math:`0.71` with 8 trainable factor parameters, versus approximately
  :math:`0.019` MSE for a post-hoc unconstrained joint least-squares fit on
  the same frozen encodings (64 fitted entries; spectral distance
  approximately :math:`0.59` via ``dynamical_similarity``). The joint arm is
  not a second end-to-end trained model. Degree :math:`P>1` is opt-in.
  The same notebook also reports a hop-matched :math:`P\in\{0,1,2\}` arm
  on that surrogate (40 epochs; not the 80-epoch historical protocol) and
  still does **not** attribute that gap to hop order. See
  ``examples/38_operator_factorization_ablation.ipynb`` and
  ``examples/49_multi_hop_factorization.ipynb``.
* **Encoder mixing does not compensate for one-hop** :math:`K`. Default
  ``GNNEncoder`` uses two message-passing layers; default discrete
  ``koopman="graph"`` mixes one hop (``koopman_filter_degree=1``).
  Degree :math:`P>1` is the opt-in polynomial; it is not the factory
  default. :meth:`~koopman_graph.model.GraphKoopmanModel.fit` emits
  :class:`~koopman_graph.nn.ReceptiveFieldMismatchWarning` when the
  encoder receptive field exceeds operator hops. Deeper GCN/GAT stacks
  or DiffConv ``diffusion_steps`` do **not** substitute for raising
  :math:`P`. The check is warn-only and skips operators without
  ``receptive_field_hops`` (per-node, continuous graph, hetero,
  hypergraph).
* **Per-node** :math:`K` **is not a cochain.** Factory
  ``koopman=None`` still selects a node latent map. Opt-in
  :class:`~koopman_graph.operators.CochainKoopmanOperator` advances
  0- and 1-cochain latents on a static signed :math:`B_1`. It is not
  a factory kind, not ``koopman="hodge"``, and not TopologicX parity
  (see :ref:`limitations-remaining`).
* **Discrete sequences still require uniform** :math:`\Delta t`.
  ``fit`` and ``predict_at`` reject gaps
  (``validate_uniform_discrete_increments``). Time-of-day Fourier /
  phase recipes do not lift that rule. Irregular timestamps on a
  gEDMD sequence are unused and do **not** create a generator
  :math:`L` (``Klus2020gEDMD``). Prefer
  ``dynamics_mode="continuous"`` or supplied derivatives when the
  sampling grid itself is irregular.
* **Structural homomorphism is a separable-dictionary lemma, not a GNN
  property.** Peng, Shen & Zhu (arXiv:2606.17797,
  ``Peng2026KoopmanGKFA``) prove that block sparsity of the lifted
  coupling matches graph edges only under a node-wise dictionary
  :math:`\psi_i(x_i)`. Default GNN encoders mix neighbors.
  :class:`~koopman_graph.adaptation.JointStateTopologyObserver` with
  ``claim_homomorphism=True`` raises unless the encoder is
  :class:`~koopman_graph.nn.SeparableDictionaryEncoder` and the
  operator is a one-tap graph Koopman operator. The observer writes
  :math:`K_{\mathrm{self}}` / :math:`K_{\mathrm{nbr}}` on the
  **observed** COO; it does **not** infer a new edge set, does not
  implement Koopman-GKFA ADMM, and does not certify a three-term MSE
  bound. The 2026 arXiv preprint is provisional.
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
  radius matters. Spectral radius (and
  :class:`~koopman_graph.operators.StabilityCertificate`) is **not** a
  finite-horizon bound on :math:`\|K^{k}\|`.
* **Eigenvector conditioning is not a forecast bound.**
  :class:`~koopman_graph.spectrum_types.SpectralDiagnostics` reports
  :math:`\kappa(V)`, per-mode Wilkinson :math:`\kappa_i`, and Frobenius
  departure from normality on spectra from ``compute_spectrum``.
  Discrete spectra also report Nyquist frequency :math:`1/(2\Delta t)`
  in **cycles per unit time** (``None`` on generator spectra), a
  per-mode aliasing flag when the argument is near :math:`\pi`, and
  :math:`\operatorname{sign}(\operatorname{Re}\lambda)`. Reported
  frequencies are principal values; phase unwrapping does not recover
  aliases, and :math:`\log|\lambda|` discards sign.
  :func:`~koopman_graph.analysis.implied_timescales` uses the same
  aliasing flag without changing the :math:`|\lambda|` validity window.
  Finite-horizon transient growth of a non-normal map can exceed
  :math:`\rho(K)^{t}`. ``mode_amplitudes`` emits a warning when
  :math:`\kappa(V)` exceeds ``CONDITION_WARN`` and still solves
  :math:`Va=z`; singular :math:`V` raises
  :class:`~koopman_graph.spectrum_types.DefectiveSpectrumError` with a
  Schur-subspace hint (not an automatic Schur fallback).
  These diagnostics do not replace
  ``stability_certificate``, are not resolvent-norm certificates, and
  are not a sampling-theorem identification result.
  Resolvent grids remain a separate opt-in helper
  (:func:`~koopman_graph.analysis.resolvent_norm_grid`).
* **Measure-preserving EDMD is not a conditioning certificate.**
  :class:`~koopman_graph.baselines.MpEDMDBaseline` enforces unitarity
  in a Gram-weighted dictionary inner product
  (``Colbrook2023mpEDMD``). It does **not** replace
  :class:`~koopman_graph.spectrum_types.SpectralDiagnostics`
  (:math:`\kappa(V)`, Wilkinson :math:`\kappa_i`, departure from
  normality) on a general directed :math:`K_{\mathrm{eff}}`.
* **Orbit ties and isotypic MVP are inductive biases.** Orbit-label ties
  share ``K_{\mathrm{self}}`` blocks; they are not automatic isotypic
  reduction. Opt-in ``koopman_symmetry="isotypic"`` (exact automorphism
  groups on modest :math:`N`; see ``examples/45_isotypic_symmetry.ipynb``)
  ties isotypic projectors for the self block and, on discrete graph /
  hypergraph / Hodge operators, the neighbor factor
  (:math:`K_{\mathrm{nbr}}` / :math:`K_{\mathrm{hedge}}`). Dual
  :math:`K_{\mathrm{bwd}}` stays globally shared. Shared-:math:`d` typed
  hetero orbits are partitioned independently within each type block;
  rectangular typed orbits are unsupported. Neither path guarantees
  sample-efficiency wins
  (see :ref:`limitations-remaining`).
* **Stochastic mode is process noise, not an SDE.** Opt-in
  ``dynamics_mode="stochastic"`` adds learned diagonal process noise after a
  discrete linear map (:math:`z \mapsto Kz+\varepsilon`); it is not a
  continuous-time stochastic generator.
  :class:`~koopman_graph.operators.DriftDiffusionKoopman` is a separate
  Euler–Maruyama / Yosida MVP (conditional-expectation semigroup vs
  sampled paths). It is not certified Itô theory and not SDMD
  (``Xu2025StochasticSemigroup``, ``Zhou2025Yosida``).
* **VAMP-2 precursor and GraphVAMP teaching path.** Optional
  :func:`~koopman_graph.baselines.vamp2.vamp2_score` /
  :func:`~koopman_graph.baselines.vamp2.vamp2_loss` and
  ``LossWeights.vamp2`` act on flattened encoder latents (topology-blind).
  :class:`~koopman_graph.baselines.GraphVAMPBaseline` plus
  :mod:`koopman_graph.datasets.molecular` (synthetic contact-graph oracle;
  optional ``[md]`` / ``[msm]``) form a **teaching / diagnostic** MD-adjacent
  toolchain, not GraphVAMPnets production software, not a PyEMMA
  replacement, and not Folding@home-scale MD. A packaged alanine-dipeptide
  teaching card and ``[md]`` fetch helper are available; CI still uses the
  synthetic oracle (see :ref:`limitations-remaining`).
* **Bayesian Laplace UQ over operator factors.** 
  :class:`~koopman_graph.uq.BayesianKoopmanUQ` provides a diagonal Laplace
  posterior over linear Koopman factors with seeded sample forecasts. It is
  **not** a Bayesian neural net over nonlinear encoder weights.
  :mod:`koopman_graph.probabilistic` ships a distinct :math:`K^{2}` /
  deep-probabilistic Koopman VAE MVP (variational encoder weights plus a
  linear latent :math:`K`). Ensemble, latent-Gaussian, and conformal
  surfaces remain available (see below).
* **Invariant geometry encode (Tier A).** 
  :class:`~koopman_graph.nn.equivariant.InvariantGeometryEncoder` builds
  rotation-/translation-invariant distance and angle features from
  ``Data.pos`` and lifts with a standard GCN. **Invariant features do not
  make the latent operator** :math:`K` **E(n)/SE(3) equivariant.** Optional
  Tier B :class:`~koopman_graph.nn.equivariant.E3EquivariantEncoder`
  (``e3nn``, ``[equivariance]``) still **defaults** to invariant scalar
  latents. Pass ``project_invariants=False`` together with
  :class:`~koopman_graph.operators.EquivariantKoopmanOperator` for a
  block :math:`K` whose vector channels are learnable multiples of
  :math:`I_3` and whose optional :math:`l=2` channels are multiples
  of :math:`I_5`. This is not a full steerable-generator library and
  not a molecular MD production stack (see :ref:`limitations-remaining`).

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
  not certificates (see Spectral analysis).   Continuous ``matrix_log``
  raises :class:`~koopman_graph.spectrum_types.DefectiveSpectrumError` on
  non-diagonalizable (Jordan) or numerically singular eigenbases by
  default (``defective="error"``); opt-in ``defective="schur"`` uses
  SciPy ``logm`` on CPU and is not a topology-criticality certificate
  (:mod:`koopman_graph.operators.continuous_van_loan`).
  :func:`~koopman_graph.analysis.monitor_critical_transition` is an
  opt-in sliding-window spectral-gap / near-defectivity heuristic on a
  sequence of spectra; it is not a Ghosh-grade early-warning certificate
  and does not mitigate forecast-horizon collapse.

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
* **Granger-style scores stay non-interventional.**
  :func:`~koopman_graph.analysis.granger_latent_influence` reports
  linear residual-MSE reduction on observed latents. The labeled
  synthetic helper
  :func:`~koopman_graph.analysis.recover_synthetic_interventional_edges`
  applies paired do-interventions on
  :func:`~koopman_graph.analysis.teaching_three_node_scm` only. That
  fixture is not field-data causal discovery and is not a factory
  kind. ``FORMAT_VERSION`` remains 1.
* **Largest Lyapunov exponent (Rosenstein) is available but sensitive.**
  ``koopman_graph.statistics.largest_lyapunov_exponent`` implements the
  Rosenstein et al. (1993) small-data-set method
  (``Rosenstein1993Lyapunov``). Estimates depend on delay,
  embedding dimension, Theiler window, and fit range; nearest-neighbor search
  is :math:`O(T^{2})`. Construction oracles use documented hyperparameters
  (see the statistics unit tests); bare defaults are not a high-precision
  spectral computation.

Identification
--------------

* **Closed-form identification does not replace Adam** ``fit``.
  Default ``identification=None`` keeps the existing Adam path and
  does not import :mod:`koopman_graph.identification` at model load.
  ``fit(..., identification=IdentificationConfig(...))`` is opt-in
  and currently writes dense per-node discrete :math:`K` only
  (graph / hetero / continuous / controlled / delay / windowed /
  DDP layouts raise). ``solver="varpro"`` raises
  ``NotImplementedError``.
  :class:`~koopman_graph.identification.IdentificationReport`
  records latent one-step / short-rollout mean squared error (MSE)
  and :math:`\rho(K)` separately from optional invariance leakage
  and finite-dictionary ResDMD residuals. Those blocks are not
  Haseli–Cortés certificates and not spectral-measure certificates.
  :func:`~koopman_graph.identification.select_latent_rank` is not
  Ray Tune AutoML for encoder ``latent_dim``. See :doc:`faq` and
  :ref:`limitations-remaining`.

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

  1. ``sparsity="distributed"`` — on discrete graph, multiplex hetero,
     Zhou-symmetric hypergraph, and continuous-graph generators,
     matrix-free Arnoldi returns leading-modulus Ritz values (placeholder
     eigenvectors of size ``num_modes``). Continuous-graph distributed
     spectrum is an Arnoldi surrogate on **generator factors**, not a
     certified continuous spectrum.
  2. Else if Kronecker-sum eligible — shared self factor,
     ``adjacency`` in ``{"symmetric", "random_walk"}``, and
     ``sparsity`` in ``{"dense", "block_diagonal"}`` — exact reduction.
     Discrete :math:`K` uses
     :math:`B(\lambda)=\sum_k\lambda^k K_k` for any hop degree
     :math:`P\ge 0` (not a sum of independent factor eigenvalues).
     Continuous generator :math:`L` stays the one-tap form
     :math:`I\otimes L_{\mathrm{self}} + \widehat{A}\otimes L_{\mathrm{nbr}}`.
     Cost is dominated by a dense :math:`N\times N` eigendecomposition of
     :math:`\widehat{A}` plus :math:`N` blocks of size :math:`d\times d`
     (order :math:`O(N^3 + N d^3)`), not a full ambient
     :math:`O((N\cdot d)^3)` factorization when the helper succeeds.
  3. Else — dense :math:`(N\cdot d)\times(N\cdot d)` eigendecomposition
     of the assembled effective map (``dual_random_walk``, discrete
     orbit / isotypic self banks, helper fall-back, and other ineligible
     cases).

  Hetero spectrum remains under the dense assembled
  :math:`(N\cdot d)` (or stacked typed) ceiling unless
  ``sparsity="distributed"`` (Arnoldi surrogate). Zhou-symmetric
  hypergraph distributed spectrum / inverse uses the matrix-free path;
  other incidence modes may still assemble. Kronecker spectrum does
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
* **Multi-graph ``Batch`` collate is opt-in vectorization.**
  ``fit(..., batch_graphs=True)`` collates independent homogeneous
  trajectories so shared :math:`K` applies to a disconnected union. The
  default ``MultiTrajectory`` Python loop is unchanged. Multi-topology
  training did not require this flag. Hetero, hypergraph, windowed, and
  DDP graph-batching raise.
* **Eigenvalue regularization** with a non-zero
  ``LossWeights.eigenvalue`` on dense or ODO networked operators runs an
  :math:`O((N \cdot d)^3)` eigendecomposition of the assembled effective map
  unless ``sparsity="distributed"`` (Arnoldi surrogate on discrete graph /
  multiplex hetero; zero hinge on hypergraph / continuous distributed).
  The Kronecker spectrum path does **not** change this training-hinge
  ceiling. Prefer structural parameterizations (``schur`` / ``lyapunov`` /
  ``dissipative``) or modest :math:`N` when the assembled term is enabled.
* **``sparsity="distributed"``** is the *operator* matrix-free path
  (Richardson / Neumann inverse and Arnoldi spectrum for discrete graph,
  multiplex hetero, and Zhou-symmetric hypergraph). Continuous-graph
  distributed **inverse** is uncontrolled implicit Euler on
  :math:`I - \Delta t L_{\mathrm{eff}}` — not a matrix exponential.
  It is **not** multi-GPU training and is **not** the Kronecker
  exact-spectrum reduction. Optional *trainer* orchestration under
  :mod:`koopman_graph.distributed` shards data and synchronizes
  gradients; it does **not** shrink the dense :math:`N\cdot d`
  representation ceilings above.
  :class:`~koopman_graph.operators.LinearOperatorProtocol` unifies
  polynomial-graph and one-tap ``matrix_free`` wrappers
  (``matvec`` / ``rmatvec`` / ``solve`` / ``expm_action`` / leading
  eigpairs). Dense assembly is refused above
  :data:`~koopman_graph.operators.MAX_DENSE_LINEAR_OPERATOR_SIZE`.
  Trainer DDP is **not** a smaller operator representation.
  :math:`10^{5}`-node scaling is not a release gate.
* **Trainer taxonomy (DDP / Fabric / Ray).** Native DDP and Lightning Fabric
  remain the **recommended** multi-GPU *model* paths.
  :func:`~koopman_graph.distributed.run_ray_train_fit_loop` (optional
  ``[ray]`` / ``[distributed]``) wraps Ray Train ``TorchTrainer`` around the
  same scientific fit loop (model DDP under Ray). Separately,
  :func:`~koopman_graph.distributed.fit_ensemble_with_ray` parallelizes
  *ensemble members*. Prefer DDP / Fabric unless you already standardize on
  Ray Train. Multi-node / cluster Ray Train is documented as an opt-in
  recipe (``KOOPMAN_GRAPH_MULTINODE=1``); default CI stays single-process
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
  :class:`~koopman_graph.uq.JointCoverageSpec` names that estimand
  (``target="per_node_marginal"``). Simultaneous node–feature–horizon
  and event coverage are named but not implemented
  (``Schlembach2025Conformal``).
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
  **teaching baselines**, not dedicated-library SOTA. Protocol-matched
  LibCity / BasicTS adapters live on
  :class:`~koopman_graph.baselines.gnn.LeaderboardProtocol` (empty
  ``deviations`` only when the named protocol truly matches). GraphCast is a
  small-mesh weather teaching adapter, not a PEMS sensor-graph forecaster
  and not ERA5-scale production training (see :ref:`limitations-remaining`).
* **Example 22 teaching comparison.**
  Saved METR-LA weekday-cache output ranks GraphKoopman first on
  aggregate RMSE (z-scored speed): :math:`0.6551` versus STGCN
  :math:`0.7076`, Graph WaveNet :math:`0.9036`, and DCRNN
  :math:`1.0754`. GraphKoopman uses a longer ODO / rollout / early-stopping
  budget than the GNN teaching refs (unequal budgets); STGCN remains
  strongest at the shortest horizon. These are in-repo teaching
  baselines, not dedicated-library SOTA. See
  ``examples/22_gnn_forecaster_comparison.ipynb``.
* **Classical DMD family.** Shipped baselines include DMD, EDMD (including
  Nyström / random-feature kernel approximations), measure-preserving EDMD
  (``MpEDMDBaseline``; unitarity is in the dictionary Gram inner product,
  not :class:`~koopman_graph.spectrum_types.SpectralDiagnostics` and
  not a Euclidean certificate on general directed
  :math:`K_{\mathrm{eff}}`),
  generator EDMD (``GEDMDBaseline``; supplied dictionary derivatives,
  not snapshot finite differences or derivative-mode SINDy; irregular
  :math:`\Delta t` does not create :math:`L`),
  Hankel-DMD (``HankelDMDBaseline``; delay-embedded flattened
  states, not :class:`~koopman_graph.nn.delay.DelayEmbeddingEncoder`),
  teaching HAVOK (``HAVOKBaseline``; same delay rows; autonomous
  ``predict`` uses :math:`u=0`; omitted ``history`` zero-pads older
  delays),
  DMDc, forward–backward
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

* **Leaderboard adapters are not SOTA.**
  :class:`~koopman_graph.baselines.gnn.LeaderboardProtocol` follows named
  LibCity / BasicTS splits and 12-step horizons on METR-LA / PEMS-style
  caches. Teaching ports still declare non-empty ``deviations``. Adapter
  numbers are protocol-matched comparisons, not dedicated-library
  competition entries.
  :class:`~koopman_graph.benchmark.ExperimentManifest` locks that protocol
  in a frozen schema (dataset SHA-256, seeds, controls). The CLI
  ``run`` / ``verify`` commands hash protocol identity; they do not
  train models, invent forecast metrics, or host a leaderboard.
  Default CI verifies hashed synthetics under ``benchmarks/v0.15/``;
  full METR-LA / PEMS downloads stay documented, not a pull-request
  gate.
* **ERA5-scale / production GraphCast** training. The in-repo adapter is a
  documented small-mesh teaching slice plus an ERA5-slice fetch/checksum
  card; CI uses a tiny generated mesh.
* **Open-world remap is user-supplied, not entity resolution.**
  :class:`~koopman_graph.data.EntityRemap` and
  :func:`~koopman_graph.data.remap_node_features` grow :math:`N_{\max}`
  with an injective index map and presence masks. Silent universe merge
  is refused. Sequences that change node count without a remap raise.
  Automatic identity matching across unrelated graphs is out of scope.
* **Graphon estimation is a dense teaching fit, not sparse-limit
  theory.** :func:`~koopman_graph.operators.estimate_graphon` recovers
  constant and product kernels on aligned graphs with a shared
  :math:`N` (ceiling 256). It does not identify a unique graphon, does
  not implement Borgs–Chayes–Lovász sparse limits, and does not certify
  GNN transferability on arbitrary sparse sensor graphs
  (``Ruiz2023Transferability``). Size-transfer remains sampling at two
  :math:`N`.
* **Regime coordinates are a data record; the interpolant is opt-in.**
  Homogeneous sequences may carry ``parameter_trajectory`` and
  :class:`~koopman_graph.data.ConditioningContext`. That does not
  change the default LTI operator. ``koopman="switched"`` and
  ``"mixture"`` remain latent-gated or piecewise LTI maps
  (``Macesic2018Nonautonomous``). ``koopman="parametric"`` is a
  discrete per-node convex combination
  :math:`K(\\mu)=\\sum_j \\alpha_j(\\mu) K_j`, not a cocycle
  :math:`K(t)` and not a graph-factorized :math:`K_{\\mathrm{eff}}(\\mu)`.
  A convex combination preserves dense / row-stochastic / doubly-stochastic
  factors; symplectic and other structural mixes raise. Export refuses
  the interpolant. Discrete sequences still require uniform
  :math:`\\Delta t`.
* **The drift–diffusion stepper is an Euler–Maruyama / Yosida MVP.**
  :class:`~koopman_graph.operators.DriftDiffusionKoopman` is not a
  factory kind and does not change ``dynamics_mode="stochastic"``.
  It is not certified SDE theory, not gEDMD, and not SDMD.
  :class:`~koopman_graph.uq.JointCoverageSpec` names
  ``per_node_marginal`` coverage; simultaneous / event targets raise.
* **Cochain dynamics are node/edge maps, not a TDL stack.**
  :class:`~koopman_graph.operators.CochainKoopmanOperator` advances
  0- and 1-cochain latents on a static signed :math:`B_1`. It is not
  a factory kind and does not change ``koopman=None`` /
  ``"pernode"``. Distinct from ``koopman="hodge"``. Face latents may
  be stored; :math:`k=2` is not evolved.
  :func:`~koopman_graph.operators.boundary_nilpotency` flags
  :math:`B_1 B_2\\approx 0` at atol ``1e-6``. Not sheaf Laplacian
  theory and not TopologicX parity (``Lim2020Hodge``, ``TopoX2024``).
* **TDL teaching stops at cell-complex degree 3.**
  :func:`~koopman_graph.nn.order2_cochain_teaching` binds
  :class:`~koopman_graph.operators.CochainKoopmanOperator` to a filled
  2-simplex. Optional tetrahedra reach
  :data:`~koopman_graph.nn.MAX_CELL_COMPLEX_DEGREE`. Face latents may
  be stored; :math:`k=2` is not evolved. Sheaf restriction maps stay
  learned-optional (default diagonal). This is still not TopologicX
  / TDA ecosystem parity (``TopoX2024``).
* **Hodge mode components are combinatorial, not physical circulation.**
  :func:`~koopman_graph.analysis.hodge_decompose_modes` splits stored
  eigenvector columns on a static signed :math:`B_1`
  (:math:`k\\in\\{0,1\\}`). It is analysis-only, not a factory kind,
  and not ``koopman="hodge"``. Face-based :math:`k=1` curl is not
  shipped. Do not treat the harmonic block as a validated current
  (``Lim2020Hodge``, ``TopoX2024``).
* **Finite memory is a whiteness diagnostic plus a convolution MVP.**
  :func:`~koopman_graph.analysis.markov_closure_report` is a
  Ljung–Box-style flag on residual energy (lags in timesteps), not a
  certified Markov-closure test (``Ljung1978Box``).
  :class:`~koopman_graph.analysis.FiniteMemoryKoopman` stores
  :math:`\\Omega` and taps :math:`K_s`. It is not a factory kind, not
  :class:`~koopman_graph.nn.delay.DelayEmbeddingEncoder`, and not
  HAVOK. Delay embedding and HAVOK are not Mori–Zwanzig decompositions
  (``Brunton2017HAVOK``, ``Lin2021MoriZwanzig``). Memory-order
  selection is not implemented.
* **Time-of-day is a control / phase recipe, not a calendar serializer.**
  :func:`~koopman_graph.data.diurnal_control_features` writes Fourier
  columns onto existing ``control_inputs``.
  :func:`~koopman_graph.data.diurnal_phase_index` fills an optional
  switched ``phase_index``. There is no native calendar field and no
  checkpoint key. Discrete ``fit`` / ``predict_at`` still reject
  non-uniform increments. Use ``dynamics_mode="continuous"`` or
  supply gEDMD derivatives for irregular sampling. Irregular
  timestamps do not create :math:`L` (``Klus2020gEDMD``).
* **Multi-node Ray is opt-in.** Default CI stays single-process. Set
  ``KOOPMAN_GRAPH_MULTINODE=1`` for the documented ``torchrun`` / Ray Train
  recipe smoke. Measured cluster speedups are outside that contract.
* **TDA / TDL extras are bridges, not replacements.**
  :mod:`koopman_graph.analysis.tda` ships 0-dimensional union-find
  persistence (optional ``[tda]``). The ``[tdl]`` TopologicX helper
  converts an external complex into in-repo incidence tensors. Neither
  replaces a dedicated TDA library or TopologicX feature parity.
* **Alanine-dipeptide fetch is not Folding@home.** The ``[md]`` loader and
  packaged card are a public teaching path on real coordinates. GraphVAMP
  + deeptime interop remain diagnostic; they are not a PyEMMA replacement.
* **Guaranteed sample-efficiency wins** from isotypic ties, transfer APIs,
  or leaderboard adapters. Negative transfer advantage and null ablations
  are allowed outcomes. Neighbor-factor isotypic tying is shipped; it is
  still not a guaranteed sample-efficiency win.
* **Finite-dictionary ResDMD.** Residual norms may attach to
  ``evaluate(..., include_resdmd=True)`` and to an optional fit callback.
  The dictionary remains finite — not certified infinite-dimensional
  spectra or spectral measures.
  :func:`~koopman_graph.identification.select_resdmd_gated` and
  ``ResDMDFitCallback(mode="gate")`` use the same max-residual cutoff
  (default :math:`10^{-2}`). They may reject an RMSE-only winner; they
  do not certify a spectral measure.
* **Finite-sample subspace invariance.** ``evaluate(...,
  include_invariance=True)`` and
  :meth:`~koopman_graph.model.GraphKoopmanModel.subspace_invariance_report`
  report projection leakage :math:`\eta` on a truncated-SVD encoding
  span. That ratio is not the Haseli–Cortés invariance-proximity
  certificate (``HaseliCortes2023``). Discrete dense per-node
  :math:`K` only.
* **Sparse graph-factor identification is not latent SINDy.**
  :func:`~koopman_graph.identification.identify_sparse_graph_factors`
  thresholds :math:`K_{\\mathrm{self}}` / :math:`K_{\\mathrm{nbr}}` on
  frozen encodings. :func:`~koopman_graph.analysis.identify_sparse_dynamics`
  and :class:`~koopman_graph.losses.KoopmanSparsityLoss` still ship.
  Dual random-walk and polynomial :math:`P>1` maps are out of scope.
* **Joint state–topology observation is not Koopman-GKFA.**
  :class:`~koopman_graph.adaptation.JointStateTopologyObserver` filters
  latents then updates graph factors or dense :math:`K`. It does not
  recover a new adjacency :math:`A`, does not run ADMM, and does not
  certify a three-term MSE bound. Homomorphism claims require a
  separable dictionary (``Peng2026KoopmanGKFA``).
* **Latent-rank selection is not AutoML for** ``latent_dim``.
  :func:`~koopman_graph.identification.select_latent_rank` truncates a
  frozen encoding cloud. :mod:`koopman_graph.tuning` Ray Tune helpers
  remain example scaffolds; they are not a scientific default for
  encoder width.
* **Decoded constraint heads do not make latent symplectic**
  :math:`K` **a mass certificate.**
  :class:`~koopman_graph.nn.MassConservingDecoder`,
  :class:`~koopman_graph.nn.LinearConservingDecoder`, and
  :class:`~koopman_graph.nn.PositivityDecoder` project named
  channels **after** decode (simplex or affine
  :math:`\\mathbf{1}^{\\top}x=M`, linear :math:`Cx=c_0`, or
  positivity). Row- / doubly-stochastic and symplectic maps still
  constrain only the operator matrix (``Greydanus2019HNN``).
  IEEE-118 ``generate`` / ``generate_typed`` remain Laplacian
  diffusion plus a load ramp — not SCADA, OPF, or AC power flow.
  Constraint heads do not create power-flow consistency.
* **Tube Koopman-MPC is residual-box tightening, not a closed-loop
  certificate.** :class:`~koopman_graph.mpc.TubeKoopmanMPC`
  erodes nominal output boxes by conformal or ensemble residual
  radii on additive discrete plants and reports
  constraint-violation rate, feasibility rate, and quadratic
  stage cost. Local decoder linearization remains. The helper is
  not a chance-constraint solver, not a recursive-feasibility
  proof, and not a Lyapunov closed-loop certificate. Zhang et
  al., *Automatica* 137:110114 (2022), address an r-KMPC scheme
  with an offline nonlinear ancillary law
  (``Zhang2022TubeMPC``); this MVP does not implement that
  controller. Bilinear, networked, and continuous operators
  raise. ``FORMAT_VERSION`` remains 1.
* **Equivariant** :math:`K` **is a block MVP.** Vector channels are
  learnable multiples of :math:`I_3`. One additional irrep
  (:math:`l=2`) is ``scale * I_5`` (``n_tensors``). Irreps are
  not mixed. Default :class:`~koopman_graph.nn.E3EquivariantEncoder`
  still projects to invariant scalars. This is not a full steerable
  generator library and not a molecular MD production stack
  (``Thomas2018TFN``, ``Geiger2022e3nn``).
* **Portable export is a restricted MVP.** :mod:`koopman_graph.export`
  covers fixed-topology, discrete, homogeneous, ``n_delays=1``, no
  adaptive adjacency. Hetero / hypergraph / control / switched kinds are
  refused. The path is not edge-device certified. ONNX is a second-tier
  converter behind ``[export]``.
* **Federated averaging is not differential privacy.**
  :mod:`koopman_graph.federated` keeps data on the client (FedAvg). It is
  not DP-SGD.
* **FDI tests are data-integrity, not pickle trust.** Sensor-corruption
  helpers evaluate forecast / certificate sensitivity. Checkpoint
  adversarial weights remain a ``SECURITY.md`` concern, distinct from
  this threat model.
* **Continuous-graph distributed inverse is implicit Euler**,
  uncontrolled, on :math:`I - \Delta t L_{\mathrm{eff}}` — not a
  certified matrix exponential.
* **Changing homogeneous scientific defaults** (AMP off by default;
  sparsity default ``"dense"``; linear time-invariant latent advance).

.. _limitations-when-else:

When to use something else
--------------------------

* **Traffic forecasting leaderboards.** Prefer dedicated traffic-forecasting
  libraries and their maintained SOTA models when the goal is competition
  numbers. KoopmanGraph’s leaderboard adapters match named protocols for
  comparison; they are not a forecasting competition entry. Teaching ports
  still declare deviations.
* **Full sheaf / cell-complex TDL stacks.** Prefer libraries built for rich
  sheaf Laplacians and cellular TDL when the domain model requires
  complex-native operators end-to-end. KoopmanGraph’s sheaf / cell MVPs,
  Hodge-structured :math:`K`, the order-2 cochain teaching hook, and
  the TopologicX incidence bridge are precursors with a linear
  Koopman head. Cell-complex degree 3 is the teaching ceiling, not a
  TopologicX replacement.
* **Production biomolecular Koopman / MSM analysis.** Prefer GraphVAMPnets /
  PyEMMA-scale MSM toolchains aimed at molecular trajectory data when you
  need that production surface. The alanine-dipeptide teaching fetch and
  GraphVAMP path are not that toolchain.
* **Certified infinite-dimensional spectral computation.** Prefer full
  ResDMD / spectral-measure packages when you need residual-based
  certificates beyond the finite-dictionary MVP wired into ``evaluate``.
* **Edge-certified portable inference.** 0.14.0 ships a **restricted**
  TorchScript / ``torch.export`` (and optional ONNX) path for
  fixed-topology discrete homogeneous models. Share general weights with
  ``safetensors_v1``. Hetero / hypergraph / control / switched export is
  refused.
* **Certified SDE / SDMD generators.** Prefer dedicated stochastic
  Koopman / SDMD toolchains when the object of study is a certified
  semigroup or generator identification method
  (``Xu2025StochasticSemigroup``, ``Zhou2025Yosida``). The in-repo
  stepper is a discrete-semigroup approximation.
* **Native calendar / irregular discrete Koopman.** Discrete ``fit``
  still requires uniform :math:`\\Delta t`. Time-of-day helpers are
  Fourier / phase-bin recipes on existing control and switched APIs.
  Prefer a dedicated calendar-aware or continuous-time toolchain when
  the sampling grid itself is the scientific object.
* **Flat-vector deep Koopman without graph structure.** Packages such as
  PyKoopman (``PyKoopman2024``), PyDMD (``PyDMD2018``), kooplearn
  (``kooplearn2026``), and DLKoopman remain appropriate when the state is
  naturally a vector and edge-based coupling is not part of the model.

Related pages
-------------

* :doc:`capabilities` — feature inventory
* :doc:`architecture` — public vs power-user API contract
* :doc:`data` — dataset cards and per-dataset limitations
* :doc:`tutorials` — notebook gallery
* :doc:`faq` — install and runtime troubleshooting
* :doc:`graphon` — graphon sampling, dense teaching estimation, and
  continuum-limit citations
* :doc:`benchmarks` — identity-bound manifests and smoke fixtures
* :doc:`identification` — opt-in solvers and finite-sample reports
* :doc:`spectral_diagnostics` — conditioning and Nyquist fields
* :doc:`graph_dynamics` — recursive topology head versus hold-last
* :doc:`matrix_free` — dense-assembly ceiling and Arnoldi Ritz pairs
* :doc:`criticality` — gap-closure heuristic (not Ghosh)
* :doc:`time_conditioning` — uniform :math:`\Delta t` and calendar recipes
