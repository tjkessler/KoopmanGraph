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
:ref:`limitations-when-else` are **scoped non-goals**, not overlooked gaps.

Graph structure
---------------

* **Default coupling is undirected-symmetric.** Networked operators
  (``koopman="graph"``) default to ``adjacency="symmetric"`` (normalized
  undirected adjacency). Directed coupling is available via
  ``adjacency="random_walk"`` (row-normalized
  :math:`D_{\mathrm{out}}^{-1}A`) or ``adjacency="dual_random_walk"``
  (forward plus reverse walks). Hypergraph operators keep Zhou-style
  symmetric incidence normalization; directed hypergraph coupling is out of
  scope.
* **Fixed node cardinality per sequence.** ``GraphSnapshotSequence`` locks
  :math:`N` to the first snapshot. Unobserved nodes are handled with
  ``observation_masks``; variable node cardinality / node churn is not
  supported.
* **Homogeneous single edge type.** Heterogeneous and multiplex graphs
  (multiple node or edge types) are out of scope.
* **Hypergraphs, not simplicial complexes.** Incidence-based hypergraph
  encode / decode / operators are supported. Simplicial, cell-complex,
  sheaf, and Hodge-Laplacian layers are out of scope.
* **Cross-topology transfer is not automatic.** On a seeded path-diffusion
  transfer from :math:`N_1=4` to :math:`N_2=6` nodes, the factorized graph
  operator achieved hold-out mean squared error (MSE) approximately
  :math:`0.26` both in-distribution and after the node-count change; a
  per-node dense control reached approximately :math:`0.21`. That run did
  **not** show a transfer advantage for the graph factorization.
  Self-adaptive topology and orbit-tied ``K_{\mathrm{self}}`` configurations
  raise on node-count changes and are excluded from any transfer claim.
  See ``examples/37_cross_topology_transfer.ipynb``.

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
* **Orbit ties are an inductive bias**, not isotypic / irreducible-
  representation block diagonalization. A representation-theoretic reduction
  would require automorphism generators beyond orbit labels and is out of
  scope.
* **No stochastic Koopman generator**, transfer-operator dual, or VAMP-2
  objective family.
* **No Bayesian prior over the operator.** Uncertainty surfaces are ensemble,
  latent-Gaussian, and conformal (see below), not posterior sampling of
  :math:`K`.
* **No molecular-dynamics / Markov-state-model toolchain** (for example
  VAMPnets / GraphVAMPnets).
* **No E(n) / SE(3) equivariance** for coordinate-bearing nodes.

Spectral analysis
-----------------

* **Data-driven residuals are diagnostics, not certified ResDMD bounds.**
  ``koopman_graph.analysis.spectral_residuals`` measures whether claimed
  eigenpairs propagate as :math:`a(t+1) \approx \lambda\, a(t)` in the
  *learned* latent / observable space on held-out trajectories. That is
  distinct from residual dynamic mode decomposition (ResDMD), which builds
  Galerkin matrices for :math:`\mathcal{K}^{*}\mathcal{K}` in a fixed
  dictionary and supplies error control on the infinite-dimensional spectrum
  (Colbrook & Townsend, *Commun. Pure Appl. Math.*, 2023/2024;
  ``ColbrookTownsend2023ResDMD``; see also ``Colbrook2023ResidualDMD``).
  Full ResDMD (certified pseudospectra and spectral measures) is out of scope.
* **No pseudospectra or spectral-measure estimators** are shipped.
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

* **Exact spectrum and exact inverse** of networked operators assemble an
  effective dense matrix of size :math:`(N \cdot d) \times (N \cdot d)`.
  Prefer modest :math:`N` or ``sparsity="block_diagonal"`` (approximate
  Jacobi / self-dominated path) when that cost dominates. Static topology
  may reuse a precomputed dense inverse within a training-loss evaluation;
  the assembly size is unchanged.
* **Eigenvalue regularization** with a non-zero
  ``LossWeights.eigenvalue`` on dense or ODO networked operators runs an
  :math:`O((N \cdot d)^3)` eigendecomposition of the effective map. Prefer
  structural parameterizations (``schur`` / ``lyapunov`` / ``dissipative``)
  or modest :math:`N` when that term is enabled.
* **``sparsity="distributed"`` is not implemented** and continues to raise
  ``ValueError``. Do not confuse this reserved operator flag with optional
  *trainer* orchestration under :mod:`koopman_graph.distributed` (native
  DDP / Lightning Fabric). Trainer DDP shards data and synchronizes
  gradients; it does **not** shrink the dense :math:`N\cdot d`
  representation ceilings above.
* **Multi-process / multi-GPU CI is not guaranteed.** Default test jobs
  stay single-process. Optional gloo smokes are opt-in
  (``KOOPMAN_GRAPH_DISTRIBUTED_TESTS=1`` / ``@pytest.mark.distributed``).
  Multi-node launches and measured speedups are outside the documented
  contract.
* **No Dask training loop.** Dask is docs-only in 0.8.0: the library does
  not import Dask on the training path, and the ``[dask]`` extra is unused
  by public APIs (reserved pin). Offline materialization of trajectories
  or window lists may use Dask in user code; training remains PyTorch /
  :mod:`koopman_graph.distributed` (see :doc:`faq`).
* **DiffConv** bidirectional diffusion supports are dense
  :math:`N \times N` tensors for static graphs. Identical
  ``edge_index`` / weight storage may reuse cached supports
  (``clear_support_cache``); caching does not sparsify the representation.
* **Continuous-graph dense advances** form
  :math:`\Phi = \exp(\Delta t\, L_{\mathrm{eff}})` on the same
  :math:`(N \cdot d) \times (N \cdot d)` scale. Prefer
  ``sparsity="block_diagonal"`` (self-dominated shortcut) for large
  :math:`N`. Within one training-loss evaluation the dense path may reuse
  :math:`\Phi` (and assembled :math:`L_{\mathrm{eff}}`) for repeated
  topology / :math:`\Delta t` keys; see :doc:`faq`.
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
  (``koopman_graph.uq.ConformalKoopmanUQ``).
* **``score="node_wise"``** yields per-node marginal half-widths (optional
  neighbor smoothing), not joint coverage across nodes. The legacy
  ``score="per_node"`` mode still max-pools over nodes into a scalar score;
  it is not heteroscedastic per node.
* **Ensemble and latent-Gaussian UQ** are power-user surfaces with their own
  modeling assumptions (empirical quantiles; linear-Gaussian latent
  propagation). They are not deep probabilistic Koopman (DPK) or a full
  :math:`K^{2}` variational autoencoder.

Benchmarks and baselines
------------------------

* **Measured telemetry** (observed time series): METR-LA, PEMS-BAY,
  PEMS03/04/07/08, and SocioPatterns contact networks (see :doc:`data` for
  acquisition and license cards).
* **Simulated or surrogate dynamics:** synthetic Laplacian / advection
  graphs, networked SIR, Lorenz-96, Kuramoto–Sivashinsky, IEEE 118 voltage /
  load diffusion, and the Hopf / Stuart–Landau cylinder-wake teaching
  surrogate. IEEE 118 uses a real bus topology with *simulated* dynamics, not
  SCADA telemetry.
* **In-repo GNN forecasters** (STGCN, DCRNN, Graph WaveNet in
  ``koopman_graph.baselines.gnn``) are **teaching baselines**, not
  reproductions of dedicated-library state-of-the-art (SOTA) implementations.
  Adding further SOTA spatiotemporal GNN baselines (for example AGCRN,
  MTGNN, STGODE, GraphCast) is out of scope.
* **Classical DMD family coverage is partial.** Shipped baselines include
  DMD, EDMD, and DMDc. Deferred on scope grounds: optDMD / variable
  projection, total-least-squares DMD, forward-backward DMD, multi-resolution
  DMD, streaming DMD, and Nyström / random-feature kernel EDMD.
  ``RecursiveKoopmanAdapter`` is not a streaming-DMD substitute; it refines a
  learned latent operator and requires dense parameterization.
* **Long-horizon chaotic evaluation** should report distributional statistics
  (``koopman_graph.statistics``) alongside short-horizon RMSE; pointwise error
  alone saturates past the predictability horizon. See
  ``examples/24_nonlinear_chaotic_benchmarks.ipynb``.

.. _limitations-when-else:

When to use something else
--------------------------

* **Traffic forecasting leaderboards.** Prefer dedicated traffic-forecasting
  libraries and their maintained SOTA models when the goal is protocol-matched
  leaderboard numbers on METR-LA / PEMS-style benchmarks. KoopmanGraph’s
  traffic demos and in-repo GNN baselines are teaching comparisons around an
  inspectable linear latent operator, not a forecasting competition entry.
* **Topological signal processing on simplicial or cellular complexes.**
  Prefer libraries built for Hodge / simplicial convolutions and related
  topological deep-learning stacks when the domain model is a complex rather
  than a graph or hypergraph incidence structure.
* **Biomolecular / molecular-dynamics Koopman analysis.** Prefer
  Markov-state-model and VAMPnet-style toolchains aimed at molecular
  trajectory data and transfer-operator objectives. KoopmanGraph does not
  target that domain.
* **Certified infinite-dimensional spectral computation.** Prefer ResDMD and
  related rigorous data-driven spectral methods when you need residual-based
  certificates on a fixed dictionary, not an a posteriori check in a learned
  latent space.
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
