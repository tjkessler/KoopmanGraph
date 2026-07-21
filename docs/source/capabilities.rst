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
  (GraphSAGE: Hamilton et al. 2017; DiffConv: DCRNN-style bidirectional
  diffusion, Li et al. 2018; Transformer: PyG ``TransformerConv`` / Shi et al.
  masked attention on edges — typically denser compute than GCN/GAT/DiffConv
  per edge × heads)
* ``DelayEmbeddingEncoder`` / ``n_delays`` for Hankel-style partial
  observability
* Per-snapshot ``edge_index`` (dynamic topology) and end-to-end
  ``edge_weight`` support

Dynamics
~~~~~~~~

* Discrete ``KoopmanOperator`` with soft modes (``dense``, ``odo``) or
  structural guarantees (``schur``, ``dissipative``, ``lyapunov``)
* Networked ``GraphKoopmanOperator`` (``koopman="graph"``) with
  self/neighbor coupling
* Continuous-time ``ContinuousKoopmanOperator``
  (``dynamics_mode="continuous"``), irregular timestamps, and
  ``predict_at``
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
  worst-case (:math:`L_\\infty`-style) reconstruction term (robust training
  only — not a generalization bound)
* LR schedulers, per-term loss history, ``MultiTrajectory`` fit, and
  windowed mini-batching
* Temporal train/val/test splits and per-horizon MAE, RMSE, and MAPE via
  ``koopman_graph.metrics.evaluate_forecast``

Analysis
~~~~~~~~

* ``KoopmanSpectrum`` / ``compute_spectrum`` with mode decoding helpers
* Dynamical similarity and anomaly utilities via
  ``koopman_graph.analysis``
* ``plot_spectrum`` for unit-disk / data-zoom views

Control, adaptation, and observation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``koopman_graph.adaptation.RecursiveKoopmanAdapter`` / ``adapt_step`` for
  online RLS updates
* ``KoopmanObserver`` for latent Kalman filtering / imputation under
  ``observation_masks``
* ``koopman_graph.env.GraphKoopmanEnv`` / ``to_latent_env`` for Gymnasium
  closed-loop control
* Hybrid physics observables: Laplacian, nodewise graph-gradient magnitude,
  graph curvature (``L_sym² x``), polynomial dictionaries, or custom lifting
  callables. Residual losses are soft penalties, not symplectic/Hamiltonian
  structure guarantees or a PIKE/SPIKE implementation.

Uncertainty quantification (power-user)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``koopman_graph.uq.EnsembleGraphKoopmanModel`` — deep ensembles with
  empirical ``predict_interval`` mean / quantile bounds (not on root
  ``__all__``)
* ``koopman_graph.uq.LatentGaussianKoopmanUQ`` — linear-Gaussian latent
  forecast with closed-form covariance propagation and optional Kalman
  refinement (not DPK; not a full K²VAE)
* See notebook ``21_uncertainty_quantification.ipynb`` for an ensemble vs
  latent-Gaussian comparison

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
  :math:`O(T^2)` — small/medium ``T``), ``DMDcBaseline``
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
     - IEEE 118-bus topology with simulated voltage/load dynamics
   * - ``MetrLaTrafficBenchmark``
     - Traffic
     - METR-LA sensor graph with cached speed snapshots

Related pages
-------------

* :doc:`quickstart` — runnable train/predict walkthrough
* :doc:`tutorials` — notebook gallery
* :doc:`architecture` — public vs power-user API layers
* :doc:`api` — module reference
* Repository ``README.md`` — short landing page with a Related software niche map
