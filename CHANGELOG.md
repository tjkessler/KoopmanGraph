# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Documented [uv](https://docs.astral.sh/uv/) install paths alongside pip
  (README, CONTRIBUTING, installation guide, FAQ); added ``[tool.uv]`` CPU
  PyTorch index defaults, committed ``uv.lock``, and switched CI to
  ``astral-sh/setup-uv``.
- Speed up CI: path-filtered jobs (shell/`git` detection; no third-party
  filter action — repo Actions allowlist), separate Ruff/`uvx` lint job, drop
  the Python 3.11 matrix leg, parallelize pytest with ``-n auto``, collect
  coverage only on 3.12, use ``uv sync --frozen`` / ``uv run --no-sync``, and
  require the aggregator ``ci`` check instead of every leaf job.


## [0.5.0] - 2026-07-18

### Added

- Deep-ensemble forecasting under ``koopman_graph.uq`` with empirical
  predictive intervals, coverage measurement, and format-1 member
  serialization.
- Linear-Gaussian latent uncertainty propagation with optional Kalman
  refinement via ``LatentGaussianKoopmanUQ``.
- Continuous ``koopman_parameterization="auxiliary_spectral"``: Lusch-style
  state-dependent block-diagonal generators via ``generator_at(z)`` /
  ``instantaneous_spectrum(z)``, configurable auxiliary-network depth, and
  controlled advance support.
- Physics-informed graph-gradient, graph-curvature, and polynomial observable
  presets; fit-time ``LieConsistencyLoss`` / ``PDEResidualLoss`` composition
  through ``ExtraLosses`` and ``LossWeights``.
- Drop-in ``SAGEEncoder`` / ``SAGEDecoder`` (GraphSAGE) and
  ``DiffConvEncoder`` / ``DiffConvDecoder`` (DCRNN-style bidirectional
  diffusion convolution) under ``koopman_graph.nn``, promoted to the root
  ``__all__`` façade with format-1 checkpoint types ``sage`` / ``diffconv``
  (default ``diffusion_steps=2``). See notebook
  ``examples/09_topology_ablation.ipynb`` encoder-zoo section and citations
  Hamilton2017GraphSAGE / Li2018DCRNN.
- ``GraphTransformerEncoder`` / ``GraphTransformerDecoder`` peers based on
  PyG ``TransformerConv``, including optional scalar edge conditioning and
  format-1 checkpoint reconstruction.
- RBF and kernel dictionaries for ``EDMDBaseline``, including Gaussian and
  polynomial Gram regression with documented quadratic scaling limits.
- ``KoopmanSparsityLoss`` and ``WorstCaseReconstructionLoss`` training terms;
  the latter is an empirical robust objective, not a generalization
  certificate.
- Hierarchical TopK/SAG forecasting under ``koopman_graph.hierarchical`` with
  coarse Koopman evolution, scatter unpooling, multi-resolution prediction,
  control pooling, and format-1 sidecar serialization.
- Tutorial coverage for auxiliary spectra, uncertainty
  quantification, hierarchical forecasting, and sparse latent operators.

### Changed

- **Breaking:** moved fifteen specialized root exports to capability modules
  (no root aliases). Root ``__all__`` retains exactly twenty core-workflow
  names (model, encoder/decoder family including delay, operators including
  graph, ``GraphSnapshotSequence`` / ``MultiTrajectory``, ``KoopmanSpectrum``
  / ``compute_spectrum``, ``__version__``). Import baselines, primary losses,
  ``FitHistory`` / ``LossWeights``, ``RecursiveKoopmanAdapter``,
  ``GraphKoopmanEnv``, temporal-split helpers, and ``evaluate_forecast`` /
  ``EvaluationResult`` from their capability packages.
- **Breaking (power-user):** ``TrainingLossBreakdown`` is no longer exported
  from ``koopman_graph.training``; import the internal frozen snapshot from
  ``koopman_graph.training.history`` when needed. ``FitHistory``,
  ``LossWeights``, and ``mean_training_loss_breakdown`` remain on the
  training package surface.
- Slimmed repository ``README.md`` into a JOSS-oriented landing page; detailed
  feature inventory, dataset catalog, and notebook gallery now live under
  Sphinx (`capabilities` / `tutorials`) with architecture and showcase figures
  in `docs/source/_static/`.
- `koopman_graph.graph_utils` is now a shallow capability package
  (`topology` / `propagation` peers) with the same import surface
  (`from koopman_graph.graph_utils import …`). No numerical or public-API
  signature changes.
- Symmetric normalized Laplacian now uses the pseudoinverse form
  `L_sym = P - Â` (Chung / Wikipedia): isolated nodes map to zeros under
  `graph_laplacian_features` and are unchanged under
  `normalized_step_operator`. On graphs with no isolates this matches the
  previous `I - Â` behavior. No checkpoint `FORMAT_VERSION` bump.
- Checkpoint schema reset: `FORMAT_VERSION` is now `1` for the current full
  architecture config. New saves write `format_version: 1`. Previously
  published format-2 checkpoints and sparse historical format-1 payloads are
  rejected (no silent migration). Future incompatible changes bump
  `FORMAT_VERSION` and add an explicit migration branch.
- Expanded the documented architecture contract for the v0.5.0 API tiers:
  SAGE/DiffConv/Transformer encoder-decoder pairs are root-façade peers, while
  uncertainty, hierarchy, physics-residual, and sparsity helpers remain
  capability-module or power-user APIs.

## [0.4.0] - 2026-07-17

### Added

- Networked discrete Koopman step via `GraphKoopmanOperator` (`koopman="graph"`) with self/neighbor coupling through `edge_index`
- Delay / Hankel encoder wrapper `DelayEmbeddingEncoder` and optional `GraphKoopmanModel(n_delays=...)` composition for partial observability
- Bilinear / control-affine Koopman terms (`control_mode="bilinear"`, optional `bilinear_rank`) on discrete, continuous, and graph operators
- Latent-space Kalman filtering and imputation with `KoopmanObserver` (`filter` / `smooth` / `impute`) under observation masks
- Nonlinear and chaotic graph benchmarks: epidemic network, Lorenz-96, Kuramoto–Sivashinsky, and cylinder-wake teaching cache
- Spatiotemporal GNN reference forecasters in `koopman_graph.baselines.gnn`: STGCN, DCRNN, and Graph WaveNet
- Tutorial notebooks for delay embeddings, networked operators, bilinear control, GNN baseline comparison, chaotic PDE benchmarks, and Kalman–Koopman estimation
- Bibliography and paper citations for Consistent KAEs, EDMD dictionary learning, Koopman–MPC predictors, compositional/networked Koopman, delay/HAVOK constructions, bilinear control, and Kalman–Koopman observers

### Changed

- v0.4.0 fields (`koopman_kind`, `control_mode`, `bilinear_rank`, `n_delays`) serialize as part of the checkpoint config (see Unreleased for the later format-1 schema reset)
- README, Sphinx docs, and JOSS paper draft updated for the expanded forecasting, control, and estimation surface

## [0.3.0] - 2026-07-16

### Added

- Structural stability parameterizations for the Koopman operator (`schur`, `dissipative`, `lyapunov`) with `stability_certificate()` for long-horizon rollouts
- Continuous-time dynamics via `ContinuousKoopmanOperator`, `dynamics_mode="continuous"`, irregular timestamps, and `predict_at`
- Online adaptation with `RecursiveKoopmanAdapter` and `GraphKoopmanModel.adapt_step` (RLS updates with a frozen encoder)
- Hybrid physics-informed observables (`physics_preset` / custom `physics_lifting_fn`) concatenated with GNN latents
- Dynamical similarity and anomaly helpers: `spectrum_distance`, `koopman_std`, `dynamical_similarity`, `detect_anomaly`, and `calibrate_anomaly_threshold`
- RL environment wrappers `GraphKoopmanEnv` / `to_latent_env` (optional `[rl]` extra for Gymnasium / Stable-Baselines3)
- Partial node observation masks on sequences with masked fit/evaluate losses
- Symmetric `GATDecoder` paired with `GATEncoder`
- Optional injectable Koopman operator (`koopman=`) for composition-style models
- Shared autoregressive latent rollout primitives used by the model, losses, and RL env
- Spectrum plotting helper `koopman_graph.analysis.plot_spectrum` with `limits="unit_disk"` or `limits="data"`
- Capability packages for power-user imports: `training`, `data`, `operators`, `nn`, `analysis`, and `baselines`

### Changed

- Checkpoint format version 2 with in-memory migration from v1 saves
- Continuous-time online adaptation write-back aligned with Van Loan integration
- `GraphSnapshotSequence.snapshots` is an immutable tuple (borrowed `Data` elements remain mutable)
- Classical baselines and the RL env reject dynamic topology; DMDc rejects per-node (3-D) controls
- Public root façade thinned: secondary metrics, analysis helpers, and data/adaptation/observables utilities import from their capability modules
- Removed compatibility shims for former top-level `encoder` / `decoder` / `gnn` / `operator` / `continuous` modules (use `nn` / `operators`)
- Frozen result types for fit/adaptation history and related training breakdowns
- EDMD reconstruction matrix renamed for clarity (`reconstruction_matrix`)
- Tutorial notebooks updated for continuous-time, stability, adaptation, physics-informed, RL, and spectrum workflows, with scoped findings for IEEE 118, METR-LA, epidemic, RL, and physics-informed examples

### Fixed

- Continuous structured-parameter eigenvalue regularization and related certificate/sign edge cases
- Invalid PyPI `Development Status` classifier that blocked the first 0.3.0 upload attempt

## [0.2.0] - 2026-07-12

### Added

- Spectral analysis API: `KoopmanSpectrum`, `compute_spectrum`, `decode_mode_shapes`, and `GraphKoopmanModel.spectrum()`
- Model persistence: `GraphKoopmanModel.save` / `load` with architecture config and optional best-epoch restoration in `fit`
- Temporal train/val/test splits and per-horizon MAE, RMSE, and MAPE via `evaluate_forecast` / `GraphKoopmanModel.evaluate`
- Soft operator stability via ODO parameterization, eigenvalue hinge regularization, and optimized backward-consistency inverse
- End-to-end `edge_weight` support through data containers, GCN encoder/decoder, model APIs, and the METR-LA benchmark
- Classical baselines: `DMDBaseline`, `EDMDBaseline`, and `DMDcBaseline`
- Koopman-with-control dynamics (`B` matrix), per-timestep controls on sequences, and control-aware `fit` / `predict` / `evaluate`
- Dynamic topology via optional per-snapshot `edge_index` with hold-last-known rollout
- Advanced training: LR schedulers, per-term loss history, multi-trajectory `fit`, rollout start indices, and windowed mini-batching

### Changed

- README, Sphinx docs, JOSS paper draft, and tutorial notebooks updated for the expanded public surface
- Version bumped to 0.2.0 across packaging and citation metadata

## [0.1.0] - 2026-07-11

### Added

- Initial public release of KoopmanGraph
- End-to-end `GraphKoopmanModel` encode → Koopman advance → decode pipeline with `fit` and `predict`
- Topology-aware `GNNEncoder` / `GNNDecoder` (GCN) and `GATEncoder`
- Learnable dense Koopman operator with forward and backward latent consistency losses
- `GraphSnapshotSequence` time-ordered container for PyTorch Geometric snapshots
- Built-in benchmarks: synthetic diffusion, 2D grid, IEEE 118-bus, and METR-LA traffic loaders
- Sphinx documentation, Jupyter tutorials, pytest suite with CI, and Apache-2.0 packaging for PyPI

[0.5.0]: https://github.com/tjkessler/KoopmanGraph/releases/tag/0.5.0
[0.4.0]: https://github.com/tjkessler/KoopmanGraph/releases/tag/0.4.0
[0.3.0]: https://github.com/tjkessler/KoopmanGraph/releases/tag/0.3.0
[0.2.0]: https://github.com/tjkessler/KoopmanGraph/releases/tag/0.2.0
[0.1.0]: https://github.com/tjkessler/KoopmanGraph/releases/tag/0.1.0
