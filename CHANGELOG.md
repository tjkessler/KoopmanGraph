# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- Tutorial notebooks updated for continuous-time, stability, adaptation, physics-informed, RL, and spectrum workflows; claim↔result alignment on IEEE 118, METR-LA, epidemic, RL, and physics-informed examples

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

[0.3.0]: https://github.com/tjkessler/KoopmanGraph/releases/tag/0.3.0
[0.2.0]: https://github.com/tjkessler/KoopmanGraph/releases/tag/0.2.0
[0.1.0]: https://github.com/tjkessler/KoopmanGraph/releases/tag/0.1.0
