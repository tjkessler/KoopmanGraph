# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.12.0] - 2026-08-05

Scale-and-honesty release on the 0.11 stack: Kronecker-sum exact spectrum
for eligible networked graph operators, and docs-only topology-criticality
honesty. Homogeneous scientific defaults and the linear latent operator
contract are unchanged.

### Added

- Exact Kronecker-sum spectrum path for eligible discrete and continuous
  graph operators (`koopman_graph.operators.kronecker_spectrum`;
  `GraphKoopmanOperator.spectrum` /
  `ContinuousGraphKoopmanOperator.spectrum`). Eligibility requires a
  shared self factor, `adjacency` in `{"symmetric", "random_walk"}`, and
  `sparsity` in `{"dense", "block_diagonal"}`. Eigenvalues and ambient
  eigenvectors match the dense assembly oracle within float tolerance on
  path/ring fixtures.
- Three-path spectrum contract: distributed leading-modulus Arnoldi
  (discrete / multiplex hetero), Kronecker-sum exact reduction when
  eligible, otherwise dense assembly. Continuous `distributed` has no
  Arnoldi spectrum path and uses dense `L_eff`.
- Sphinx Scale / architecture / FAQ prose distinguishing the three
  spectrum paths and residual dense ceilings for exact inverse, Jacobi /
  Richardson, and eig-regularization hinges. Tutorial
  `examples/29_large_graph_block_diagonal.ipynb` records a dense vs
  Kronecker spectrum timing probe at the existing probe \(N\).
- Topology-criticality honesty in limitations (cross-linked from
  architecture / capabilities), with `paper.bib` entry `Ghosh2025`.
  Tutorial `examples/18_networked_koopman_dynamic_topology.ipynb` notes
  that dynamic topology remains mechanical (no spectral-degeneracy /
  early-warning diagnostic API).

### Notes

- Kronecker reduction applies to **spectrum only**. Exact inverse /
  Jacobi / Richardson and eig-regularization hinges remain dense
  \((N\cdot d)\) ceilings.
- No Kronecker-style spectrum for hetero / hypergraph ambient operators
  in this release.
- Homogeneous training defaults are unchanged relative to 0.11.0.

## [0.11.0] - 2026-08-02

Domain-and-structure release on the 0.10 stack: measured topology transfer,
directed hypergraph incidence, Ray Train model DDP, fixed-union presence-mask
churn, teaching traffic forecaster ports, in-repo sheaf / cell-complex MVPs,
GraphVAMP + molecular teaching path, and exact-automorphism isotypic
self-block ties. Homogeneous scientific defaults and the linear latent
operator contract are unchanged.

### Added

- Public measured cross-topology transfer API
  (`koopman_graph.analysis.evaluate_topology_transfer` /
  `TopologyTransferReport`) with mandatory `pernode` control; negative
  transfer advantage is a valid outcome. Tutorial
  `examples/37_cross_topology_transfer.ipynb` calls the API.
- Directed hypergraph incidence modes on
  `HypergraphKoopmanOperator`: `zhou_symmetric` (default),
  `forward_random_walk`, `dual_random_walk` (factory
  `koopman_hypergraph_incidence_mode`). Forward walk follows Ducournau &
  Bretto (`Ducournau2014DirectedHypergraphs`); encode/advance orientations
  may differ by design.
- Optional Ray Train model-DDP fit path
  (`koopman_graph.distributed.run_ray_train_fit_loop`, `[ray]` /
  `[distributed]` with `ray[train]`). Distinct from
  `fit_ensemble_with_ray` (ensemble *members*). Native DDP / Fabric remain
  the recommended multi-GPU defaults; multi-node Ray Train is outside CI.
- Fixed-union node churn: `allow_node_churn` + `presence_masks` at capacity
  \(N_{\max}\) (typed hetero peers included). Losses ignore inactive nodes;
  matvecs stay at capacity. Tutorial
  `examples/41_node_churn_presence_masks.ipynb`.
- Teaching traffic forecaster ports with `ForecasterProtocol` deviation
  tables: `AGCRNBaseline` (`Bai2020AGCRN`), `MTGNNBaseline`
  (`Wu2020MTGNN`), `STGODEBaseline` (`Fang2021STGODE`; optional
  `[baselines-ode]`), `GraphCastBaseline` (`Lam2023GraphCast`; small-mesh
  weather teaching slice). Tutorial
  `examples/42_traffic_teaching_baselines.ipynb`.
- Sheaf and cell-complex encode/decode MVPs (`encoder="sheaf"` /
  `"cell_complex"`; `Hansen2020SheafNeuralNetworks`,
  `Bodnar2022NeuralSheaf`, `Bodnar2021CWNetworks` motivation). Same linear
  Koopman head as GNN / simplicial-1. Tutorial
  `examples/43_tdl_sheaf_cell_ablation.ipynb`.
- Contact-graph GraphVAMP teaching baseline (`GraphVAMPBaseline`) with
  synthetic molecular oracle under `koopman_graph.datasets.molecular` and
  optional deeptime interop (`koopman_graph.interop`, `[msm]` / `[md]`).
  Positioning relative to GraphVAMPnets (`Ghorbani2022GraphVAMPNet`) /
  VAMPnets (`Mardt2018VAMPnets`, `Wu2020VAMP`). Tutorial
  `examples/44_graphvamp_md.ipynb`.
- Exact-automorphism isotypic self-block ties
  (`koopman_symmetry="isotypic"`; `compute_isotypic_decomposition`;
  textbook \(K_n\) / \(C_n\) oracles — `Serre1977LinearRepresentations`).
  Tutorial `examples/45_isotypic_symmetry.ipynb`. Neighbor-factor
  (\(K_{\mathrm{nbr}}\)) isotypic tying is **not** shipped.
- Sphinx: rewritten `limitations.rst` / `capabilities.rst`, architecture
  v0.11 map, FAQ (Ray Train choice, transfer honesty, teaching vs
  leaderboard), API / installation extras, molecular dataset card and
  presence-vs-observation mask semantics, tutorials 37 / 41–45.

### Notes

- Teaching baselines are **not** LibCity / BasicTS leaderboard reproductions.
  GraphCast is **not** a PEMS sensor-graph forecaster and **not** ERA5-scale
  production training.
- Presence-mask churn is a fixed-union \(N_{\max}\) MVP — not unbounded
  open-world graph growth.
- Sheaf / cell MVPs are **not** full TopologicX / cellular TDL parity. An
  optional richer external-complex bridge was **cut** for this release.
- GraphVAMP path is teaching / diagnostic — not Folding@home-scale MD, not
  a PyEMMA replacement. A public alanine-dipeptide loader was **cut**; CI
  uses the synthetic two-state oracle.
- Isotypic mode requires exact Aut (pynauty); WL methods are refused. Do
  not read orbit-label ties as isotypic reduction. Sample-efficiency wins
  are not guaranteed.
- Dense \(N\cdot d\) (or \(N_{\max}\cdot d\)) ceilings are unchanged by
  trainers or presence masks.
- Remaining honesty boundaries (leaderboard protocols, multi-node Ray Train,
  infinite-dimensional ResDMD, equivariant latent \(K\), full DPK /
  \(K^{2}\)VAE, etc.) are listed in Sphinx `limitations` without a deferred
  roadmap of the same shipped items.
- Zenodo version DOI:
  [10.5281/zenodo.21763908](https://doi.org/10.5281/zenodo.21763908).

## [0.10.0] - 2026-08-01

Heterogeneous-operator completion, matrix-free sparsity, ResDMD MVP,
classical DMD peers, and topology-blind VAMP-2 precursor. Homogeneous
defaults and the 0.9 RelGraph hetero path remain the baseline; shared-\(d\)
bit-compat is preserved where rectangular maps are not requested.

### Added

- Windowed hetero `fit`, opt-in reverse-relation synthesis
  (`synthesize_reverse_relations` / `synthesize_reverse_edge_types`), and
  continuous hetero dynamics (`dynamics_mode="continuous"`).
- Per-type latent widths \(d_\tau\) with rectangular relation maps \(K_r\);
  joint structural certificate on \(\rho(K_{\mathrm{eff}})\) (Gershgorin-style
  bound via `stability_certificate`, distinct from factor-wise proxies).
- Hetero parity for conformal UQ, `GraphKoopmanEnv`, hierarchical models
  (multiplex union-bank and typed per-type pooling; hetero ``fit`` uses
  ``hold_perm``), and evaluate paths; hetero orbit ties
  (`koopman_auto_orbits`).
- Operator flag `sparsity="distributed"`: matrix-free matvec path (Lanczos /
  Arnoldi spectrum with `num_modes`; multi-step Neumann / fixed-point
  inverse). Distinct from multi-GPU *trainer* extras.
- Finite-dictionary ResDMD MVP (`koopman_graph.analysis.resdmd`) and
  resolvent-norm grid (`pseudospectra`); tutorial
  `examples/40_resdmd_pseudospectra.ipynb`. Remains separate from
  diagnostic `spectral_residuals`.
- Bayesian Laplace UQ over operator factors (`koopman_graph.uq.bayesian`)
  and opt-in stochastic latent dynamics; Ulam-style transfer-operator
  baseline.
- Dask materialization helpers (`koopman_graph.distributed.dask_prep`,
  `[dask]` extra) — prep only; no Dask training loop.
- Classical DMD family peers: fbDMD, tlsDMD, optDMD, streaming DMD,
  depth-2 mrDMD; EDMD Nyström / random-feature approximations.
- Simplicial-1 / Hodge encode path (`encoder="simplicial"`); Tier A
  `InvariantGeometryEncoder` (rotation-invariant features; not equivariant
  \(K\)).
- Optional Tier B `E3EquivariantEncoder` via `e3nn` (`[equivariance]`
  extra): steerable encode to invariant scalar latents for linear \(K\)
  (still **not** an equivariant operator). Citations: `Thomas2018TFN`,
  `Geiger2022e3nn`.
- Topology-blind VAMP-2 precursor (`vamp2_score` / `LossWeights.vamp2`,
  optional `[msm]` deeptime oracles) — not GraphVAMPnets / MD toolchain.
- Sphinx: limitations **0.11 roadmap**, capabilities / architecture / FAQ /
  API / tutorials updates for the 0.10 surface.

### Notes

- Dense \(N\cdot d\) (or \(\sum_\tau N_\tau d_\tau\)) assembly ceilings still
  apply when spectrum / inverse require a dense effective matrix; the
  matrix-free path avoids assembly only under `sparsity="distributed"`.
- ResDMD MVP is finite-dictionary only (Colbrook & Townsend / related
  residual DMD literature; keys `ColbrookTownsend2023ResDMD`,
  `Colbrook2023ResidualDMD`). Do not treat `spectral_residuals` as ResDMD.
- VAMP-2 precursor cites Wu & Noé (`Wu2020VAMP`) and Mardt et al.
  (`Mardt2018VAMPnets`); it is not a GraphVAMPnets or MSM production stack.
- **0.11 targets (high level):** variable node cardinality / churn, directed
  hypergraph incidence modes, public cross-topology transfer API, sheaf /
  cell TDL, GraphVAMPnets + MD/MSM toolchains, Ray Train as model DDP,
  isotypic symmetry / equivariant \(K\).
- Zenodo version DOI:
  [10.5281/zenodo.21752937](https://doi.org/10.5281/zenodo.21752937).

## [0.9.0] - 2026-07-30

Opt-in heterogeneous / multiplex graph Koopman operators and RelGraph
encode/decode, composed with the 0.8 trainer adapters. Homogeneous defaults
unchanged.

### Added

- `HeteroGraphSnapshotSequence` and typed stacked latent layout helpers
  (`koopman_graph.data.hetero_layout`) for PyG `HeteroData` multiplex /
  multi-node-type sequences at a shared latent width \(d\).
- R-GCN-lite `RelGraphEncoder` / `RelGraphDecoder` and
  `HeteroGraphKoopmanOperator` via factory `koopman="hetero_graph"`
  (\(K_{\mathrm{eff}} = I\otimes K_{\mathrm{self}} + \sum_r \widehat{A}_r
  \otimes K_r\); optional `relation_tying="basis"`).
- Optional `HGTEncoder` / `HGTDecoder` under `koopman_graph.nn` (PyG
  `HGTConv`; not factory defaults).
- Hetero composition with native DDP (`strategy="ddp"`), Lightning Fabric,
  optional Lightning `Trainer`, and Ray ensemble helpers; exposed
  `find_unused_parameters` (default `True` for RelGraph hetero stacks).
- Interpretive `attribute_mode_energy` / `ModeEnergyAttribution` in
  `koopman_graph.analysis`.
- Typed IEEE-118 topology helpers and tutorial
  `examples/39_heterogeneous_relational_koopman.ipynb`.
- Sphinx: limitations / FAQ / capabilities / architecture / API updates for
  hetero and trainer composition.

### Notes

- Dense \(N\cdot d\) spectrum / inverse ceilings are unchanged by multi-GPU
  *trainer* orchestration. Do not confuse trainers with the unimplemented
  operator flag `sparsity="distributed"`.
- Factor-wise Schur / Lyapunov / dissipative modes do **not** certify
  \(\rho(K_{\mathrm{eff}})\). Relational factorization must earn its keep in
  ablations (negative results allowed).
- Deferred: per-type \(d_\tau\), auto reverse relations, continuous hetero
  operators, env / conformal / hierarchical hetero parity, joint structural
  certificates.
- Zenodo version DOI:
  [10.5281/zenodo.21711185](https://doi.org/10.5281/zenodo.21711185).

## [0.8.2] - 2026-07-28

Second patch re-release for Zenodo GitHub archival after tag `0.8.1` failed
with the same missing-zip ingest error (`KoopmanGraph-0.8.1.zip`, draft
`21656413`). No scientific API changes relative to 0.8.0.

### Notes

- Zenodo version DOI (manual upload after GitHub→Zenodo ingest failures):
  [10.5281/zenodo.21682959](https://doi.org/10.5281/zenodo.21682959).

## [0.8.1] - 2026-07-28

Patch re-release so Zenodo can archive the 0.8.0 software surface after the
GitHub→Zenodo webhook for tag `0.8.0` failed during Zenodo downtime (draft
ingest missing `KoopmanGraph-0.8.0.zip`). No scientific API changes relative
to 0.8.0.

### Notes

- Zenodo ingest for this tag later failed (missing `KoopmanGraph-0.8.1.zip`);
  see 0.8.2 for the follow-up archive retry.

## [0.8.0] - 2026-07-28

Optional multi-process / multi-GPU *trainer orchestration* around the same
scientific fit loop. No claimed wall-time percentages. Operator flag
`sparsity="distributed"` remains unimplemented and unrelated.

### Added

- Power-user package `koopman_graph.distributed`: rank / world-size helpers,
  `seed_everything`, `DistributedWindowSampler`,
  `shard_sequences_for_rank` (off root `__all__`).
- Native PyTorch DDP via `run_ddp_fit_loop` and
  `GraphKoopmanModel.fit(..., strategy="ddp")` (core install; typically
  `torchrun`). Example: `examples/scripts/ddp_fit_torchrun.py`.
- Lightning Fabric adapter `fit_with_fabric` (`[lightning]` extra).
- Optional Lightning `Trainer` sugar: `KoopmanLightningModule` with
  `export_format1_checkpoint` (`[lightning]`; prefer Fabric / DDP for
  full loss schedules and rank-aware sampling).
- Ray parallel ensemble member fits: `fit_ensemble_with_ray` and
  `EnsembleGraphKoopmanModel.fit(..., parallel_backend="ray",
  member_factory=...)` (`[ray]`; sequential ensemble fit remains the
  default). Does not change UQ coverage guarantees.
- Examples-only Ray Tune script
  `examples/scripts/ray_tune_koopman_example.py` (search space stays in
  the script; no library Tune / AutoML API).
- Optional extras `lightning`, `ray`, reserved `dask`, and meta
  `distributed` (Lightning + Ray + Dask pins).
- Sphinx: installation extras, capabilities (Distributed training),
  limitations / FAQ (multi-GPU, sparsity disambiguation, Dask pattern),
  architecture (`distributed/` layout and dependency rule), tutorials
  scripts pointer.

### Changed

- `GraphKoopmanModel.fit` accepts optional `strategy="ddp"` that
  delegates to `run_ddp_fit_loop` without importing Lightning into
  `model/`. Default `fit` / `run_fit_loop` remain single-process.
- Format-1 checkpoints from DDP / Trainer paths snapshot the unwrapped
  module (no `module.` prefix).

### Notes

- **Dask:** docs-only in 0.8.0; `[dask]` extra reserved; no library
  `dask_prep` API and no Dask training loop (offline materialization in
  user code only).
- **Ray Train** as a multi-GPU *model* DDP backend is out of scope; use
  native DDP or Fabric.
- Data-parallel training does **not** reduce dense \(N \cdot d\)
  operator ceilings.
- Multi-process smoke tests are opt-in
  (`KOOPMAN_GRAPH_DISTRIBUTED_TESTS=1`); default CI stays single-process.

## [0.7.1] - 2026-07-27

Performance and training-path reuse. Same public scientific defaults; no
claimed wall-time percentages.

### Added

- Opt-in CUDA automatic mixed precision on ``GraphKoopmanModel.fit`` /
  ``run_fit_loop`` (``use_amp``, ``amp_dtype``; FP32 fallback on CPU/MPS).
- Hierarchical ``pool_schedule="hold_perm"`` to amortize TopK / SAG
  pooling across a sequence (default remains ``"per_snapshot"``).
- ``clear_hyperedge_cache`` for the ephemeral Zhou ``Ĥ`` cache; DiffConv
  ``clear_support_cache`` for diffusion supports.

### Changed

- ``compute_training_loss`` shares sequence latents across pair terms
  (``SequenceLatentCache``); reconstruction / PDE / worst-case terms share
  one-step predictions when composed together.
- Networked dense inverse, continuous ``Φ`` / ``L_eff``, structural ``K`` /
  ``L``, and low-rank bilinear assemblies reuse ephemeral keyed buffers
  within an evaluation (not checkpointed).
- Self-adaptive topology materializes at most once per top-level
  ``forward`` / ``encode_rollout_origin``; multi-start rollout encodes each
  distinct origin once.
- Fit / epoch loops use ``zero_grad(set_to_none=True)``.
- Sphinx docs: Scale / Training performance / architecture contracts / FAQ
  entries for large-``N`` cost, ``block_diagonal``, AMP, hierarchical pool
  schedule, and hypergraph ``Ĥ`` caching.

## [0.7.0] - 2026-07-26

### Added

- Held-out data-driven spectral residuals
  (``koopman_graph.analysis.spectral_residuals``,
  ``SpectralResidualReport.trustworthy_mask``) and optional
  ``plot_spectrum(..., annotate_untrustworthy=True)`` overlay. Diagnostic
  in the learned latent / observable space — not a certified ResDMD bound.
- Pairwise graph / continuous-graph ``adjacency`` modes:
  ``"symmetric"`` (default), ``"random_walk"``, and ``"dual_random_walk"``,
  with factory ``koopman_adjacency`` and format-1 ``config.adjacency``.
  Row-walk normalization helpers live in ``koopman_graph.graph_utils``.
- Long-horizon statistics leaf ``koopman_graph.statistics`` (Welch PSD,
  ``spectral_distance``, ``invariant_measure_distance``, Rosenstein
  ``largest_lyapunov_exponent``, ``LongHorizonReport``). Power-user only;
  off root ``__all__``.
- Conformal ``score="node_wise"`` with optional ``neighbor_smoothing``;
  calibration payload kind ``ConformalKoopmanUQ.calibration.v2``. Legacy
  ``score="per_node"`` (max-pool) retained.
- Classical DMD-family ``rank="auto"`` (Gavish–Donoho median threshold) with
  recorded ``selected_rank``.
- Public scope page ``docs/source/limitations.rst`` (linked from README and
  the Sphinx toctree).
- Tutorials: residual / spurious-mode section on notebook 07; long-horizon
  PSD / W1 on notebook 24; ``37_cross_topology_transfer.ipynb``;
  ``38_operator_factorization_ablation.ipynb``.

### Changed

- Undirected-symmetric coupling is no longer the unconditional graph-operator
  contract; directed modes are first-class (see architecture / FAQ).
- JOSS ``paper.md`` wording: compositional / networked Koopman citations are
  inspiration for factorized self/neighbor blocks, not a demonstrated
  cross-``N`` transfer claim.
- Chaotic-benchmark guidance: report distributional statistics alongside
  short-horizon RMSE past the predictability horizon.
- Notebook ``24_nonlinear_chaotic_benchmarks.ipynb`` follows the nine-section
  scientific-notebook arc (Setup through Further reading), cites Li (2017) /
  Lorenz (1996) in Further reading, and documents CI-budget ranking caveats.

## [0.6.0] - 2026-07-25

### Added

- Hypergraph encode / decode / operator path: optional
  ``hyperedge_index`` / ``hyperedge_weight`` on
  ``GraphSnapshotSequence``, root ``HypergraphEncoder`` /
  ``HypergraphDecoder``, and ``HypergraphKoopmanOperator``
  (``koopman="hypergraph"``).
- Self-adaptive pairwise topology via ``AdaptiveAdjacency`` and
  ``learn_topology="self_adaptive"``.
- Large-graph training aids: ``sparsity="block_diagonal"`` for graph /
  hypergraph / continuous-graph operators and neighbor-sampling
  mini-batches (hypergraph shares the graph Jacobi approximate inverse
  via Zhou ``Ĥ``; forward advance unchanged vs ``dense``).
- ``GlobalLocalKoopmanOperator`` (``koopman="global_local"``) and
  ``ContinuousGraphKoopmanOperator`` (continuous networked generator;
  dense ``N·d`` cost caveat documented).
- Symmetry-adapted orbit ties for ``K_self``
  (``koopman_auto_orbits`` / ``koopman_orbit_partition``) behind the
  optional ``[symmetry]`` extra (``networkx``).
- Analysis helpers: SINDy sparse identification, Koopman spectral
  clustering, and DMD-style coupling estimation under
  ``koopman_graph.analysis``.
- Split / ACI conformal prediction via ``ConformalKoopmanUQ``.
- Receding-horizon ``KoopmanMPC`` under ``koopman_graph.mpc`` with the
  optional ``[mpc]`` (OSQP) extra and conformal constraint tightening.
- PEMS traffic and SocioPatterns contact-epidemic dataset loaders,
  download scripts, and FAIR dataset cards (``docs/source/data.rst``).
- Tutorials ``examples/27``–``36`` covering the surfaces above.
- Optional extras ``[mpc]`` and ``[symmetry]``; core
  ``import koopman_graph`` remains free of those dependencies
  (fail-at-call guided ``ImportError``).
- ``h5py`` in the ``[dev]`` extra so METR-LA / PEMS teaching-cache tests run
  in CI; ``[mpc]`` lists ``scipy`` directly (QP sparse matrices).

### Changed

- Extended format-1 checkpoint schema in place with keys for
  ``sparsity``, ``learn_topology``, ``topology_embedding_dim``,
  ``symmetry``, and global/local window fields. ``FORMAT_VERSION``
  stays ``1`` through 0.x (reject incomplete payloads; no migration).
- Expanded root ``__all__`` to twenty-five core-workflow names
  (hypergraph encoder/decoder plus the three new operator classes);
  other 0.6.0 surfaces remain capability-module imports.
- Documented [uv](https://docs.astral.sh/uv/) install paths alongside pip
  (README, CONTRIBUTING, installation guide, FAQ); added ``[tool.uv]`` CPU
  PyTorch index defaults, committed ``uv.lock``, and switched CI to
  ``astral-sh/setup-uv``.
- Speed up CI: path-filtered jobs (shell/`git` detection; no third-party
  filter action — repo Actions allowlist), separate Ruff/`uvx` lint job, drop
  the Python 3.11 matrix leg, parallelize pytest with ``-n auto``, collect
  coverage only on 3.12, use ``uv sync --frozen`` / ``uv run --no-sync``,
  pin BLAS/PyTorch thread counts under xdist, shard notebook jobs on ``main``,
  run a 10-notebook PR smoke subset (full suite on push to ``main``), and
  require the aggregator ``ci`` check instead of every leaf job.
- Moved ``discrete_spectrum_at_delta_t`` onto the neutral
  ``koopman_graph.spectrum_types`` leaf so the model façade no longer imports
  ``koopman_graph.analysis`` for spectrum assembly (still re-exported from
  ``koopman_graph.analysis``).
- Expanded ``docs/source/architecture.rst`` for the v0.6.0 capability map,
  peer inventories (``operators.orbit_ties``, ``data.sequence_types``,
  ``training.loop.bind_pending_orbit_ties``), and wrapper-local UQ /
  hierarchy persistence patterns.
- Regenerated ``uv.lock`` so ``[mpc]`` / ``[symmetry]`` resolve under
  ``uv sync --frozen``; CI and the release verify job install those extras.
- METR-LA / PEMS-BAY ``--fetch`` always verifies SHA256 (pinned digests for
  the default HuggingFace mirrors; ``--expected-sha256`` required for
  non-default ``--h5-url``).
- Release workflow runs lint, tests (coverage floor), and Sphinx ``-W`` before
  PyPI publish.
- ``SECURITY.md`` documents the trust boundary for ``torch.load`` checkpoints
  and ``*.pt`` teaching caches.

### Breaking

- **Checkpoints:** 0.5.0 (and earlier) format-1 saves that omit the new
  required keys are rejected with a clear re-save error. Retrain or
  re-save under 0.6.0; there is no silent migration while the package
  is pre-1.0. Historical format-2 payloads remain unsupported.


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

[0.10.0]: https://github.com/tjkessler/KoopmanGraph/compare/0.9.0...0.10.0
[0.9.0]: https://github.com/tjkessler/KoopmanGraph/compare/0.8.2...0.9.0
[0.8.2]: https://github.com/tjkessler/KoopmanGraph/releases/tag/0.8.2
[0.8.1]: https://github.com/tjkessler/KoopmanGraph/releases/tag/0.8.1
[0.8.0]: https://github.com/tjkessler/KoopmanGraph/releases/tag/0.8.0
[0.7.1]: https://github.com/tjkessler/KoopmanGraph/releases/tag/0.7.1
[0.7.0]: https://github.com/tjkessler/KoopmanGraph/releases/tag/0.7.0
[0.6.0]: https://github.com/tjkessler/KoopmanGraph/releases/tag/0.6.0
[0.5.0]: https://github.com/tjkessler/KoopmanGraph/releases/tag/0.5.0
[0.4.0]: https://github.com/tjkessler/KoopmanGraph/releases/tag/0.4.0
[0.3.0]: https://github.com/tjkessler/KoopmanGraph/releases/tag/0.3.0
[0.2.0]: https://github.com/tjkessler/KoopmanGraph/releases/tag/0.2.0
[0.1.0]: https://github.com/tjkessler/KoopmanGraph/releases/tag/0.1.0
