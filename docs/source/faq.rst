FAQ and troubleshooting
=======================

Common install and runtime friction for KoopmanGraph. For a full install walkthrough
see :doc:`installation`. For the public vs power-user API contract see
:doc:`architecture`.

Installation order (PyTorch / PyG / wheels)
-------------------------------------------

Install **PyTorch**, then **PyTorch Geometric (PyG)**, then **KoopmanGraph**.
KoopmanGraph depends on both; installing the package first often pulls an
incompatible or source-built stack.

1. Pick a PyTorch build (CPU or CUDA) from the
   `PyTorch Get Started <https://pytorch.org/get-started/locally/>`_ selector.
2. Install matching PyG wheels from the
   `PyG installation guide <https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html>`_.
3. Install KoopmanGraph (``pip install koopman-graph``,
   ``uv pip install koopman-graph``, or an editable clone / ``uv sync``).

If the installer tries to compile extensions or cannot find wheels, re-check
that the installed ``torch`` version and CUDA tag match the PyG wheel index you
used. With uv, ``uv pip install torch --torch-backend=auto`` (or a specific
backend) helps pick a matching PyTorch index; see :doc:`installation`.

Which platforms does CI cover?
------------------------------

CI runs the full Ubuntu test matrix on Python 3.10–3.12 and a macOS core smoke
job on Python 3.12. Windows is best-effort community support (not in CI). See
:doc:`installation` (Supported platforms / CI).

CUDA vs CPU mismatches
----------------------

Symptoms include CUDA-related import errors, ``RuntimeError`` about devices, or
kernels failing only on GPU.

* Confirm ``torch.cuda.is_available()`` matches the build you intended.
* Reinstall PyTorch and PyG for the **same** CUDA (or CPU) choice; mixing a
  CPU ``torch`` wheel with CUDA PyG extensions (or the reverse) is a common
  failure mode.
* When reporting install failures, include ``python --version``,
  ``torch.__version__``, and whether CUDA is expected.

Editable installs and extras
----------------------------

From a clone of the repository:

.. code-block:: bash

   pip install -e .              # runtime package only
   pip install -e ".[dev]"       # tests, Ruff, pre-commit
   pip install -e ".[docs]"      # Sphinx documentation build

   # uv equivalents:
   uv sync                       # runtime package only (CPU torch by default)
   uv sync --extra dev
   uv sync --extra docs

Use ``.[dev]`` for local testing and ``.[docs]`` before ``cd docs && make html``.
The ``[dev]`` and ``[docs]`` extras do not replace the PyTorch / PyG prerequisite
order above when you need a non-default (non-CPU) accelerator.

Optional feature extras
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install "koopman-graph[mpc]"               # OSQP for KoopmanMPC
   pip install "koopman-graph[symmetry]"          # networkx for auto node orbits
   pip install "koopman-graph[rl]"                # Gymnasium / Stable-Baselines3
   pip install "koopman-graph[lightning]"         # Fabric + KoopmanLightningModule
   pip install "koopman-graph[ray]"               # Ray Train + ensemble Ray
   pip install "koopman-graph[dask]"              # offline dask_prep helpers
   pip install "koopman-graph[msm]"               # deeptime / GraphVAMP interop
   pip install "koopman-graph[md]"                # mdtraj molecular I/O stubs
   pip install "koopman-graph[equivariance]"      # e3nn Tier-B encoder
   pip install "koopman-graph[baselines-ode]"     # torchdiffeq for STGODE
   pip install "koopman-graph[baselines-graphcast]"  # reserved; teaching GraphCast is pure PyTorch
   pip install "koopman-graph[distributed]"       # meta: lightning + ray + dask

* **MPC:** ``from koopman_graph.mpc import KoopmanMPC, TubeKoopmanMPC``.
  Construction works without OSQP; ``solve`` / ``rollout`` /
  ``evaluate`` raise with install guidance if OSQP is missing.
* **Symmetry:** ``koopman_auto_orbits=True`` uses ``networkx``
  (``method="auto"``). Without ``[symmetry]``, ``node_orbit_partition``
  warns and returns the identity partition (no tying). Exact orbits /
  ``koopman_symmetry="isotypic"`` need optional ``pynauty`` separately.
* **Distributed trainers:** native DDP / ``torchrun`` need only core
  PyTorch. Fabric and optional
  :class:`~koopman_graph.distributed.KoopmanLightningModule` Trainer sugar
  need ``[lightning]``. ``[ray]`` covers both
  :func:`~koopman_graph.distributed.run_ray_train_fit_loop` (model DDP)
  and :func:`~koopman_graph.distributed.fit_ensemble_with_ray` (ensemble
  members). Prefer Fabric / DDP for multi-GPU *model* training unless you
  already standardize on Ray Train (see below). ``[dask]`` activates
  :mod:`koopman_graph.distributed.dask_prep` materialize helpers (not a
  training loop; see “Can I use Dask?” below). See :doc:`installation`
  and :doc:`capabilities` (Distributed training).
* **MD / MSM:** ``[msm]`` pins deeptime for GraphVAMP /
  :mod:`koopman_graph.interop`. ``[md]`` pins mdtraj for optional
  trajectory I/O under :mod:`koopman_graph.datasets.molecular`; the
  synthetic contact-graph oracle needs no extra.
* See :doc:`installation` for the full extras table.

Import paths after 0.6
----------------------

Version **0.6** keeps a thin root façade. Core workflow symbols remain
``from koopman_graph import …`` (model; encoders/decoders including delay and
hypergraph; operators including graph / hypergraph / global-local /
continuous-graph; snapshot containers; primary spectrum helpers;
``__version__``).

Specialized symbols are **capability-module imports only** (hard cut; no root
aliases), for example:

.. code-block:: python

   from koopman_graph.baselines import DMDBaseline, EDMDBaseline
   from koopman_graph.losses import ForwardConsistencyLoss
   from koopman_graph.training import FitHistory, LossWeights
   from koopman_graph.adaptation import RecursiveKoopmanAdapter
   from koopman_graph.env import GraphKoopmanEnv
   from koopman_graph.data import temporal_split, WindowSampler
   from koopman_graph.metrics import evaluate_forecast, EvaluationResult
   from koopman_graph.uq import (
       BayesianKoopmanUQ,
       ConformalKoopmanUQ,
       EnsembleGraphKoopmanModel,
   )
   from koopman_graph.mpc import KoopmanMPC, TubeKoopmanMPC, TubeMPCReport
   from koopman_graph.nn import AdaptiveAdjacency
   from koopman_graph.analysis import (
       identify_sparse_dynamics,
       koopman_spectral_clustering,
       resdmd,
       spectral_residuals,
   )
   from koopman_graph.baselines import vamp2_score
   from koopman_graph.statistics import spectral_distance
   # Root façade also exports SimplicialEncoder / InvariantGeometryEncoder

``ImportError: cannot import name '…' from 'koopman_graph'`` for one of these
names usually means the import should use the capability module. See the Keep-in
/ Demote inventories in :doc:`architecture`.

Choosing an ``adjacency`` mode (graph operators)
------------------------------------------------

Pairwise networked operators (``koopman="graph"`` and
``koopman="continuous_graph"``) take ``adjacency`` (factory:
``koopman_adjacency``):

* ``"symmetric"`` (default) — undirected
  :math:`D^{-1/2} A D^{-1/2}`. Use when the graph is undirected or when you
  intentionally symmetrize directed edges.
* ``"random_walk"`` — row-normalized :math:`D_{\mathrm{out}}^{-1} A`. Use for
  one-way / directed coupling that the symmetric mode cannot represent.
* ``"dual_random_walk"`` — forward walk plus
  :math:`D_{\mathrm{in}}^{-1} A^{\top}` (extra ``K_bwd`` / ``L_bwd`` factors).
  Use when both directions matter but should not be forced into a single
  symmetric matrix (DCRNN-style bidirectional diffusion).

Hypergraph operators do **not** expose ``adjacency`` (Zhou symmetric
incidence). Self-adaptive topology and orbit-tied ``K_{\mathrm{self}}`` are
separate options and are not substitutes for directed normalization. See
:doc:`architecture` (adjacency contract) and :doc:`limitations`.

Can I train on several graphs or topologies?
--------------------------------------------

Yes. Pass a :class:`~koopman_graph.data.MultiTrajectory` of homogeneous
(or all-hetero) sequences to :meth:`~koopman_graph.model.GraphKoopmanModel.fit`.
That path already averages per-trajectory losses; it is not a
single-topology restriction.

``fit(..., batch_graphs=True)`` is an opt-in **vectorization** of that
loop: independent graphs are collated into one PyG ``Batch`` so shared
:math:`K` applies to the disconnected union with per-graph shifts.
Reconstruction (and forward consistency when weighted) match the mean of
the per-sequence losses on modest batches. The default Python loop is
unchanged. Hetero, hypergraph, windowed, and DDP graph-batching are out
of scope. This flag does not enable multi-topology training for the
first time, and this page does not quote a throughput number.
See :doc:`architecture` (multi-graph ``Batch``).

Why is training slow on large :math:`N`?
----------------------------------------

Several training paths still assemble or multiply dense structures whose
size grows with :math:`N` (or :math:`N\cdot d`), even when latents and
supports are cached within an evaluation:

* Exact **inverse** and continuous dense
  :math:`\Phi=\exp(\Delta t\, L_{\mathrm{eff}})` on
  :math:`(N\cdot d)\times(N\cdot d)` matrices
* Exact **spectrum** when the Kronecker path does not apply (dense
  :math:`(N\cdot d)` eigendecomposition, or discrete distributed Arnoldi
  surrogate); eligible graph / continuous-graph spectrum uses Kronecker-sum
  reduction instead — see below and :doc:`limitations` (Scale)
* DiffConv diffusion supports and hypergraph Zhou :math:`\hat{H}` as dense
  :math:`N\times N` tensors
* Eigenvalue hinge on dense / ODO networked operators
  (:math:`O((N\cdot d)^3)` eigendecomposition of the assembled map)
* Self-adaptive topology materializing full :math:`N^2` COO

Shared pair latents, inverse / support / :math:`\hat{H}` / :math:`\Phi`
reuse, and PDE/worst-case prediction sharing reduce repeated work; they
do not change those representation sizes. See :doc:`limitations` (Scale),
:doc:`matrix_free`, and :doc:`capabilities` (Training performance).

Does spectrum still assemble :math:`N\cdot d`?
----------------------------------------------

Not always for discrete ``GraphKoopmanOperator`` /
``ContinuousGraphKoopmanOperator``. ``.spectrum`` auto-routes:

* ``sparsity="distributed"`` — Arnoldi leading-modulus surrogate on discrete
  graph / multiplex hetero; continuous graph has no Arnoldi spectrum path
  and uses dense :math:`L_{\mathrm{eff}}`
* Else if eligible (shared self; ``adjacency`` in
  ``{"symmetric", "random_walk"}``; ``sparsity`` in
  ``{"dense", "block_diagonal"}``) — Kronecker-sum exact spectrum
  (order :math:`O(N^3 + N d^3)` via dense :math:`N\times N`
  :math:`\widehat{A}` plus :math:`N` blocks of size :math:`d\times d`).
  Discrete hop degree :math:`P\ge 0` is eligible; the pencil is
  :math:`B(\lambda)=\sum_k\lambda^k K_k`, not a sum of independent
  factor eigenvalues. Continuous graph stays the one-tap generator.
* Else — dense :math:`(N\cdot d)` eigendecomposition
  (``dual_random_walk``, discrete orbit / isotypic self banks, hetero /
  hypergraph, helper fall-back)

Exact inverse and eigenvalue-regularization hinges are unchanged (still
dense assembled ceilings where documented). Details:
:doc:`limitations` (Scale) and :doc:`architecture` (spectrum routing).

Does a deeper GNN encoder make up for one-hop :math:`K`?
--------------------------------------------------------

No. Encoder neighborhood mixing and the Koopman factor hop degree are
different maps. :meth:`~koopman_graph.model.GraphKoopmanModel.fit`
warns when encoder hops exceed discrete graph ``filter_degree``
(GCN/GAT depth, or DiffConv ``num_layers * diffusion_steps``). Raise
``koopman_filter_degree`` or reduce encoder depth if you want them
matched. The check does not fail training and skips operators without
a hop radius. See :doc:`limitations`,
``examples/38_operator_factorization_ablation.ipynb`` (hop-matched
:math:`P` arm; not a rewrite of the historical joint-LS gap), and
``examples/49_multi_hop_factorization.ipynb`` (Kronecker versus dense
:math:`P=2` spectrum).

Does identification replace Adam ``fit``?
-----------------------------------------

No. :meth:`~koopman_graph.model.GraphKoopmanModel.fit` still defaults to
Adam (``identification=None``). Pass
:class:`~koopman_graph.identification.IdentificationConfig` to alternate
frozen-encoder closed-form :math:`K` updates (ridge, TLS, or
constrained least squares) with encoder/decoder Adam steps. That path
currently supports discrete dense per-node
:class:`~koopman_graph.operators.KoopmanOperator` only. After fit, read
``model.identification_report`` for latent one-step / short-rollout
mean squared error (MSE) and :math:`\rho(K)` — not a Haseli–Cortés,
ResDMD, or stability certificate. Types and solvers stay off the root
façade (``from koopman_graph.identification import ...``). Tutorials
continue to use Adam unless they opt in. See :doc:`identification`
and ``examples/48_identification_invariance.ipynb``.

Is ``evaluate(..., include_invariance=True)`` a Haseli–Cortés certificate?
--------------------------------------------------------------------------

No. Opt-in ``include_invariance`` (and
:meth:`~koopman_graph.model.GraphKoopmanModel.subspace_invariance_report`)
reports a dimensionless finite-sample projection leakage
:math:`\eta` on a truncated-SVD basis of encoded snapshots. Default
evaluate MAE / RMSE / MAPE are unchanged. The helper currently
supports discrete dense per-node
:class:`~koopman_graph.operators.KoopmanOperator` only. It is **not**
the Haseli–Cortés invariance-proximity certificate (principal angles /
worst-case bound; ``HaseliCortes2023``), **not**
:class:`~koopman_graph.losses.ForwardConsistencyLoss`, and **not**
:func:`~koopman_graph.analysis.spectral_residuals`. Closed-form
``fit`` still does not fill ``IdentificationReport.invariance``.

Does residual-aware selection certify a ResDMD spectral measure?
----------------------------------------------------------------

No. :func:`~koopman_graph.identification.select_resdmd_gated` compares
already-scored dictionaries: lowest train one-step mean squared error
(MSE) wins, unless ``gate_resdmd=True`` first drops candidates whose
max finite-dictionary ResDMD residual exceeds the default cutoff
:math:`10^{-2}` (same as
:func:`~koopman_graph.analysis.resdmd`).
``IdentificationConfig.gate_resdmd=True`` fills
``IdentificationReport.spectral`` on the final identification ``fit``;
it does not abort training.
:class:`~koopman_graph.training.ResDMDFitCallback` defaults to
``mode="observe"``. ``mode="gate"`` raises at ``on_fit_end`` when the
observed max residual exceeds that cutoff, without mutating parameters.
None of these is an infinite-dimensional residual certificate.

Does ``identify_sparse_graph_factors`` replace SINDy or L1 training?
--------------------------------------------------------------------

No. :func:`~koopman_graph.identification.identify_sparse_graph_factors`
fits shared :math:`K_{\mathrm{self}}` / :math:`K_{\mathrm{nbr}}` on
frozen encodings (STLSQ or a teaching proximal group-lasso, then an
unpenalized refit). It is not
:func:`~koopman_graph.analysis.identify_sparse_dynamics` (polynomial /
graph library on learned latents) and not
:class:`~koopman_graph.losses.KoopmanSparsityLoss` (soft training
penalty on operator entries). Those tools still ship. Dual
random-walk and polynomial :math:`P>1` hops are out of scope. Related
sparse-Koopman literature: Pan, Arnold-Medabalimi, and Duraisamy
(*J. Fluid Mech.*, 2021; ``Pan2021SparseSubspace``). The identifier is
not that paper's multi-task EDMD dictionary pruning.

Does ``select_latent_rank`` replace Ray Tune for ``latent_dim``?
----------------------------------------------------------------

No. :func:`~koopman_graph.identification.select_latent_rank` scores a
truncated-SVD grid of **frozen** encodings (in-tree VAMP-2, a
finite-dictionary ResDMD residual elbow, or stability-penalized
held-out one-step mean squared error). It does not train an encoder
per candidate and does not choose
:class:`~koopman_graph.model.GraphKoopmanModel` ``latent_dim``.
:mod:`koopman_graph.tuning` Ray Tune helpers remain caller-owned
example scaffolds; KoopmanGraph is not an AutoML product. deeptime
(``[msm]``; ``deeptime2021``) is an optional VAMP-2 cross-check, not a
runtime requirement. See :doc:`identification` and
``examples/53_latent_rank_selection.ipynb``.

Does ``koopman-graph benchmark run`` train a model?
---------------------------------------------------

No. ``koopman-graph benchmark run --manifest … --data … --out …``
verifies the dataset SHA-256 against a frozen
:class:`~koopman_graph.benchmark.ExperimentManifest` and writes
identity-bound ``summary.json`` (canonical digest;
``executed=False``). ``verify`` recomputes that digest and fails on a
tampered hash. Neither command fits
:class:`~koopman_graph.model.GraphKoopmanModel`, downloads METR-LA, or
hosts a LibCity / BasicTS leaderboard. Default CI verifies hashed
stand-ins under ``benchmarks/v0.15/`` (see :doc:`cli` and
:doc:`benchmarks`); it does not download full telemetry.

When should I use ``sparsity="block_diagonal"``?
------------------------------------------------

Use ``sparsity="block_diagonal"`` on graph / hypergraph / continuous-graph
operators when the dense :math:`N\cdot d` path dominates wall time and a
self-dominated (Jacobi-style) approximation is acceptable for your
advance / inverse use case. Prefer ``sparsity="dense"`` when you need the
full coupled effective map for advance / inverse. Eligible graph /
continuous-graph ``.spectrum`` may still Kronecker-route the full coupled
factors under ``block_diagonal``. Tutorial:
``examples/29_large_graph_block_diagonal.ipynb``.

How do I enable automatic mixed precision (AMP)?
------------------------------------------------

Pass ``use_amp=True`` to ``GraphKoopmanModel.fit`` (or ``run_fit_loop``).
Optional ``amp_dtype`` selects the autocast dtype (default
``torch.float16``). AMP is supported on **CUDA** only; on CPU or MPS the
fit loop warns once and continues in FP32. AMP does not change loss
definitions — only numeric precision during the forward / backward pass.
With Lightning Fabric, prefer Fabric ``precision`` **or** ``use_amp`` —
not both stacked (``fit_with_fabric`` raises if both own autocast).

How do I use multiple GPUs or processes?
----------------------------------------

Use the power-user :mod:`koopman_graph.distributed` helpers (not root
``__all__``):

* **Native DDP / ``torchrun``** (core install)::

      torchrun --standalone --nproc_per_node=2 \\
        examples/scripts/ddp_fit_torchrun.py

  Or call ``model.fit(..., strategy="ddp")`` /
  :func:`~koopman_graph.distributed.run_ddp_fit_loop` under a process
  group. Prefer
  :class:`~koopman_graph.distributed.DistributedWindowSampler` (or
  ``window_length=...``) so a single trajectory can shard across ranks;
  full-sequence mode requires at least as many trajectories as ranks.
* **Lightning Fabric** — :func:`~koopman_graph.distributed.fit_with_fabric`
  after ``pip install "koopman-graph[lightning]"``.
* **Lightning Trainer** (optional sugar) —
  :class:`~koopman_graph.distributed.KoopmanLightningModule` wraps a
  composed :class:`~koopman_graph.model.GraphKoopmanModel`. Collate
  ``DataLoader`` batches as a ``GraphSnapshotSequence`` or a list of
  sequences; export with ``export_format1_checkpoint``. Prefer Fabric /
  DDP when you need full loss schedules or
  :class:`~koopman_graph.distributed.DistributedWindowSampler`.
* **Ray Train model DDP** (optional) —
  :func:`~koopman_graph.distributed.run_ray_train_fit_loop` after
  ``pip install "koopman-graph[ray]"``. Same scientific epoch driver under
  Ray Train ``TorchTrainer``. Prefer DDP / Fabric unless you already
  standardize on Ray (see “Which trainer should I choose?” below).
  Multi-node Ray Train is outside the CI contract.
* **Ray ensemble members** (optional) —
  :func:`~koopman_graph.distributed.fit_ensemble_with_ray` after
  ``pip install "koopman-graph[ray]"``, or
  ``EnsembleGraphKoopmanModel.fit(..., parallel_backend="ray",
  member_factory=...)``. Sequential ensemble fit remains the default.
  Prefer a picklable (ideally module-level) factory. This does **not**
  change UQ coverage guarantees and does **not** shard one model across
  GPUs — that is Ray Train / DDP / Fabric.
* **Ray Tune HPO** — power-user helpers in :mod:`koopman_graph.tuning`
  (``fit_history_metrics``, ``run_ray_tune``, optional ``example_*``
  smoke scaffolds) plus ``examples/scripts/ray_tune_koopman_example.py``.
  The search configuration stays script-/caller-owned; KoopmanGraph is
  not an AutoML product. Optuna is examples-only (no library Optuna API).

Default ``fit`` / ``run_fit_loop`` remain single-process when
``strategy`` is unset. Distributed training does **not** reduce dense
:math:`N\cdot d` ceilings (see :doc:`limitations`). Multi-node behavior
is not covered by default CI.

Which trainer should I choose — DDP, Fabric, Ray Train, or Ray ensemble?
------------------------------------------------------------------------

They answer different questions:

* **Native DDP / Fabric** — recommended default for multi-GPU *model*
  training and full loss schedules / window sharding.
* **Ray Train** (``run_ray_train_fit_loop``) — optional model-DDP backend
  when your cluster already uses Ray Train. Same scientific fit loop;
  not a multi-node production path in CI.
* **Ray ensemble** (``fit_ensemble_with_ray``) — parallel *independent*
  member fits for
  :class:`~koopman_graph.uq.EnsembleGraphKoopmanModel`. Not model DDP.

Do not stack Ray Train and Fabric autocast ownership, and do not confuse
any of these with operator ``sparsity="distributed"``.

Does heterogeneous / multiplex ``koopman="hetero_graph"`` work with DDP /
Fabric / Lightning / Ray?
--------------------------------------------------------------------------

**Yes.** RelGraph / ``HeteroGraphKoopmanOperator`` models compose with the
same trainer adapters:

* ``model.fit(..., strategy="ddp")`` /
  :func:`~koopman_graph.distributed.run_ddp_fit_loop`
* :func:`~koopman_graph.distributed.fit_with_fabric`
* :class:`~koopman_graph.distributed.KoopmanLightningModule` (hetero
  batches coerce via sequence helpers)
* :func:`~koopman_graph.distributed.run_ray_train_fit_loop`
* Ray ensemble member fits when members accept hetero inputs

``find_unused_parameters`` defaults to ``True`` for hetero RelGraph stacks
(override on the DDP fit path if you know every parameter is used).
Single-process windowed ``run_fit_loop`` accepts windowed hetero sequences
(parity with world-size-1 DDP window sampling). Dense :math:`N\cdot d` /
stacked width ceilings are unchanged by multi-GPU training.

Is trainer “distributed” the same as ``sparsity="distributed"``?
----------------------------------------------------------------

**No.** They are unrelated:

* **Trainer orchestration** — optional DDP / Fabric /
  :class:`~koopman_graph.distributed.KoopmanLightningModule` paths under
  :mod:`koopman_graph.distributed` (data-parallel gradients / devices).
* **``sparsity="distributed"``** — an *operator* sparsity mode for
  matrix-free inverse and Arnoldi spectrum on discrete graph and multiplex
  hetero constructors (hypergraph / continuous peers may still assemble).
  It does **not** enable multi-GPU training.
  :class:`~koopman_graph.operators.LinearOperatorProtocol` is the same
  operator-math surface (polynomial graph + one-tap ``matrix_free``).
  Trainer DDP does **not** shrink
  :data:`~koopman_graph.operators.MAX_DENSE_LINEAR_OPERATOR_SIZE`.

Can I use Dask with KoopmanGraph?
---------------------------------

**Yes, for offline data prep** — not as a second training runtime.
``pip install "koopman-graph[dask]"`` activates
:mod:`koopman_graph.distributed.dask_prep` helpers
(``materialize_sequences``, ``materialize_window_index_list``). The library
does **not** import Dask from ``training``; trainers remain native DDP /
Fabric / Ray Train / Ray ensemble.

Typical pattern::

   import dask
   from koopman_graph.distributed import materialize_sequences

   delayed_seqs = [dask.delayed(load_sequence)(path) for path in paths]
   sequences = materialize_sequences(delayed_seqs)
   model.fit(list(sequences), epochs=10)  # or windowed / DDP fit

Do **not** replace :func:`~koopman_graph.training.run_fit_loop` /
DDP / Fabric with a Dask-worker training loop.

How do I log fits to CSV, TensorBoard, W&B, or MLflow?
-------------------------------------------------------

Use observe-only :class:`~koopman_graph.FitCallback` hooks on
single-process ``fit`` / :func:`~koopman_graph.training.run_fit_loop`:

* **CSV / TensorBoard (in-tree)** —
  :class:`~koopman_graph.tracking.CsvFitLogger` and
  :class:`~koopman_graph.tracking.TensorBoardFitLogger`::

      from koopman_graph.tracking import CsvFitLogger, TensorBoardFitLogger

      model.fit(
          sequence,
          epochs=20,
          callbacks=[
              CsvFitLogger("runs/fit.csv"),
              TensorBoardFitLogger("runs/tb"),  # needs: pip install tensorboard
          ],
      )

* **W&B / MLflow (DIY, no library pin)** — implement ``FitCallback`` and call
  ``wandb.log`` / ``mlflow.log_metrics`` from ``on_epoch_end`` using
  ``train_breakdown.to_floats()``. Sketch classes live in
  ``examples/tracking/wandb_mlflow_callback.py``. Install those SDKs
  yourself; KoopmanGraph does not declare them as dependencies.
* **Lightning Trainer** — attach Lightning loggers to
  :class:`~koopman_graph.distributed.KoopmanLightningModule` /
  ``Trainer``; do not expect ``fit(..., callbacks=)`` on the Lightning
  path. Native ``strategy="ddp"`` rejects non-None ``callbacks`` until
  that path is wired.

What is the difference between ``spectral_residuals`` and ResDMD?
-----------------------------------------------------------------

They answer different questions:

* **``spectral_residuals``** — held-out data-driven check that claimed
  eigenpairs propagate as :math:`a(t+1)\approx\lambda\, a(t)` in the
  *learned* latent / observable space. Diagnostic filter
  (``trustworthy_mask``), not a residual-DMD certificate.
* **``resdmd`` / resolvent-norm grid** — finite-dictionary ResDMD MVP and
  finite-matrix resolvent helpers in :mod:`koopman_graph.analysis`. Useful
  for residual-aware spectral diagnostics on a fixed dictionary; **not**
  infinite-dimensional certified pseudospectra / spectral measures.
  See ``examples/40_resdmd_pseudospectra.ipynb`` and :doc:`limitations`.

Does ``MpEDMDBaseline`` replace Euclidean EDMD or spectral conditioning?
------------------------------------------------------------------------

No. :class:`~koopman_graph.baselines.MpEDMDBaseline` is measure-preserving
EDMD (Colbrook, *SIAM J. Numer. Anal.*, 2023; ``Colbrook2023mpEDMD``):
a Gram-weighted orthogonal Procrustes polar factor of the dictionary
map. Unitarity is in that Gram inner product. On a regular polygonal
planar rotation (identity dictionary; empirical Gram a multiple of the
identity), mpEDMD matches :class:`~koopman_graph.baselines.EDMDBaseline`;
on a contraction it does **not** recover the dissipative map
(eigenvalues stay on the unit circle). It does not obsolete Euclidean
conditioning diagnostics on a general directed
:math:`K_{\mathrm{eff}}`. Use
:func:`~koopman_graph.metrics.evaluate_forecast` with the same Data-only
``predict`` call site as EDMD. Euclidean conditioning on a general
directed :math:`K` is :doc:`spectral_diagnostics`.

Does ``GEDMDBaseline`` infer a generator from irregular timestamps?
-------------------------------------------------------------------

No. :class:`~koopman_graph.baselines.GEDMDBaseline` is generator EDMD
(Klus et al., *Physica D*, 2020; ``Klus2020gEDMD``): least squares of
:math:`\dot\psi \approx \psi L^{\top}` on a polynomial dictionary.
Callers must supply :math:`dx/dt` as ``Data.dx_dt``
or ``fit(..., derivatives=)``. Discrete neural ``fit`` still rejects
non-uniform :math:`\Delta t`. Irregular timestamps on the gEDMD
sequence are unused and do not create :math:`L`. This is not
:func:`~koopman_graph.analysis.identify_sparse_dynamics` (SINDy / STLSQ
on learned latents, including ``mode="derivative"``). ``predict``
advances by :math:`\exp(L\,\Delta t)` with fitted ``time_step``;
:func:`~koopman_graph.metrics.evaluate_forecast` uses the same Data-only
call site as EDMD.

Does SpectralDiagnostics certify a finite-horizon bound?
--------------------------------------------------------

No. :class:`~koopman_graph.spectrum_types.SpectralDiagnostics` reports
:math:`\kappa(V)`, Wilkinson :math:`\kappa_i`, departure from
normality, discrete Nyquist :math:`1/(2\Delta t)` in cycles per unit
time, and per-mode aliasing flags. Discrete
:func:`~koopman_graph.spectrum_types.compute_spectrum` warns when a
mode is Nyquist-adjacent.
:meth:`~koopman_graph.spectrum_types.KoopmanSpectrum.mode_amplitudes`
warns when :math:`\kappa(V)` exceeds
:data:`~koopman_graph.spectrum_types.CONDITION_WARN` (:math:`10^{6}`)
and still solves :math:`Va=z^{\top}`. None of these is a bound on
:math:`\|K^{k}\|`. See :doc:`spectral_diagnostics` and
``examples/51_spectral_diagnostics.ipynb``.

Does ``monitor_critical_transition`` certify a critical transition?
-------------------------------------------------------------------

No. :func:`~koopman_graph.analysis.monitor_critical_transition` is a
sliding-window spectral-gap heuristic. A positive rate means the
closest-eigenvalue gap shrank. It is not a Ghosh-grade
topology-criticality certificate (``Ghosh2025``) and not
:meth:`~koopman_graph.operators.KoopmanOperator.stability_certificate`.
See :doc:`criticality` and ``examples/54_criticality_monitor.ipynb``.

Does LinearOperatorProtocol replace Kronecker spectrum or DDP?
--------------------------------------------------------------

No. :class:`~koopman_graph.operators.LinearOperatorProtocol` is
``matvec`` / ``solve`` / Arnoldi algebra without assembling
:math:`K_{\mathrm{eff}}`. Leading eigpairs are Ritz values, not
:math:`\operatorname{eig}(B(\lambda))`. Trainer DDP does not shrink
the representation. Dense assembly is refused above
:data:`~koopman_graph.operators.MAX_DENSE_LINEAR_OPERATOR_SIZE`.
See :doc:`matrix_free`.

Does GraphDynamicsConfig close the topology loop by default?
------------------------------------------------------------

No. Default ``graph_dynamics=None`` keeps the 0.14 hold-last path.
Pass :class:`~koopman_graph.data.GraphDynamicsConfig` to attach a
topology head (default ``sparse_candidate``). Recursive prediction
is opt-in and mutually exclusive with
``learn_topology="self_adaptive"`` when the head is not ``none``.
See :doc:`graph_dynamics` and
``examples/50_graph_state_closure.ipynb`` (wiring check, not a
learned-forecast claim).

Can discrete neural ``fit`` use irregular :math:`\\Delta t`?
------------------------------------------------------------

No. Discrete ``fit`` and ``predict_at`` still require a uniform
increment equal to ``time_step``. Gaps raise
(``validate_uniform_discrete_increments``). Set
``dynamics_mode="continuous"`` so ``predict_at`` integrates a
generator over the supplied intervals. Generator EDMD is a different
escape hatch and **requires supplied derivatives**; irregular
timestamps on that sequence do not create :math:`L`
(``Klus2020gEDMD``). See :doc:`time_conditioning` and
``examples/12_irregular_sampling_continuous_time.ipynb``.

How do I encode time of day?
----------------------------

:func:`~koopman_graph.data.diurnal_control_features` returns Fourier
sine/cosine columns for existing additive / bilinear
``control_inputs``. :func:`~koopman_graph.data.diurnal_phase_index`
bins timestamps for a per-step ``phase_index`` on
``koopman="switched"``. These are recipes, not a native calendar
field or checkpoint key. Discrete uniform-:math:`\\Delta t`
validation is unchanged. Heterogeneous sequences have no calendar
helper. See :doc:`time_conditioning`.

What does example 22 report for GraphKoopman versus the GNN ports?
------------------------------------------------------------------

Saved METR-LA weekday-cache output ranks GraphKoopman first on
aggregate RMSE (z-scored speed): :math:`0.6551` versus STGCN
:math:`0.7076`, DCRNN :math:`1.0754`, and Graph WaveNet
:math:`0.9036`. GraphKoopman uses a longer ODO / rollout /
early-stopping budget than the GNN teaching refs (unequal budgets).
These are in-repo teaching baselines, not dedicated-library SOTA.
See ``examples/22_gnn_forecaster_comparison.ipynb``.

Does ``CochainKoopmanOperator`` replace ``koopman="hodge"`` or TopologicX?
--------------------------------------------------------------------------

No. :class:`~koopman_graph.operators.CochainKoopmanOperator` advances
node and edge latents on a static signed :math:`B_1`. It is not a
factory kind; ``koopman=None`` stays ``"pernode"``.
``koopman="hodge"`` is a node Laplacian neighbor term
(:class:`~koopman_graph.operators.HodgeKoopmanOperator`). Face latents
may be stored; :math:`k=2` is not evolved.
:func:`~koopman_graph.operators.boundary_nilpotency` flags
:math:`B_1 B_2\\approx 0`. This is not sheaf theory and not
TopologicX parity (``Lim2020Hodge``, ``TopoX2024``).

Is the order-2 teaching path TopologicX or TDA parity?
------------------------------------------------------

No. :func:`~koopman_graph.nn.order2_cochain_teaching` binds
:class:`~koopman_graph.operators.CochainKoopmanOperator` to a filled
triangle and scores :math:`B_1 B_2\\approx 0`. Face latents may be
stored; :math:`k=2` is not evolved. Optional tetrahedra reach
:data:`~koopman_graph.nn.MAX_CELL_COMPLEX_DEGREE` (3). Sheaf
restriction maps stay learned-optional (default diagonal). This is
not TopologicX or TDA ecosystem parity (``TopoX2024``).

Are Hodge mode components physical circulation?
-----------------------------------------------

No. :func:`~koopman_graph.analysis.hodge_decompose_modes` projects
stored eigenvector columns onto the combinatorial gradient / curl /
harmonic subspaces of a static signed :math:`B_1`
(``Lim2020Hodge``). On a consistently oriented cycle the constant
1-cochain is harmonic; that algebraic kernel is not a validated
fluid or electrical current. The helper is analysis-only, not a
factory kind, not ``koopman="hodge"``, and not TopologicX / sheaf
parity (``TopoX2024``).

Is ``dynamics_mode="stochastic"`` a continuous-time SDE?
--------------------------------------------------------

No. That factory string adds learned diagonal process noise after a
discrete linear map. :class:`~koopman_graph.operators.DriftDiffusionKoopman`
is a separate Euler–Maruyama / Yosida stepper: ``forward`` is the
conditional-expectation semigroup and ``advance`` samples a path. It
is not certified Itô theory and not SDMD
(``Xu2025StochasticSemigroup``, ``Zhou2025Yosida``).

What coverage does conformal UQ claim?
--------------------------------------

:attr:`~koopman_graph.uq.ConformalKoopmanUQ.coverage` names
:class:`~koopman_graph.uq.JointCoverageSpec`
``target="per_node_marginal"``. That is frequentist marginal coverage
under exchangeability, approximate on graph time series. Simultaneous
node–feature–horizon boxes and event coverage are named but not
implemented (``Schlembach2025Conformal``). Proper scores
(:func:`~koopman_graph.uq.gaussian_crps`,
:func:`~koopman_graph.uq.gaussian_nll`,
:func:`~koopman_graph.uq.energy_score`) evaluate forecasts; they do
not certify coverage.

Does Hankel-DMD or HAVOK replace ``DelayEmbeddingEncoder``?
-----------------------------------------------------------

No. :class:`~koopman_graph.nn.delay.DelayEmbeddingEncoder` stacks
Takens-style channels around a sized GNN encoder.
:class:`~koopman_graph.baselines.HankelDMDBaseline` (Arbabi and Mezić,
*SIAM J. Appl. Dyn. Syst.*, 2017; ``Arbabi2017HankelDMD``) and
:class:`~koopman_graph.baselines.HAVOKBaseline` (Brunton et al.,
*Nature Communications*, 2017; ``Brunton2017HAVOK``) fit operators on
delay-embedded flattened snapshots. HAVOK ``predict`` is autonomous
(:math:`u=0`). Optional ``history`` supplies older delay slots
(oldest → newest); without it, those slots are zeros, so
``predict(data, steps)`` is not a faithful delay initial condition
when ``n_delays > 1``.

Is delay embedding or HAVOK a Mori–Zwanzig memory model?
--------------------------------------------------------

No. :class:`~koopman_graph.nn.delay.DelayEmbeddingEncoder` stacks
Takens-style channels. :class:`~koopman_graph.baselines.HAVOKBaseline`
is delay-plus-forcing (``Brunton2017HAVOK``), not a projection-operator
memory kernel. :func:`~koopman_graph.analysis.markov_closure_report`
flags residual-energy autocorrelation (Ljung–Box-style;
``Ljung1978Box``). :class:`~koopman_graph.analysis.FiniteMemoryKoopman`
is a convolution MVP at the same latent width as a delay encoder, not
a factory kind and not Mori–Zwanzig identification
(``Lin2021MoriZwanzig``). Recovered memory length is an oracle test,
not a general theorem.

Does ``InvariantGeometryEncoder`` make the Koopman operator equivariant?
------------------------------------------------------------------------

**No.** Tier A
:class:`~koopman_graph.nn.InvariantGeometryEncoder` builds
rotation-/translation-invariant features from ``Data.pos`` and lifts them
with a standard GCN. Optional Tier B
:class:`~koopman_graph.nn.E3EquivariantEncoder` (``e3nn``,
``[equivariance]``) uses steerable message passing but still **defaults**
to invariant scalar latents. A separate
:class:`~koopman_graph.operators.EquivariantKoopmanOperator` is a
block MVP: scalars, ``scale * I_3`` vectors, and optional
:math:`l=2` ``scale * I_5`` tensors. It is not a factory kind
and not a molecular MD production stack (see :doc:`limitations`).

Does symplectic :math:`K` conserve decoded mass?
------------------------------------------------

**No.** ``parameterization="symplectic"`` constrains the latent
operator matrix. A nonlinear decoder can still break mass in
feature space (``Greydanus2019HNN``). Use
:class:`~koopman_graph.nn.MassConservingDecoder` or
:class:`~koopman_graph.nn.LinearConservingDecoder` when a named
decoded channel must satisfy a linear conservation law. Those
heads do not turn IEEE-118 Laplacian diffusion into AC power
flow.

Does TubeKoopmanMPC prove recursive feasibility?
------------------------------------------------

**No.** :class:`~koopman_graph.mpc.TubeKoopmanMPC` erodes nominal
output boxes by conformal quantiles or ensemble residual radii
and reports constraint-violation rate, feasibility rate, and
quadratic stage cost on a toy closed loop. Local decoder
linearization is unchanged. The helper is not a chance-constraint
solver and not a Lyapunov closed-loop certificate. Zhang et al.,
*Automatica* 137:110114 (2022), prove robustness for an r-KMPC
scheme with an offline nonlinear ancillary law; this MVP does not
implement that controller or inherit those proofs
(``Zhang2022TubeMPC``).

Does ``granger_latent_influence`` recover interventional edges?
---------------------------------------------------------------

**No.** :func:`~koopman_graph.analysis.granger_latent_influence`
is **non-interventional**: it reports linear residual-MSE
reduction on observed latents. The labeled synthetic helper
:func:`~koopman_graph.analysis.recover_synthetic_interventional_edges`
recovers a known do-edge on
:func:`~koopman_graph.analysis.teaching_three_node_scm` only. That
protocol is not observational discovery on field data.

Does ``evaluate_topology_transfer`` mean factorization transfers well?
----------------------------------------------------------------------

**No.** :func:`~koopman_graph.analysis.evaluate_topology_transfer`
**measures** zero-shot or fine-tune transfer across a node-count change.
It always reports a mandatory ``pernode`` control and may return
**negative** transfer advantage (as on the seeded path-diffusion fixture
in ``examples/37_cross_topology_transfer.ipynb``). Self-adaptive, orbit,
and isotypic configurations bind :math:`N` and are excluded. Naming is
deliberately ``evaluate_*`` / measure-style — not a success path. See
:doc:`limitations`.

Are AGCRN / MTGNN / STGODE / GraphCast leaderboard reproductions?
-----------------------------------------------------------------

**No.** The in-repo ports under :mod:`koopman_graph.baselines.gnn` are
**teaching baselines** with documented ``ForecasterProtocol`` deviation
tables for side-by-side comparisons on the same PEMS / METR slices. They
are not protocol-matched LibCity / BasicTS leaderboard entries. GraphCast
is a small-mesh weather teaching adapter, not a PEMS sensor-graph
forecaster and not ERA5-scale production training. Prefer dedicated
traffic libraries when you need competition numbers (see
:doc:`limitations`).

Hierarchical pooling: ``per_snapshot`` vs ``hold_perm``
-------------------------------------------------------

``HierarchicalGraphKoopmanModel`` defaults to
``pool_schedule="per_snapshot"``: TopK / SAG scores are recomputed each
timestep from that snapshot's features (0.7.0-compatible). Set
``pool_schedule="hold_perm"`` to pool from the first snapshot and reuse
that permutation for the rest of the sequence — fewer pool passes, but
pool assignments no longer track per-timestep feature changes.

Hypergraph dense :math:`\hat{H}` and ``clear_hyperedge_cache``
--------------------------------------------------------------

``koopman="hypergraph"`` advances through a dense Zhou :math:`\hat{H}`
(:math:`N\times N`). Static incidence reuses a pointer-keyed cache shared
by advance, eigen, and dense inverse assembly. Call
``clear_hyperedge_cache()`` (or
``HypergraphKoopmanOperator.clear_hyperedge_cache()``) after in-place
edits to ``hyperedge_index`` / ``hyperedge_weight`` that keep the same
storage pointers. New incidence tensors invalidate automatically.
Caching does **not** remove the dense :math:`O(N^2)` ceiling — see
:doc:`limitations`.

Training cost: eigenvalue loss and continuous dense :math:`\Phi`
----------------------------------------------------------------

Two training terms can dominate on large graphs:

* **Eigenvalue hinge** (``LossWeights.eigenvalue > 0``) with dense or ODO
  ``koopman="graph"`` / ``"hypergraph"`` / continuous-graph peers builds the
  effective :math:`N\cdot d` operator and calls ``torch.linalg.eigvals``.
  Cost is cubic in :math:`N\cdot d`. Prefer structural parameterizations
  (``schur``, ``lyapunov``, ``dissipative``) or keep :math:`N` modest when
  this weight is non-zero.
* **Continuous dense advance** (``dynamics_mode="continuous"`` with
  ``koopman="graph"`` / ``"continuous_graph"`` and ``sparsity="dense"``)
  forms :math:`\Phi=\exp(\Delta t\, L_{\mathrm{eff}})`. For large :math:`N`,
  prefer ``sparsity="block_diagonal"`` (self-only shortcut). The dense path
  caches :math:`\Phi` for repeated topology / :math:`\Delta t` within a
  single ``compute_training_loss`` evaluation and clears that cache at the
  next evaluation so optimizer steps never see a stale transition.

See :doc:`limitations` (Scale) and :doc:`capabilities`.

Checkpoint format and load failures
-----------------------------------

Checkpoints use ``FORMAT_VERSION``. Current saves write ``format_version: 1``.
Loaders accept only supported versions (currently ``{1}``).

* Previously published **format-2** checkpoints and sparse historical format-1
  payloads are **rejected** (no silent migration).
* Typical failure: ``ValueError`` / load error naming an unsupported
  ``format_version``. Retrain or re-save under the current schema, or use the
  package version that produced the checkpoint.
* Serialization details and Built-in operator kinds are documented in
  :doc:`architecture` (checkpoint / serialization sections).

Where to ask for help
---------------------

Reuse the project support routing (also in repository ``CONTRIBUTING.md``):

* **Usage / how-to** — `GitHub Discussions
  <https://github.com/tjkessler/KoopmanGraph/discussions>`_
* **Bugs** (crash, wrong results, install failure with a repro) —
  `bug report
  <https://github.com/tjkessler/KoopmanGraph/issues/new?template=bug_report.yml>`_
* **Features / API changes** —
  `feature request
  <https://github.com/tjkessler/KoopmanGraph/issues/new?template=feature_request.yml>`_

Responses are best-effort; there is no SLA. Security vulnerabilities should be
reported privately — see the repository ``SECURITY.md``.
