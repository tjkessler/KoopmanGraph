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

Optional feature extras (0.6)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install "koopman-graph[mpc]"        # OSQP for KoopmanMPC
   pip install "koopman-graph[symmetry]"   # networkx for auto node orbits
   pip install "koopman-graph[rl]"         # Gymnasium / Stable-Baselines3

* **MPC:** ``from koopman_graph.mpc import KoopmanMPC``. Construction works
  without OSQP; ``solve`` / ``rollout`` raise with install guidance if OSQP
  is missing.
* **Symmetry:** ``koopman_auto_orbits=True`` uses ``networkx``
  (``method="auto"``). Without ``[symmetry]``, ``node_orbit_partition``
  warns and returns the identity partition (no tying). Exact orbits need
  optional ``pynauty`` separately.
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
   from koopman_graph.uq import ConformalKoopmanUQ, EnsembleGraphKoopmanModel
   from koopman_graph.mpc import KoopmanMPC
   from koopman_graph.nn import AdaptiveAdjacency
   from koopman_graph.analysis import (
       identify_sparse_dynamics,
       koopman_spectral_clustering,
       spectral_residuals,
   )
   from koopman_graph.statistics import spectral_distance

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

Why is training slow on large :math:`N`?
----------------------------------------

Several training paths still assemble or multiply dense structures whose
size grows with :math:`N` (or :math:`N\cdot d`), even when latents and
supports are cached within an evaluation:

* Exact spectrum / inverse and continuous dense
  :math:`\Phi=\exp(\Delta t\, L_{\mathrm{eff}})` on
  :math:`(N\cdot d)\times(N\cdot d)` matrices
* DiffConv diffusion supports and hypergraph Zhou :math:`\hat{H}` as dense
  :math:`N\times N` tensors
* Eigenvalue hinge on dense / ODO networked operators
  (:math:`O((N\cdot d)^3)` eigendecomposition)
* Self-adaptive topology materializing full :math:`N^2` COO

Shared pair latents, inverse / support / :math:`\hat{H}` / :math:`\Phi`
reuse, and PDE/worst-case prediction sharing reduce repeated work; they
do not change those representation sizes. See :doc:`limitations` (Scale)
and :doc:`capabilities` (Training performance).

When should I use ``sparsity="block_diagonal"``?
------------------------------------------------

Use ``sparsity="block_diagonal"`` on graph / hypergraph / continuous-graph
operators when the dense :math:`N\cdot d` path dominates wall time and a
self-dominated (Jacobi-style) approximation is acceptable for your
advance / inverse / spectrum use case. Prefer ``sparsity="dense"`` when you
need the full coupled effective map. Tutorial:
``examples/29_large_graph_block_diagonal.ipynb``.

How do I enable automatic mixed precision (AMP)?
------------------------------------------------

Pass ``use_amp=True`` to ``GraphKoopmanModel.fit`` (or ``run_fit_loop``).
Optional ``amp_dtype`` selects the autocast dtype (default
``torch.float16``). AMP is supported on **CUDA** only; on CPU or MPS the
fit loop warns once and continues in FP32. AMP does not change loss
definitions — only numeric precision during the forward / backward pass.

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
