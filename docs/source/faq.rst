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

Import paths after 0.5
----------------------

Version **0.5** uses a thin root façade. Core workflow symbols remain
``from koopman_graph import …`` (model, encoders/decoders including delay,
operators including graph, snapshot containers, primary spectrum helpers,
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

``ImportError: cannot import name '…' from 'koopman_graph'`` for one of these
names usually means the import should use the capability module. See the Keep-in
/ Demote inventories in :doc:`architecture`.

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
