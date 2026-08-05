Installation
==============

KoopmanGraph requires Python 3.10 or newer, PyTorch, and PyTorch Geometric (PyG).
Install PyTorch and PyG **before** installing KoopmanGraph so your installer can
resolve compatible wheels for your platform.

Commands below show **pip** and **uv** where both apply. Pip remains the
canonical path; uv is fully supported for the same workflows.

Prerequisites
-------------

Python
~~~~~~

Use Python 3.10, 3.11, or 3.12. Check your version:

.. code-block:: bash

   python --version

With uv you can also pin an interpreter for the project:

.. code-block:: bash

   uv python pin 3.12

PyTorch
~~~~~~~

Install the PyTorch build that matches your system (CPU or CUDA). Follow the
selector at `PyTorch Get Started <https://pytorch.org/get-started/locally/>`_.
Example for CPU-only on Linux or macOS:

.. code-block:: bash

   pip install torch
   # or: uv pip install torch

uv can pick a matching accelerator index automatically (``uv pip`` only):

.. code-block:: bash

   uv pip install torch --torch-backend=auto

See `Using uv with PyTorch <https://docs.astral.sh/uv/guides/integration/pytorch/>`_
for CUDA / ROCm / XPU indexes and ``UV_TORCH_BACKEND``.

PyTorch Geometric
~~~~~~~~~~~~~~~~~

PyG depends on the PyTorch version already installed. Use the official
`PyG installation guide <https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html>`_
to pick the correct ``torch-geometric`` (and optional extension) wheels.

Minimal install after PyTorch is in place:

.. code-block:: bash

   pip install torch-geometric
   # or: uv pip install torch-geometric

Install KoopmanGraph
--------------------

From source (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~

Clone the repository and install in editable mode:

.. code-block:: bash

   git clone https://github.com/tjkessler/KoopmanGraph.git
   cd KoopmanGraph
   pip install -e .

For development (tests, linting, pre-commit):

.. code-block:: bash

   pip install -e ".[dev]"

To build documentation locally:

.. code-block:: bash

   pip install -e ".[docs]"
   cd docs && make html

Optional extras for symmetry-adapted operators (node-orbit partitions via
``networkx``; exact automorphisms need a separate ``pynauty`` install):

.. code-block:: bash

   pip install -e ".[symmetry]"

Optional extras for Koopman model-predictive control (OSQP QP solver):

.. code-block:: bash

   pip install -e ".[mpc]"

Optional extras for distributed *trainer orchestration* (Lightning Fabric,
Ray Train / Ray ensemble, and Dask), MD / MSM teaching paths, and related
baselines. Native PyTorch DistributedDataParallel (DDP) / ``torchrun``
paths use core PyTorch only — no extra required:

.. code-block:: bash

   pip install -e ".[lightning]"           # Fabric + KoopmanLightningModule
   pip install -e ".[ray]"                 # Ray Train + ensemble Ray (ray[train])
   pip install -e ".[dask]"                # dask_prep materialize (offline)
   pip install -e ".[msm]"                 # deeptime (GraphVAMP / interop)
   pip install -e ".[md]"                  # mdtraj molecular I/O stubs
   pip install -e ".[equivariance]"        # e3nn Tier-B steerable encoder
   pip install -e ".[baselines-ode]"       # torchdiffeq for STGODEBaseline
   pip install -e ".[baselines-graphcast]" # reserved no-op (pure PyTorch)
   pip install -e ".[distributed]"         # meta: lightning + ray + dask

These trainer extras are **not** related to operator
``sparsity="distributed"`` (matrix-free inverse / spectrum on discrete
graph and multiplex hetero; see :doc:`faq` and :doc:`limitations`). The
``[lightning]`` extra covers Fabric and the optional
:class:`~koopman_graph.distributed.KoopmanLightningModule` Trainer sugar.
The ``[ray]`` extra pins ``ray[train]`` for both
:func:`~koopman_graph.distributed.run_ray_train_fit_loop` (model DDP under
Ray Train) and
:func:`~koopman_graph.distributed.fit_ensemble_with_ray` (parallel
independent ensemble member fits; sequential remains default). Prefer
native DDP / Fabric unless you already standardize on Ray Train
(see :doc:`faq`). The same extra is used by the examples-only Tune script
``examples/scripts/ray_tune_koopman_example.py`` (search space stays in the
script; no library Tune / AutoML API). The ``[dask]`` extra activates
:mod:`koopman_graph.distributed.dask_prep` materialize helpers; it is
**not** a Dask training loop (see :doc:`faq`). The ``[msm]`` extra pins
deeptime for VAMP-2 oracle tests, GraphVAMP helpers, and
:mod:`koopman_graph.interop`; the topology-blind precursor score/loss
helpers themselves need no extra. The ``[md]`` extra pins mdtraj for
optional I/O under :mod:`koopman_graph.datasets.molecular`; the synthetic
contact-graph oracle needs no extra. The ``[equivariance]`` extra pins
``e3nn`` for :class:`~koopman_graph.nn.E3EquivariantEncoder` (steerable
encode to invariant latents; latent :math:`K` remains non-equivariant).
``[baselines-ode]`` is required to construct
:class:`~koopman_graph.baselines.gnn.STGODEBaseline`.
``[baselines-graphcast]`` is a reserved no-op — the GraphCast teaching
path is pure PyTorch. Exact-automorphism isotypic mode
(``koopman_symmetry="isotypic"``) needs optional ``pynauty`` separately
from ``[symmetry]`` (``networkx`` approximate orbits).

uv (project sync)
~~~~~~~~~~~~~~~~~

From a clone, ``uv sync`` creates ``.venv`` and installs the project. The
repository’s ``pyproject.toml`` pins ``torch`` to the official **CPU** wheel
index by default (same choice as CI). Install PyTorch / PyG first only when you
need a non-default accelerator; otherwise sync is enough for CPU development:

.. code-block:: bash

   git clone https://github.com/tjkessler/KoopmanGraph.git
   cd KoopmanGraph
   uv sync --extra dev
   uv run pytest

Documentation and optional capability extras (``uv.lock`` includes ``mpc`` /
``symmetry`` for frozen sync; add ``lightning`` / ``distributed`` when you
need Fabric locally):

.. code-block:: bash

   uv sync --extra docs
   uv sync --extra mpc --extra symmetry
   uv sync --extra lightning
   cd docs && make html

pip-compatible uv installs (after creating a venv) mirror the pip commands:

.. code-block:: bash

   uv venv
   uv pip install -e ".[dev]"

For GPU builds with uv, prefer ``uv pip install torch --torch-backend=auto``
(or a specific backend such as ``cu126``) before installing KoopmanGraph, and
see the Astral PyTorch guide linked above. Override ``[tool.uv.sources]`` if you
want ``uv sync`` to resolve a non-CPU index.

PyPI
~~~~

After PyTorch and PyG are installed, install KoopmanGraph from PyPI:

.. code-block:: bash

   pip install koopman-graph
   # or: uv pip install koopman-graph

Pin a specific **published** release when reproducing results (replace the
version with the tag you intend; see PyPI and ``CHANGELOG.md``):

.. code-block:: bash

   pip install koopman-graph==0.12.0
   # or: uv pip install koopman-graph==0.12.0

Releases are published automatically when a maintainer creates a GitHub Release
(see ``CONTRIBUTING.md`` in the repository). For the latest in-tree development
checkout, use the editable install from source above.

Verify
------

Confirm the package imports:

.. code-block:: bash

   python -c "import koopman_graph; print(koopman_graph.__version__)"
   # or: uv run python -c "import koopman_graph; print(koopman_graph.__version__)"

For a full development check after ``pip install -e ".[dev]"`` or
``uv sync --extra dev``:

.. code-block:: bash

   pytest tests/ -v
   # or: uv run pytest tests/ -v

Next steps
----------

See :doc:`quickstart` for a minimal train-and-predict workflow.
