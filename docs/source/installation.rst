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

Documentation extras:

.. code-block:: bash

   uv sync --extra docs
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

Pin a specific release when reproducing results:

.. code-block:: bash

   pip install koopman-graph==0.5.0
   # or: uv pip install koopman-graph==0.5.0

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
