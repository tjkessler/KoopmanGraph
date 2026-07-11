Installation
==============

KoopmanGraph requires Python 3.10 or newer, PyTorch, and PyTorch Geometric (PyG).
Install PyTorch and PyG **before** installing KoopmanGraph so pip can resolve
compatible wheels for your platform.

Prerequisites
-------------

Python
~~~~~~

Use Python 3.10, 3.11, or 3.12. Check your version:

.. code-block:: bash

   python --version

PyTorch
~~~~~~~

Install the PyTorch build that matches your system (CPU or CUDA). Follow the
selector at `PyTorch Get Started <https://pytorch.org/get-started/locally/>`_.
Example for CPU-only on Linux or macOS:

.. code-block:: bash

   pip install torch

PyTorch Geometric
~~~~~~~~~~~~~~~~~

PyG depends on the PyTorch version already installed. Use the official
`PyG installation guide <https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html>`_
to pick the correct ``torch-geometric`` (and optional extension) wheels.

Minimal install after PyTorch is in place:

.. code-block:: bash

   pip install torch-geometric

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

PyPI
~~~~

After PyTorch and PyG are installed, install KoopmanGraph from PyPI:

.. code-block:: bash

   pip install koopman-graph

Pin a specific release when reproducing results:

.. code-block:: bash

   pip install koopman-graph==0.1.0

Releases are published automatically when a maintainer creates a GitHub Release
(see ``CONTRIBUTING.md`` in the repository). Until the first release is published,
use the editable install from source above.

Verify
------

Confirm the package imports:

.. code-block:: bash

   python -c "import koopman_graph; print(koopman_graph.__version__)"

For a full development check after ``pip install -e ".[dev]"``:

.. code-block:: bash

   pytest tests/ -v

Next steps
----------

See :doc:`quickstart` for a minimal train-and-predict workflow.
