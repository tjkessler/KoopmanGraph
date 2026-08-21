Command-line interface
======================

The ``koopman-graph`` console script is a config-driven façade over the library
``fit`` / ``predict`` / ``save`` / ``load`` path, plus identity-bound
``benchmark run`` / ``verify``. It does not fork training mathematics.
Prefer the Python API when you need custom encoders, distributed
strategies, or experiment callbacks; use the CLI for reproducible train and
predict smokes from JSON (or YAML) configs, and for hashing a frozen
:class:`~koopman_graph.benchmark.ExperimentManifest`.

Install the package as usual (:doc:`installation`). The console script is then
available as ``koopman-graph``. YAML configs additionally need
``pip install "koopman-graph[cli]"`` (PyYAML); JSON configs do not.

Subcommands
-----------

+-------------------+----------------------------------------------------------+
| Command           | Role                                                     |
+===================+==========================================================+
| ``version``       | Print ``koopman-graph {version}`` and exit 0.            |
|                   | Equivalent to ``--version``.                             |
+-------------------+----------------------------------------------------------+
| ``train``         | Load a train config, build an allowlisted model and      |
|                   | sequence, call ``fit``, write a checkpoint               |
|                   | (default ``safetensors_v1``).                            |
+-------------------+----------------------------------------------------------+
| ``predict``       | Load a checkpoint, take the first snapshot from          |
|                   | ``--data``, roll out ``--steps``, write a ``.pt``        |
|                   | forecast payload.                                        |
+-------------------+----------------------------------------------------------+
| ``benchmark``     | Nested ``run`` / ``verify``: hash protocol identity.     |
|                   | Does not train a model or invent forecast metrics.       |
+-------------------+----------------------------------------------------------+

Global flags include ``--help`` and ``--version``.

Train
~~~~~

.. code-block:: bash

   koopman-graph train \
     --config examples/cli/synthetic_train.json \
     --out /tmp/kg-cli

* ``--config`` (required) — JSON always; ``.yaml`` / ``.yml`` require
  ``pip install "koopman-graph[cli]"`` (PyYAML).
* ``--out`` (optional) — when set, relative ``checkpoint.path`` values resolve
  under this directory.

On success, stdout reports ``wrote checkpoint: …`` and the process exits 0.

Predict
~~~~~~~

.. code-block:: bash

   koopman-graph predict \
     --checkpoint /tmp/kg-cli/model.kgckpt \
     --data examples/cli/synthetic_train.json \
     --steps 5 \
     --out /tmp/kg-cli/forecast.pt

* ``--checkpoint`` — ``safetensors_v1`` directory or ``.kgckpt`` zip, or a
  legacy ``.pt`` checkpoint (auto-detected).
* ``--data`` — trusted ``.pt`` ``GraphSnapshotSequence`` (or list of
  ``Data``), or JSON/YAML with a ``data`` section (or a bare data mapping with
  ``kind``).
* ``--steps`` — autoregressive horizon (default 5).
* ``--out`` — destination ``.pt`` file.

The forecast payload is a ``torch.save`` dict with keys ``steps``,
``forecasts`` (list of PyG ``Data``), and ``summary``.

Benchmark
~~~~~~~~~

.. code-block:: bash

   koopman-graph benchmark run \
     --manifest /tmp/kg-bench/manifest.json \
     --data /tmp/kg-bench/payload.bin \
     --out /tmp/kg-bench/artifacts

   koopman-graph benchmark verify \
     --manifest /tmp/kg-bench/manifest.json \
     --against /tmp/kg-bench/artifacts

* ``run --manifest`` (required) — JSON always; ``.yaml`` / ``.yml`` require
  ``pip install "koopman-graph[cli]"`` (PyYAML).
* ``run --data`` (required) — dataset payload whose SHA-256 must match
  ``dataset.sha256`` on the manifest.
* ``run --out`` (required) — directory that receives ``summary.json``.
* ``verify --against`` (required) — that directory, or the ``summary.json``
  file itself.

``run`` writes schema ``benchmark_summary_v1`` with ``executed=False``
and a canonical SHA-256 of the identity fields (UTF-8 JSON,
``sort_keys=True``, compact separators, ``summary_sha256`` omitted from
the digest). It does **not** fit
:class:`~koopman_graph.model.GraphKoopmanModel` or GNN teaching ports
and does not invent MAE / RMSE / MAPE numbers. ``verify`` recomputes
the digest and binds the summary to the loaded manifest; a tampered
hash or identity field fails with exit code 1.

``koopman-graph benchmark --help`` lists ``run`` and ``verify``. Bare
``benchmark`` (no nested command) prints that help and exits 0.

Tracked smoke fixtures live under ``benchmarks/v0.15/`` (three tracks:
telemetry-like, multiphysics, topology transfer). Payloads are tiny
hashed UTF-8 stand-ins, not METR-LA HDF5. YAML needs
``pip install "koopman-graph[cli]"``. Example:

.. code-block:: bash

   koopman-graph benchmark verify \
     --manifest benchmarks/v0.15/smoke_telemetry.yaml \
     --against benchmarks/v0.15/summaries/smoke_telemetry.json

Default CI runs that verify on all three stubs. It does not download
full telemetry or invent forecast metrics. See :doc:`benchmarks` and
the walkthrough ``examples/47_benchmark_manifest.ipynb``.

Exit codes
----------

* ``0`` — success.
* ``1`` — validation, I/O, or runtime error; a short ``error: …`` message is
  written to stderr (unknown config keys use dotted paths such as
  ``model.not_allowed``).

Train config schema (MVP)
-------------------------

Top-level sections: ``model``, ``data``, optional ``fit``, optional
``checkpoint``. Unknown keys are rejected.

**model.** Required: ``encoder``, ``in_channels``, ``hidden_channels``,
``latent_dim``. Encoder kinds in the CLI MVP: ``gcn``, ``gat``, ``sage``
(matched decoder peers). Optional passthrough includes ``num_layers``,
``time_step``, ``dynamics_mode``, and other allowlisted factory kwargs.

**data.** Required ``kind``:

* ``synthetic_path`` — seeded path-graph decay trajectory
  (``num_nodes``, ``num_timesteps``, ``seed``, ``feature_dim`` / ``in_channels``).
* ``cached_sequence`` — requires ``path`` to a trusted pickle ``.pt`` sequence
  (same trust boundary as other teaching caches; see the security notes in the
  repository).

**fit.** Allowlisted ``GraphKoopmanModel.fit`` kwargs (for example ``epochs``,
``lr``, ``device``).

**checkpoint.** ``path`` (file or directory); optional ``format``
(``safetensors_v1`` default, or ``legacy_pt``).

A minimal runnable example ships at ``examples/cli/synthetic_train.json``
(see ``examples/cli/README.md``).

Limitations
-----------

* Heterogeneous, sheaf, cell, and control-heavy stacks are not exposed in the
  CLI MVP; use the Python API.
* The CLI package is a power-user façade: other library modules must not import
  ``koopman_graph.cli``.
* Experiment tracking and HPO remain separate surfaces (callbacks / ``tuning``);
  the CLI does not pin cloud SDKs.
