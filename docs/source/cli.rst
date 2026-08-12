Command-line interface
======================

The ``koopman-graph`` console script is a config-driven façade over the library
``fit`` / ``predict`` / ``save`` / ``load`` path. It does not fork training
mathematics. Prefer the Python API when you need custom encoders, distributed
strategies, or experiment callbacks; use the CLI for reproducible train and
predict smokes from JSON (or YAML) configs.

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
