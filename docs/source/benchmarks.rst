Identity-bound benchmarks
=========================

A **benchmark** in this library is a frozen protocol record plus an
identity-bound summary, not a trained forecast table. The runner hashes
dataset bytes and declared methods; it does **not** fit
:class:`~koopman_graph.model.GraphKoopmanModel` or invent MAE / RMSE.

What ships
----------

:class:`~koopman_graph.benchmark.ExperimentManifest` (schema
``benchmark_manifest_v1``) records dataset SHA-256, seeds, horizons,
declared metric *names*, and method rows. Teaching GNN methods require
non-empty ``deviations``. JSON always loads; YAML needs
``pip install "koopman-graph[cli]"`` (PyYAML).

``koopman-graph benchmark run`` writes ``summary.json`` (schema
``benchmark_summary_v1``) with ``executed=False`` and a canonical
SHA-256 of the identity fields. ``verify`` recomputes that digest and
fails on a tampered hash or a dataset-byte mismatch. Types live under
:mod:`koopman_graph.benchmark` (off root ``__all__``). No other
package imports ``benchmark`` at module load; the CLI lazy-imports
handlers.

Tracked smoke fixtures live under ``benchmarks/v0.15/`` (telemetry-like,
multiphysics, topology transfer). Payloads are tiny hashed UTF-8
stand-ins, not METR-LA HDF5. Default CI runs ``benchmark verify`` on
those stubs.

How to use it
-------------

.. code-block:: bash

   koopman-graph benchmark verify \
     --manifest benchmarks/v0.15/smoke_telemetry.yaml \
     --against benchmarks/v0.15/summaries/smoke_telemetry.json

A Python walkthrough is ``examples/47_benchmark_manifest.ipynb``. Full
telemetry remains a documented extra, not default CI.

Ceilings
--------

* ``run`` / ``verify`` do not train, download METR-LA, or host a
  LibCity / BasicTS leaderboard (``LibCity2021``, ``BasicTS2024``).
* Identity-bound summaries are **not** forecast scores. Declared metric
  names are labels on the protocol, not computed errors.
* The ``[benchmark]`` extra is empty. YAML is the only optional
  dependency (``[cli]``).

See :doc:`cli` for flags, :doc:`limitations` for the honesty boundary,
and :doc:`tutorials` for the gallery row.
