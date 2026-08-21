"""Protocol-locked experiment manifests and identity-bound summaries.

Capability layout
-----------------
``schema``
    Frozen :class:`~koopman_graph.benchmark.ExperimentManifest` and nested
    specs (``benchmark_manifest_v1``).
``io``
    :func:`~koopman_graph.benchmark.load_manifest` /
    :func:`~koopman_graph.benchmark.dump_manifest` (JSON always; YAML via
    PyYAML / ``[cli]``) and
    :func:`~koopman_graph.benchmark.verify_dataset_hash`.
``runner``
    :func:`~koopman_graph.benchmark.run_manifest` /
    :func:`~koopman_graph.benchmark.verify_summary` write and check
    canonical ``summary.json`` (schema ``benchmark_summary_v1``).

Honesty
-------
This package is importable and **off** root ``__all__``. ``run_manifest``
does not train models, download telemetry, or reproduce LibCity /
BasicTS leaderboards. Teaching GNN methods require non-empty
``deviations``. The optional ``[benchmark]`` extra is empty; YAML still
needs PyYAML. The CLI ``benchmark run`` / ``verify`` commands are
identity-bound (dataset SHA-256 plus a canonical summary digest).

Import rules
------------
No other :mod:`koopman_graph` package may import this package at module
load. :mod:`koopman_graph.cli` lazy-imports it inside ``benchmark``
handlers. This package may import ``baselines``, ``datasets``,
``metrics``, and ``model``; the schema/load path imports
``datasets.download`` only for SHA-256 helpers. The identity-bound
runner does not import ``model`` or ``training``.
"""

from koopman_graph.benchmark.io import dump_manifest, load_manifest, verify_dataset_hash
from koopman_graph.benchmark.runner import (
    SUMMARY_FILENAME,
    SUMMARY_SCHEMA_VERSION,
    BenchmarkSummary,
    SummaryError,
    SummaryMethodRef,
    build_summary,
    canonical_sha256,
    dump_summary,
    load_summary,
    resolve_summary_path,
    run_manifest,
    summary_from_mapping,
    summary_to_mapping,
    verify_summary,
)
from koopman_graph.benchmark.schema import (
    BENCHMARK_METRICS,
    BENCHMARK_TRACKS,
    CONTROL_TOKENS,
    METHOD_ROLES,
    MIN_MANIFEST_SEEDS,
    SCHEMA_VERSION,
    ComputeBudget,
    DatasetRef,
    EmptyMethodDeviationsError,
    ExperimentManifest,
    ManifestError,
    MethodSpec,
    OODShiftSpec,
    PreprocessingSpec,
    SplitSpec,
    UQSpec,
    manifest_from_mapping,
    manifest_to_mapping,
)

__all__ = [
    "BENCHMARK_METRICS",
    "BENCHMARK_TRACKS",
    "CONTROL_TOKENS",
    "METHOD_ROLES",
    "MIN_MANIFEST_SEEDS",
    "SCHEMA_VERSION",
    "SUMMARY_FILENAME",
    "SUMMARY_SCHEMA_VERSION",
    "BenchmarkSummary",
    "ComputeBudget",
    "DatasetRef",
    "EmptyMethodDeviationsError",
    "ExperimentManifest",
    "ManifestError",
    "MethodSpec",
    "OODShiftSpec",
    "PreprocessingSpec",
    "SplitSpec",
    "SummaryError",
    "SummaryMethodRef",
    "UQSpec",
    "build_summary",
    "canonical_sha256",
    "dump_manifest",
    "dump_summary",
    "load_manifest",
    "load_summary",
    "manifest_from_mapping",
    "manifest_to_mapping",
    "resolve_summary_path",
    "run_manifest",
    "summary_from_mapping",
    "summary_to_mapping",
    "verify_dataset_hash",
    "verify_summary",
]
