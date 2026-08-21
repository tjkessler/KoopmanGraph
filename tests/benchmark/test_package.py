"""Package surface for ``koopman_graph.benchmark``."""

from __future__ import annotations

import pytest

import koopman_graph
import koopman_graph.benchmark as benchmark
from koopman_graph.benchmark import (
    ExperimentManifest,
    ManifestError,
    SummaryError,
    dump_manifest,
    load_manifest,
    run_manifest,
    verify_dataset_hash,
    verify_summary,
)


def test_benchmark_not_on_root_facade() -> None:
    """Manifest types stay off the thin root façade."""
    for name in (
        "ExperimentManifest",
        "ManifestError",
        "SummaryError",
        "load_manifest",
        "dump_manifest",
        "verify_dataset_hash",
        "run_manifest",
        "verify_summary",
    ):
        assert name not in koopman_graph.__all__
        assert not hasattr(koopman_graph, name)
        assert name in benchmark.__all__
    with pytest.raises(ImportError):
        exec("from koopman_graph import ExperimentManifest")


def test_package_reexports_match_submodules() -> None:
    """Package ``__all__`` names resolve to the submodule objects."""
    assert benchmark.ExperimentManifest is ExperimentManifest
    assert benchmark.ManifestError is ManifestError
    assert benchmark.SummaryError is SummaryError
    assert benchmark.load_manifest is load_manifest
    assert benchmark.dump_manifest is dump_manifest
    assert benchmark.verify_dataset_hash is verify_dataset_hash
    assert benchmark.run_manifest is run_manifest
    assert benchmark.verify_summary is verify_summary
    assert benchmark.SCHEMA_VERSION == "benchmark_manifest_v1"
    assert benchmark.SUMMARY_SCHEMA_VERSION == "benchmark_summary_v1"
    assert benchmark.MIN_MANIFEST_SEEDS == 3


def test_benchmark_extra_is_empty() -> None:
    """``[benchmark]`` is an empty extra; YAML still uses ``[cli]``."""
    from tests.helpers import REPO_ROOT

    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "benchmark = []" in text
    assert '"pyyaml>=6"' in text
