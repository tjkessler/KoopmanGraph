"""Identity-bound summary hashing for ``benchmark run`` / ``verify``."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from koopman_graph import __version__
from koopman_graph.benchmark import (
    SCHEMA_VERSION,
    SUMMARY_FILENAME,
    SUMMARY_SCHEMA_VERSION,
    BenchmarkSummary,
    ComputeBudget,
    DatasetRef,
    ExperimentManifest,
    ManifestError,
    MethodSpec,
    PreprocessingSpec,
    SplitSpec,
    SummaryError,
    SummaryMethodRef,
    build_summary,
    canonical_sha256,
    dump_manifest,
    load_summary,
    run_manifest,
    summary_from_mapping,
    summary_to_mapping,
    verify_summary,
)

_BYTES = b"fixture-bytes"
_DIGEST = hashlib.sha256(_BYTES).hexdigest()


def _manifest() -> ExperimentManifest:
    """Return a valid telemetry manifest.

    Returns
    -------
    ExperimentManifest
        Toy record.
    """
    return ExperimentManifest(
        manifest_id="smoke-telemetry",
        schema_version=SCHEMA_VERSION,
        track="telemetry",
        dataset=DatasetRef(
            name="toy-path",
            version="1",
            sha256=_DIGEST,
            card="docs/data/toy.md",
        ),
        split=SplitSpec(0.7, 0.1, 0.2, history_len=12),
        preprocessing=PreprocessingSpec(zscore=True),
        methods=(MethodSpec(name="graph_koopman", role="koopman"),),
        seeds=(0, 1, 2),
        horizons=(1, 3, 12),
        metrics=("mae", "rmse"),
        compute_budget=ComputeBudget(max_epochs=2),
        controls=("pernode",),
    )


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Write a manifest and matching payload.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory.

    Returns
    -------
    tuple of Path
        ``(manifest, data, out_dir)``.
    """
    manifest_path = tmp_path / "manifest.json"
    data_path = tmp_path / "payload.bin"
    out_dir = tmp_path / "artifacts"
    dump_manifest(_manifest(), manifest_path)
    data_path.write_bytes(_BYTES)
    return manifest_path, data_path, out_dir


def test_run_writes_unexecuted_hashed_summary(tmp_path: Path) -> None:
    """``run_manifest`` writes identity fields and ``executed=False``."""
    manifest_path, data_path, out_dir = _write_inputs(tmp_path)
    written = run_manifest(manifest_path, data_path, out_dir)
    assert written == out_dir / SUMMARY_FILENAME
    summary = load_summary(written)
    assert summary.schema_version == SUMMARY_SCHEMA_VERSION
    assert summary.executed is False
    assert summary.package_version == __version__
    assert summary.dataset_sha256 == _DIGEST
    assert summary.manifest_id == "smoke-telemetry"
    payload = json.loads(written.read_text(encoding="utf-8"))
    assert payload["metrics"] == ["mae", "rmse"]
    assert "results" not in payload
    body = summary_to_mapping(summary)
    body.pop("summary_sha256")
    assert summary.summary_sha256 == canonical_sha256(body)


def test_run_rejects_dataset_hash_mismatch(tmp_path: Path) -> None:
    """Tampered payload bytes fail before a summary is written."""
    manifest_path, data_path, out_dir = _write_inputs(tmp_path)
    data_path.write_bytes(b"tampered-bytes")
    with pytest.raises(ManifestError, match="SHA256 mismatch"):
        run_manifest(manifest_path, data_path, out_dir)
    assert not (out_dir / SUMMARY_FILENAME).exists()


def test_verify_accepts_directory_or_file(tmp_path: Path) -> None:
    """``--against`` may be the artifact directory or ``summary.json``."""
    manifest_path, data_path, out_dir = _write_inputs(tmp_path)
    run_manifest(manifest_path, data_path, out_dir)
    from_dir = verify_summary(manifest_path, out_dir)
    from_file = verify_summary(manifest_path, out_dir / SUMMARY_FILENAME)
    assert from_dir == from_file
    assert from_dir.executed is False


def test_verify_fails_on_tampered_summary_hash(tmp_path: Path) -> None:
    """Changing ``summary_sha256`` without matching the body fails."""
    manifest_path, data_path, out_dir = _write_inputs(tmp_path)
    written = run_manifest(manifest_path, data_path, out_dir)
    payload = json.loads(written.read_text(encoding="utf-8"))
    payload["summary_sha256"] = "0" * 64
    written.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(SummaryError, match="summary_sha256 mismatch"):
        verify_summary(manifest_path, out_dir)


def test_verify_fails_on_tampered_field(tmp_path: Path) -> None:
    """Changing a bound field without updating the digest fails."""
    manifest_path, data_path, out_dir = _write_inputs(tmp_path)
    written = run_manifest(manifest_path, data_path, out_dir)
    payload = json.loads(written.read_text(encoding="utf-8"))
    payload["track"] = "multiphysics"
    written.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(SummaryError, match="summary_sha256 mismatch"):
        verify_summary(manifest_path, out_dir)


def test_verify_fails_when_rehashed_field_unbound(tmp_path: Path) -> None:
    """Rehashing after changing identity still fails the manifest bind."""
    manifest_path, data_path, out_dir = _write_inputs(tmp_path)
    written = run_manifest(manifest_path, data_path, out_dir)
    payload = json.loads(written.read_text(encoding="utf-8"))
    payload["track"] = "multiphysics"
    body = {key: value for key, value in payload.items() if key != "summary_sha256"}
    payload["summary_sha256"] = canonical_sha256(body)
    written.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(SummaryError, match="track"):
        verify_summary(manifest_path, out_dir)


def test_build_summary_is_deterministic() -> None:
    """Canonical hashing does not depend on dict insertion order."""
    first = build_summary(_manifest())
    second = build_summary(_manifest())
    assert first.summary_sha256 == second.summary_sha256
    assert first.manifest_sha256 == second.manifest_sha256


def test_load_summary_rejects_invalid_json(tmp_path: Path) -> None:
    """A non-mapping JSON root is a ``SummaryError``."""
    path = tmp_path / "summary.json"
    path.write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(SummaryError, match="mapping"):
        load_summary(path)


def test_load_summary_rejects_malformed_json(tmp_path: Path) -> None:
    """Broken JSON text is a ``SummaryError``."""
    path = tmp_path / "summary.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(SummaryError, match="Invalid JSON"):
        load_summary(path)


def test_load_summary_missing_file(tmp_path: Path) -> None:
    """Missing ``summary.json`` is a ``SummaryError``."""
    with pytest.raises(SummaryError, match="not found"):
        load_summary(tmp_path / "missing.json")


def test_run_missing_data_file(tmp_path: Path) -> None:
    """A missing ``--data`` path raises ``FileNotFoundError``."""
    manifest_path, data_path, out_dir = _write_inputs(tmp_path)
    data_path.unlink()
    with pytest.raises(FileNotFoundError):
        run_manifest(manifest_path, data_path, out_dir)


def _summary_kwargs(**overrides: object) -> dict[str, object]:
    """Return valid :class:`BenchmarkSummary` constructor kwargs.

    Parameters
    ----------
    **overrides
        Field replacements.

    Returns
    -------
    dict
        Keyword arguments for :class:`BenchmarkSummary`.
    """
    payload: dict[str, object] = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "manifest_id": "smoke-telemetry",
        "manifest_sha256": _DIGEST,
        "dataset_sha256": _DIGEST,
        "track": "telemetry",
        "methods": (SummaryMethodRef(name="graph_koopman", role="koopman"),),
        "seeds": (0, 1, 2),
        "horizons": (1, 3, 12),
        "metrics": ("mae", "rmse"),
        "controls": ("pernode",),
        "package_version": "0.0.0",
        "executed": False,
        "summary_sha256": _DIGEST,
    }
    payload.update(overrides)
    return payload


def _rehash_summary_file(path: Path, **updates: object) -> None:
    """Rewrite identity fields and refresh ``summary_sha256``.

    Parameters
    ----------
    path : Path
        ``summary.json`` path.
    **updates
        Mapping keys to overwrite before hashing.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(updates)
    body = {key: value for key, value in payload.items() if key != "summary_sha256"}
    payload["summary_sha256"] = canonical_sha256(body)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_summary_method_ref_rejects_empty_fields() -> None:
    """Empty method name or role is a ``SummaryError``."""
    with pytest.raises(SummaryError, match="summary.method.name"):
        SummaryMethodRef(name="", role="koopman")
    with pytest.raises(SummaryError, match="summary.method.role"):
        SummaryMethodRef(name="graph_koopman", role="")


def test_benchmark_summary_rejects_schema_and_identity_fields() -> None:
    """Schema, empty identity, empty methods, and bad method types raise."""
    with pytest.raises(SummaryError, match="schema_version"):
        BenchmarkSummary(**_summary_kwargs(schema_version="benchmark_summary_v0"))
    with pytest.raises(SummaryError, match="summary.track"):
        BenchmarkSummary(**_summary_kwargs(track=""))
    with pytest.raises(SummaryError, match="must be non-empty"):
        BenchmarkSummary(**_summary_kwargs(methods=()))
    with pytest.raises(SummaryError, match="SummaryMethodRef"):
        BenchmarkSummary(**_summary_kwargs(methods=("graph_koopman",)))


def test_benchmark_summary_rejects_invalid_digests() -> None:
    """Digest fields must be 64-character hexadecimal strings."""
    with pytest.raises(SummaryError, match="64-character hex"):
        BenchmarkSummary(**_summary_kwargs(manifest_sha256="abc"))
    with pytest.raises(SummaryError, match="64-character hex"):
        BenchmarkSummary(**_summary_kwargs(dataset_sha256="g" * 64))


def test_summary_from_mapping_rejects_shape_and_keys() -> None:
    """Mapping load rejects non-objects, unknown keys, and incomplete methods."""
    with pytest.raises(SummaryError, match="must be a mapping"):
        summary_from_mapping([1, 2])  # type: ignore[arg-type]
    valid = summary_to_mapping(build_summary(_manifest()))
    extra = dict(valid)
    extra["unexpected"] = True
    with pytest.raises(SummaryError, match="unknown summary keys"):
        summary_from_mapping(extra)
    missing = {key: value for key, value in valid.items() if key != "track"}
    with pytest.raises(SummaryError, match="missing required keys"):
        summary_from_mapping(missing)
    as_string = dict(valid)
    as_string["methods"] = "graph_koopman"
    with pytest.raises(SummaryError, match="sequence of mappings"):
        summary_from_mapping(as_string)
    with pytest.raises(SummaryError, match="must be a mapping"):
        summary_from_mapping({**valid, "methods": ["graph_koopman"]})
    with pytest.raises(SummaryError, match="missing required keys"):
        summary_from_mapping({**valid, "methods": [{"name": "graph_koopman"}]})


def test_verify_fails_when_rehashed_identity_unbound(tmp_path: Path) -> None:
    """Each bound identity field is checked after a matching digest."""
    manifest_path, data_path, out_dir = _write_inputs(tmp_path)
    written = run_manifest(manifest_path, data_path, out_dir)
    cases: tuple[tuple[dict[str, object], str], ...] = (
        ({"manifest_sha256": "a" * 64}, "manifest_sha256"),
        ({"manifest_id": "other-id"}, "manifest_id"),
        ({"dataset_sha256": "b" * 64}, "dataset_sha256"),
        ({"methods": [{"name": "other", "role": "koopman"}]}, "methods"),
        ({"seeds": [9, 8, 7]}, "seeds"),
        ({"horizons": [2, 4]}, "horizons"),
        ({"metrics": ["mae"]}, "metrics"),
        ({"controls": ["joint_ls"]}, "controls"),
    )
    original = written.read_text(encoding="utf-8")
    for updates, match in cases:
        written.write_text(original, encoding="utf-8")
        _rehash_summary_file(written, **updates)
        with pytest.raises(SummaryError, match=match):
            verify_summary(manifest_path, out_dir)
