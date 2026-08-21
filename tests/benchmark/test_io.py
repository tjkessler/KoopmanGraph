"""JSON/YAML round-trip and dataset SHA-256 mismatch tests."""

from __future__ import annotations

import builtins
import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from koopman_graph.benchmark import (
    SCHEMA_VERSION,
    ComputeBudget,
    DatasetRef,
    ExperimentManifest,
    ManifestError,
    MethodSpec,
    PreprocessingSpec,
    SplitSpec,
    dump_manifest,
    load_manifest,
    verify_dataset_hash,
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


def test_json_round_trip(tmp_path: Path) -> None:
    """JSON dump then load recovers the frozen dataclass."""
    path = tmp_path / "manifest.json"
    original = _manifest()
    dump_manifest(original, path)
    restored = load_manifest(path)
    assert restored == original
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["seeds"] == [0, 1, 2]


def test_yaml_round_trip_when_installed(tmp_path: Path) -> None:
    """YAML dump then load recovers the frozen dataclass when PyYAML is present."""
    pytest.importorskip("yaml")
    path = tmp_path / "manifest.yaml"
    original = _manifest()
    dump_manifest(original, path)
    restored = load_manifest(path)
    assert restored == original


def test_yaml_requires_pyyaml(tmp_path: Path) -> None:
    """Missing PyYAML raises ImportError with the ``[cli]`` install hint."""
    path = tmp_path / "manifest.yaml"
    path.write_text("manifest_id: x\n", encoding="utf-8")
    real_import = builtins.__import__

    def blocked_import(
        name: str,
        globals: object = None,
        locals: object = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "yaml":
            raise ImportError("simulated missing yaml")
        return real_import(name, globals, locals, fromlist, level)

    with (
        patch.object(builtins, "__import__", side_effect=blocked_import),
        pytest.raises(ImportError, match="YAML manifests require PyYAML"),
    ):
        load_manifest(path)


def test_invalid_yaml_and_suffix(tmp_path: Path) -> None:
    """Malformed YAML and unknown suffixes raise ManifestError."""
    bad_suffix = tmp_path / "manifest.txt"
    bad_suffix.write_text("{}", encoding="utf-8")
    with pytest.raises(ManifestError, match="Unsupported manifest suffix"):
        load_manifest(bad_suffix)
    path = tmp_path / "invalid.yaml"
    path.write_text("invalid: [", encoding="utf-8")

    class FakeYamlError(Exception):
        """Synthetic YAML parser error."""

    fake_yaml = ModuleType("yaml")
    fake_yaml.YAMLError = FakeYamlError  # type: ignore[attr-defined]
    fake_yaml.safe_load = MagicMock(side_effect=FakeYamlError("bad yaml"))  # type: ignore[attr-defined]
    with (
        patch.dict(sys.modules, {"yaml": fake_yaml}),
        pytest.raises(ManifestError, match="Invalid YAML"),
    ):
        load_manifest(path)


def test_dataset_hash_mismatch_rejected(tmp_path: Path) -> None:
    """Tampered bytes and files fail the declared SHA-256."""
    dataset = _manifest().dataset
    verify_dataset_hash(dataset, _BYTES)
    payload = tmp_path / "payload.bin"
    payload.write_bytes(_BYTES)
    verify_dataset_hash(dataset, payload)
    with pytest.raises(ManifestError, match="SHA256 mismatch"):
        verify_dataset_hash(dataset, b"tampered")
    payload.write_bytes(b"tampered")
    with pytest.raises(ManifestError, match="SHA256 mismatch"):
        verify_dataset_hash(dataset, payload)


def test_missing_file_and_invalid_json(tmp_path: Path) -> None:
    """Missing paths and malformed JSON raise ManifestError."""
    with pytest.raises(ManifestError, match="manifest file not found"):
        load_manifest(tmp_path / "missing.json")
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    with pytest.raises(ManifestError, match="Invalid JSON"):
        load_manifest(invalid)
