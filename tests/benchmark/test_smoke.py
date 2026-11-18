"""Identity-bound manifests under ``benchmarks/v0.15/`` and ``benchmarks/v0.16/``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.helpers import REPO_ROOT

from koopman_graph.benchmark import (
    SummaryError,
    load_manifest,
    load_summary,
    run_manifest,
    verify_dataset_hash,
    verify_summary,
)

pytest.importorskip("yaml")

pytestmark = pytest.mark.benchmark_smoke

_BENCHMARKS = REPO_ROOT / "benchmarks"
#: ``(directory, manifest stem, payload path relative to the directory)``.
_TRACKS = (
    ("v0.15", "smoke_telemetry", "data/smoke_telemetry.txt"),
    ("v0.15", "smoke_multiphysics", "data/smoke_multiphysics.txt"),
    ("v0.15", "smoke_topology_transfer", "data/smoke_topology_transfer.txt"),
    ("v0.16", "reaction_michaelis_menten", "data/reaction_michaelis_menten_v1.pt"),
    (
        "v0.16",
        "reaction_van_de_vusse_dilution",
        "data/reaction_van_de_vusse_dilution_v1.pt",
    ),
    ("v0.16", "reaction_random_family", "data/reaction_random_family_v1/index.json"),
)
_TRACK_IDS = [f"{directory}/{name}" for directory, name, _ in _TRACKS]
_REQUIRED_CONTROLS = {
    "telemetry": frozenset({"pernode"}),
    "multiphysics": frozenset({"joint_ls"}),
    "topology_transfer": frozenset({"hold_last", "pernode", "joint_ls"}),
}


def _manifest_path(directory: str, name: str) -> Path:
    """Return the tracked YAML path for ``name``.

    Parameters
    ----------
    directory : str
        Benchmark directory such as ``v0.15``.
    name : str
        Stem such as ``smoke_telemetry``.

    Returns
    -------
    Path
        Manifest path.
    """
    return _BENCHMARKS / directory / f"{name}.yaml"


def _data_path(directory: str, payload: str) -> Path:
    """Return the tracked payload path.

    Parameters
    ----------
    directory : str
        Benchmark directory such as ``v0.15``.
    payload : str
        Payload path relative to ``directory``.

    Returns
    -------
    Path
        Payload path.
    """
    return _BENCHMARKS / directory / payload


def _summary_path(directory: str, name: str) -> Path:
    """Return the checked-in summary path for ``name``.

    Parameters
    ----------
    directory : str
        Benchmark directory such as ``v0.15``.
    name : str
        Stem such as ``smoke_telemetry``.

    Returns
    -------
    Path
        Summary JSON path.
    """
    return _BENCHMARKS / directory / "summaries" / f"{name}.json"


@pytest.mark.parametrize(("directory", "name", "payload"), _TRACKS, ids=_TRACK_IDS)
def test_smoke_manifest_controls_and_payload(
    directory: str, name: str, payload: str
) -> None:
    """Each manifest lists required controls and a matching payload digest."""
    manifest = load_manifest(_manifest_path(directory, name))
    required = _REQUIRED_CONTROLS[manifest.track]
    assert required <= set(manifest.controls)
    assert all(method.role != "teaching_gnn" for method in manifest.methods)
    assert any(method.role == "koopman" for method in manifest.methods)
    verify_dataset_hash(manifest.dataset, _data_path(directory, payload))
    assert (REPO_ROOT / manifest.dataset.card).is_file()
    checked = load_summary(_summary_path(directory, name))
    assert checked.executed is False
    assert checked.manifest_id == manifest.manifest_id


@pytest.mark.parametrize(("directory", "name", "payload"), _TRACKS, ids=_TRACK_IDS)
def test_smoke_run_then_verify(
    directory: str, name: str, payload: str, tmp_path: Path
) -> None:
    """``run_manifest`` then ``verify_summary`` succeeds on every manifest."""
    manifest_path = _manifest_path(directory, name)
    written = run_manifest(manifest_path, _data_path(directory, payload), tmp_path)
    live = verify_summary(manifest_path, written)
    assert live.executed is False
    checked = verify_summary(manifest_path, _summary_path(directory, name))
    assert checked.manifest_id == live.manifest_id
    assert checked.dataset_sha256 == live.dataset_sha256
    assert checked.summary_sha256 == live.summary_sha256 or (
        checked.package_version != live.package_version
    )


def test_checked_in_summary_tamper_fails(tmp_path: Path) -> None:
    """Tampering a checked-in ``summary_sha256`` fails ``verify``."""
    source = _summary_path("v0.15", "smoke_telemetry")
    tampered = tmp_path / "summary.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["summary_sha256"] = "0" * 64
    tampered.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(SummaryError, match="summary_sha256 mismatch"):
        verify_summary(_manifest_path("v0.15", "smoke_telemetry"), tampered)
