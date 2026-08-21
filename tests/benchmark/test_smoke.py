"""Identity-bound smoke manifests under ``benchmarks/v0.15/``."""

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

_SMOKE_ROOT = REPO_ROOT / "benchmarks" / "v0.15"
_TRACKS = (
    "smoke_telemetry",
    "smoke_multiphysics",
    "smoke_topology_transfer",
)
_REQUIRED_CONTROLS = {
    "telemetry": frozenset({"pernode"}),
    "multiphysics": frozenset({"joint_ls"}),
    "topology_transfer": frozenset({"hold_last", "pernode", "joint_ls"}),
}


def _manifest_path(name: str) -> Path:
    """Return the tracked YAML path for ``name``.

    Parameters
    ----------
    name : str
        Stem such as ``smoke_telemetry``.

    Returns
    -------
    Path
        Manifest path.
    """
    return _SMOKE_ROOT / f"{name}.yaml"


def _data_path(name: str) -> Path:
    """Return the tracked payload path for ``name``.

    Parameters
    ----------
    name : str
        Stem such as ``smoke_telemetry``.

    Returns
    -------
    Path
        Payload path.
    """
    return _SMOKE_ROOT / "data" / f"{name}.txt"


def _summary_path(name: str) -> Path:
    """Return the checked-in summary path for ``name``.

    Parameters
    ----------
    name : str
        Stem such as ``smoke_telemetry``.

    Returns
    -------
    Path
        Summary JSON path.
    """
    return _SMOKE_ROOT / "summaries" / f"{name}.json"


@pytest.mark.parametrize("name", _TRACKS)
def test_smoke_manifest_controls_and_payload(name: str) -> None:
    """Each track stub lists required controls and a matching payload digest."""
    manifest = load_manifest(_manifest_path(name))
    required = _REQUIRED_CONTROLS[manifest.track]
    assert required <= set(manifest.controls)
    assert all(method.role != "teaching_gnn" for method in manifest.methods)
    assert any(method.role == "koopman" for method in manifest.methods)
    verify_dataset_hash(manifest.dataset, _data_path(name))
    checked = load_summary(_summary_path(name))
    assert checked.executed is False
    assert checked.manifest_id == manifest.manifest_id


@pytest.mark.parametrize("name", _TRACKS)
def test_smoke_run_then_verify(name: str, tmp_path: Path) -> None:
    """``run_manifest`` then ``verify_summary`` succeeds on the stand-in."""
    written = run_manifest(_manifest_path(name), _data_path(name), tmp_path)
    live = verify_summary(_manifest_path(name), written)
    assert live.executed is False
    checked = verify_summary(_manifest_path(name), _summary_path(name))
    assert checked.manifest_id == live.manifest_id
    assert checked.dataset_sha256 == live.dataset_sha256
    assert checked.summary_sha256 == live.summary_sha256 or (
        checked.package_version != live.package_version
    )


def test_checked_in_summary_tamper_fails(tmp_path: Path) -> None:
    """Tampering a checked-in ``summary_sha256`` fails ``verify``."""
    source = _summary_path("smoke_telemetry")
    tampered = tmp_path / "summary.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["summary_sha256"] = "0" * 64
    tampered.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(SummaryError, match="summary_sha256 mismatch"):
        verify_summary(_manifest_path("smoke_telemetry"), tampered)
