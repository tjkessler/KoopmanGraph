"""CLI tests for ``koopman-graph benchmark run`` / ``verify``."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from koopman_graph.benchmark import (
    SCHEMA_VERSION,
    SUMMARY_FILENAME,
    ComputeBudget,
    DatasetRef,
    ExperimentManifest,
    MethodSpec,
    PreprocessingSpec,
    SplitSpec,
    dump_manifest,
)
from koopman_graph.cli import main

_BYTES = b"fixture-bytes"
_DIGEST = hashlib.sha256(_BYTES).hexdigest()


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Write a telemetry manifest and matching payload.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory.

    Returns
    -------
    tuple of Path
        ``(manifest, data, out_dir)``.
    """
    manifest = ExperimentManifest(
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
    manifest_path = tmp_path / "manifest.json"
    data_path = tmp_path / "payload.bin"
    out_dir = tmp_path / "artifacts"
    dump_manifest(manifest, manifest_path)
    data_path.write_bytes(_BYTES)
    return manifest_path, data_path, out_dir


def test_benchmark_help_lists_run_and_verify(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``koopman-graph benchmark --help`` lists ``run`` and ``verify``."""
    with pytest.raises(SystemExit) as exc_info:
        main(["benchmark", "--help"])
    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "run" in help_text
    assert "verify" in help_text


def test_benchmark_without_subcommand_prints_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Bare ``benchmark`` prints nested help and exits 0."""
    assert main(["benchmark"]) == 0
    help_text = capsys.readouterr().out
    assert "run" in help_text
    assert "verify" in help_text


def test_benchmark_run_requires_data(tmp_path: Path) -> None:
    """``run`` without ``--data`` is an argparse error."""
    manifest_path, _, out_dir = _write_inputs(tmp_path)
    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "benchmark",
                "run",
                "--manifest",
                str(manifest_path),
                "--out",
                str(out_dir),
            ]
        )
    assert exc_info.value.code == 2


def test_benchmark_run_and_verify_succeed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``run`` then ``verify`` against the artifact directory exits 0."""
    manifest_path, data_path, out_dir = _write_inputs(tmp_path)
    run_code = main(
        [
            "benchmark",
            "run",
            "--manifest",
            str(manifest_path),
            "--data",
            str(data_path),
            "--out",
            str(out_dir),
        ]
    )
    assert run_code == 0
    summary_path = out_dir / SUMMARY_FILENAME
    assert summary_path.is_file()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["executed"] is False

    verify_code = main(
        [
            "benchmark",
            "verify",
            "--manifest",
            str(manifest_path),
            "--against",
            str(out_dir),
        ]
    )
    assert verify_code == 0
    out = capsys.readouterr().out
    assert "wrote summary:" in out
    assert "verified summary:" in out


def test_benchmark_verify_tampered_hash_exits_one(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Tampering ``summary_sha256`` exits 1 with ``error:`` on stderr."""
    manifest_path, data_path, out_dir = _write_inputs(tmp_path)
    assert (
        main(
            [
                "benchmark",
                "run",
                "--manifest",
                str(manifest_path),
                "--data",
                str(data_path),
                "--out",
                str(out_dir),
            ]
        )
        == 0
    )
    written = out_dir / SUMMARY_FILENAME
    payload = json.loads(written.read_text(encoding="utf-8"))
    payload["summary_sha256"] = "0" * 64
    written.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    capsys.readouterr()
    code = main(
        [
            "benchmark",
            "verify",
            "--manifest",
            str(manifest_path),
            "--against",
            str(out_dir),
        ]
    )
    assert code == 1
    err = capsys.readouterr().err
    assert "error:" in err
    assert "summary_sha256" in err
