"""Tests for ``koopman_graph.tracking`` fit adapters."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch
from tests.helpers import REPO_ROOT

from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.tracking import (
    CSV_FIT_FIELDNAMES,
    CsvFitLogger,
    TensorBoardFitLogger,
)


def _tiny_model() -> GraphKoopmanModel:
    """Build a small discrete GCN for tracking integration tests."""
    return GraphKoopmanModel(
        encoder=GNNEncoder(
            in_channels=3, hidden_channels=8, latent_dim=4, num_layers=2
        ),
        decoder=GNNDecoder(
            latent_dim=4, hidden_channels=8, out_channels=3, num_layers=2
        ),
        latent_dim=4,
        time_step=0.1,
    )


def test_csv_fit_logger_writes_expected_rows(
    scaling_sequence: GraphSnapshotSequence,
    tmp_path: Path,
) -> None:
    """Tiny fit with CsvFitLogger writes header + one row per epoch."""
    csv_path = tmp_path / "fit_log.csv"
    logger = CsvFitLogger(csv_path)
    model = _tiny_model()
    torch.manual_seed(0)
    history = model.fit(
        scaling_sequence,
        epochs=3,
        lr=1e-2,
        device="cpu",
        callbacks=[logger],
    )

    assert csv_path.is_file()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(CSV_FIT_FIELDNAMES)
        rows = list(reader)

    assert len(rows) == history.epochs == 3
    assert [int(row["epoch"]) for row in rows] == [0, 1, 2]
    for row, expected_total in zip(rows, history.loss, strict=True):
        assert float(row["train_total"]) == expected_total
        assert row["train_reconstruction"] != ""
        assert row["val_total"] == ""


def test_csv_fit_logger_writes_val_columns(
    scaling_sequence: GraphSnapshotSequence,
    tmp_path: Path,
) -> None:
    """Validation breakdown populates ``val_*`` columns when provided."""
    csv_path = tmp_path / "fit_val.csv"
    model = _tiny_model()
    torch.manual_seed(1)
    history = model.fit(
        scaling_sequence,
        epochs=2,
        lr=1e-2,
        device="cpu",
        validation_sequence=scaling_sequence,
        callbacks=[CsvFitLogger(csv_path)],
    )

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2
    assert history.val_loss is not None
    for row, expected_val in zip(rows, history.val_loss, strict=True):
        assert float(row["val_total"]) == expected_val
        assert row["val_reconstruction"] != ""


def test_training_does_not_import_tracking() -> None:
    """Layer rule: training sources must not import tracking adapters."""
    package_root = REPO_ROOT / "src" / "koopman_graph"
    training_root = package_root / "training"
    offenders: list[str] = []
    for path in training_root.rglob("*.py"):
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            from_tracking = stripped.startswith("from koopman_graph.tracking")
            import_tracking = stripped.startswith("import koopman_graph.tracking")
            if from_tracking or import_tracking:
                offenders.append(str(path.relative_to(package_root.parent)))
                break
    assert offenders == []


def test_tensorboard_fit_logger_writes_event_files(
    scaling_sequence: GraphSnapshotSequence,
    tmp_path: Path,
) -> None:
    """Tiny fit writes TensorBoard event files when tensorboard is installed."""
    pytest.importorskip("tensorboard")
    log_dir = tmp_path / "tb_runs"
    model = _tiny_model()
    torch.manual_seed(2)
    model.fit(
        scaling_sequence,
        epochs=2,
        lr=1e-2,
        device="cpu",
        validation_sequence=scaling_sequence,
        callbacks=[TensorBoardFitLogger(log_dir)],
    )

    event_files = list(log_dir.rglob("events.out.tfevents.*"))
    assert event_files, f"expected TensorBoard events under {log_dir}"


def test_tensorboard_fit_logger_import_error_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing tensorboard raises ImportError with an install hint."""
    import builtins

    import koopman_graph.tracking.tensorboard as tb_mod

    real_import = builtins.__import__

    def blocked(
        name: str,
        globals: object = None,
        locals: object = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "torch.utils.tensorboard":
            raise ImportError("simulated missing tensorboard")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError, match="pip install tensorboard"):
        tb_mod._require_summary_writer()
