"""Coverage and error-path tests for :mod:`koopman_graph.cli`."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch_geometric.data import Data

import koopman_graph.cli.config as config_mod
import koopman_graph.cli.predict as predict_mod
from koopman_graph.data import GraphSnapshotSequence


def _snapshot_sequence() -> GraphSnapshotSequence:
    """Return a deterministic two-snapshot homogeneous sequence."""
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    snapshots = [
        Data(x=torch.full((2, 2), float(step)), edge_index=edge_index)
        for step in range(2)
    ]
    return GraphSnapshotSequence(snapshots)


def test_predict_sequence_loader_covers_config_and_file_branches(
    tmp_path: Path,
) -> None:
    """Prediction data loading covers config, cache, and suffix validation."""
    config_path = tmp_path / "data.json"
    config_path.write_text("{}", encoding="utf-8")
    sentinel = _snapshot_sequence()

    with (
        patch.object(predict_mod, "load_config", return_value={"data": []}),
        pytest.raises(config_mod.ConfigError, match="config.data must be a mapping"),
    ):
        predict_mod.load_sequence_from_path(config_path)

    with (
        patch.object(
            predict_mod,
            "load_config",
            return_value={"kind": "synthetic_path"},
        ),
        patch.object(
            predict_mod,
            "build_sequence_from_config",
            return_value=sentinel,
        ),
    ):
        assert predict_mod.load_sequence_from_path(config_path) is sentinel

    with (
        patch.object(predict_mod, "load_config", return_value={"other": "value"}),
        pytest.raises(config_mod.ConfigError, match="must contain a 'data' section"),
    ):
        predict_mod.load_sequence_from_path(config_path)

    cache_path = tmp_path / "data.pt"
    cache_path.write_bytes(b"placeholder")
    with patch.object(
        predict_mod,
        "load_cached_sequence",
        return_value=sentinel,
    ) as cached_loader:
        assert predict_mod.load_sequence_from_path(cache_path) is sentinel
    cached_loader.assert_called_once()

    unsupported = tmp_path / "data.csv"
    unsupported.write_text("x\n", encoding="utf-8")
    with pytest.raises(config_mod.ConfigError, match="Unsupported data path suffix"):
        predict_mod.load_sequence_from_path(unsupported)

    with pytest.raises(config_mod.ConfigError, match="Data path not found"):
        predict_mod.load_sequence_from_path(tmp_path / "missing.pt")


def test_predict_run_rejects_invalid_steps_empty_data_and_empty_forecast(
    tmp_path: Path,
) -> None:
    """Prediction validates horizon, input sequence, and forecast output."""
    with pytest.raises(config_mod.ConfigError, match="--steps must be >= 1"):
        predict_mod.run_predict(
            checkpoint="model",
            data="data",
            steps=0,
            out=tmp_path / "out.pt",
        )

    with (
        patch.object(predict_mod, "load_checkpoint", return_value=MagicMock()),
        patch.object(predict_mod, "load_sequence_from_path", return_value=[]),
        pytest.raises(config_mod.ConfigError, match="Data sequence is empty"),
    ):
        predict_mod.run_predict(
            checkpoint="model",
            data="data",
            steps=1,
            out=tmp_path / "out.pt",
        )

    model = MagicMock()
    model.predict.return_value = []
    with (
        patch.object(predict_mod, "load_checkpoint", return_value=model),
        patch.object(
            predict_mod,
            "load_sequence_from_path",
            return_value=[Data(x=torch.ones(1, 1))],
        ),
        pytest.raises(config_mod.ConfigError, match="empty forecast"),
    ):
        predict_mod.run_predict(
            checkpoint="model",
            data="data",
            steps=1,
            out=tmp_path / "out.pt",
        )


def test_predict_handler_prints_path_without_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The predict handler reports successful forecasts without shape metadata."""
    output = tmp_path / "forecast.pt"
    args = SimpleNamespace(
        checkpoint="model",
        data="data",
        steps=2,
        out=output,
    )
    with patch.object(predict_mod, "run_predict", return_value=(output, None)):
        assert predict_mod.handle_predict(args) == 0  # type: ignore[arg-type]
    assert f"wrote forecast: {output} (steps=2)" in capsys.readouterr().out
