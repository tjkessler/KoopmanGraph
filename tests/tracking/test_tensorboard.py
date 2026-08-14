"""Coverage and error-path tests for :mod:`koopman_graph.tracking`."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest
import torch

import koopman_graph.tracking.tensorboard as tensorboard_mod
from koopman_graph.training.history import TrainingLossBreakdown


def _breakdown() -> TrainingLossBreakdown:
    """Return a deterministic all-zero loss breakdown."""
    return TrainingLossBreakdown.zeros(torch.device("cpu"))


def test_tensorboard_summary_writer_import_success() -> None:
    """The TensorBoard peer import returns SummaryWriter when available."""
    writer_cls = object()
    fake_module = ModuleType("torch.utils.tensorboard")
    fake_module.SummaryWriter = writer_cls  # type: ignore[attr-defined]
    with patch.dict(sys.modules, {"torch.utils.tensorboard": fake_module}):
        assert tensorboard_mod._require_summary_writer() is writer_cls


def test_tensorboard_logger_lifecycle_with_mocked_writer(tmp_path: Path) -> None:
    """Mocked TensorBoard logging covers start, train/val, flush, and close."""
    writer = MagicMock()
    writer_cls = MagicMock(return_value=writer)
    logger = tensorboard_mod.TensorBoardFitLogger(
        tmp_path / "events",
        comment="coverage",
    )

    with pytest.raises(RuntimeError, match="before on_fit_start"):
        logger.on_epoch_end(
            epoch=0,
            train_breakdown=_breakdown(),
            val_breakdown=None,
            history_so_far=None,
        )

    with patch.object(
        tensorboard_mod,
        "_require_summary_writer",
        return_value=writer_cls,
    ):
        logger.on_fit_start(model=object(), fit_kwargs={})
    logger.on_epoch_end(
        epoch=3,
        train_breakdown=_breakdown(),
        val_breakdown=_breakdown(),
        history_so_far=None,
    )
    logger.on_fit_end(history=MagicMock())

    writer_cls.assert_called_once_with(
        log_dir=str(tmp_path / "events"),
        comment="coverage",
    )
    assert writer.add_scalar.call_count == 22
    assert writer.flush.call_count == 2
    writer.close.assert_called_once_with()
    assert logger._writer is None
