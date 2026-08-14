"""Coverage and error-path tests for :mod:`koopman_graph.tracking`."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from koopman_graph.tracking.csv import CsvFitLogger
from koopman_graph.training.history import TrainingLossBreakdown


def _breakdown() -> TrainingLossBreakdown:
    """Return a deterministic all-zero loss breakdown."""
    return TrainingLossBreakdown.zeros(torch.device("cpu"))


def test_csv_logger_rejects_epoch_before_start(tmp_path: Path) -> None:
    """CSV logging requires on_fit_start before the first epoch."""
    logger = CsvFitLogger(tmp_path / "fit.csv")
    with pytest.raises(RuntimeError, match="before on_fit_start"):
        logger.on_epoch_end(
            epoch=0,
            train_breakdown=_breakdown(),
            val_breakdown=None,
            history_so_far=None,
        )
