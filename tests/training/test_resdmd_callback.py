"""Coverage and error-path tests for :mod:`koopman_graph.training`."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from koopman_graph.distributed.ray_train import multinode_recipe_enabled
from koopman_graph.operators import (
    sparse_leading_eigenvalues,
)
from koopman_graph.training import ResDMDFitCallback
from koopman_graph.training.history import TrainingLossBreakdown


def test_resdmd_callback_hooks() -> None:
    """Observe-only ResDMD hooks reset, ignore epoch events, and no-op on end."""
    callback = ResDMDFitCallback()
    callback.record(0.5)
    callback.on_fit_start(model=None, fit_kwargs={})
    assert callback.last_mean_residual is None
    callback.on_epoch_end(
        epoch=0,
        train_breakdown=TrainingLossBreakdown.zeros(torch.device("cpu")),
        val_breakdown=None,
        history_so_far=None,
    )
    callback.on_fit_end(history=SimpleNamespace())


def test_resdmd_callback_and_sparse_eigs() -> None:
    """Observe-only ResDMD callback and scipy/dense sparse eigensolve."""
    callback = ResDMDFitCallback()
    callback.record(0.25)
    assert callback.last_mean_residual == pytest.approx(0.25)
    vals = sparse_leading_eigenvalues(torch.eye(4), num_modes=2)
    assert vals.numel() == 2
    assert multinode_recipe_enabled() is False
