"""Coverage and error-path tests for :mod:`koopman_graph.training`."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from tests.identification.test_fit_identification import (
    _identity_model,
    _linear_sequence,
)

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
    assert callback.polluted is True
    assert callback.rejected is False
    callback.on_fit_start(model=None, fit_kwargs={})
    assert callback.last_mean_residual is None
    assert callback.last_residual_max is None
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


def test_resdmd_callback_rejects_invalid_mode() -> None:
    """Unknown ``mode`` values raise at construction."""
    with pytest.raises(ValueError, match="mode must be"):
        ResDMDFitCallback(mode="abort")  # type: ignore[arg-type]


def test_resdmd_gate_mode_raises_at_fit_end() -> None:
    """Gate mode records pollution and raises from ``on_fit_end``."""
    callback = ResDMDFitCallback(mode="gate")
    callback.record(0.5)
    assert callback.rejected is True
    with pytest.raises(ValueError, match="ResDMD gate rejected"):
        callback.on_fit_end(history=SimpleNamespace())


def test_resdmd_observe_encodings_records_residual() -> None:
    """Identity-dictionary stacks with ``T >= 3`` populate residuals."""
    callback = ResDMDFitCallback()
    true_k = torch.diag(torch.tensor([0.8, 0.5], dtype=torch.float64))
    state = torch.tensor([0.4, -0.2], dtype=torch.float64)
    rows = []
    for _ in range(12):
        rows.append(state.clone())
        state = state @ true_k.T
    encodings = torch.stack(rows, dim=0)
    callback.observe_encodings(encodings)
    assert callback.last_mean_residual is not None
    assert callback.last_residual_max is not None
    assert callback.polluted is False
    stacked = ResDMDFitCallback()
    stacked.observe_encodings(encodings.unsqueeze(1))
    assert stacked.last_mean_residual is not None
    short = ResDMDFitCallback()
    short.observe_encodings(encodings[:2])
    assert short.last_mean_residual is None
    with pytest.raises(ValueError, match="shape"):
        callback.observe_encodings(torch.zeros(4, dtype=torch.float64))


def test_resdmd_callback_observe_during_identity_fit() -> None:
    """Fit-loop wiring calls ``observe_encodings`` on the first sequence."""
    true_k = torch.diag(torch.tensor([0.8, 0.5], dtype=torch.float64))
    sequence, _ = _linear_sequence(true_k, n_nodes=1, n_times=16)
    model = _identity_model(2, dtype=torch.float64)
    callback = ResDMDFitCallback()
    model.fit(sequence, epochs=1, callbacks=[callback], lr=0.0)
    assert callback.last_mean_residual is not None
    assert callback.last_residual_max is not None
    assert callback.polluted is False
    gated = ResDMDFitCallback(mode="gate")
    model.fit(sequence, epochs=1, callbacks=[gated], lr=0.0)
    assert gated.rejected is False
