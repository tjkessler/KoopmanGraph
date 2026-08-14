"""Smoke tests for the observe-only :class:`FitCallback` protocol."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict
from typing import Any

import pytest
import torch

from koopman_graph import (
    FitCallback,
    GNNDecoder,
    GNNEncoder,
    GraphKoopmanModel,
    NoOpFitCallback,
)
from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.training import FitCallback as TrainingFitCallback
from koopman_graph.training import NoOpFitCallback as TrainingNoOpFitCallback
from koopman_graph.training import run_fit_loop
from koopman_graph.training.history import FitHistory, TrainingLossBreakdown


def test_noop_is_runtime_checkable_fit_callback() -> None:
    """``NoOpFitCallback`` satisfies ``FitCallback`` at runtime."""
    callback = NoOpFitCallback()
    assert isinstance(callback, FitCallback)


def test_missing_hook_fails_runtime_checkable() -> None:
    """Structural objects missing a required hook are not FitCallbacks."""

    class _Incomplete:
        def on_fit_start(self, *, model: Any, fit_kwargs: Mapping[str, Any]) -> None:
            return None

        def on_fit_end(self, *, history: FitHistory) -> None:
            return None

    assert not isinstance(_Incomplete(), FitCallback)


def test_noop_hooks_accept_protocol_arguments() -> None:
    """No-op hooks accept the designed keyword shapes without error."""
    callback = NoOpFitCallback()
    breakdown = TrainingLossBreakdown.zeros(torch.device("cpu"))
    history = FitHistory(loss=(1.0,), epochs=1)

    callback.on_fit_start(model=object(), fit_kwargs={"epochs": 1})
    callback.on_epoch_end(
        epoch=0,
        train_breakdown=breakdown,
        val_breakdown=None,
        history_so_far=None,
    )
    callback.on_fit_end(history=history)


def test_fit_callback_exported_from_training_package() -> None:
    """Protocol symbols are part of the curated ``training`` ``__all__``."""
    import koopman_graph.training as training

    assert "FitCallback" in training.__all__
    assert "NoOpFitCallback" in training.__all__
    assert training.FitCallback is TrainingFitCallback
    assert training.NoOpFitCallback is TrainingNoOpFitCallback


def test_fit_callback_exported_from_root_package() -> None:
    """Q11: ``FitCallback`` / ``NoOpFitCallback`` are root ``__all__`` exports."""
    import koopman_graph as kg

    assert "FitCallback" in kg.__all__
    assert "NoOpFitCallback" in kg.__all__
    assert kg.FitCallback is FitCallback
    assert kg.NoOpFitCallback is NoOpFitCallback
    assert FitCallback is TrainingFitCallback
    assert NoOpFitCallback is TrainingNoOpFitCallback


class _RecordingFitCallback:
    """Test double that records hook order and epoch indices."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self.epochs: list[int] = []
        self.final_history: FitHistory | None = None

    def on_fit_start(self, *, model: Any, fit_kwargs: Mapping[str, Any]) -> None:
        assert "epochs" in fit_kwargs
        self.events.append("start")

    def on_epoch_end(
        self,
        *,
        epoch: int,
        train_breakdown: TrainingLossBreakdown,
        val_breakdown: TrainingLossBreakdown | None,
        history_so_far: FitHistory | None,
    ) -> None:
        assert history_so_far is not None
        assert history_so_far.epochs == epoch + 1
        self.events.append(f"epoch:{epoch}")
        self.epochs.append(epoch)

    def on_fit_end(self, *, history: FitHistory) -> None:
        self.events.append("end")
        self.final_history = history


def _tiny_model() -> GraphKoopmanModel:
    """Build a small discrete GCN for callback loop tests."""
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


def test_run_fit_loop_callback_order(
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Recording callback sees start → epochs → end with zero-based indices."""
    model = _tiny_model()
    recorder = _RecordingFitCallback()
    torch.manual_seed(0)
    history = run_fit_loop(
        model,
        [scaling_sequence],
        epochs=3,
        lr=1e-2,
        device="cpu",
        callbacks=[recorder],
    )

    assert recorder.events == ["start", "epoch:0", "epoch:1", "epoch:2", "end"]
    assert recorder.epochs == [0, 1, 2]
    assert recorder.final_history == history
    assert history.epochs == 3


def test_noop_callbacks_preserve_fit_history(
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Same seed: NoOp callbacks leave FitHistory identical to no callbacks."""
    torch.manual_seed(1)
    model_a = _tiny_model()
    state = deepcopy(model_a.state_dict())
    model_b = _tiny_model()
    model_b.load_state_dict(deepcopy(state))

    torch.manual_seed(2)
    history_a = run_fit_loop(
        model_a,
        [scaling_sequence],
        epochs=2,
        lr=1e-2,
        device="cpu",
        callbacks=None,
    )
    torch.manual_seed(2)
    history_b = run_fit_loop(
        model_b,
        [scaling_sequence],
        epochs=2,
        lr=1e-2,
        device="cpu",
        callbacks=[NoOpFitCallback()],
    )

    assert asdict(history_a) == asdict(history_b)
    for key in state:
        torch.testing.assert_close(
            model_a.state_dict()[key],
            model_b.state_dict()[key],
            rtol=0.0,
            atol=0.0,
        )


def test_model_fit_forwards_callbacks(
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Public ``GraphKoopmanModel.fit`` forwards callbacks to the fit loop."""
    model = _tiny_model()
    recorder = _RecordingFitCallback()
    torch.manual_seed(3)
    history = model.fit(
        scaling_sequence,
        epochs=2,
        lr=1e-2,
        device="cpu",
        callbacks=[recorder],
    )

    assert recorder.events == ["start", "epoch:0", "epoch:1", "end"]
    assert recorder.final_history == history


def test_model_fit_ddp_rejects_callbacks(
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """``strategy="ddp"`` rejects non-None callbacks until DDP wiring exists."""
    model = _tiny_model()
    with pytest.raises(ValueError, match='strategy="ddp"'):
        model.fit(
            scaling_sequence,
            epochs=1,
            lr=1e-2,
            device="cpu",
            callbacks=[NoOpFitCallback()],
            strategy="ddp",
        )
