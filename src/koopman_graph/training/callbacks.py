"""Observe-only hooks for :func:`~koopman_graph.training.loop.run_fit_loop`.

Callbacks must not mutate model parameters. Wiring into the fit loop is
added in a later task; this module defines the protocol and a no-op
implementation for typing and smoke tests.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from koopman_graph.training.history import FitHistory, TrainingLossBreakdown


@runtime_checkable
class FitCallback(Protocol):
    """Protocol for epoch-level observers during ``fit`` / ``run_fit_loop``.

    Implementations should treat all arguments as read-only. Do not mutate
    model parameters, optimizer state, or loss breakdown tensors in place.

    Notes
    -----
    Concrete adapters (CSV, TensorBoard, custom loggers) implement this
    protocol; :class:`NoOpFitCallback` is the observe-nothing default.
    """

    def on_fit_start(self, *, model: Any, fit_kwargs: Mapping[str, Any]) -> None:
        """Called once before the first training epoch.

        Parameters
        ----------
        model : Any
            Trainable model about to be fit (typically
            :class:`~koopman_graph.model.GraphKoopmanModel`).
        fit_kwargs : Mapping[str, Any]
            Keyword arguments supplied to the fit loop (epochs, device,
            loss weights, etc.). Contents are informational and may omit
            internal-only resolved objects.
        """
        ...

    def on_epoch_end(
        self,
        *,
        epoch: int,
        train_breakdown: TrainingLossBreakdown,
        val_breakdown: TrainingLossBreakdown | None,
        history_so_far: FitHistory | None,
    ) -> None:
        """Called after each completed epoch.

        Parameters
        ----------
        epoch : int
            Zero-based epoch index that just finished.
        train_breakdown : TrainingLossBreakdown
            Mean training loss terms for the epoch.
        val_breakdown : TrainingLossBreakdown or None
            Mean validation loss terms when validation is enabled.
        history_so_far : FitHistory or None
            History accumulated through this epoch when available; may be
            ``None`` until the loop builds a public history object.
        """
        ...

    def on_fit_end(self, *, history: FitHistory) -> None:
        """Called once after training finishes (including early stop).

        Parameters
        ----------
        history : FitHistory
            Final immutable training history returned to the caller.
        """
        ...


class NoOpFitCallback:
    """Concrete :class:`FitCallback` that records nothing and mutates nothing.

    Notes
    -----
    Useful as a typed default when no logging adapter is attached.
    """

    def on_fit_start(self, *, model: Any, fit_kwargs: Mapping[str, Any]) -> None:
        """No-op fit start hook.

        Parameters
        ----------
        model : Any
            Unused model reference.
        fit_kwargs : Mapping[str, Any]
            Unused fit keyword arguments.
        """

    def on_epoch_end(
        self,
        *,
        epoch: int,
        train_breakdown: TrainingLossBreakdown,
        val_breakdown: TrainingLossBreakdown | None,
        history_so_far: FitHistory | None,
    ) -> None:
        """No-op epoch-end hook.

        Parameters
        ----------
        epoch : int
            Unused epoch index.
        train_breakdown : TrainingLossBreakdown
            Unused training loss breakdown.
        val_breakdown : TrainingLossBreakdown or None
            Unused validation loss breakdown.
        history_so_far : FitHistory or None
            Unused partial history.
        """

    def on_fit_end(self, *, history: FitHistory) -> None:
        """No-op fit end hook.

        Parameters
        ----------
        history : FitHistory
            Unused final fit history.
        """
