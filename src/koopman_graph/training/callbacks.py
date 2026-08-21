"""Fit-loop observer hooks for :func:`~koopman_graph.training.loop.run_fit_loop`.

Callbacks must not mutate model parameters. The loop invokes
``on_fit_start``, ``on_epoch_end``, and ``on_fit_end``. Callbacks that
expose ``observe_encodings`` receive a frozen time-major latent stack
from the first training sequence at epoch end (identity-dictionary
ResDMD layout).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Literal, Protocol, runtime_checkable

from torch import Tensor

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


class ResDMDFitCallback:
    """Finite-dictionary ResDMD observer with optional opt-in gate.

    Default ``mode="observe"`` records residuals and does not abort.
    ``mode="gate"`` sets :attr:`rejected` when the max residual exceeds
    ``tolerance`` and raises :class:`ValueError` from :meth:`on_fit_end`.
    Does not mutate model parameters. Not a certified infinite-dimensional
    residual bound.

    Attributes
    ----------
    mode : {"observe", "gate"}
        Observe-only or reject-at-fit-end.
    tolerance : float
        Max-residual cutoff (same default as
        :func:`~koopman_graph.analysis.resdmd`).
    last_mean_residual : float or None
        Mean residual from the last :meth:`record` / :meth:`observe_encodings`.
    last_residual_max : float or None
        Max residual from the last observation.
    polluted : bool or None
        Whether the last max residual exceeded ``tolerance``.
    rejected : bool
        True when ``mode="gate"`` and a polluted residual was recorded.
    """

    def __init__(
        self,
        *,
        mode: Literal["observe", "gate"] = "observe",
        tolerance: float = 1e-2,
    ) -> None:
        """Initialize with no recorded residual.

        Parameters
        ----------
        mode : {"observe", "gate"}, optional
            Default ``"observe"``.
        tolerance : float, optional
            Pollution cutoff. Default ``1e-2``.

        Raises
        ------
        ValueError
            If ``mode`` is unknown or ``tolerance`` is invalid.
        """
        if mode not in {"observe", "gate"}:
            msg = f"mode must be 'observe' or 'gate', got {mode!r}"
            raise ValueError(msg)
        if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
            msg = f"tolerance must be a finite float, got {type(tolerance).__name__}"
            raise ValueError(msg)
        if not math.isfinite(float(tolerance)) or float(tolerance) < 0.0:
            msg = f"tolerance must be a finite non-negative float, got {tolerance!r}"
            raise ValueError(msg)
        self.mode: Literal["observe", "gate"] = mode
        self.tolerance = float(tolerance)
        self.last_mean_residual: float | None = None
        self.last_residual_max: float | None = None
        self.polluted: bool | None = None
        self.rejected = False

    def on_fit_start(self, *, model: Any, fit_kwargs: Mapping[str, Any]) -> None:
        """Reset recorded residuals and gate flags.

        Parameters
        ----------
        model : Any
            Unused model reference.
        fit_kwargs : Mapping[str, Any]
            Unused fit keyword arguments.
        """
        del model, fit_kwargs
        self.last_mean_residual = None
        self.last_residual_max = None
        self.polluted = None
        self.rejected = False

    def on_epoch_end(
        self,
        *,
        epoch: int,
        train_breakdown: TrainingLossBreakdown,
        val_breakdown: TrainingLossBreakdown | None,
        history_so_far: FitHistory | None,
    ) -> None:
        """No parameter updates; residuals arrive via :meth:`observe_encodings`.

        Parameters
        ----------
        epoch : int
            Unused epoch index.
        train_breakdown : TrainingLossBreakdown
            Unused training loss.
        val_breakdown : TrainingLossBreakdown or None
            Unused validation loss.
        history_so_far : FitHistory or None
            Unused history.
        """
        del epoch, train_breakdown, val_breakdown, history_so_far

    def on_fit_end(self, *, history: FitHistory) -> None:
        """Raise in gate mode when a polluted residual was recorded.

        Parameters
        ----------
        history : FitHistory
            Unused final history.

        Raises
        ------
        ValueError
            If ``mode="gate"`` and :attr:`rejected` is True.
        """
        del history
        if self.mode == "gate" and self.rejected:
            peak = self.last_residual_max
            msg = (
                "ResDMD gate rejected the fit dictionary "
                f"(residual_max={peak!r} > tolerance={self.tolerance:g})"
            )
            raise ValueError(msg)

    def record(self, residual: float, *, residual_max: float | None = None) -> None:
        """Store a scalar residual observation.

        Parameters
        ----------
        residual : float
            Mean finite-dictionary ResDMD residual.
        residual_max : float or None, optional
            Max residual used for the pollution test. Default uses
            ``residual`` as both mean and max.

        Raises
        ------
        ValueError
            If a residual is non-finite or negative.
        """
        mean_residual = float(residual)
        peak = mean_residual if residual_max is None else float(residual_max)
        for name, value in (("residual", mean_residual), ("residual_max", peak)):
            if not math.isfinite(value) or value < 0.0:
                msg = f"{name} must be a finite non-negative float, got {value!r}"
                raise ValueError(msg)
        self.last_mean_residual = mean_residual
        self.last_residual_max = peak
        self.polluted = peak > self.tolerance
        if self.mode == "gate" and self.polluted:
            self.rejected = True

    def observe_encodings(self, encodings: Tensor) -> None:
        """Run identity-dictionary ResDMD on a time-major latent stack.

        Layout matches ``evaluate(..., include_resdmd=True)``: one row per
        snapshot, width :math:`N\\cdot d` (or already ``(T, m)``). Needs
        ``T >= 3``. Lazy-imports :mod:`koopman_graph.analysis.resdmd`.

        Parameters
        ----------
        encodings : Tensor
            Time-major latents ``(T, N, d)`` or ``(T, m)``.

        Raises
        ------
        ValueError
            If the tensor rank is not 2 or 3.
        """
        if encodings.ndim == 3:
            stacked = encodings.reshape(encodings.shape[0], -1)
        elif encodings.ndim == 2:
            stacked = encodings
        else:
            msg = (
                "encodings must have shape (T, d) or (T, N, d), "
                f"got {tuple(encodings.shape)}"
            )
            raise ValueError(msg)
        if stacked.shape[0] < 3:
            return
        from koopman_graph.analysis.resdmd import resdmd

        report = resdmd(
            stacked[:-1].detach(),
            stacked[1:].detach(),
            tolerance=self.tolerance,
        )
        residuals = report.residuals.real
        self.record(
            float(residuals.mean().item()),
            residual_max=float(residuals.max().item()),
        )
