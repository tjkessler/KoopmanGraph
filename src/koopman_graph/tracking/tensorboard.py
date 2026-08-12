"""TensorBoard adapter for :class:`~koopman_graph.training.FitCallback`."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from koopman_graph.training.history import FitHistory, TrainingLossBreakdown

# Match :mod:`koopman_graph.tracking.csv` term order for comparable dashboards.
_LOSS_TERM_KEYS: tuple[str, ...] = (
    "total",
    "reconstruction",
    "forward",
    "backward",
    "rollout",
    "eigenvalue",
    "lie",
    "pde",
    "sparsity",
    "worst_case",
    "vamp2",
)


def _require_summary_writer() -> Any:
    """Import ``SummaryWriter`` or raise with an install hint.

    Returns
    -------
    type
        ``torch.utils.tensorboard.SummaryWriter`` class.

    Raises
    ------
    ImportError
        If the ``tensorboard`` peer package is not installed.
    """
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        msg = (
            "TensorBoardFitLogger requires the tensorboard package "
            "(peer install for torch.utils.tensorboard). "
            "Install with: pip install tensorboard"
        )
        raise ImportError(msg) from exc
    return SummaryWriter


def _log_breakdown(
    writer: Any,
    *,
    prefix: str,
    breakdown: TrainingLossBreakdown,
    epoch: int,
) -> None:
    """Write scalar tags ``{prefix}/{term}`` for one breakdown.

    Parameters
    ----------
    writer : SummaryWriter
        Open TensorBoard writer.
    prefix : str
        Tag prefix (``"train"`` or ``"val"``).
    breakdown : TrainingLossBreakdown
        Epoch loss terms to log.
    epoch : int
        Global step / epoch index.
    """
    floats = breakdown.to_floats()
    for key in _LOSS_TERM_KEYS:
        writer.add_scalar(f"{prefix}/{key}", floats[key], global_step=epoch)


class TensorBoardFitLogger:
    """Log per-epoch train/val loss terms with TensorBoard ``SummaryWriter``.

    Implements :class:`~koopman_graph.training.FitCallback`. Requires the
    peer ``tensorboard`` package (not pinned in this library; Q10). Observe-only.

    Parameters
    ----------
    log_dir : str or Path
        Directory for TensorBoard event files (created if missing).
    **writer_kwargs
        Forwarded to ``torch.utils.tensorboard.SummaryWriter`` (for example
        ``comment`` or ``max_queue``).

    Raises
    ------
    ImportError
        If ``tensorboard`` / ``SummaryWriter`` is not importable. The message
        includes an install hint.
    """

    def __init__(self, log_dir: str | Path, **writer_kwargs: Any) -> None:
        """Store log directory and SummaryWriter kwargs.

        Parameters
        ----------
        log_dir : str or Path
            Directory for TensorBoard event files.
        **writer_kwargs
            Forwarded to ``SummaryWriter`` on ``on_fit_start``.
        """
        self.log_dir = Path(log_dir)
        self._writer_kwargs = writer_kwargs
        self._writer: Any | None = None

    def on_fit_start(self, *, model: Any, fit_kwargs: Mapping[str, Any]) -> None:
        """Create a ``SummaryWriter`` under ``log_dir``.

        Parameters
        ----------
        model : Any
            Unused model reference (observe-only).
        fit_kwargs : Mapping[str, Any]
            Unused fit keyword arguments.
        """
        del model, fit_kwargs
        summary_writer_cls = _require_summary_writer()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = summary_writer_cls(
            log_dir=str(self.log_dir),
            **self._writer_kwargs,
        )

    def on_epoch_end(
        self,
        *,
        epoch: int,
        train_breakdown: TrainingLossBreakdown,
        val_breakdown: TrainingLossBreakdown | None,
        history_so_far: FitHistory | None,
    ) -> None:
        """Log train (and optional val) scalars for the completed epoch.

        Parameters
        ----------
        epoch : int
            Zero-based epoch index used as ``global_step``.
        train_breakdown : TrainingLossBreakdown
            Mean training loss terms for the epoch.
        val_breakdown : TrainingLossBreakdown or None
            Mean validation loss terms when validation is enabled.
        history_so_far : FitHistory or None
            Unused partial history.
        """
        del history_so_far
        if self._writer is None:
            msg = "TensorBoardFitLogger.on_epoch_end called before on_fit_start"
            raise RuntimeError(msg)
        _log_breakdown(
            self._writer,
            prefix="train",
            breakdown=train_breakdown,
            epoch=epoch,
        )
        if val_breakdown is not None:
            _log_breakdown(
                self._writer,
                prefix="val",
                breakdown=val_breakdown,
                epoch=epoch,
            )
        self._writer.flush()

    def on_fit_end(self, *, history: FitHistory) -> None:
        """Flush and close the ``SummaryWriter``.

        Parameters
        ----------
        history : FitHistory
            Unused final fit history.
        """
        del history
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
        self._writer = None
