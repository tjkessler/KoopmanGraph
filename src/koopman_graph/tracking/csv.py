"""CSV adapter for :class:`~koopman_graph.training.FitCallback`."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TextIO

from koopman_graph.training.history import FitHistory, TrainingLossBreakdown

# Keys from :meth:`TrainingLossBreakdown.to_floats` in stable column order.
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

CSV_FIT_FIELDNAMES: tuple[str, ...] = (
    "epoch",
    *(f"train_{key}" for key in _LOSS_TERM_KEYS),
    *(f"val_{key}" for key in _LOSS_TERM_KEYS),
)


def _breakdown_row(
    prefix: str,
    breakdown: TrainingLossBreakdown | None,
) -> dict[str, str]:
    """Map a loss breakdown to CSV cells under ``prefix_`` keys.

    Parameters
    ----------
    prefix : str
        Column prefix (``"train"`` or ``"val"``).
    breakdown : TrainingLossBreakdown or None
        Epoch loss terms; ``None`` yields empty string cells.

    Returns
    -------
    dict of str to str
        Mapping from ``{prefix}_{term}`` to stringified floats (or ``""``).
    """
    if breakdown is None:
        return {f"{prefix}_{key}": "" for key in _LOSS_TERM_KEYS}
    floats = breakdown.to_floats()
    return {f"{prefix}_{key}": f"{floats[key]!s}" for key in _LOSS_TERM_KEYS}


class CsvFitLogger:
    """Write per-epoch train/val loss terms to a CSV file.

    Implements :class:`~koopman_graph.training.FitCallback`. Opens the file on
    ``on_fit_start``, appends one row per ``on_epoch_end``, and closes on
    ``on_fit_end``. Observe-only: does not mutate the model.

    Parameters
    ----------
    path : str or Path
        Destination CSV path. Parent directories are created when missing.
    overwrite : bool, optional
        When ``True`` (default), truncate an existing file. When ``False``,
        append without rewriting the header (caller must ensure the file
        already has a compatible header).

    Notes
    -----
    Column names are :data:`CSV_FIT_FIELDNAMES`. Validation columns are left
    empty when ``val_breakdown`` is ``None``.
    """

    def __init__(self, path: str | Path, *, overwrite: bool = True) -> None:
        """Store destination path and overwrite policy.

        Parameters
        ----------
        path : str or Path
            Destination CSV path.
        overwrite : bool, optional
            Truncate on start when ``True`` (default).
        """
        self.path = Path(path)
        self.overwrite = overwrite
        self._file: TextIO | None = None
        self._writer: csv.DictWriter[str] | None = None

    def on_fit_start(self, *, model: Any, fit_kwargs: Mapping[str, Any]) -> None:
        """Open the CSV and write the header when starting a new file.

        Parameters
        ----------
        model : Any
            Unused model reference (observe-only).
        fit_kwargs : Mapping[str, Any]
            Unused fit keyword arguments.
        """
        del model, fit_kwargs
        self.path.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if self.overwrite else "a"
        self._file = self.path.open(mode, newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=list(CSV_FIT_FIELDNAMES),
            extrasaction="ignore",
        )
        if self.overwrite or self.path.stat().st_size == 0:
            self._writer.writeheader()
            self._file.flush()

    def on_epoch_end(
        self,
        *,
        epoch: int,
        train_breakdown: TrainingLossBreakdown,
        val_breakdown: TrainingLossBreakdown | None,
        history_so_far: FitHistory | None,
    ) -> None:
        """Append one epoch row of train/val loss terms.

        Parameters
        ----------
        epoch : int
            Zero-based epoch index.
        train_breakdown : TrainingLossBreakdown
            Mean training loss terms for the epoch.
        val_breakdown : TrainingLossBreakdown or None
            Mean validation loss terms when validation is enabled.
        history_so_far : FitHistory or None
            Unused partial history.
        """
        del history_so_far
        if self._writer is None or self._file is None:
            msg = "CsvFitLogger.on_epoch_end called before on_fit_start"
            raise RuntimeError(msg)
        row: dict[str, str] = {"epoch": str(epoch)}
        row.update(_breakdown_row("train", train_breakdown))
        row.update(_breakdown_row("val", val_breakdown))
        self._writer.writerow(row)
        self._file.flush()

    def on_fit_end(self, *, history: FitHistory) -> None:
        """Flush and close the CSV file.

        Parameters
        ----------
        history : FitHistory
            Unused final fit history.
        """
        del history
        if self._file is not None:
            self._file.flush()
            self._file.close()
        self._file = None
        self._writer = None
