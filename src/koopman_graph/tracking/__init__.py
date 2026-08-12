"""Experiment-tracking adapters for fit callbacks.

Capability layout
-----------------
``csv``
    :class:`~koopman_graph.tracking.CsvFitLogger` — stdlib CSV writer
    implementing :class:`~koopman_graph.training.FitCallback`.
``tensorboard``
    :class:`~koopman_graph.tracking.TensorBoardFitLogger` — optional
    ``SummaryWriter`` adapter (requires peer ``tensorboard`` install).

Power-user module: import as ``koopman_graph.tracking``. Types are
intentionally omitted from root ``koopman_graph.__all__`` (callbacks are
root-exported; adapters stay here). This package may import
:mod:`koopman_graph.training` types; ``training`` must not import tracking
adapters.
"""

from koopman_graph.tracking.csv import CSV_FIT_FIELDNAMES, CsvFitLogger
from koopman_graph.tracking.tensorboard import TensorBoardFitLogger

__all__ = [
    "CSV_FIT_FIELDNAMES",
    "CsvFitLogger",
    "TensorBoardFitLogger",
]
