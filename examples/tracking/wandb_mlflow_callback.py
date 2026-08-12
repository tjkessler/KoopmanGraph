#!/usr/bin/env python3
"""Reference FitCallback adapters for W&B and MLflow (peer installs).

These classes are teaching sketches — not library APIs. Cloud SDKs are
intentionally not pinned by KoopmanGraph. Install peers yourself when
needed::

    pip install wandb
    pip install mlflow
    pip install tensorboard   # for TensorBoardFitLogger

In-tree loggers (no cloud SDKs)::

    from koopman_graph.tracking import CsvFitLogger, TensorBoardFitLogger

The ``__main__`` block below runs a tiny CSV-logged fit only (no W&B /
MLflow import), so the script stays usable without optional SDKs.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Data

from koopman_graph import FitCallback, GNNDecoder, GNNEncoder, GraphKoopmanModel
from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.tracking import CsvFitLogger
from koopman_graph.training.history import FitHistory, TrainingLossBreakdown


class WandbFitLogger:
    """Sketch :class:`~koopman_graph.FitCallback` that calls ``wandb.log``.

    Requires a peer ``wandb`` install. Not part of ``koopman_graph.tracking``.
    """

    def __init__(self, *, project: str = "koopman-graph", **init_kwargs: Any) -> None:
        self.project = project
        self.init_kwargs = init_kwargs
        self._wandb: Any | None = None

    def on_fit_start(self, *, model: Any, fit_kwargs: Mapping[str, Any]) -> None:
        del model
        import wandb

        self._wandb = wandb
        self._wandb.init(
            project=self.project,
            config=dict(fit_kwargs),
            **self.init_kwargs,
        )

    def on_epoch_end(
        self,
        *,
        epoch: int,
        train_breakdown: TrainingLossBreakdown,
        val_breakdown: TrainingLossBreakdown | None,
        history_so_far: FitHistory | None,
    ) -> None:
        del history_so_far
        if self._wandb is None:
            msg = "WandbFitLogger.on_epoch_end called before on_fit_start"
            raise RuntimeError(msg)
        payload = {
            f"train/{key}": value for key, value in train_breakdown.to_floats().items()
        }
        if val_breakdown is not None:
            val_floats = val_breakdown.to_floats()
            payload.update({f"val/{key}": value for key, value in val_floats.items()})
        self._wandb.log(payload, step=epoch)

    def on_fit_end(self, *, history: FitHistory) -> None:
        del history
        if self._wandb is not None:
            self._wandb.finish()
        self._wandb = None


class MLflowFitLogger:
    """Sketch :class:`~koopman_graph.FitCallback` that calls ``mlflow.log_metrics``.

    Requires a peer ``mlflow`` install. Not part of ``koopman_graph.tracking``.
    """

    def __init__(self, *, run_name: str | None = None) -> None:
        self.run_name = run_name
        self._mlflow: Any | None = None

    def on_fit_start(self, *, model: Any, fit_kwargs: Mapping[str, Any]) -> None:
        del model
        import mlflow

        self._mlflow = mlflow
        self._mlflow.start_run(run_name=self.run_name)
        for key, value in fit_kwargs.items():
            if isinstance(value, (bool, int, float, str)):
                self._mlflow.log_param(key, value)

    def on_epoch_end(
        self,
        *,
        epoch: int,
        train_breakdown: TrainingLossBreakdown,
        val_breakdown: TrainingLossBreakdown | None,
        history_so_far: FitHistory | None,
    ) -> None:
        del history_so_far
        if self._mlflow is None:
            msg = "MLflowFitLogger.on_epoch_end called before on_fit_start"
            raise RuntimeError(msg)
        metrics = {
            f"train_{key}": value for key, value in train_breakdown.to_floats().items()
        }
        if val_breakdown is not None:
            metrics.update(
                {
                    f"val_{key}": value
                    for key, value in val_breakdown.to_floats().items()
                }
            )
        self._mlflow.log_metrics(metrics, step=epoch)

    def on_fit_end(self, *, history: FitHistory) -> None:
        del history
        if self._mlflow is not None:
            self._mlflow.end_run()
        self._mlflow = None


def _assert_fit_callback(callback: FitCallback) -> None:
    """Runtime-check that a sketch satisfies the FitCallback protocol."""
    assert isinstance(callback, FitCallback)


def _tiny_sequence() -> GraphSnapshotSequence:
    """Deterministic five-step path graph for the CSV smoke path."""
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]],
        dtype=torch.long,
    )
    x0 = torch.ones(5, 3)
    snapshots = [Data(x=x0 * (0.9**t), edge_index=edge_index) for t in range(5)]
    return GraphSnapshotSequence(snapshots)


def main() -> None:
    """Run a short CSV-logged fit (no W&B / MLflow required)."""
    out_dir = Path("artifacts") / "tracking_demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "fit.csv"
    logger = CsvFitLogger(csv_path)
    _assert_fit_callback(logger)
    _assert_fit_callback(WandbFitLogger())
    _assert_fit_callback(MLflowFitLogger())

    model = GraphKoopmanModel(
        encoder=GNNEncoder(
            in_channels=3, hidden_channels=8, latent_dim=4, num_layers=2
        ),
        decoder=GNNDecoder(
            latent_dim=4, hidden_channels=8, out_channels=3, num_layers=2
        ),
        latent_dim=4,
        time_step=0.1,
    )
    torch.manual_seed(0)
    history = model.fit(
        _tiny_sequence(),
        epochs=2,
        lr=1e-2,
        device="cpu",
        callbacks=[logger],
    )
    print(f"wrote {csv_path} ({history.epochs} epochs)")


if __name__ == "__main__":
    main()
