#!/usr/bin/env python3
"""Examples-only Ray Tune HPO pattern for KoopmanGraph.

This script owns the search configuration. It may call *example-only*
library scaffolds such as ``example_lr_loguniform_space``; those are smoke
ranges, not scientific defaults. KoopmanGraph is **not** an AutoML product.
Requires ``pip install "koopman-graph[ray]"``.

Smoke run (two learning-rate trials on a tiny decay trajectory)::

    python examples/scripts/ray_tune_koopman_example.py --epochs 1 --num-samples 2

Uses :func:`~koopman_graph.tuning.fit_history_metrics` and
:func:`~koopman_graph.tuning.run_ray_tune`. Prefer native DDP / Fabric for
multi-GPU *model* training; use Ray for trial / ensemble-member parallelism.
This script is intentionally outside ``nbmake`` CI.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Data

from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.tuning import (
    example_lr_loguniform_space,
    fit_history_metrics,
    run_ray_tune,
)

_RAY_INSTALL_HINT = 'pip install "koopman-graph[ray]"'


def _decay_sequence(*, num_timesteps: int = 5) -> GraphSnapshotSequence:
    """Build a deterministic decay trajectory ``x_{t+1} = 0.9 * x_t``."""
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 4],
            [1, 0, 2, 1, 3, 2, 4, 3],
        ],
        dtype=torch.long,
    )
    x0 = torch.ones(5, 3)
    snapshots = [
        Data(x=x0 * (0.9**t), edge_index=edge_index) for t in range(num_timesteps)
    ]
    return GraphSnapshotSequence(snapshots)


def _build_model() -> GraphKoopmanModel:
    """Construct a small identically structured trainable model."""
    return GraphKoopmanModel(
        encoder=GNNEncoder(in_channels=3, hidden_channels=16, latent_dim=8),
        decoder=GNNDecoder(latent_dim=8, hidden_channels=16, out_channels=3),
        latent_dim=8,
        time_step=0.1,
    )


def _train_koopman(config: dict[str, Any]) -> None:
    """Tune trainable: fit one model and report metrics from FitHistory."""
    from ray import tune

    torch.manual_seed(int(config["seed"]))
    sequence = _decay_sequence()
    model = _build_model()
    history = model.fit(
        sequence,
        epochs=int(config["epochs"]),
        lr=float(config["lr"]),
        device="cpu",
    )
    metrics = fit_history_metrics(history)
    # Ensure a finite loss key for Tune's metric="loss" even on empty history.
    if "loss" not in metrics:
        metrics["loss"] = float("nan")
    tune.report(metrics)


def parse_args() -> argparse.Namespace:
    """Parse CLI flags for the Ray Tune example script."""
    parser = argparse.ArgumentParser(
        description=(
            "Examples-only Ray Tune HPO for a tiny GraphKoopmanModel. "
            "Search configuration is owned by this script (not an AutoML API)."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Training epochs per trial (default: 2).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=2,
        help="Number of random lr samples (default: 2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed for model init inside each trial (default: 0).",
    )
    parser.add_argument(
        "--storage-path",
        type=Path,
        default=None,
        help="Optional Tune storage directory (default: Ray temp dir).",
    )
    return parser.parse_args()


def main() -> None:
    """Run a tiny Tune search over learning rate and print the best trial."""
    args = parse_args()
    if args.epochs < 1:
        raise SystemExit(f"--epochs must be >= 1; got {args.epochs}")
    if args.num_samples < 1:
        raise SystemExit(f"--num-samples must be >= 1; got {args.num_samples}")

    # Script-owned search config: example_* scaffold is smoke-only, not science.
    # Search space is owned by this script (not an AutoML API).
    try:
        param_space = example_lr_loguniform_space(
            epochs=args.epochs,
            seed=args.seed,
        )
        results = run_ray_tune(
            _train_koopman,
            param_space,
            num_samples=args.num_samples,
            metric="loss",
            mode="min",
            storage_path=args.storage_path,
            run_name="koopman_graph_tune_example",
            ray_init_kwargs={
                "num_cpus": min(2, args.num_samples),
                "ignore_reinit_error": True,
            },
        )
    except ImportError as exc:
        raise SystemExit(
            f"Ray Tune is required for this example; install with: {_RAY_INSTALL_HINT}"
        ) from exc

    best = results.get_best_result(metric="loss", mode="min")
    best_loss = best.metrics.get("loss") if best.metrics else None
    print(
        "ray_tune_koopman_example: "
        f"best_lr={best.config.get('lr')!r} "
        f"best_loss={best_loss!r} "
        f"num_trials={len(results)}"
    )
    if best_loss is None or not torch.isfinite(torch.tensor(float(best_loss))).item():
        raise SystemExit("Tune finished without a finite best loss")


if __name__ == "__main__":
    main()
