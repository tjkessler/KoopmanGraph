#!/usr/bin/env python3
"""Examples-only Ray Tune HPO pattern for KoopmanGraph.

Search space stays in this script — KoopmanGraph does **not** ship a Tune
search-space DSL or AutoML API. Requires ``pip install "koopman-graph[ray]"``.

Smoke run (two learning-rate trials on a tiny decay trajectory)::

    python examples/scripts/ray_tune_koopman_example.py --epochs 1 --num-samples 2

Use native DDP / Fabric for multi-GPU *model* training; use Ray for trial /
ensemble-member parallelism. This script is intentionally outside ``nbmake`` CI.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Data

from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
from koopman_graph.data import GraphSnapshotSequence

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
    """Tune trainable: fit one model and report the final training loss."""
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
    final_loss = float(history.loss[-1]) if history.loss else float("nan")
    tune.report({"loss": final_loss, "epochs": int(history.epochs)})


def parse_args() -> argparse.Namespace:
    """Parse CLI flags for the Ray Tune example script."""
    parser = argparse.ArgumentParser(
        description=(
            "Examples-only Ray Tune HPO for a tiny GraphKoopmanModel. "
            "Search space lives in this script (not in the library)."
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

    try:
        import ray
        from ray import tune
        from ray.train import RunConfig
        from ray.tune import Tuner
    except ImportError as exc:
        raise SystemExit(
            f"Ray Tune is required for this example; install with: {_RAY_INSTALL_HINT}"
        ) from exc

    if not ray.is_initialized():
        ray.init(num_cpus=min(2, args.num_samples), ignore_reinit_error=True)

    # Search space is owned by this script (not by koopman_graph).
    param_space = {
        "lr": tune.loguniform(1e-3, 1e-1),
        "epochs": args.epochs,
        "seed": args.seed,
    }
    tune_config = tune.TuneConfig(
        num_samples=args.num_samples,
        metric="loss",
        mode="min",
    )
    run_config: RunConfig | None = None
    if args.storage_path is not None:
        run_config = RunConfig(
            name="koopman_graph_tune_example",
            storage_path=str(args.storage_path.resolve()),
        )

    tuner = Tuner(
        _train_koopman,
        param_space=param_space,
        tune_config=tune_config,
        run_config=run_config,
    )
    results = tuner.fit()
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
