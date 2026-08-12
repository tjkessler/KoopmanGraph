"""``koopman-graph predict`` — load checkpoint, roll out, write forecasts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Data

from koopman_graph.cli.config import ConfigError, load_config
from koopman_graph.cli.train import (
    build_sequence_from_config,
    load_cached_sequence,
)
from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.serialization import load_checkpoint


def load_sequence_from_path(path: str | Path) -> GraphSnapshotSequence:
    """Load a sequence from a ``.pt`` cache or a JSON/YAML data config.

    Parameters
    ----------
    path : str or Path
        Either a pickle ``GraphSnapshotSequence`` / list of ``Data``, or a
        JSON/YAML file containing a ``data`` section (or a bare data mapping
        with ``kind``).

    Returns
    -------
    GraphSnapshotSequence
        Loaded or synthesized trajectory.
    """
    destination = Path(path)
    if not destination.is_file():
        raise ConfigError(f"Data path not found: {destination}")

    suffix = destination.suffix.lower()
    if suffix in {".json", ".yaml", ".yml"}:
        payload = load_config(destination)
        if "data" in payload:
            data_cfg = payload["data"]
            if not isinstance(data_cfg, dict):
                raise ConfigError("config.data must be a mapping")
        elif "kind" in payload:
            data_cfg = payload
        else:
            raise ConfigError(
                f"{destination} must contain a 'data' section or a top-level "
                "'kind' field"
            )
        return build_sequence_from_config(data_cfg)

    if suffix in {".pt", ".pth"}:
        return load_cached_sequence(
            {"kind": "cached_sequence", "path": str(destination)}
        )

    msg = (
        f"Unsupported data path suffix {suffix!r} for {destination}; "
        "use .json/.yaml/.yml or .pt/.pth"
    )
    raise ConfigError(msg)


def run_predict(
    *,
    checkpoint: str | Path,
    data: str | Path,
    steps: int,
    out: str | Path,
) -> tuple[Path, dict[str, Any] | None]:
    """Load a checkpoint, forecast ``steps``, and write a ``.pt`` payload.

    Parameters
    ----------
    checkpoint : str or Path
        Safe or legacy checkpoint path.
    data : str or Path
        Sequence source (see :func:`load_sequence_from_path`).
    steps : int
        Autoregressive forecast horizon (must be >= 1).
    out : str or Path
        Destination ``.pt`` file for the forecast payload.

    Returns
    -------
    path : Path
        Path written.
    summary : dict or None
        Optional shape summary for stdout.
    """
    if steps < 1:
        raise ConfigError(f"--steps must be >= 1, got {steps}")

    model = load_checkpoint(checkpoint)
    sequence = load_sequence_from_path(data)
    if len(sequence) < 1:
        raise ConfigError("Data sequence is empty")

    initial = sequence[0]
    forecasts = model.predict(initial, steps=steps)
    if not forecasts:
        raise ConfigError("predict returned an empty forecast list")

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    first = forecasts[0]
    summary: dict[str, Any] | None = None
    if isinstance(first, Data) and first.x is not None:
        summary = {
            "num_nodes": int(first.x.shape[0]),
            "feature_dim": int(first.x.shape[1]),
            "num_forecasts": len(forecasts),
        }
    payload: dict[str, Any] = {
        "steps": steps,
        "forecasts": forecasts,
        "summary": summary,
    }
    torch.save(payload, out_path)
    return out_path, summary


def handle_predict(args: argparse.Namespace) -> int:
    """Argparse handler for the ``predict`` subcommand.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI namespace with ``checkpoint``, ``data``, ``steps``, and
        ``out``.

    Returns
    -------
    int
        ``0`` on success; ``1`` on config / I/O / type errors.
    """
    try:
        out_path, summary = run_predict(
            checkpoint=args.checkpoint,
            data=args.data,
            steps=args.steps,
            out=args.out,
        )
        if summary is not None:
            print(
                "wrote forecast: "
                f"{out_path} (steps={args.steps}, "
                f"nodes={summary.get('num_nodes')}, "
                f"feature_dim={summary.get('feature_dim')})"
            )
        else:
            print(f"wrote forecast: {out_path} (steps={args.steps})")
    except (ConfigError, ValueError, FileNotFoundError, OSError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0
