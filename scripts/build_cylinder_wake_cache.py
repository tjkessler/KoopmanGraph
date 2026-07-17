#!/usr/bin/env python3
"""Build or refresh the small cylinder-wake teaching cache."""

from __future__ import annotations

import argparse
from pathlib import Path

from koopman_graph.datasets.nonlinear import ensure_wake_cache


def main() -> None:
    """CLI entry point for building the cylinder-wake cache.

    Notes
    -----
    Writes ``wake.pt`` under ``data/cylinder_wake`` (or ``--cache-dir``).
    """
    parser = argparse.ArgumentParser(
        description="Build the cylinder-wake Hopf surrogate cache (wake.pt)."
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Cache directory (default: data/cylinder_wake).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if wake.pt already exists.",
    )
    args = parser.parse_args()
    path = ensure_wake_cache(args.cache_dir, force=args.force)
    print(f"Wrote cylinder-wake cache to {path}")


if __name__ == "__main__":
    main()
