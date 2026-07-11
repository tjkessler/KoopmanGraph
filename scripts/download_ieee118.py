#!/usr/bin/env python3
"""Download and cache IEEE 118-bus topology from MATPOWER."""

from __future__ import annotations

import argparse
from pathlib import Path

from koopman_graph.datasets.ieee118 import (
    DEFAULT_CACHE_DIR,
    MATPOWER_CASE118_URL,
    ensure_topology_cache,
)


def main() -> None:
    """Parse CLI arguments and build the topology cache.

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(
        description="Download MATPOWER case118 and cache parsed topology tensors.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"Output directory (default: {DEFAULT_CACHE_DIR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and overwrite an existing cache file.",
    )
    args = parser.parse_args()

    path = ensure_topology_cache(args.cache_dir, force=args.force)
    print(f"Cached IEEE 118 topology at {path}")
    print(f"Source: {MATPOWER_CASE118_URL}")


if __name__ == "__main__":
    main()
