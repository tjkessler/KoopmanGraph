#!/usr/bin/env python3
"""Download and cache IEEE 118-bus topology from MATPOWER."""

from __future__ import annotations

from koopman_graph.datasets.cache_cli import (
    add_cache_dir_argument,
    add_force_argument,
    add_print_acquisition_argument,
    configure_acquisition_parser,
    maybe_print_acquisition,
)
from koopman_graph.datasets.ieee118 import (
    DEFAULT_CACHE_DIR,
    MATPOWER_CASE118_URL,
    ensure_topology_cache,
)

ACQUISITION_NOTES = f"""
IEEE 118-bus topology acquisition
---------------------------------

Source (MATPOWER case118.m):
  {MATPOWER_CASE118_URL}

The script downloads the case file, parses bus/branch topology, and writes
``topology.pt`` under the cache directory. No separate raw archive is kept.

Rebuild the teaching cache:

  python scripts/download_ieee118.py
  python scripts/download_ieee118.py --force
"""


def main() -> None:
    """Parse CLI arguments and build the topology cache.

    Returns
    -------
    None
    """
    parser = configure_acquisition_parser(
        "Download MATPOWER case118 and cache parsed topology tensors.",
        acquisition_notes=ACQUISITION_NOTES,
    )
    add_cache_dir_argument(parser, default=DEFAULT_CACHE_DIR)
    add_force_argument(
        parser,
        help="Re-download and overwrite an existing cache file.",
    )
    add_print_acquisition_argument(parser)
    args = parser.parse_args()

    if maybe_print_acquisition(args, ACQUISITION_NOTES):
        return

    path = ensure_topology_cache(args.cache_dir, force=args.force)
    print(f"Cached IEEE 118 topology at {path}")
    print(f"Source: {MATPOWER_CASE118_URL}")


if __name__ == "__main__":
    main()
