#!/usr/bin/env python3
"""Build or refresh the small cylinder-wake teaching cache."""

from __future__ import annotations

from koopman_graph.datasets.cache_cli import (
    add_cache_dir_argument,
    add_force_argument,
    add_print_acquisition_argument,
    configure_acquisition_parser,
    maybe_print_acquisition,
)
from koopman_graph.datasets.cylinder_wake import (
    DEFAULT_WAKE_CACHE_DIR,
    ensure_wake_cache,
)

ACQUISITION_NOTES = f"""
Cylinder-wake teaching cache
----------------------------

This benchmark is generated locally (Hopf / Stuart–Landau surrogate on a coarse
wake mesh). There is no remote download; the script writes ``wake.pt`` under
the cache directory (default: {DEFAULT_WAKE_CACHE_DIR}).

Rebuild the teaching cache:

  python scripts/download_cylinder_wake.py
  python scripts/download_cylinder_wake.py --force
"""


def main() -> None:
    """CLI entry point for building the cylinder-wake cache.

    Returns
    -------
    None
    """
    parser = configure_acquisition_parser(
        "Build the cylinder-wake Hopf surrogate cache (wake.pt).",
        acquisition_notes=ACQUISITION_NOTES,
    )
    add_cache_dir_argument(parser, default=DEFAULT_WAKE_CACHE_DIR)
    add_force_argument(parser, help="Rebuild even if wake.pt already exists.")
    add_print_acquisition_argument(parser)
    args = parser.parse_args()

    if maybe_print_acquisition(args, ACQUISITION_NOTES):
        return

    path = ensure_wake_cache(args.cache_dir, force=args.force)
    print(f"Cached cylinder-wake benchmark at {path}")
    print("Source: local Hopf/Stuart-Landau surrogate (no remote download)")


if __name__ == "__main__":
    main()
