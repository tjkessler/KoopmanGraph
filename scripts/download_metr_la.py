#!/usr/bin/env python3
"""Download and cache METR-LA traffic benchmark data."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve

from koopman_graph.datasets.metr_la import (
    DCRNN_H5_GOOGLE_DRIVE,
    DEFAULT_CACHE_DIR,
    DEFAULT_H5_MIRROR_URL,
    DEFAULT_NUM_TIMESTEPS,
    DEFAULT_TIMESTEP_OFFSET,
    ensure_traffic_cache,
)

ACQUISITION_NOTES = f"""
METR-LA dataset acquisition
-----------------------------

Speed readings (metr-la.h5)
  • Original DCRNN release (Google Drive):
    {DCRNN_H5_GOOGLE_DRIVE}
  • Public mirror used by this script (--fetch-h5):
    {DEFAULT_H5_MIRROR_URL}

Graph metadata (sensor IDs, road distances)
  • Bundled automatically from the DCRNN GitHub repository when building cache.

After downloading metr-la.h5 locally, rebuild the tutorial cache with:

  python scripts/download_metr_la.py --h5-path /path/to/metr-la.h5 --force

Defaults cache one weekday (288 five-minute samples) starting at HDF5 row
offset {DEFAULT_TIMESTEP_OFFSET} (a high-contrast congestion day used by the
METR-LA tutorial notebook). Override with --num-timesteps / --offset.
"""


def _download_h5(destination: Path, *, url: str) -> Path:
    """Download an HDF5 file to ``destination``.

    Parameters
    ----------
    destination : Path
        Local path where the downloaded file is written.
    url : str
        Remote URL for the HDF5 archive.

    Returns
    -------
    Path
        ``destination`` after the download completes.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, destination)
    return destination


def main() -> None:
    """Parse CLI arguments and build the METR-LA traffic cache.

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(
        description="Build cached METR-LA topology and speed subset for tutorials.",
        epilog=ACQUISITION_NOTES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"Output directory (default: {DEFAULT_CACHE_DIR})",
    )
    parser.add_argument(
        "--h5-path",
        type=Path,
        default=None,
        help="Local metr-la.h5 file used to extract speed readings.",
    )
    parser.add_argument(
        "--fetch-h5",
        type=Path,
        default=None,
        metavar="PATH",
        help=("Download metr-la.h5 from the public mirror to PATH, then build cache."),
    )
    parser.add_argument(
        "--h5-url",
        type=str,
        default=DEFAULT_H5_MIRROR_URL,
        help="HDF5 download URL when using --fetch-h5 (default: mirror URL).",
    )
    parser.add_argument(
        "--num-timesteps",
        type=int,
        default=DEFAULT_NUM_TIMESTEPS,
        help=(
            "Number of 5-minute speed readings to cache "
            f"(default: {DEFAULT_NUM_TIMESTEPS})."
        ),
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=DEFAULT_TIMESTEP_OFFSET,
        help=(
            "Starting row offset in the HDF5 speed table "
            f"(default: {DEFAULT_TIMESTEP_OFFSET})."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the cache even when traffic.pt already exists.",
    )
    parser.add_argument(
        "--print-acquisition",
        action="store_true",
        help="Print dataset acquisition notes and exit.",
    )
    args = parser.parse_args()

    if args.print_acquisition:
        print(ACQUISITION_NOTES.strip())
        return

    h5_path = args.h5_path
    if args.fetch_h5 is not None:
        print(f"Downloading METR-LA HDF5 from {args.h5_url} ...")
        h5_path = _download_h5(args.fetch_h5, url=args.h5_url)
        print(f"Saved HDF5 to {h5_path}")

    path = ensure_traffic_cache(
        args.cache_dir,
        force=args.force,
        h5_path=h5_path,
        num_timesteps=args.num_timesteps,
        offset=args.offset,
    )
    print(f"Cached METR-LA traffic benchmark at {path}")
    if h5_path is not None:
        print(f"Speed source: {h5_path}")
    print("Graph metadata source: DCRNN sensor_graph files")


if __name__ == "__main__":
    main()
