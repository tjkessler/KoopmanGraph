#!/usr/bin/env python3
"""Download and cache METR-LA traffic benchmark data."""

from __future__ import annotations

from pathlib import Path

from koopman_graph.datasets.cache_cli import (
    add_cache_dir_argument,
    add_expected_sha256_argument,
    add_fetch_argument,
    add_force_argument,
    add_print_acquisition_argument,
    configure_acquisition_parser,
    maybe_print_acquisition,
)
from koopman_graph.datasets.download import (
    download_url_to_path,
    resolve_fetch_sha256,
)
from koopman_graph.datasets.metr_la import (
    DCRNN_H5_GOOGLE_DRIVE,
    DEFAULT_CACHE_DIR,
    DEFAULT_H5_FILENAME,
    DEFAULT_H5_MIRROR_URL,
    DEFAULT_H5_SHA256,
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
  • Public mirror used by this script (--fetch):
    {DEFAULT_H5_MIRROR_URL}
    SHA256: {DEFAULT_H5_SHA256}

Graph metadata (sensor IDs, road distances)
  • Bundled automatically from the DCRNN GitHub repository when building cache.

Rebuild the tutorial cache:

  python scripts/download_metr_la.py --fetch
  python scripts/download_metr_la.py --h5-path /path/to/metr-la.h5 --force

--fetch always verifies SHA256 (pinned digest for the default mirror; supply
--expected-sha256 when overriding --h5-url).

Defaults cache one weekday (288 five-minute samples) starting at HDF5 row
offset {DEFAULT_TIMESTEP_OFFSET} (a high-contrast congestion day used by the
METR-LA tutorial notebook). Override with --num-timesteps / --offset.
"""


def main() -> None:
    """Parse CLI arguments and build the METR-LA traffic cache.

    Returns
    -------
    None
    """
    parser = configure_acquisition_parser(
        "Build cached METR-LA topology and speed subset for tutorials.",
        acquisition_notes=ACQUISITION_NOTES,
    )
    add_cache_dir_argument(parser, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--h5-path",
        type=Path,
        default=None,
        help="Local metr-la.h5 file used to extract speed readings.",
    )
    add_fetch_argument(
        parser,
        help=(
            f"Download metr-la.h5 from --h5-url into "
            f"--cache-dir/{DEFAULT_H5_FILENAME}, then build cache."
        ),
    )
    parser.add_argument(
        "--h5-url",
        type=str,
        default=DEFAULT_H5_MIRROR_URL,
        help="HDF5 download URL when using --fetch (default: mirror URL).",
    )
    add_expected_sha256_argument(
        parser,
        help=(
            "SHA256 hex digest for metr-la.h5. Required with --fetch when "
            "--h5-url is not the default mirror; otherwise defaults to the "
            f"pinned digest ({DEFAULT_H5_SHA256[:12]}…)."
        ),
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
    add_force_argument(parser)
    add_print_acquisition_argument(parser)
    args = parser.parse_args()

    if maybe_print_acquisition(args, ACQUISITION_NOTES):
        return

    cache_dir = args.cache_dir
    h5_path = args.h5_path
    if args.fetch:
        try:
            expected_sha256 = resolve_fetch_sha256(
                expected_sha256=args.expected_sha256,
                url=args.h5_url,
                default_url=DEFAULT_H5_MIRROR_URL,
                default_sha256=DEFAULT_H5_SHA256,
                label="METR-LA HDF5",
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        destination = h5_path or (cache_dir / DEFAULT_H5_FILENAME)
        print(f"Downloading METR-LA HDF5 from {args.h5_url} ...")
        h5_path = download_url_to_path(
            args.h5_url,
            destination,
            label="METR-LA HDF5",
            expected_sha256=expected_sha256,
        )
        print(f"Saved HDF5 to {h5_path}")

    # When --fetch already verified the digest during download, skip a second
    # pass inside ensure_traffic_cache.
    expected_h5 = None if args.fetch else args.expected_sha256
    path = ensure_traffic_cache(
        cache_dir,
        force=args.force,
        h5_path=h5_path,
        num_timesteps=args.num_timesteps,
        offset=args.offset,
        expected_h5_sha256=expected_h5,
    )
    print(f"Cached METR-LA traffic benchmark at {path}")
    if h5_path is not None:
        print(f"Speed source: {h5_path}")
    print("Graph metadata source: DCRNN sensor_graph files")


if __name__ == "__main__":
    main()
