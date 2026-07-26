#!/usr/bin/env python3
"""Download and cache PEMS-BAY / PEMS03-08 traffic benchmark data."""

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
from koopman_graph.datasets.pems import (
    BAY_DISTANCES_URL,
    BAY_SENSOR_LOCATIONS_URL,
    DCRNN_BAY_H5_GOOGLE_DRIVE,
    DEFAULT_BAY_CACHE_DIR,
    DEFAULT_BAY_H5_FILENAME,
    DEFAULT_BAY_H5_MIRROR_URL,
    DEFAULT_BAY_H5_SHA256,
    DEFAULT_BAY_TIMESTEP_OFFSET,
    DEFAULT_NUM_TIMESTEPS,
    VALID_PEMS_VARIANTS,
    default_variant_cache_dir,
    ensure_bay_traffic_cache,
    ensure_variant_traffic_cache,
    normalize_pems_variant,
)

ACQUISITION_NOTES = f"""
PEMS traffic dataset acquisition
--------------------------------

PEMS-BAY (speeds, 325 sensors)
  • Original DCRNN release (Google Drive bundle with METR-LA):
    {DCRNN_BAY_H5_GOOGLE_DRIVE}
  • Public mirror used by --fetch:
    {DEFAULT_BAY_H5_MIRROR_URL}
    SHA256: {DEFAULT_BAY_H5_SHA256}
  • Graph metadata (auto-fetched with SHA256 verification):
    {BAY_SENSOR_LOCATIONS_URL}
    {BAY_DISTANCES_URL}

PEMS03 / 04 / 07 / 08 (flows)
  • Upstream: Caltrans Performance Measurement System (PeMS)
  • Community NPZ + adjacency CSV packaging (ASTGCN / STFGNN-style).
    Obtain ``PEMSXX.npz`` and the matching adjacency CSV from a trusted
    research archive, then pass --npz-path / --adj-csv. Always supply
    --expected-sha256 for the NPZ when downloading from a mirror.

Rebuild teaching caches (one day of 5-minute samples by default):

  python scripts/download_pems.py --benchmark bay --fetch
  python scripts/download_pems.py --benchmark bay --h5-path /path/to/pems-bay.h5
  python scripts/download_pems.py --benchmark 04 \\
      --npz-path /path/to/PEMS04.npz --adj-csv /path/to/PEMS04.csv \\
      --expected-sha256 <hex>

--fetch for --benchmark bay always verifies SHA256 (pinned digest for the
default mirror; supply --expected-sha256 when overriding --h5-url).
"""


def main() -> None:
    """Parse CLI arguments and build a PEMS teaching cache.

    Returns
    -------
    None
    """
    parser = configure_acquisition_parser(
        "Build cached PEMS-BAY / PEMS0X topology and series subsets.",
        acquisition_notes=ACQUISITION_NOTES,
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        help="bay | 03 | 04 | 07 | 08 (required unless --print-acquisition)",
    )
    add_cache_dir_argument(
        parser,
        default=None,
        help_suffix="Output cache directory (defaults under data/pems*).",
    )
    parser.add_argument(
        "--h5-path",
        type=Path,
        default=None,
        help="Local pems-bay.h5 for --benchmark bay.",
    )
    add_fetch_argument(
        parser,
        help=(
            f"For --benchmark bay: download pems-bay.h5 from --h5-url into "
            f"--cache-dir/{DEFAULT_BAY_H5_FILENAME}, then build cache."
        ),
    )
    parser.add_argument(
        "--h5-url",
        type=str,
        default=DEFAULT_BAY_H5_MIRROR_URL,
        help="HDF5 URL when using --fetch with --benchmark bay.",
    )
    parser.add_argument(
        "--npz-path",
        type=Path,
        default=None,
        help="Local PEMSXX.npz for PEMS0X variants.",
    )
    parser.add_argument(
        "--adj-csv",
        type=Path,
        default=None,
        help="Local dense adjacency CSV for PEMS0X variants.",
    )
    add_expected_sha256_argument(
        parser,
        help=(
            "SHA256 hex digest for the primary raw file (H5 or NPZ). Required "
            "with --fetch when --h5-url is not the default BAY mirror; "
            "recommended for PEMS0X NPZ inputs."
        ),
    )
    parser.add_argument(
        "--num-timesteps",
        type=int,
        default=DEFAULT_NUM_TIMESTEPS,
        help=f"Cached timesteps (default: {DEFAULT_NUM_TIMESTEPS}).",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=None,
        help=(
            "Starting row offset in the raw series (default: 0; BAY default "
            f"{DEFAULT_BAY_TIMESTEP_OFFSET})."
        ),
    )
    add_force_argument(parser)
    add_print_acquisition_argument(parser)
    args = parser.parse_args()

    if maybe_print_acquisition(args, ACQUISITION_NOTES):
        return

    if args.benchmark is None:
        raise SystemExit("--benchmark is required unless --print-acquisition")

    benchmark = args.benchmark.strip().lower()
    if benchmark in {"bay", "pems-bay", "pems_bay"}:
        cache_dir = args.cache_dir or DEFAULT_BAY_CACHE_DIR
        h5_path = args.h5_path
        if args.fetch:
            try:
                expected_sha256 = resolve_fetch_sha256(
                    expected_sha256=args.expected_sha256,
                    url=args.h5_url,
                    default_url=DEFAULT_BAY_H5_MIRROR_URL,
                    default_sha256=DEFAULT_BAY_H5_SHA256,
                    label="PEMS-BAY HDF5",
                )
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
            destination = h5_path or (cache_dir / DEFAULT_BAY_H5_FILENAME)
            print(f"Downloading PEMS-BAY HDF5 from {args.h5_url} ...")
            h5_path = download_url_to_path(
                args.h5_url,
                destination,
                label="PEMS-BAY HDF5",
                expected_sha256=expected_sha256,
            )
            print(f"Saved HDF5 to {h5_path}")
        offset = (
            DEFAULT_BAY_TIMESTEP_OFFSET if args.offset is None else int(args.offset)
        )
        # When --fetch already verified the digest during download, skip a
        # second pass inside ensure_bay_traffic_cache.
        expected_h5 = None
        if h5_path is not None and not args.fetch:
            expected_h5 = args.expected_sha256
        path = ensure_bay_traffic_cache(
            cache_dir,
            force=args.force,
            h5_path=h5_path,
            num_timesteps=args.num_timesteps,
            offset=offset,
            expected_h5_sha256=expected_h5,
        )
        print(f"Cached PEMS-BAY traffic benchmark at {path}")
        if h5_path is not None:
            print(f"Speed source: {h5_path}")
        print("Graph metadata source: DCRNN sensor_graph files")
        return

    if args.fetch:
        msg = (
            "--fetch applies only to --benchmark bay; "
            "PEMS0X variants require --npz-path and --adj-csv."
        )
        raise SystemExit(msg)

    variant = normalize_pems_variant(benchmark)
    if variant not in VALID_PEMS_VARIANTS:
        msg = f"Unknown --benchmark {args.benchmark!r}"
        raise SystemExit(msg)
    cache_dir = args.cache_dir or default_variant_cache_dir(variant)
    offset = 0 if args.offset is None else int(args.offset)
    path = ensure_variant_traffic_cache(
        variant,
        cache_dir,
        force=args.force,
        npz_path=args.npz_path,
        adj_csv_path=args.adj_csv,
        num_timesteps=args.num_timesteps,
        offset=offset,
        expected_npz_sha256=args.expected_sha256,
        source_url=str(args.npz_path) if args.npz_path is not None else None,
    )
    print(f"Cached PEMS{variant} traffic benchmark at {path}")


if __name__ == "__main__":
    main()
