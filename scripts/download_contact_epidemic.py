#!/usr/bin/env python3
"""Download and cache the SocioPatterns primary-school contact benchmark."""

from __future__ import annotations

from pathlib import Path

from koopman_graph.datasets.cache_cli import (
    add_cache_dir_argument,
    add_fetch_argument,
    add_force_argument,
    add_print_acquisition_argument,
    configure_acquisition_parser,
    maybe_print_acquisition,
)
from koopman_graph.datasets.contact_epidemic import (
    CONTACTS_SHA256,
    CONTACTS_URL,
    DATASET_PAGE_URL,
    DEFAULT_BIN_SECONDS,
    DEFAULT_CACHE_DIR,
    DEFAULT_NUM_BINS,
    METADATA_SHA256,
    METADATA_URL,
    ensure_contact_cache,
)

ACQUISITION_NOTES = f"""
SocioPatterns primary-school contact network
--------------------------------------------

Page: {DATASET_PAGE_URL}

Assets (CC-BY-NC-SA — do not redistribute in this repository):
  • Contacts (gzipped TSV): {CONTACTS_URL}
    SHA256: {CONTACTS_SHA256}
  • Metadata (id, class, gender): {METADATA_URL}
    SHA256: {METADATA_SHA256}

Cite Gemmetto et al., BMC Infectious Diseases 14:695 (2014) and Stehlé et al.,
PLoS ONE 6(8):e23176 (2011). Acknowledge the SocioPatterns collaboration
(https://www.sociopatterns.org). Non-commercial / ShareAlike obligations remain
with the user.

Build the teaching cache (24 × 1-hour contact-intensity bins by default):

  python scripts/download_contact_epidemic.py --fetch
  python scripts/download_contact_epidemic.py \\
      --contacts-path primaryschool.csv.gz \\
      --metadata-path primaryschool_metadata.txt
"""


def main() -> None:
    """Parse CLI arguments and build the contact-epidemic teaching cache.

    Returns
    -------
    None
    """
    parser = configure_acquisition_parser(
        "Build cached SocioPatterns primary-school contact benchmark.",
        acquisition_notes=ACQUISITION_NOTES,
    )
    add_cache_dir_argument(parser, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--contacts-path",
        type=Path,
        default=None,
        help="Local primaryschool.csv.gz (or plain TSV).",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=None,
        help="Local primaryschool_metadata.txt.",
    )
    add_fetch_argument(
        parser,
        help="Download SocioPatterns assets (verified SHA256) then build cache.",
    )
    parser.add_argument(
        "--expected-contacts-sha256",
        type=str,
        default=None,
        help="Optional override digest for a local contacts file.",
    )
    parser.add_argument(
        "--bin-seconds",
        type=int,
        default=DEFAULT_BIN_SECONDS,
        help=f"Bin width in seconds (default: {DEFAULT_BIN_SECONDS}).",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=DEFAULT_NUM_BINS,
        help=f"Number of teaching bins (default: {DEFAULT_NUM_BINS}).",
    )
    add_force_argument(parser, help="Rebuild even when contact.pt exists.")
    add_print_acquisition_argument(parser)
    args = parser.parse_args()

    if maybe_print_acquisition(args, ACQUISITION_NOTES):
        return

    path = ensure_contact_cache(
        args.cache_dir,
        force=args.force,
        contacts_path=args.contacts_path,
        metadata_path=args.metadata_path,
        bin_seconds=args.bin_seconds,
        num_bins=args.num_bins,
        expected_contacts_sha256=args.expected_contacts_sha256,
        fetch=args.fetch,
    )
    print(f"Cached contact-epidemic benchmark at {path}")
    print("License: CC-BY-NC-SA (SocioPatterns); do not redistribute raw assets.")


if __name__ == "__main__":
    main()
