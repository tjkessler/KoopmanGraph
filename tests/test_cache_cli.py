"""Unit tests for shared teaching-cache CLI helpers."""

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


def test_configure_acquisition_parser_common_flags(tmp_path: Path) -> None:
    """Shared helpers register a consistent acquisition CLI surface."""
    notes = "Acquisition notes for tests."
    parser = configure_acquisition_parser(
        "Test cache builder.",
        acquisition_notes=notes,
    )
    add_cache_dir_argument(parser, default=tmp_path / "cache")
    add_force_argument(parser)
    add_fetch_argument(parser, help="Fetch remote assets.")
    add_expected_sha256_argument(parser)
    add_print_acquisition_argument(parser)

    args = parser.parse_args(
        ["--fetch", "--force", "--expected-sha256", "abc", "--print-acquisition"]
    )
    assert args.cache_dir == tmp_path / "cache"
    assert args.fetch is True
    assert args.force is True
    assert args.expected_sha256 == "abc"
    assert maybe_print_acquisition(args, notes) is True


def test_add_cache_dir_argument_none_default() -> None:
    """``default=None`` uses the provided help suffix."""
    parser = configure_acquisition_parser("x", acquisition_notes="")
    add_cache_dir_argument(
        parser,
        default=None,
        help_suffix="Output cache directory (defaults under data/).",
    )
    args = parser.parse_args([])
    assert args.cache_dir is None
