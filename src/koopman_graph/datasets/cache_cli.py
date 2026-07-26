"""Shared argparse helpers for teaching-cache acquisition scripts.

Power-user module used by ``scripts/download_*.py``. Prefer calling
``ensure_*`` helpers from dataset modules; use these helpers only to keep
CLI surfaces consistent across acquisition scripts.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def add_cache_dir_argument(
    parser: argparse.ArgumentParser,
    *,
    default: Path | None,
    help_suffix: str | None = None,
) -> None:
    """Add a standard ``--cache-dir`` argument.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to mutate.
    default : Path or None
        Default cache directory, or ``None`` when the dataset module resolves
        the default inside ``ensure_*``.
    help_suffix : str, optional
        Extra help text when ``default`` is ``None`` (e.g. dataset-specific
        path description).
    """
    if default is None:
        help_text = help_suffix or "Output cache directory."
    else:
        help_text = f"Output directory (default: {default})"
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=default,
        help=help_text,
    )


def add_force_argument(
    parser: argparse.ArgumentParser,
    *,
    help: str = "Rebuild the cache even when the artifact already exists.",
) -> None:
    """Add a standard ``--force`` flag.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to mutate.
    help : str, optional
        Help string for ``--force``.
    """
    parser.add_argument("--force", action="store_true", help=help)


def add_print_acquisition_argument(parser: argparse.ArgumentParser) -> None:
    """Add ``--print-acquisition`` (print notes and exit).

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to mutate.
    """
    parser.add_argument(
        "--print-acquisition",
        action="store_true",
        help="Print dataset acquisition notes and exit.",
    )


def add_fetch_argument(
    parser: argparse.ArgumentParser,
    *,
    help: str,
) -> None:
    """Add a boolean ``--fetch`` flag for remote acquisition into ``--cache-dir``.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to mutate.
    help : str
        Dataset-specific help describing what is downloaded.
    """
    parser.add_argument("--fetch", action="store_true", help=help)


def add_expected_sha256_argument(
    parser: argparse.ArgumentParser,
    *,
    help: str = "Optional SHA256 hex digest for the primary raw artifact.",
) -> None:
    """Add ``--expected-sha256`` for optional integrity checks.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to mutate.
    help : str, optional
        Help string for the digest flag.
    """
    parser.add_argument(
        "--expected-sha256",
        type=str,
        default=None,
        help=help,
    )


def maybe_print_acquisition(args: argparse.Namespace, notes: str) -> bool:
    """Print acquisition notes when requested.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments; looks for ``print_acquisition``.
    notes : str
        Acquisition / license notes to print.

    Returns
    -------
    bool
        ``True`` when notes were printed and the caller should exit.
    """
    if getattr(args, "print_acquisition", False):
        print(notes.strip())
        return True
    return False


def configure_acquisition_parser(
    description: str,
    *,
    acquisition_notes: str,
) -> argparse.ArgumentParser:
    """Create an ``ArgumentParser`` with acquisition epilog formatting.

    Parameters
    ----------
    description : str
        One-line parser description.
    acquisition_notes : str
        Epilog text (typically the same string as ``--print-acquisition``).

    Returns
    -------
    argparse.ArgumentParser
        Parser configured with :class:`argparse.RawDescriptionHelpFormatter`.
    """
    return argparse.ArgumentParser(
        description=description,
        epilog=acquisition_notes,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
