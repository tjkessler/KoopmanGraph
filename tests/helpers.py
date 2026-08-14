"""Repository path helpers for tests at any directory depth."""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Return the KoopmanGraph repository root.

    Returns
    -------
    Path
        Directory containing ``pyproject.toml`` and ``src/koopman_graph``.

    Raises
    ------
    RuntimeError
        If no ancestor of this file is a KoopmanGraph checkout.
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (
            parent / "src" / "koopman_graph"
        ).is_dir():
            return parent
    msg = "could not locate the KoopmanGraph repository root"
    raise RuntimeError(msg)


REPO_ROOT = repo_root()
