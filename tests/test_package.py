"""Smoke tests for package installation and imports."""

from __future__ import annotations

import importlib.metadata
import re
import subprocess
import sys
from pathlib import Path

import koopman_graph

_VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+")


def test_import_package() -> None:
    """Verify the package imports and exposes a semver-like ``__version__``."""
    assert koopman_graph.__version__
    assert _VERSION_PATTERN.match(koopman_graph.__version__)


def test_installed_version_matches_metadata() -> None:
    """Editable install should expose the same version via importlib.metadata."""
    assert importlib.metadata.version("koopman-graph") == koopman_graph.__version__


def test_build_produces_wheel_and_sdist() -> None:
    """Verify ``python -m build`` produces installable artifacts."""
    root = Path(__file__).resolve().parents[1]
    dist_dir = root / "dist"
    dist_dir.mkdir(exist_ok=True)

    result = subprocess.run(
        [sys.executable, "-m", "build", "--outdir", str(dist_dir)],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout

    assert list(dist_dir.glob("koopman_graph-*.whl"))
    assert list(dist_dir.glob("koopman_graph-*.tar.gz"))
