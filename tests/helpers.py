"""Repository path and import-isolation helpers for tests at any depth."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Iterable
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


def loaded_modules_after_import(module: str, roots: Iterable[str]) -> list[str]:
    """Import ``module`` in a fresh interpreter and report loaded ``roots``.

    A fresh interpreter is needed because the test session may already have
    imported the modules under test through other tests.

    Parameters
    ----------
    module : str
        Dotted module to import.
    roots : iterable of str
        Top-level package names to look for in ``sys.modules`` afterwards.

    Returns
    -------
    list of str
        Sorted ``sys.modules`` keys whose top-level package is in ``roots``.
    """
    code = (
        "import importlib, json, sys\n"
        f"importlib.import_module({module!r})\n"
        f"roots = {tuple(roots)!r}\n"
        "print(json.dumps(sorted(\n"
        "    m for m in sys.modules\n"
        "    if m.split('.')[0] in roots\n"
        ")))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])
