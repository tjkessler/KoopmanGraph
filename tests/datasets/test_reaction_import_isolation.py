"""Import isolation for :mod:`koopman_graph.datasets.reaction`.

The reaction subpackage keeps SciPy and Cantera behind the optional
``[reaction]`` / ``[cantera]`` extras. Importing it must not pull either into
``sys.modules``. Checks run in a fresh interpreter because the test session
may already have imported SciPy through other modules.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

_HEAVY = ("scipy", "cantera")


def _loaded_heavy_modules(module: str) -> list[str]:
    code = (
        "import importlib, json, sys\n"
        f"importlib.import_module({module!r})\n"
        f"heavy = {_HEAVY!r}\n"
        "print(json.dumps(sorted(\n"
        "    m for m in sys.modules\n"
        "    if m.split('.')[0] in heavy\n"
        ")))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize(
    "module",
    [
        "koopman_graph",
        "koopman_graph.datasets",
        "koopman_graph.datasets.reaction",
    ],
)
def test_import_does_not_load_scipy_or_cantera(module: str) -> None:
    assert _loaded_heavy_modules(module) == []


def test_reaction_all_is_empty_skeleton() -> None:
    import koopman_graph.datasets.reaction as reaction

    assert reaction.__all__ == []


def test_root_all_does_not_export_reaction_names() -> None:
    import koopman_graph

    assert "reaction" not in koopman_graph.__all__
    assert "ReactionSystem" not in koopman_graph.__all__
