"""Package-wide import hygiene: optional heavy dependencies stay lazy.

``import koopman_graph`` (and the always-imported subpackages) must not pull
SciPy, Cantera, or torchdiffeq into ``sys.modules``. Modules that need them
import inside the function that uses them (see ``CONTRIBUTING.md``). This
holds even when the matching extras are installed, which is how CI runs.
"""

from __future__ import annotations

import pytest

from tests.helpers import loaded_modules_after_import

_LAZY_ONLY = ("scipy", "cantera", "torchdiffeq")


@pytest.mark.parametrize(
    "module",
    [
        "koopman_graph",
        "koopman_graph.datasets",
        "koopman_graph.baselines.gnn",
        "koopman_graph.baselines.gnn.stgode",
    ],
)
def test_import_does_not_load_lazy_only_dependencies(module: str) -> None:
    assert loaded_modules_after_import(module, _LAZY_ONLY) == []
