"""Import isolation for :mod:`koopman_graph.datasets.reaction`.

The reaction subpackage keeps SciPy and Cantera behind the optional
``[reaction]`` / ``[cantera]`` extras. Importing it must not pull either into
``sys.modules``. The package-wide version of this rule lives in
``tests/test_import_hygiene.py``.
"""

from __future__ import annotations

import importlib.util

import pytest
from tests.helpers import loaded_modules_after_import


def test_reaction_import_does_not_load_scipy_or_cantera() -> None:
    loaded = loaded_modules_after_import(
        "koopman_graph.datasets.reaction", ("scipy", "cantera")
    )
    assert loaded == []


def test_reaction_all_lists_importable_names() -> None:
    import koopman_graph.datasets.reaction as reaction

    assert "ReactionSystem" in reaction.__all__
    for name in reaction.__all__:
        assert hasattr(reaction, name), name


def test_root_all_does_not_export_reaction_names() -> None:
    import koopman_graph

    assert "reaction" not in koopman_graph.__all__
    assert "ReactionSystem" not in koopman_graph.__all__


@pytest.mark.cantera
def test_cantera_marker_runs_only_with_extra() -> None:
    """``tests/conftest.py`` skips this unless ``[cantera]`` is installed."""
    assert importlib.util.find_spec("cantera") is not None
