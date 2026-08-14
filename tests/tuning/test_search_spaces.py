"""Coverage and error-path tests for :mod:`koopman_graph.tuning`."""

from __future__ import annotations

from types import ModuleType
from unittest.mock import patch

import pytest

import koopman_graph.tuning.search_spaces as search_spaces_mod


def test_search_space_import_helper_success_and_failure() -> None:
    """The Ray Tune search-space importer handles both lazy-import outcomes."""
    fake_tune = ModuleType("ray.tune")
    with patch.object(
        search_spaces_mod.importlib,
        "import_module",
        return_value=fake_tune,
    ):
        assert search_spaces_mod._import_tune() is fake_tune

    with (
        patch.object(
            search_spaces_mod.importlib,
            "import_module",
            side_effect=ImportError("missing ray"),
        ),
        pytest.raises(ImportError, match=r"koopman-graph\[ray\]"),
    ):
        search_spaces_mod._import_tune()


@pytest.mark.parametrize(
    ("function", "kwargs", "message"),
    [
        (
            search_spaces_mod.example_lr_loguniform_space,
            {"lr_low": 0.0},
            "must be positive",
        ),
        (
            search_spaces_mod.example_lr_latent_dim_space,
            {"latent_dims": ()},
            "at least one",
        ),
        (
            search_spaces_mod.example_lr_latent_dim_space,
            {"latent_dims": (2, 0)},
            "entries must be >= 1",
        ),
    ],
)
def test_search_space_validates_positive_domains(
    function: object,
    kwargs: dict[str, object],
    message: str,
) -> None:
    """Example search spaces reject non-positive and empty domains."""
    with pytest.raises(ValueError, match=message):
        function(**kwargs)  # type: ignore[operator]
