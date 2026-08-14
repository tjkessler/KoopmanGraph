"""Coverage and error-path tests for :mod:`koopman_graph.tuning`."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import koopman_graph.tuning.ray_tune as ray_tune_mod


def test_ray_import_helpers_cover_success_failure_and_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ray helpers set the uv flag and wrap lazy-import failures."""
    monkeypatch.delenv("RAY_ENABLE_UV_RUN_RUNTIME_ENV", raising=False)
    ray_tune_mod._prefer_driver_interpreter_for_uv_run()
    assert sys.modules["os"].environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] == "0"

    fake_ray = ModuleType("ray")
    fake_tune = ModuleType("ray.tune")
    fake_run_config = object()
    fake_train = SimpleNamespace(RunConfig=fake_run_config)

    def import_ray(name: str) -> object:
        return {
            "ray": fake_ray,
            "ray.tune": fake_tune,
            "ray.train": fake_train,
        }[name]

    with patch.object(
        ray_tune_mod.importlib,
        "import_module",
        side_effect=import_ray,
    ):
        assert ray_tune_mod._import_ray_tune() == (
            fake_ray,
            fake_tune,
            fake_run_config,
        )

    with (
        patch.object(
            ray_tune_mod.importlib,
            "import_module",
            side_effect=ImportError("missing ray"),
        ),
        pytest.raises(ImportError, match=r"koopman-graph\[ray\]"),
    ):
        ray_tune_mod._import_ray_tune()


def test_ray_tune_initializes_uninitialized_runtime() -> None:
    """The wrapper initializes Ray with caller-provided kwargs when needed."""
    fake_ray = ModuleType("ray")
    fake_ray.is_initialized = MagicMock(return_value=False)  # type: ignore[attr-defined]
    fake_ray.init = MagicMock()  # type: ignore[attr-defined]
    fake_tuner = MagicMock()
    fake_tuner.fit.return_value = "results"
    fake_tune = ModuleType("ray.tune")
    fake_tune.TuneConfig = MagicMock(return_value=object())  # type: ignore[attr-defined]
    fake_tune.Tuner = MagicMock(return_value=fake_tuner)  # type: ignore[attr-defined]

    with patch.object(
        ray_tune_mod,
        "_import_ray_tune",
        return_value=(fake_ray, fake_tune, MagicMock()),
    ):
        result = ray_tune_mod.run_ray_tune(
            lambda config: config,
            {"lr": 0.01},
            ray_init_kwargs={"local_mode": True},
        )
    assert result == "results"
    fake_ray.init.assert_called_once_with(local_mode=True)
