"""Unit tests for ``koopman_graph.tuning`` (no Ray required for unit suite)."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from tests.helpers import REPO_ROOT

from koopman_graph.training.history import FitHistory
from koopman_graph.tuning import (
    example_lr_latent_dim_space,
    example_lr_loguniform_space,
    fit_history_metrics,
    run_ray_tune,
)


def test_fit_history_metrics_train_only() -> None:
    """Final train loss and reconstruction are reported without val keys."""
    history = FitHistory(
        loss=(1.0, 0.5, 0.25),
        epochs=3,
        reconstruction_loss=(0.9, 0.4, 0.2),
        stopped_early=False,
    )
    metrics = fit_history_metrics(history)
    assert metrics["epochs"] == 3.0
    assert metrics["stopped_early"] == 0.0
    assert metrics["loss"] == 0.25
    assert metrics["final_loss"] == 0.25
    assert metrics["reconstruction_loss"] == 0.2
    assert "val_loss" not in metrics
    assert "best_loss" not in metrics


def test_fit_history_metrics_with_val_and_best() -> None:
    """Validation and best-epoch fields flatten when present."""
    history = FitHistory(
        loss=(1.0, 0.8),
        epochs=2,
        reconstruction_loss=(0.7, 0.6),
        val_loss=(1.2, 0.9),
        val_reconstruction_loss=(1.1, 0.85),
        stopped_early=True,
        best_epoch=1,
        best_loss=0.8,
    )
    metrics = fit_history_metrics(history)
    assert metrics["stopped_early"] == 1.0
    assert metrics["val_loss"] == 0.9
    assert metrics["final_val_loss"] == 0.9
    assert metrics["val_reconstruction_loss"] == 0.85
    assert metrics["best_epoch"] == 1.0
    assert metrics["best_loss"] == 0.8


def test_fit_history_metrics_empty_loss_omits_loss_keys() -> None:
    """Empty loss series omit ``loss`` rather than inventing NaNs."""
    history = FitHistory(loss=(), epochs=0)
    metrics = fit_history_metrics(history)
    assert metrics == {"epochs": 0.0, "stopped_early": 0.0}


def test_fit_history_metrics_early_stop_flag() -> None:
    """Early stopping encodes ``stopped_early`` as ``1.0`` with short epochs."""
    history = FitHistory(
        loss=(1.0, 0.5),
        epochs=2,
        reconstruction_loss=(0.8, 0.4),
        stopped_early=True,
        best_epoch=1,
        best_loss=0.5,
    )
    metrics = fit_history_metrics(history)
    assert metrics["stopped_early"] == 1.0
    assert metrics["epochs"] == 2.0
    assert metrics["loss"] == 0.5
    assert metrics["best_loss"] == 0.5
    assert metrics["best_epoch"] == 1.0


def test_fit_history_metrics_missing_val_series_omitted() -> None:
    """``val_loss=None`` and empty val tuples omit validation metric keys."""
    none_history = FitHistory(
        loss=(0.5,),
        epochs=1,
        reconstruction_loss=(0.4,),
        val_loss=None,
        val_reconstruction_loss=None,
    )
    none_metrics = fit_history_metrics(none_history)
    assert "val_loss" not in none_metrics
    assert "final_val_loss" not in none_metrics
    assert "val_reconstruction_loss" not in none_metrics

    empty_history = FitHistory(
        loss=(0.5,),
        epochs=1,
        reconstruction_loss=(0.4,),
        val_loss=(),
        val_reconstruction_loss=(),
    )
    empty_metrics = fit_history_metrics(empty_history)
    assert "val_loss" not in empty_metrics
    assert "final_val_loss" not in empty_metrics
    assert "val_reconstruction_loss" not in empty_metrics


def test_fit_history_metrics_partial_optionals() -> None:
    """Empty reconstruction and one-sided best fields stay omitted."""
    no_recon = FitHistory(
        loss=(1.0, 0.2),
        epochs=2,
        reconstruction_loss=(),
    )
    metrics = fit_history_metrics(no_recon)
    assert metrics["loss"] == 0.2
    assert "reconstruction_loss" not in metrics

    best_loss_only = FitHistory(loss=(0.3,), epochs=1, best_loss=0.3)
    metrics_loss = fit_history_metrics(best_loss_only)
    assert metrics_loss["best_loss"] == 0.3
    assert "best_epoch" not in metrics_loss

    best_epoch_only = FitHistory(loss=(0.3,), epochs=1, best_epoch=0)
    metrics_epoch = fit_history_metrics(best_epoch_only)
    assert metrics_epoch["best_epoch"] == 0.0
    assert "best_loss" not in metrics_epoch


def test_fit_history_metrics_values_are_floats() -> None:
    """Returned mapping values are plain floats (Tune/Optuna-friendly)."""
    history = FitHistory(
        loss=(1.0, 0.5),
        epochs=2,
        reconstruction_loss=(0.9, 0.4),
        val_loss=(1.1, 0.6),
        val_reconstruction_loss=(1.0, 0.55),
        stopped_early=True,
        best_epoch=1,
        best_loss=0.5,
    )
    metrics = fit_history_metrics(history)
    assert metrics
    assert all(type(value) is float for value in metrics.values())
    assert all(value == value for value in metrics.values())  # no NaN


def test_tuning_not_on_root_all() -> None:
    """``tuning`` stays a power-user import (off root ``__all__``)."""
    import koopman_graph

    assert "fit_history_metrics" not in koopman_graph.__all__
    assert "run_ray_tune" not in koopman_graph.__all__


def _top_level_ray_import_lines(path: Path) -> list[str]:
    """Return source lines that import Ray at module scope."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "ray" or alias.name.startswith("ray."):
                    offenders.append(f"{path.name}:{node.lineno}")
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module is not None
            and (node.module == "ray" or node.module.startswith("ray."))
        ):
            offenders.append(f"{path.name}:{node.lineno}")
    return offenders


def test_tuning_has_no_toplevel_ray_imports() -> None:
    """Importing ``tuning`` must not execute a top-level ``import ray``."""
    package_root = REPO_ROOT / "src" / "koopman_graph"
    tuning_root = package_root / "tuning"
    offenders: list[str] = []
    for path in tuning_root.rglob("*.py"):
        offenders.extend(_top_level_ray_import_lines(path))
    assert offenders == []


def test_importing_tuning_does_not_load_ray() -> None:
    """``import koopman_graph.tuning`` must not import Ray by itself."""
    import subprocess

    script = (
        "import sys\n"
        "import koopman_graph.tuning\n"
        "assert 'ray' not in sys.modules, sorted(sys.modules)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_training_does_not_import_tuning() -> None:
    """Layer rule: ``training`` must not import ``tuning``."""
    package_root = REPO_ROOT / "src" / "koopman_graph"
    training_root = package_root / "training"
    offenders: list[str] = []
    for path in training_root.rglob("*.py"):
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            from_tuning = stripped.startswith("from koopman_graph.tuning")
            import_tuning = stripped.startswith("import koopman_graph.tuning")
            if from_tuning or import_tuning:
                offenders.append(str(path.relative_to(package_root.parent)))
                break
    assert offenders == []


def test_run_ray_tune_rejects_nonpositive_samples() -> None:
    """``num_samples`` must be at least 1."""
    with pytest.raises(ValueError, match="num_samples"):
        run_ray_tune(lambda config: None, {"lr": 1e-3}, num_samples=0)


def test_run_ray_tune_missing_ray_message() -> None:
    """Missing Ray raises ImportError with the ``[ray]`` install hint."""
    from koopman_graph.tuning import ray_tune as ray_tune_mod

    with (
        patch.object(
            ray_tune_mod,
            "_import_ray_tune",
            side_effect=ImportError(
                "Ray Tune is required for koopman_graph.tuning.run_ray_tune; "
                'install with: pip install "koopman-graph[ray]"'
            ),
        ),
        pytest.raises(ImportError, match=r"koopman-graph\[ray\]"),
    ):
        ray_tune_mod.run_ray_tune(lambda config: None, {"lr": 1e-3}, num_samples=1)


def test_run_ray_tune_builds_tuner_with_mocks(tmp_path: Path) -> None:
    """Wrapper constructs TuneConfig / Tuner and returns ``fit()`` result."""
    from koopman_graph.tuning import ray_tune as ray_tune_mod

    fake_results = SimpleNamespace(ok=True)
    fake_tune_config = object()
    fake_run_config = object()
    fake_tuner = MagicMock()
    fake_tuner.fit.return_value = fake_results

    fake_ray = ModuleType("ray")
    fake_ray.is_initialized = MagicMock(return_value=True)  # type: ignore[attr-defined]
    fake_ray.init = MagicMock()  # type: ignore[attr-defined]

    fake_tune = ModuleType("ray.tune")
    fake_tune.TuneConfig = MagicMock(return_value=fake_tune_config)  # type: ignore[attr-defined]
    fake_tune.Tuner = MagicMock(return_value=fake_tuner)  # type: ignore[attr-defined]

    fake_run_config_cls = MagicMock(return_value=fake_run_config)

    def _trainable(config: dict[str, object]) -> None:
        return None

    with patch.object(
        ray_tune_mod,
        "_import_ray_tune",
        return_value=(fake_ray, fake_tune, fake_run_config_cls),
    ):
        result = ray_tune_mod.run_ray_tune(
            _trainable,
            {"lr": 0.01, "epochs": 1},
            num_samples=2,
            metric="loss",
            mode="min",
            storage_path=tmp_path / "tune",
            run_name="unit_test_tune",
        )

    assert result is fake_results
    fake_ray.init.assert_not_called()
    fake_tune.TuneConfig.assert_called_once_with(
        num_samples=2,
        metric="loss",
        mode="min",
    )
    fake_run_config_cls.assert_called_once()
    run_kwargs = fake_run_config_cls.call_args.kwargs
    assert run_kwargs["name"] == "unit_test_tune"
    assert run_kwargs["storage_path"] == str((tmp_path / "tune").resolve())
    fake_tune.Tuner.assert_called_once()
    tuner_kwargs = fake_tune.Tuner.call_args
    assert tuner_kwargs.args[0] is _trainable
    assert tuner_kwargs.kwargs["param_space"] == {"lr": 0.01, "epochs": 1}
    assert tuner_kwargs.kwargs["tune_config"] is fake_tune_config
    assert tuner_kwargs.kwargs["run_config"] is fake_run_config
    fake_tuner.fit.assert_called_once_with()


def test_example_search_space_names_and_docs_are_honest() -> None:
    """Helpers are explicitly example-only (not scientific defaults)."""
    from koopman_graph.tuning import search_spaces

    assert example_lr_loguniform_space.__name__.startswith("example_")
    assert example_lr_latent_dim_space.__name__.startswith("example_")
    module_doc = search_spaces.__doc__ or ""
    assert "not" in module_doc.lower()
    assert "scientific" in module_doc.lower() or "defaults" in module_doc.lower()
    assert "smoke" in module_doc.lower() or "example" in module_doc.lower()


def test_example_lr_loguniform_space_with_mocked_tune() -> None:
    """LR helper returns a Tune-compatible dict via lazy ``ray.tune``."""
    from koopman_graph.tuning import search_spaces as spaces_mod

    fake_sampler = object()
    fake_tune = ModuleType("ray.tune")
    fake_tune.loguniform = MagicMock(return_value=fake_sampler)  # type: ignore[attr-defined]

    with patch.object(spaces_mod, "_import_tune", return_value=fake_tune):
        space = spaces_mod.example_lr_loguniform_space(
            lr_low=1e-4,
            lr_high=1e-2,
            epochs=2,
            seed=0,
        )

    assert space["lr"] is fake_sampler
    assert space["epochs"] == 2
    assert space["seed"] == 0
    fake_tune.loguniform.assert_called_once_with(1e-4, 1e-2)


def test_example_lr_latent_dim_space_with_mocked_tune() -> None:
    """Joint lr / latent_dim example space includes both samplers."""
    from koopman_graph.tuning import search_spaces as spaces_mod

    fake_lr = object()
    fake_dim = object()
    fake_tune = ModuleType("ray.tune")
    fake_tune.loguniform = MagicMock(return_value=fake_lr)  # type: ignore[attr-defined]
    fake_tune.choice = MagicMock(return_value=fake_dim)  # type: ignore[attr-defined]

    with patch.object(spaces_mod, "_import_tune", return_value=fake_tune):
        space = spaces_mod.example_lr_latent_dim_space(
            latent_dims=(4, 8),
            epochs=1,
        )

    assert space["lr"] is fake_lr
    assert space["latent_dim"] is fake_dim
    assert space["epochs"] == 1
    fake_tune.choice.assert_called_once_with([4, 8])


def test_example_search_space_missing_ray_message() -> None:
    """Missing Ray raises ImportError with the ``[ray]`` install hint."""
    from koopman_graph.tuning import search_spaces as spaces_mod

    with (
        patch.object(
            spaces_mod,
            "_import_tune",
            side_effect=ImportError(
                "Ray Tune is required for koopman_graph.tuning search-space "
                'helpers; install with: pip install "koopman-graph[ray]"'
            ),
        ),
        pytest.raises(ImportError, match=r"koopman-graph\[ray\]"),
    ):
        spaces_mod.example_lr_loguniform_space()


def test_example_search_space_validates_bounds() -> None:
    """Invalid learning-rate bounds raise ``ValueError`` before Ray use."""
    with pytest.raises(ValueError, match="lr_low"):
        example_lr_loguniform_space(lr_low=0.1, lr_high=0.01)
