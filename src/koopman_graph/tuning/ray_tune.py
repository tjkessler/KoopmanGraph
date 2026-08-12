"""Thin Ray Tune wrapper (lazy import; requires ``[ray]``).

Search spaces and trainables remain caller-owned — this module is not an
AutoML product. Prefer the examples-only script
``examples/scripts/ray_tune_koopman_example.py`` for a full smoke pattern
(including metrics from :func:`~koopman_graph.tuning.fit_history_metrics`).
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

_RAY_INSTALL_HINT = 'pip install "koopman-graph[ray]"'


def _prefer_driver_interpreter_for_uv_run() -> None:
    """Disable Ray's ``uv run`` worker rewrite when unset.

    Must run before ``import ray`` so Ray reads the flag at import time.

    Notes
    -----
    Sets ``RAY_ENABLE_UV_RUN_RUNTIME_ENV=0`` via ``setdefault`` so existing
    user overrides are preserved.
    """
    os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")


def _import_ray_tune() -> tuple[Any, Any, Any]:
    """Import ``ray``, ``ray.tune``, and ``ray.train.RunConfig`` lazily.

    Returns
    -------
    ray_module, tune_module, RunConfig
        Imported symbols.

    Raises
    ------
    ImportError
        If Ray / Ray Tune is not installed (``[ray]`` extra missing).
    """
    _prefer_driver_interpreter_for_uv_run()
    try:
        ray_module = importlib.import_module("ray")
        tune_module = importlib.import_module("ray.tune")
        run_config_cls = importlib.import_module("ray.train").RunConfig
    except ImportError as exc:
        msg = (
            "Ray Tune is required for koopman_graph.tuning.run_ray_tune; "
            f"install with: {_RAY_INSTALL_HINT}"
        )
        raise ImportError(msg) from exc
    return ray_module, tune_module, run_config_cls


def run_ray_tune(
    trainable: Callable[..., Any] | type,
    param_space: Mapping[str, Any],
    *,
    num_samples: int = 1,
    metric: str = "loss",
    mode: str = "min",
    storage_path: str | Path | None = None,
    run_name: str | None = None,
    ray_init_kwargs: Mapping[str, Any] | None = None,
    **tuner_kwargs: Any,
) -> Any:
    """Run a Ray Tune search with a caller-owned trainable and param space.

    This is a thin façade over :class:`ray.tune.Tuner`. It does not define
    searchable hyperparameters or scientific defaults. Requires the
    ``[ray]`` extra.

    Parameters
    ----------
    trainable : callable or class
        Ray Tune trainable (function or Trainable subclass).
    param_space : mapping
        Search space / fixed config owned by the caller.
    num_samples : int, optional
        Number of samples for :class:`~ray.tune.TuneConfig` (default 1).
    metric : str, optional
        Metric name reported by the trainable (default ``\"loss\"``).
    mode : str, optional
        Optimization mode ``\"min\"`` or ``\"max\"`` (default ``\"min\"``).
    storage_path : str, Path, or None, optional
        When set, builds a :class:`~ray.train.RunConfig` with this storage
        directory.
    run_name : str or None, optional
        Optional Tune run name when ``storage_path`` is set (default
        ``koopman_graph_tune``).
    ray_init_kwargs : mapping or None, optional
        Forwarded to ``ray.init`` when Ray is not already initialized.
    **tuner_kwargs
        Extra keyword arguments forwarded to :class:`ray.tune.Tuner`
        (for example a pre-built ``tune_config`` overrides the default
        constructed here only if you pass ``tune_config=`` explicitly —
        prefer the dedicated parameters above for the common case).

    Returns
    -------
    ResultGrid
        Return value of :meth:`ray.tune.Tuner.fit`.

    Raises
    ------
    ImportError
        If Ray Tune is not installed.
    ValueError
        If ``num_samples < 1``.

    See Also
    --------
    examples/scripts/ray_tune_koopman_example.py
        Examples-only smoke that owns the search space and trainable body.
    """
    if num_samples < 1:
        msg = f"num_samples must be >= 1, got {num_samples}"
        raise ValueError(msg)

    ray_module, tune_module, run_config_cls = _import_ray_tune()

    if not ray_module.is_initialized():
        init_kwargs = dict(ray_init_kwargs or {})
        ray_module.init(**init_kwargs)

    tune_config = tuner_kwargs.pop("tune_config", None)
    if tune_config is None:
        tune_config = tune_module.TuneConfig(
            num_samples=num_samples,
            metric=metric,
            mode=mode,
        )

    run_config = tuner_kwargs.pop("run_config", None)
    if run_config is None and storage_path is not None:
        run_config = run_config_cls(
            name=run_name or "koopman_graph_tune",
            storage_path=str(Path(storage_path).resolve()),
        )

    tuner = tune_module.Tuner(
        trainable,
        param_space=dict(param_space),
        tune_config=tune_config,
        run_config=run_config,
        **tuner_kwargs,
    )
    return tuner.fit()
