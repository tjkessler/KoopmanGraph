"""Optional Ray helpers for parallel ensemble **member** fits.

Embarrassingly parallel member training for
:class:`~koopman_graph.uq.EnsembleGraphKoopmanModel`. Each Ray task builds a
member from a factory, runs single-process ``fit``, and returns
``(state_dict, FitHistory)``. Default ensemble fitting remains sequential.

Ray is imported lazily so :mod:`koopman_graph.distributed` stays importable
without the ``[ray]`` extra until this submodule is used.

Notes
-----
Prefer module-level ``member_factory`` callables (cloudpickle-friendly).
This path does **not** change UQ coverage guarantees — members remain
independent fits. It does **not** shard one model across GPUs.

For multi-GPU *model* training prefer
:func:`~koopman_graph.distributed.run_ddp_fit_loop` or
:func:`~koopman_graph.distributed.fit_with_fabric` (recommended defaults).
Optional Ray Train model-DDP orchestration is
:func:`~koopman_graph.distributed.run_ray_train_fit_loop` — a separate API.
"""

from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Callable, Sequence
from typing import Any

from koopman_graph.training import FitHistory

__all__ = ["fit_ensemble_with_ray"]

_RAY_INSTALL_HINT = 'pip install "koopman-graph[ray]"'

# Avoid importing ``koopman_graph.model`` here — ``GraphKoopmanModel.fit``
# lazy-imports this package and a top-level model import would cycle.
MemberFactory = Callable[[], Any]


def _prefer_driver_interpreter_for_uv_run() -> None:
    """Disable Ray's ``uv run`` worker rewrite when unset.

    Under ``uv run --no-sync`` after ``uv sync``, Ray's default UV runtime-env
    hook rewrites workers as ``uv run ... python`` and can miss the synced
    project environment (``ModuleNotFoundError: ray``). Prefer the driver
    interpreter via :func:`_ray_init_kwargs` instead.

    Notes
    -----
    Must run before ``import ray`` so Ray reads the flag at import time.
    """
    os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")


def _import_ray() -> Any:
    """Import the ``ray`` package with an actionable install hint.

    Returns
    -------
    module
        The ``ray`` package.

    Raises
    ------
    ImportError
        If Ray is not installed.
    """
    _prefer_driver_interpreter_for_uv_run()
    try:
        return importlib.import_module("ray")
    except ImportError as exc:
        msg = (
            "Ray is required for fit_ensemble_with_ray; "
            f"install with: {_RAY_INSTALL_HINT}"
        )
        raise ImportError(msg) from exc


def _ray_init_kwargs(
    *,
    ray_address: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Build ``ray.init`` kwargs that keep workers on the driver interpreter.

    Pinning ``runtime_env['py_executable']`` to :data:`sys.executable` keeps
    local workers aligned with the driver when Ray's ``uv run`` rewrite is
    disabled (see :func:`_prefer_driver_interpreter_for_uv_run`).

    Parameters
    ----------
    ray_address : str or None, optional
        Optional Ray cluster address forwarded as ``address=...``.
    **extra
        Additional ``ray.init`` keyword arguments (for example ``num_cpus``).

    Returns
    -------
    dict of str to object
        Keyword arguments suitable for ``ray.init(**kwargs)``.
    """
    _prefer_driver_interpreter_for_uv_run()
    kwargs: dict[str, Any] = {
        "ignore_reinit_error": True,
        "runtime_env": {"py_executable": sys.executable},
    }
    if ray_address is not None:
        kwargs["address"] = ray_address
    kwargs.update(extra)
    return kwargs


def _fit_member_task(
    member_factory: MemberFactory,
    data_ref: Any,
    seed: int | None,
    fit_kwargs: dict[str, Any],
) -> tuple[dict[str, Any], FitHistory]:
    """Fit one ensemble member inside a Ray worker.

    Parameters
    ----------
    member_factory : callable
        Zero-argument factory returning a fresh trainable Koopman model
        (typically :class:`~koopman_graph.model.GraphKoopmanModel`).
    data_ref : object
        Training input (often a Ray ObjectRef resolved by the runtime).
    seed : int or None
        Optional ``torch.manual_seed`` value before construction / fit.
    fit_kwargs : dict
        Keyword arguments forwarded to ``member.fit``.

    Returns
    -------
    state_dict : dict
        CPU ``state_dict`` of the fitted member.
    history : FitHistory
        Per-member training history.
    """
    import torch

    if seed is not None:
        torch.manual_seed(int(seed))
    member = member_factory()
    if seed is not None:
        torch.manual_seed(int(seed))
    history = member.fit(data_ref, **fit_kwargs)
    state_dict = {
        key: value.detach().cpu() for key, value in member.state_dict().items()
    }
    return state_dict, history


def fit_ensemble_with_ray(
    member_factory: MemberFactory,
    data_sequence: Any,
    *,
    num_members: int,
    seeds: Sequence[int] | None = None,
    ray_address: str | None = None,
    **fit_kwargs: Any,
) -> tuple[list[dict[str, Any]], list[FitHistory]]:
    """Fit independent ensemble members as Ray tasks.

    Parameters
    ----------
    member_factory : callable
        Zero-argument callable returning a fresh trainable Koopman model
        (typically :class:`~koopman_graph.model.GraphKoopmanModel`). Prefer
        a **module-level** function so Ray's cloudpickle can serialize it.
    data_sequence
        Training input forwarded to each member's ``fit`` (put once via
        ``ray.put``). Accepts the same homogeneous or multiplex
        snapshot sequences / multi-trajectory inputs as single-process
        ``fit``.
    num_members : int
        Ensemble size (must be >= 1).
    seeds : sequence of int or None, optional
        One seed per member. When omitted, uses ``0 .. num_members-1``.
    ray_address : str or None, optional
        Passed to ``ray.init(address=...)`` only when Ray is not already
        initialized. Default ``None`` starts / joins a local runtime.
    **fit_kwargs
        Forwarded unchanged to each member's ``fit`` (must not include
        ``parallel_backend`` / ``member_factory``).

    Returns
    -------
    state_dicts : list of dict
        Per-member ``state_dict`` values in member order (CPU tensors).
    histories : list of FitHistory
        Per-member training histories in member order.

    Raises
    ------
    ImportError
        If Ray is not installed.
    ValueError
        If ``num_members < 1`` or ``seeds`` length mismatches.

    Notes
    -----
    Does not claim new predictive-coverage guarantees relative to sequential
    ensemble fits. This API parallelizes *members* only. For multi-GPU *model*
    training prefer
    :func:`~koopman_graph.distributed.run_ddp_fit_loop` /
    :func:`~koopman_graph.distributed.fit_with_fabric`, or optional
    :func:`~koopman_graph.distributed.run_ray_train_fit_loop`.
    """
    if num_members < 1:
        msg = f"num_members must be >= 1; got {num_members}"
        raise ValueError(msg)
    resolved_seeds: list[int | None]
    if seeds is None:
        resolved_seeds = list(range(num_members))
    else:
        resolved_seeds = [int(seed) for seed in seeds]
        if len(resolved_seeds) != num_members:
            msg = (
                "seeds must have length num_members; "
                f"got len(seeds)={len(resolved_seeds)}, num_members={num_members}"
            )
            raise ValueError(msg)

    banned = {"parallel_backend", "member_factory"}
    overlap = banned.intersection(fit_kwargs)
    if overlap:
        msg = (
            "fit_ensemble_with_ray does not accept "
            f"{sorted(overlap)}; pass member fit kwargs only"
        )
        raise TypeError(msg)

    ray = _import_ray()
    if not ray.is_initialized():
        ray.init(**_ray_init_kwargs(ray_address=ray_address))

    data_ref = ray.put(data_sequence)
    remote_fit = ray.remote(_fit_member_task)
    futures = [
        remote_fit.remote(member_factory, data_ref, seed, fit_kwargs)
        for seed in resolved_seeds
    ]
    results = ray.get(futures)
    state_dicts = [state for state, _history in results]
    histories = [history for _state, history in results]
    return state_dicts, histories
