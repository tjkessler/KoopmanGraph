"""Optional Ray helpers for parallel ensemble member fits.

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
independent fits. Ray Train as a DDP backend is out of scope; use native
DDP / Fabric for multi-GPU *model* training.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from typing import Any

from koopman_graph.training import FitHistory

__all__ = ["fit_ensemble_with_ray"]

_RAY_INSTALL_HINT = 'pip install "koopman-graph[ray]"'

# Avoid importing ``koopman_graph.model`` here — ``GraphKoopmanModel.fit``
# lazy-imports this package and a top-level model import would cycle.
MemberFactory = Callable[[], Any]


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
    try:
        return importlib.import_module("ray")
    except ImportError as exc:
        msg = (
            "Ray is required for fit_ensemble_with_ray; "
            f"install with: {_RAY_INSTALL_HINT}"
        )
        raise ImportError(msg) from exc


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
    ensemble fits. Ray Train / multi-GPU *model* DDP is out of scope — use
    :func:`~koopman_graph.distributed.run_ddp_fit_loop` or
    :func:`~koopman_graph.distributed.fit_with_fabric` for that.
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
        init_kwargs: dict[str, Any] = {"ignore_reinit_error": True}
        if ray_address is not None:
            init_kwargs["address"] = ray_address
        ray.init(**init_kwargs)

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
