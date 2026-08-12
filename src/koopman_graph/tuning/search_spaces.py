"""Example-only Ray Tune param-space scaffolds (smoke tests, not science).

These helpers build small Ray Tune–compatible ``param_space`` dicts for local
smoke runs. They are **not** recommended scientific defaults, domain oracles,
or an AutoML search-space DSL. Real studies must define their own ranges and
searchable keys.

Requires the ``[ray]`` extra. Ray is imported lazily when a helper is called
so :mod:`koopman_graph.tuning` stays importable without Ray.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any

_RAY_INSTALL_HINT = 'pip install "koopman-graph[ray]"'


def _import_tune() -> Any:
    """Import ``ray.tune`` with an actionable install hint.

    Returns
    -------
    module
        The ``ray.tune`` module.

    Raises
    ------
    ImportError
        If Ray Tune is not installed (``[ray]`` extra missing).
    """
    try:
        return importlib.import_module("ray.tune")
    except ImportError as exc:
        msg = (
            "Ray Tune is required for koopman_graph.tuning search-space "
            f"helpers; install with: {_RAY_INSTALL_HINT}"
        )
        raise ImportError(msg) from exc


def example_lr_loguniform_space(
    *,
    lr_low: float = 1e-3,
    lr_high: float = 1e-1,
    **fixed: Any,
) -> dict[str, Any]:
    """Build an *example* param space with log-uniform learning rate.

    For smoke tests only — not a recommended scientific default.

    Parameters
    ----------
    lr_low, lr_high : float, optional
        Bounds for ``tune.loguniform`` (defaults ``1e-3`` … ``1e-1``).
    **fixed
        Additional fixed config keys (for example ``epochs``, ``seed``).

    Returns
    -------
    dict
        Ray Tune–compatible ``param_space`` mapping.

    Raises
    ------
    ImportError
        If Ray Tune is not installed.
    ValueError
        If ``lr_low >= lr_high`` or bounds are non-positive.
    """
    if lr_low <= 0 or lr_high <= 0:
        raise ValueError("lr_low and lr_high must be positive")
    if lr_low >= lr_high:
        raise ValueError(f"lr_low must be < lr_high, got {lr_low} >= {lr_high}")

    tune = _import_tune()
    space: dict[str, Any] = {"lr": tune.loguniform(lr_low, lr_high)}
    space.update(fixed)
    return space


def example_lr_latent_dim_space(
    *,
    lr_low: float = 1e-3,
    lr_high: float = 1e-1,
    latent_dims: Sequence[int] = (8, 16),
    **fixed: Any,
) -> dict[str, Any]:
    """Build an *example* param space over ``lr`` and ``latent_dim``.

    For smoke tests only — not a recommended scientific default. The
    ``latent_dims`` values are illustrative integers for tiny models.

    Parameters
    ----------
    lr_low, lr_high : float, optional
        Bounds for ``tune.loguniform`` on ``lr``.
    latent_dims : sequence of int, optional
        Discrete choices for ``latent_dim`` (default ``(8, 16)``).
    **fixed
        Additional fixed config keys.

    Returns
    -------
    dict
        Ray Tune–compatible ``param_space`` mapping.

    Raises
    ------
    ImportError
        If Ray Tune is not installed.
    ValueError
        If ``latent_dims`` is empty or contains non-positive values, or if
        learning-rate bounds are invalid.
    """
    dims = tuple(int(d) for d in latent_dims)
    if not dims:
        raise ValueError("latent_dims must contain at least one positive int")
    if any(d < 1 for d in dims):
        raise ValueError("latent_dims entries must be >= 1")

    space = example_lr_loguniform_space(lr_low=lr_low, lr_high=lr_high, **fixed)
    tune = _import_tune()
    space["latent_dim"] = tune.choice(list(dims))
    return space
