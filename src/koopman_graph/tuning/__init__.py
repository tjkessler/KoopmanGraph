"""Hyperparameter-search helpers for user-defined Tune / Optuna trials.

Capability layout
-----------------
``metrics``
    :func:`~koopman_graph.tuning.fit_history_metrics` — flatten a
    :class:`~koopman_graph.training.FitHistory` into scalar floats for
    trial reporters.
``ray_tune``
    :func:`~koopman_graph.tuning.run_ray_tune` — thin Ray Tune
    :class:`~ray.tune.Tuner` façade (lazy import; requires ``[ray]``).
``search_spaces``
    :func:`~koopman_graph.tuning.example_lr_loguniform_space` and
    :func:`~koopman_graph.tuning.example_lr_latent_dim_space` — *example*
    smoke scaffolds only (not scientific defaults; requires ``[ray]``).

This package is **not** an AutoML product. Search spaces, searchable
hyperparameters, and objective choice remain caller-owned. Helpers support
Ray Tune / Optuna (and similar) integrations; they do not invent
recommended scientific defaults.

Power-user module: import as ``koopman_graph.tuning``. Intentionally
omitted from root ``koopman_graph.__all__``. May import
:mod:`koopman_graph.training` types; ``training`` must not import
``tuning``. Importing this package does not import Ray; Ray is loaded
lazily inside Ray-backed helpers.
"""

from koopman_graph.tuning.metrics import fit_history_metrics
from koopman_graph.tuning.ray_tune import run_ray_tune
from koopman_graph.tuning.search_spaces import (
    example_lr_latent_dim_space,
    example_lr_loguniform_space,
)

__all__ = [
    "example_lr_latent_dim_space",
    "example_lr_loguniform_space",
    "fit_history_metrics",
    "run_ray_tune",
]
