"""Timestamp / increment policy helpers for GraphKoopmanModel.

Shallow peer of :mod:`koopman_graph.model.estimator`. Prefer calling these through
the model façade; peer imports are for power-user / package-internal use.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


def resolve_time_increments(
    *,
    query_times: Sequence[float] | Sequence[Tensor] | None,
    step_deltas: Sequence[float] | Sequence[Tensor] | None,
) -> list[float]:
    """Convert query specification into positive integration increments.

    Parameters
    ----------
    query_times : sequence of float or Tensor, or None
        Absolute query times from the origin (mutually exclusive with
        ``step_deltas``).
    step_deltas : sequence of float or Tensor, or None
        Per-step positive increments (mutually exclusive with ``query_times``).

    Returns
    -------
    list of float
        Strictly positive integration intervals.

    Raises
    ------
    ValueError
        If neither or both inputs are provided, or values are invalid.
    """
    if (query_times is None) == (step_deltas is None):
        msg = "exactly one of query_times or step_deltas must be provided"
        raise ValueError(msg)

    if step_deltas is not None:
        increments = [float(torch.as_tensor(value).item()) for value in step_deltas]
        if not increments or any(value <= 0 for value in increments):
            msg = "step_deltas must be non-empty and strictly positive"
            raise ValueError(msg)
        return increments

    assert query_times is not None
    times = [float(torch.as_tensor(value).item()) for value in query_times]
    if not times or any(value <= 0 for value in times):
        msg = "query_times must be non-empty and strictly positive"
        raise ValueError(msg)
    previous = 0.0
    increments = []
    for value in times:
        if value <= previous:
            msg = "query_times must be strictly increasing"
            raise ValueError(msg)
        increments.append(value - previous)
        previous = value
    return increments


def validate_uniform_discrete_increments(
    *,
    time_step: float,
    increments: Sequence[float],
) -> None:
    """Ensure discrete models only receive uniform time increments.

    Parameters
    ----------
    time_step : float
        Model discrete time step.
    increments : sequence of float
        Proposed integration increments.

    Raises
    ------
    ValueError
        If any increment differs from ``time_step``.
    """
    tolerance = max(1e-6, 1e-4 * time_step)
    for value in increments:
        if abs(value - time_step) > tolerance:
            msg = (
                "discrete dynamics_mode requires uniform increments equal to "
                f"time_step={time_step}; got {value}. Use "
                "dynamics_mode='continuous' for irregular sampling."
            )
            raise ValueError(msg)
