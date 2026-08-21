"""Time-of-day control and phase-index recipes.

These helpers turn caller timestamps into additive / bilinear control
features or integer phase bins. They are not a calendar serializer and
do not change discrete uniform-:math:`\\Delta t` validation. This module
must not import :mod:`koopman_graph.nn` or :mod:`koopman_graph.model`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor

DEFAULT_HARMONICS = 1
MAX_HARMONICS = 8

__all__ = [
    "DEFAULT_HARMONICS",
    "MAX_HARMONICS",
    "diurnal_control_features",
    "diurnal_phase_index",
]


def _as_1d_timestamps(timestamps: Sequence[float] | Tensor) -> Tensor:
    """Coerce timestamps to a finite 1-D float tensor.

    Parameters
    ----------
    timestamps : sequence of float or Tensor
        Caller times in one shared unit.

    Returns
    -------
    Tensor
        Shape ``(T,)`` with ``T >= 1``.

    Raises
    ------
    ValueError
        If the input is empty, not 1-D after a scalar wrap, or non-finite.
    """
    tensor = torch.as_tensor(timestamps)
    if tensor.ndim == 0:
        tensor = tensor.reshape(1)
    if tensor.ndim != 1:
        msg = (
            "timestamps must be a scalar or 1-D sequence, "
            f"got shape {tuple(tensor.shape)}"
        )
        raise ValueError(msg)
    if tensor.numel() < 1:
        msg = "timestamps must contain at least one value"
        raise ValueError(msg)
    values = tensor.to(dtype=torch.float64)
    if not bool(torch.all(torch.isfinite(values))):
        msg = "timestamps must be finite"
        raise ValueError(msg)
    return values


def _require_positive_period(period: float) -> float:
    """Validate the shared period.

    Parameters
    ----------
    period : float
        Cycle length in the same unit as the timestamps.

    Returns
    -------
    float
        The accepted period.

    Raises
    ------
    ValueError
        If ``period`` is not a positive finite float.
    """
    scale = float(period)
    if not math.isfinite(scale) or scale <= 0.0:
        msg = f"period must be a positive finite float, got {period!r}"
        raise ValueError(msg)
    return scale


def diurnal_control_features(
    timestamps: Sequence[float] | Tensor,
    *,
    period: float,
    harmonics: int = DEFAULT_HARMONICS,
) -> Tensor:
    """Return Fourier time-of-day features for additive / bilinear control.

    For harmonic :math:`k=1,\\ldots,H` the columns are
    :math:`\\sin(2\\pi k t / T)` and :math:`\\cos(2\\pi k t / T)` in the
    same time unit as ``timestamps`` and ``period``. Pass the result as
    ``control_inputs`` on a homogeneous sequence. This is a documented
    recipe, not a native calendar field, and it does not relax discrete
    uniform-:math:`\\Delta t` validation.

    Parameters
    ----------
    timestamps : sequence of float or Tensor
        Times with shape ``(T,)`` (or a scalar) in the caller unit.
    period : float
        Cycle length :math:`T` in the same unit (for example 24 hours
        when timestamps are hours).
    harmonics : int, optional
        Number of Fourier pairs. Default is 1 (one sine and one cosine).
        Maximum is 8.

    Returns
    -------
    Tensor
        Features with shape ``(T, 2 H)`` and dtype ``float32``.

    Raises
    ------
    ValueError
        If ``timestamps`` or ``period`` are invalid, or ``harmonics`` is
        outside ``[1, 8]``.
    """
    times = _as_1d_timestamps(timestamps)
    cycle = _require_positive_period(period)
    if type(harmonics) is not int or not 1 <= harmonics <= MAX_HARMONICS:
        msg = f"harmonics must be an int in [1, {MAX_HARMONICS}], got {harmonics!r}"
        raise ValueError(msg)
    angle = (2.0 * math.pi) * times / cycle
    columns: list[Tensor] = []
    for order in range(1, harmonics + 1):
        scaled = float(order) * angle
        columns.append(torch.sin(scaled))
        columns.append(torch.cos(scaled))
    return torch.stack(columns, dim=-1).to(dtype=torch.float32)


def diurnal_phase_index(
    timestamps: Sequence[float] | Tensor,
    *,
    period: float,
    num_phases: int,
) -> Tensor:
    """Return integer phase bins for switched-by-clock maps.

    Each time is reduced modulo ``period`` and assigned
    :math:`\\lfloor n \\, \\mathrm{frac}(t / T) \\rfloor` in
    ``[0, num_phases)``. Use the bins as ``phase_index`` on
    :class:`~koopman_graph.operators.SwitchedKoopmanOperator`. Discrete
    uniform-:math:`\\Delta t` validation is unchanged.

    Parameters
    ----------
    timestamps : sequence of float or Tensor
        Times with shape ``(T,)`` (or a scalar) in the caller unit.
    period : float
        Cycle length :math:`T` in the same unit.
    num_phases : int
        Number of bins (typically the switched ``num_modes``).

    Returns
    -------
    Tensor
        Integer indices with shape ``(T,)`` and dtype ``int64``.

    Raises
    ------
    ValueError
        If ``timestamps`` or ``period`` are invalid, or ``num_phases``
        is not a positive int.
    """
    times = _as_1d_timestamps(timestamps)
    cycle = _require_positive_period(period)
    if type(num_phases) is not int or num_phases < 1:
        msg = f"num_phases must be a positive int, got {num_phases!r}"
        raise ValueError(msg)
    fraction = torch.remainder(times, cycle) / cycle
    raw = torch.floor(float(num_phases) * fraction)
    clipped = torch.clamp(raw, min=0, max=num_phases - 1)
    return clipped.to(dtype=torch.int64)
