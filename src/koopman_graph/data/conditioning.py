"""Per-snapshot time, parameter, and control records.

A :class:`ConditioningContext` is a data record for one timestep. It does
not select a factory operator. ``koopman="switched"`` and
``koopman="mixture"`` remain latent-gated or piecewise LTI maps, not a
parameter interpolant :math:`K(\\mu)` (``Macesic2018Nonautonomous``).

This module must not import :mod:`koopman_graph.nn`.
``koopman="parametric"`` evaluates :math:`K(\\mu)` from the stored
trajectory; this record does not select that factory kind.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from koopman_graph.data.calendar import diurnal_phase_index
from koopman_graph.data.containers import GraphSnapshotSequence

__all__ = [
    "ConditioningContext",
    "conditioning_at",
]


@dataclass(frozen=True, eq=False)
class ConditioningContext:
    """Exogenous conditioning at one homogeneous snapshot.

    Attributes
    ----------
    time : float or None
        Timestamp at this index in the caller's time unit, or ``None``
        when the sequence has no timestamps.
    parameters : Tensor or None
        Regime coordinates :math:`\\mu` with shape ``(d_mu,)``. Units are
        caller-defined (dimensionless if unspecified). ``None`` when
        the sequence has no
        :attr:`~koopman_graph.data.GraphSnapshotSequence.parameter_trajectory`.
    control : Tensor or None
        Control driving the transition from this snapshot, same layout as
        :meth:`~koopman_graph.data.GraphSnapshotSequence.control_at`
        (global ``(C,)`` or per-node ``(N, C)``). ``None`` when controls
        are absent.
    phase_index : int or None
        Optional diurnal or switched-by-clock index. Default
        :func:`conditioning_at` leaves this ``None``. Pass ``period``
        and ``num_phases`` to fill it from timestamps via
        :func:`~koopman_graph.data.diurnal_phase_index`.

    Notes
    -----
    Equality is disabled because tensor fields do not define value
    equality. This record does not itself evaluate :math:`K(\\mu)`;
    ``koopman="parametric"`` does. Discrete uniform-:math:`\\Delta t`
    validation is unchanged.

    References
    ----------
    Maćešić, S., Črnjarić-Žic, N. and Mezić, I. (2018). Koopman operator
    family spectrum for nonautonomous systems. *SIAM Journal on Applied
    Dynamical Systems* 17:2478–2515. doi:10.1137/17M1133610
    (``Macesic2018Nonautonomous``).
    """

    time: float | None
    parameters: Tensor | None
    control: Tensor | None
    phase_index: int | None = None

    def __post_init__(self) -> None:
        """Validate optional tensor ranks and scalar fields.

        Raises
        ------
        ValueError
            If a present field has an invalid rank, width, or value.
        """
        if self.time is not None and not math.isfinite(self.time):
            msg = f"time must be a finite float, got {self.time!r}"
            raise ValueError(msg)
        if self.parameters is not None:
            if self.parameters.ndim != 1 or int(self.parameters.shape[0]) < 1:
                msg = (
                    "parameters must have shape (d_mu,) with d_mu >= 1, "
                    f"got {tuple(self.parameters.shape)}"
                )
                raise ValueError(msg)
            if self.parameters.is_floating_point() and not torch.all(
                torch.isfinite(self.parameters)
            ):
                msg = "parameters must be finite"
                raise ValueError(msg)
        if self.control is not None and self.control.ndim not in (1, 2):
            msg = (
                "control must have shape (C,) or (N, C), "
                f"got {tuple(self.control.shape)}"
            )
            raise ValueError(msg)
        if self.phase_index is not None and (
            type(self.phase_index) is not int or self.phase_index < 0
        ):
            msg = (
                "phase_index must be a non-negative int or None, "
                f"got {self.phase_index!r}"
            )
            raise ValueError(msg)


def conditioning_at(
    sequence: GraphSnapshotSequence,
    index: int,
    *,
    period: float | None = None,
    num_phases: int | None = None,
) -> ConditioningContext:
    """Return the exogenous record at ``index``.

    Parameters
    ----------
    sequence : GraphSnapshotSequence
        Homogeneous snapshot sequence.
    index : int
        Timestep in ``[0, num_timesteps)``.
    period : float or None, optional
        Cycle length in the timestamp unit. Required with ``num_phases``
        to fill :attr:`ConditioningContext.phase_index`.
    num_phases : int or None, optional
        Number of phase bins. Required with ``period``.

    Returns
    -------
    ConditioningContext
        Time, :math:`\\mu`, and control at ``index``. ``phase_index`` is
        ``None`` unless ``period`` and ``num_phases`` are set. Missing
        sequence fields yield ``None`` payload slots; an all-``None``
        record is valid.

    Raises
    ------
    TypeError
        If ``sequence`` is not a homogeneous
        :class:`~koopman_graph.data.GraphSnapshotSequence`.
    ValueError
        If ``index`` is out of range, only one of ``period`` /
        ``num_phases`` is set, or phase bins are requested without
        timestamps.
    """
    if type(sequence) is not GraphSnapshotSequence:
        msg = (
            "conditioning_at requires GraphSnapshotSequence, "
            f"got {type(sequence).__name__}"
        )
        raise TypeError(msg)
    num_timesteps = sequence.num_timesteps
    if index < 0 or index >= num_timesteps:
        msg = (
            f"conditioning index {index} is out of range for {num_timesteps} timesteps"
        )
        raise ValueError(msg)
    time = None
    if sequence.timestamps is not None:
        time = float(sequence.timestamps[index].item())
    parameters = None
    if sequence.parameter_trajectory is not None:
        parameters = sequence.parameter_trajectory[index]
    control = None
    if sequence.has_controls:
        control = sequence.control_at(index)
    phase_index = None
    if (period is None) != (num_phases is None):
        msg = "period and num_phases must be supplied together"
        raise ValueError(msg)
    if period is not None and num_phases is not None:
        if sequence.timestamps is None:
            msg = "phase_index requires sequence timestamps"
            raise ValueError(msg)
        bins = diurnal_phase_index(
            sequence.timestamps[index : index + 1],
            period=period,
            num_phases=num_phases,
        )
        phase_index = int(bins[0].item())
    return ConditioningContext(
        time=time,
        parameters=parameters,
        control=control,
        phase_index=phase_index,
    )
