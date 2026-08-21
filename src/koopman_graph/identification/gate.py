"""Residual-aware candidate selection for finite-dictionary ResDMD.

``select_resdmd_gated`` compares already-scored dictionaries. It does
**not** fit an operator, does **not** run
:func:`~koopman_graph.analysis.resdmd`, and is **not** a certified
infinite-dimensional residual bound. Callers supply a train one-step
mean squared error (MSE) and a max Colbrook–Townsend residual per
candidate.

This module must not import :mod:`koopman_graph.training` or
:mod:`koopman_graph.model`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "DEFAULT_RESDMD_GATE_TOLERANCE",
    "ResDMDGateCandidate",
    "ResDMDGateResult",
    "select_resdmd_gated",
]

# Matches :func:`~koopman_graph.analysis.resdmd` default ``tolerance``.
DEFAULT_RESDMD_GATE_TOLERANCE = 1e-2


@dataclass(frozen=True)
class ResDMDGateCandidate:
    """One dictionary scored by train MSE and max ResDMD residual.

    ``mse`` is a dimensionless one-step mean squared error on that
    dictionary, not a physical unit.

    Attributes
    ----------
    name : str
        Non-empty candidate label.
    mse : float
        Train one-step mean squared error (non-negative, finite).
    residual_max : float
        Maximum finite-dictionary ResDMD residual (non-negative, finite).
    """

    name: str
    mse: float
    residual_max: float

    def __post_init__(self) -> None:
        """Validate name and finite non-negative scalars.

        Raises
        ------
        ValueError
            If ``name`` is empty or a scalar is invalid.
        """
        if not isinstance(self.name, str) or self.name == "":
            msg = "name must be a non-empty str"
            raise ValueError(msg)
        for field, value in (("mse", self.mse), ("residual_max", self.residual_max)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                msg = f"{field} must be a finite float, got {type(value).__name__}"
                raise ValueError(msg)
            if not math.isfinite(float(value)) or float(value) < 0.0:
                msg = f"{field} must be a finite non-negative float, got {value!r}"
                raise ValueError(msg)


@dataclass(frozen=True)
class ResDMDGateResult:
    """Outcome of :func:`select_resdmd_gated`.

    ``rejected_alternatives`` lists names dropped for exceeding the
    residual tolerance when ``gated`` is True. Non-winners that passed
    the residual check are not listed.

    Attributes
    ----------
    selected : str
        Name of the retained candidate.
    rejected_alternatives : tuple of str
        Polluted names dropped by the gate (empty when ``gated`` is
        False).
    residual_tolerance : float
        Cutoff applied to ``residual_max``.
    gated : bool
        Whether residual filtering ran.
    """

    selected: str
    rejected_alternatives: tuple[str, ...]
    residual_tolerance: float
    gated: bool

    def __post_init__(self) -> None:
        """Validate selected name, rejected list, and tolerance.

        Raises
        ------
        ValueError
            If fields are malformed.
        """
        if not isinstance(self.selected, str) or self.selected == "":
            msg = "selected must be a non-empty str"
            raise ValueError(msg)
        if not isinstance(self.rejected_alternatives, tuple):
            msg = "rejected_alternatives must be a tuple of str"
            raise ValueError(msg)
        if any(
            not isinstance(name, str) or name == ""
            for name in self.rejected_alternatives
        ):
            msg = "rejected_alternatives entries must be non-empty strings"
            raise ValueError(msg)
        if isinstance(self.residual_tolerance, bool) or not isinstance(
            self.residual_tolerance, (int, float)
        ):
            msg = (
                "residual_tolerance must be a finite float, got "
                f"{type(self.residual_tolerance).__name__}"
            )
            raise ValueError(msg)
        if (
            not math.isfinite(float(self.residual_tolerance))
            or float(self.residual_tolerance) < 0.0
        ):
            msg = (
                "residual_tolerance must be a finite non-negative float, "
                f"got {self.residual_tolerance!r}"
            )
            raise ValueError(msg)
        if type(self.gated) is not bool:
            msg = f"gated must be a bool, got {type(self.gated).__name__}"
            raise ValueError(msg)


def select_resdmd_gated(
    candidates: Sequence[ResDMDGateCandidate],
    *,
    residual_tolerance: float = DEFAULT_RESDMD_GATE_TOLERANCE,
    gate_resdmd: bool = True,
) -> ResDMDGateResult:
    """Pick a candidate by train MSE, optionally dropping polluted dictionaries.

    When ``gate_resdmd`` is False, the lowest ``mse`` wins (RMSE-only
    rule on these scalars). When True, candidates with
    ``residual_max > residual_tolerance`` are dropped first. This is
    finite-dictionary selection, not a certified ResDMD spectral
    measure.

    Parameters
    ----------
    candidates : sequence of ResDMDGateCandidate
        Named scores. Names must be unique.
    residual_tolerance : float, optional
        Max-residual cutoff. Default ``1e-2`` (same as
        :func:`~koopman_graph.analysis.resdmd`).
    gate_resdmd : bool, optional
        Apply residual filtering. Default is ``True``.

    Returns
    -------
    ResDMDGateResult
        Selected name and any residual-rejected alternatives.

    Raises
    ------
    TypeError
        If ``gate_resdmd`` is not a ``bool``.
    ValueError
        If ``candidates`` is empty, names collide, the tolerance is
        invalid, or every candidate is polluted when gating.
    """
    if type(gate_resdmd) is not bool:
        msg = f"gate_resdmd must be a bool, got {type(gate_resdmd).__name__}"
        raise TypeError(msg)
    if isinstance(residual_tolerance, bool) or not isinstance(
        residual_tolerance, (int, float)
    ):
        msg = (
            "residual_tolerance must be a finite float, got "
            f"{type(residual_tolerance).__name__}"
        )
        raise ValueError(msg)
    if not math.isfinite(float(residual_tolerance)) or float(residual_tolerance) < 0.0:
        msg = (
            "residual_tolerance must be a finite non-negative float, "
            f"got {residual_tolerance!r}"
        )
        raise ValueError(msg)
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        msg = "candidates must be a sequence of ResDMDGateCandidate"
        raise ValueError(msg)
    pool = tuple(candidates)
    if not pool:
        msg = "select_resdmd_gated requires at least one candidate"
        raise ValueError(msg)
    if any(not isinstance(item, ResDMDGateCandidate) for item in pool):
        msg = "candidates entries must be ResDMDGateCandidate"
        raise ValueError(msg)
    names = [item.name for item in pool]
    if len(set(names)) != len(names):
        msg = "candidate names must be unique"
        raise ValueError(msg)
    rejected: tuple[str, ...] = ()
    eligible = pool
    if gate_resdmd:
        rejected = tuple(
            item.name for item in pool if item.residual_max > residual_tolerance
        )
        eligible = tuple(
            item for item in pool if item.residual_max <= residual_tolerance
        )
        if not eligible:
            msg = (
                "ResDMD gate rejected every candidate "
                f"(residual_max > {residual_tolerance:g})"
            )
            raise ValueError(msg)
    selected = min(eligible, key=lambda item: (item.mse, item.name))
    return ResDMDGateResult(
        selected=selected.name,
        rejected_alternatives=rejected,
        residual_tolerance=float(residual_tolerance),
        gated=gate_resdmd,
    )
