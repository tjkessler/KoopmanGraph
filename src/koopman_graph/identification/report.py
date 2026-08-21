"""Frozen identification report and placeholder metric blocks.

Field groups follow the public identification record: reconstruction,
one-step, rollout, closure, invariance, spectral reliability, and
stability, plus optional rank metadata. Values default to ``None`` /
empty until solvers and diagnostics populate them. This is **not** a
Haseli–Cortés invariance certificate, not a ResDMD residual bound, and
not a stability certificate.

This module must not import :mod:`koopman_graph.training` or
:mod:`koopman_graph.model`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

__all__ = [
    "IdentificationReport",
    "InvarianceBlock",
    "MetricBlock",
    "SpectralReliabilityBlock",
    "StabilityBlock",
]


def _optional_finite_float(name: str, value: float | None) -> None:
    """Reject non-finite optional floats.

    Parameters
    ----------
    name : str
        Field name for the error message.
    value : float or None
        Optional scalar.

    Raises
    ------
    ValueError
        If ``value`` is not ``None`` and is NaN or infinite.
    """
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f"{name} must be a finite float or None, got {type(value).__name__}"
        raise ValueError(msg)
    if not math.isfinite(float(value)):
        msg = f"{name} must be a finite float or None, got {value!r}"
        raise ValueError(msg)


def _optional_nonneg_int(name: str, value: int | None) -> None:
    """Reject negative optional integers.

    Parameters
    ----------
    name : str
        Field name for the error message.
    value : int or None
        Optional count.

    Raises
    ------
    ValueError
        If ``value`` is not ``None`` and is not a non-negative ``int``.
    """
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        msg = f"{name} must be a non-negative int or None, got {value!r}"
        raise ValueError(msg)


@dataclass(frozen=True)
class MetricBlock:
    """Named error summary for one identification arm.

    ``mse`` is a dimensionless mean squared error in the coordinates of
    that arm (latent or decoded), not a physical unit. ``None`` means
    the arm was not populated.

    Attributes
    ----------
    mse : float or None
        Mean squared error, or ``None`` when unset.
    n_samples : int or None
        Number of terms in the mean, or ``None`` when unset.
    """

    mse: float | None = None
    n_samples: int | None = None

    def __post_init__(self) -> None:
        """Validate optional MSE and sample count.

        Raises
        ------
        ValueError
            If ``mse`` is non-finite or ``n_samples`` is negative.
        """
        _optional_finite_float("mse", self.mse)
        _optional_nonneg_int("n_samples", self.n_samples)


@dataclass(frozen=True)
class InvarianceBlock:
    """Placeholder for subspace-invariance proximity on a fit report.

    ``leakage`` is the slot for finite-sample :math:`\\eta`. The
    populated report type is
    :class:`~koopman_graph.identification.SubspaceInvarianceReport`
    from :func:`~koopman_graph.identification.subspace_invariance_report`
    (evaluate / model helper). Closed-form ``fit`` still leaves this
    block unset. It is **not** a Haseli–Cortés certificate.

    Attributes
    ----------
    leakage : float or None
        Non-negative proximity scalar, or ``None`` when unset.
    """

    leakage: float | None = None

    def __post_init__(self) -> None:
        """Validate optional leakage.

        Raises
        ------
        ValueError
            If ``leakage`` is non-finite or negative.
        """
        _optional_finite_float("leakage", self.leakage)
        if self.leakage is not None and self.leakage < 0.0:
            msg = f"leakage must be non-negative, got {self.leakage!r}"
            raise ValueError(msg)


@dataclass(frozen=True)
class SpectralReliabilityBlock:
    """Finite-dictionary residual / pollution flags.

    Filled when identification ``gate_resdmd=True``. Not a certified
    infinite-dimensional residual bound and not
    :func:`~koopman_graph.analysis.spectral_residuals`.

    Attributes
    ----------
    residual_max : float or None
        Maximum dictionary residual, or ``None`` when unset.
    polluted : bool or None
        ``True`` when a later gate marks the spectrum as polluted.
    """

    residual_max: float | None = None
    polluted: bool | None = None

    def __post_init__(self) -> None:
        """Validate residual and pollution flag.

        Raises
        ------
        ValueError
            If ``residual_max`` is non-finite or ``polluted`` is not a
            ``bool`` or ``None``.
        """
        _optional_finite_float("residual_max", self.residual_max)
        if self.polluted is not None and type(self.polluted) is not bool:
            msg = f"polluted must be a bool or None, got {type(self.polluted).__name__}"
            raise ValueError(msg)


@dataclass(frozen=True)
class StabilityBlock:
    """Placeholder for a spectral-radius snapshot of identified ``K``.

    Not a joint Gershgorin certificate and not
    :meth:`~koopman_graph.operators.KoopmanOperator.stability_certificate`.

    Attributes
    ----------
    spectral_radius : float or None
        :math:`\\rho(K)` when populated, dimensionless for discrete maps.
    """

    spectral_radius: float | None = None

    def __post_init__(self) -> None:
        """Validate optional spectral radius.

        Raises
        ------
        ValueError
            If ``spectral_radius`` is non-finite or negative.
        """
        _optional_finite_float("spectral_radius", self.spectral_radius)
        if self.spectral_radius is not None and self.spectral_radius < 0.0:
            msg = f"spectral_radius must be non-negative, got {self.spectral_radius!r}"
            raise ValueError(msg)


@dataclass(frozen=True)
class IdentificationReport:
    """Structured identification record with named metric groups.

    Empty reports are valid. Closed-form ``fit(..., identification=...)``
    fills ``one_step``, short latent ``rollout``, and
    ``stability.spectral_radius``. ``spectral`` fills only when
    ``gate_resdmd=True``. Invariance, reconstruction, and closure stay
    unset. Evaluate-time :math:`\\eta` lives on
    :class:`~koopman_graph.metrics.EvaluationResult.invariance`, not here.

    Attributes
    ----------
    reconstruction : MetricBlock
        Decode / reconstruction error.
    one_step : MetricBlock
        One-step latent or decoded error.
    rollout : MetricBlock
        Multi-step rollout error.
    closure : MetricBlock
        Topology / graph-state closure error when applicable.
    invariance : InvarianceBlock
        Subspace leakage slot.
    spectral : SpectralReliabilityBlock
        Finite-dictionary residual / pollution slot.
    stability : StabilityBlock
        Spectral-radius slot.
    selected_rank : int or None
        Latent rank chosen by a later selector, or ``None``.
    rejected_alternatives : tuple of str
        Names of candidates a later gate discarded.
    """

    reconstruction: MetricBlock = field(default_factory=MetricBlock)
    one_step: MetricBlock = field(default_factory=MetricBlock)
    rollout: MetricBlock = field(default_factory=MetricBlock)
    closure: MetricBlock = field(default_factory=MetricBlock)
    invariance: InvarianceBlock = field(default_factory=InvarianceBlock)
    spectral: SpectralReliabilityBlock = field(default_factory=SpectralReliabilityBlock)
    stability: StabilityBlock = field(default_factory=StabilityBlock)
    selected_rank: int | None = None
    rejected_alternatives: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate rank metadata and rejected-name sequence.

        Raises
        ------
        ValueError
            If ``selected_rank`` is negative or
            ``rejected_alternatives`` is not a string sequence.
        """
        _optional_nonneg_int("selected_rank", self.selected_rank)
        if not isinstance(self.rejected_alternatives, Sequence) or isinstance(
            self.rejected_alternatives, (str, bytes)
        ):
            msg = "rejected_alternatives must be a sequence of strings"
            raise ValueError(msg)
        names = tuple(self.rejected_alternatives)
        if any(not isinstance(name, str) or name == "" for name in names):
            msg = "rejected_alternatives entries must be non-empty strings"
            raise ValueError(msg)
        if names != self.rejected_alternatives:
            object.__setattr__(self, "rejected_alternatives", names)
