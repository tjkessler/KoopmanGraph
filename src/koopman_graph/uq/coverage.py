"""Named coverage targets for forecast intervals.

:class:`JointCoverageSpec` records the estimand. It does not itself
compute intervals. Shipped conformal methods name
``target="per_node_marginal"``; simultaneous and event targets are
documented but not implemented (``Schlembach2025Conformal``).
``dynamics_mode="stochastic"`` does not change this contract.

This module must not import :mod:`koopman_graph.model` or
:mod:`koopman_graph.operators`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CoverageTarget = Literal[
    "per_node_marginal",
    "simultaneous_node_feature_horizon",
    "event",
]
CoverageBlock = Literal["none", "temporal", "graph"]

DEFAULT_COVERAGE_ALPHA = 0.1
DEFAULT_COVERAGE_TARGET: CoverageTarget = "per_node_marginal"
DEFAULT_COVERAGE_BLOCK: CoverageBlock = "none"
SHIPPED_COVERAGE_TARGET: CoverageTarget = "per_node_marginal"
SHIPPED_COVERAGE_BLOCK: CoverageBlock = "none"

__all__ = [
    "DEFAULT_COVERAGE_ALPHA",
    "DEFAULT_COVERAGE_BLOCK",
    "DEFAULT_COVERAGE_TARGET",
    "JointCoverageSpec",
    "require_shipped_coverage",
]


@dataclass(frozen=True)
class JointCoverageSpec:
    """Named coverage estimand for interval UQ.

    Attributes
    ----------
    target : {"per_node_marginal", "simultaneous_node_feature_horizon", "event"}
        What the interval claims to cover. Default is
        ``per_node_marginal``. Simultaneous node–feature–horizon
        boxes and event coverage are named only.
    alpha : float
        Miscoverage rate in ``(0, 1)``. Default ``0.1``.
    block : {"none", "temporal", "graph"}
        Dependence assumption attached to the claim. Default ``none``.
        Temporal and graph blocks are named only.

    Notes
    -----
    Coverage is assumption-dependent (exchangeability or a named block
    model). This record does not certify joint graph UQ.
    """

    target: CoverageTarget = DEFAULT_COVERAGE_TARGET
    alpha: float = DEFAULT_COVERAGE_ALPHA
    block: CoverageBlock = DEFAULT_COVERAGE_BLOCK

    def __post_init__(self) -> None:
        """Validate the named estimand.

        Raises
        ------
        ValueError
            If ``target``, ``alpha``, or ``block`` is invalid.
        """
        if self.target not in {
            "per_node_marginal",
            "simultaneous_node_feature_horizon",
            "event",
        }:
            msg = (
                "coverage target must be 'per_node_marginal', "
                "'simultaneous_node_feature_horizon', or 'event', "
                f"got {self.target!r}"
            )
            raise ValueError(msg)
        if not 0.0 < float(self.alpha) < 1.0:
            msg = f"coverage alpha must lie in (0, 1), got {self.alpha!r}"
            raise ValueError(msg)
        if self.block not in {"none", "temporal", "graph"}:
            msg = (
                "coverage block must be 'none', 'temporal', or 'graph', "
                f"got {self.block!r}"
            )
            raise ValueError(msg)


def require_shipped_coverage(spec: JointCoverageSpec) -> JointCoverageSpec:
    """Refuse estimands that are named but not implemented.

    Parameters
    ----------
    spec : JointCoverageSpec
        Caller coverage record.

    Returns
    -------
    JointCoverageSpec
        The same spec when it is the shipped marginal / ``block="none"``
        MVP.

    Raises
    ------
    ValueError
        If ``target`` or ``block`` is not the shipped pair.
    """
    if spec.target != SHIPPED_COVERAGE_TARGET:
        msg = (
            "shipped coverage target is 'per_node_marginal'; "
            f"{spec.target!r} is named but not implemented"
        )
        raise ValueError(msg)
    if spec.block != SHIPPED_COVERAGE_BLOCK:
        msg = (
            "shipped coverage block is 'none'; "
            f"{spec.block!r} is named but not implemented"
        )
        raise ValueError(msg)
    return spec
