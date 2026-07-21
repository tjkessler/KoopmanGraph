"""Shared non-private helpers and result types for UQ peers.

Ensemble and latent-Gaussian paths share interval construction utilities
and :class:`PredictionInterval` here so peer modules never import each
other's leading-``_`` symbols or each other's public result types.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor
from torch_geometric.data import Data


@dataclass(frozen=True)
class PredictionInterval:
    """Empirical predictive interval from an ensemble or Gaussian forecast.

    Public result types in this package are frozen dataclasses with attribute
    access (not mapping/dict styles). Collection fields are immutable
    sequences (``tuple``) after construction: callers cannot ``.append`` or
    replace slots in place. Individual ``Data`` objects are **borrowed**, not
    cloned — in-place mutation of node features or topology remains possible.
    Clone explicitly when isolation is required.

    Attributes
    ----------
    mean : tuple of Data
        Ensemble-mean (or predictive-mean) node features per forecast step.
    lower : tuple of Data
        Lower empirical quantile per step (same topology as ``mean``).
    upper : tuple of Data
        Upper empirical quantile per step (same topology as ``mean``).
    level : float
        Nominal central coverage requested at construction (e.g. ``0.9``).
    n_members : int
        Number of ensemble members **or** Monte Carlo latent draws used to
        form the interval (context-dependent).
    """

    mean: tuple[Data, ...]
    lower: tuple[Data, ...]
    upper: tuple[Data, ...]
    level: float
    n_members: int

    def __post_init__(self) -> None:
        """Freeze collection fields as tuples (accept any sequence input).

        Notes
        -----
        Coerces sequence fields to ``tuple`` so the frozen dataclass remains
        hashable and immutable after construction.
        """
        object.__setattr__(self, "mean", tuple(self.mean))
        object.__setattr__(self, "lower", tuple(self.lower))
        object.__setattr__(self, "upper", tuple(self.upper))


def quantile_levels(level: float) -> tuple[float, float]:
    """Map a central coverage level to lower/upper quantile probabilities.

    Parameters
    ----------
    level : float
        Nominal central coverage in ``(0, 1)``.

    Returns
    -------
    tuple of float
        ``(lower_quantile, upper_quantile)`` probabilities in ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``level`` is not strictly inside ``(0, 1)``.
    """
    if not 0.0 < level < 1.0:
        msg = f"level must lie in (0, 1); got {level}"
        raise ValueError(msg)
    alpha = 1.0 - level
    lower_q = alpha / 2.0
    upper_q = 1.0 - lower_q
    return lower_q, upper_q


def snapshot_with_features(template: Data, features: Tensor) -> Data:
    """Clone topology from ``template`` and replace node features.

    Parameters
    ----------
    template : Data
        Snapshot supplying ``edge_index`` and optional ``edge_weight``.
    features : Tensor
        Replacement node-feature matrix.

    Returns
    -------
    Data
        New snapshot with ``features`` and the template topology.
    """
    fields: dict[str, Tensor] = {
        "x": features,
        "edge_index": template.edge_index,
    }
    edge_weight = getattr(template, "edge_weight", None)
    if edge_weight is not None:
        fields["edge_weight"] = edge_weight
    return Data(**fields)
