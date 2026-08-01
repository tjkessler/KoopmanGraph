"""Shared non-private helpers and result types for UQ peers.

Ensemble and latent-Gaussian paths share interval construction utilities
and :class:`PredictionInterval` here so peer modules never import each
other's leading-``_`` symbols or each other's public result types.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from torch import Tensor
from torch_geometric.data import Data, HeteroData

from koopman_graph.data.hetero_layout import (
    snapshot_num_nodes_dict,
    unstack_typed_features,
)

SnapshotLike = Data | HeteroData


@dataclass(frozen=True)
class PredictionInterval:
    """Empirical predictive interval from an ensemble or Gaussian forecast.

    Public result types in this package are frozen dataclasses with attribute
    access (not mapping/dict styles). Collection fields are immutable
    sequences (``tuple``) after construction: callers cannot ``.append`` or
    replace slots in place. Individual ``Data`` / ``HeteroData`` objects are
    **borrowed**, not cloned — in-place mutation of node features or
    topology remains possible. Clone explicitly when isolation is required.

    Attributes
    ----------
    mean : tuple of Data or HeteroData
        Ensemble-mean (or predictive-mean) node features per forecast step.
        Homogeneous paths use ``Data``; hetero conformal paths use
        ``HeteroData``.
    lower : tuple of Data or HeteroData
        Lower empirical quantile per step (same topology as ``mean``).
    upper : tuple of Data or HeteroData
        Upper empirical quantile per step (same topology as ``mean``).
    level : float
        Nominal central coverage requested at construction (e.g. ``0.9``).
    n_members : int
        Number of ensemble members **or** Monte Carlo latent draws used to
        form the interval (context-dependent).
    """

    mean: tuple[SnapshotLike, ...]
    lower: tuple[SnapshotLike, ...]
    upper: tuple[SnapshotLike, ...]
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


def hetero_snapshot_with_features(
    template: HeteroData,
    features: Tensor,
    node_type_names: Sequence[str],
) -> HeteroData:
    """Clone a hetero snapshot and replace stacked node features.

    Parameters
    ----------
    template : HeteroData
        Snapshot supplying per-type / per-relation topology.
    features : Tensor
        Stacked replacement features with shape ``(Σ_τ N_τ, F)`` in
        ``node_type_names`` order (shared trailing width ``F``).
    node_type_names : sequence of str
        Ordered node-type names defining the stacking order.

    Returns
    -------
    HeteroData
        Cloned snapshot with unstacked ``x`` per node type and the template
        relation banks.

    Raises
    ------
    ValueError
        If ``features`` row count disagrees with the template schema or
        widths cannot be unstacked.
    """
    counts = snapshot_num_nodes_dict(template, node_type_names)
    unstacked = unstack_typed_features(features, node_type_names, counts)
    out = template.clone()
    for name, block in unstacked.items():
        out[name].x = block
    return out
