"""0-dimensional persistence observables (core, no TDA extra required).

Computes birth–death pairs of connected components on a pairwise distance
filtration via union-find. Optional ``[tda]`` libraries are not required.
This is not a persistent-homology library replacement.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class PersistenceDiagram:
    """0-dimensional persistence pairs.

    Attributes
    ----------
    pairs : Tensor
        Shape ``(k, 2)`` with columns ``(birth, death)``. Infinite death is
        replaced by the filtration diameter.
    betti_0 : int
        Number of components at filtration 0 (equals ``num_points``).
    """

    pairs: Tensor
    betti_0: int


def pairwise_distance_filtration(points: Tensor) -> Tensor:
    """Return pairwise distances.

    Parameters
    ----------
    points : Tensor
        Point cloud ``(n, dim)``.

    Returns
    -------
    Tensor
        Square distance matrix.
    """
    if points.ndim != 2:
        raise ValueError(f"points must have shape (n, dim), got {tuple(points.shape)}")
    return torch.cdist(points, points)


def _union_find_parent(parent: list[int], index: int) -> int:
    """Path-compressed find.

    Parameters
    ----------
    parent : list of int
        Parent table mutated in place.
    index : int
        Node to resolve.

    Returns
    -------
    int
        Root of ``index``.
    """
    while parent[index] != index:
        parent[index] = parent[parent[index]]
        index = parent[index]
    return index


def persistence_diagram_0d(points: Tensor) -> PersistenceDiagram:
    """Compute 0-dimensional persistence via Kruskal / union-find.

    Parameters
    ----------
    points : Tensor
        Point cloud ``(n, dim)``.

    Returns
    -------
    PersistenceDiagram
        Birth–death pairs (births are 0 for this union-find).
    """
    distances = pairwise_distance_filtration(points)
    num_points = int(points.shape[0])
    parent = list(range(num_points))
    pairs: list[list[float]] = []
    triu = torch.triu_indices(num_points, num_points, offset=1)
    order = distances[triu[0], triu[1]].argsort()
    for loc in order.tolist():
        i = int(triu[0, loc])
        j = int(triu[1, loc])
        pi, pj = _union_find_parent(parent, i), _union_find_parent(parent, j)
        if pi == pj:
            continue
        parent[pi] = pj
        pairs.append([0.0, float(distances[i, j])])
        if len(pairs) == num_points - 1:
            break
    diameter = float(distances.max()) if num_points else 0.0
    if not pairs:
        tensor = torch.zeros(0, 2, dtype=points.dtype)
    else:
        tensor = torch.tensor(pairs, dtype=points.dtype)
        tensor[:, 1] = tensor[:, 1].clamp(max=diameter)
    return PersistenceDiagram(pairs=tensor, betti_0=num_points)


def betti_curve(diagram: PersistenceDiagram, thresholds: Tensor) -> Tensor:
    """Count components still alive at each threshold.

    Parameters
    ----------
    diagram : PersistenceDiagram
        0-dimensional diagram.
    thresholds : Tensor
        Filtration values.

    Returns
    -------
    Tensor
        Betti-0 counts aligned to ``thresholds``.
    """
    if diagram.pairs.numel() == 0:
        return torch.full_like(thresholds, float(diagram.betti_0))
    deaths = diagram.pairs[:, 1]
    alive = (deaths.unsqueeze(0) > thresholds.unsqueeze(1)).sum(dim=1)
    return alive.to(dtype=thresholds.dtype) + 1.0
