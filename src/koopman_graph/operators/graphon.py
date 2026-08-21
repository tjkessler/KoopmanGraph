r"""Graphon sampling and dense-kernel estimation for transfer experiments.

Samples a dense adjacency from a kernel :math:`W:[0,1]^2\to[0,1]` at
:math:`N` uniform nodes, or estimates a constant / product / low-rank
kernel from aligned graphs. Theory bounds are cited, not proved
in-repo. Sparse-graph graphon limits are out of scope
(``LovaszSzegedy2006``; ``Ruiz2023Transferability``).

References
----------
Lovász, L. and Szegedy, B. (2006). Limits of dense graph sequences.
*Journal of Combinatorial Theory, Series B* 96:933–957.
doi:10.1016/j.jctb.2006.05.001 (``LovaszSzegedy2006``).

Ruiz, L., Chamon, L. F. O. and Ribeiro, A. (2023). Transferability
properties of graph neural networks. *IEEE Transactions on Signal
Processing* 71:3474–3489. doi:10.1109/TSP.2023.3297848
(``Ruiz2023Transferability``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor
from torch_geometric.data import Data

__all__ = [
    "MAX_GRAPHON_NODES",
    "GraphonEstimate",
    "GraphonKernelFamily",
    "estimate_graphon",
    "sample_graphon_adjacency",
]

GraphonKernelFamily = Literal["constant", "product", "low_rank"]
_FAMILIES = frozenset({"constant", "product", "low_rank"})
MAX_GRAPHON_NODES = 256
"""Dense teaching ceiling on aligned node count for ``estimate_graphon``."""
_FAMILIES_DOC = '{"constant", "product", "low_rank"}'


def sample_graphon_adjacency(
    num_nodes: int,
    *,
    kernel: str = "constant",
    density: float = 0.3,
    generator: torch.Generator | None = None,
    positions: Tensor | None = None,
) -> Tensor:
    r"""Sample a symmetric 0/1 adjacency from a simple graphon.

    Parameters
    ----------
    num_nodes : int
        Node count :math:`N`.
    kernel : {"constant", "product"}, optional
        ``constant`` uses edge probability ``density``; ``product`` uses
        :math:`W(u,v)=u v` at uniform (or supplied) latent positions.
    density : float, optional
        Edge probability for the constant graphon. Dimensionless in
        ``[0, 1]``. Default ``0.3``.
    generator : torch.Generator or None, optional
        Optional RNG.
    positions : Tensor or None, optional
        Latent coordinates ``(N,)`` in ``[0, 1]``. Drawn uniformly when
        omitted. Used by the product kernel.

    Returns
    -------
    Tensor
        Integer ``edge_index`` with shape ``(2, E)`` (undirected, both
        orientations).

    Raises
    ------
    ValueError
        If ``num_nodes < 2``, ``density`` is outside ``[0, 1]``,
        ``positions`` has the wrong shape or lies outside ``[0, 1]``,
        or ``kernel`` is unknown.

    Notes
    -----
    Dense-graph limit viewpoint of Lovász and Szegedy (2006;
    ``LovaszSzegedy2006``). Transferability of filters / GNNs across
    sizes is not automatic on arbitrary sparse graphs
    (``Ruiz2023Transferability``). Bounds are cited, not proved here.
    """
    if num_nodes < 2:
        raise ValueError(f"num_nodes must be >= 2, got {num_nodes}")
    if kernel == "constant" and not 0.0 <= float(density) <= 1.0:
        raise ValueError(f"density must lie in [0, 1], got {density}")
    if positions is None:
        coords = torch.rand(num_nodes, generator=generator)
    else:
        if positions.ndim != 1 or int(positions.shape[0]) != num_nodes:
            msg = (
                f"positions must have shape (num_nodes,), got {tuple(positions.shape)}"
            )
            raise ValueError(msg)
        coords = positions.to(dtype=torch.float32)
        if float(coords.min()) < 0.0 or float(coords.max()) > 1.0:
            raise ValueError("positions must lie in [0, 1]")
    if kernel == "constant":
        probs = torch.full((num_nodes, num_nodes), float(density))
    elif kernel == "product":
        probs = coords.unsqueeze(1) * coords.unsqueeze(0)
    else:
        raise ValueError(f"unknown graphon kernel {kernel!r}")
    probs = torch.triu(probs, diagonal=1)
    samples = torch.rand((num_nodes, num_nodes), generator=generator) < probs
    src, dst = samples.nonzero(as_tuple=True)
    return torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)


def _graph_num_nodes(graph: Data) -> int:
    """Return node count for a homogeneous snapshot.

    Parameters
    ----------
    graph : Data
        Homogeneous graph.

    Returns
    -------
    int
        Node count.

    Raises
    ------
    TypeError
        If ``graph`` is not homogeneous ``Data``.
    ValueError
        If the node count cannot be inferred.
    """
    if type(graph) is not Data:
        msg = (
            "estimate_graphon supports homogeneous Data only, "
            f"got {type(graph).__name__}"
        )
        raise TypeError(msg)
    if graph.x is not None and graph.x.ndim == 2:
        return int(graph.x.shape[0])
    if "num_nodes" in graph and graph["num_nodes"] is not None:
        return int(graph["num_nodes"])
    edge_index = graph.edge_index
    if edge_index is not None and int(edge_index.numel()) > 0:
        return int(edge_index.max()) + 1
    msg = "cannot infer num_nodes from graph without x, num_nodes, or edges"
    raise ValueError(msg)


def _dense_adjacency(graph: Data, num_nodes: int) -> Tensor:
    """Return a loopless symmetric 0/1 adjacency.

    Parameters
    ----------
    graph : Data
        Homogeneous snapshot.
    num_nodes : int
        Union size.

    Returns
    -------
    Tensor
        Dense ``(N, N)`` adjacency (float64).
    """
    adjacency = torch.zeros(num_nodes, num_nodes, dtype=torch.float64)
    edge_index = graph.edge_index
    if edge_index is not None and int(edge_index.numel()) > 0:
        if int(edge_index.min()) < 0 or int(edge_index.max()) >= num_nodes:
            msg = (
                "edge_index endpoints must lie in "
                f"[0, {num_nodes}), got min={int(edge_index.min())}, "
                f"max={int(edge_index.max())}"
            )
            raise ValueError(msg)
        src = edge_index[0].long()
        dst = edge_index[1].long()
        adjacency[src, dst] = 1.0
        adjacency[dst, src] = 1.0
    adjacency.fill_diagonal_(0.0)
    return adjacency


def _mean_offdiag(adjacency: Tensor) -> float:
    """Return the mean of the strict upper triangle.

    Parameters
    ----------
    adjacency : Tensor
        Square dense matrix.

    Returns
    -------
    float
        Mean off-diagonal (upper) entry.
    """
    num_nodes = int(adjacency.shape[0])
    tri = torch.triu(adjacency, diagonal=1)
    count = num_nodes * (num_nodes - 1) / 2.0
    return float(tri.sum() / count)


def _zero_diagonal(matrix: Tensor) -> Tensor:
    """Return a copy with a zero diagonal.

    Parameters
    ----------
    matrix : Tensor
        Square matrix.

    Returns
    -------
    Tensor
        Loopless copy.
    """
    out = matrix.clone()
    out.fill_diagonal_(0.0)
    return out


@dataclass(frozen=True, eq=False)
class GraphonEstimate:
    """Dense graphon-family fit on aligned homogeneous graphs.

    Attributes
    ----------
    kernel_family : {"constant", "product", "low_rank"}
        Requested family.
    density : float
        Mean off-diagonal edge probability (dimensionless).
    probability_matrix : Tensor
        Estimated loopless edge probabilities ``(N, N)``.
    latent_scores : Tensor or None
        Product-kernel scores ``(N,)`` in ``[0, 1]``.
    factors : Tensor or None
        Low-rank factors ``(N, r)``.
    rank : int or None
        Requested SVD rank when ``kernel_family="low_rank"``.
    n_graphs : int
        Number of input graphs.
    num_nodes : int
        Shared node count.

    Notes
    -----
    Equality is disabled because the payload holds tensors. This is a
    teaching estimator, not a unique graphon identifier. Low-rank
    ``factors`` use non-negative singular values and need not satisfy
    :math:`P = FF^{\\top}`.
    """

    kernel_family: GraphonKernelFamily
    density: float
    probability_matrix: Tensor
    latent_scores: Tensor | None
    factors: Tensor | None
    rank: int | None
    n_graphs: int
    num_nodes: int


def estimate_graphon(
    graphs: Sequence[Data],
    *,
    kernel_family: GraphonKernelFamily = "low_rank",
    rank: int = 4,
) -> GraphonEstimate:
    """Estimate a constant, product, or low-rank kernel from aligned graphs.

    Graphs must share a finite node count :math:`N` (dense teaching path).
    Constant recovery is the mean off-diagonal density. Product recovery
    uses :math:`\\hat u_i = 2 d_i / (N-1)` on the mean adjacency (for
    :math:`W(u,v)=uv`). Low-rank is a truncated SVD of that mean
    adjacency, clipped to ``[0, 1]``. This does **not** identify a unique
    graphon, does not transfer to arbitrary sparse sensor graphs, and is
    not Borgs–Chayes–Lovász sparse-graph theory
    (``LovaszSzegedy2006``; ``Ruiz2023Transferability``).

    Parameters
    ----------
    graphs : sequence of Data
        Homogeneous snapshots with a shared ``N``.
    kernel_family : {"constant", "product", "low_rank"}, optional
        Family to fit. Default ``"low_rank"``.
    rank : int, optional
        SVD rank for ``low_rank``. Default ``4``. Ignored otherwise.

    Returns
    -------
    GraphonEstimate
        Fitted kernel on the observed node set.

    Raises
    ------
    ValueError
        If the graph list is empty, ``N`` is mixed or outside
        ``[2, MAX_GRAPHON_NODES]``, or ``kernel_family`` / ``rank`` is
        invalid.
    TypeError
        If a graph is not homogeneous ``Data``.
    """
    if kernel_family not in _FAMILIES:
        raise ValueError(
            f"kernel_family must be one of {_FAMILIES_DOC}, got {kernel_family!r}"
        )
    snapshots = tuple(graphs)
    if not snapshots:
        raise ValueError("estimate_graphon requires at least one graph")
    counts = [_graph_num_nodes(graph) for graph in snapshots]
    num_nodes = counts[0]
    if any(count != num_nodes for count in counts):
        msg = (
            "estimate_graphon requires aligned graphs with a shared N; "
            f"got {sorted(set(counts))}. Unaligned / unbounded node "
            "growth is refused."
        )
        raise ValueError(msg)
    if num_nodes < 2:
        raise ValueError(f"num_nodes must be >= 2, got {num_nodes}")
    if num_nodes > MAX_GRAPHON_NODES:
        msg = (
            "estimate_graphon refuses N > "
            f"{MAX_GRAPHON_NODES} (dense O(N^2) teaching ceiling), "
            f"got {num_nodes}"
        )
        raise ValueError(msg)
    stacked = torch.stack(
        [_dense_adjacency(graph, num_nodes) for graph in snapshots],
        dim=0,
    )
    mean_adj = stacked.mean(dim=0)
    density = _mean_offdiag(mean_adj)
    latent_scores: Tensor | None = None
    factors: Tensor | None = None
    fitted_rank: int | None = None
    if kernel_family == "constant":
        probs = density * torch.ones(
            num_nodes,
            num_nodes,
            dtype=torch.float64,
        )
        probability_matrix = _zero_diagonal(probs)
    elif kernel_family == "product":
        degrees = mean_adj.sum(dim=1)
        latent_scores = (2.0 * degrees / float(num_nodes - 1)).clamp(0.0, 1.0)
        probability_matrix = _zero_diagonal(
            latent_scores.unsqueeze(1) * latent_scores.unsqueeze(0)
        )
    else:
        if int(rank) < 1:
            raise ValueError(f"rank must be a positive int, got {rank}")
        fitted_rank = min(int(rank), num_nodes - 1)
        left, values, right = torch.linalg.svd(mean_adj, full_matrices=False)
        kept = left[:, :fitted_rank] * values[:fitted_rank]
        reconstructed = kept @ right[:fitted_rank]
        reconstructed = 0.5 * (reconstructed + reconstructed.T)
        probability_matrix = _zero_diagonal(reconstructed.clamp(0.0, 1.0))
        scale = values[:fitted_rank].clamp(min=0.0).sqrt()
        factors = left[:, :fitted_rank] * scale
    return GraphonEstimate(
        kernel_family=kernel_family,
        density=density,
        probability_matrix=probability_matrix,
        latent_scores=latent_scores,
        factors=factors,
        rank=fitted_rank,
        n_graphs=len(snapshots),
        num_nodes=num_nodes,
    )
