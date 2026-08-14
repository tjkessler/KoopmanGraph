"""Random node-cluster subgraph partitions (ClusterGCN / GraphSAINT teaching).

Training approximation over induced subgraphs. Default predict/evaluate
remain full-graph.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import subgraph


def cluster_node_partition(
    num_nodes: int,
    num_clusters: int,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Assign each node a cluster id in ``[0, num_clusters)``.

    Parameters
    ----------
    num_nodes : int
        Graph size.
    num_clusters : int
        Number of partitions.
    generator : torch.Generator or None, optional
        Optional RNG.

    Returns
    -------
    Tensor
        Integer assignment of shape ``(num_nodes,)``.
    """
    if num_clusters < 1:
        raise ValueError(f"num_clusters must be positive, got {num_clusters}")
    if num_nodes < 1:
        raise ValueError(f"num_nodes must be positive, got {num_nodes}")
    return torch.randint(
        0,
        int(num_clusters),
        (int(num_nodes),),
        generator=generator,
    )


def induced_cluster_subgraph(
    data: Data,
    cluster_id: int,
    assignment: Tensor,
) -> Data:
    """Return the induced subgraph of one cluster.

    Parameters
    ----------
    data : Data
        Homogeneous snapshot.
    cluster_id : int
        Cluster to extract.
    assignment : Tensor
        Per-node cluster ids.

    Returns
    -------
    Data
        Induced subgraph with relabeled nodes.

    Raises
    ------
    ValueError
        If features/edges are missing or the cluster is empty.
    """
    if data.x is None or data.edge_index is None:
        raise ValueError("induced_cluster_subgraph requires Data.x and edge_index")
    mask = assignment == int(cluster_id)
    node_idx = mask.nonzero(as_tuple=False).view(-1)
    if node_idx.numel() == 0:
        raise ValueError(f"cluster {cluster_id} is empty")
    edge_index, _ = subgraph(
        node_idx,
        data.edge_index,
        relabel_nodes=True,
        num_nodes=data.x.shape[0],
    )
    return Data(x=data.x[node_idx], edge_index=edge_index)


def iter_cluster_subgraphs(
    data: Data,
    num_clusters: int,
    *,
    generator: torch.Generator | None = None,
) -> Iterator[Data]:
    """Yield induced cluster subgraphs for sampled training.

    Compose with :class:`~koopman_graph.data.NeighborWindowSampler` by
    sampling windows on each subgraph. Default ``predict`` / ``evaluate``
    remain full-graph.

    Parameters
    ----------
    data : Data
        Homogeneous snapshot.
    num_clusters : int
        Number of random partitions.
    generator : torch.Generator or None, optional
        Optional RNG.

    Yields
    ------
    Data
        Non-empty induced subgraphs.
    """
    if data.x is None:
        raise ValueError("iter_cluster_subgraphs requires Data.x")
    assignment = cluster_node_partition(
        int(data.x.shape[0]),
        num_clusters,
        generator=generator,
    )
    for cluster_id in range(int(num_clusters)):
        mask = assignment == int(cluster_id)
        if int(mask.sum()) == 0:
            continue
        yield induced_cluster_subgraph(data, cluster_id, assignment)
