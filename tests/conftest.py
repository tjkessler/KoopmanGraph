"""Shared pytest fixtures for KoopmanGraph tests."""

from collections.abc import Callable

import pytest
import torch
from torch_geometric.data import Data


@pytest.fixture
def synthetic_edge_index() -> torch.Tensor:
    """Provide a shared edge index for small undirected 5-node test graphs.

    Returns
    -------
    Tensor
        Edge index with shape ``(2, 8)`` forming a path graph on 5 nodes.
    """
    return torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 4],
            [1, 0, 2, 1, 3, 2, 4, 3],
        ],
        dtype=torch.long,
    )


@pytest.fixture
def synthetic_graph(synthetic_edge_index: torch.Tensor) -> Data:
    """Provide a small graph with random 3-dimensional node features.

    Parameters
    ----------
    synthetic_edge_index : Tensor
        Shared topology fixture.

    Returns
    -------
    Data
        PyG graph with 5 nodes and ``x`` of shape ``(5, 3)``.
    """
    x = torch.randn(5, 3)
    return Data(x=x, edge_index=synthetic_edge_index)


@pytest.fixture
def make_snapshots(
    synthetic_edge_index: torch.Tensor,
) -> Callable[..., list[Data]]:
    """Provide a factory for building random snapshot lists.

    Parameters
    ----------
    synthetic_edge_index : Tensor
        Default edge index used when the factory is called without one.

    Returns
    -------
    callable
        Factory accepting ``edge_index``, ``num_timesteps``, ``num_nodes``, and
        ``in_channels`` keyword arguments.
    """

    def _make(
        edge_index: torch.Tensor | None = None,
        *,
        num_timesteps: int = 3,
        num_nodes: int = 5,
        in_channels: int = 3,
    ) -> list[Data]:
        """Build a list of random graph snapshots sharing one topology.

        Parameters
        ----------
        edge_index : Tensor, optional
            Shared edge index. Defaults to ``synthetic_edge_index``.
        num_timesteps : int, optional
            Number of snapshots to generate. Default is ``3``.
        num_nodes : int, optional
            Number of nodes per snapshot. Default is ``5``.
        in_channels : int, optional
            Node feature dimension. Default is ``3``.

        Returns
        -------
        list of Data
            Random snapshots with consistent topology.
        """
        edges = synthetic_edge_index if edge_index is None else edge_index
        return [
            Data(x=torch.randn(num_nodes, in_channels), edge_index=edges)
            for _ in range(num_timesteps)
        ]

    return _make


@pytest.fixture
def synthetic_graph_sequence(make_snapshots: Callable[..., list[Data]]):
    """Provide a three-timestep dynamic graph sequence.

    Parameters
    ----------
    make_snapshots : callable
        Snapshot factory fixture.

    Returns
    -------
    GraphSnapshotSequence
        Sequence of three random snapshots on the shared 5-node topology.
    """
    from koopman_graph.data import GraphSnapshotSequence

    return GraphSnapshotSequence(make_snapshots(num_timesteps=3))


@pytest.fixture
def scaling_sequence(synthetic_edge_index: torch.Tensor):
    """Provide a deterministic decay sequence for training smoke tests.

    The sequence follows ``x_{t+1} = 0.9 * x_t`` over five timesteps.

    Parameters
    ----------
    synthetic_edge_index : Tensor
        Shared topology fixture.

    Returns
    -------
    GraphSnapshotSequence
        Five-step sequence with ``x`` initialized to ones.
    """
    from koopman_graph.data import GraphSnapshotSequence

    x0 = torch.ones(5, 3)
    snapshots = [
        Data(x=x0 * (0.9**t), edge_index=synthetic_edge_index) for t in range(5)
    ]
    return GraphSnapshotSequence(snapshots)
