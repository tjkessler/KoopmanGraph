"""Utilities for spatiotemporal graph snapshot sequences."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data

ArrayLike = Tensor | np.ndarray


def _as_tensor(value: ArrayLike, *, dtype: torch.dtype | None = None) -> Tensor:
    """Convert an array-like value to a :class:`torch.Tensor`.

    Parameters
    ----------
    value : Tensor or ndarray
        Input array or tensor.
    dtype : torch.dtype, optional
        Target dtype. When ``value`` is already a tensor, conversion is applied
        only if the dtypes differ.

    Returns
    -------
    Tensor
        Tensor representation of ``value``.
    """
    if isinstance(value, Tensor):
        if dtype is not None and value.dtype != dtype:
            return value.to(dtype=dtype)
        return value
    return torch.as_tensor(value, dtype=dtype)


def _validate_shared_topology(snapshots: Sequence[Data]) -> None:
    """Verify that all snapshots share node count, features, and topology.

    Parameters
    ----------
    snapshots : sequence of Data
        Graph snapshots to validate.

    Raises
    ------
    ValueError
        If the sequence is empty or any snapshot differs in ``edge_index``,
        node count, or feature dimension from the first snapshot.
    """
    if not snapshots:
        msg = "GraphSnapshotSequence requires at least one snapshot"
        raise ValueError(msg)

    reference = snapshots[0]
    ref_edge_index = reference.edge_index
    ref_num_nodes = reference.num_nodes
    ref_in_channels = reference.x.shape[1]

    for idx, snapshot in enumerate(snapshots[1:], start=1):
        if snapshot.num_nodes != ref_num_nodes:
            msg = (
                f"Snapshot {idx} has {snapshot.num_nodes} nodes, "
                f"expected {ref_num_nodes}"
            )
            raise ValueError(msg)
        if snapshot.x.shape[1] != ref_in_channels:
            msg = (
                f"Snapshot {idx} has feature dimension {snapshot.x.shape[1]}, "
                f"expected {ref_in_channels}"
            )
            raise ValueError(msg)
        if not torch.equal(snapshot.edge_index, ref_edge_index):
            msg = f"Snapshot {idx} has a different edge_index than snapshot 0"
            raise ValueError(msg)


class GraphSnapshotSequence:
    """Container for a time-ordered sequence of PyG ``Data`` graph snapshots.

    All snapshots must share the same ``edge_index``, node count, and feature
    dimension. Downstream training APIs should require at least two snapshots;
    construction here allows a single snapshot for inspection or prediction-only
    workflows.

    Notes
    -----
    Read-only views of the shared topology and sequence metadata are exposed as
    :attr:`snapshots`, :attr:`edge_index`, :attr:`num_nodes`,
    :attr:`num_timesteps`, and :attr:`in_channels`.
    """

    def __init__(self, snapshots: Sequence[Data]) -> None:
        """Initialize from a sequence of graph snapshots.

        Parameters
        ----------
        snapshots : sequence of Data
            Time-ordered graph snapshots. Validated for shared topology on
            construction.
        """
        snapshot_list = list(snapshots)
        _validate_shared_topology(snapshot_list)
        self._snapshots = snapshot_list

    @classmethod
    def from_arrays(
        cls,
        node_features: ArrayLike,
        edge_index: ArrayLike,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> GraphSnapshotSequence:
        """Build a sequence from node feature arrays and a shared topology.

        Parameters
        ----------
        node_features : array-like
            Array with shape ``(num_timesteps, num_nodes, in_channels)``.
        edge_index : array-like
            Shared edge index with shape ``(2, num_edges)``.
        dtype : torch.dtype, optional
            Floating dtype used when converting numpy inputs to torch tensors.
            Default is ``torch.float32``.

        Returns
        -------
        :class:`~koopman_graph.data.GraphSnapshotSequence`
            Validated snapshot sequence.

        Raises
        ------
        ValueError
            If ``node_features`` or ``edge_index`` have invalid shape.
        """
        features = _as_tensor(node_features, dtype=dtype)
        edges = _as_tensor(edge_index, dtype=torch.long)

        if features.ndim != 3:
            msg = (
                f"node_features must have shape "
                f"(num_timesteps, num_nodes, in_channels), got {tuple(features.shape)}"
            )
            raise ValueError(msg)
        if edges.ndim != 2 or edges.shape[0] != 2:
            msg = f"edge_index must have shape (2, num_edges), got {tuple(edges.shape)}"
            raise ValueError(msg)
        if features.shape[0] < 1:
            msg = "node_features must contain at least one timestep"
            raise ValueError(msg)

        snapshots = [
            Data(x=features[t], edge_index=edges) for t in range(features.shape[0])
        ]
        return cls(snapshots)

    @property
    def snapshots(self) -> list[Data]:
        """Return the underlying list of graph snapshots.

        Returns
        -------
        list of Data
            Time-ordered PyG graph snapshots.
        """
        return self._snapshots

    @property
    def edge_index(self) -> Tensor:
        """Return the shared edge index for all snapshots.

        Returns
        -------
        Tensor
            Edge index with shape ``(2, num_edges)``.
        """
        return self._snapshots[0].edge_index

    @property
    def num_nodes(self) -> int:
        """Return the number of nodes in the graph topology.

        Returns
        -------
        int
            Node count shared across all snapshots.
        """
        return int(self._snapshots[0].num_nodes)

    @property
    def num_timesteps(self) -> int:
        """Return the number of timesteps in the sequence.

        Returns
        -------
        int
            Length of the temporal sequence.
        """
        return len(self._snapshots)

    @property
    def in_channels(self) -> int:
        """Return the node feature dimension.

        Returns
        -------
        int
            Feature dimension shared across all snapshots.
        """
        return int(self._snapshots[0].x.shape[1])

    def __len__(self) -> int:
        """Return the number of timesteps in the sequence.

        Returns
        -------
        int
            Same value as :attr:`num_timesteps`.
        """
        return len(self._snapshots)

    def __getitem__(self, index: int) -> Data:
        """Return the graph snapshot at ``index``.

        Parameters
        ----------
        index : int
            Timestep index.

        Returns
        -------
        Data
            Graph snapshot at the requested timestep.
        """
        return self._snapshots[index]

    def __iter__(self) -> Iterator[Data]:
        """Iterate over graph snapshots in temporal order.

        Yields
        ------
        Data
            Graph snapshot at each timestep.
        """
        return iter(self._snapshots)
