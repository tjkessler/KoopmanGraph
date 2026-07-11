"""Synthetic spatiotemporal graph benchmarks for tests and tutorials."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.datasets.dynamics import (
    InitialStateName,
    diffusion_sequence_from_features,
    laplacian_diffusion_rollout,
    make_generator,
    validate_diffusion_generation_params,
)

TopologyName = Literal["path", "ring"]


def _path_edge_index(num_nodes: int) -> Tensor:
    """Build bidirectional path-graph edges.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the path.

    Returns
    -------
    Tensor
        Edge index with shape ``(2, num_edges)``.
    """
    if num_nodes < 2:
        return torch.zeros((2, 0), dtype=torch.long)

    src: list[int] = []
    dst: list[int] = []
    for node in range(num_nodes - 1):
        src.extend([node, node + 1])
        dst.extend([node + 1, node])
    return torch.tensor([src, dst], dtype=torch.long)


def _ring_edge_index(num_nodes: int) -> Tensor:
    """Build bidirectional ring-graph edges.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the ring.

    Returns
    -------
    Tensor
        Edge index with shape ``(2, num_edges)``.
    """
    if num_nodes < 2:
        return torch.zeros((2, 0), dtype=torch.long)

    src: list[int] = []
    dst: list[int] = []
    for node in range(num_nodes):
        nxt = (node + 1) % num_nodes
        src.extend([node, nxt])
        dst.extend([nxt, node])
    return torch.tensor([src, dst], dtype=torch.long)


def _build_topology(topology: TopologyName, num_nodes: int) -> Tensor:
    """Return the edge index for a supported synthetic topology.

    Parameters
    ----------
    topology : {"path", "ring"}
        Graph topology name.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    Tensor
        Shared edge index used by all snapshots.

    Raises
    ------
    ValueError
        If ``topology`` is not supported.
    """
    if topology == "path":
        return _path_edge_index(num_nodes)
    if topology == "ring":
        return _ring_edge_index(num_nodes)
    msg = f"Unsupported topology {topology!r}; expected 'path' or 'ring'"
    raise ValueError(msg)


class SyntheticDynamicGraphBenchmark:
    """Reproducible synthetic graph dynamics for benchmarks and tutorials.

    Node features evolve via graph Laplacian diffusion with optional global
    decay and additive Gaussian noise:

    .. math::

        x_{t+1} = \\text{decay\\_rate} \\cdot S x_t + \\mathcal{N}(0, \\sigma^2)

    where ``S = (1 - diffusion_rate) * I + diffusion_rate * D^{-1/2} A D^{-1/2}``.

    Parameters
    ----------
    num_nodes : int, optional
        Number of nodes in the graph. Default is ``20``.
    num_timesteps : int, optional
        Number of temporal snapshots. Default is ``50``.
    in_channels : int, optional
        Node feature dimension. Default is ``3``.
    topology : {"path", "ring"}, optional
        Static graph topology shared across timesteps. Default is ``"path"``.
    diffusion_rate : float, optional
        Laplacian diffusion strength in ``[0, 1]``. Default is ``0.5``.
    decay_rate : float, optional
        Global amplitude decay applied each step. Default is ``0.95``.
    noise_std : float, optional
        Standard deviation of additive Gaussian noise. Default is ``0.0``.
    seed : int, optional
        Random seed for the initial state and noise. ``None`` uses unseeded
        randomness.
    initial_state : {"random", "ones"}, optional
        Initial node feature pattern. Default is ``"random"``.
    dtype : torch.dtype, optional
        Floating dtype for generated features. Default is ``torch.float32``.
    """

    @classmethod
    def generate(
        cls,
        *,
        num_nodes: int = 20,
        num_timesteps: int = 50,
        in_channels: int = 3,
        topology: TopologyName = "path",
        diffusion_rate: float = 0.5,
        decay_rate: float = 0.95,
        noise_std: float = 0.0,
        seed: int | None = None,
        initial_state: InitialStateName = "random",
        dtype: torch.dtype = torch.float32,
    ) -> GraphSnapshotSequence:
        """Generate a synthetic dynamic graph snapshot sequence.

        Parameters
        ----------
        num_nodes : int, optional
            Number of nodes in the graph. Default is ``20``.
        num_timesteps : int, optional
            Number of temporal snapshots. Default is ``50``.
        in_channels : int, optional
            Node feature dimension. Default is ``3``.
        topology : {"path", "ring"}, optional
            Static graph topology shared across timesteps. Default is ``"path"``.
        diffusion_rate : float, optional
            Laplacian diffusion strength in ``[0, 1]``. Default is ``0.5``.
        decay_rate : float, optional
            Global amplitude decay applied each step. Default is ``0.95``.
        noise_std : float, optional
            Standard deviation of additive Gaussian noise. Default is ``0.0``.
        seed : int, optional
            Random seed for the initial state and noise. ``None`` uses
            unseeded randomness.
        initial_state : {"random", "ones"}, optional
            Initial node feature pattern. Default is ``"random"``.
        dtype : torch.dtype, optional
            Floating dtype for generated features. Default is ``torch.float32``.

        Returns
        -------
        :class:`~koopman_graph.data.GraphSnapshotSequence`
            Time-ordered snapshots with shared topology and documented dynamics.

        Raises
        ------
        ValueError
            If any generation parameter is invalid.
        """
        if num_nodes < 1:
            msg = f"num_nodes must be >= 1, got {num_nodes}"
            raise ValueError(msg)
        if num_timesteps < 1:
            msg = f"num_timesteps must be >= 1, got {num_timesteps}"
            raise ValueError(msg)
        if in_channels < 1:
            msg = f"in_channels must be >= 1, got {in_channels}"
            raise ValueError(msg)
        validate_diffusion_generation_params(
            diffusion_rate=diffusion_rate,
            decay_rate=decay_rate,
            noise_std=noise_std,
            initial_state=initial_state,
        )

        edge_index = _build_topology(topology, num_nodes)
        features = laplacian_diffusion_rollout(
            edge_index=edge_index,
            num_nodes=num_nodes,
            num_timesteps=num_timesteps,
            in_channels=in_channels,
            diffusion_rate=diffusion_rate,
            decay_rate=decay_rate,
            noise_std=noise_std,
            initial_state=initial_state,
            dtype=dtype,
            generator=make_generator(seed),
        )
        return diffusion_sequence_from_features(features, edge_index, dtype=dtype)
