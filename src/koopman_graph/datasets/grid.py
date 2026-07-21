"""2D grid spatiotemporal graph benchmarks for tests and tutorials."""

from __future__ import annotations

import torch
from torch import Tensor

from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.datasets.dynamics import (
    InitialStateName,
    add_gaussian_noise,
    diffusion_sequence_from_features,
    initial_node_features,
    laplacian_diffusion_rollout,
    make_generator,
    validate_advection_decay_rate,
    validate_diffusion_generation_params,
)


def _grid_edge_index(num_rows: int, num_cols: int) -> Tensor:
    """Build bidirectional edges for a 4-connected 2D grid.

    Parameters
    ----------
    num_rows : int
        Number of grid rows.
    num_cols : int
        Number of grid columns.

    Returns
    -------
    Tensor
        Edge index with shape ``(2, num_edges)``.
    """
    if num_rows < 1 or num_cols < 1:
        return torch.zeros((2, 0), dtype=torch.long)

    src: list[int] = []
    dst: list[int] = []
    for row in range(num_rows):
        for col in range(num_cols):
            node = row * num_cols + col
            if col < num_cols - 1:
                neighbor = row * num_cols + (col + 1)
                src.extend([node, neighbor])
                dst.extend([neighbor, node])
            if row < num_rows - 1:
                neighbor = (row + 1) * num_cols + col
                src.extend([node, neighbor])
                dst.extend([neighbor, node])
    return torch.tensor([src, dst], dtype=torch.long)


def grid_node_index(row: int, col: int, *, num_cols: int) -> int:
    """Return the flattened node index for a grid coordinate.

    Parameters
    ----------
    row : int
        Zero-based row index.
    col : int
        Zero-based column index.
    num_cols : int
        Number of columns in the grid.

    Returns
    -------
    int
        Flattened node index ``row * num_cols + col``.
    """
    return row * num_cols + col


class GridDynamicGraphBenchmark:
    """Reproducible Laplacian diffusion on a 2D lattice graph.

    Node features evolve via the same graph diffusion dynamics as
    :class:`~koopman_graph.datasets.SyntheticDynamicGraphBenchmark`, but on a
    4-connected grid. Corner, edge, and interior nodes have different degrees,
    which makes graph attention encoders a natural fit.

    Parameters
    ----------
    num_rows : int, optional
        Grid height. Default is ``10``.
    num_cols : int, optional
        Grid width. Default is ``10``.
    num_timesteps : int, optional
        Number of temporal snapshots. Default is ``40``.
    in_channels : int, optional
        Node feature dimension. Default is ``3``.
    diffusion_rate : float, optional
        Laplacian diffusion strength in ``[0, 1]``. Default is ``0.1``.
    decay_rate : float, optional
        Global amplitude decay applied each step. Default is ``0.99``.
    noise_std : float, optional
        Standard deviation of additive Gaussian noise. Default is ``0.01``.
    seed : int, optional
        Random seed for the initial state and noise. ``None`` uses unseeded
        randomness; tutorials should pass an explicit seed (e.g. ``42``).
    initial_state : {"random", "ones"}, optional
        Initial node feature pattern. Default is ``"ones"``.
    dtype : torch.dtype, optional
        Floating dtype for generated features. Default is ``torch.float32``.
    """

    @classmethod
    def generate(
        cls,
        *,
        num_rows: int = 10,
        num_cols: int = 10,
        num_timesteps: int = 40,
        in_channels: int = 3,
        diffusion_rate: float = 0.1,
        decay_rate: float = 0.99,
        noise_std: float = 0.01,
        seed: int | None = None,
        initial_state: InitialStateName = "ones",
        dtype: torch.dtype = torch.float32,
    ) -> GraphSnapshotSequence:
        """Generate a dynamic 2D grid snapshot sequence.

        Parameters
        ----------
        num_rows : int, optional
            Grid height. Default is ``10``.
        num_cols : int, optional
            Grid width. Default is ``10``.
        num_timesteps : int, optional
            Number of temporal snapshots. Default is ``40``.
        in_channels : int, optional
            Node feature dimension. Default is ``3``.
        diffusion_rate : float, optional
            Laplacian diffusion strength in ``[0, 1]``. Default is ``0.1``.
        decay_rate : float, optional
            Global amplitude decay applied each step. Default is ``0.99``.
        noise_std : float, optional
            Standard deviation of additive Gaussian noise. Default is ``0.01``.
        seed : int, optional
            Random seed for the initial state and noise. ``None`` uses unseeded
            randomness; tutorials should pass an explicit seed (e.g. ``42``).
        initial_state : {"random", "ones"}, optional
            Initial node feature pattern. Default is ``"ones"``.
        dtype : torch.dtype, optional
            Floating dtype for generated features. Default is ``torch.float32``.

        Returns
        -------
        :class:`~koopman_graph.data.GraphSnapshotSequence`
            Time-ordered snapshots on the grid graph.

        Raises
        ------
        ValueError
            If any generation parameter is invalid.
        """
        if num_rows < 1:
            msg = f"num_rows must be >= 1, got {num_rows}"
            raise ValueError(msg)
        if num_cols < 1:
            msg = f"num_cols must be >= 1, got {num_cols}"
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

        num_nodes = num_rows * num_cols
        edge_index = _grid_edge_index(num_rows, num_cols)
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


def _grid_neighbors(
    row: int,
    col: int,
    *,
    num_rows: int,
    num_cols: int,
) -> dict[str, int]:
    """Return named grid neighbors for a lattice coordinate.

    Parameters
    ----------
    row : int
        Zero-based row index.
    col : int
        Zero-based column index.
    num_rows : int
        Number of rows in the grid.
    num_cols : int
        Number of columns in the grid.

    Returns
    -------
    dict of str to int
        Mapping from direction names (``"west"``, ``"east"``, ``"north"``,
        ``"south"``) to flattened neighbor node indices.
    """
    neighbors: dict[str, int] = {}
    if col > 0:
        neighbors["west"] = grid_node_index(row, col - 1, num_cols=num_cols)
    if col < num_cols - 1:
        neighbors["east"] = grid_node_index(row, col + 1, num_cols=num_cols)
    if row > 0:
        neighbors["north"] = grid_node_index(row - 1, col, num_cols=num_cols)
    if row < num_rows - 1:
        neighbors["south"] = grid_node_index(row + 1, col, num_cols=num_cols)
    return neighbors


def _anisotropic_advection_step(
    state: Tensor,
    *,
    num_rows: int,
    num_cols: int,
    decay_rate: float,
    west_weight: float,
    north_weight: float,
) -> Tensor:
    """Apply one noiseless anisotropic advection step (no parameter checks)."""
    in_channels = state.shape[1]
    dtype = state.dtype
    updated = decay_rate * state
    for row in range(num_rows):
        for col in range(num_cols):
            node = grid_node_index(row, col, num_cols=num_cols)
            neighbors = _grid_neighbors(
                row,
                col,
                num_rows=num_rows,
                num_cols=num_cols,
            )
            if not neighbors:
                continue

            weights: dict[int, float] = {}
            if "west" in neighbors:
                weights[neighbors["west"]] = west_weight
            if "north" in neighbors:
                weights[neighbors["north"]] = north_weight
            other_neighbors = [
                index
                for name, index in neighbors.items()
                if name not in {"west", "north"}
            ]
            if other_neighbors:
                remaining = 1.0 - west_weight - north_weight
                share = remaining / len(other_neighbors)
                for index in other_neighbors:
                    weights[index] = share

            weight_sum = sum(weights.values())
            if weight_sum <= 0.0:
                continue
            mixture = torch.zeros(in_channels, dtype=dtype)
            for neighbor, weight in weights.items():
                mixture = mixture + weight * state[neighbor]
            mixture = mixture / weight_sum
            updated[node] = updated[node] + (1.0 - decay_rate) * mixture
    return updated


def anisotropic_advection_step(
    state: Tensor,
    *,
    num_rows: int,
    num_cols: int,
    decay_rate: float = 0.85,
    west_weight: float = 0.7,
    north_weight: float = 0.2,
) -> Tensor:
    """Apply one noiseless anisotropic advection step on a 2D lattice.

    This is the deterministic update used by
    :class:`~koopman_graph.datasets.grid.AnisotropicAdvectionGridBenchmark`
    before optional Gaussian noise is added. Tutorials can call it directly for
    impulse-response diagnostics without reimplementing the neighbor mixture.

    Parameters
    ----------
    state : Tensor
        Node features with shape ``(num_rows * num_cols, in_channels)``.
    num_rows : int
        Grid height.
    num_cols : int
        Grid width.
    decay_rate : float, optional
        Self-retention factor in ``(0, 1)``. Default is ``0.85``.
    west_weight : float, optional
        Relative influence of the western neighbor. Default is ``0.7``.
    north_weight : float, optional
        Relative influence of the northern neighbor. Default is ``0.2``.

    Returns
    -------
    Tensor
        Updated node features with the same shape as ``state``.

    Raises
    ------
    ValueError
        If grid dimensions, ``state`` shape, or advection weights are invalid.
    """
    if num_rows < 1:
        msg = f"num_rows must be >= 1, got {num_rows}"
        raise ValueError(msg)
    if num_cols < 1:
        msg = f"num_cols must be >= 1, got {num_cols}"
        raise ValueError(msg)
    if state.ndim != 2:
        msg = (
            "state must be 2D (num_nodes, in_channels), "
            f"got shape {tuple(state.shape)}"
        )
        raise ValueError(msg)
    expected_nodes = num_rows * num_cols
    if state.shape[0] != expected_nodes:
        msg = (
            f"state.shape[0] must equal num_rows * num_cols ({expected_nodes}), "
            f"got {state.shape[0]}"
        )
        raise ValueError(msg)
    if state.shape[1] < 1:
        msg = f"state in_channels must be >= 1, got {state.shape[1]}"
        raise ValueError(msg)
    validate_advection_decay_rate(decay_rate)
    if west_weight < 0.0 or north_weight < 0.0:
        msg = "west_weight and north_weight must be non-negative"
        raise ValueError(msg)
    if west_weight + north_weight >= 1.0:
        msg = (
            f"west_weight + north_weight must be < 1, got "
            f"{west_weight + north_weight}"
        )
        raise ValueError(msg)
    return _anisotropic_advection_step(
        state,
        num_rows=num_rows,
        num_cols=num_cols,
        decay_rate=decay_rate,
        west_weight=west_weight,
        north_weight=north_weight,
    )


class AnisotropicAdvectionGridBenchmark:
    """Directional advection on a 2D lattice with asymmetric neighbor weights.

    Each node updates from a weighted mixture of neighbors where the **west**
    and **north** directions dominate. This breaks the symmetry that GCN layers
    assume when they aggregate neighbors uniformly, making graph attention a
    better fit than plain convolution for rollout forecasting.

    The one-step update is

    .. math::

        x_{t+1} = \\text{decay\\_rate} \\cdot x_t
                  + (1 - \\text{decay\\_rate}) \\cdot
                    \\frac{\\sum_{j \\in \\mathcal{N}(i)} w_{ij} x_{j,t}}
                         {\\sum_{j \\in \\mathcal{N}(i)} w_{ij}}

    with ``w_{i,\\text{west}} = west_weight``, ``w_{i,\\text{north}} = north_weight``,
    and remaining neighbors sharing leftover mass
    ``1 - west_weight - north_weight`` equally. If a preferred direction has
    no neighbor (grid border), that weight is still reserved from the leftover
    budget but never assigned, so border nodes mix less strongly than
    interior nodes; the mixture is then renormalized by the sum of assigned
    weights. When that assigned-weight sum is zero (for example both preferred
    weights are zero at a corner that only has west/north neighbors), the node
    keeps pure self-retention ``decay_rate · x_t`` with no neighbor mixture.

    Parameters
    ----------
    num_rows : int, optional
        Grid height. Default is ``8``.
    num_cols : int, optional
        Grid width. Default is ``8``.
    num_timesteps : int, optional
        Number of temporal snapshots. Default is ``40``.
    in_channels : int, optional
        Node feature dimension. Default is ``3``.
    decay_rate : float, optional
        Self-retention factor in ``(0, 1)``. Default is ``0.85``.
    west_weight : float, optional
        Relative influence of the western neighbor. Default is ``0.7``.
    north_weight : float, optional
        Relative influence of the northern neighbor. Default is ``0.2``.
    noise_std : float, optional
        Standard deviation of additive Gaussian noise. Default is ``0.005``.
    seed : int, optional
        Random seed for the initial state and noise. ``None`` uses unseeded
        randomness; tutorials should pass an explicit seed (e.g. ``42``).
    initial_state : {"random", "ones"}, optional
        Initial node feature pattern. Default is ``"ones"``.
    dtype : torch.dtype, optional
        Floating dtype for generated features. Default is ``torch.float32``.
    """

    @classmethod
    def generate(
        cls,
        *,
        num_rows: int = 8,
        num_cols: int = 8,
        num_timesteps: int = 40,
        in_channels: int = 3,
        decay_rate: float = 0.85,
        west_weight: float = 0.7,
        north_weight: float = 0.2,
        noise_std: float = 0.005,
        seed: int | None = None,
        initial_state: InitialStateName = "ones",
        dtype: torch.dtype = torch.float32,
    ) -> GraphSnapshotSequence:
        """Generate a directional advection snapshot sequence on a grid.

        Parameters
        ----------
        num_rows : int, optional
            Grid height. Default is ``8``.
        num_cols : int, optional
            Grid width. Default is ``8``.
        num_timesteps : int, optional
            Number of temporal snapshots. Default is ``40``.
        in_channels : int, optional
            Node feature dimension. Default is ``3``.
        decay_rate : float, optional
            Self-retention factor in ``(0, 1)``. Default is ``0.85``.
        west_weight : float, optional
            Relative influence of the western neighbor. Default is ``0.7``.
        north_weight : float, optional
            Relative influence of the northern neighbor. Default is ``0.2``.
        noise_std : float, optional
            Standard deviation of additive Gaussian noise. Default is ``0.005``.
        seed : int, optional
            Random seed for the initial state and noise. ``None`` uses unseeded
            randomness; tutorials should pass an explicit seed (e.g. ``42``).
        initial_state : {"random", "ones"}, optional
            Initial node feature pattern. Default is ``"ones"``.
        dtype : torch.dtype, optional
            Floating dtype for generated features. Default is ``torch.float32``.

        Returns
        -------
        :class:`~koopman_graph.data.GraphSnapshotSequence`
            Time-ordered snapshots on the grid graph.

        Raises
        ------
        ValueError
            If any generation parameter is invalid.
        """
        if num_rows < 1:
            msg = f"num_rows must be >= 1, got {num_rows}"
            raise ValueError(msg)
        if num_cols < 1:
            msg = f"num_cols must be >= 1, got {num_cols}"
            raise ValueError(msg)
        if num_timesteps < 1:
            msg = f"num_timesteps must be >= 1, got {num_timesteps}"
            raise ValueError(msg)
        if in_channels < 1:
            msg = f"in_channels must be >= 1, got {in_channels}"
            raise ValueError(msg)
        validate_advection_decay_rate(decay_rate)
        if west_weight < 0.0 or north_weight < 0.0:
            msg = "west_weight and north_weight must be non-negative"
            raise ValueError(msg)
        if west_weight + north_weight >= 1.0:
            msg = (
                f"west_weight + north_weight must be < 1, got "
                f"{west_weight + north_weight}"
            )
            raise ValueError(msg)
        validate_diffusion_generation_params(
            decay_rate=decay_rate,
            noise_std=noise_std,
            initial_state=initial_state,
        )

        num_nodes = num_rows * num_cols
        generator = make_generator(seed)
        edge_index = _grid_edge_index(num_rows, num_cols)
        state = initial_node_features(
            num_nodes,
            in_channels,
            initial_state,
            generator=generator,
            dtype=dtype,
        )

        snapshots = [state.clone()]
        for _ in range(num_timesteps - 1):
            updated = _anisotropic_advection_step(
                state,
                num_rows=num_rows,
                num_cols=num_cols,
                decay_rate=decay_rate,
                west_weight=west_weight,
                north_weight=north_weight,
            )
            state = add_gaussian_noise(
                updated,
                noise_std,
                generator=generator,
                dtype=dtype,
            )
            snapshots.append(state.clone())

        features = torch.stack(snapshots, dim=0)
        return diffusion_sequence_from_features(features, edge_index, dtype=dtype)
