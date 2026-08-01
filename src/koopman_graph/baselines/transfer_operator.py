"""Ulam Galerkin transfer-operator baseline on a fixed indicator dictionary.

Approximates the discrete-time transfer / Perron–Frobenius operator by
counting transitions between axis-aligned boxes on flattened graph states
(Ulam's method). Topology-blind: edges are ignored during fit and only
copied onto predicted ``Data`` snapshots.

This is **not** ResDMD, **not** a stochastic Koopman generator / SDE, and
**not** a GNN forecaster. Sphinx API listing is deferred to Phase 61.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor
from torch_geometric.data import Data

from koopman_graph.baselines.base import (
    ClassicalBaseline,
    check_initial_graph,
    copy_topology,
    flatten_snapshots,
    require_static_topology,
)
from koopman_graph.data import (
    GraphSnapshotSequence,
    resolve_sequence,
)
from koopman_graph.spectrum_types import KoopmanSpectrum, compute_spectrum

_MAX_CELLS = 4096
_EDGE_EPS = 1e-8


class UlamTransferOperatorBaseline(ClassicalBaseline):
    """Ulam transfer-operator baseline with a fixed box indicator dictionary.

    Partitions the axis-aligned bounding box of training flattened states into
    a regular product grid (``bins_per_dim`` bins along each coordinate). The
    fitted row-stochastic matrix ``P`` satisfies ``ρ_{t+1} = ρ_t @ P``.
    :attr:`K` aliases ``P`` so :meth:`spectrum` reports transfer-matrix
    eigenvalues. :meth:`predict` reconstructs expected cell-center states
    from propagated one-hot densities (ForecastModel-compatible smoke).

    Parameters
    ----------
    time_step : float, optional
        Physical duration of one snapshot transition. Default ``1.0``.
    bins_per_dim : int, optional
        Number of equal-width bins along each flattened coordinate.
        Default ``4``. Total cells must satisfy
        ``bins_per_dim ** state_dim ≤ 4096``.

    Attributes
    ----------
    P : Tensor or None
        Row-stochastic transfer matrix ``(n_cells, n_cells)``.
    bin_edges : tuple of Tensor or None
        Per-dimension edge vectors of length ``bins_per_dim + 1``.
    cell_centers : Tensor or None
        Cell centers with shape ``(n_cells, state_dim)``.
    n_cells : int or None
        Number of indicator cells after fit.
    """

    def __init__(
        self,
        *,
        time_step: float = 1.0,
        bins_per_dim: int = 4,
    ) -> None:
        """Validate partition knobs and initialize unfitted state.

        Parameters
        ----------
        time_step : float, optional
            See class docstring.
        bins_per_dim : int, optional
            See class docstring.

        Raises
        ------
        ValueError
            If ``bins_per_dim < 1``.
        """
        super().__init__(time_step=time_step, rank=None)
        if bins_per_dim < 1:
            msg = f"bins_per_dim must be >= 1, got {bins_per_dim}"
            raise ValueError(msg)
        self.bins_per_dim = int(bins_per_dim)
        self.P: Tensor | None = None
        self.bin_edges: tuple[Tensor, ...] | None = None
        self.cell_centers: Tensor | None = None
        self.n_cells: int | None = None

    def _is_fitted(self) -> bool:
        """Return whether the Ulam transfer matrix has been fit.

        Returns
        -------
        object
            Function result.
        """
        return self.P is not None and self.K is not None

    def fit(
        self,
        sequence: GraphSnapshotSequence | Sequence[Data],
    ) -> UlamTransferOperatorBaseline:
        """Estimate the Ulam transfer matrix from consecutive snapshots.

        Parameters
        ----------
        sequence : GraphSnapshotSequence or sequence of Data
            Training snapshots with shared topology and length ≥ 2.

        Returns
        -------
        UlamTransferOperatorBaseline
            Fitted baseline (``self``) for sklearn-style chaining.

        Raises
        ------
        ValueError
            If the sequence is too short, has dynamic topology, or the
            product grid exceeds ``4096`` cells.
        """
        resolved = resolve_sequence(sequence)
        require_static_topology(resolved)
        if resolved.num_timesteps < 2:
            msg = "UlamTransferOperatorBaseline.fit requires at least two snapshots"
            raise ValueError(msg)

        states = flatten_snapshots(resolved)
        state_dim = int(states.shape[1])
        n_cells = self.bins_per_dim**state_dim
        if n_cells > _MAX_CELLS:
            msg = (
                "Ulam product grid exceeds "
                f"{_MAX_CELLS} cells "
                f"(bins_per_dim={self.bins_per_dim}, state_dim={state_dim}, "
                f"n_cells={n_cells}); reduce bins_per_dim or state dimension"
            )
            raise ValueError(msg)

        edges = _build_bin_edges(states, self.bins_per_dim)
        cell_ids = _assign_cells(states, edges, self.bins_per_dim)
        counts = torch.zeros(n_cells, n_cells, dtype=states.dtype)
        sources = cell_ids[:-1]
        targets = cell_ids[1:]
        for src, dst in zip(sources.tolist(), targets.tolist(), strict=True):
            counts[src, dst] += 1.0

        row_sums = counts.sum(dim=1)
        p_mat = torch.zeros_like(counts)
        nonempty = row_sums > 0
        p_mat[nonempty] = counts[nonempty] / row_sums[nonempty].unsqueeze(1)
        if (~nonempty).any():
            p_mat[~nonempty] = 1.0 / float(n_cells)

        centers = _cell_centers(edges, self.bins_per_dim, state_dim, states.dtype)

        self.P = p_mat
        self.K = p_mat
        self.bin_edges = edges
        self.cell_centers = centers
        self.n_cells = n_cells
        self.num_nodes = resolved.num_nodes
        self.in_channels = resolved.in_channels
        self.state_dim = state_dim
        self.selected_rank = None
        return self

    def propagate_density(self, rho: Tensor, steps: int) -> Tensor:
        """Propagate a density row-vector under the fitted transfer matrix.

        Parameters
        ----------
        rho : Tensor
            Density with shape ``(n_cells,)`` or ``(batch, n_cells)``.
        steps : int
            Number of transfer steps (``≥ 1``).

        Returns
        -------
        Tensor
            Propagated density with the same leading shape as ``rho``.

        Raises
        ------
        RuntimeError
            If the baseline has not been fit.
        ValueError
            If ``steps < 1`` or ``rho`` width disagrees with ``n_cells``.
        """
        p_mat = self._require_transfer()
        if steps < 1:
            msg = f"steps must be >= 1, got {steps}"
            raise ValueError(msg)
        if rho.shape[-1] != p_mat.shape[0]:
            msg = (
                f"rho trailing dimension must be n_cells={p_mat.shape[0]}, "
                f"got {tuple(rho.shape)}"
            )
            raise ValueError(msg)
        out = rho.to(dtype=p_mat.dtype, device=p_mat.device)
        for _ in range(steps):
            out = out @ p_mat
        return out

    def predict(self, initial_graph: Data, steps: int) -> list[Data]:
        """Predict expected cell-center states from a one-hot density rollout.

        Parameters
        ----------
        initial_graph : Data
            Initial graph snapshot (topology copied onto predictions).
        steps : int
            Forecast horizon (``≥ 1``).

        Returns
        -------
        list of Data
            Predicted snapshots with shape ``(num_nodes, in_channels)``.
        """
        p_mat = self._require_transfer()
        num_nodes, in_channels = self._require_graph_metadata()
        if steps < 1:
            msg = f"steps must be >= 1, got {steps}"
            raise ValueError(msg)
        check_initial_graph(
            initial_graph,
            num_nodes=num_nodes,
            in_channels=in_channels,
        )
        assert self.bin_edges is not None
        assert self.cell_centers is not None
        assert self.bins_per_dim is not None

        state = initial_graph.x.reshape(1, -1)
        cell = int(_assign_cells(state, self.bin_edges, self.bins_per_dim).item())
        rho = torch.zeros(p_mat.shape[0], dtype=p_mat.dtype, device=p_mat.device)
        rho[cell] = 1.0
        topology = copy_topology(initial_graph)
        predictions: list[Data] = []
        for _ in range(steps):
            rho = rho @ p_mat
            expected = rho @ self.cell_centers
            predictions.append(
                Data(x=expected.reshape(num_nodes, in_channels), **topology)
            )
        return predictions

    def spectrum(self) -> KoopmanSpectrum:
        """Return the spectrum of the fitted transfer matrix ``P``.

        Returns
        -------
        KoopmanSpectrum
            Eigendecomposition of :attr:`P` (aliased as :attr:`K`).
        """
        return compute_spectrum(self._require_transfer(), self.time_step)

    def _require_transfer(self) -> Tensor:
        """Return the fitted transfer matrix after a fitted-state check.

        Returns
        -------
        object
            Function result.
        """
        self._check_fitted()
        if self.P is None:
            raise RuntimeError(self._unfitted_message())
        return self.P


def _build_bin_edges(states: Tensor, bins_per_dim: int) -> tuple[Tensor, ...]:
    """Build per-dimension equal-width edges covering training states.

    Parameters
    ----------
    states : Tensor
        Flattened states ``(T, state_dim)``.
    bins_per_dim : int
        Bins along each coordinate.

    Returns
    -------
    tuple of Tensor
        Edge vectors of length ``bins_per_dim + 1`` per dimension.
    """
    lo = states.min(dim=0).values
    hi = states.max(dim=0).values
    span = (hi - lo).clamp(min=_EDGE_EPS)
    lo = lo - _EDGE_EPS * span
    hi = hi + _EDGE_EPS * span
    edges: list[Tensor] = []
    for dim in range(states.shape[1]):
        edges.append(
            torch.linspace(
                float(lo[dim]),
                float(hi[dim]),
                bins_per_dim + 1,
                dtype=states.dtype,
            )
        )
    return tuple(edges)


def _assign_cells(
    states: Tensor,
    edges: tuple[Tensor, ...],
    bins_per_dim: int,
) -> Tensor:
    """Map flattened states to linear cell indices (row-major product grid).

    Parameters
    ----------
    states : Tensor
        States ``(T, state_dim)``.
    edges : tuple of Tensor
        Per-dimension bin edges.
    bins_per_dim : int
        Bins along each coordinate.

    Returns
    -------
    Tensor
        Integer cell ids with shape ``(T,)``.
    """
    state_dim = states.shape[1]
    coords = []
    for dim in range(state_dim):
        # Bucketize right edges exclude the rightmost; clamp into [0, bins-1].
        idx = torch.bucketize(states[:, dim], edges[dim][1:-1], right=False)
        coords.append(idx.clamp(0, bins_per_dim - 1))
    cell = coords[0]
    for dim in range(1, state_dim):
        cell = cell + coords[dim] * (bins_per_dim**dim)
    return cell.to(dtype=torch.long)


def _cell_centers(
    edges: tuple[Tensor, ...],
    bins_per_dim: int,
    state_dim: int,
    dtype: torch.dtype,
) -> Tensor:
    """Return product-grid cell centers with shape ``(n_cells, state_dim)``.

    Parameters
    ----------
    edges : tuple of Tensor
        Per-dimension bin edges.
    bins_per_dim : int
        Bins along each coordinate.
    state_dim : int
        Flattened state dimension.
    dtype : torch.dtype
        Output dtype.

    Returns
    -------
    Tensor
        Cell centers ordered by linear cell index.
    """
    centers_1d = [0.5 * (edge[:-1] + edge[1:]) for edge in edges]
    # Linear index matches _assign_cells: dimension 0 is least significant.
    n_cells = bins_per_dim**state_dim
    out = torch.empty(n_cells, state_dim, dtype=dtype)
    for cell in range(n_cells):
        rem = cell
        for dim in range(state_dim):
            coord = rem % bins_per_dim
            rem //= bins_per_dim
            out[cell, dim] = centers_1d[dim][coord]
    return out
