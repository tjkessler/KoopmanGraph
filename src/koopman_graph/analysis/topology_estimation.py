"""DMD-estimated inter-node coupling from graph snapshots.

Fits a flattened linear map with the classical DMD least-squares helper
(:func:`~koopman_graph.baselines.base.fit_row_operator`), then reduces the
``(N·F)×(N·F)`` operator to an ``N×N`` coupling magnitude matrix. Intended for
diagnostics and warm-starting adaptive topology — a **linear coupling
estimate**, not causal structure discovery.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor
from torch_geometric.data import Data

from koopman_graph.baselines.base import (
    fit_row_operator,
    flatten_snapshots,
    require_static_topology,
)
from koopman_graph.data import GraphSnapshotSequence, resolve_sequence


@dataclass(frozen=True)
class CouplingEstimate:
    """Data-driven inter-node coupling recovered from snapshot DMD.

    Attributes
    ----------
    coupling : Tensor
        Dense coupling magnitudes with shape ``(num_nodes, num_nodes)``.
        Entry ``(i, j)`` is ``|K_ij|`` when ``in_channels == 1``, otherwise the
        Frobenius norm of the ``(i, j)`` feature block of the flattened DMD
        operator.
    edge_index : Tensor
        Thresholded directed edges with shape ``(2, E)`` (diagonal excluded).
    edge_weight : Tensor
        Coupling magnitudes for those edges with shape ``(E,)``.
    rank : int or None
        Truncated-SVD rank passed to the DMD fit, or ``None`` for full
        least squares.
    """

    coupling: Tensor
    edge_index: Tensor
    edge_weight: Tensor
    rank: int | None


def _block_frobenius_coupling(
    operator: Tensor,
    num_nodes: int,
    in_channels: int,
) -> Tensor:
    """Reduce a flattened DMD operator to an ``N×N`` block-Frobenius matrix.

    Parameters
    ----------

    operator : Tensor
        See the function signature / summary for ``operator``.
    num_nodes : int
        See the function signature / summary for ``num_nodes``.
    in_channels : int
        See the function signature / summary for ``in_channels``.

    Returns
    -------

    Tensor
        See summary line.

    Raises
    ------

    ValueError
        Raised when inputs are invalid."""
    state_dim = num_nodes * in_channels
    if operator.shape != (state_dim, state_dim):
        msg = (
            f"operator must have shape {(state_dim, state_dim)}, "
            f"got {tuple(operator.shape)}"
        )
        raise ValueError(msg)
    if in_channels == 1:
        return operator.abs()

    coupling = torch.empty(
        num_nodes,
        num_nodes,
        dtype=operator.dtype,
        device=operator.device,
    )
    for row in range(num_nodes):
        row_slice = slice(row * in_channels, (row + 1) * in_channels)
        for col in range(num_nodes):
            col_slice = slice(col * in_channels, (col + 1) * in_channels)
            block = operator[row_slice, col_slice]
            coupling[row, col] = torch.linalg.vector_norm(block)
    return coupling


def _threshold_edges(coupling: Tensor, threshold: float) -> tuple[Tensor, Tensor]:
    """Build directed COO edges with ``|C_ij| >= threshold``, excluding diagonal.

    Parameters
    ----------

    coupling : Tensor
        See the function signature / summary for ``coupling``.
    threshold : float
        See the function signature / summary for ``threshold``.

    Returns
    -------

    tuple[Tensor, Tensor]
        See summary line.

    Raises
    ------

    ValueError
        Raised when inputs are invalid."""
    if threshold < 0:
        msg = f"threshold must be non-negative, got {threshold}"
        raise ValueError(msg)
    num_nodes = coupling.shape[0]
    mask = coupling >= threshold
    eye = torch.eye(num_nodes, dtype=torch.bool, device=coupling.device)
    mask = mask & ~eye
    rows, cols = torch.nonzero(mask, as_tuple=True)
    edge_index = torch.stack([rows, cols], dim=0)
    edge_weight = coupling[rows, cols]
    return edge_index, edge_weight


def estimate_coupling_from_snapshots(
    sequence: GraphSnapshotSequence | Sequence[Data],
    *,
    rank: int | None = None,
    threshold: float,
) -> CouplingEstimate:
    """Estimate inter-node coupling via flattened DMD least squares.

    Parameters
    ----------
    sequence : GraphSnapshotSequence or sequence of Data
        Static-topology snapshots (no hyperedges). Needs at least two frames.
    rank : int or None, optional
        Optional truncated-SVD rank for
        :func:`~koopman_graph.baselines.base.fit_row_operator`. ``None`` uses
        full least squares.
    threshold : float
        Absolute floor on coupling magnitude for the returned COO edges.
        Required (no silent default).

    Returns
    -------
    CouplingEstimate
        Dense coupling matrix, thresholded directed edges (no self-loops), and
        the rank used for the fit.

    Raises
    ------
    ValueError
        If the sequence has dynamic topology or hyperedges, fewer than two
        snapshots, invalid ``rank`` / ``threshold``, or inconsistent shapes.
    """
    resolved = resolve_sequence(sequence)
    require_static_topology(resolved)
    if resolved.num_timesteps < 2:
        msg = "estimate_coupling_from_snapshots requires at least two snapshots"
        raise ValueError(msg)

    states = flatten_snapshots(resolved)
    operator = fit_row_operator(states[:-1], states[1:], rank)
    num_nodes = resolved.num_nodes
    in_channels = resolved.in_channels
    coupling = _block_frobenius_coupling(operator, num_nodes, in_channels)
    edge_index, edge_weight = _threshold_edges(coupling, threshold)
    return CouplingEstimate(
        coupling=coupling,
        edge_index=edge_index,
        edge_weight=edge_weight,
        rank=rank,
    )
