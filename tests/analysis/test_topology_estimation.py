"""Tests for DMD-estimated coupling (TASK-1309)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch
from torch_geometric.data import Data

from koopman_graph.analysis import CouplingEstimate, estimate_coupling_from_snapshots
from koopman_graph.data import GraphSnapshotSequence


def _path_edge_index(num_nodes: int) -> torch.Tensor:
    edges: list[list[int]] = []
    for node in range(num_nodes - 1):
        edges.extend([[node, node + 1], [node + 1, node]])
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _linear_diffusion_sequence(
    *,
    num_nodes: int = 4,
    num_timesteps: int = 40,
    seed: int = 0,
) -> tuple[GraphSnapshotSequence, set[tuple[int, int]], torch.Tensor]:
    """Build a noiseless linear diffusion trajectory with planted off-diagonals."""
    torch.manual_seed(seed)
    edge_index = _path_edge_index(num_nodes)
    true_edges = {
        (int(edge_index[0, e]), int(edge_index[1, e]))
        for e in range(edge_index.shape[1])
    }
    k_true = 0.55 * torch.eye(num_nodes)
    for i, j in true_edges:
        k_true[i, j] = 0.2
    x = torch.randn(num_nodes, 1)
    snapshots: list[Data] = [Data(x=x.clone(), edge_index=edge_index)]
    for _ in range(num_timesteps - 1):
        # Package convention: x_next = x @ K.T with row feature layout (N, F=1).
        flat = x.reshape(1, -1)
        x = (flat @ k_true.T).reshape(num_nodes, 1)
        snapshots.append(Data(x=x.clone(), edge_index=edge_index))
    return GraphSnapshotSequence(snapshots), true_edges, k_true.abs()


def test_recovers_planted_diffusion_coupling() -> None:
    """Thresholded edges recover the planted path support with high precision."""
    sequence, true_edges, _ = _linear_diffusion_sequence()
    estimate = estimate_coupling_from_snapshots(sequence, threshold=0.1)
    assert isinstance(estimate, CouplingEstimate)
    assert estimate.coupling.shape == (4, 4)
    assert estimate.rank is None

    recovered = {
        (int(estimate.edge_index[0, e]), int(estimate.edge_index[1, e]))
        for e in range(estimate.edge_index.shape[1])
    }
    # Precision: fraction of recovered edges that are true.
    assert recovered
    precision = len(recovered & true_edges) / len(recovered)
    recall = len(recovered & true_edges) / len(true_edges)
    assert precision >= 0.9
    assert recall >= 0.9


def test_coupling_estimate_is_frozen() -> None:
    """CouplingEstimate rejects attribute assignment."""
    sequence, _, _ = _linear_diffusion_sequence(num_timesteps=5)
    estimate = estimate_coupling_from_snapshots(sequence, threshold=0.05)
    with pytest.raises(FrozenInstanceError):
        estimate.rank = 1  # type: ignore[misc]


def test_excludes_diagonal_from_coo() -> None:
    """Self-loops stay in dense coupling but not in thresholded edges."""
    sequence, _, _ = _linear_diffusion_sequence(num_timesteps=8)
    estimate = estimate_coupling_from_snapshots(sequence, threshold=0.0)
    assert torch.diagonal(estimate.coupling).min() > 0
    if estimate.edge_index.numel() > 0:
        assert not torch.any(estimate.edge_index[0] == estimate.edge_index[1])


def test_rejects_dynamic_topology() -> None:
    """Dynamic-topology sequences raise like classical baselines."""
    edge_a = _path_edge_index(3)
    edge_b = torch.tensor([[0, 2], [2, 0]], dtype=torch.long)
    snapshots = [
        Data(x=torch.randn(3, 1), edge_index=edge_a),
        Data(x=torch.randn(3, 1), edge_index=edge_b),
    ]
    sequence = GraphSnapshotSequence(snapshots, allow_dynamic_topology=True)
    with pytest.raises(ValueError, match="fixed graph topology"):
        estimate_coupling_from_snapshots(sequence, threshold=0.1)


def test_rejects_hyperedge_sequence() -> None:
    """Hyperedge-carrying sequences are rejected."""
    edge_index = _path_edge_index(3)
    hyperedge_index = torch.tensor([[0, 1, 1, 2], [0, 0, 1, 1]], dtype=torch.long)
    snapshots = [
        Data(
            x=torch.randn(3, 1),
            edge_index=edge_index,
            hyperedge_index=hyperedge_index,
        ),
        Data(
            x=torch.randn(3, 1),
            edge_index=edge_index,
            hyperedge_index=hyperedge_index,
        ),
    ]
    sequence = GraphSnapshotSequence(snapshots)
    with pytest.raises(ValueError, match="hyperedge"):
        estimate_coupling_from_snapshots(sequence, threshold=0.1)


def test_rejects_short_sequence_and_negative_threshold() -> None:
    """Fewer than two snapshots / negative threshold raise."""
    edge_index = _path_edge_index(2)
    with pytest.raises(ValueError, match="at least two snapshots"):
        estimate_coupling_from_snapshots(
            GraphSnapshotSequence([Data(x=torch.randn(2, 1), edge_index=edge_index)]),
            threshold=0.1,
        )
    sequence, _, _ = _linear_diffusion_sequence(num_timesteps=3)
    with pytest.raises(ValueError, match="non-negative"):
        estimate_coupling_from_snapshots(sequence, threshold=-0.01)


def test_multifeature_block_frobenius_smoke() -> None:
    """F>1 uses block-Frobenius reduction and returns N×N coupling."""
    torch.manual_seed(2)
    num_nodes = 3
    in_channels = 2
    edge_index = _path_edge_index(num_nodes)
    state_dim = num_nodes * in_channels
    k_true = 0.6 * torch.eye(state_dim)
    # Couple node 0→1 feature blocks.
    k_true[0:2, 2:4] = 0.15
    flat = torch.randn(1, state_dim)
    snapshots: list[Data] = []
    for _ in range(20):
        x = flat.reshape(num_nodes, in_channels)
        snapshots.append(Data(x=x.clone(), edge_index=edge_index))
        flat = flat @ k_true.T
    estimate = estimate_coupling_from_snapshots(
        GraphSnapshotSequence(snapshots),
        threshold=0.05,
        rank=None,
    )
    assert estimate.coupling.shape == (num_nodes, num_nodes)
    assert estimate.coupling[0, 1] > estimate.coupling[0, 2]


def test_rank_truncation_smoke() -> None:
    """Optional rank is forwarded and stored on the result."""
    sequence, _, _ = _linear_diffusion_sequence(num_timesteps=12)
    estimate = estimate_coupling_from_snapshots(sequence, rank=2, threshold=0.05)
    assert estimate.rank == 2
    assert estimate.coupling.shape == (4, 4)
