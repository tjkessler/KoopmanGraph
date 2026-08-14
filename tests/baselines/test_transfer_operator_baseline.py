"""Tests for Ulam transfer-operator baseline (TASK-1842)."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data

import koopman_graph
import koopman_graph.baselines as baselines_pkg
from koopman_graph import GraphSnapshotSequence
from koopman_graph.baselines import UlamTransferOperatorBaseline


def _sequence_from_scalar_states(values: list[float]) -> GraphSnapshotSequence:
    """Build a 1-node, 1-feature static sequence from scalar states."""
    edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    snaps = [
        Data(x=torch.tensor([[v]], dtype=torch.float64), edge_index=edge_index)
        for v in values
    ]
    return GraphSnapshotSequence(snaps)


def test_ulam_rejects_invalid_bins_per_dim() -> None:
    """Constructor rejects non-positive bin counts."""
    with pytest.raises(ValueError, match="bins_per_dim must be >= 1"):
        UlamTransferOperatorBaseline(bins_per_dim=0)


def test_ulam_rejects_single_snapshot_sequence() -> None:
    """Fit requires at least two snapshots."""
    sequence = _sequence_from_scalar_states([0.5])
    with pytest.raises(ValueError, match="at least two snapshots"):
        UlamTransferOperatorBaseline().fit(sequence)


def test_ulam_recovers_deterministic_bin_cycle() -> None:
    """Deterministic cell-to-cell jumps recover a cyclic transfer matrix."""
    # Four well-separated scalars → four bins; always advance to the next cell.
    values = [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0]
    sequence = _sequence_from_scalar_states(values)
    baseline = UlamTransferOperatorBaseline(bins_per_dim=4, time_step=0.5).fit(sequence)
    assert baseline.n_cells == 4
    assert baseline.P is not None
    assert baseline.K is not None
    assert torch.allclose(baseline.P, baseline.K)

    # Each row should be a one-hot to the next cell (mod 4).
    expected = torch.zeros(4, 4, dtype=torch.float64)
    for i in range(4):
        expected[i, (i + 1) % 4] = 1.0
    assert torch.allclose(baseline.P, expected, atol=1e-12)

    rho0 = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    rho1 = baseline.propagate_density(rho0, steps=1)
    assert torch.allclose(rho1, torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float64))
    rho4 = baseline.propagate_density(rho0, steps=4)
    assert torch.allclose(rho4, rho0, atol=1e-12)

    preds = baseline.predict(sequence[0], steps=3)
    assert len(preds) == 3
    assert preds[0].x.shape == (1, 1)
    # Expected centers for cells 1, 2, 3 after starting in cell 0.
    assert baseline.cell_centers is not None
    for step, pred in enumerate(preds, start=1):
        assert torch.allclose(
            pred.x.reshape(-1),
            baseline.cell_centers[(step) % 4],
            atol=1e-10,
        )

    spectrum = baseline.spectrum()
    assert spectrum.eigenvalues.numel() == 4
    assert torch.isfinite(spectrum.eigenvalues).all()


def test_ulam_rejects_dynamic_topology() -> None:
    """Dynamic-topology sequences are rejected at fit."""
    snaps = [
        Data(
            x=torch.randn(2, 1),
            edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        ),
        Data(
            x=torch.randn(2, 1),
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        ),
    ]
    sequence = GraphSnapshotSequence(snaps, allow_dynamic_topology=True)
    assert sequence.is_dynamic_topology
    with pytest.raises(ValueError, match="is_dynamic_topology"):
        UlamTransferOperatorBaseline(bins_per_dim=2).fit(sequence)


def test_ulam_rejects_oversized_grid() -> None:
    """Product grids larger than 4096 cells raise ValueError."""
    # state_dim=3, bins=20 → 8000 cells.
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    snaps = [
        Data(x=torch.randn(1, 3), edge_index=edge_index),
        Data(x=torch.randn(1, 3), edge_index=edge_index),
    ]
    with pytest.raises(ValueError, match="4096"):
        UlamTransferOperatorBaseline(bins_per_dim=20).fit(GraphSnapshotSequence(snaps))


def test_ulam_export_and_not_on_root_all() -> None:
    """Ulam baseline is on baselines.__all__ only (capability module)."""
    assert "UlamTransferOperatorBaseline" in baselines_pkg.__all__
    assert baselines_pkg.UlamTransferOperatorBaseline is UlamTransferOperatorBaseline
    assert "UlamTransferOperatorBaseline" not in set(koopman_graph.__all__)


def test_ulam_uniform_fallback_for_unvisited_source_cells() -> None:
    """Rows with no outgoing mass receive uniform transition probabilities."""
    sequence = _sequence_from_scalar_states([0.0, 0.0, 0.0, 1.0])
    baseline = UlamTransferOperatorBaseline(bins_per_dim=2).fit(sequence)
    assert baseline.P is not None
    # Cell 1 never appears as a transition source; row should be uniform.
    assert float(baseline.P[1].sum()) == pytest.approx(1.0, abs=1e-12)
    assert torch.allclose(
        baseline.P[1],
        torch.full((2,), 0.5, dtype=torch.float64),
        atol=1e-12,
    )


def test_ulam_propagate_density_validation() -> None:
    """propagate_density rejects invalid steps and rho width."""
    sequence = _sequence_from_scalar_states([0.0, 1.0, 0.0, 1.0])
    baseline = UlamTransferOperatorBaseline(bins_per_dim=2).fit(sequence)
    rho0 = torch.tensor([1.0, 0.0], dtype=torch.float64)
    with pytest.raises(ValueError, match="steps must be >= 1"):
        baseline.propagate_density(rho0, steps=0)
    with pytest.raises(ValueError, match="rho trailing dimension"):
        baseline.propagate_density(torch.ones(3), steps=1)


def test_ulam_predict_rejects_invalid_steps() -> None:
    """predict rejects non-positive rollout horizons."""
    sequence = _sequence_from_scalar_states([0.0, 1.0, 0.0, 1.0])
    baseline = UlamTransferOperatorBaseline(bins_per_dim=2).fit(sequence)
    with pytest.raises(ValueError, match="steps must be >= 1"):
        baseline.predict(sequence[0], steps=0)


def test_ulam_require_transfer_when_p_cleared() -> None:
    """_require_transfer raises when the transfer matrix was cleared."""
    sequence = _sequence_from_scalar_states([0.0, 1.0, 0.0, 1.0])
    baseline = UlamTransferOperatorBaseline(bins_per_dim=2).fit(sequence)
    baseline.P = None
    with pytest.raises(RuntimeError, match="must be fit"):
        baseline._require_transfer()


def test_ulam_multidimensional_state_bin_assignment() -> None:
    """Two-dimensional flattened states use the product grid cell index."""
    edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    snaps = [
        Data(x=torch.tensor([[0.0, 0.0]], dtype=torch.float64), edge_index=edge_index),
        Data(x=torch.tensor([[1.0, 0.0]], dtype=torch.float64), edge_index=edge_index),
        Data(x=torch.tensor([[0.0, 1.0]], dtype=torch.float64), edge_index=edge_index),
        Data(x=torch.tensor([[1.0, 1.0]], dtype=torch.float64), edge_index=edge_index),
    ]
    baseline = UlamTransferOperatorBaseline(bins_per_dim=2).fit(
        GraphSnapshotSequence(snaps)
    )
    assert baseline.n_cells == 4
    assert baseline.P is not None
    assert torch.isfinite(baseline.P).all()


def test_ulam_predict_requires_fit() -> None:
    """predict / propagate_density before fit raise RuntimeError."""
    baseline = UlamTransferOperatorBaseline()
    snap = Data(
        x=torch.zeros(1, 1),
        edge_index=torch.tensor([[0], [0]], dtype=torch.long),
    )
    with pytest.raises(RuntimeError, match="must be fit"):
        baseline.predict(snap, steps=1)
    with pytest.raises(RuntimeError, match="must be fit"):
        baseline.propagate_density(torch.ones(4), steps=1)
