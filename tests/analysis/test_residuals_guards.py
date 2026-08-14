"""Coverage and error-path tests for :mod:`koopman_graph.analysis`."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data

from koopman_graph import (
    GNNDecoder,
    GNNEncoder,
    GraphKoopmanModel,
    GraphSnapshotSequence,
)
from koopman_graph.analysis.residuals import (
    _compute_mode_residuals,
    _prepare_amplitude_state,
    _resolve_spectrum,
    spectral_residuals,
)
from koopman_graph.spectrum_types import compute_spectrum


def _path_edges(num_nodes: int = 3) -> torch.Tensor:
    sources = list(range(num_nodes - 1)) + list(range(1, num_nodes))
    targets = list(range(1, num_nodes)) + list(range(num_nodes - 1))
    return torch.tensor([sources, targets], dtype=torch.long)


def test_residuals_hypergraph_and_encode_override_paths() -> None:
    """Hypergraph spectrum kwargs, encode overrides, and delta_t guards."""
    edge_index = _path_edges(3)
    # Missing hyperedge_index on a hypergraph model.
    hyp = GraphKoopmanModel(
        encoder=GNNEncoder(2, 4, 2, num_layers=1),
        decoder=GNNDecoder(2, 4, 2, num_layers=1),
        latent_dim=2,
        koopman="hypergraph",
        time_step=1.0,
    )
    seq = GraphSnapshotSequence(
        [
            Data(x=torch.randn(3, 2), edge_index=edge_index),
            Data(x=torch.randn(3, 2), edge_index=edge_index),
        ]
    )
    with pytest.raises(ValueError, match="hyperedge_index"):
        spectral_residuals(hyp, seq)

    # Continuous graph with non-positive delta_t.
    cg = GraphKoopmanModel(
        encoder=GNNEncoder(2, 4, 2, num_layers=1),
        decoder=GNNDecoder(2, 4, 2, num_layers=1),
        latent_dim=2,
        dynamics_mode="continuous",
        koopman="continuous_graph",
        time_step=0.1,
    )
    with pytest.raises(ValueError, match="delta_t must be positive"):
        spectral_residuals(cg, seq, delta_t=0.0)

    # Edge override forces the encode(snapshot, edges, weights) branch.
    model = GraphKoopmanModel(
        encoder=GNNEncoder(2, 4, 2, num_layers=1),
        decoder=GNNDecoder(2, 4, 2, num_layers=1),
        latent_dim=2,
        time_step=1.0,
    )
    report = spectral_residuals(model, seq, edge_index=edge_index)
    assert report.num_pairs == 1
    assert torch.all(torch.isfinite(report.residuals))


def test_residuals_private_helpers_validation() -> None:
    """Cover residual helper ValueError branches."""
    spectrum = compute_spectrum(0.8 * torch.eye(2), time_step=1.0)
    with pytest.raises(ValueError, match="at least two latent frames"):
        _compute_mode_residuals(spectrum, [torch.randn(3, 2)])
    with pytest.raises(ValueError, match="at least 1-D"):
        _prepare_amplitude_state(torch.tensor(1.0), 2)
    with pytest.raises(ValueError, match="incompatible with spectrum"):
        _prepare_amplitude_state(torch.randn(5), 2)

    # Hypergraph kwargs assembly when hyperedge_index is present.
    hyp = GraphKoopmanModel(
        encoder=GNNEncoder(2, 4, 2, num_layers=1),
        decoder=GNNDecoder(2, 4, 2, num_layers=1),
        latent_dim=2,
        koopman="hypergraph",
        time_step=1.0,
    )
    hyperedge_index = torch.tensor([[0, 1, 2], [0, 0, 0]], dtype=torch.long)
    hyp_seq = GraphSnapshotSequence(
        [
            Data(
                x=torch.randn(3, 2),
                edge_index=_path_edges(3),
                hyperedge_index=hyperedge_index,
            ),
            Data(
                x=torch.randn(3, 2),
                edge_index=_path_edges(3),
                hyperedge_index=hyperedge_index,
            ),
        ]
    )
    resolved = _resolve_spectrum(
        hyp, hyp_seq, edge_index=None, edge_weight=None, delta_t=None
    )
    assert resolved.eigenvalues.numel() == 6
