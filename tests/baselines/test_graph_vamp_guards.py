"""Coverage and error-path tests for :mod:`koopman_graph.baselines`."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
from torch_geometric.data import Data

from koopman_graph.baselines.graph_vamp import GraphVAMPBaseline


def test_graph_vamp_fit_and_score_guards() -> None:
    """Fit/score edge-resolution and lag/epochs validation."""
    edge = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    snaps = [Data(x=torch.randn(2, 3), edge_index=edge) for _ in range(4)]
    model = GraphVAMPBaseline(in_channels=3, hidden_channels=4, latent_dim=2)

    with pytest.raises(ValueError, match="at least one snapshot"):
        model.fit([])
    with pytest.raises(ValueError, match="lag must be >= 1"):
        model.fit(snaps, lag=0)
    with pytest.raises(ValueError, match="num_timesteps > lag"):
        model.fit(snaps, lag=4)
    with pytest.raises(ValueError, match="epochs must be >= 1"):
        model.fit(snaps, epochs=0)
    with pytest.raises(ValueError, match="either edge_index or positions_nm"):
        model.fit(snaps, edge_index=edge, positions_nm=torch.zeros(2, 3))
    with pytest.raises(ValueError, match="positions_nm is required"):
        model.fit(snaps, cutoff_nm=0.5)
    with pytest.raises(ValueError, match="cutoff_nm is required"):
        model.fit(snaps, positions_nm=torch.zeros(2, 3))

    from koopman_graph.baselines.graph_vamp import _resolve_fit_edges

    traj = MagicMock()
    traj.__getitem__.return_value = Data(x=torch.randn(2, 3))
    with pytest.raises(ValueError, match="could not resolve contact edges"):
        _resolve_fit_edges(
            traj,
            edge_index=None,
            positions_nm=None,
            cutoff_nm=None,
        )

    with pytest.raises(RuntimeError, match="score requires a prior fit"):
        model.score(snaps)

    encoded = model.encode_frame(snaps[0])
    assert encoded.shape == (2,)
