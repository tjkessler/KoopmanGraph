"""Coverage and error-path tests for :mod:`koopman_graph.analysis`."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data

from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
from koopman_graph.data import GraphSnapshotSequence


def _tiny_sequence() -> GraphSnapshotSequence:
    edge = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    return GraphSnapshotSequence(
        [Data(x=torch.ones(2, 3) * (0.9**t), edge_index=edge) for t in range(6)]
    )


def _tiny_model(*, seed: int = 0) -> GraphKoopmanModel:
    torch.manual_seed(seed)
    return GraphKoopmanModel(
        encoder=GNNEncoder(3, 8, 4, num_layers=1),
        decoder=GNNDecoder(4, 8, 3, num_layers=1),
        latent_dim=4,
        time_step=0.1,
    )


def test_topology_transfer_and_implied_timescale_guards() -> None:
    """Easy evaluate_topology_transfer and implied_timescales input guards."""
    from koopman_graph.analysis import (
        evaluate_topology_transfer,
        implied_timescales,
    )

    with pytest.raises(TypeError, match="GraphKoopmanModel architecture template"):
        evaluate_topology_transfer(
            object(),  # type: ignore[arg-type]
            _tiny_sequence(),
            _tiny_sequence(),
            _tiny_sequence(),
            steps=1,
        )
    with pytest.raises(ValueError, match="steps must be >= 1"):
        evaluate_topology_transfer(
            _tiny_model(),
            _tiny_sequence(),
            _tiny_sequence(),
            _tiny_sequence(),
            steps=0,
        )
    with pytest.raises(ValueError, match="controls must be a non-empty"):
        evaluate_topology_transfer(
            _tiny_model(),
            _tiny_sequence(),
            _tiny_sequence(),
            _tiny_sequence(),
            controls=(),
            steps=1,
        )

    eigenvalues = torch.tensor([0.5], dtype=torch.float64)
    with pytest.raises(ValueError, match="magnitude_atol must be > 0"):
        implied_timescales(eigenvalues, lag_steps=1, magnitude_atol=0.0)
    with pytest.raises(ValueError, match="magnitude_atol must be < 0.5"):
        implied_timescales(eigenvalues, lag_steps=1, magnitude_atol=0.5)
