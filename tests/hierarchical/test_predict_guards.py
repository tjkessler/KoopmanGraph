"""Coverage and error-path tests for :mod:`koopman_graph.hierarchical`."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data

from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.hierarchical import HierarchicalGraphKoopmanModel


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


def test_hierarchical_predict_rejects_future_presence() -> None:
    """Presence-mask kwargs are accepted for parity but must stay None."""
    hier = HierarchicalGraphKoopmanModel(_tiny_model(), pool_ratios=(0.5,))
    sequence = _tiny_sequence()
    preds = hier.predict(sequence[0], steps=1, future_presence=None)
    assert len(preds) == 1
    with pytest.raises(ValueError, match="does not support future_presence"):
        hier.predict(
            sequence[0],
            steps=1,
            future_presence=torch.ones(1, sequence.num_nodes, dtype=torch.bool),
        )
