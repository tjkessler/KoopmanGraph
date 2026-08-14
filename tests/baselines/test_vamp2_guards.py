"""Coverage and error-path tests for :mod:`koopman_graph.baselines`."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch_geometric.data import Data

from koopman_graph import GraphSnapshotSequence
from koopman_graph.training.extra_objectives import compute_vamp2_loss


def test_vamp2_short_sequence_guard() -> None:
    """VAMP-2 requires at least one lagged snapshot pair."""
    model = nn.Linear(2, 2)
    sequence = GraphSnapshotSequence(
        [Data(x=torch.zeros(2, 2), edge_index=torch.empty(2, 0, dtype=torch.long))]
    )
    with pytest.raises(ValueError, match="at least 2 snapshots"):
        compute_vamp2_loss(model, sequence, weight=1.0)  # type: ignore[arg-type]
