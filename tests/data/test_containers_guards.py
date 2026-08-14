"""Coverage and error-path tests for :mod:`koopman_graph.data`."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data

from koopman_graph.data import GraphSnapshotSequence


def _edge_index() -> torch.Tensor:
    return torch.tensor([[0, 1], [1, 0]], dtype=torch.long)


def test_containers_timestamp_and_mask_validation() -> None:
    edge = _edge_index()
    snapshots = [Data(x=torch.randn(2, 2), edge_index=edge) for _ in range(3)]
    with pytest.raises(ValueError, match="timestamps must have shape"):
        GraphSnapshotSequence(snapshots, timestamps=torch.ones(2, 2))
    with pytest.raises(ValueError, match="entries, expected"):
        GraphSnapshotSequence(snapshots, timestamps=torch.arange(2.0))
    with pytest.raises(ValueError, match="observation_masks"):
        GraphSnapshotSequence(
            snapshots,
            observation_masks=torch.ones(2, 2, dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="observation_masks"):
        GraphSnapshotSequence(
            snapshots,
            observation_masks=torch.ones(3, 3, dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="only 0 and 1"):
        GraphSnapshotSequence(
            snapshots,
            observation_masks=torch.full((3, 2), 2, dtype=torch.int64),
        )

    sequence = GraphSnapshotSequence(snapshots)
    with pytest.raises(ValueError, match="does not contain timestamps"):
        sequence.delta_t_at(0)
    with pytest.raises(ValueError, match="does not contain observation_masks"):
        sequence.observation_mask_at(0)
    with pytest.raises(ValueError, match="does not contain observation_masks"):
        sequence.pair_observation_mask(0)

    stamped = GraphSnapshotSequence(
        snapshots,
        timestamps=torch.tensor([0.0, 0.5, 1.0]),
    )
    with pytest.raises(ValueError, match="out of range"):
        stamped.delta_t_at(2)

    masked = GraphSnapshotSequence(
        snapshots,
        observation_masks=torch.ones(3, 2, dtype=torch.bool),
    )
    with pytest.raises(ValueError, match="out of range"):
        masked.observation_mask_at(3)
    with pytest.raises(ValueError, match="pair_observation_mask|out of range"):
        masked.pair_observation_mask(2)
