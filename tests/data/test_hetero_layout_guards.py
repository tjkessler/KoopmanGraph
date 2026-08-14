"""Coverage and error-path tests for :mod:`koopman_graph.data`."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import HeteroData

from koopman_graph.data.hetero_layout import (
    latent_type_slices,
    mask_hetero_snapshot_features,
    snapshot_num_nodes_dict,
    stack_typed_features,
)


def _typed_snapshot() -> HeteroData:
    """Build a two-type snapshot with one cross relation."""
    data = HeteroData()
    data["gen"].x = torch.randn(2, 3)
    data["load"].x = torch.randn(3, 2)
    data["gen", "feeds", "load"].edge_index = torch.tensor(
        [[0, 1], [0, 2]],
        dtype=torch.long,
    )
    return data


def test_hetero_layout_and_mask_helpers() -> None:
    """Typed layout helpers reject malformed slices, stacks, and masks."""
    with pytest.raises(ValueError, match="latent_dim must be positive"):
        latent_type_slices({"a": slice(0, 2)}, latent_dim=0)
    with pytest.raises(ValueError, match="bounded slice"):
        latent_type_slices({"a": slice(None, 2)}, latent_dim=2)
    with pytest.raises(ValueError, match="stop >= start"):
        latent_type_slices({"a": slice(3, 1)}, latent_dim=2)

    snap = _typed_snapshot()
    with pytest.raises(ValueError, match="missing node type"):
        snapshot_num_nodes_dict(snap, ("gen", "bus"))
    with pytest.raises(ValueError, match="missing node type"):
        stack_typed_features({"gen": torch.randn(2, 3)}, ("gen", "load"))
    with pytest.raises(ValueError, match="must have shape"):
        stack_typed_features(
            {"gen": torch.randn(4), "load": torch.randn(3, 2)},
            ("gen", "load"),
        )
    with pytest.raises(ValueError, match="shared trailing width"):
        stack_typed_features(
            {"gen": torch.randn(2, 3), "load": torch.randn(3, 4)},
            ("gen", "load"),
        )
    with pytest.raises(ValueError, match="expected"):
        mask_hetero_snapshot_features(
            snap,
            {
                "gen": torch.ones(3, dtype=torch.bool),
                "load": torch.ones(3, dtype=torch.bool),
            },
        )
