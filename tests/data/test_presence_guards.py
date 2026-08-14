"""Coverage and error-path tests for :mod:`koopman_graph.data`."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data

from koopman_graph.data import GraphSnapshotSequence


def test_presence_mask_accessor_and_validation_guards() -> None:
    """Presence-mask dtype / accessor / pair-mask edge branches."""
    from koopman_graph.data.validation import validate_presence_masks

    with pytest.raises(ValueError, match="presence_masks must have shape"):
        validate_presence_masks(
            torch.ones(3, dtype=torch.bool),
            num_timesteps=3,
            num_nodes=2,
        )
    with pytest.raises(ValueError, match="boolean or numeric"):
        validate_presence_masks(
            torch.ones(2, 2, dtype=torch.complex64),
            num_timesteps=2,
            num_nodes=2,
        )

    edge = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    snaps = [Data(x=torch.ones(2, 1), edge_index=edge) for _ in range(3)]
    presence = torch.ones(3, 2, dtype=torch.bool)
    presence[1, 0] = False
    sequence = GraphSnapshotSequence(
        snaps,
        presence_masks=presence,
        allow_node_churn=True,
    )
    with pytest.raises(ValueError, match="presence mask index"):
        sequence.presence_mask_at(3)
    with pytest.raises(ValueError, match="pair presence mask index"):
        sequence.pair_presence_mask(2)
    pair = sequence.pair_presence_mask(0)
    assert pair.shape == (2,)
    assert bool(pair[0].item()) is False


def test_resolve_future_presence_at_guards() -> None:
    """Tensor / sequence shape and length guards for forecast presence."""
    from koopman_graph.model.inference import resolve_future_presence_at

    assert resolve_future_presence_at(None, steps=2, num_nodes=3) is None

    schedule = resolve_future_presence_at(
        torch.ones(2, 3, dtype=torch.bool),
        steps=2,
        num_nodes=3,
    )
    assert schedule is not None
    assert schedule(0).shape == (3,)

    with pytest.raises(ValueError, match="future_presence tensor must have shape"):
        resolve_future_presence_at(torch.ones(2, 2), steps=2, num_nodes=3)
    with pytest.raises(ValueError, match="future_presence length"):
        resolve_future_presence_at(
            [torch.ones(3, dtype=torch.bool)],
            steps=2,
            num_nodes=3,
        )
    with pytest.raises(ValueError, match=r"future_presence\[1\] must have shape"):
        resolve_future_presence_at(
            [torch.ones(3, dtype=torch.bool), torch.ones(2, dtype=torch.bool)],
            steps=2,
            num_nodes=3,
        )


def test_containers_presence_observation_accessor_gaps() -> None:
    """Missing-mask raises, presence-only loss masks, hetero index errors."""
    from koopman_graph.data import HeteroGraphSnapshotSequence

    edge = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    snaps = [Data(x=torch.ones(2, 1), edge_index=edge) for _ in range(3)]
    bare = GraphSnapshotSequence(snaps)
    with pytest.raises(ValueError, match="does not contain presence_masks"):
        bare.pair_presence_mask(0)
    with pytest.raises(ValueError, match="does not contain observation_masks"):
        bare.observation_mask_at(0)
    assert bare.loss_mask_at(0) is None

    presence = torch.ones(3, 2, dtype=torch.bool)
    presence[1, 0] = False
    presence_only = GraphSnapshotSequence(
        snaps,
        presence_masks=presence,
        allow_node_churn=True,
    )
    assert presence_only.has_presence_masks
    assert torch.equal(presence_only.loss_mask_at(1), presence[1])
    assert torch.equal(
        presence_only.pair_loss_mask(0),
        presence[0] & presence[1],
    )

    from torch_geometric.data import HeteroData

    hetero_snap = HeteroData()
    hetero_snap["node"].x = torch.randn(3, 2)
    hetero_snap["node", "to", "node"].edge_index = torch.tensor(
        [[0, 1], [1, 2]],
        dtype=torch.long,
    )
    hetero_bare = HeteroGraphSnapshotSequence([hetero_snap, hetero_snap, hetero_snap])
    with pytest.raises(ValueError, match="does not contain presence_masks"):
        hetero_bare.presence_mask_at(0)
    with pytest.raises(ValueError, match="does not contain presence_masks"):
        hetero_bare.pair_presence_mask(0)

    masks = {"node": torch.ones(3, 3, dtype=torch.bool)}
    hetero = HeteroGraphSnapshotSequence(
        [hetero_snap, hetero_snap, hetero_snap],
        presence_masks=masks,
        allow_node_churn=True,
    )
    with pytest.raises(IndexError, match="0 <= index < 3"):
        hetero.presence_mask_at(3)
    with pytest.raises(IndexError, match="0 <= index < 2"):
        hetero.pair_presence_mask(2)
    assert set(hetero.presence_mask_at(0)) == {"node"}
