"""Hetero per-type presence masks / node churn (TASK-1919).

Presence marks whether an entity **exists** in each type's fixed union
``N_τ``; observation marks whether an existing entity was **measured**.
``allow_node_churn`` gates presence drops. Losses use ``loss_mask_at`` /
``pair_loss_mask`` (present ∧ observed). Rectangular ``latent_dims`` reject
stacked latent consistency under presence; feature-space losses remain OK.
"""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import HeteroData

from koopman_graph.data import HeteroGraphSnapshotSequence
from koopman_graph.data.validation import (
    validate_hetero_presence_masks,
    validate_node_churn_policy,
)
from koopman_graph.model import GraphKoopmanModel
from koopman_graph.nn import RelGraphDecoder, RelGraphEncoder
from koopman_graph.operators import HeteroGraphKoopmanOperator
from koopman_graph.training.device import sequence_to_device
from koopman_graph.training.pair_objectives import (
    _FORWARD_CONSISTENCY_LOSS,
    _hetero_pair_mask,
    _hetero_target_masks,
    _one_step_pair,
)


def _multiplex_snapshot(*, num_nodes: int = 4, in_channels: int = 3) -> HeteroData:
    data = HeteroData()
    data["node"].x = torch.randn(num_nodes, in_channels)
    data["node", "r1", "node"].edge_index = torch.tensor(
        [[0, 1, 2], [1, 2, 0]],
        dtype=torch.long,
    )
    data["node", "r2", "node"].edge_index = torch.tensor(
        [[0, 2], [2, 3]],
        dtype=torch.long,
    )
    return data


def _typed_snapshot() -> HeteroData:
    data = HeteroData()
    data["gen"].x = torch.randn(2, 3)
    data["load"].x = torch.randn(3, 2)
    data["gen", "feeds", "load"].edge_index = torch.tensor(
        [[0, 1], [0, 2]],
        dtype=torch.long,
    )
    return data


def _multiplex_presence(
    num_timesteps: int,
    num_nodes: int = 4,
) -> dict[str, torch.Tensor]:
    masks = torch.ones(num_timesteps, num_nodes, dtype=torch.bool)
    if num_timesteps >= 3 and num_nodes >= 2:
        masks[2:, -1] = False
    return {"node": masks}


def _multiplex_model(*, latent_dim: int = 4) -> GraphKoopmanModel:
    return GraphKoopmanModel(
        encoder=RelGraphEncoder(
            3,
            hidden_channels=8,
            latent_dim=latent_dim,
            num_relations=2,
            num_layers=1,
        ),
        decoder=RelGraphDecoder(
            latent_dim=latent_dim,
            hidden_channels=8,
            out_channels=3,
            num_relations=2,
            num_layers=1,
        ),
        latent_dim=latent_dim,
        time_step=1.0,
        koopman="hetero_graph",
    )


def test_hetero_presence_masks_validation_and_accessors() -> None:
    """Accept validated per-type presence when churn is enabled."""
    snapshots = [_multiplex_snapshot() for _ in range(3)]
    presence = _multiplex_presence(3)
    sequence = HeteroGraphSnapshotSequence(
        snapshots,
        presence_masks=presence,
        allow_node_churn=True,
    )

    assert sequence.has_presence_masks
    assert sequence.allow_node_churn
    assert sequence.presence_masks is not None
    assert torch.equal(sequence.presence_masks["node"], presence["node"])
    assert torch.equal(sequence.presence_mask_at(2)["node"], presence["node"][2])
    assert not bool(presence["node"][2, -1].item())


def test_hetero_presence_rejects_bad_keys_shapes_and_empty_type() -> None:
    """Reject key mismatches, bad shapes, and empty-type timesteps."""
    snapshots = [_multiplex_snapshot() for _ in range(3)]

    with pytest.raises(ValueError, match="presence_masks keys"):
        HeteroGraphSnapshotSequence(
            snapshots,
            presence_masks={"wrong": torch.ones(3, 4, dtype=torch.bool)},
            allow_node_churn=True,
        )
    with pytest.raises(ValueError, match=r"presence_masks\['node'\]"):
        HeteroGraphSnapshotSequence(
            snapshots,
            presence_masks={"node": torch.ones(3, 3, dtype=torch.bool)},
            allow_node_churn=True,
        )
    with pytest.raises(ValueError, match="at least one present entity"):
        HeteroGraphSnapshotSequence(
            snapshots,
            presence_masks={"node": torch.zeros(3, 4, dtype=torch.bool)},
            allow_node_churn=True,
        )


def test_validate_hetero_presence_masks_coerces_numeric() -> None:
    """Numeric 0/1 per-type presence coerces to bool."""
    masks = validate_hetero_presence_masks(
        {"node": torch.tensor([[1, 0], [1, 1]], dtype=torch.float32)},
        num_timesteps=2,
        num_nodes={"node": 2},
    )
    assert masks["node"].dtype == torch.bool
    assert torch.equal(
        masks["node"],
        torch.tensor([[True, False], [True, True]]),
    )


def test_allow_node_churn_false_rejects_hetero_presence_drops() -> None:
    """Default churn flag rejects per-type presence drops."""
    snapshots = [_multiplex_snapshot() for _ in range(3)]
    with pytest.raises(ValueError, match="allow_node_churn=False"):
        HeteroGraphSnapshotSequence(
            snapshots,
            presence_masks=_multiplex_presence(3),
        )


def test_allow_node_churn_true_requires_presence() -> None:
    """Churn mode without presence masks is rejected."""
    snapshots = [_multiplex_snapshot() for _ in range(2)]
    with pytest.raises(ValueError, match="requires presence_masks"):
        HeteroGraphSnapshotSequence(snapshots, allow_node_churn=True)


def test_validate_node_churn_policy_accepts_hetero_mapping() -> None:
    """Policy helper accepts per-type presence mappings."""
    masks = {"a": torch.ones(2, 2, dtype=torch.bool)}
    validate_node_churn_policy(allow_node_churn=True, presence_masks=masks)
    validate_node_churn_policy(allow_node_churn=False, presence_masks=masks)


def test_loss_mask_composes_presence_and_observation() -> None:
    """Per-type loss masks are present ∧ observed."""
    snapshots = [_multiplex_snapshot() for _ in range(3)]
    presence = _multiplex_presence(3)
    observation = {
        "node": torch.tensor(
            [
                [True, True, True, True],
                [True, True, False, True],
                [True, True, True, True],
            ]
        )
    }
    sequence = HeteroGraphSnapshotSequence(
        snapshots,
        observation_masks=observation,
        presence_masks=presence,
        allow_node_churn=True,
    )

    loss_t2 = sequence.loss_mask_at(2)
    assert loss_t2 is not None
    # Presence drops node 3; observation is all-True at t=2.
    assert torch.equal(
        loss_t2["node"],
        torch.tensor([True, True, True, False]),
    )

    pair = sequence.pair_loss_mask(1)
    assert pair is not None
    # Pair AND: obs at t=1 drops node 2; presence AND drops node 3 at t=2.
    assert torch.equal(
        pair["node"],
        torch.tensor([True, True, False, False]),
    )


def test_no_presence_matches_observation_only_baseline() -> None:
    """Absent presence keeps observation-only / unmasked hetero behavior."""
    snapshots = [_multiplex_snapshot() for _ in range(3)]
    observation = {
        "node": torch.ones(3, 4, dtype=torch.bool),
    }
    observation["node"][1, 0] = False
    sequence = HeteroGraphSnapshotSequence(snapshots, observation_masks=observation)

    assert not sequence.has_presence_masks
    loss_t1 = sequence.loss_mask_at(1)
    obs_t1 = sequence.observation_mask_at(1)
    assert loss_t1 is not None
    assert torch.equal(loss_t1["node"], obs_t1["node"])
    pair = sequence.pair_loss_mask(0)
    pair_obs = sequence.pair_observation_mask(0)
    assert pair is not None
    assert torch.equal(pair["node"], pair_obs["node"])
    target = _hetero_target_masks(sequence, 1)
    assert target is not None
    assert torch.equal(target["node"], obs_t1["node"])


def test_slice_propagates_presence_and_churn_flag() -> None:
    """Temporal slice keeps presence masks and allow_node_churn."""
    snapshots = [_multiplex_snapshot() for _ in range(4)]
    presence = _multiplex_presence(4)
    sequence = HeteroGraphSnapshotSequence(
        snapshots,
        presence_masks=presence,
        allow_node_churn=True,
    )
    sliced = sequence.slice(1, 4)
    assert sliced.allow_node_churn
    assert sliced.has_presence_masks
    assert sliced.presence_masks is not None
    assert sliced.num_timesteps == 3
    assert torch.equal(sliced.presence_masks["node"], presence["node"][1:4])


def test_sequence_to_device_preserves_hetero_presence() -> None:
    """Device move keeps hetero presence and churn flag."""
    snapshots = [_multiplex_snapshot() for _ in range(3)]
    sequence = HeteroGraphSnapshotSequence(
        snapshots,
        presence_masks=_multiplex_presence(3),
        allow_node_churn=True,
    )
    moved = sequence_to_device(sequence, torch.device("cpu"))
    assert isinstance(moved, HeteroGraphSnapshotSequence)
    assert moved.allow_node_churn
    assert moved.has_presence_masks
    assert moved.presence_masks is not None
    assert torch.equal(
        moved.presence_masks["node"],
        sequence.presence_masks["node"],  # type: ignore[index]
    )


def test_typed_presence_same_mapping_path() -> None:
    """Two-type sequences use the same per-type presence contract."""
    snapshots = [_typed_snapshot() for _ in range(3)]
    presence = {
        "gen": torch.ones(3, 2, dtype=torch.bool),
        "load": torch.ones(3, 3, dtype=torch.bool),
    }
    presence["load"][2, -1] = False
    sequence = HeteroGraphSnapshotSequence(
        snapshots,
        presence_masks=presence,
        allow_node_churn=True,
    )
    assert set(sequence.presence_mask_at(2)) == {"gen", "load"}
    assert not bool(sequence.presence_mask_at(2)["load"][-1].item())
    loss = sequence.loss_mask_at(2)
    assert loss is not None
    assert torch.equal(loss["load"], presence["load"][2])


def test_multiplex_one_step_ignores_inactive_rows() -> None:
    """One-step reconstruction does not score presence-False rows."""
    torch.manual_seed(0)
    snapshots = [_multiplex_snapshot() for _ in range(3)]
    # Poison inactive node features at t=2 so unmasked loss would spike.
    snapshots[2]["node"].x[-1] = 1.0e3
    presence = _multiplex_presence(3)
    sequence = HeteroGraphSnapshotSequence(
        snapshots,
        presence_masks=presence,
        allow_node_churn=True,
    )
    model = _multiplex_model()
    model.eval()

    loss = _one_step_pair(model, sequence, 1)
    assert torch.isfinite(loss)

    # Baseline without presence: poisoned target inflates loss.
    baseline = HeteroGraphSnapshotSequence(snapshots)
    loss_poisoned = _one_step_pair(model, baseline, 1)
    assert loss.item() < loss_poisoned.item()


def test_rectangular_latent_dims_reject_stacked_presence_pair_mask() -> None:
    """Rectangular latent_dims + presence raise on stacked pair-mask path."""
    node_types = ("a", "b")
    edge_types = (("a", "to_b", "b"), ("b", "to_a", "a"))
    num_nodes = {"a": 2, "b": 3}
    feature_dims = {"a": 2, "b": 2}
    latent_dims = {"a": 2, "b": 3}
    shared_d = 4

    snapshots: list[HeteroData] = []
    for _ in range(2):
        snap = HeteroData()
        snap["a"].x = torch.randn(num_nodes["a"], feature_dims["a"])
        snap["b"].x = torch.randn(num_nodes["b"], feature_dims["b"])
        snap["a", "to_b", "b"].edge_index = torch.tensor(
            [[0, 1], [0, 2]], dtype=torch.long
        )
        snap["b", "to_a", "a"].edge_index = torch.tensor(
            [[0, 1], [0, 1]], dtype=torch.long
        )
        snapshots.append(snap)

    presence = {
        "a": torch.ones(2, 2, dtype=torch.bool),
        "b": torch.ones(2, 3, dtype=torch.bool),
    }
    presence["b"][1, -1] = False
    sequence = HeteroGraphSnapshotSequence(
        snapshots,
        presence_masks=presence,
        allow_node_churn=True,
    )

    model = GraphKoopmanModel(
        encoder=RelGraphEncoder(
            feature_dims,
            hidden_channels=4,
            latent_dim=shared_d,
            num_relations=len(edge_types),
            num_layers=1,
            node_types=node_types,
            edge_types=edge_types,
            latent_dims=latent_dims,
        ),
        decoder=RelGraphDecoder(
            latent_dim=shared_d,
            hidden_channels=4,
            out_channels=feature_dims,
            num_relations=len(edge_types),
            num_layers=1,
            node_types=node_types,
            edge_types=edge_types,
            latent_dims=latent_dims,
        ),
        latent_dim=shared_d,
        time_step=1.0,
        koopman="hetero_graph",
        koopman_node_types=node_types,
        koopman_edge_types=edge_types,
        koopman_latent_dims=latent_dims,
    )
    assert isinstance(model.koopman, HeteroGraphKoopmanOperator)
    assert model.koopman.is_rectangular

    with pytest.raises(ValueError, match="rectangular latent_dims"):
        _hetero_pair_mask(sequence, 0, node_types, model=model)

    # Feature-space target masks remain available.
    target = _hetero_target_masks(sequence, 1)
    assert target is not None
    assert not bool(target["b"][-1].item())

    # Shared-d path accepts stacked presence masks.
    shared = _multiplex_model()
    multiplex = HeteroGraphSnapshotSequence(
        [_multiplex_snapshot() for _ in range(2)],
        presence_masks=_multiplex_presence(2),
        allow_node_churn=True,
    )
    stacked = _hetero_pair_mask(multiplex, 0, None, model=shared)
    assert stacked is not None
    assert stacked.shape == (4,)
    assert bool(stacked[:-1].all())
    # Presence drops only start at t>=2 in helper; pair (0,1) is all present.
    assert bool(stacked.all())


def test_forward_consistency_uses_composed_pair_mask() -> None:
    """Forward consistency stacks the composed hetero pair loss mask."""
    snapshots = [_multiplex_snapshot() for _ in range(3)]
    presence = _multiplex_presence(3)
    sequence = HeteroGraphSnapshotSequence(
        snapshots,
        presence_masks=presence,
        allow_node_churn=True,
    )
    model = _multiplex_model()
    mask = _hetero_pair_mask(sequence, 1, None, model=model)
    assert mask is not None
    # Pair (1, 2): node 3 absent at t=2 ⇒ False in stacked mask.
    assert torch.equal(mask, torch.tensor([True, True, True, False]))

    # Smoke: loss object accepts the mask shape.
    z = torch.randn(4, 4)
    loss = _FORWARD_CONSISTENCY_LOSS(
        z,
        z,
        model.koopman,
        mask=mask,
        edge_indices=[
            snapshots[2]["node", "r1", "node"].edge_index,
            snapshots[2]["node", "r2", "node"].edge_index,
        ],
        num_nodes_dict={"node": 4},
    )
    assert torch.isfinite(loss)
