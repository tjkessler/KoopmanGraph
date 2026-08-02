"""Presence-mask / node-churn sequence contract (TASK-1915–1917).

Presence marks whether an entity **exists** in a fixed union universe
``N_max``; observation marks whether an existing entity was **measured**.
``allow_node_churn`` gates presence drops; ``entity_ids`` name universe rows.
Training losses use ``loss_mask_at`` / ``pair_loss_mask`` (present ∧ observed)
and normalize by contributing-row count, not ``N_max``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
from koopman_graph.data import GraphSnapshotSequence, temporal_split
from koopman_graph.data.validation import (
    validate_entity_ids,
    validate_node_churn_policy,
    validate_presence_masks,
)
from koopman_graph.losses import masked_mse_loss
from koopman_graph.metrics import evaluate_forecast, masked_mae
from koopman_graph.serialization import (
    FORMAT_VERSION,
    build_checkpoint,
    build_model_config,
    load_checkpoint,
    reconstruct_model,
    save_checkpoint,
)
from koopman_graph.training.device import sequence_to_device
from koopman_graph.training.pair_objectives import _one_step_pair


def _presence_pattern(num_timesteps: int, num_nodes: int) -> torch.Tensor:
    """Build a valid presence mask that drops one entity mid-sequence."""
    masks = torch.ones(num_timesteps, num_nodes, dtype=torch.bool)
    if num_timesteps >= 3 and num_nodes >= 2:
        masks[2:, -1] = False
    return masks


def test_presence_masks_validation_and_accessors(
    synthetic_edge_index: torch.Tensor,
    make_snapshots: Callable[..., list[Data]],
) -> None:
    """Accept validated presence masks when churn is enabled."""
    snapshots = make_snapshots(synthetic_edge_index, num_timesteps=3, num_nodes=4)
    valid = _presence_pattern(3, 4)
    sequence = GraphSnapshotSequence(
        snapshots,
        presence_masks=valid,
        allow_node_churn=True,
    )

    assert sequence.has_presence_masks
    assert sequence.allow_node_churn
    assert not sequence.has_observation_masks
    assert sequence.presence_masks is not None
    assert torch.equal(sequence.presence_masks, valid)
    assert torch.equal(sequence.presence_mask_at(2), valid[2])
    assert not bool(valid[2, -1].item())


def test_presence_masks_reject_bad_shape_dtype_and_empty_timestep(
    synthetic_edge_index: torch.Tensor,
    make_snapshots: Callable[..., list[Data]],
) -> None:
    """Reject shape mismatches, non-0/1 values, and empty universes."""
    snapshots = make_snapshots(synthetic_edge_index, num_timesteps=3, num_nodes=4)

    with pytest.raises(ValueError, match="presence_masks shape"):
        GraphSnapshotSequence(
            snapshots,
            presence_masks=torch.ones(3, 3, dtype=torch.bool),
            allow_node_churn=True,
        )
    with pytest.raises(ValueError, match="at least one present entity"):
        GraphSnapshotSequence(
            snapshots,
            presence_masks=torch.zeros(3, 4, dtype=torch.bool),
            allow_node_churn=True,
        )
    with pytest.raises(
        ValueError, match="numeric presence_masks must contain only 0 and 1"
    ):
        GraphSnapshotSequence(
            snapshots,
            presence_masks=torch.full((3, 4), 0.5),
            allow_node_churn=True,
        )


def test_validate_presence_masks_coerces_numeric_0_1() -> None:
    """Numeric 0/1 presence masks coerce to bool like observation masks."""
    mask = validate_presence_masks(
        torch.tensor([[1, 0], [1, 1]], dtype=torch.float32),
        num_timesteps=2,
        num_nodes=2,
    )
    assert mask.dtype == torch.bool
    assert torch.equal(mask, torch.tensor([[True, False], [True, True]]))


def test_sequence_without_presence_masks_matches_default_contract(
    synthetic_edge_index: torch.Tensor,
    make_snapshots: Callable[..., list[Data]],
) -> None:
    """Absent presence masks keep the fixed-cardinality 0.10 surface."""
    snapshots = make_snapshots(synthetic_edge_index, num_timesteps=3, num_nodes=4)
    sequence = GraphSnapshotSequence(snapshots)
    assert not sequence.has_presence_masks
    assert sequence.presence_masks is None
    assert not sequence.allow_node_churn
    assert not sequence.has_entity_ids
    with pytest.raises(ValueError, match="does not contain presence_masks"):
        sequence.presence_mask_at(0)


def test_allow_node_churn_false_rejects_presence_drops(
    synthetic_edge_index: torch.Tensor,
    make_snapshots: Callable[..., list[Data]],
) -> None:
    """Default churn flag rejects presence masks that drop entities."""
    snapshots = make_snapshots(synthetic_edge_index, num_timesteps=3, num_nodes=4)
    masks = _presence_pattern(3, 4)
    with pytest.raises(ValueError, match="allow_node_churn=False"):
        GraphSnapshotSequence(snapshots, presence_masks=masks)

    # All-present masks remain valid without enabling churn.
    always_present = torch.ones(3, 4, dtype=torch.bool)
    sequence = GraphSnapshotSequence(snapshots, presence_masks=always_present)
    assert sequence.has_presence_masks
    assert not sequence.allow_node_churn


def test_allow_node_churn_true_requires_presence_masks(
    synthetic_edge_index: torch.Tensor,
    make_snapshots: Callable[..., list[Data]],
) -> None:
    """Churn without a presence contract is undefined."""
    snapshots = make_snapshots(synthetic_edge_index, num_timesteps=3, num_nodes=4)
    with pytest.raises(ValueError, match="requires presence_masks"):
        GraphSnapshotSequence(snapshots, allow_node_churn=True)


def test_entity_ids_validation_and_accessors(
    synthetic_edge_index: torch.Tensor,
    make_snapshots: Callable[..., list[Data]],
) -> None:
    """Accept unique entity ids; reject wrong length and duplicates."""
    snapshots = make_snapshots(synthetic_edge_index, num_timesteps=3, num_nodes=4)
    ids = ("a", "b", "c", "d")
    sequence = GraphSnapshotSequence(snapshots, entity_ids=ids)
    assert sequence.has_entity_ids
    assert sequence.entity_ids == ids

    with pytest.raises(ValueError, match="entity_ids length"):
        GraphSnapshotSequence(snapshots, entity_ids=("a", "b", "c"))
    with pytest.raises(ValueError, match="unique"):
        GraphSnapshotSequence(snapshots, entity_ids=("a", "b", "a", "c"))
    with pytest.raises(ValueError, match="str or int"):
        GraphSnapshotSequence(snapshots, entity_ids=("a", "b", "c", 1.5))  # type: ignore[arg-type]

    assert validate_entity_ids([0, 1, 2, 3], num_nodes=4) == (0, 1, 2, 3)


def test_validate_node_churn_policy_direct() -> None:
    """Policy helper encodes churn / presence coupling."""
    drops = torch.tensor([[True, False], [True, True]])
    with pytest.raises(ValueError, match="allow_node_churn=False"):
        validate_node_churn_policy(allow_node_churn=False, presence_masks=drops)
    with pytest.raises(ValueError, match="requires presence_masks"):
        validate_node_churn_policy(allow_node_churn=True, presence_masks=None)
    validate_node_churn_policy(allow_node_churn=False, presence_masks=None)
    validate_node_churn_policy(allow_node_churn=True, presence_masks=drops)


def test_slice_and_windowed_propagate_presence_masks(
    synthetic_edge_index: torch.Tensor,
    make_snapshots: Callable[..., list[Data]],
) -> None:
    """Contiguous slices and delay windows carry presence masks and churn flag."""
    snapshots = make_snapshots(synthetic_edge_index, num_timesteps=5, num_nodes=4)
    masks = _presence_pattern(5, 4)
    ids = ("n0", "n1", "n2", "n3")
    sequence = GraphSnapshotSequence(
        snapshots,
        presence_masks=masks,
        entity_ids=ids,
        allow_node_churn=True,
    )

    window = sequence.slice(1, 4)
    assert window.has_presence_masks
    assert window.allow_node_churn
    assert window.entity_ids == ids
    assert torch.equal(window.presence_masks, masks[1:4])

    hankel = sequence.windowed(n_delays=2, stride=1, pad=True)
    # Window ends are 0, 1, 2, 3, 4 when pad=True.
    assert hankel.has_presence_masks
    assert hankel.allow_node_churn
    assert hankel.entity_ids == ids
    assert torch.equal(hankel.presence_masks, masks)


def test_from_arrays_and_temporal_split_propagate_presence_masks(
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Array construction and temporal splits preserve presence / ids / churn."""
    features = torch.randn(6, 4, 2)
    masks = _presence_pattern(6, 4)
    ids = (10, 20, 30, 40)
    sequence = GraphSnapshotSequence.from_arrays(
        features,
        synthetic_edge_index,
        presence_masks=masks,
        entity_ids=ids,
        allow_node_churn=True,
    )
    assert sequence.has_presence_masks
    assert sequence.allow_node_churn
    assert sequence.entity_ids == ids
    assert torch.equal(sequence.presence_masks, masks.bool())

    split = temporal_split(
        sequence,
        train_ratio=0.5,
        val_ratio=1.0 / 6.0,
        test_ratio=1.0 / 3.0,
        min_val_timesteps=1,
    )
    assert split.train.has_presence_masks
    assert split.val.allow_node_churn
    assert split.test.entity_ids == ids
    assert split.train.presence_masks is not None
    assert split.val.presence_masks is not None
    assert split.test.presence_masks is not None
    assert torch.equal(
        torch.cat(
            [
                split.train.presence_masks,
                split.val.presence_masks,
                split.test.presence_masks,
            ],
            dim=0,
        ),
        masks.bool(),
    )


def test_presence_and_observation_masks_coexist(
    synthetic_edge_index: torch.Tensor,
    make_snapshots: Callable[..., list[Data]],
) -> None:
    """Presence (exists) and observation (measured) are independent fields."""
    snapshots = make_snapshots(synthetic_edge_index, num_timesteps=3, num_nodes=4)
    presence = _presence_pattern(3, 4)
    observation = torch.ones(3, 4, dtype=torch.bool)
    observation[:, 0] = False
    # Keep at least one observed node per timestep.
    observation[:, 1] = True

    sequence = GraphSnapshotSequence(
        snapshots,
        presence_masks=presence,
        observation_masks=observation,
        allow_node_churn=True,
    )
    assert sequence.has_presence_masks
    assert sequence.has_observation_masks
    assert torch.equal(sequence.presence_mask_at(0), presence[0])
    assert torch.equal(sequence.observation_mask_at(0), observation[0])


def test_sequence_to_device_preserves_presence_masks(
    synthetic_edge_index: torch.Tensor,
    make_snapshots: Callable[..., list[Data]],
) -> None:
    """Device transfer keeps presence masks, entity ids, and churn flag."""
    snapshots = make_snapshots(synthetic_edge_index, num_timesteps=3, num_nodes=4)
    masks = _presence_pattern(3, 4)
    ids = ("w", "x", "y", "z")
    sequence = GraphSnapshotSequence(
        snapshots,
        presence_masks=masks,
        entity_ids=ids,
        allow_node_churn=True,
    )
    moved = sequence_to_device(sequence, torch.device("cpu"))
    assert moved.has_presence_masks
    assert moved.allow_node_churn
    assert moved.entity_ids == ids
    assert torch.equal(moved.presence_masks, masks)


def test_inactive_nonzero_features_permitted_when_churn_enabled(
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Churn permits padded inactive rows without requiring zeros (convention)."""
    features = torch.ones(3, 4, 2)
    masks = _presence_pattern(3, 4)
    # Non-zero inactive padding is allowed; zeros are conventional only.
    sequence = GraphSnapshotSequence.from_arrays(
        features,
        synthetic_edge_index,
        presence_masks=masks,
        allow_node_churn=True,
    )
    assert sequence.allow_node_churn
    assert sequence[2].x[-1].abs().sum() > 0


def test_loss_mask_composes_presence_and_observation(
    synthetic_edge_index: torch.Tensor,
    make_snapshots: Callable[..., list[Data]],
) -> None:
    """Contributing mask is present ∧ observed; absent families pass through."""
    snapshots = make_snapshots(synthetic_edge_index, num_timesteps=3, num_nodes=4)
    presence = torch.tensor(
        [
            [True, True, True, True],
            [True, True, True, False],
            [True, True, False, False],
        ]
    )
    observation = torch.tensor(
        [
            [True, False, True, True],
            [True, True, False, True],
            [False, True, True, True],
        ]
    )
    bare = GraphSnapshotSequence(snapshots)
    assert bare.loss_mask_at(0) is None
    assert bare.pair_loss_mask(0) is None

    obs_only = GraphSnapshotSequence(snapshots, observation_masks=observation)
    assert torch.equal(obs_only.loss_mask_at(1), observation[1])
    assert torch.equal(obs_only.pair_loss_mask(0), observation[0] & observation[1])

    both = GraphSnapshotSequence(
        snapshots,
        presence_masks=presence,
        observation_masks=observation,
        allow_node_churn=True,
    )
    assert torch.equal(both.loss_mask_at(1), presence[1] & observation[1])
    assert torch.equal(
        both.pair_loss_mask(1),
        (presence[1] & presence[2]) & (observation[1] & observation[2]),
    )


def test_masked_mse_normalizes_by_contributing_rows_not_n_max() -> None:
    """Dropped node matches MSE on the contributing subset alone."""
    prediction = torch.tensor(
        [[1.0, 0.0], [2.0, 0.0], [100.0, 100.0]],
        dtype=torch.float32,
    )
    target = torch.zeros_like(prediction)
    mask = torch.tensor([True, True, False])
    loss = masked_mse_loss(prediction, target, mask)
    subset = ((prediction[:2] - target[:2]) ** 2).mean()
    assert loss.item() == pytest.approx(subset.item())


def test_one_step_loss_without_presence_matches_unmasked_path(
    synthetic_hypergraph_edge_index: torch.Tensor,
) -> None:
    """Sequences without presence masks keep the 0.10 unmasked loss path."""
    torch.manual_seed(0)
    features = torch.randn(3, 4, 3)
    sequence = GraphSnapshotSequence.from_arrays(
        features, synthetic_hypergraph_edge_index
    )
    model = GraphKoopmanModel(
        GNNEncoder(3, 16, 8, num_layers=1),
        GNNDecoder(8, 16, 3, num_layers=1),
        latent_dim=8,
        time_step=1.0,
    )
    torch.manual_seed(1)
    loss = _one_step_pair(model, sequence, timestep=0)
    torch.manual_seed(1)
    prediction = model(sequence[0])
    expected = torch.nn.functional.mse_loss(prediction, sequence[1].x)
    assert loss.item() == pytest.approx(expected.item(), abs=1e-6)


def test_one_step_loss_ignores_inactive_like_row_removal(
    synthetic_hypergraph_edge_index: torch.Tensor,
) -> None:
    """Presence-False target rows do not contribute to one-step reconstruction."""
    torch.manual_seed(0)
    features = torch.randn(3, 4, 3)
    presence = torch.ones(3, 4, dtype=torch.bool)
    presence[1:, -1] = False
    sequence = GraphSnapshotSequence.from_arrays(
        features,
        synthetic_hypergraph_edge_index,
        presence_masks=presence,
        allow_node_churn=True,
    )
    model = GraphKoopmanModel(
        GNNEncoder(3, 16, 8, num_layers=1),
        GNNDecoder(8, 16, 3, num_layers=1),
        latent_dim=8,
        time_step=1.0,
    )
    torch.manual_seed(2)
    loss = _one_step_pair(model, sequence, timestep=0)
    torch.manual_seed(2)
    prediction = model(sequence[0])
    mask = sequence.loss_mask_at(1)
    assert mask is not None
    assert not bool(mask[-1].item())
    expected = masked_mse_loss(prediction, sequence[1].x, mask)
    assert loss.item() == pytest.approx(expected.item(), abs=1e-6)
    # Equivalent to removing the inactive row from the mean.
    subset = ((prediction[:3] - sequence[1].x[:3]) ** 2).mean()
    assert loss.item() == pytest.approx(subset.item(), abs=1e-6)


def _tiny_model() -> GraphKoopmanModel:
    return GraphKoopmanModel(
        GNNEncoder(3, 16, 8, num_layers=1),
        GNNDecoder(8, 16, 3, num_layers=1),
        latent_dim=8,
        time_step=1.0,
    )


def test_predict_holds_inactive_and_resumes_on_reentry(
    synthetic_hypergraph_edge_index: torch.Tensor,
) -> None:
    """Hold last active features while absent; resume advancing on re-entry."""
    torch.manual_seed(0)
    features = torch.randn(4, 4, 3)
    sequence = GraphSnapshotSequence.from_arrays(
        features, synthetic_hypergraph_edge_index
    )
    model = _tiny_model()
    # step 0 present; step 1 drop last; step 2 re-enter last
    future_presence = torch.tensor(
        [
            [True, True, True, True],
            [True, True, True, False],
            [True, True, True, True],
        ]
    )
    preds = model.predict(sequence[0], steps=3, future_presence=future_presence)
    assert torch.equal(preds[1].x[-1], preds[0].x[-1])
    # Re-entry advances from the held latent; decoded row should leave the hold.
    assert not torch.allclose(preds[2].x[-1], preds[1].x[-1])


def test_evaluate_forecast_ignores_inactive_rows(
    synthetic_hypergraph_edge_index: torch.Tensor,
) -> None:
    """evaluate_forecast scores with loss_mask_at (present ∧ observed)."""
    torch.manual_seed(0)
    features = torch.randn(6, 4, 3)
    presence = torch.ones(6, 4, dtype=torch.bool)
    presence[2:, -1] = False
    sequence = GraphSnapshotSequence.from_arrays(
        features,
        synthetic_hypergraph_edge_index,
        presence_masks=presence,
        allow_node_churn=True,
    )
    # Poison inactive target rows; metrics must ignore them.
    poisoned = [snap.clone() for snap in sequence.snapshots]
    for t in range(2, 6):
        poisoned[t].x = poisoned[t].x.clone()
        poisoned[t].x[-1] = 1.0e6
    poisoned_seq = GraphSnapshotSequence(
        poisoned,
        presence_masks=presence,
        allow_node_churn=True,
    )
    model = _tiny_model()
    result = evaluate_forecast(model, poisoned_seq, horizons=(1, 2), start_indices=(0,))
    clean = evaluate_forecast(model, sequence, horizons=(1, 2), start_indices=(0,))
    result_by_h = {h.horizon: h.mae for h in result.horizons}
    clean_by_h = {h.horizon: h.mae for h in clean.horizons}
    assert result_by_h[1] == pytest.approx(clean_by_h[1], abs=1e-5)
    assert result_by_h[2] == pytest.approx(clean_by_h[2], abs=1e-5)

    # Explicit: horizon-1 mask at t=1 is all-present; at t=2 drops last node.
    assert sequence.loss_mask_at(2) is not None
    assert not bool(sequence.loss_mask_at(2)[-1].item())
    # Sanity: masked_mae with a full-true mask would see the poison if mis-wired.
    pred = model.predict(
        sequence[0],
        steps=2,
        future_presence=presence[1:3],
    )[1].x
    poison_mae = float(masked_mae(pred, poisoned[2].x, torch.ones(4, dtype=torch.bool)))
    assert poison_mae > 1.0e3


def test_checkpoint_round_trips_churn_contract_without_presence_tensors(
    synthetic_hypergraph_edge_index: torch.Tensor,
    tmp_path: Path,
) -> None:
    """Fit stamps churn keys; save/load restores them; masks stay off-checkpoint."""
    features = torch.randn(4, 4, 3)
    presence = _presence_pattern(4, 4)
    ids = ("a", "b", "c", "d")
    sequence = GraphSnapshotSequence.from_arrays(
        features,
        synthetic_hypergraph_edge_index,
        presence_masks=presence,
        entity_ids=ids,
        allow_node_churn=True,
    )
    model = _tiny_model()
    model.fit(sequence, epochs=1, lr=1e-2, device="cpu")

    assert model.allow_node_churn
    assert model.has_presence_masks
    assert model.entity_ids == ids

    config = build_model_config(model)
    assert config["allow_node_churn"] is True
    assert config["has_presence_masks"] is True
    assert config["entity_ids"] == list(ids)
    assert "presence_masks" not in config
    checkpoint = build_checkpoint(model)
    assert checkpoint["format_version"] == FORMAT_VERSION
    assert "presence_masks" not in checkpoint
    assert "presence_masks" not in checkpoint["config"]

    path = tmp_path / "churn.pt"
    save_checkpoint(model, path)
    loaded = load_checkpoint(path)
    assert loaded.allow_node_churn
    assert loaded.has_presence_masks
    assert loaded.entity_ids == ids


def test_checkpoint_absent_churn_keys_load_as_fixed_cardinality() -> None:
    """Configs without churn keys reconstruct as the 0.10 contract."""
    model = GraphKoopmanModel(
        GNNEncoder(3, 8, 4, num_layers=1),
        GNNDecoder(4, 8, 3, num_layers=1),
        latent_dim=4,
        time_step=0.1,
    )
    config = build_model_config(model)
    assert config["allow_node_churn"] is False
    assert config["has_presence_masks"] is False
    assert "entity_ids" not in config

    # Simulate a pre-0.11 payload that never wrote the additive keys.
    del config["allow_node_churn"]
    del config["has_presence_masks"]
    restored = reconstruct_model(config)
    assert not restored.allow_node_churn
    assert not restored.has_presence_masks
    assert restored.entity_ids is None


def test_checkpoint_entity_ids_length_mismatch_vs_orbit_n_max() -> None:
    """entity_ids length disagreeing with orbit-bound N_max raises clearly."""
    model = GraphKoopmanModel(
        GNNEncoder(3, 8, 4, num_layers=1),
        GNNDecoder(4, 8, 3, num_layers=1),
        latent_dim=4,
        time_step=0.1,
        koopman="graph",
        koopman_orbit_partition=((0, 1), (2, 3)),
    )
    config = build_model_config(model)
    config["allow_node_churn"] = True
    config["has_presence_masks"] = True
    config["entity_ids"] = ["a", "b", "c"]  # N_max is 4
    with pytest.raises(ValueError, match="entity_ids length .* N_max"):
        reconstruct_model(config)


def test_fit_without_presence_keeps_default_churn_contract(
    synthetic_hypergraph_edge_index: torch.Tensor,
) -> None:
    """Fitting a fixed-cardinality sequence leaves 0.10 checkpoint defaults."""
    features = torch.randn(3, 4, 3)
    sequence = GraphSnapshotSequence.from_arrays(
        features, synthetic_hypergraph_edge_index
    )
    model = _tiny_model()
    model.fit(sequence, epochs=1, lr=1e-2, device="cpu")
    assert not model.allow_node_churn
    assert not model.has_presence_masks
    assert model.entity_ids is None
    config = build_model_config(model)
    assert config["allow_node_churn"] is False
    assert config["has_presence_masks"] is False
    assert "entity_ids" not in config
