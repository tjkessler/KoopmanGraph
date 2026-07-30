"""Typed multi-node-type hetero path (shared latent width ``d``)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch_geometric.data import HeteroData

from koopman_graph.data import (
    HeteroGraphSnapshotSequence,
    global_relation_edge_indices,
    node_type_offsets,
    node_type_slices,
    offset_edge_index,
    snapshot_num_nodes_dict,
    stack_typed_features,
    unstack_typed_features,
)
from koopman_graph.model import GraphKoopmanModel
from koopman_graph.nn import RelGraphDecoder, RelGraphEncoder
from koopman_graph.operators import HeteroGraphKoopmanOperator
from koopman_graph.training import LossWeights, compute_training_loss

NODE_TYPES = ("a", "b")
EDGE_TYPES = (("a", "r0", "b"), ("b", "r1", "a"), ("a", "r2", "a"))
FEATURE_DIMS = {"a": 2, "b": 3}
NUM_NODES = {"a": 4, "b": 3}
LATENT_DIM = 4

_EDGES_AB = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)
_EDGES_BA = torch.tensor([[0, 1], [1, 3]], dtype=torch.long)
_EDGES_AA = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)


def _typed_snapshot(
    x_a: torch.Tensor | None = None,
    x_b: torch.Tensor | None = None,
) -> HeteroData:
    """Build a two-node-type snapshot with a fixed relation schema."""
    snapshot = HeteroData()
    snapshot["a"].x = (
        torch.randn(NUM_NODES["a"], FEATURE_DIMS["a"]) if x_a is None else x_a
    )
    snapshot["b"].x = (
        torch.randn(NUM_NODES["b"], FEATURE_DIMS["b"]) if x_b is None else x_b
    )
    snapshot["a", "r0", "b"].edge_index = _EDGES_AB
    snapshot["b", "r1", "a"].edge_index = _EDGES_BA
    snapshot["a", "r2", "a"].edge_index = _EDGES_AA
    return snapshot


def _typed_encoder(**kwargs: object) -> RelGraphEncoder:
    """Build a typed relational encoder over :data:`NODE_TYPES`."""
    return RelGraphEncoder(
        FEATURE_DIMS,
        hidden_channels=8,
        latent_dim=LATENT_DIM,
        num_relations=len(EDGE_TYPES),
        num_layers=1,
        node_types=NODE_TYPES,
        edge_types=EDGE_TYPES,
        **kwargs,  # type: ignore[arg-type]
    )


def _typed_decoder(**kwargs: object) -> RelGraphDecoder:
    """Build a typed relational decoder over :data:`NODE_TYPES`."""
    return RelGraphDecoder(
        latent_dim=LATENT_DIM,
        hidden_channels=8,
        out_channels=FEATURE_DIMS,
        num_relations=len(EDGE_TYPES),
        num_layers=1,
        node_types=NODE_TYPES,
        edge_types=EDGE_TYPES,
        **kwargs,  # type: ignore[arg-type]
    )


def _typed_model() -> GraphKoopmanModel:
    """Build a typed hetero Koopman model with matching peers and operator."""
    return GraphKoopmanModel(
        encoder=_typed_encoder(),
        decoder=_typed_decoder(),
        latent_dim=LATENT_DIM,
        time_step=1.0,
        koopman="hetero_graph",
        koopman_node_types=NODE_TYPES,
        koopman_edge_types=EDGE_TYPES,
    )


def _typed_operator() -> HeteroGraphKoopmanOperator:
    """Build a typed relational Koopman operator."""
    return HeteroGraphKoopmanOperator(
        latent_dim=LATENT_DIM,
        num_relations=len(EDGE_TYPES),
        node_types=NODE_TYPES,
        edge_types=EDGE_TYPES,
    )


# --------------------------------------------------------------------------
# Layout helpers
# --------------------------------------------------------------------------


def test_node_type_slices_tile_the_stacked_block() -> None:
    """Type slices tile ``[0, sum N_tau)`` without gaps or overlap."""
    slices = node_type_slices(NODE_TYPES, NUM_NODES)
    total = sum(NUM_NODES.values())

    covered: list[int] = []
    for name in NODE_TYPES:
        type_slice = slices[name]
        assert type_slice.stop - type_slice.start == NUM_NODES[name]
        covered.extend(range(type_slice.start, type_slice.stop))
    assert covered == list(range(total))
    assert node_type_offsets(NODE_TYPES, NUM_NODES) == {"a": 0, "b": NUM_NODES["a"]}


def test_node_type_slices_follow_declared_order() -> None:
    """Reversing ``node_type_names`` reverses the stacked layout."""
    slices = node_type_slices(("b", "a"), NUM_NODES)
    assert slices["b"].start == 0
    assert slices["a"].start == NUM_NODES["b"]


def test_node_type_slices_rejects_missing_count() -> None:
    """A missing per-type count is rejected with the expected message."""
    with pytest.raises(ValueError, match="missing node type"):
        node_type_slices(NODE_TYPES, {"a": 4})


def test_stack_unstack_typed_features_round_trip() -> None:
    """Stacking then unstacking equal-width per-type tensors is lossless."""
    features = {
        name: torch.randn(count, LATENT_DIM) for name, count in NUM_NODES.items()
    }

    stacked = stack_typed_features(features, NODE_TYPES)
    assert stacked.shape == (sum(NUM_NODES.values()), LATENT_DIM)

    recovered = unstack_typed_features(stacked, NODE_TYPES, NUM_NODES)
    for name in NODE_TYPES:
        assert torch.equal(recovered[name], features[name])


@pytest.mark.parametrize(
    "names, match",
    [
        ((), "at least one node type"),
        (("a", ""), "non-empty strings"),
        (("a", "a"), "must be unique"),
    ],
)
def test_node_type_names_validation(names: tuple[str, ...], match: str) -> None:
    """Malformed node-type name sequences are rejected."""
    with pytest.raises(ValueError, match=match):
        node_type_offsets(names, NUM_NODES)


def test_node_type_offsets_rejects_extra_counts() -> None:
    """Counts outside the declared stacking order are rejected."""
    with pytest.raises(ValueError, match="outside node_type_names"):
        node_type_offsets(NODE_TYPES, {**NUM_NODES, "c": 2})


def test_node_type_offsets_rejects_nonpositive_count() -> None:
    """Every node type must contribute at least one stacked row."""
    with pytest.raises(ValueError, match="must be positive"):
        node_type_offsets(NODE_TYPES, {"a": 4, "b": 0})


def test_unstack_typed_features_validates_shape() -> None:
    """Unstacking rejects non-2D input and row-count mismatches."""
    with pytest.raises(ValueError, match="stacked"):
        unstack_typed_features(torch.randn(7), NODE_TYPES, NUM_NODES)
    with pytest.raises(ValueError, match="num_nodes_dict sums to"):
        unstack_typed_features(torch.randn(9, LATENT_DIM), NODE_TYPES, NUM_NODES)


def test_offset_edge_index_validates_inputs() -> None:
    """Offsetting rejects malformed edge indices and negative offsets."""
    with pytest.raises(ValueError, match=r"\(2, num_edges\)"):
        offset_edge_index(torch.zeros(3, 3, dtype=torch.long), 0, 0)
    with pytest.raises(ValueError, match="non-negative offsets"):
        offset_edge_index(_EDGES_AB, -1, 0)


def test_global_relation_edge_indices_validates_edge_types() -> None:
    """Relation banks must exist and reference declared node types."""
    snapshot = _typed_snapshot()
    with pytest.raises(ValueError, match="outside node_type_names"):
        global_relation_edge_indices(
            snapshot, (("a", "r0", "c"),), NODE_TYPES, NUM_NODES
        )
    with pytest.raises(ValueError, match="missing edge type"):
        global_relation_edge_indices(
            snapshot, (("b", "r9", "b"),), NODE_TYPES, NUM_NODES
        )
    with pytest.raises(ValueError, match="triples"):
        global_relation_edge_indices(snapshot, (("a", "r0"),), NODE_TYPES, NUM_NODES)


def test_snapshot_num_nodes_dict_requires_features() -> None:
    """Reading per-type counts requires an ``x`` matrix on every node type."""
    snapshot = _typed_snapshot()
    with pytest.raises(ValueError, match="missing node type"):
        snapshot_num_nodes_dict(snapshot, ("a", "c"))


def test_stack_typed_features_rejects_mixed_widths() -> None:
    """Stacking requires one shared trailing width."""
    features = {
        "a": torch.randn(NUM_NODES["a"], 2),
        "b": torch.randn(NUM_NODES["b"], 3),
    }
    with pytest.raises(ValueError, match="one shared trailing width"):
        stack_typed_features(features, NODE_TYPES)


def test_offset_edge_index_round_trip() -> None:
    """Offsetting then subtracting the same offsets recovers local indices."""
    offsets = node_type_offsets(NODE_TYPES, NUM_NODES)

    shifted = offset_edge_index(_EDGES_AB, offsets["a"], offsets["b"])
    assert torch.equal(shifted[0], _EDGES_AB[0] + offsets["a"])
    assert torch.equal(shifted[1], _EDGES_AB[1] + offsets["b"])

    unshifted = offset_edge_index(shifted, 0, 0)
    unshifted = unshifted - torch.tensor([[offsets["a"]], [offsets["b"]]])
    assert torch.equal(unshifted, _EDGES_AB)


def test_global_relation_edge_indices_stay_in_range() -> None:
    """Global banks index stacked rows and respect the declared bank order."""
    snapshot = _typed_snapshot()
    counts = snapshot_num_nodes_dict(snapshot, NODE_TYPES)
    assert counts == NUM_NODES

    banks = global_relation_edge_indices(snapshot, EDGE_TYPES, NODE_TYPES, counts)
    total = sum(counts.values())
    assert len(banks) == len(EDGE_TYPES)
    for bank in banks:
        assert int(bank.min()) >= 0
        assert int(bank.max()) < total

    offsets = node_type_offsets(NODE_TYPES, counts)
    assert torch.equal(banks[0][1], _EDGES_AB[1] + offsets["b"])
    assert torch.equal(banks[2], _EDGES_AA)


# --------------------------------------------------------------------------
# Encoder / decoder
# --------------------------------------------------------------------------


def test_typed_encoder_returns_stacked_latents() -> None:
    """A typed encoder maps per-type features onto one shared-width block."""
    encoder = _typed_encoder()
    assert encoder.is_typed
    assert encoder.node_types == NODE_TYPES
    assert encoder.in_channels_dict == FEATURE_DIMS

    latents = encoder(_typed_snapshot())
    assert latents.shape == (sum(NUM_NODES.values()), LATENT_DIM)


def test_typed_decoder_returns_per_type_features() -> None:
    """A typed decoder returns one ``(N_tau, F_tau)`` tensor per node type."""
    snapshot = _typed_snapshot()
    encoder = _typed_encoder()
    decoder = _typed_decoder()

    latents = encoder(snapshot)
    banks = global_relation_edge_indices(snapshot, EDGE_TYPES, NODE_TYPES, NUM_NODES)
    reconstructed = decoder(latents, banks, num_nodes_dict=NUM_NODES)

    assert set(reconstructed) == set(NODE_TYPES)
    for name in NODE_TYPES:
        assert reconstructed[name].shape == (NUM_NODES[name], FEATURE_DIMS[name])


def test_typed_encode_decode_gradients_reach_per_type_heads() -> None:
    """Gradients flow through per-type projections and per-type output heads."""
    snapshot = _typed_snapshot()
    encoder = _typed_encoder()
    decoder = _typed_decoder()

    latents = encoder(snapshot)
    banks = global_relation_edge_indices(snapshot, EDGE_TYPES, NODE_TYPES, NUM_NODES)
    reconstructed = decoder(latents, banks, num_nodes_dict=NUM_NODES)
    loss = sum(
        torch.nn.functional.mse_loss(reconstructed[name], snapshot[name].x)
        for name in NODE_TYPES
    )
    loss.backward()

    for name in NODE_TYPES:
        assert encoder.type_input[name].weight.grad is not None
        assert decoder.type_output[name].weight.grad is not None
        assert torch.any(encoder.type_input[name].weight.grad != 0)
        assert torch.any(decoder.type_output[name].weight.grad != 0)


def test_typed_channels_require_explicit_node_types() -> None:
    """A mapping ``in_channels`` without ``node_types`` is rejected."""
    with pytest.raises(ValueError, match="node_types is required"):
        RelGraphEncoder(
            FEATURE_DIMS,
            hidden_channels=8,
            latent_dim=LATENT_DIM,
            num_relations=len(EDGE_TYPES),
            edge_types=EDGE_TYPES,
        )


def test_typed_channels_require_matching_node_types() -> None:
    """Mapping keys must match the declared stacking order exactly."""
    with pytest.raises(ValueError, match="must match node_types"):
        RelGraphEncoder(
            FEATURE_DIMS,
            hidden_channels=8,
            latent_dim=LATENT_DIM,
            num_relations=len(EDGE_TYPES),
            node_types=("a", "c"),
            edge_types=EDGE_TYPES,
        )


def test_typed_channels_require_explicit_edge_types() -> None:
    """Typed peers must pin the relation-bank order used by the operator."""
    with pytest.raises(ValueError, match="edge_types is required"):
        RelGraphEncoder(
            FEATURE_DIMS,
            hidden_channels=8,
            latent_dim=LATENT_DIM,
            num_relations=len(EDGE_TYPES),
            node_types=NODE_TYPES,
        )


def test_typed_decoder_requires_num_nodes_dict() -> None:
    """A typed decoder cannot split stacked rows without per-type counts."""
    snapshot = _typed_snapshot()
    decoder = _typed_decoder()
    banks = global_relation_edge_indices(snapshot, EDGE_TYPES, NODE_TYPES, NUM_NODES)
    latents = torch.randn(sum(NUM_NODES.values()), LATENT_DIM)

    with pytest.raises(ValueError, match="num_nodes_dict is required"):
        decoder(latents, banks)


def test_typed_modules_reject_per_type_latent_dims() -> None:
    """Latent width stays scalar: mapping ``latent_dim`` is not accepted."""
    with pytest.raises((TypeError, ValueError)):
        RelGraphEncoder(
            FEATURE_DIMS,
            hidden_channels=8,
            latent_dim={"a": 4, "b": 5},  # type: ignore[arg-type]
            num_relations=len(EDGE_TYPES),
            node_types=NODE_TYPES,
            edge_types=EDGE_TYPES,
        )
    with pytest.raises((TypeError, ValueError)):
        HeteroGraphKoopmanOperator(
            latent_dim={"a": 4, "b": 5},  # type: ignore[arg-type]
            num_relations=len(EDGE_TYPES),
            node_types=NODE_TYPES,
            edge_types=EDGE_TYPES,
        )


def test_typed_encoder_accepts_a_feature_mapping_with_global_banks() -> None:
    """Typed encoders accept per-type tensors plus already-global banks."""
    snapshot = _typed_snapshot()
    encoder = _typed_encoder()
    banks = global_relation_edge_indices(snapshot, EDGE_TYPES, NODE_TYPES, NUM_NODES)
    features = {name: snapshot[name].x for name in NODE_TYPES}

    from_mapping = encoder(features, banks)
    from_snapshot = encoder(snapshot)

    assert torch.allclose(from_mapping, from_snapshot, atol=1e-6)


def test_typed_feature_mapping_requires_edge_banks() -> None:
    """A per-type feature mapping carries no topology of its own."""
    encoder = _typed_encoder()
    features = {
        name: torch.randn(NUM_NODES[name], FEATURE_DIMS[name]) for name in NODE_TYPES
    }
    with pytest.raises(ValueError, match="edge_index relation banks are required"):
        encoder(features)


def test_typed_feature_mapping_rejects_missing_node_type() -> None:
    """Every declared node type must appear in the feature mapping."""
    encoder = _typed_encoder()
    banks = global_relation_edge_indices(
        _typed_snapshot(), EDGE_TYPES, NODE_TYPES, NUM_NODES
    )
    features = {"a": torch.randn(NUM_NODES["a"], FEATURE_DIMS["a"])}
    with pytest.raises(ValueError, match="missing node type"):
        encoder(features, banks)


def test_typed_encoder_rejects_unexpected_snapshot_node_types() -> None:
    """A snapshot with extra node types is rejected rather than silently sliced."""
    snapshot = _typed_snapshot()
    snapshot["c"].x = torch.randn(2, 2)
    encoder = _typed_encoder()

    with pytest.raises(ValueError, match="expects exactly"):
        encoder(snapshot)


def test_typed_decoder_rejects_inconsistent_num_nodes_dict() -> None:
    """Per-type counts must sum to the number of stacked latent rows."""
    snapshot = _typed_snapshot()
    decoder = _typed_decoder()
    banks = global_relation_edge_indices(snapshot, EDGE_TYPES, NODE_TYPES, NUM_NODES)
    latents = torch.randn(sum(NUM_NODES.values()), LATENT_DIM)

    with pytest.raises(ValueError, match="rows"):
        decoder(latents, banks, num_nodes_dict={"a": 2, "b": 2})


# --------------------------------------------------------------------------
# Operator
# --------------------------------------------------------------------------


def test_typed_operator_holds_one_self_block_per_node_type() -> None:
    """Typed operators expose per-type ``K_self`` and reject the scalar alias."""
    operator = _typed_operator()
    assert operator.is_typed
    assert operator.node_types == NODE_TYPES

    for name in NODE_TYPES:
        assert operator.k_self_for(name).shape == (LATENT_DIM, LATENT_DIM)
    with pytest.raises(ValueError, match="k_self_for"):
        _ = operator.K_self


def test_typed_effective_matrix_matches_forward() -> None:
    """``effective_matrix @ vec(Z)`` reproduces the sparse typed advance."""
    torch.manual_seed(0)
    operator = _typed_operator()
    snapshot = _typed_snapshot()
    banks = global_relation_edge_indices(snapshot, EDGE_TYPES, NODE_TYPES, NUM_NODES)
    total = sum(NUM_NODES.values())
    latents = torch.randn(total, LATENT_DIM)

    advanced = operator(latents, edge_indices=banks, num_nodes_dict=NUM_NODES)
    dense = operator.effective_matrix(banks, total, num_nodes_dict=NUM_NODES)

    assert dense.shape == (total * LATENT_DIM, total * LATENT_DIM)
    assert torch.allclose(
        advanced.reshape(-1),
        dense @ latents.reshape(-1),
        atol=1e-5,
    )


def test_typed_self_blocks_apply_per_type_dynamics() -> None:
    """With no relations, each type advances under its own ``K_self``."""
    operator = HeteroGraphKoopmanOperator(
        latent_dim=2,
        num_relations=1,
        node_types=NODE_TYPES,
        edge_types=(("a", "r0", "b"),),
    )
    k_a = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    k_b = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    operator.set_dense_matrices(
        k_self={"a": k_a, "b": k_b},
        k_relations=[torch.zeros(2, 2)],
    )
    empty = torch.zeros(2, 0, dtype=torch.long)
    latents = torch.randn(sum(NUM_NODES.values()), 2)

    advanced = operator(latents, edge_indices=[empty], num_nodes_dict=NUM_NODES)

    slices = node_type_slices(NODE_TYPES, NUM_NODES)
    assert torch.allclose(
        advanced[slices["a"]], latents[slices["a"]] @ k_a.T, atol=1e-6
    )
    assert torch.allclose(
        advanced[slices["b"]], latents[slices["b"]] @ k_b.T, atol=1e-6
    )


def test_typed_operator_requires_num_nodes_dict() -> None:
    """Typed advance without per-type counts fails loudly."""
    operator = _typed_operator()
    snapshot = _typed_snapshot()
    banks = global_relation_edge_indices(snapshot, EDGE_TYPES, NODE_TYPES, NUM_NODES)
    latents = torch.randn(sum(NUM_NODES.values()), LATENT_DIM)

    with pytest.raises(ValueError, match="num_nodes_dict"):
        operator(latents, edge_indices=banks)


def test_typed_operator_rejects_edges_outside_node_types() -> None:
    """Edge types must reference declared node types."""
    with pytest.raises(ValueError, match="node_types"):
        HeteroGraphKoopmanOperator(
            latent_dim=LATENT_DIM,
            num_relations=1,
            node_types=NODE_TYPES,
            edge_types=(("a", "r0", "c"),),
        )


def test_typed_operator_rejects_shared_dense_self_matrix() -> None:
    """Typed operators require a per-type mapping in ``set_dense_matrices``."""
    operator = _typed_operator()
    with pytest.raises(ValueError, match="mapping"):
        operator.set_dense_matrices(
            k_self=torch.eye(LATENT_DIM),
            k_relations=[torch.zeros(LATENT_DIM, LATENT_DIM)] * len(EDGE_TYPES),
        )


def test_typed_operator_rejects_unknown_node_type_lookup() -> None:
    """``k_self_for`` only accepts declared node types."""
    operator = _typed_operator()
    with pytest.raises(KeyError, match="unknown node type"):
        operator.k_self_for("c")


@pytest.mark.parametrize(
    "counts, match",
    [
        ({"a": 4}, "must match operator node_types"),
        ({"a": 4, "b": 0}, "must be positive"),
    ],
)
def test_typed_operator_validates_per_type_counts(
    counts: dict[str, int], match: str
) -> None:
    """Per-type counts must cover every node type with a positive value."""
    operator = _typed_operator()
    with pytest.raises(ValueError, match=match):
        operator.typed_k_self_blocks(counts)


def test_typed_operator_rejects_count_sum_mismatch() -> None:
    """Advance rejects counts that disagree with the stacked row count."""
    operator = _typed_operator()
    snapshot = _typed_snapshot()
    banks = global_relation_edge_indices(snapshot, EDGE_TYPES, NODE_TYPES, NUM_NODES)
    latents = torch.randn(sum(NUM_NODES.values()), LATENT_DIM)

    with pytest.raises(ValueError, match="sums to"):
        operator(latents, edge_indices=banks, num_nodes_dict={"a": 4, "b": 4})


def test_typed_operator_rejects_control() -> None:
    """Typed operators have no control surface."""
    with pytest.raises(ValueError, match="control is unsupported"):
        HeteroGraphKoopmanOperator(
            latent_dim=LATENT_DIM,
            num_relations=len(EDGE_TYPES),
            node_types=NODE_TYPES,
            edge_types=EDGE_TYPES,
            control_dim=2,
        )


def test_typed_operator_requires_explicit_edge_types() -> None:
    """Default relation triples are defined for the multiplex schema only."""
    with pytest.raises(ValueError, match="edge_types is required"):
        HeteroGraphKoopmanOperator(
            latent_dim=LATENT_DIM,
            num_relations=len(EDGE_TYPES),
            node_types=NODE_TYPES,
        )


def test_typed_dense_self_mapping_keys_must_match_node_types() -> None:
    """``set_dense_matrices`` rejects a mapping that misses a node type."""
    operator = _typed_operator()
    with pytest.raises(ValueError, match="k_self keys must match"):
        operator.set_dense_matrices(
            k_self={"a": torch.eye(LATENT_DIM)},
            k_relations=[torch.zeros(LATENT_DIM, LATENT_DIM)] * len(EDGE_TYPES),
        )


def test_multiplex_dense_self_rejects_a_mapping() -> None:
    """The multiplex path keeps its single-tensor ``k_self`` contract."""
    operator = HeteroGraphKoopmanOperator(latent_dim=2, num_relations=1)
    with pytest.raises(ValueError, match="not a mapping"):
        operator.set_dense_matrices(
            k_self={"node": torch.eye(2)},
            k_relations=[torch.zeros(2, 2)],
        )


def test_typed_reset_parameters_reinitializes_every_self_block() -> None:
    """``reset_parameters`` reaches each per-type self factor."""
    operator = _typed_operator()
    operator.set_dense_matrices(
        k_self={name: torch.zeros(LATENT_DIM, LATENT_DIM) for name in NODE_TYPES},
        k_relations=[torch.zeros(LATENT_DIM, LATENT_DIM)] * len(EDGE_TYPES),
    )
    assert all(
        torch.count_nonzero(operator.k_self_for(name)) == 0 for name in NODE_TYPES
    )

    operator.reset_parameters()

    # Default ``identity_noise`` init: each block returns to I plus noise drawn
    # at ``init_scale``, so allow a few standard deviations of slack.
    identity = torch.eye(LATENT_DIM)
    tolerance = 5.0 * operator.init_scale
    for name in NODE_TYPES:
        assert torch.allclose(operator.k_self_for(name), identity, atol=tolerance)


def test_typed_operator_reports_a_self_stability_certificate() -> None:
    """Stability reporting reads the first typed self factor."""
    operator = _typed_operator()
    assert operator.stability_certificate() is None


# --------------------------------------------------------------------------
# Model / factory
# --------------------------------------------------------------------------


def test_typed_model_forward_returns_per_type_predictions() -> None:
    """A typed model's single-step forward returns one tensor per node type."""
    model = _typed_model()
    prediction = model(_typed_snapshot())

    assert isinstance(prediction, dict)
    for name in NODE_TYPES:
        assert prediction[name].shape == (NUM_NODES[name], FEATURE_DIMS[name])


def test_typed_model_predict_packs_hetero_snapshots() -> None:
    """Typed rollouts preserve the origin node / edge type schema."""
    model = _typed_model()
    origin = _typed_snapshot()

    forecasts = model.predict(origin, steps=3)

    assert len(forecasts) == 3
    for snapshot in forecasts:
        assert tuple(snapshot.node_types) == NODE_TYPES
        assert set(map(tuple, snapshot.edge_types)) == set(EDGE_TYPES)
        for name in NODE_TYPES:
            assert snapshot[name].x.shape == (NUM_NODES[name], FEATURE_DIMS[name])


def test_factory_rejects_typed_operator_with_shared_peer_channels() -> None:
    """A typed operator needs typed peers so latent slices line up."""
    with pytest.raises(ValueError, match="mapping of per-type feature widths"):
        GraphKoopmanModel(
            encoder=RelGraphEncoder(
                3,
                hidden_channels=8,
                latent_dim=LATENT_DIM,
                num_relations=len(EDGE_TYPES),
                num_layers=1,
            ),
            decoder=RelGraphDecoder(
                latent_dim=LATENT_DIM,
                hidden_channels=8,
                out_channels=3,
                num_relations=len(EDGE_TYPES),
                num_layers=1,
            ),
            latent_dim=LATENT_DIM,
            time_step=1.0,
            koopman="hetero_graph",
            koopman_node_types=NODE_TYPES,
            koopman_edge_types=EDGE_TYPES,
        )


def test_factory_rejects_typed_node_type_order_mismatch() -> None:
    """Peer node-type order must match the operator's stacking order."""
    with pytest.raises(ValueError, match="must match HeteroGraphKoopmanOperator"):
        GraphKoopmanModel(
            encoder=RelGraphEncoder(
                {"b": 3, "a": 2},
                hidden_channels=8,
                latent_dim=LATENT_DIM,
                num_relations=len(EDGE_TYPES),
                num_layers=1,
                node_types=("b", "a"),
                edge_types=EDGE_TYPES,
            ),
            decoder=_typed_decoder(),
            latent_dim=LATENT_DIM,
            time_step=1.0,
            koopman="hetero_graph",
            koopman_node_types=NODE_TYPES,
            koopman_edge_types=EDGE_TYPES,
        )


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


def _typed_sequence(
    num_timesteps: int = 6,
    *,
    seed: int = 0,
) -> HeteroGraphSnapshotSequence:
    """Build a short typed sequence of random snapshots."""
    torch.manual_seed(seed)
    return HeteroGraphSnapshotSequence(
        [_typed_snapshot() for _ in range(num_timesteps)]
    )


def test_typed_fit_smoke_decreases_loss() -> None:
    """Typed fit runs reconstruction + forward + rollout terms end to end."""
    torch.manual_seed(0)
    model = _typed_model()
    sequence = _typed_sequence()

    history = model.fit(
        sequence,
        epochs=8,
        lr=1e-2,
        loss_weights=LossWeights(reconstruction=1.0, forward=1.0, rollout=0.5),
        rollout_horizon=2,
    )

    assert len(history.loss) == 8
    assert history.loss[-1] < history.loss[0]
    assert history.forward_loss[-1] > 0.0
    assert history.rollout_loss[-1] > 0.0


def test_typed_observation_masks_change_the_loss() -> None:
    """Per-type observation masks select which nodes contribute to the loss."""
    torch.manual_seed(0)
    snapshots = [_typed_snapshot() for _ in range(4)]
    unmasked = HeteroGraphSnapshotSequence(snapshots)

    masks = {
        name: torch.ones(len(snapshots), count, dtype=torch.bool)
        for name, count in NUM_NODES.items()
    }
    masks["a"][:, 0] = False
    masked = HeteroGraphSnapshotSequence(snapshots, observation_masks=masks)

    torch.manual_seed(1)
    model = _typed_model()
    weights = LossWeights(reconstruction=1.0, forward=1.0)

    unmasked_loss = compute_training_loss(model, unmasked, weights)
    masked_loss = compute_training_loss(model, masked, weights)

    assert not torch.isclose(unmasked_loss.total, masked_loss.total)


def test_typed_masked_nodes_do_not_contribute_to_the_loss() -> None:
    """Perturbing masked rows of one node type leaves the loss unchanged."""
    torch.manual_seed(0)
    snapshots = [_typed_snapshot() for _ in range(3)]

    masks = {
        name: torch.ones(len(snapshots), count, dtype=torch.bool)
        for name, count in NUM_NODES.items()
    }
    masks["b"][:, 1:] = False

    perturbed_snapshots = []
    for snapshot in snapshots:
        x_b = snapshot["b"].x.clone()
        x_b[1:] = x_b[1:] + 5.0
        perturbed_snapshots.append(_typed_snapshot(x_a=snapshot["a"].x, x_b=x_b))

    torch.manual_seed(1)
    model = _typed_model()
    weights = LossWeights(reconstruction=1.0, forward=1.0)

    original = compute_training_loss(
        model,
        HeteroGraphSnapshotSequence(snapshots, observation_masks=masks),
        weights,
    )
    perturbed = compute_training_loss(
        model,
        HeteroGraphSnapshotSequence(perturbed_snapshots, observation_masks=masks),
        weights,
    )

    assert torch.allclose(original.total, perturbed.total, atol=1e-6)


# --------------------------------------------------------------------------
# Serialization
# --------------------------------------------------------------------------


def test_typed_checkpoint_round_trip(tmp_path: Path) -> None:
    """Typed checkpoints restore per-type self blocks and per-type heads."""
    torch.manual_seed(0)
    model = _typed_model()
    snapshot = _typed_snapshot()
    model.fit(_typed_sequence(4), epochs=3, lr=1e-2)

    path = tmp_path / "typed.pt"
    model.save(path)
    loaded = GraphKoopmanModel.load(path)

    assert isinstance(loaded.encoder, RelGraphEncoder)
    assert isinstance(loaded.decoder, RelGraphDecoder)
    assert loaded.encoder.is_typed
    assert loaded.decoder.is_typed
    assert loaded.encoder.in_channels_dict == FEATURE_DIMS
    assert loaded.decoder.out_channels_dict == FEATURE_DIMS
    assert tuple(loaded.koopman.node_types) == NODE_TYPES

    selves_keys = {key for key in loaded.state_dict() if "_selves" in key}
    assert selves_keys == {f"koopman._selves.{name}.K" for name in NODE_TYPES}

    model.eval()
    loaded.eval()
    with torch.no_grad():
        original = model(snapshot)
        restored = loaded(snapshot)
    for name in NODE_TYPES:
        assert torch.allclose(original[name], restored[name], atol=1e-6)


# --------------------------------------------------------------------------
# Ablation
# --------------------------------------------------------------------------

_ABLATION_NODE_TYPES = ("a", "b")
_ABLATION_EDGE_TYPES = (("a", "r0", "b"), ("b", "r1", "a"))
_ABLATION_FEATURE_DIMS = {"a": 2, "b": 3}
_ABLATION_NUM_NODES = {"a": 6, "b": 5}
_ABLATION_EDGES_AB = torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]], dtype=torch.long)
_ABLATION_EDGES_BA = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long)


def _ablation_snapshot(x_a: torch.Tensor, x_b: torch.Tensor) -> HeteroData:
    """Build one snapshot of the two-type ablation trajectory."""
    snapshot = HeteroData()
    snapshot["a"].x = x_a
    snapshot["b"].x = x_b
    snapshot["a", "r0", "b"].edge_index = _ABLATION_EDGES_AB
    snapshot["b", "r1", "a"].edge_index = _ABLATION_EDGES_BA
    return snapshot


def _ablation_trajectory(num_timesteps: int, seed: int) -> list[HeteroData]:
    """Roll a linear trajectory whose two node types have distinct dynamics."""
    generator = torch.Generator().manual_seed(seed)
    x_a = torch.randn(
        _ABLATION_NUM_NODES["a"],
        _ABLATION_FEATURE_DIMS["a"],
        generator=generator,
    )
    x_b = torch.randn(
        _ABLATION_NUM_NODES["b"],
        _ABLATION_FEATURE_DIMS["b"],
        generator=generator,
    )
    # Type "a" decays with a sign-flipping mode; type "b" rotates slowly.
    dynamics_a = torch.tensor([[0.95, 0.0], [0.0, -0.30]])
    dynamics_b = (
        torch.tensor(
            [[0.50, -0.85, 0.0], [0.85, 0.50, 0.0], [0.0, 0.0, 0.99]],
        )
        * 0.98
    )
    snapshots = [_ablation_snapshot(x_a.clone(), x_b.clone())]
    for _ in range(num_timesteps - 1):
        coupling = 0.05 * x_b[:, :2].mean(dim=0, keepdim=True)
        x_a = x_a @ dynamics_a.T + coupling.expand(_ABLATION_NUM_NODES["a"], 2)
        x_b = x_b @ dynamics_b.T
        snapshots.append(_ablation_snapshot(x_a.clone(), x_b.clone()))
    return snapshots


def _ablation_model(*, tie_self_blocks: bool) -> GraphKoopmanModel:
    """Build the ablation model, optionally forcing one shared self block."""
    torch.manual_seed(0)
    model = GraphKoopmanModel(
        encoder=RelGraphEncoder(
            _ABLATION_FEATURE_DIMS,
            hidden_channels=16,
            latent_dim=4,
            num_relations=len(_ABLATION_EDGE_TYPES),
            node_types=_ABLATION_NODE_TYPES,
            edge_types=_ABLATION_EDGE_TYPES,
        ),
        decoder=RelGraphDecoder(
            latent_dim=4,
            hidden_channels=16,
            out_channels=_ABLATION_FEATURE_DIMS,
            num_relations=len(_ABLATION_EDGE_TYPES),
            node_types=_ABLATION_NODE_TYPES,
            edge_types=_ABLATION_EDGE_TYPES,
        ),
        latent_dim=4,
        time_step=1.0,
        koopman="hetero_graph",
        koopman_node_types=_ABLATION_NODE_TYPES,
        koopman_edge_types=_ABLATION_EDGE_TYPES,
    )
    if tie_self_blocks:
        operator = model.koopman
        assert isinstance(operator, HeteroGraphKoopmanOperator)
        operator._selves["b"] = operator._selves["a"]  # noqa: SLF001
    return model


def _one_step_mse(
    model: GraphKoopmanModel,
    sequence: HeteroGraphSnapshotSequence,
) -> float:
    """Average per-type one-step MSE over consecutive snapshots."""
    total = 0.0
    with torch.no_grad():
        for timestep in range(sequence.num_timesteps - 1):
            prediction = model(sequence[timestep])
            target = sequence[timestep + 1]
            for name in _ABLATION_NODE_TYPES:
                total += float(
                    torch.nn.functional.mse_loss(prediction[name], target[name].x)
                )
    pairs = (sequence.num_timesteps - 1) * len(_ABLATION_NODE_TYPES)
    return total / pairs


def test_per_type_self_blocks_beat_forced_sharing_on_holdout() -> None:
    """Per-type ``K_self`` beats forced sharing on a temporal hold-out window.

    Honest scope: the advantage is real but not universal. Across the seeds
    surveyed while writing this test (three trajectories x two initializations)
    per-type self blocks won five of six times; the remaining configuration
    favored the tied variant. This test therefore pins one reproducible
    configuration rather than asserting a seed-independent result.
    """
    trajectory = _ablation_trajectory(20, seed=3)
    train = HeteroGraphSnapshotSequence(trajectory[:12])
    holdout = HeteroGraphSnapshotSequence(trajectory[11:])
    weights = LossWeights(reconstruction=1.0, forward=1.0)

    scores: dict[bool, float] = {}
    for tie_self_blocks in (False, True):
        model = _ablation_model(tie_self_blocks=tie_self_blocks)
        model.fit(train, epochs=300, lr=5e-3, loss_weights=weights)
        scores[tie_self_blocks] = _one_step_mse(model, holdout)

    assert scores[False] < scores[True]
