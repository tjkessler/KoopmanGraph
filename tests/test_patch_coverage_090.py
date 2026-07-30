"""Patch-coverage gaps for the 0.9.0 Codecov patch gate (target ≥ 90%)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch_geometric.data import Data, HeteroData

from koopman_graph import (
    GNNDecoder,
    GNNEncoder,
    GraphKoopmanModel,
    GraphSnapshotSequence,
    RelGraphDecoder,
    RelGraphEncoder,
)
from koopman_graph.analysis.spectrum import attribute_mode_energy
from koopman_graph.data import HeteroGraphSnapshotSequence
from koopman_graph.data.hetero_layout import (
    latent_type_slices,
    mask_hetero_snapshot_features,
    snapshot_num_nodes_dict,
    stack_typed_features,
)
from koopman_graph.data.validation import (
    _hetero_edge_weight,
    _require_hetero_node_x,
    hetero_snapshots_have_dynamic_topology,
    infer_hetero_schema,
    validate_hetero_control_inputs,
    validate_hetero_observation_masks,
    validate_hetero_snapshot_metadata,
    validate_shared_hetero_topology,
)
from koopman_graph.datasets.ieee118 import (
    bus_type_name,
    homogeneous_features_to_typed_hetero,
    partition_buses_by_type,
)
from koopman_graph.losses.rollout import (
    _bind_hetero_decoder,
    _hetero_rollout_step_loss,
    _multiplex_target_features,
    _relation_topology_at_from_targets,
    _typed_target_features,
)
from koopman_graph.model.factory import validate_typed_relgraph_peers
from koopman_graph.model.validation import (
    uses_relgraph_modules,
    validate_sequence_hyperedges,
)
from koopman_graph.nn.heterogeneous import (
    HGTDecoder,
    RelGraphConv,
    _normalize_edge_type_order,
    _relgraph_message_passing,
    _resolve_hgt_activation,
    resolve_hgt_typed_inputs,
    resolve_multiplex_relation_inputs,
    resolve_typed_relation_inputs,
)
from koopman_graph.nn.heterogeneous import (
    RelGraphDecoder as RelGraphDecoderCls,
)
from koopman_graph.operators.heterogeneous import (
    HeteroGraphKoopmanOperator,
    _normalize_edge_types,
    _normalize_node_types,
)
from koopman_graph.serialization import _require_hetero_schema
from koopman_graph.training import LossWeights
from koopman_graph.training.history import ExtraLosses
from koopman_graph.training.inputs import (
    resolve_training_sequences,
    resolve_validation_sequences,
)
from koopman_graph.training.objectives import (
    _hetero_eigenvalue_regularization_over_sequence,
    _hetero_relation_banks,
    _validate_hetero_fit_surface,
    compute_eigenvalue_regularization_loss,
)
from koopman_graph.training.pair_objectives import (
    _backward_consistency_pair,
    _hetero_num_relations,
    _one_step_pair,
    multiplex_node_features,
    one_step_loss,
    one_step_prediction,
    pair_control,
    stack_typed_masks,
    typed_node_features,
)


def _multiplex_snapshot(*, num_nodes: int = 4, in_channels: int = 3) -> HeteroData:
    """Build a one-type, two-relation multiplex snapshot."""
    data = HeteroData()
    data["node"].x = torch.randn(num_nodes, in_channels)
    data["node", "r1", "node"].edge_index = torch.tensor(
        [[0, 1, 2], [1, 2, 3]],
        dtype=torch.long,
    )
    data["node", "r2", "node"].edge_index = torch.tensor(
        [[0, 2], [2, 0]],
        dtype=torch.long,
    )
    return data


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


def _multiplex_model(*, num_relations: int = 2) -> GraphKoopmanModel:
    """Factory-built multiplex RelGraph model."""
    return GraphKoopmanModel(
        encoder=RelGraphEncoder(3, 8, 4, num_relations=num_relations, num_layers=1),
        decoder=RelGraphDecoder(4, 8, 3, num_relations=num_relations, num_layers=1),
        latent_dim=4,
        time_step=1.0,
        koopman="hetero_graph",
    )


# ---------------------------------------------------------------------------
# data.validation
# ---------------------------------------------------------------------------


def test_hetero_edge_weight_rejects_non_tensor() -> None:
    """Non-tensor edge_weight on a store raises TypeError."""
    store = SimpleNamespace(edge_weight=[1.0, 2.0])
    with pytest.raises(TypeError, match="must be a Tensor"):
        _hetero_edge_weight(store)


def test_require_hetero_node_x_error_paths() -> None:
    """Missing type / None features / wrong ndim raise named ValueErrors."""
    snap = _typed_snapshot()
    with pytest.raises(ValueError, match="missing node type"):
        _require_hetero_node_x(snap, "ghost", index=2)

    class _Store:
        def __init__(self, features: torch.Tensor | None) -> None:
            self.x = features

    class _Snap:
        node_types = ("gen",)

        def __getitem__(self, _key: str) -> _Store:
            return _Store(None)

    with pytest.raises(ValueError, match="has no feature tensor x"):
        _require_hetero_node_x(_Snap(), "gen", index=0)  # type: ignore[arg-type]

    snap["gen"].x = torch.randn(4)
    with pytest.raises(ValueError, match="must have shape"):
        _require_hetero_node_x(snap, "gen", index=0)


def test_infer_hetero_schema_error_paths() -> None:
    """Schema inference rejects empty / malformed hetero snapshots."""
    with pytest.raises(ValueError, match="has no node types"):
        infer_hetero_schema(HeteroData())

    no_edges = HeteroData()
    no_edges["node"].x = torch.randn(3, 2)
    with pytest.raises(ValueError, match=r"\|R\| >= 1"):
        infer_hetero_schema(no_edges)

    bad_src = HeteroData()
    bad_src["load"].x = torch.randn(2, 2)
    bad_src["ghost", "r", "load"].edge_index = torch.tensor(
        [[0], [0]],
        dtype=torch.long,
    )
    with pytest.raises(ValueError, match="unknown source node type"):
        infer_hetero_schema(bad_src)

    bad_dst = HeteroData()
    bad_dst["gen"].x = torch.randn(2, 2)
    bad_dst["gen", "r", "ghost"].edge_index = torch.tensor(
        [[0], [0]],
        dtype=torch.long,
    )
    with pytest.raises(ValueError, match="unknown destination"):
        infer_hetero_schema(bad_dst)

    bad_shape = HeteroData()
    bad_shape["node"].x = torch.randn(2, 2)
    bad_shape["node", "r", "node"].edge_index = torch.ones(3, 2, dtype=torch.long)
    with pytest.raises(ValueError, match=r"\(2, num_edges\)"):
        infer_hetero_schema(bad_shape)


def test_validate_hetero_metadata_and_shared_topology_errors() -> None:
    """Metadata / shared-topology helpers reject drift and weight mismatches."""
    snap0 = _typed_snapshot()
    snap1 = HeteroData()
    snap1["gen"].x = torch.randn(2, 3)
    snap1["load"].x = torch.randn(3, 2)
    snap1["bus"].x = torch.randn(1, 2)
    snap1["gen", "feeds", "load"].edge_index = torch.tensor(
        [[0, 1], [0, 2]],
        dtype=torch.long,
    )
    with pytest.raises(ValueError, match="node types"):
        validate_hetero_snapshot_metadata([snap0, snap1])

    snap1b = _typed_snapshot()
    feeds = snap1b["gen", "feeds", "load"].edge_index
    snap1b["gen", "other", "load"].edge_index = feeds
    del snap1b["gen", "feeds", "load"]
    with pytest.raises(ValueError, match="edge-type set"):
        validate_hetero_snapshot_metadata([snap0, snap1b])

    edge_types = [("node", "r1", "node"), ("node", "r2", "node")]
    assert hetero_snapshots_have_dynamic_topology([], edge_types) is False
    m0 = _multiplex_snapshot()
    m1 = _multiplex_snapshot()
    m1["node", "r1", "node"].edge_index = torch.tensor(
        [[0, 1], [1, 0]],
        dtype=torch.long,
    )
    assert hetero_snapshots_have_dynamic_topology([m0, m1], edge_types) is True

    w0 = _multiplex_snapshot()
    w1 = _multiplex_snapshot()
    w0["node", "r1", "node"].edge_weight = torch.ones(3)
    with pytest.raises(ValueError, match="presence does not match"):
        validate_shared_hetero_topology([w0, w1])
    w1["node", "r1", "node"].edge_weight = torch.full((3,), 2.0)
    with pytest.raises(ValueError, match="different edge_weight"):
        validate_shared_hetero_topology([w0, w1])


def test_validate_hetero_observation_masks_errors() -> None:
    """Per-type observation mask validation wraps key / shape failures."""
    with pytest.raises(ValueError, match="keys must match node types"):
        validate_hetero_observation_masks(
            {"gen": torch.ones(2, 2, dtype=torch.bool)},
            num_timesteps=2,
            num_nodes={"gen": 2, "load": 3},
        )
    with pytest.raises(ValueError, match=r"observation_masks\['gen'\]"):
        validate_hetero_observation_masks(
            {
                "gen": torch.ones(2, 5, dtype=torch.bool),
                "load": torch.ones(2, 3, dtype=torch.bool),
            },
            num_timesteps=2,
            num_nodes={"gen": 2, "load": 3},
        )


# ---------------------------------------------------------------------------
# operators.heterogeneous
# ---------------------------------------------------------------------------


def test_hetero_operator_constructor_validation() -> None:
    """Constructor rejects bad dims, tying, normalization, and type metadata."""
    with pytest.raises(ValueError, match="latent_dim must be positive"):
        HeteroGraphKoopmanOperator(latent_dim=0, num_relations=1)
    with pytest.raises(ValueError, match="num_relations must be positive"):
        HeteroGraphKoopmanOperator(latent_dim=2, num_relations=0)
    with pytest.raises(ValueError, match="control_dim must be non-negative"):
        HeteroGraphKoopmanOperator(latent_dim=2, num_relations=1, control_dim=-1)
    with pytest.raises(ValueError, match="normalization must be one of"):
        HeteroGraphKoopmanOperator(
            latent_dim=2,
            num_relations=1,
            normalization="symmetric",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="must be a positive int"):
        HeteroGraphKoopmanOperator(
            latent_dim=2,
            num_relations=2,
            relation_tying="basis",
            basis_size=0,
        )
    with pytest.raises(ValueError, match="at least one node type"):
        _normalize_node_types(())
    with pytest.raises(ValueError, match="non-empty strings"):
        _normalize_node_types(("", "a"))
    with pytest.raises(ValueError, match="must be unique"):
        _normalize_node_types(("a", "a"))
    with pytest.raises(ValueError, match="edge_types is required for typed"):
        HeteroGraphKoopmanOperator(
            latent_dim=2,
            num_relations=1,
            node_types=("a", "b"),
        )
    with pytest.raises(ValueError, match=r"\(src, rel, dst\)"):
        _normalize_edge_types(
            [("a", "b")],
            num_relations=1,
            node_types=("a", "b"),
        )
    with pytest.raises(ValueError, match="non-empty strings"):
        _normalize_edge_types(
            [("", "r", "b")],
            num_relations=1,
            node_types=("a", "b"),
        )
    with pytest.raises(ValueError, match="must match"):
        _normalize_edge_types(
            [("a", "r0", "b")],
            num_relations=2,
            node_types=("a", "b"),
        )
    with pytest.raises(ValueError, match="unique after key encoding"):
        _normalize_edge_types(
            [("a", "r0", "b"), ("a", "r0", "b")],
            num_relations=2,
            node_types=("a", "b"),
        )
    with pytest.raises(ValueError, match="outside node_types"):
        _normalize_edge_types(
            [("a", "r0", "ghost")],
            num_relations=1,
            node_types=("a", "b"),
        )


def test_hetero_operator_runtime_validation_and_basis_paths() -> None:
    """Forward / effective / basis helpers reject invalid overrides."""
    edges = [
        torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        torch.tensor([[0], [1]], dtype=torch.long),
    ]
    op = HeteroGraphKoopmanOperator(latent_dim=2, num_relations=2)
    z = torch.randn(3, 2)
    with pytest.raises(ValueError, match="expects z with shape"):
        op(torch.randn(3), edges)
    with pytest.raises(ValueError, match="Expected trailing dimension"):
        op(torch.randn(3, 3), edges)
    with pytest.raises(ValueError, match="relation weight banks"):
        op(z, edges, edge_weights=[None])

    with pytest.raises(ValueError, match="at most one of k_self"):
        op.effective_matrix(
            edges,
            num_nodes=3,
            k_self=torch.eye(2),
            k_self_blocks=torch.eye(2).expand(3, 2, 2).clone(),
        )
    with pytest.raises(ValueError, match="k_self_blocks must have shape"):
        op.effective_matrix(
            edges,
            num_nodes=3,
            k_self_blocks=torch.zeros(3, 2, 3),
        )

    block = HeteroGraphKoopmanOperator(
        latent_dim=2,
        num_relations=1,
        sparsity="block_diagonal",
    )
    block_edges = [torch.tensor([[0], [1]], dtype=torch.long)]
    with pytest.raises(ValueError, match="requires sparsity='dense'"):
        block.dense_effective_inverse(block_edges, num_nodes=2)

    basis = HeteroGraphKoopmanOperator(
        latent_dim=2,
        num_relations=2,
        relation_tying="basis",
        basis_size=1,
    )
    with pytest.raises(ValueError, match="requires relation_tying='basis'"):
        op.set_basis_factors([torch.eye(2)], torch.ones(2, 1))
    with pytest.raises(ValueError, match="coefficients must have shape"):
        basis.set_basis_factors([torch.eye(2)], torch.ones(2, 2))
    with pytest.raises(ValueError, match="only defined for relation_tying"):
        basis._relation_modules()  # noqa: SLF001
    metric = basis.bound_metric()
    assert torch.isfinite(metric).all()

    control_op = HeteroGraphKoopmanOperator(
        latent_dim=2,
        num_relations=1,
        control_dim=1,
        control_mode="bilinear",
    )
    edge = [torch.tensor([[0, 1], [1, 0]], dtype=torch.long)]
    z_c = torch.randn(2, 2)
    nxt = control_op(z_c, edge, control=torch.ones(2, 1))
    back = control_op.inverse_advance(
        nxt,
        edge_indices=edge,
        control=torch.ones(2, 1),
    )
    assert back.shape == z_c.shape
    with pytest.raises(ValueError, match="only supported for"):
        block.inverse_advance(
            torch.randn(2, 2),
            edge_indices=block_edges,
            inverse_matrix=torch.eye(4),
        )


# ---------------------------------------------------------------------------
# nn.heterogeneous
# ---------------------------------------------------------------------------


def test_relgraph_conv_validation_and_bias_none() -> None:
    """RelGraphConv validates inputs and supports bias=False."""
    layer = RelGraphConv(3, 4, num_relations=2, bias=False)
    assert layer.bias is None
    layer.reset_parameters()
    x = torch.randn(4, 3)
    edges = [
        torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        torch.tensor([[0], [3]], dtype=torch.long),
    ]
    with pytest.raises(ValueError, match="Expected x with shape"):
        layer(torch.randn(4, 3, 1), edges)
    with pytest.raises(ValueError, match="Expected in_channels"):
        layer(torch.randn(4, 2), edges)
    with pytest.raises(ValueError, match="relation edge banks"):
        layer(x, edges[:1])
    with pytest.raises(ValueError, match="relation weight banks"):
        layer(x, edges, edge_weights=[None])
    weights = [torch.ones(2), torch.ones(1)]
    out = layer(x, edges, edge_weights=weights)
    assert out.shape == (4, 4)


def test_resolve_relation_input_helpers_errors() -> None:
    """Multiplex / typed / HGT resolvers reject malformed inputs."""
    with pytest.raises(ValueError, match="relation banks"):
        resolve_multiplex_relation_inputs(
            torch.randn(3, 2),
            edge_index={"r1": torch.tensor([[0], [1]], dtype=torch.long)},
            num_relations=2,
        )
    with pytest.raises(ValueError, match="exactly one node type"):
        resolve_multiplex_relation_inputs(_typed_snapshot(), num_relations=1)

    with pytest.raises(ValueError, match="node_types must be unique"):
        RelGraphEncoder(
            {"a": 2, "b": 3},
            hidden_channels=4,
            latent_dim=2,
            num_relations=1,
            node_types=("a", "a"),
            edge_types=(("a", "r", "b"),),
        )
    with pytest.raises(ValueError, match=r"\(src, rel, dst\)"):
        _normalize_edge_type_order([("a", "b")], num_relations=1)
    with pytest.raises(ValueError, match="non-empty strings"):
        _normalize_edge_type_order([("", "r", "b")], num_relations=1)
    with pytest.raises(ValueError, match="must match"):
        _normalize_edge_type_order([("a", "r0", "b")], num_relations=2)
    with pytest.raises(ValueError, match="must be unique"):
        _normalize_edge_type_order(
            [("a", "r0", "b"), ("a", "r0", "b")],
            num_relations=2,
        )
    with pytest.raises(ValueError, match="edge_types is required"):
        _normalize_edge_type_order(None, num_relations=1, required=True)

    typed = _typed_snapshot()
    feats, edges, _weights, counts = resolve_typed_relation_inputs(
        typed,
        node_types=("gen", "load"),
        edge_types=None,
        num_relations=1,
    )
    assert set(feats) == {"gen", "load"}
    assert len(edges) == 1
    assert counts["gen"] == 2

    with pytest.raises(TypeError, match="HeteroData or a mapping"):
        resolve_typed_relation_inputs(
            42,  # type: ignore[arg-type]
            node_types=("a",),
            num_relations=1,
        )
    with pytest.raises(ValueError, match="must have shape"):
        resolve_typed_relation_inputs(
            {"gen": torch.randn(4), "load": torch.randn(3, 2)},
            edge_index=[torch.tensor([[0], [2]], dtype=torch.long)],
            node_types=("gen", "load"),
            edge_types=(("gen", "feeds", "load"),),
            num_relations=1,
        )

    with pytest.raises(ValueError, match="Unknown activation"):
        _resolve_hgt_activation("gelu")  # type: ignore[arg-type]

    hgt_snap = HeteroData()
    hgt_snap["a"].x = torch.randn(2, 2)
    hgt_snap["b"].x = torch.randn(2, 2)
    hgt_snap["c"].x = torch.randn(1, 2)
    hgt_snap["a", "r0", "b"].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    with pytest.raises(ValueError, match="expects exactly"):
        resolve_hgt_typed_inputs(
            hgt_snap,
            node_types=("a", "b"),
            edge_types=(("a", "r0", "b"),),
        )
    missing_edge = HeteroData()
    missing_edge["a"].x = torch.randn(2, 2)
    missing_edge["b"].x = torch.randn(2, 2)
    with pytest.raises(ValueError, match="missing edge type"):
        resolve_hgt_typed_inputs(
            missing_edge,
            node_types=("a", "b"),
            edge_types=(("a", "r0", "b"),),
        )
    with pytest.raises(ValueError, match="edge_index relation mapping"):
        resolve_hgt_typed_inputs(
            {"a": torch.randn(2, 2), "b": torch.randn(2, 2)},
            edge_index=None,
            node_types=("a", "b"),
            edge_types=(("a", "r0", "b"),),
        )


# ---------------------------------------------------------------------------
# training.pair_objectives / objectives
# ---------------------------------------------------------------------------


def test_pair_objective_hetero_guards() -> None:
    """Pair helpers reject unsupported hetero training surfaces."""
    typed = _typed_snapshot()
    with pytest.raises(ValueError, match="exactly one node type"):
        multiplex_node_features(typed)

    homo = GraphKoopmanModel(
        encoder=GNNEncoder(3, 8, 4, num_layers=1),
        decoder=GNNDecoder(4, 8, 3, num_layers=1),
        latent_dim=4,
        time_step=1.0,
    )
    with pytest.raises(TypeError, match="HeteroGraphKoopmanOperator"):
        _hetero_num_relations(homo)

    with pytest.raises(ValueError, match="missing node type"):
        typed_node_features(_typed_snapshot(), ("gen", "bus"))

    class _StoreNone:
        x = None

    class _TypedMissing:
        node_types = ("gen",)

        def __getitem__(self, _key: str) -> _StoreNone:
            return _StoreNone()

    with pytest.raises(ValueError, match="missing feature matrix x"):
        typed_node_features(_TypedMissing(), ("gen",))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="missing node type"):
        stack_typed_masks(
            {"gen": torch.ones(2, dtype=torch.bool)},
            ("gen", "load"),
        )

    seq = HeteroGraphSnapshotSequence([_multiplex_snapshot() for _ in range(2)])
    controlled = HeteroGraphSnapshotSequence(
        [_multiplex_snapshot() for _ in range(2)],
        control_inputs=torch.randn(2, 1),
    )
    with pytest.raises(ValueError, match="controlled HeteroGraphSnapshotSequence"):
        pair_control(controlled, 0)

    model = _multiplex_model()
    with pytest.raises(ValueError, match="backward consistency is unsupported"):
        _backward_consistency_pair(model, seq, 0)

    pred = one_step_prediction(model, seq, 0)
    assert isinstance(pred, torch.Tensor)
    assert pred.shape[0] == 4


def test_objectives_hetero_fit_surface_guards() -> None:
    """Eigenvalue / fit-surface helpers reject unsupported hetero mixes."""
    multi = [_multiplex_snapshot() for _ in range(2)]
    multi[1]["node", "r1", "node"].edge_index = torch.tensor(
        [[0, 1], [1, 0]],
        dtype=torch.long,
    )
    dynamic = HeteroGraphSnapshotSequence(multi, allow_dynamic_topology=True)
    with pytest.raises(ValueError, match="dynamic-topology"):
        _hetero_relation_banks(dynamic)

    seq = HeteroGraphSnapshotSequence([_multiplex_snapshot() for _ in range(3)])
    one_rel = _multiplex_model(num_relations=1)
    with pytest.raises(ValueError, match="num_relations"):
        _hetero_eigenvalue_regularization_over_sequence(one_rel, seq)

    graph = GraphKoopmanModel(
        encoder=GNNEncoder(3, 8, 4, num_layers=1),
        decoder=GNNDecoder(4, 8, 3, num_layers=1),
        latent_dim=4,
        time_step=1.0,
        koopman="graph",
    )
    with pytest.raises(ValueError, match="not HeteroGraphSnapshotSequence"):
        compute_eigenvalue_regularization_loss(graph, seq)

    model = _multiplex_model()
    homo_seq = GraphSnapshotSequence(
        [
            Data(
                x=torch.randn(4, 3),
                edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            )
            for _ in range(2)
        ]
    )
    with pytest.raises(ValueError, match="requires a HeteroGraphSnapshotSequence"):
        compute_eigenvalue_regularization_loss(model, homo_seq)

    weights = LossWeights(reconstruction=1.0, forward=1.0)
    with pytest.raises(ValueError, match="dynamic-topology"):
        _validate_hetero_fit_surface(dynamic, weights, extra_losses=None)

    controlled = HeteroGraphSnapshotSequence(
        [_multiplex_snapshot() for _ in range(2)],
        control_inputs=torch.randn(2, 1),
    )
    with pytest.raises(ValueError, match="controlled"):
        _validate_hetero_fit_surface(controlled, weights, extra_losses=None)

    stamped = HeteroGraphSnapshotSequence(
        [_multiplex_snapshot() for _ in range(2)],
        timestamps=torch.tensor([0.0, 1.0]),
    )
    with pytest.raises(ValueError, match="timestamped"):
        _validate_hetero_fit_surface(stamped, weights, extra_losses=None)

    with pytest.raises(ValueError, match="lie / pde / worst_case"):
        _validate_hetero_fit_surface(
            seq,
            LossWeights(reconstruction=1.0, lie=0.1),
            extra_losses=None,
        )
    with pytest.raises(ValueError, match="lie_dynamics_fn"):
        _validate_hetero_fit_surface(
            seq,
            weights,
            extra_losses=ExtraLosses(lie_dynamics_fn=lambda *_a, **_k: 0.0),
        )


# ---------------------------------------------------------------------------
# Second wave: layout / factory / serialization / IEEE / HGT / containers
# ---------------------------------------------------------------------------


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


def test_factory_typed_peer_validation() -> None:
    """Typed RelGraph peers must agree with the hetero operator schema."""
    operator = HeteroGraphKoopmanOperator(
        latent_dim=2,
        num_relations=1,
        node_types=("a", "b"),
        edge_types=(("a", "r0", "b"),),
    )
    enc_multi = RelGraphEncoder(3, 4, 2, num_relations=1, num_layers=1)
    dec_typed = RelGraphDecoderCls(
        2,
        4,
        {"a": 3, "b": 2},
        num_relations=1,
        node_types=("a", "b"),
        edge_types=(("a", "r0", "b"),),
        num_layers=1,
    )
    with pytest.raises(ValueError, match="must agree on typed channels"):
        validate_typed_relgraph_peers(enc_multi, dec_typed, operator)

    enc_typed = RelGraphEncoder(
        {"a": 3, "b": 2},
        4,
        2,
        num_relations=1,
        node_types=("a", "b"),
        edge_types=(("a", "r0", "b"),),
        num_layers=1,
    )
    with pytest.raises(ValueError, match="declares node types"):
        validate_typed_relgraph_peers(
            enc_multi,
            RelGraphDecoder(2, 4, 3, num_relations=1, num_layers=1),
            operator,
        )
    wrong_enc = RelGraphEncoder(
        {"b": 2, "a": 3},
        4,
        2,
        num_relations=1,
        node_types=("b", "a"),
        edge_types=(("a", "r0", "b"),),
        num_layers=1,
    )
    with pytest.raises(ValueError, match="RelGraphEncoder node types"):
        validate_typed_relgraph_peers(wrong_enc, dec_typed, operator)
    wrong_dec = RelGraphDecoderCls(
        2,
        4,
        {"a": 3, "b": 2},
        num_relations=1,
        node_types=("b", "a"),
        edge_types=(("a", "r0", "b"),),
        num_layers=1,
    )
    with pytest.raises(ValueError, match="RelGraphDecoder node types"):
        validate_typed_relgraph_peers(enc_typed, wrong_dec, operator)
    wrong_edge_dec = RelGraphDecoderCls(
        2,
        4,
        {"a": 3, "b": 2},
        num_relations=1,
        node_types=("a", "b"),
        edge_types=(("b", "r0", "a"),),
        num_layers=1,
    )
    with pytest.raises(ValueError, match="edge types"):
        validate_typed_relgraph_peers(enc_typed, wrong_edge_dec, operator)


def test_require_hetero_schema_validation() -> None:
    """Hetero checkpoint schema rejects incomplete / inconsistent fields."""
    base = {
        "node_types": ["node"],
        "edge_types": [("node", "r0", "node")],
        "relation_tying": "independent",
        "basis_size": None,
        "relation_normalization": "rgcn_in_degree",
    }
    with pytest.raises(ValueError, match="Incomplete hetero checkpoint"):
        _require_hetero_schema({"node_types": ["node"]})
    with pytest.raises(ValueError, match="non-empty sequence of strings"):
        _require_hetero_schema({**base, "node_types": []})
    with pytest.raises(ValueError, match="entries must be non-empty strings"):
        _require_hetero_schema({**base, "node_types": [""]})
    with pytest.raises(ValueError, match="non-empty sequence of"):
        _require_hetero_schema({**base, "edge_types": []})
    with pytest.raises(ValueError, match=r"\(src, rel, dst\)"):
        _require_hetero_schema({**base, "edge_types": [("node", "r0")]})
    with pytest.raises(ValueError, match="triples must use non-empty"):
        _require_hetero_schema({**base, "edge_types": [("node", "", "node")]})
    with pytest.raises(ValueError, match="relation_tying must be one of"):
        _require_hetero_schema({**base, "relation_tying": "tied"})
    with pytest.raises(ValueError, match="basis_size must be null"):
        _require_hetero_schema({**base, "basis_size": 1})
    with pytest.raises(ValueError, match="positive int"):
        _require_hetero_schema({**base, "relation_tying": "basis", "basis_size": True})
    with pytest.raises(ValueError, match="positive int"):
        _require_hetero_schema({**base, "relation_tying": "basis", "basis_size": 0})
    with pytest.raises(ValueError, match="relation_normalization must be"):
        _require_hetero_schema({**base, "relation_normalization": "symmetric"})
    with pytest.raises(ValueError, match="must match"):
        _require_hetero_schema(
            {
                **base,
                "encoder": {"normalization": "random_walk"},
            }
        )


def test_ieee118_typed_helpers_and_rollout_guards() -> None:
    """IEEE typed helpers and rollout multiplex guards raise on bad inputs."""
    with pytest.raises(ValueError, match="Unsupported MATPOWER BUS_TYPE"):
        bus_type_name(9)
    with pytest.raises(ValueError, match="rank-1"):
        partition_buses_by_type(torch.ones(2, 2, dtype=torch.long))
    with pytest.raises(ValueError, match="Unsupported MATPOWER BUS_TYPE codes"):
        partition_buses_by_type(torch.tensor([1, 9], dtype=torch.long))

    with pytest.raises(ValueError, match="exactly one node type"):
        _multiplex_target_features(_typed_snapshot())
    topo = _relation_topology_at_from_targets(
        [_multiplex_snapshot()],
        num_relations=2,
    )
    edges, weights = topo(0)
    assert len(edges) == 2
    typed_topo = _relation_topology_at_from_targets(
        [_typed_snapshot()],
        num_relations=1,
        node_types=("gen", "load"),
        edge_types=(("gen", "feeds", "load"),),
    )
    t_edges, _t_weights = typed_topo(0)
    assert len(t_edges) == 1
    typed_dec = RelGraphDecoderCls(
        2,
        4,
        {"gen": 3, "load": 2},
        num_relations=1,
        node_types=("gen", "load"),
        edge_types=(("gen", "feeds", "load"),),
        num_layers=1,
    )
    bound = _bind_hetero_decoder(typed_dec, {"gen": 2, "load": 3})
    assert callable(bound)


def test_hgt_decoder_and_container_error_paths() -> None:
    """HGT decoder and hetero sequence accessors reject invalid inputs."""
    decoder = HGTDecoder(
        latent_dim=2,
        hidden_channels=4,
        out_channels={"a": 2, "b": 2},
        node_types=("a", "b"),
        edge_types=(("a", "r0", "b"),),
        num_layers=1,
        heads=1,
    )
    snap = HeteroData()
    snap["a"].x = torch.randn(2, 2)
    snap["b"].x = torch.randn(2, 2)
    snap["a", "r0", "b"].edge_index = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    z = torch.randn(4, 2)
    with pytest.raises(ValueError, match="does not match HeteroData counts"):
        decoder(z, snap, num_nodes_dict={"a": 9, "b": 2})
    edge_map = {("a", "r0", "b"): snap["a", "r0", "b"].edge_index}
    with pytest.raises(ValueError, match="num_nodes_dict is required"):
        decoder(z, edge_map, num_nodes_dict=None)
    with pytest.raises(ValueError, match="z must have shape"):
        decoder(torch.randn(4, 3), snap)
    with pytest.raises(ValueError, match="num_nodes_dict sums"):
        decoder(torch.randn(3, 2), snap)

    multi = HeteroGraphSnapshotSequence([_multiplex_snapshot() for _ in range(2)])
    with pytest.raises(KeyError, match="unknown node type"):
        multi.num_nodes_of("ghost")
    typed_seq = HeteroGraphSnapshotSequence([_typed_snapshot() for _ in range(2)])
    with pytest.raises(ValueError, match="num_nodes is defined only"):
        _ = typed_seq.num_nodes
    with pytest.raises(ValueError, match="does not contain observation_masks"):
        multi.observation_mask_at(0)
    with pytest.raises(ValueError, match="does not contain observation_masks"):
        multi.pair_observation_mask(0)
    masked = HeteroGraphSnapshotSequence(
        [_multiplex_snapshot() for _ in range(2)],
        observation_masks={"node": torch.ones(2, 4, dtype=torch.bool)},
    )
    with pytest.raises(IndexError, match="0 <= index"):
        masked.observation_mask_at(5)
    with pytest.raises(IndexError, match="0 <= index"):
        masked.pair_observation_mask(5)
    # Per-type control dict path on control_dim.
    ctrl = HeteroGraphSnapshotSequence(
        [_typed_snapshot() for _ in range(2)],
        control_inputs={
            "gen": torch.randn(2, 1),
            "load": torch.randn(2, 1),
        },
    )
    assert ctrl.control_dim == 1
    assert ctrl.has_controls
    assert len(ctrl.snapshots) == 2


# ---------------------------------------------------------------------------
# Third wave: close the last ~20 executable patch lines to ≥90%
# ---------------------------------------------------------------------------


def test_attribute_mode_energy_and_control_input_guards() -> None:
    """Spectrum attribution and hetero control validation reject bad inputs."""
    k_eff = torch.eye(6)
    eigenvectors = torch.eye(6)
    with pytest.raises(ValueError, match="eigenvectors must have shape"):
        attribute_mode_energy(k_eff, torch.ones(6), latent_dim=2)
    with pytest.raises(ValueError, match="rows must match"):
        attribute_mode_energy(k_eff, torch.eye(4, 6), latent_dim=2)
    with pytest.raises(ValueError, match="latent_dim must be positive"):
        attribute_mode_energy(k_eff, eigenvectors, latent_dim=0)
    with pytest.raises(ValueError, match="not divisible by"):
        attribute_mode_energy(k_eff, eigenvectors, latent_dim=5)
    with pytest.raises(ValueError, match="latent slice"):
        attribute_mode_energy(
            k_eff,
            eigenvectors,
            latent_dim=2,
            node_type_slices={"a": slice(0, 4)},
            mode_indices=(0,),
        )
    report = attribute_mode_energy(
        k_eff,
        eigenvectors,
        latent_dim=2,
        node_type_slices={"a": slice(0, 2), "b": slice(2, 3)},
        mode_indices=(0,),
    )
    assert report.type_fractions["a"].shape == (1,)

    with pytest.raises(ValueError, match="non-empty"):
        validate_hetero_control_inputs({}, num_timesteps=2, num_nodes={"node": 4})
    with pytest.raises(ValueError, match="unknown node types"):
        validate_hetero_control_inputs(
            {"ghost": torch.randn(2, 1)},
            num_timesteps=2,
            num_nodes={"node": 4},
        )
    with pytest.raises(ValueError, match="control_inputs\\['node'\\]"):
        validate_hetero_control_inputs(
            {"node": torch.randn(2, 5, 1)},
            num_timesteps=2,
            num_nodes={"node": 4},
        )
    with pytest.raises(ValueError, match="share a common control_dim"):
        validate_hetero_control_inputs(
            {
                "gen": torch.randn(2, 1),
                "load": torch.randn(2, 2),
            },
            num_timesteps=2,
            num_nodes={"gen": 2, "load": 3},
        )
    with pytest.raises(ValueError, match=r"\(T, N, C\) require a single node type"):
        validate_hetero_control_inputs(
            torch.randn(2, 5, 1),
            num_timesteps=2,
            num_nodes={"gen": 2, "load": 3},
        )


def test_ieee_typed_conversion_and_model_validation_guards() -> None:
    """IEEE typed conversion and RelGraph peer checks cover remaining branches."""
    features = torch.randn(4, 4)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    bus_types = torch.tensor([1, 2, 3, 1], dtype=torch.long)
    with pytest.raises(ValueError, match=r"\(num_buses, F\)"):
        homogeneous_features_to_typed_hetero(features[0], edge_index, bus_types)
    with pytest.raises(ValueError, match="must match"):
        homogeneous_features_to_typed_hetero(
            features,
            edge_index,
            torch.tensor([1, 2, 3], dtype=torch.long),
        )
    with pytest.raises(ValueError, match="partition is empty"):
        homogeneous_features_to_typed_hetero(
            features,
            edge_index,
            bus_types,
            node_type_order=("ghost",),
        )
    hetero = homogeneous_features_to_typed_hetero(features, edge_index, bus_types)
    assert set(hetero.node_types) <= {"generator", "load", "slack"}

    with pytest.raises(ValueError, match="must be used together"):
        uses_relgraph_modules(
            RelGraphEncoder(3, 4, 2, num_relations=1, num_layers=1),
            GNNDecoder(2, 4, 3),
        )
    with pytest.raises(ValueError, match="num_relations"):
        uses_relgraph_modules(
            RelGraphEncoder(3, 4, 2, num_relations=1, num_layers=1),
            RelGraphDecoder(2, 4, 3, num_relations=2, num_layers=1),
        )
    with pytest.raises(ValueError, match="normalization"):
        uses_relgraph_modules(
            RelGraphEncoder(
                3, 4, 2, num_relations=1, num_layers=1, normalization="rgcn_in_degree"
            ),
            RelGraphDecoder(
                2, 4, 3, num_relations=1, num_layers=1, normalization="random_walk"
            ),
        )
    seq = HeteroGraphSnapshotSequence([_multiplex_snapshot() for _ in range(2)])
    with pytest.raises(ValueError, match="homogeneous hyperedge-carrying"):
        validate_sequence_hyperedges(seq, allow_hyperedges=True)
    validate_sequence_hyperedges(seq, allow_hyperedges=False)


def test_training_input_and_pair_objective_remaining_guards() -> None:
    """Fit-input classifiers and hetero pair helpers hit remaining miss lines."""
    with pytest.raises(TypeError, match="single HeteroData is not a trajectory"):
        resolve_training_sequences(_multiplex_snapshot())
    with pytest.raises(TypeError, match="single HeteroData is not a trajectory"):
        resolve_validation_sequences(_multiplex_snapshot(), num_training_sequences=1)

    class _Snap:
        node_types = ("node",)

        def __getitem__(self, _key: str) -> SimpleNamespace:
            return SimpleNamespace(x=None)

    with pytest.raises(ValueError, match="missing feature matrix x"):
        multiplex_node_features(_Snap())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="missing feature matrix x"):
        _multiplex_target_features(_Snap())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="missing node type"):
        _typed_target_features(_typed_snapshot(), ("gen", "ghost"))
    with pytest.raises(ValueError, match="missing feature matrix x"):
        _typed_target_features(_Snap(), ("node",))  # type: ignore[arg-type]

    model = _multiplex_model()
    snap_t = _multiplex_snapshot()
    snap_t1 = _multiplex_snapshot()
    with pytest.raises(ValueError, match="target_mask is unsupported"):
        one_step_loss(
            model,
            snap_t,
            snap_t1,
            target_mask=torch.ones(4, dtype=torch.bool),
        )
    seq = HeteroGraphSnapshotSequence([_multiplex_snapshot() for _ in range(2)])
    loss = _one_step_pair(model, seq, 0)
    assert loss.ndim == 0

    pred = torch.randn(4, 3)
    step = _hetero_rollout_step_loss(pred, seq[1], node_types=None, masks=None)
    assert step.ndim == 0
    masked = HeteroGraphSnapshotSequence(
        [_multiplex_snapshot() for _ in range(2)],
        observation_masks={"node": torch.ones(2, 4, dtype=torch.bool)},
    )
    step_m = _hetero_rollout_step_loss(
        pred,
        masked[1],
        node_types=None,
        masks=masked.observation_mask_at(1),
    )
    assert step_m.ndim == 0

    enc = RelGraphEncoder(3, 4, 2, num_relations=1, num_layers=1)
    edges = [torch.tensor([[0, 1], [1, 0]], dtype=torch.long)]
    with pytest.raises(ValueError, match="Expected x with shape"):
        _relgraph_message_passing(enc, torch.randn(3), edges, [None])
    with pytest.raises(ValueError, match="Expected in_channels"):
        _relgraph_message_passing(enc, torch.randn(3, 5), edges, [None])

    op = HeteroGraphKoopmanOperator(
        latent_dim=2,
        num_relations=1,
        init_mode="identity_noise",
        init_scale=0.01,
        parameterization="odo",
    )
    op.reset_parameters()
    assert op.num_relations == 1
