"""Multiplex hetero tests for ``HierarchicalGraphKoopmanModel``."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch_geometric.data import Data, HeteroData

from koopman_graph.data import GraphSnapshotSequence, HeteroGraphSnapshotSequence
from koopman_graph.hierarchical import HierarchicalGraphKoopmanModel
from koopman_graph.hierarchical.model import _channels_by_type
from koopman_graph.hierarchical.pooling import (
    PoolStep,
    apply_pool_layer,
    build_pool_layer,
    filter_subgraph,
    induce_typed_relation_banks,
    multiplex_snapshot_from_features,
    pool_control,
    pool_control_sequence,
    pool_features_with_steps,
    pool_multiplex_features_with_steps,
    pool_typed_features_with_steps,
    resolve_snapshot_inputs,
    snapshot_from_features,
    typed_snapshot_from_features,
)
from koopman_graph.model import GraphKoopmanModel
from koopman_graph.nn import GNNDecoder, GNNEncoder, RelGraphDecoder, RelGraphEncoder


def _multiplex_snapshot(*, seed: int = 0, num_nodes: int = 8) -> HeteroData:
    generator = torch.Generator().manual_seed(seed)
    data = HeteroData()
    data["node"].x = torch.randn(num_nodes, 3, generator=generator)
    # Dense-ish cycle + extras so TopK retention still has edges.
    src = torch.arange(num_nodes, dtype=torch.long)
    dst = (src + 1) % num_nodes
    data["node", "r1", "node"].edge_index = torch.stack([src, dst], dim=0)
    data["node", "r2", "node"].edge_index = torch.stack([dst, src], dim=0)
    return data


def _multiplex_sequence(
    *,
    timesteps: int = 4,
    seed: int = 0,
    num_nodes: int = 8,
) -> HeteroGraphSnapshotSequence:
    return HeteroGraphSnapshotSequence(
        [
            _multiplex_snapshot(seed=seed + t, num_nodes=num_nodes)
            for t in range(timesteps)
        ]
    )


def _multiplex_model(*, seed: int = 0) -> GraphKoopmanModel:
    torch.manual_seed(seed)
    return GraphKoopmanModel(
        encoder=RelGraphEncoder(
            3,
            hidden_channels=8,
            latent_dim=4,
            num_relations=2,
            num_layers=1,
        ),
        decoder=RelGraphDecoder(
            latent_dim=4,
            hidden_channels=8,
            out_channels=3,
            num_relations=2,
            num_layers=1,
        ),
        latent_dim=4,
        time_step=1.0,
        koopman="hetero_graph",
        koopman_edge_types=(
            ("node", "r1", "node"),
            ("node", "r2", "node"),
        ),
    )


def test_multiplex_pool_down_preserves_relation_schema() -> None:
    """Coarse multiplex snapshot keeps operator edge types."""
    model = _multiplex_model()
    hier = HierarchicalGraphKoopmanModel(model, pool_ratios=(0.5,))
    fine = _multiplex_snapshot(seed=0, num_nodes=8)
    coarse, steps = hier.pool_down(fine)
    assert isinstance(coarse, HeteroData)
    assert coarse["node"].x is not None
    assert coarse["node"].x.size(0) == 4
    assert set(coarse.edge_types) == {
        ("node", "r1", "node"),
        ("node", "r2", "node"),
    }
    assert steps[0].relation_edge_indices is not None
    assert len(steps[0].relation_edge_indices) == 2


def test_multiplex_predict_fine_and_coarse() -> None:
    """Predict returns HeteroData at fine and coarse resolutions."""
    model = _multiplex_model()
    hier = HierarchicalGraphKoopmanModel(model, pool_ratios=(0.5,))
    origin = _multiplex_snapshot(seed=1, num_nodes=8)
    fine_preds = hier.predict(origin, steps=2, resolution="fine")
    assert len(fine_preds) == 2
    assert isinstance(fine_preds[0], HeteroData)
    assert fine_preds[0]["node"].x.shape == (8, 3)
    assert set(fine_preds[0].edge_types) == set(origin.edge_types)

    coarse_preds = hier.predict(origin, steps=1, resolution="coarse")
    assert isinstance(coarse_preds[0], HeteroData)
    assert coarse_preds[0]["node"].x.shape[0] == 4


def test_multiplex_fit_smoke() -> None:
    """Short multiplex hierarchical fit completes with finite loss.

    Hetero ``fit`` rejects dynamic relation banks, so multiplex hierarchical
    training uses ``hold_perm`` (static coarse banks). ``per_snapshot`` still
    works for ``pool_down`` / ``predict``.
    """
    torch.manual_seed(0)
    model = _multiplex_model()
    hier = HierarchicalGraphKoopmanModel(
        model,
        pool_ratios=(0.5,),
        pool_schedule="hold_perm",
    )
    sequence = _multiplex_sequence(timesteps=5, num_nodes=8)
    history = hier.fit(sequence, epochs=1, lr=1e-2, unpool_epochs=1)
    assert len(history.loss) == 1
    assert torch.isfinite(torch.tensor(history.loss[0]))


def test_multiplex_per_snapshot_fit_requires_hold_perm() -> None:
    """``per_snapshot`` multiplex fit raises toward ``hold_perm``."""
    model = _multiplex_model()
    hier = HierarchicalGraphKoopmanModel(model, pool_ratios=(0.5,))
    sequence = _multiplex_sequence(timesteps=4, num_nodes=8)
    with pytest.raises(ValueError, match="hold_perm"):
        hier.fit(sequence, epochs=1, lr=1e-2, unpool_epochs=0)


def test_checkpoint_preserves_coarse_relation_schema(tmp_path: Path) -> None:
    """Manifest records multiplex node/edge types for coarse schema."""
    model = _multiplex_model()
    hier = HierarchicalGraphKoopmanModel(model, pool_ratios=(0.5,))
    out = tmp_path / "hier"
    hier.save(out)
    manifest = json.loads((out / "hierarchical_manifest.json").read_text())
    assert manifest["hetero_mode"] == "multiplex"
    assert manifest["node_types"] == ["node"]
    assert manifest["edge_types"] == [
        ["node", "r1", "node"],
        ["node", "r2", "node"],
    ]
    loaded = HierarchicalGraphKoopmanModel.load(out)
    assert loaded._uses_hetero is True
    assert loaded._edge_types == (
        ("node", "r1", "node"),
        ("node", "r2", "node"),
    )
    coarse, _ = loaded.pool_down(_multiplex_snapshot(seed=2, num_nodes=8))
    assert isinstance(coarse, HeteroData)


def test_homo_default_pool_schedule_unchanged() -> None:
    """Homogeneous hierarchical default pool_schedule remains per_snapshot."""
    from koopman_graph.nn import GNNDecoder, GNNEncoder

    model = GraphKoopmanModel(
        encoder=GNNEncoder(in_channels=1, hidden_channels=4, latent_dim=2),
        decoder=GNNDecoder(latent_dim=2, hidden_channels=4, out_channels=1),
        latent_dim=2,
        time_step=0.1,
    )
    hier = HierarchicalGraphKoopmanModel(model, pool_ratios=(0.5,))
    assert hier.pool_schedule == "per_snapshot"
    assert hier._uses_hetero is False


_TYPED_EDGE_TYPES = (
    ("a", "r0", "b"),
    ("b", "r1", "a"),
    ("a", "r2", "a"),
)


def _typed_snapshot(*, seed: int = 0, n_a: int = 8, n_b: int = 6) -> HeteroData:
    generator = torch.Generator().manual_seed(seed)
    data = HeteroData()
    data["a"].x = torch.randn(n_a, 2, generator=generator)
    data["b"].x = torch.randn(n_b, 3, generator=generator)
    src_a = torch.arange(n_a, dtype=torch.long)
    data["a", "r0", "b"].edge_index = torch.stack(
        [src_a % n_a, src_a % n_b],
        dim=0,
    )
    src_b = torch.arange(n_b, dtype=torch.long)
    data["b", "r1", "a"].edge_index = torch.stack(
        [src_b % n_b, src_b % n_a],
        dim=0,
    )
    data["a", "r2", "a"].edge_index = torch.stack(
        [src_a, (src_a + 1) % n_a],
        dim=0,
    )
    return data


def _typed_sequence(
    *,
    timesteps: int = 4,
    seed: int = 0,
    n_a: int = 8,
    n_b: int = 6,
) -> HeteroGraphSnapshotSequence:
    return HeteroGraphSnapshotSequence(
        [_typed_snapshot(seed=seed + t, n_a=n_a, n_b=n_b) for t in range(timesteps)]
    )


def _typed_model(*, seed: int = 0) -> GraphKoopmanModel:
    torch.manual_seed(seed)
    return GraphKoopmanModel(
        encoder=RelGraphEncoder(
            {"a": 2, "b": 3},
            hidden_channels=8,
            latent_dim=4,
            num_relations=3,
            num_layers=1,
            node_types=("a", "b"),
            edge_types=_TYPED_EDGE_TYPES,
        ),
        decoder=RelGraphDecoder(
            latent_dim=4,
            hidden_channels=8,
            out_channels={"a": 2, "b": 3},
            num_relations=3,
            num_layers=1,
            node_types=("a", "b"),
            edge_types=_TYPED_EDGE_TYPES,
        ),
        latent_dim=4,
        time_step=1.0,
        koopman="hetero_graph",
        koopman_node_types=("a", "b"),
        koopman_edge_types=_TYPED_EDGE_TYPES,
    )


def test_typed_pool_down_preserves_relation_schema() -> None:
    """Coarse typed snapshot keeps all relation banks and shrinks per type."""
    model = _typed_model()
    hier = HierarchicalGraphKoopmanModel(model, pool_ratios=(0.5,))
    fine = _typed_snapshot(seed=0, n_a=8, n_b=6)
    coarse, steps = hier.pool_down(fine)
    assert isinstance(coarse, HeteroData)
    assert coarse["a"].x is not None and coarse["b"].x is not None
    assert coarse["a"].x.size(0) == 4
    assert coarse["b"].x.size(0) == 3
    assert set(coarse.edge_types) == set(_TYPED_EDGE_TYPES)
    assert steps[0].typed_perms is not None
    assert len(steps[0].typed_perms) == 2
    assert steps[0].relation_edge_indices is not None
    assert len(steps[0].relation_edge_indices) == 3


def test_typed_predict_fine_and_coarse() -> None:
    """Typed predict returns HeteroData at fine and coarse resolutions."""
    model = _typed_model()
    hier = HierarchicalGraphKoopmanModel(model, pool_ratios=(0.5,))
    origin = _typed_snapshot(seed=1, n_a=8, n_b=6)
    fine_preds = hier.predict(origin, steps=2, resolution="fine")
    assert len(fine_preds) == 2
    assert isinstance(fine_preds[0], HeteroData)
    assert fine_preds[0]["a"].x.shape == (8, 2)
    assert fine_preds[0]["b"].x.shape == (6, 3)
    assert set(fine_preds[0].edge_types) == set(origin.edge_types)

    coarse_preds = hier.predict(origin, steps=1, resolution="coarse")
    assert isinstance(coarse_preds[0], HeteroData)
    assert coarse_preds[0]["a"].x.shape[0] == 4
    assert coarse_preds[0]["b"].x.shape[0] == 3


def test_typed_fit_smoke() -> None:
    """Short typed hierarchical fit completes with finite loss under hold_perm."""
    torch.manual_seed(0)
    model = _typed_model()
    hier = HierarchicalGraphKoopmanModel(
        model,
        pool_ratios=(0.5,),
        pool_schedule="hold_perm",
    )
    sequence = _typed_sequence(timesteps=5, n_a=8, n_b=6)
    history = hier.fit(sequence, epochs=1, lr=1e-2, unpool_epochs=1)
    assert len(history.loss) == 1
    assert torch.isfinite(torch.tensor(history.loss[0]))


def test_typed_per_snapshot_fit_requires_hold_perm() -> None:
    """``per_snapshot`` typed fit raises toward ``hold_perm``."""
    model = _typed_model()
    hier = HierarchicalGraphKoopmanModel(model, pool_ratios=(0.5,))
    sequence = _typed_sequence(timesteps=4, n_a=8, n_b=6)
    with pytest.raises(ValueError, match="hold_perm"):
        hier.fit(sequence, epochs=1, lr=1e-2, unpool_epochs=0)


def test_typed_checkpoint_preserves_coarse_relation_schema(tmp_path: Path) -> None:
    """Manifest records typed node/edge types for coarse schema."""
    model = _typed_model()
    hier = HierarchicalGraphKoopmanModel(model, pool_ratios=(0.5,))
    out = tmp_path / "hier_typed"
    hier.save(out)
    manifest = json.loads((out / "hierarchical_manifest.json").read_text())
    assert manifest["hetero_mode"] == "typed"
    assert manifest["node_types"] == ["a", "b"]
    assert manifest["edge_types"] == [list(t) for t in _TYPED_EDGE_TYPES]
    loaded = HierarchicalGraphKoopmanModel.load(out)
    assert loaded._uses_typed is True
    assert loaded._node_types == ("a", "b")
    coarse, _ = loaded.pool_down(_typed_snapshot(seed=2, n_a=8, n_b=6))
    assert isinstance(coarse, HeteroData)
    assert set(coarse.edge_types) == set(_TYPED_EDGE_TYPES)


def test_hierarchical_constructor_rejects_invalid_hetero_metadata() -> None:
    """Constructor guards reject inconsistent hetero schemas and channels."""

    def fake_model(
        *,
        node_types: tuple[str, ...],
        typed: bool,
        in_channels: object = 2,
        out_channels: object = 2,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            uses_hetero_koopman=True,
            uses_typed_hetero=typed,
            koopman=SimpleNamespace(node_types=node_types, edge_types=()),
            encoder=SimpleNamespace(in_channels=in_channels),
            decoder=SimpleNamespace(out_channels=out_channels),
        )

    with pytest.raises(ValueError, match="could not infer typed"):
        _channels_by_type(None, role="encoder.in_channels")
    with pytest.raises(ValueError, match="must be positive"):
        _channels_by_type({"a": 0}, role="encoder.in_channels")
    with pytest.raises(TypeError, match="at least two node types"):
        HierarchicalGraphKoopmanModel(
            fake_model(node_types=("a",), typed=True),  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="exactly one node type"):
        HierarchicalGraphKoopmanModel(
            fake_model(node_types=("a", "b"), typed=False),  # type: ignore[arg-type]
        )

    typed = fake_model(
        node_types=("a", "b"),
        typed=True,
        in_channels={"a": 2, "b": 3},
        out_channels={"a": 2, "b": 3},
    )
    with pytest.raises(ValueError, match="do not pass in_channels"):
        HierarchicalGraphKoopmanModel(typed, in_channels=2)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must cover all node types"):
        HierarchicalGraphKoopmanModel(
            fake_model(
                node_types=("a", "b"),
                typed=True,
                in_channels={"a": 2},
                out_channels={"a": 2, "b": 3},
            ),  # type: ignore[arg-type]
        )


def test_pool_down_rejects_mismatched_snapshot_kinds_and_schema() -> None:
    """Pool dispatch and hetero schema validation fail with clear errors."""
    multiplex = HierarchicalGraphKoopmanModel(_multiplex_model(), pool_ratios=(0.5,))
    with pytest.raises(TypeError, match="requires HeteroData"):
        multiplex.pool_down(Data(x=torch.randn(4, 3), edge_index=torch.zeros(2, 0)))

    homo_model = GraphKoopmanModel(
        encoder=GNNEncoder(in_channels=3, hidden_channels=4, latent_dim=2),
        decoder=GNNDecoder(latent_dim=2, hidden_channels=4, out_channels=3),
        latent_dim=2,
        time_step=1.0,
    )
    homo = HierarchicalGraphKoopmanModel(homo_model, pool_ratios=(0.5,))
    with pytest.raises(TypeError, match="cannot pool HeteroData"):
        homo.pool_down(_multiplex_snapshot())

    extra_type = _multiplex_snapshot()
    extra_type["other"].x = torch.randn(2, 3)
    with pytest.raises(ValueError, match="exactly one node type"):
        multiplex.pool_down(extra_type)

    missing_edge = _multiplex_snapshot()
    del missing_edge["node", "r2", "node"]
    with pytest.raises(ValueError, match="missing edge type"):
        multiplex.pool_down(missing_edge)

    typed = HierarchicalGraphKoopmanModel(_typed_model(), pool_ratios=(0.5,))
    wrong_nodes = _typed_snapshot()
    wrong_nodes["c"].x = torch.randn(2, 1)
    with pytest.raises(ValueError, match="requires node types"):
        typed.pool_down(wrong_nodes)

    typed_missing_edge = _typed_snapshot()
    del typed_missing_edge["a", "r0", "b"]
    with pytest.raises(ValueError, match="missing edge type"):
        typed.pool_down(typed_missing_edge)


def test_unpool_rejects_wrong_feature_container_and_metadata() -> None:
    """Unpool guards distinguish tensor, typed mapping, and typed metadata."""
    typed = HierarchicalGraphKoopmanModel(_typed_model(), pool_ratios=(0.5,))
    coarse, steps = typed.pool_down(_typed_snapshot())
    with pytest.raises(TypeError, match="requires a.*mapping"):
        typed.unpool_up(coarse["a"].x, steps)

    multiplex = HierarchicalGraphKoopmanModel(_multiplex_model(), pool_ratios=(0.5,))
    multiplex_coarse, multiplex_steps = multiplex.pool_down(_multiplex_snapshot())
    with pytest.raises(TypeError, match="expects a feature Tensor"):
        multiplex.unpool_up({"node": multiplex_coarse["node"].x}, multiplex_steps)

    malformed = PoolStep(
        perm=steps[0].perm,
        num_fine=steps[0].num_fine,
        edge_index=steps[0].edge_index,
        edge_weight=None,
    )
    with pytest.raises(ValueError, match="typed perm fields"):
        typed.unpool_up(
            {"a": coarse["a"].x, "b": coarse["b"].x},
            [malformed],
        )


def test_hetero_predict_future_history_and_intermediate_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hetero prediction pools optional context and intermediate outputs."""
    multiplex = HierarchicalGraphKoopmanModel(
        _multiplex_model(), pool_ratios=(0.75, 0.5)
    )
    monkeypatch.setattr(
        multiplex.model,
        "predict",
        lambda initial, steps, **kwargs: [initial for _ in range(steps)],
    )
    origin = _multiplex_snapshot(seed=30)
    preds = multiplex.predict(
        origin,
        steps=1,
        future_topologies=[_multiplex_snapshot(seed=31)],
        history=[_multiplex_snapshot(seed=29)],
        resolution=1,
    )
    assert preds[0]["node"].x.shape[0] == 6
    with pytest.raises(TypeError, match="HeteroData origin"):
        multiplex.predict(torch.randn(8, 3), steps=1)
    with pytest.raises(TypeError, match="future_topologies"):
        multiplex.predict(origin, steps=1, future_topologies=[Data()])
    with pytest.raises(TypeError, match="history"):
        multiplex.predict(origin, steps=1, history=[Data()])

    typed = HierarchicalGraphKoopmanModel(_typed_model(), pool_ratios=(0.75, 0.5))
    monkeypatch.setattr(
        typed.model,
        "predict",
        lambda initial, steps, **kwargs: [initial for _ in range(steps)],
    )
    typed_origin = _typed_snapshot(seed=40)
    typed_preds = typed.predict(
        typed_origin,
        steps=1,
        future_topologies=[_typed_snapshot(seed=41)],
        history=[_typed_snapshot(seed=39)],
        resolution=1,
    )
    assert typed_preds[0]["a"].x.shape[0] == 6
    assert typed_preds[0]["b"].x.shape[0] == 5
    with pytest.raises(TypeError, match="HeteroData origin"):
        typed.predict(torch.randn(8, 2), steps=1)
    with pytest.raises(TypeError, match="future_topologies"):
        typed.predict(typed_origin, steps=1, future_topologies=[Data()])
    with pytest.raises(TypeError, match="history"):
        typed.predict(typed_origin, steps=1, history=[Data()])


def test_sequence_and_feature_validation_paths() -> None:
    """Sequence pooling and feature extraction reject incompatible inputs."""
    multiplex = HierarchicalGraphKoopmanModel(_multiplex_model(), pool_ratios=(0.5,))
    homo_sequence = GraphSnapshotSequence(
        [Data(x=torch.randn(4, 3), edge_index=torch.zeros(2, 0, dtype=torch.long))]
    )
    with pytest.raises(TypeError, match="HeteroGraphSnapshotSequence"):
        multiplex._pool_sequence(homo_sequence)

    homo_model = GraphKoopmanModel(
        encoder=GNNEncoder(in_channels=3, hidden_channels=4, latent_dim=2),
        decoder=GNNDecoder(latent_dim=2, hidden_channels=4, out_channels=3),
        latent_dim=2,
        time_step=1.0,
    )
    homo = HierarchicalGraphKoopmanModel(homo_model, pool_ratios=(0.5,))
    hetero_sequence = _multiplex_sequence(timesteps=2)
    with pytest.raises(TypeError, match="cannot use HeteroGraphSnapshotSequence"):
        homo._pool_sequence(hetero_sequence)

    typed = HierarchicalGraphKoopmanModel(_typed_model(), pool_ratios=(0.5,))
    with pytest.raises(TypeError, match="requires HeteroData snapshots"):
        typed._typed_snapshot_features(Data())


def test_pooling_helper_validation_and_weight_paths() -> None:
    """Pooling utilities retain weights and reject incomplete metadata."""
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_weight = torch.tensor([0.25, 0.75])
    features = torch.randn(2, 2)

    class MultiEdgePool(nn.Module):
        ratio = 0.5

        def forward(
            self,
            x: torch.Tensor,
            edges: torch.Tensor,
            edge_attr: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, torch.Tensor, None]:
            del edge_attr
            attrs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
            return x, edges, attrs, None, torch.arange(x.size(0)), None

    _, _, trimmed, _ = apply_pool_layer(
        MultiEdgePool(),  # type: ignore[arg-type]
        features,
        edge_index,
    )
    assert trimmed is not None and trimmed.shape == (2,)

    with pytest.raises(ValueError, match="expected 1 relation banks"):
        multiplex_snapshot_from_features(
            features,
            node_type="node",
            edge_types=(("node", "r", "node"),),
            relation_edge_indices=(),
        )
    weighted_multiplex = multiplex_snapshot_from_features(
        features,
        node_type="node",
        edge_types=(("node", "r", "node"),),
        relation_edge_indices=(edge_index,),
        relation_edge_weights=(edge_weight,),
    )
    assert torch.equal(weighted_multiplex["node", "r", "node"].edge_weight, edge_weight)

    with pytest.raises(ValueError, match="steps must contain"):
        pool_features_with_steps(features, [])
    with pytest.raises(ValueError, match="steps must contain"):
        pool_multiplex_features_with_steps(
            features,
            [],
            node_type="node",
            edge_types=(("node", "r", "node"),),
        )
    plain_step = PoolStep(
        perm=torch.tensor([0]),
        num_fine=2,
        edge_index=torch.zeros(2, 0, dtype=torch.long),
        edge_weight=None,
    )
    with pytest.raises(ValueError, match="requires relation banks"):
        pool_multiplex_features_with_steps(
            features,
            [plain_step],
            node_type="node",
            edge_types=(("node", "r", "node"),),
        )

    with pytest.raises(ValueError, match="expected 1 relation banks"):
        typed_snapshot_from_features(
            {"a": features},
            node_types=("a",),
            edge_types=(("a", "r", "a"),),
            relation_edge_indices=(),
        )
    weighted_typed = typed_snapshot_from_features(
        {"a": features},
        node_types=("a",),
        edge_types=(("a", "r", "a"),),
        relation_edge_indices=(edge_index,),
        relation_edge_weights=(edge_weight,),
    )
    assert torch.equal(weighted_typed["a", "r", "a"].edge_weight, edge_weight)

    with pytest.raises(ValueError, match="steps must contain"):
        pool_typed_features_with_steps(
            {"a": features},
            [],
            node_types=("a",),
            edge_types=(("a", "r", "a"),),
        )
    with pytest.raises(ValueError, match="typed perm fields"):
        pool_typed_features_with_steps(
            {"a": features},
            [plain_step],
            node_types=("a",),
            edge_types=(("a", "r", "a"),),
        )
    typed_step = PoolStep(
        perm=torch.tensor([0]),
        num_fine=2,
        edge_index=torch.zeros(2, 0, dtype=torch.long),
        edge_weight=None,
        typed_node_types=("a",),
        typed_perms=(torch.tensor([0]),),
        typed_num_fine=(2,),
    )
    with pytest.raises(ValueError, match="requires relation banks"):
        pool_typed_features_with_steps(
            {"a": features},
            [typed_step],
            node_types=("a",),
            edge_types=(("a", "r", "a"),),
        )

    empty = torch.zeros(2, 0, dtype=torch.long)
    induced_edges, induced_weights = induce_typed_relation_banks(
        (("a", "empty", "a"), ("a", "r", "a")),
        (empty, edge_index),
        (None, edge_weight),
        perms_by_type={"a": torch.tensor([0, 1])},
        num_fine_by_type={"a": 2},
    )
    assert induced_edges[0].numel() == 0
    assert torch.equal(induced_weights[1], edge_weight)

    resolved_features, resolved_edges, resolved_weights = resolve_snapshot_inputs(
        features, edge_index, edge_weight
    )
    assert resolved_features is features
    assert resolved_edges is edge_index
    assert resolved_weights is edge_weight


def test_homogeneous_pool_predict_and_sequence_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Homogeneous paths cover pooling, forecasting, and held permutations."""
    model = GraphKoopmanModel(
        encoder=GNNEncoder(in_channels=1, hidden_channels=4, latent_dim=2),
        decoder=GNNDecoder(latent_dim=2, hidden_channels=4, out_channels=1),
        latent_dim=2,
        time_step=0.1,
        control_dim=1,
    )
    hier = HierarchicalGraphKoopmanModel(
        model,
        pool_ratios=(0.75, 0.5),
        pool_schedule="hold_perm",
    )
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7, 0]],
        dtype=torch.long,
    )
    snapshots = [Data(x=torch.randn(8, 1), edge_index=edge_index) for _ in range(3)]
    sequence = GraphSnapshotSequence(
        snapshots,
        control_inputs=torch.randn(3, 8, 1),
        timestamps=torch.arange(3, dtype=torch.float32),
    )
    coarse, pool_steps = hier.pool_down(snapshots[0].x, edge_index)
    assert coarse.x.shape[0] == 3
    assert hier.n_levels == 2
    assert hier.control_dim == 1
    assert hier._pool_controls(None, pool_steps) is None
    assert hier._pool_controls([torch.randn(8, 1)], pool_steps)[0].shape == (3, 1)

    monkeypatch.setattr(
        hier.model,
        "predict",
        lambda initial, steps, **kwargs: [initial for _ in range(steps)],
    )
    intermediate = hier.predict(
        snapshots[0],
        steps=1,
        controls=[torch.randn(8, 1)],
        future_topologies=[snapshots[1]],
        history=[snapshots[0]],
        resolution=1,
    )
    assert intermediate[0].x.shape[0] == 6
    assert hier.predict(snapshots[0], steps=1, resolution="fine")[0].x.shape[0] == 8
    assert hier.predict(snapshots[0], steps=1, resolution="coarse")[0].x.shape[0] == 3
    with pytest.raises(ValueError, match="resolution int"):
        hier.predict(snapshots[0], steps=1, resolution=3)
    with pytest.raises(ValueError, match="resolution must be"):
        hier.predict(snapshots[0], steps=1, resolution="middle")  # type: ignore[arg-type]

    coarse_sequence, all_steps = hier._pool_sequence(sequence)
    assert coarse_sequence.timestamps is not None
    assert coarse_sequence.control_inputs.shape == (3, 3, 1)
    assert all_steps[0] is all_steps[-1]

    per_snapshot = HierarchicalGraphKoopmanModel(
        GraphKoopmanModel(
            encoder=GNNEncoder(in_channels=1, hidden_channels=4, latent_dim=2),
            decoder=GNNDecoder(latent_dim=2, hidden_channels=4, out_channels=1),
            latent_dim=2,
            time_step=0.1,
        ),
        pool_ratios=(0.5,),
    )
    dynamic, dynamic_steps = per_snapshot._pool_sequence(
        GraphSnapshotSequence(snapshots)
    )
    assert dynamic.is_dynamic_topology
    assert len(dynamic_steps) == 3


def test_model_validation_and_unpool_training_guards() -> None:
    """Model validation covers resolution, sequence, and malformed fit metadata."""
    homo = HierarchicalGraphKoopmanModel(
        GraphKoopmanModel(
            encoder=GNNEncoder(in_channels=1, hidden_channels=4, latent_dim=2),
            decoder=GNNDecoder(latent_dim=2, hidden_channels=4, out_channels=1),
            latent_dim=2,
            time_step=0.1,
        ),
        pool_ratios=(0.5,),
    )
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    fine = Data(x=torch.randn(2, 1), edge_index=edge_index)
    coarse, steps = homo.pool_down(fine)
    with pytest.raises(ValueError, match="expected 1 pool steps"):
        homo.unpool_up(coarse.x, [])
    with pytest.raises(ValueError, match="levels must be"):
        homo.unpool_up(coarse.x, steps, levels=2)
    with pytest.raises(ValueError, match="snapshot.x"):
        homo._snapshot_features(Data(edge_index=edge_index))

    homo.unpool_layers = nn.ModuleList([nn.Identity()])
    homo._fit_unpool(
        GraphSnapshotSequence([fine]),
        [steps],
        epochs=1,
        lr=1e-2,
    )

    typed = HierarchicalGraphKoopmanModel(_typed_model(), pool_ratios=(0.5,))
    typed_sequence = _typed_sequence(timesteps=1)
    typed_coarse, typed_steps = typed.pool_down(typed_sequence[0])
    malformed = PoolStep(
        perm=typed_steps[0].perm,
        num_fine=typed_steps[0].num_fine,
        edge_index=typed_steps[0].edge_index,
        edge_weight=None,
        relation_edge_indices=typed_steps[0].relation_edge_indices,
    )
    assert typed_coarse["a"].x is not None
    with pytest.raises(ValueError, match="typed unpool fit requires"):
        typed._fit_unpool(typed_sequence, [[malformed]], epochs=1, lr=1e-2)


def test_general_pooling_helper_paths() -> None:
    """General pooling helpers cover identity, controls, snapshots, and errors."""
    with pytest.raises(ValueError, match="pool ratio"):
        build_pool_layer(2, 0.0)
    assert build_pool_layer(2, 0.5, pooling="sag") is not None
    with pytest.raises(ValueError, match="pooling must be"):
        build_pool_layer(2, 0.5, pooling="invalid")  # type: ignore[arg-type]

    features = torch.randn(3, 2)
    edges = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    weights = torch.ones(3)
    identity = build_pool_layer(2, 1.0)
    pooled_x, pooled_edges, pooled_weights, perm = apply_pool_layer(
        identity, features, edges, weights
    )
    assert torch.equal(pooled_x, features)
    assert torch.equal(pooled_edges, edges)
    assert torch.equal(pooled_weights, weights)
    assert torch.equal(perm, torch.arange(3))

    assert torch.equal(pool_control(torch.tensor([1.0]), [perm]), torch.tensor([1.0]))
    per_node = pool_control(features, [torch.tensor([0, 2])])
    assert torch.equal(per_node, features[[0, 2]])
    with pytest.raises(ValueError, match="control must have shape"):
        pool_control(torch.zeros(1, 1, 1), [])

    global_controls = torch.randn(2, 1)
    assert pool_control_sequence(global_controls, []) is global_controls
    with pytest.raises(ValueError, match="control_inputs must have shape"):
        pool_control_sequence(torch.zeros(1, 1, 1, 1), [])
    with pytest.raises(ValueError, match="expected 2 perm chains"):
        pool_control_sequence(torch.randn(2, 3, 1), [[]])
    pooled_controls = pool_control_sequence(
        torch.randn(2, 3, 1),
        [[torch.tensor([0, 2])], [torch.tensor([1, 2])]],
    )
    assert pooled_controls.shape == (2, 2, 1)

    weighted = snapshot_from_features(features, edges, weights)
    assert weighted.edge_weight is not None
    resolved_x, resolved_edges, resolved_weight = resolve_snapshot_inputs(weighted)
    assert resolved_x is weighted.x
    assert resolved_edges is weighted.edge_index
    assert resolved_weight is weighted.edge_weight
    with pytest.raises(ValueError, match="node features"):
        resolve_snapshot_inputs(Data(edge_index=edges))
    with pytest.raises(ValueError, match="edge_index is required"):
        resolve_snapshot_inputs(features)

    mask = torch.tensor([True, True, False])
    filtered_edges, filtered_weights = filter_subgraph(edges, mask, weights)
    assert filtered_edges.shape[0] == 2
    assert filtered_weights is not None
