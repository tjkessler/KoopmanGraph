"""Multiplex hetero tests for ``HierarchicalGraphKoopmanModel``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch_geometric.data import HeteroData

from koopman_graph.data import HeteroGraphSnapshotSequence
from koopman_graph.hierarchical import HierarchicalGraphKoopmanModel
from koopman_graph.model import GraphKoopmanModel
from koopman_graph.nn import RelGraphDecoder, RelGraphEncoder


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
