"""Tests for hierarchical / multi-resolution GraphKoopman forecasting."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from koopman_graph import (
    GNNDecoder,
    GNNEncoder,
    GraphKoopmanModel,
    GraphSnapshotSequence,
)
from koopman_graph.datasets import (
    GridDynamicGraphBenchmark,
    SyntheticDynamicGraphBenchmark,
)
from koopman_graph.hierarchical import (
    HierarchicalGraphKoopmanModel,
    ScatterUnpool,
    build_pool_layer,
    pool_control,
)


def _tiny_model(
    *,
    koopman: str | None = None,
    control_dim: int = 0,
) -> GraphKoopmanModel:
    """Build a small composed model for hierarchical tests."""
    encoder = GNNEncoder(in_channels=1, hidden_channels=4, latent_dim=2)
    decoder = GNNDecoder(latent_dim=2, hidden_channels=4, out_channels=1)
    kwargs: dict[str, object] = {
        "encoder": encoder,
        "decoder": decoder,
        "latent_dim": 2,
        "time_step": 0.1,
        "control_dim": control_dim,
    }
    if koopman is not None:
        kwargs["koopman"] = koopman
    return GraphKoopmanModel(**kwargs)  # type: ignore[arg-type]


def test_build_pool_layer_rejects_bad_ratio() -> None:
    """Invalid pool ratios raise ``ValueError``."""
    with pytest.raises(ValueError, match="pool ratio"):
        build_pool_layer(3, 0.0)
    with pytest.raises(ValueError, match="pooling must be"):
        build_pool_layer(3, 0.5, pooling="diff")  # type: ignore[arg-type]


def test_pool_unpool_shape_round_trip() -> None:
    """Pool then unpool restores fine node count."""
    model = _tiny_model()
    hier = HierarchicalGraphKoopmanModel(model, pool_ratios=(0.5,))
    sequence = SyntheticDynamicGraphBenchmark.generate(
        num_nodes=8,
        num_timesteps=4,
        in_channels=1,
        noise_std=0.0,
        seed=0,
    )
    coarse, steps = hier.pool_down(sequence[0])
    assert coarse.x is not None
    assert coarse.x.size(0) == 4
    restored = hier.unpool_up(coarse.x, steps)
    assert restored.shape == sequence[0].x.shape


def test_noop_ratio_preserves_node_count_and_features() -> None:
    """``pool_ratios=(1.0,)`` keeps all nodes; identity unpool recovers ``x``."""
    model = _tiny_model()
    hier = HierarchicalGraphKoopmanModel(
        model, pool_ratios=(1.0,), refine_unpool=True
    )
    sequence = SyntheticDynamicGraphBenchmark.generate(
        num_nodes=6,
        num_timesteps=3,
        in_channels=1,
        noise_std=0.0,
        seed=1,
    )
    fine = sequence[0]
    assert fine.x is not None
    coarse, steps = hier.pool_down(fine)
    assert coarse.x is not None
    assert coarse.x.size(0) == fine.x.size(0)
    # Teacher scatter path used by unpool training.
    permuted = fine.x[steps[0].perm]
    restored = hier.unpool_up(permuted, steps)
    assert torch.allclose(restored, fine.x, atol=1e-5)


def test_predict_resolution_shapes() -> None:
    """``resolution`` switches between coarse and fine forecast shapes."""
    model = _tiny_model()
    hier = HierarchicalGraphKoopmanModel(model, pool_ratios=(0.5,))
    sequence = SyntheticDynamicGraphBenchmark.generate(
        num_nodes=8,
        num_timesteps=6,
        in_channels=1,
        noise_std=0.0,
        seed=2,
    )
    hier.fit(sequence, epochs=2, lr=1e-2, unpool_epochs=2)

    fine_preds = hier.predict(sequence[0], steps=2, resolution="fine")
    coarse_preds = hier.predict(sequence[0], steps=2, resolution="coarse")
    assert fine_preds[0].x.shape[0] == 8
    assert coarse_preds[0].x.shape[0] == 4
    mid = hier.predict(sequence[0], steps=1, resolution=0)
    assert mid[0].x.shape[0] == 4


def test_sag_pooling_smoke() -> None:
    """Optional ``pooling='sag'`` constructs and pools."""
    model = _tiny_model()
    hier = HierarchicalGraphKoopmanModel(
        model, pool_ratios=(0.5,), pooling="sag"
    )
    sequence = SyntheticDynamicGraphBenchmark.generate(
        num_nodes=8,
        num_timesteps=3,
        in_channels=1,
        noise_std=0.0,
        seed=3,
    )
    coarse, steps = hier.pool_down(sequence[0])
    assert coarse.x is not None
    assert coarse.x.size(0) == 4
    assert len(steps) == 1


def test_graph_spectrum_uses_pooled_topology() -> None:
    """Graph-operator spectrum requires pooled topology (not bare ``K_self``)."""
    model = _tiny_model(koopman="graph")
    hier = HierarchicalGraphKoopmanModel(model, pool_ratios=(0.5,))
    sequence = SyntheticDynamicGraphBenchmark.generate(
        num_nodes=6,
        num_timesteps=4,
        in_channels=1,
        noise_std=0.0,
        seed=4,
    )
    with pytest.raises(ValueError, match="edge_index and num_nodes"):
        model.spectrum()

    spectrum = hier.spectrum(sequence[0])
    # Effective size is N_coarse * latent_dim.
    assert spectrum.eigenvalues.numel() == 3 * 2


def test_global_and_per_node_controls() -> None:
    """Global controls pass through; per-node controls follow ``perm``."""
    perm = torch.tensor([0, 2, 4])
    global_u = torch.tensor([0.5, -0.25])
    assert torch.equal(pool_control(global_u, [perm]), global_u)

    per_node = torch.arange(6, dtype=torch.float32).view(6, 1)
    pooled = pool_control(per_node, [perm])
    assert pooled.shape == (3, 1)
    assert torch.equal(pooled, per_node[perm])

    model = _tiny_model(control_dim=1)
    hier = HierarchicalGraphKoopmanModel(model, pool_ratios=(0.5,))
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4, 4, 5], [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]],
        dtype=torch.long,
    )
    snaps = [
        Data(x=torch.randn(6, 1), edge_index=edge_index) for _ in range(5)
    ]
    # Global controls (T, C).
    sequence = GraphSnapshotSequence(
        snaps,
        control_inputs=torch.randn(5, 1),
    )
    hier.fit(sequence, epochs=2, lr=1e-2, unpool_epochs=1)
    preds = hier.predict(
        sequence[0],
        steps=2,
        controls=[torch.tensor([0.1]), torch.tensor([-0.2])],
    )
    assert len(preds) == 2
    assert preds[0].x.shape[0] == 6

    # Per-node controls (T, N, C).
    per_seq = GraphSnapshotSequence(
        snaps,
        control_inputs=torch.randn(5, 6, 1),
    )
    hier2 = HierarchicalGraphKoopmanModel(
        _tiny_model(control_dim=1),
        pool_ratios=(0.5,),
    )
    hier2.fit(per_seq, epochs=2, lr=1e-2, unpool_epochs=1)
    per_controls = [torch.randn(6, 1), torch.randn(6, 1)]
    preds2 = hier2.predict(per_seq[0], steps=2, controls=per_controls)
    assert preds2[0].x.shape[0] == 6


def test_save_load_round_trip(tmp_path: Path) -> None:
    """Wrapper + format-1 inner model round-trip through ``save`` / ``load``."""
    model = _tiny_model()
    hier = HierarchicalGraphKoopmanModel(model, pool_ratios=(0.5, 0.5))
    sequence = SyntheticDynamicGraphBenchmark.generate(
        num_nodes=8,
        num_timesteps=5,
        in_channels=1,
        noise_std=0.0,
        seed=5,
    )
    hier.fit(sequence, epochs=2, lr=1e-2, unpool_epochs=1)
    original = hier.predict(sequence[0], steps=2)

    out = tmp_path / "hier"
    hier.save(out)
    loaded = HierarchicalGraphKoopmanModel.load(out)
    restored = loaded.predict(sequence[0], steps=2)
    for a, b in zip(original, restored, strict=True):
        assert torch.allclose(a.x, b.x, atol=1e-5)


def test_scatter_unpool_identity_init() -> None:
    """Identity-initialized refine leaves scattered features unchanged."""
    unpool = ScatterUnpool(2, refine=True)
    coarse = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    perm = torch.tensor([1, 3])
    out = unpool(coarse, perm, num_fine=4)
    expected = torch.zeros(4, 2)
    expected[1] = coarse[0]
    expected[3] = coarse[1]
    assert torch.allclose(out, expected)


def test_grid_fit_predict_smoke() -> None:
    """Smoke fit/predict on a small grid (notebook-scale stand-in)."""
    sequence = GridDynamicGraphBenchmark.generate(
        num_rows=4,
        num_cols=4,
        num_timesteps=12,
        in_channels=1,
        seed=0,
    )
    model = GraphKoopmanModel(
        encoder=GNNEncoder(in_channels=1, hidden_channels=8, latent_dim=4),
        decoder=GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=1),
        latent_dim=4,
        time_step=0.1,
    )
    hier = HierarchicalGraphKoopmanModel(model, pool_ratios=(0.5,))
    hier.fit(sequence, epochs=3, lr=1e-2, unpool_epochs=2)
    preds = hier.predict(sequence[0], steps=3)
    assert len(preds) == 3
    assert preds[0].x.shape == sequence[0].x.shape
