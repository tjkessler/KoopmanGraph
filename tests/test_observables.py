"""Tests for physics-informed hybrid observables."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
from koopman_graph.data import GraphSnapshotSequence, temporal_split
from koopman_graph.datasets import AnisotropicAdvectionGridBenchmark
from koopman_graph.observables import (
    PHYSICS_PRESETS,
    concatenate_observables,
    graph_curvature_features,
    graph_gradient_features,
    graph_laplacian_features,
    make_polynomial_features,
    polynomial_features,
    resolve_physics_lifting_fn,
    resolve_physics_position,
)
from koopman_graph.serialization import (
    build_model_config,
    load_checkpoint,
    reconstruct_model,
)


@pytest.fixture
def small_snapshot() -> Data:
    """Provide a tiny weighted graph snapshot."""
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    edge_weight = torch.tensor([1.0, 1.0, 2.0, 2.0])
    x = torch.tensor(
        [
            [1.0, 0.0, 0.5],
            [0.0, 1.0, 0.25],
            [0.5, 0.5, 1.0],
        ]
    )
    return Data(x=x, edge_index=edge_index, edge_weight=edge_weight)


def _hybrid_model(
    *,
    in_channels: int = 3,
    gnn_latent_dim: int = 8,
    physics_dim: int = 3,
    use_preset: bool = True,
) -> GraphKoopmanModel:
    """Build a hybrid GraphKoopmanModel for tests."""
    encoder = GNNEncoder(
        in_channels=in_channels,
        hidden_channels=16,
        latent_dim=gnn_latent_dim,
    )
    total_latent_dim = gnn_latent_dim + physics_dim
    decoder = GNNDecoder(
        latent_dim=total_latent_dim,
        hidden_channels=16,
        out_channels=in_channels,
    )
    kwargs: dict[str, object] = {
        "encoder": encoder,
        "decoder": decoder,
        "latent_dim": total_latent_dim,
        "time_step": 0.1,
        "physics_dim": physics_dim,
    }
    if use_preset:
        kwargs["physics_preset"] = "graph_laplacian"
    else:
        kwargs["physics_lifting_fn"] = graph_laplacian_features
    return GraphKoopmanModel(**kwargs)


def test_graph_laplacian_features_shape(small_snapshot: Data) -> None:
    """Laplacian features should match node and channel counts."""
    physics = graph_laplacian_features(small_snapshot)
    assert physics.shape == (3, 3)


def test_graph_gradient_and_curvature_presets_preserve_shape_and_isolates() -> None:
    """Graph derivative presets preserve channels and map isolates to zero."""
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    snapshot = Data(
        x=torch.tensor([[0.0], [2.0], [7.0]]),
        edge_index=edge_index,
    )

    gradient = graph_gradient_features(snapshot)
    curvature = graph_curvature_features(snapshot)

    assert gradient.shape == snapshot.x.shape
    assert curvature.shape == snapshot.x.shape
    assert torch.all(gradient >= 0)
    assert gradient[2].item() == pytest.approx(0.0)
    assert curvature[2].item() == pytest.approx(0.0)


def test_polynomial_features_factory_and_dynamic_preset(small_snapshot: Data) -> None:
    """Polynomial lifting concatenates powers and resolves dynamic preset names."""
    direct = polynomial_features(small_snapshot, degree=3)
    factory = make_polynomial_features(3)
    resolved = resolve_physics_lifting_fn(physics_preset="polynomial(3)")

    assert direct.shape == (3, 9)
    assert torch.equal(direct[:, :3], small_snapshot.x)
    assert torch.equal(direct[:, 3:6], small_snapshot.x.square())
    assert torch.equal(factory(small_snapshot), direct)
    assert resolved is not None
    assert torch.equal(resolved(small_snapshot), direct)


def test_polynomial_features_rejects_invalid_degree(small_snapshot: Data) -> None:
    """Polynomial lifting requires a positive integer degree."""
    with pytest.raises(ValueError, match="degree"):
        polynomial_features(small_snapshot, degree=0)
    with pytest.raises(ValueError, match="physics_preset"):
        resolve_physics_lifting_fn(physics_preset="polynomial(0)")


def test_graph_laplacian_features_matches_dense_normalized_laplacian(
    small_snapshot: Data,
) -> None:
    """Physics lifting should equal ``(P - Â) x`` (``I - Â`` without isolates)."""
    from koopman_graph.graph_utils import dense_symmetric_normalized_laplacian

    physics = graph_laplacian_features(small_snapshot)
    laplacian = dense_symmetric_normalized_laplacian(
        small_snapshot.edge_index,
        num_nodes=small_snapshot.x.shape[0],
        edge_weight=small_snapshot.edge_weight,
        dtype=small_snapshot.x.dtype,
    )
    expected = laplacian @ small_snapshot.x
    assert torch.allclose(physics, expected, atol=1e-6)


def test_graph_laplacian_features_zeros_isolated_nodes() -> None:
    """Isolated nodes must map to zeros under L_sym (not identity passthrough)."""
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    data = Data(
        x=torch.tensor([[1.0, 0.5], [0.0, 1.0], [2.0, -1.0]]),
        edge_index=edge_index,
    )
    physics = graph_laplacian_features(data)
    assert torch.allclose(physics[2], torch.zeros(2))
    assert not torch.allclose(physics[0], torch.zeros(2))


def test_concatenate_observables_prepends_physics() -> None:
    """Physics features should be prepended to GNN latents."""
    physics = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    gnn = torch.tensor([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]])
    combined = concatenate_observables(physics, gnn)
    assert combined.shape == (2, 5)
    assert torch.allclose(combined[:, :2], physics)
    assert torch.allclose(combined[:, 2:], gnn)


def test_resolve_physics_lifting_fn_prefers_custom_callable() -> None:
    """Custom lifting functions should override presets."""

    def custom_fn(data: Data) -> torch.Tensor:
        return torch.zeros(data.num_nodes, 2)

    resolved = resolve_physics_lifting_fn(
        physics_preset="graph_laplacian",
        physics_lifting_fn=custom_fn,
    )
    assert resolved is custom_fn


def test_constructor_requires_explicit_physics_dim() -> None:
    """Physics lifting without physics_dim should fail."""
    encoder = GNNEncoder(in_channels=3, hidden_channels=8, latent_dim=4)
    decoder = GNNDecoder(latent_dim=7, hidden_channels=8, out_channels=3)
    with pytest.raises(ValueError, match="physics_dim must be positive"):
        GraphKoopmanModel(
            encoder=encoder,
            decoder=decoder,
            latent_dim=7,
            time_step=0.1,
            physics_preset="graph_laplacian",
            physics_dim=0,
        )


def test_constructor_validates_total_latent_dim() -> None:
    """latent_dim must equal encoder.latent_dim + physics_dim."""
    encoder = GNNEncoder(in_channels=3, hidden_channels=8, latent_dim=4)
    decoder = GNNDecoder(latent_dim=8, hidden_channels=8, out_channels=3)
    with pytest.raises(ValueError, match="latent_dim"):
        GraphKoopmanModel(
            encoder=encoder,
            decoder=decoder,
            latent_dim=8,
            time_step=0.1,
            physics_preset="graph_laplacian",
            physics_dim=3,
        )


def test_encode_concatenates_physics_and_gnn(small_snapshot: Data) -> None:
    """Hybrid encode should return prepended physics and GNN features."""
    model = _hybrid_model()
    latent = model.encode(small_snapshot)
    assert latent.shape == (small_snapshot.num_nodes, 11)

    gnn_only = model.encoder(
        small_snapshot.x,
        small_snapshot.edge_index,
        small_snapshot.edge_weight,
    )
    physics = graph_laplacian_features(small_snapshot)
    expected = concatenate_observables(physics, gnn_only)
    assert torch.allclose(latent, expected)


def test_encode_removed_encode_latent_alias(small_snapshot: Data) -> None:
    """encode_latent was removed; encode remains the only lifting API."""
    model = _hybrid_model()
    assert not hasattr(model, "encode_latent")
    assert model.encode(small_snapshot).ndim == 2


def test_encode_supports_gradients(small_snapshot: Data) -> None:
    """Gradients should flow through GNN parameters in hybrid mode."""
    model = _hybrid_model()
    small_snapshot.x = small_snapshot.x.clone().requires_grad_(True)
    latent = model.encode(small_snapshot)
    loss = latent.sum()
    loss.backward()
    assert small_snapshot.x.grad is not None
    assert any(param.grad is not None for param in model.encoder.parameters())


def test_forward_and_predict_use_hybrid_latent(
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Forward and predict should operate on the hybrid latent size."""
    in_channels = scaling_sequence[0].x.size(-1)
    model = _hybrid_model(in_channels=in_channels, physics_dim=in_channels)
    output = model.forward(scaling_sequence[0])
    assert output.shape == scaling_sequence[0].x.shape

    predictions = model.predict(scaling_sequence[0], steps=2)
    assert len(predictions) == 2
    assert predictions[0].x.shape == scaling_sequence[0].x.shape


def test_build_model_config_stores_physics_metadata() -> None:
    """Checkpoint config should record hybrid observable settings."""
    model = _hybrid_model()
    config = build_model_config(model)
    assert config["physics"] == {
        "dim": 3,
        "preset": "graph_laplacian",
        "position": "prepend",
    }


def test_resolve_physics_position_defaults_and_rejects_unknown() -> None:
    """Position resolver should default to prepend and reject unsupported values."""
    assert resolve_physics_position(None) == "prepend"
    assert resolve_physics_position("prepend") == "prepend"
    with pytest.raises(ValueError, match="Unsupported physics position"):
        resolve_physics_position("append")


def test_reconstruct_model_round_trips_physics_position() -> None:
    """reconstruct_model should restore physics.position from config."""
    model = _hybrid_model()
    config = build_model_config(model)
    restored = reconstruct_model(config)
    assert restored.physics_position == "prepend"
    assert restored.physics_dim == model.physics_dim
    assert restored.physics_preset == model.physics_preset


def test_reconstruct_model_rejects_unsupported_physics_position() -> None:
    """Invalid physics.position in a hybrid checkpoint should raise on load."""
    model = _hybrid_model()
    config = build_model_config(model)
    assert config["physics"] is not None
    config["physics"]["position"] = "append"
    with pytest.raises(ValueError, match="Unsupported physics position"):
        reconstruct_model(config)


def test_reconstruct_model_defaults_missing_physics_position() -> None:
    """Older hybrid configs without position should default to prepend."""
    model = _hybrid_model()
    config = build_model_config(model)
    assert config["physics"] is not None
    del config["physics"]["position"]
    restored = reconstruct_model(config)
    assert restored.physics_position == "prepend"


def test_checkpoint_round_trip_with_preset(
    scaling_sequence: GraphSnapshotSequence,
    tmp_path: Path,
) -> None:
    """Preset-based hybrid models should round-trip through save/load."""
    model = _hybrid_model(in_channels=scaling_sequence[0].x.size(-1))
    torch.manual_seed(0)
    model.fit(scaling_sequence, epochs=2, lr=1e-2)

    checkpoint = tmp_path / "hybrid.pt"
    model.save(checkpoint)
    loaded = GraphKoopmanModel.load(checkpoint)

    assert loaded.physics_dim == 3
    assert loaded.physics_preset == "graph_laplacian"
    assert loaded.physics_position == "prepend"
    assert loaded.latent_dim == model.latent_dim

    initial = scaling_sequence[0]
    original = [graph.x.detach().clone() for graph in model.predict(initial, steps=2)]
    restored = [graph.x.detach().clone() for graph in loaded.predict(initial, steps=2)]
    for left, right in zip(original, restored, strict=True):
        assert torch.allclose(left, right)


def test_load_requires_custom_physics_fn_without_preset(
    scaling_sequence: GraphSnapshotSequence,
    tmp_path: Path,
) -> None:
    """Custom hybrid checkpoints must re-supply physics_lifting_fn on load."""
    model = _hybrid_model(
        in_channels=scaling_sequence[0].x.size(-1),
        use_preset=False,
    )
    checkpoint = tmp_path / "custom_hybrid.pt"
    model.save(checkpoint)

    with pytest.raises(ValueError, match="physics_lifting_fn"):
        load_checkpoint(checkpoint)

    loaded = load_checkpoint(
        checkpoint,
        physics_lifting_fn=graph_laplacian_features,
    )
    assert loaded.physics_dim == 3
    assert loaded.physics_preset is None


def test_physics_ablation_improves_advection_recovery() -> None:
    """Hybrid Laplacian features should beat a pure GNN on advection dynamics."""
    torch.manual_seed(7)
    sequence = AnisotropicAdvectionGridBenchmark.generate(
        num_rows=6,
        num_cols=6,
        num_timesteps=30,
        in_channels=3,
        seed=7,
    )
    split = temporal_split(sequence, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    in_channels = sequence[0].x.size(-1)
    gnn_latent_dim = 12

    pure_encoder = GNNEncoder(
        in_channels=in_channels,
        hidden_channels=24,
        latent_dim=gnn_latent_dim,
    )
    pure_decoder = GNNDecoder(
        latent_dim=gnn_latent_dim,
        hidden_channels=24,
        out_channels=in_channels,
    )
    pure_model = GraphKoopmanModel(
        encoder=pure_encoder,
        decoder=pure_decoder,
        latent_dim=gnn_latent_dim,
        time_step=0.1,
    )

    hybrid_model = _hybrid_model(
        in_channels=in_channels,
        gnn_latent_dim=gnn_latent_dim,
        physics_dim=in_channels,
    )

    pure_history = pure_model.fit(
        split.train,
        validation_sequence=split.val,
        epochs=40,
        lr=5e-3,
    )
    hybrid_history = hybrid_model.fit(
        split.train,
        validation_sequence=split.val,
        epochs=40,
        lr=5e-3,
    )

    assert hybrid_history.val_loss[-1] < pure_history.val_loss[-1]


def test_registered_physics_presets_include_graph_laplacian() -> None:
    """Static graph derivative presets should be publicly registered."""
    expected = {"graph_laplacian", "graph_gradient", "graph_curvature"}
    assert expected <= PHYSICS_PRESETS.keys()
