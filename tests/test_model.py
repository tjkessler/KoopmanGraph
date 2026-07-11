"""Integration tests for GraphKoopmanModel."""

import pytest
import torch
from torch_geometric.data import Data

from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
from koopman_graph.data import GraphSnapshotSequence


@pytest.fixture
def graph_koopman_model() -> GraphKoopmanModel:
    """Provide a small GraphKoopmanModel for integration tests.

    Returns
    -------
    GraphKoopmanModel
        Model with 3 input/output channels and latent dimension 4.
    """
    encoder = GNNEncoder(in_channels=3, hidden_channels=8, latent_dim=4)
    decoder = GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=3)
    return GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=4,
        time_step=0.1,
    )


def test_constructor_stores_time_step(graph_koopman_model: GraphKoopmanModel) -> None:
    """Verify ``time_step`` is stored on construction."""
    assert graph_koopman_model.time_step == 0.1


def test_constructor_rejects_mismatched_encoder_latent_dim() -> None:
    """Verify encoder latent mismatch raises ``ValueError``."""
    encoder = GNNEncoder(in_channels=3, hidden_channels=8, latent_dim=4)
    decoder = GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=3)
    with pytest.raises(ValueError, match="encoder.latent_dim"):
        GraphKoopmanModel(
            encoder=encoder,
            decoder=decoder,
            latent_dim=8,
            time_step=0.1,
        )


def test_constructor_rejects_mismatched_decoder_latent_dim() -> None:
    """Verify decoder latent mismatch raises ``ValueError``."""
    encoder = GNNEncoder(in_channels=3, hidden_channels=8, latent_dim=8)
    decoder = GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=3)
    with pytest.raises(ValueError, match="decoder.latent_dim"):
        GraphKoopmanModel(
            encoder=encoder,
            decoder=decoder,
            latent_dim=8,
            time_step=0.1,
        )


def test_constructor_rejects_non_positive_time_step() -> None:
    """Verify non-positive ``time_step`` raises ``ValueError``."""
    encoder = GNNEncoder(in_channels=3, hidden_channels=8, latent_dim=4)
    decoder = GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=3)
    with pytest.raises(ValueError, match="time_step"):
        GraphKoopmanModel(
            encoder=encoder,
            decoder=decoder,
            latent_dim=4,
            time_step=0.0,
        )


def test_constructor_rejects_non_positive_latent_dim() -> None:
    """Verify non-positive ``latent_dim`` raises ``ValueError``."""
    encoder = GNNEncoder(in_channels=3, hidden_channels=8, latent_dim=4)
    decoder = GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=3)
    with pytest.raises(ValueError, match="latent_dim must be positive"):
        GraphKoopmanModel(
            encoder=encoder,
            decoder=decoder,
            latent_dim=0,
            time_step=0.1,
        )


def test_resolve_edge_index_raises_for_tensor_without_edges() -> None:
    """Verify ``_resolve_edge_index`` requires ``edge_index`` for tensor input."""
    x = torch.randn(5, 3)
    with pytest.raises(ValueError, match="edge_index is required"):
        GraphKoopmanModel._resolve_edge_index(x, None)


def test_forward_with_data_object(
    graph_koopman_model: GraphKoopmanModel,
    synthetic_graph: Data,
) -> None:
    """Verify forward accepts a PyG ``Data`` object."""
    out = graph_koopman_model(synthetic_graph)
    assert out.shape == (5, 3)
    assert torch.isfinite(out).all()


def test_forward_with_tensor_inputs(
    graph_koopman_model: GraphKoopmanModel,
    synthetic_graph: Data,
) -> None:
    """Verify forward accepts separate tensor inputs."""
    out = graph_koopman_model(synthetic_graph.x, synthetic_graph.edge_index)
    assert out.shape == (5, 3)


def test_rollout_returns_correct_shapes(
    graph_koopman_model: GraphKoopmanModel,
    synthetic_graph: Data,
) -> None:
    """Verify ``_rollout`` returns the expected tensor shapes."""
    predictions = graph_koopman_model._rollout(synthetic_graph, steps=3)
    assert len(predictions) == 3
    for pred in predictions:
        assert pred.shape == (5, 3)
        assert torch.isfinite(pred).all()


def test_rollout_rejects_invalid_steps(
    graph_koopman_model: GraphKoopmanModel,
    synthetic_graph: Data,
) -> None:
    """Verify invalid rollout ``steps`` raises ``ValueError``."""
    with pytest.raises(ValueError, match="steps"):
        graph_koopman_model._rollout(synthetic_graph, steps=0)


def test_predict_returns_data_objects_with_correct_shapes(
    graph_koopman_model: GraphKoopmanModel,
    synthetic_graph: Data,
) -> None:
    """Verify ``predict`` returns ``Data`` with correct shapes."""
    steps = 4
    predictions = graph_koopman_model.predict(synthetic_graph, steps=steps)
    assert len(predictions) == steps
    for pred in predictions:
        assert isinstance(pred, Data)
        assert pred.x.shape == synthetic_graph.x.shape
        assert pred.edge_index.shape == synthetic_graph.edge_index.shape
        assert torch.isfinite(pred.x).all()


def test_predict_with_tensor_inputs(
    graph_koopman_model: GraphKoopmanModel,
    synthetic_graph: Data,
) -> None:
    """Verify ``predict`` accepts separate tensor inputs."""
    predictions = graph_koopman_model.predict(
        synthetic_graph.x,
        steps=2,
        edge_index=synthetic_graph.edge_index,
    )
    assert len(predictions) == 2
    assert all(pred.x.shape == synthetic_graph.x.shape for pred in predictions)


def test_predict_runs_without_gradients(
    graph_koopman_model: GraphKoopmanModel,
    synthetic_graph: Data,
) -> None:
    """Verify ``predict`` disables autograd tracking."""
    x = synthetic_graph.x.detach().clone().requires_grad_(True)
    graph = Data(x=x, edge_index=synthetic_graph.edge_index)
    predictions = graph_koopman_model.predict(graph, steps=2)
    for pred in predictions:
        assert pred.x.grad_fn is None


def test_predict_restores_training_mode(
    graph_koopman_model: GraphKoopmanModel,
    synthetic_graph: Data,
) -> None:
    """Verify ``predict`` restores the prior training mode."""
    graph_koopman_model.train()
    graph_koopman_model.predict(synthetic_graph, steps=1)
    assert graph_koopman_model.training

    graph_koopman_model.eval()
    graph_koopman_model.predict(synthetic_graph, steps=1)
    assert not graph_koopman_model.training


def test_predict_rejects_invalid_steps(
    graph_koopman_model: GraphKoopmanModel,
    synthetic_graph: Data,
) -> None:
    """Verify invalid predict ``steps`` raises ``ValueError``."""
    with pytest.raises(ValueError, match="steps"):
        graph_koopman_model.predict(synthetic_graph, steps=0)


def test_gradient_flow_end_to_end(
    graph_koopman_model: GraphKoopmanModel,
    synthetic_graph: Data,
) -> None:
    """Verify end-to-end gradients flow through the full model."""
    out = graph_koopman_model(synthetic_graph)
    loss = out.sum()
    loss.backward()

    for param in graph_koopman_model.parameters():
        assert param.grad is not None


def test_integration_with_graph_snapshot_sequence(
    synthetic_graph_sequence: GraphSnapshotSequence,
) -> None:
    """Verify model integration with ``GraphSnapshotSequence``."""
    encoder = GNNEncoder(
        in_channels=synthetic_graph_sequence.in_channels,
        hidden_channels=8,
        latent_dim=4,
    )
    decoder = GNNDecoder(
        latent_dim=4,
        hidden_channels=8,
        out_channels=synthetic_graph_sequence.in_channels,
    )
    model = GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=4,
        time_step=0.1,
    )

    snapshot = synthetic_graph_sequence[0]
    prediction = model(snapshot)
    assert prediction.shape == snapshot.x.shape

    rollout = model._rollout(snapshot, steps=2)
    assert len(rollout) == 2
    assert all(pred.shape == snapshot.x.shape for pred in rollout)


def test_public_export() -> None:
    """Verify ``GraphKoopmanModel`` is exported from the package root."""
    from koopman_graph import GraphKoopmanModel as ExportedModel

    assert ExportedModel is GraphKoopmanModel
