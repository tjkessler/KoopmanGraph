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
    """Verify ``resolve_edge_index`` requires ``edge_index`` for tensor input."""
    from koopman_graph.graph_utils import resolve_edge_index

    x = torch.randn(5, 3)
    with pytest.raises(ValueError, match="edge_index is required"):
        resolve_edge_index(x, None)


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


def test_forward_matches_decode_advance_encode(
    graph_koopman_model: GraphKoopmanModel,
    synthetic_graph: Data,
) -> None:
    """Forward must equal decoder(K · encode(x)) for the discrete pipeline."""
    graph_koopman_model.eval()
    with torch.no_grad():
        predicted = graph_koopman_model(synthetic_graph)
        z = graph_koopman_model.encode(synthetic_graph)
        z_next = graph_koopman_model.koopman.advance(z)
        expected = graph_koopman_model.decoder(
            z_next,
            synthetic_graph.edge_index,
        )
    assert torch.allclose(predicted, expected, atol=1e-6)


def test_predict_matches_repeated_operator_composition(
    graph_koopman_model: GraphKoopmanModel,
    synthetic_graph: Data,
) -> None:
    """Multi-step predict must match repeated latent advances then decode."""
    graph_koopman_model.eval()
    steps = 3
    with torch.no_grad():
        predictions = graph_koopman_model.predict(synthetic_graph, steps=steps)
        z = graph_koopman_model.encode(synthetic_graph)
        for prediction in predictions:
            z = graph_koopman_model.koopman.advance(z)
            expected = graph_koopman_model.decoder(
                z,
                synthetic_graph.edge_index,
            )
            assert torch.allclose(prediction.x, expected, atol=1e-6)


def test_rollout_returns_correct_shapes(
    graph_koopman_model: GraphKoopmanModel,
    synthetic_graph: Data,
) -> None:
    """Verify ``_rollout`` returns the expected tensor shapes."""
    predictions = graph_koopman_model._rollout(synthetic_graph, steps=3)
    assert len(predictions) == 3
    for pred, _, _ in predictions:
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


def test_predict_hold_last_known_topology(
    graph_koopman_model: GraphKoopmanModel,
    synthetic_graph: Data,
) -> None:
    """Verify partial future topologies are held for remaining rollout steps."""
    alt_edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    future_topologies = [Data(x=synthetic_graph.x, edge_index=alt_edges)]
    predictions = graph_koopman_model.predict(
        synthetic_graph,
        steps=3,
        future_topologies=future_topologies,
    )
    assert torch.equal(predictions[0].edge_index, alt_edges)
    assert torch.equal(predictions[1].edge_index, alt_edges)
    assert torch.equal(predictions[2].edge_index, alt_edges)


def test_predict_without_future_topologies_uses_initial_topology(
    graph_koopman_model: GraphKoopmanModel,
    synthetic_graph: Data,
) -> None:
    """Verify rollout without a schedule keeps the initial topology."""
    predictions = graph_koopman_model.predict(synthetic_graph, steps=2)
    for pred in predictions:
        assert torch.equal(pred.edge_index, synthetic_graph.edge_index)


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


def test_predict_preserves_edge_weight(
    graph_koopman_model: GraphKoopmanModel,
    synthetic_graph: Data,
) -> None:
    """Verify weighted predictions retain edge weights on output snapshots."""
    edge_weight = torch.tensor(
        [1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 0.5, 0.5],
        dtype=torch.float32,
    )
    graph = Data(
        x=synthetic_graph.x,
        edge_index=synthetic_graph.edge_index,
        edge_weight=edge_weight,
    )
    predictions = graph_koopman_model.predict(graph, steps=2)
    for pred in predictions:
        assert pred.edge_weight is not None
        assert torch.equal(pred.edge_weight, edge_weight)


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
    assert all(pred.shape == snapshot.x.shape for pred, _, _ in rollout)


def test_public_export() -> None:
    """Verify ``GraphKoopmanModel`` is exported from the package root."""
    from koopman_graph import GraphKoopmanModel as ExportedModel

    assert ExportedModel is GraphKoopmanModel


def test_constructor_rejects_negative_control_dim() -> None:
    """Verify negative ``control_dim`` raises ``ValueError``."""
    encoder = GNNEncoder(in_channels=3, hidden_channels=8, latent_dim=4)
    decoder = GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=3)
    with pytest.raises(ValueError, match="control_dim must be non-negative"):
        GraphKoopmanModel(
            encoder=encoder,
            decoder=decoder,
            latent_dim=4,
            time_step=0.1,
            control_dim=-1,
        )


def test_model_evaluate_delegates_to_metrics(
    graph_koopman_model: GraphKoopmanModel,
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Verify ``evaluate`` returns per-horizon forecast metrics."""
    snapshots = [
        Data(x=torch.randn(5, 3), edge_index=synthetic_edge_index) for _ in range(5)
    ]
    result = graph_koopman_model.evaluate(snapshots, horizons=(1, 2))
    assert len(result.horizons) == 2
    assert result.num_origins > 0


def test_model_spectrum_uses_time_step(
    graph_koopman_model: GraphKoopmanModel,
) -> None:
    """Verify spectrum analysis reflects the model time step."""
    spectrum = graph_koopman_model.spectrum()
    assert spectrum.time_step == graph_koopman_model.time_step
    assert spectrum.eigenvalues.shape == (graph_koopman_model.latent_dim,)


def test_predict_rejects_controls_on_uncontrolled_model(
    graph_koopman_model: GraphKoopmanModel,
    synthetic_graph: Data,
) -> None:
    """Verify controls passed to an uncontrolled model raise."""
    with pytest.raises(ValueError, match="uncontrolled model"):
        graph_koopman_model.predict(
            synthetic_graph,
            steps=1,
            controls=[torch.zeros(1)],
        )


def _controlled_model() -> GraphKoopmanModel:
    """Build a small controlled GraphKoopmanModel."""
    encoder = GNNEncoder(in_channels=3, hidden_channels=8, latent_dim=4)
    decoder = GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=3)
    return GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=4,
        time_step=0.1,
        control_dim=1,
    )


def test_controlled_predict_requires_matching_controls(
    synthetic_graph: Data,
) -> None:
    """Verify controlled prediction validates control presence and length."""
    model = _controlled_model()
    with pytest.raises(ValueError, match="controls are required"):
        model.predict(synthetic_graph, steps=1)
    with pytest.raises(ValueError, match="expected 2 control inputs"):
        model.predict(synthetic_graph, steps=2, controls=[torch.zeros(1)])


def test_controlled_fit_rejects_control_dim_mismatch(
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Verify sequences with a different control dimension are rejected."""
    model = _controlled_model()
    snapshots = [
        Data(x=torch.randn(5, 3), edge_index=synthetic_edge_index) for _ in range(3)
    ]
    sequence = GraphSnapshotSequence(snapshots, control_inputs=torch.randn(3, 2))
    with pytest.raises(ValueError, match="must match"):
        model.fit(sequence, epochs=1)


def test_fit_moves_edge_weights_to_device(
    graph_koopman_model: GraphKoopmanModel,
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Verify training sequences with edge weights are device-transferred."""
    weight = torch.rand(synthetic_edge_index.shape[1])
    snapshots = [
        Data(
            x=torch.randn(5, 3),
            edge_index=synthetic_edge_index,
            edge_weight=weight,
        )
        for _ in range(3)
    ]
    history = graph_koopman_model.fit(
        GraphSnapshotSequence(snapshots),
        epochs=1,
        device="cpu",
    )
    assert history.epochs == 1


class _MinimalDenseOperator(torch.nn.Module):
    """Minimal discrete dense operator for injection smoke tests.

    Implements :class:`~koopman_graph.operators.KoopmanOperatorContract` only
    (no built-in ``dense_inverse_matrix``). Default ``fit`` evaluates backward
    consistency via Protocol ``inverse_advance``.
    """

    def __init__(self, latent_dim: int, *, control_dim: int = 0) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.control_dim = control_dim
        self.parameterization = "dense"
        self._matrix = torch.nn.Parameter(torch.eye(latent_dim))
        if control_dim > 0:
            self.B = torch.nn.Parameter(torch.zeros(latent_dim, control_dim))
        else:
            self.register_parameter("B", None)

    @property
    def matrix(self) -> torch.Tensor:
        return self._matrix

    @property
    def K(self) -> torch.Tensor:
        return self._matrix

    def advance(
        self,
        z: torch.Tensor,
        delta_t: float | torch.Tensor | None = None,
        *,
        control: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del delta_t
        advanced = z @ self._matrix.T
        if control is not None and self.B is not None:
            advanced = advanced + control @ self.B.T
        return advanced

    def inverse_advance(
        self,
        z: torch.Tensor,
        delta_t: float | torch.Tensor | None = None,
        *,
        control: torch.Tensor | None = None,
        inverse_matrix: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del delta_t
        recovered = z
        if control is not None and self.B is not None:
            recovered = recovered - control @ self.B.T
        inv = (
            inverse_matrix
            if inverse_matrix is not None
            else torch.linalg.pinv(self._matrix)
        )
        return recovered @ inv.T

    def bound_metric(self) -> torch.Tensor:
        return torch.linalg.eigvals(self._matrix).abs().max()


def test_inject_builtin_koopman_operator() -> None:
    """Verify a pre-built ``KoopmanOperator`` can be injected."""
    from koopman_graph.operators import KoopmanOperator

    encoder = GNNEncoder(in_channels=3, hidden_channels=8, latent_dim=4)
    decoder = GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=3)
    operator = KoopmanOperator(4, parameterization="odo")
    model = GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=4,
        time_step=0.1,
        koopman=operator,
    )
    assert model.koopman is operator
    assert model.koopman.parameterization == "odo"


def test_inject_custom_operator_train_predict_smoke(
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Verify train/predict work with a minimal custom injected operator."""
    encoder = GNNEncoder(in_channels=3, hidden_channels=8, latent_dim=4)
    decoder = GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=3)
    operator = _MinimalDenseOperator(4)
    model = GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=4,
        time_step=0.1,
        koopman=operator,
    )
    assert model.koopman is operator

    snapshots = [
        Data(x=torch.randn(5, 3), edge_index=synthetic_edge_index) for _ in range(4)
    ]
    sequence = GraphSnapshotSequence(snapshots)
    history = model.fit(sequence, epochs=1)
    assert history.epochs == 1

    future = model.predict(snapshots[0], steps=2)
    assert len(future) == 2
    assert future[0].x.shape == (5, 3)


class _MinimalContinuousOperator(torch.nn.Module):
    """Minimal continuous generator for injection smoke tests."""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.control_dim = 0
        self.parameterization = "dense"
        self._generator = torch.nn.Parameter(-0.5 * torch.eye(latent_dim))

    @property
    def matrix(self) -> torch.Tensor:
        return self._generator

    def advance(
        self,
        z: torch.Tensor,
        delta_t: float | torch.Tensor | None = None,
        *,
        control: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if delta_t is None:
            msg = "delta_t is required for continuous advance"
            raise ValueError(msg)
        del control
        delta = torch.as_tensor(delta_t, dtype=z.dtype, device=z.device)
        transition = torch.linalg.matrix_exp(self._generator * delta)
        return z @ transition.T

    def inverse_advance(
        self,
        z: torch.Tensor,
        delta_t: float | torch.Tensor | None = None,
        *,
        control: torch.Tensor | None = None,
        inverse_matrix: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del control, inverse_matrix
        if delta_t is None:
            msg = "delta_t is required for continuous inverse_advance"
            raise ValueError(msg)
        delta = torch.as_tensor(delta_t, dtype=z.dtype, device=z.device)
        transition = torch.linalg.matrix_exp(self._generator * delta)
        return z @ torch.linalg.inv(transition).T

    def bound_metric(self) -> torch.Tensor:
        return torch.linalg.eigvals(self._generator).real.max()


def test_inject_custom_continuous_operator_train_predict_smoke(
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Custom continuous contract modules route delta_t via unified advance."""
    encoder = GNNEncoder(in_channels=3, hidden_channels=8, latent_dim=4)
    decoder = GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=3)
    operator = _MinimalContinuousOperator(4)
    model = GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=4,
        time_step=0.1,
        dynamics_mode="continuous",
        koopman=operator,
    )
    assert model.koopman is operator
    assert model.is_continuous

    spectrum = model.spectrum()
    assert spectrum.eigenvalues.shape[0] == 4

    snapshots = [
        Data(x=torch.randn(5, 3), edge_index=synthetic_edge_index) for _ in range(4)
    ]
    sequence = GraphSnapshotSequence(snapshots)
    history = model.fit(sequence, epochs=1)
    assert history.epochs == 1

    future = model.predict(snapshots[0], steps=2)
    assert len(future) == 2
    assert future[0].x.shape == (5, 3)


def test_inject_rejects_non_default_factory_kwargs() -> None:
    """Verify injection conflicts with non-default factory kwargs."""
    from koopman_graph.operators import KoopmanOperator

    encoder = GNNEncoder(in_channels=3, hidden_channels=8, latent_dim=4)
    decoder = GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=3)
    operator = KoopmanOperator(4)
    with pytest.raises(ValueError, match="mutually exclusive"):
        GraphKoopmanModel(
            encoder=encoder,
            decoder=decoder,
            latent_dim=4,
            time_step=0.1,
            koopman=operator,
            koopman_parameterization="odo",
        )


def test_inject_rejects_latent_dim_mismatch() -> None:
    """Verify injected operator latent_dim must match the model."""
    from koopman_graph.operators import KoopmanOperator

    encoder = GNNEncoder(in_channels=3, hidden_channels=8, latent_dim=4)
    decoder = GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=3)
    operator = KoopmanOperator(8)
    with pytest.raises(ValueError, match="koopman.latent_dim"):
        GraphKoopmanModel(
            encoder=encoder,
            decoder=decoder,
            latent_dim=4,
            time_step=0.1,
            koopman=operator,
        )


def test_inject_rejects_control_dim_mismatch() -> None:
    """Verify injected operator control_dim must match the model."""
    from koopman_graph.operators import KoopmanOperator

    encoder = GNNEncoder(in_channels=3, hidden_channels=8, latent_dim=4)
    decoder = GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=3)
    operator = KoopmanOperator(4, control_dim=2)
    with pytest.raises(ValueError, match="koopman.control_dim"):
        GraphKoopmanModel(
            encoder=encoder,
            decoder=decoder,
            latent_dim=4,
            time_step=0.1,
            control_dim=0,
            koopman=operator,
        )


def test_inject_rejects_dynamics_mode_mismatch_for_builtin() -> None:
    """Verify built-in continuous operators require continuous dynamics_mode."""
    from koopman_graph.operators import ContinuousKoopmanOperator

    encoder = GNNEncoder(in_channels=3, hidden_channels=8, latent_dim=4)
    decoder = GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=3)
    operator = ContinuousKoopmanOperator(4)
    with pytest.raises(ValueError, match="dynamics_mode='continuous'"):
        GraphKoopmanModel(
            encoder=encoder,
            decoder=decoder,
            latent_dim=4,
            time_step=0.1,
            dynamics_mode="discrete",
            koopman=operator,
        )


def test_inject_rejects_non_module_operator() -> None:
    """Verify injected operators must be ``nn.Module`` instances."""
    encoder = GNNEncoder(in_channels=3, hidden_channels=8, latent_dim=4)
    decoder = GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=3)

    class _NotAModule:
        latent_dim = 4
        control_dim = 0
        parameterization = "dense"

    with pytest.raises(TypeError, match="nn.Module"):
        GraphKoopmanModel(
            encoder=encoder,
            decoder=decoder,
            latent_dim=4,
            time_step=0.1,
            koopman=_NotAModule(),  # type: ignore[arg-type]
        )


def test_default_factory_path_unchanged() -> None:
    """Verify omitting ``koopman`` still builds a dense discrete operator."""
    encoder = GNNEncoder(in_channels=3, hidden_channels=8, latent_dim=4)
    decoder = GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=3)
    model = GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=4,
        time_step=0.1,
    )
    from koopman_graph.operators import KoopmanOperator

    assert isinstance(model.koopman, KoopmanOperator)
    assert model.koopman.parameterization == "dense"
