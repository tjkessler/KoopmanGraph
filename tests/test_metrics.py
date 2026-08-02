"""Unit tests for forecast evaluation metrics."""

import pytest
import torch
from torch_geometric.data import Data, HeteroData

from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
from koopman_graph.data import GraphSnapshotSequence, HeteroGraphSnapshotSequence
from koopman_graph.metrics import (
    EvaluationResult,
    HorizonMetrics,
    _hetero_eval_feature_vector,
    _resolve_evaluate_sequence,
    _validate_hetero_evaluate_surface,
    evaluate_forecast,
    mae,
    mape,
    masked_mae,
    masked_mape,
    masked_rmse,
    rmse,
)


@pytest.fixture
def trainable_model() -> GraphKoopmanModel:
    """Provide a small model for metric evaluation tests."""
    encoder = GNNEncoder(in_channels=3, hidden_channels=16, latent_dim=8)
    decoder = GNNDecoder(latent_dim=8, hidden_channels=16, out_channels=3)
    return GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=8,
        time_step=0.1,
    )


def test_mae_matches_hand_computed() -> None:
    """Verify MAE against a manual calculation."""
    prediction = torch.tensor([1.0, 4.0, 6.0])
    target = torch.tensor([2.0, 2.0, 8.0])
    expected = torch.mean(torch.abs(prediction - target))
    assert torch.allclose(mae(prediction, target), expected)


def test_rmse_matches_hand_computed() -> None:
    """Verify RMSE against a manual calculation."""
    prediction = torch.tensor([1.0, 4.0, 6.0])
    target = torch.tensor([2.0, 2.0, 8.0])
    expected = torch.sqrt(torch.mean((prediction - target) ** 2))
    assert torch.allclose(rmse(prediction, target), expected)


def test_mape_matches_hand_computed() -> None:
    """Verify MAPE against a manual calculation."""
    prediction = torch.tensor([1.0, 4.0, 6.0])
    target = torch.tensor([2.0, 2.0, 8.0])
    expected = torch.mean(torch.abs((prediction - target) / target.abs()))
    assert torch.allclose(mape(prediction, target), expected)


def test_masked_metrics_match_hand_computed_multi_feature() -> None:
    """Masked MAE/RMSE/MAPE average per-node feature means over observed nodes."""
    prediction = torch.tensor(
        [
            [1.0, 2.0],
            [4.0, 6.0],
            [0.0, 0.0],
        ]
    )
    target = torch.tensor(
        [
            [1.0, 2.0],
            [2.0, 2.0],
            [1.0, 1.0],
        ]
    )
    mask = torch.tensor([False, True, True])
    # Node 1: abs err mean = (|2|+|4|)/2 = 3; mape mean = (1+2)/2 = 1.5
    # Node 2: abs err mean = (1+1)/2 = 1; mape mean = (1+1)/2 = 1
    # Masked MAE = (3+1)/2 = 2
    # Masked MSE = ((4+16)+(1+1))/4 = 22/4 = 5.5 → RMSE = sqrt(5.5)
    # Masked MAPE = (1.5+1)/2 = 1.25
    assert masked_mae(prediction, target, mask).item() == pytest.approx(2.0)
    assert masked_rmse(prediction, target, mask).item() == pytest.approx(
        (5.5) ** 0.5,
        abs=1e-6,
    )
    assert masked_mape(prediction, target, mask, eps=0.0).item() == pytest.approx(
        1.25,
        abs=1e-6,
    )


def test_evaluate_forecast_reports_per_horizon_metrics(
    trainable_model,
    scaling_sequence,
) -> None:
    """Verify multi-horizon evaluation returns expected structure."""
    result = evaluate_forecast(
        trainable_model,
        scaling_sequence,
        horizons=(1, 2),
    )
    assert isinstance(result, EvaluationResult)
    assert result.num_origins == scaling_sequence.num_timesteps - 2
    assert len(result.horizons) == 2
    assert result.horizons[0].horizon == 1
    assert result.horizons[1].horizon == 2
    for metric in result.horizons:
        assert isinstance(metric, HorizonMetrics)
        assert metric.mae >= 0.0
        assert metric.rmse >= 0.0
        assert metric.mape >= 0.0
    assert result.aggregate_mae == sum(m.mae for m in result.horizons) / 2


def test_evaluate_forecast_rejects_empty_horizons(
    trainable_model,
    scaling_sequence,
) -> None:
    """Verify empty horizons raise a clear error."""
    with pytest.raises(ValueError, match="horizons"):
        evaluate_forecast(trainable_model, scaling_sequence, horizons=())


def test_evaluate_forecast_rejects_invalid_start_index(
    trainable_model,
    scaling_sequence,
) -> None:
    """Verify invalid forecast origins raise a clear error."""
    with pytest.raises(ValueError, match="start index"):
        evaluate_forecast(
            trainable_model,
            scaling_sequence,
            horizons=(1,),
            start_indices=[99],
        )


def test_evaluate_forecast_rejects_non_positive_horizons(
    trainable_model,
    scaling_sequence,
) -> None:
    """Verify horizons below one raise a clear error."""
    with pytest.raises(ValueError, match="all horizons must be >= 1"):
        evaluate_forecast(trainable_model, scaling_sequence, horizons=(0, 2))


def test_evaluate_forecast_with_controlled_model(
    synthetic_edge_index,
) -> None:
    """Verify controlled models pull rollout controls during evaluation."""
    from torch_geometric.data import Data

    from koopman_graph.data import GraphSnapshotSequence

    model = GraphKoopmanModel(
        encoder=GNNEncoder(in_channels=3, hidden_channels=8, latent_dim=4),
        decoder=GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=3),
        latent_dim=4,
        time_step=0.1,
        control_dim=1,
    )
    snapshots = [
        Data(x=torch.randn(5, 3), edge_index=synthetic_edge_index) for _ in range(5)
    ]
    sequence = GraphSnapshotSequence(snapshots, control_inputs=torch.randn(5, 1))

    result = evaluate_forecast(model, sequence, horizons=(1, 2))

    assert result.num_origins == 3
    assert all(metric.mae >= 0.0 for metric in result.horizons)


def test_evaluate_forecast_with_dynamic_topology(trainable_model) -> None:
    """Verify dynamic-topology sequences supply future topologies."""
    from torch_geometric.data import Data

    from koopman_graph.data import GraphSnapshotSequence

    first_edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    second_edges = torch.tensor([[0, 2, 2, 1], [2, 0, 1, 2]], dtype=torch.long)
    snapshots = [
        Data(
            x=torch.randn(5, 3),
            edge_index=first_edges if t % 2 == 0 else second_edges,
        )
        for t in range(4)
    ]
    sequence = GraphSnapshotSequence(snapshots, allow_dynamic_topology=True)

    result = evaluate_forecast(trainable_model, sequence, horizons=(1,))

    assert result.num_origins == 3


def _hetero_snapshot(*, seed: int = 0) -> HeteroData:
    generator = torch.Generator().manual_seed(seed)
    data = HeteroData()
    data["node"].x = torch.randn(4, 3, generator=generator)
    data["node", "r1", "node"].edge_index = torch.tensor(
        [[0, 1, 2], [1, 2, 0]],
        dtype=torch.long,
    )
    return data


def test_hetero_eval_feature_vector_validation() -> None:
    """Stacked hetero features require present node types with ``x``."""
    snap = _hetero_snapshot()
    vec = _hetero_eval_feature_vector(snap, ("node",))
    assert vec.ndim == 1
    assert vec.numel() == 4 * 3

    with pytest.raises(ValueError, match="missing node type"):
        _hetero_eval_feature_vector(snap, ("node", "other"))

    class _NodeWithoutFeatures:
        x = None

    class _SnapshotMissingX:
        node_types = ["node"]

        def __getitem__(self, name: str) -> _NodeWithoutFeatures:
            return _NodeWithoutFeatures()

    with pytest.raises(ValueError, match="missing feature matrix x"):
        _hetero_eval_feature_vector(_SnapshotMissingX(), ("node",))


def test_validate_hetero_evaluate_surface_rejects_unsupported_options() -> None:
    """Hetero evaluate rejects dynamic topology, controls, and masks."""
    snaps = [_hetero_snapshot(seed=t) for t in range(3)]
    dynamic_snaps = [_hetero_snapshot(seed=0), _hetero_snapshot(seed=1)]
    dynamic_snaps[1]["node", "r1", "node"].edge_index = torch.tensor(
        [[0, 1], [1, 0]],
        dtype=torch.long,
    )
    dynamic = HeteroGraphSnapshotSequence(
        dynamic_snaps,
        allow_dynamic_topology=True,
    )
    assert dynamic.is_dynamic_topology
    with pytest.raises(
        ValueError,
        match="dynamic-topology HeteroGraphSnapshotSequence",
    ):
        _validate_hetero_evaluate_surface(dynamic)

    controlled = HeteroGraphSnapshotSequence(
        snaps,
        control_inputs=torch.zeros(3, 1),
    )
    with pytest.raises(ValueError, match="controlled HeteroGraphSnapshotSequence"):
        _validate_hetero_evaluate_surface(controlled)

    masks = HeteroGraphSnapshotSequence(
        snaps,
        observation_masks={
            "node": torch.ones(len(snaps), 4, dtype=torch.bool),
        },
    )
    with pytest.raises(
        ValueError,
        match="observation-masked HeteroGraphSnapshotSequence",
    ):
        _validate_hetero_evaluate_surface(masks)


def test_resolve_evaluate_sequence_wraps_inputs() -> None:
    """Plain Data / HeteroData lists and containers resolve correctly."""
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    homo = _resolve_evaluate_sequence(
        [
            Data(x=torch.randn(2, 3), edge_index=edge_index),
            Data(x=torch.randn(2, 3), edge_index=edge_index),
        ]
    )
    assert isinstance(homo, GraphSnapshotSequence)

    hetero = _resolve_evaluate_sequence([_hetero_snapshot(), _hetero_snapshot(seed=1)])
    assert isinstance(hetero, HeteroGraphSnapshotSequence)

    wrapped = GraphSnapshotSequence(
        [Data(x=torch.randn(2, 3), edge_index=edge_index) for _ in range(2)]
    )
    assert _resolve_evaluate_sequence(wrapped) is wrapped

    with pytest.raises(ValueError, match="at least one snapshot"):
        _resolve_evaluate_sequence([])
    with pytest.raises(TypeError, match="evaluate_forecast expects"):
        _resolve_evaluate_sequence(42)  # type: ignore[arg-type]


def test_evaluate_forecast_hetero_guardrails() -> None:
    """Hetero evaluate rejects delay embedding, homo models, and bad predict types."""
    from koopman_graph.nn import RelGraphDecoder, RelGraphEncoder

    snaps = [_hetero_snapshot(seed=t) for t in range(4)]
    sequence = HeteroGraphSnapshotSequence(snaps)
    hetero_model = GraphKoopmanModel(
        encoder=RelGraphEncoder(3, hidden_channels=8, latent_dim=4, num_relations=1),
        decoder=RelGraphDecoder(
            latent_dim=4,
            hidden_channels=8,
            out_channels=3,
            num_relations=1,
        ),
        latent_dim=4,
        time_step=1.0,
        koopman="hetero_graph",
        koopman_edge_types=(("node", "r1", "node"),),
    )
    homo_model = GraphKoopmanModel(
        encoder=GNNEncoder(in_channels=3, hidden_channels=8, latent_dim=4),
        decoder=GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=3),
        latent_dim=4,
        time_step=1.0,
    )

    with pytest.raises(TypeError, match="koopman='hetero_graph'"):
        evaluate_forecast(homo_model, sequence, horizons=(1,))

    delay_model = GraphKoopmanModel(
        encoder=RelGraphEncoder(3, hidden_channels=8, latent_dim=4, num_relations=1),
        decoder=RelGraphDecoder(
            latent_dim=4,
            hidden_channels=8,
            out_channels=3,
            num_relations=1,
        ),
        latent_dim=4,
        time_step=1.0,
        koopman="hetero_graph",
        koopman_edge_types=(("node", "r1", "node"),),
    )
    delay_model.n_delays = 2
    with pytest.raises(ValueError, match="delay embedding"):
        evaluate_forecast(delay_model, sequence, horizons=(1,))

    def _bad_predict(_initial, steps: int):  # noqa: ANN001
        edge_index = snaps[0]["node", "r1", "node"].edge_index
        bad = Data(x=torch.zeros(4, 3), edge_index=edge_index)
        return [bad] * steps

    hetero_model.predict = _bad_predict  # type: ignore[method-assign]
    with pytest.raises(TypeError, match="expects predict\\(\\) to return HeteroData"):
        evaluate_forecast(hetero_model, sequence, horizons=(1,), start_indices=[0])
