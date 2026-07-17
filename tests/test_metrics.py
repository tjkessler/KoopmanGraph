"""Unit tests for forecast evaluation metrics."""

import pytest
import torch

from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
from koopman_graph.metrics import (
    EvaluationResult,
    HorizonMetrics,
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
