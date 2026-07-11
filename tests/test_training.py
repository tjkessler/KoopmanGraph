"""Tests for GraphKoopmanModel.fit and training utilities."""

from unittest.mock import patch

import pytest
import torch
from torch_geometric.data import Data

from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel, LossWeights
from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.training import (
    FitHistory,
    compute_sequence_loss,
    compute_training_loss,
    constant_loss_weights,
    linear_ramp_loss_weights,
    one_step_loss,
    should_stop_early,
)


@pytest.fixture
def trainable_model() -> GraphKoopmanModel:
    """Provide a GraphKoopmanModel sized for training smoke tests.

    Returns
    -------
    GraphKoopmanModel
        Model with hidden width 16 and latent dimension 8.
    """
    encoder = GNNEncoder(in_channels=3, hidden_channels=16, latent_dim=8)
    decoder = GNNDecoder(latent_dim=8, hidden_channels=16, out_channels=3)
    return GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=8,
        time_step=0.1,
    )


def test_fit_rejects_single_snapshot(
    trainable_model: GraphKoopmanModel,
    synthetic_graph: Data,
) -> None:
    """Verify ``fit`` rejects sequences with one snapshot."""
    sequence = GraphSnapshotSequence([synthetic_graph])
    with pytest.raises(ValueError, match="at least 2 snapshots"):
        trainable_model.fit(sequence, epochs=1)


def test_fit_rejects_zero_epochs(
    trainable_model: GraphKoopmanModel,
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Verify ``fit`` rejects zero epochs."""
    with pytest.raises(ValueError, match="epochs"):
        trainable_model.fit(scaling_sequence, epochs=0)


def test_fit_returns_fit_history(
    trainable_model: GraphKoopmanModel,
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Verify ``fit`` returns a populated ``FitHistory``."""
    history = trainable_model.fit(scaling_sequence, epochs=3, lr=1e-2)
    assert isinstance(history, FitHistory)
    assert history.epochs == 3
    assert len(history.loss) == 3
    assert all(isinstance(value, float) for value in history.loss)


def test_fit_accepts_list_of_data(
    trainable_model: GraphKoopmanModel,
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Verify ``fit`` accepts a plain list of ``Data`` objects."""
    history = trainable_model.fit(list(scaling_sequence), epochs=2)
    assert len(history.loss) == 2


def test_one_step_loss_is_differentiable(
    trainable_model: GraphKoopmanModel,
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Verify ``one_step_loss`` supports backpropagation."""
    loss = one_step_loss(trainable_model, scaling_sequence[0], scaling_sequence[1])
    loss.backward()
    for param in trainable_model.parameters():
        assert param.grad is not None


def test_compute_sequence_loss_requires_two_snapshots(
    trainable_model: GraphKoopmanModel,
    synthetic_graph: Data,
) -> None:
    """Verify sequence loss requires two snapshots."""
    sequence = GraphSnapshotSequence([synthetic_graph])
    with pytest.raises(ValueError, match="at least 2 snapshots"):
        compute_sequence_loss(trainable_model, sequence)


def test_fit_loss_decreases_on_synthetic_data(
    trainable_model: GraphKoopmanModel,
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Verify training loss decreases on synthetic data."""
    torch.manual_seed(0)
    history = trainable_model.fit(scaling_sequence, epochs=100, lr=5e-3)
    assert history.loss[-1] < history.loss[0]


def test_fit_honors_device_kwarg(
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Verify ``fit`` moves the model to the requested device."""
    encoder = GNNEncoder(in_channels=3, hidden_channels=8, latent_dim=4)
    decoder = GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=3)
    model = GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=4,
        time_step=0.1,
    )
    history = model.fit(scaling_sequence, epochs=1, device="cpu")
    assert len(history.loss) == 1
    assert next(model.parameters()).device.type == "cpu"


def test_fit_honors_custom_optimizer(
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Verify ``fit`` accepts a custom optimizer class."""
    encoder = GNNEncoder(in_channels=3, hidden_channels=8, latent_dim=4)
    decoder = GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=3)
    model = GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=4,
        time_step=0.1,
    )
    history = model.fit(
        scaling_sequence,
        epochs=2,
        optimizer=torch.optim.SGD,
        lr=1e-2,
        momentum=0.9,
    )
    assert len(history.loss) == 2


def test_resolve_device_uses_explicit_device() -> None:
    """Verify ``resolve_device`` honors an explicit device argument."""
    from koopman_graph.training import resolve_device

    model = GraphKoopmanModel(
        encoder=GNNEncoder(in_channels=3, hidden_channels=8, latent_dim=4),
        decoder=GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=3),
        latent_dim=4,
        time_step=0.1,
    )
    assert resolve_device(model, "cpu").type == "cpu"


def test_resolve_device_falls_back_to_cpu_for_empty_module() -> None:
    """Verify ``resolve_device`` returns CPU for parameter-free modules."""
    from torch import nn

    from koopman_graph.training import resolve_device

    empty = nn.Module()
    assert resolve_device(empty, None).type == "cpu"


def test_constant_loss_weights_defaults() -> None:
    """Verify default loss weights enable reconstruction only."""
    weights = constant_loss_weights()
    assert weights == LossWeights(
        reconstruction=1.0,
        forward=0.0,
        backward=0.0,
        rollout=0.0,
    )


def test_compute_training_loss_skips_zero_reconstruction_weight(
    trainable_model: GraphKoopmanModel,
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Verify zero reconstruction weight skips the reconstruction term."""
    weights = LossWeights(reconstruction=0.0, forward=1.0, backward=0.0)
    loss = compute_training_loss(trainable_model, scaling_sequence, weights)
    assert loss.ndim == 0
    assert torch.isfinite(loss).item()


def test_linear_ramp_loss_weights_interpolates() -> None:
    """Verify linear ramp schedule interpolates and then holds."""
    start = constant_loss_weights(forward=0.0, backward=0.0)
    end = constant_loss_weights(forward=1.0, backward=0.5)
    schedule = linear_ramp_loss_weights(start, end, ramp_epochs=5)

    assert schedule(0) == start
    mid = schedule(2)
    assert mid.forward == pytest.approx(0.5)
    assert mid.backward == pytest.approx(0.25)
    assert schedule(4) == end
    assert schedule(10) == end


def test_linear_ramp_loss_weights_rejects_invalid_ramp_epochs() -> None:
    """Verify invalid ramp length raises."""
    start = constant_loss_weights()
    end = constant_loss_weights(forward=1.0)
    with pytest.raises(ValueError, match="ramp_epochs"):
        linear_ramp_loss_weights(start, end, ramp_epochs=0)


def test_fit_applies_gradient_clipping(
    trainable_model: GraphKoopmanModel,
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Verify ``max_grad_norm`` triggers gradient clipping."""
    with patch("koopman_graph.training.nn.utils.clip_grad_norm_") as clip_mock:
        trainable_model.fit(scaling_sequence, epochs=1, max_grad_norm=1.0)
    clip_mock.assert_called_once()
    assert clip_mock.call_args.args[1] == 1.0


def test_fit_without_gradient_clipping_skips_clip(
    trainable_model: GraphKoopmanModel,
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Verify clipping is skipped when ``max_grad_norm`` is None."""
    with patch("koopman_graph.training.nn.utils.clip_grad_norm_") as clip_mock:
        trainable_model.fit(scaling_sequence, epochs=1)
    clip_mock.assert_not_called()


def test_fit_early_stopping_stops_before_requested_epochs(
    trainable_model: GraphKoopmanModel,
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Verify early stopping can end training before ``epochs``."""
    history = trainable_model.fit(
        scaling_sequence,
        epochs=50,
        lr=1e-4,
        early_stopping_patience=2,
        early_stopping_min_delta=1e9,
    )
    assert history.stopped_early is True
    assert history.epochs < 50
    assert len(history.loss) == history.epochs


def test_fit_rejects_invalid_early_stopping_patience(
    trainable_model: GraphKoopmanModel,
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Verify invalid early stopping patience is rejected."""
    with pytest.raises(ValueError, match="early_stopping_patience"):
        trainable_model.fit(
            scaling_sequence,
            epochs=2,
            early_stopping_patience=0,
        )


def test_fit_with_loss_weights_object(
    trainable_model: GraphKoopmanModel,
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Verify ``fit`` accepts an explicit ``LossWeights`` object."""
    weights = constant_loss_weights(forward=0.5, backward=0.1)
    history = trainable_model.fit(
        scaling_sequence,
        epochs=2,
        loss_weights=weights,
        max_grad_norm=1.0,
    )
    assert len(history.loss) == 2


def test_fit_with_loss_weight_schedule(
    trainable_model: GraphKoopmanModel,
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Verify ``fit`` applies a per-epoch loss weight schedule."""
    schedule = linear_ramp_loss_weights(
        constant_loss_weights(forward=0.0, backward=0.0),
        constant_loss_weights(forward=1.0, backward=0.5),
        ramp_epochs=2,
    )
    history = trainable_model.fit(
        scaling_sequence,
        epochs=2,
        loss_weight_schedule=schedule,
    )
    assert len(history.loss) == 2


def test_fit_no_nan_with_all_loss_terms_on_synthetic_benchmark(
    trainable_model: GraphKoopmanModel,
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Verify combined losses and clipping avoid NaNs on synthetic data."""
    torch.manual_seed(0)
    history = trainable_model.fit(
        scaling_sequence,
        epochs=20,
        lr=1e-3,
        loss_weights=constant_loss_weights(forward=1.0, backward=0.1),
        max_grad_norm=1.0,
    )
    assert all(torch.isfinite(torch.tensor(value)).item() for value in history.loss)


def test_fit_with_rollout_loss_on_synthetic_benchmark(
    trainable_model: GraphKoopmanModel,
) -> None:
    """Verify rollout loss improves autoregressive prediction on the benchmark."""
    from koopman_graph.datasets import SyntheticDynamicGraphBenchmark

    sequence = SyntheticDynamicGraphBenchmark.generate(
        num_nodes=20,
        num_timesteps=30,
        in_channels=3,
        seed=42,
        noise_std=0.01,
    )
    torch.manual_seed(0)
    history = trainable_model.fit(
        sequence,
        epochs=50,
        lr=1e-3,
        loss_weights=constant_loss_weights(
            reconstruction=1.0,
            forward=1.0,
            rollout=1.0,
        ),
        rollout_horizon=10,
    )
    assert len(history.loss) == 50
    predictions = trainable_model.predict(sequence[0], steps=10)
    ground_truth = sequence[1:11]
    rollout_mse = torch.mean(
        torch.stack(
            [
                torch.mean((pred.x - truth.x) ** 2)
                for pred, truth in zip(predictions, ground_truth, strict=True)
            ]
        )
    )
    assert rollout_mse.item() < 0.05


def test_fit_on_synthetic_benchmark_dataset(
    trainable_model: GraphKoopmanModel,
) -> None:
    """Verify end-to-end training on the package synthetic benchmark."""
    from koopman_graph.datasets import SyntheticDynamicGraphBenchmark

    sequence = SyntheticDynamicGraphBenchmark.generate(
        num_nodes=10,
        num_timesteps=8,
        in_channels=3,
        seed=42,
        noise_std=0.01,
    )
    torch.manual_seed(0)
    history = trainable_model.fit(sequence, epochs=10, lr=1e-3)
    assert len(history.loss) == 10
    assert all(torch.isfinite(torch.tensor(value)).item() for value in history.loss)


def test_fit_on_ieee118_benchmark_dataset() -> None:
    """Verify end-to-end training on the IEEE 118-bus benchmark."""
    from koopman_graph.datasets import IEEE118DynamicBenchmark

    encoder = GNNEncoder(4, 32, 32)
    decoder = GNNDecoder(32, 32, 4)
    model = GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=32,
        time_step=0.1,
    )
    sequence = IEEE118DynamicBenchmark.generate(num_timesteps=8, seed=42)
    torch.manual_seed(0)
    history = model.fit(sequence, epochs=5, lr=1e-3)
    assert len(history.loss) == 5
    assert all(torch.isfinite(torch.tensor(value)).item() for value in history.loss)
    predictions = model.predict(sequence[0], steps=3)
    assert len(predictions) == 3
    assert predictions[0].x.shape == (118, 4)


def test_should_stop_early_tracks_improvement() -> None:
    """Verify early-stop helper resets patience on improvement."""
    stop, best, count = should_stop_early(
        epoch_loss=0.5,
        best_loss=1.0,
        epochs_without_improvement=2,
        patience=3,
        min_delta=0.0,
    )
    assert stop is False
    assert best == 0.5
    assert count == 0

    stop, best, count = should_stop_early(
        epoch_loss=0.6,
        best_loss=0.5,
        epochs_without_improvement=2,
        patience=3,
        min_delta=0.0,
    )
    assert stop is True
    assert best == 0.5
    assert count == 3
