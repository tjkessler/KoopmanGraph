"""Tests for GraphKoopmanModel.fit and training utilities."""

from unittest.mock import patch

import pytest
import torch
from torch_geometric.data import Data

from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel, LossWeights
from koopman_graph.data import GraphSnapshotSequence, temporal_split
from koopman_graph.training import (
    FitHistory,
    compute_sequence_loss,
    compute_training_loss,
    constant_loss_weights,
    linear_ramp_loss_weights,
    one_step_loss,
    resolve_lr_scheduler,
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
    return _make_trainable_model()


def _make_trainable_model() -> GraphKoopmanModel:
    """Construct the small model used by training tests."""
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
        eigenvalue=0.0,
    )


def test_compute_training_loss_skips_zero_reconstruction_weight(
    trainable_model: GraphKoopmanModel,
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Verify zero reconstruction weight skips the reconstruction term."""
    weights = LossWeights(reconstruction=0.0, forward=1.0, backward=0.0)
    breakdown = compute_training_loss(trainable_model, scaling_sequence, weights)
    assert breakdown.total.ndim == 0
    assert torch.isfinite(breakdown.total).item()


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


def test_fit_records_validation_loss(
    trainable_model: GraphKoopmanModel,
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Verify validation loss is recorded when a validation sequence is provided."""
    snapshots = [
        Data(x=torch.ones(5, 3) * (0.9**t), edge_index=synthetic_edge_index)
        for t in range(20)
    ]
    split = temporal_split(GraphSnapshotSequence(snapshots))
    history = trainable_model.fit(
        split.train,
        validation_sequence=split.val,
        epochs=2,
    )
    assert history.val_loss is not None
    assert len(history.val_loss) == 2
    assert all(value >= 0.0 for value in history.val_loss)


def test_fit_early_stopping_uses_validation_loss(
    trainable_model: GraphKoopmanModel,
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Verify early stopping can monitor validation loss."""
    snapshots = [
        Data(x=torch.ones(5, 3) * (0.9**t), edge_index=synthetic_edge_index)
        for t in range(20)
    ]
    split = temporal_split(GraphSnapshotSequence(snapshots))
    history = trainable_model.fit(
        split.train,
        validation_sequence=split.val,
        epochs=50,
        lr=1e-4,
        early_stopping_patience=2,
        early_stopping_min_delta=1e9,
        early_stopping_monitor="val",
    )
    assert history.stopped_early is True
    assert history.epochs < 50
    assert history.val_loss is not None


def test_fit_rejects_val_monitor_without_validation_sequence(
    trainable_model: GraphKoopmanModel,
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Verify validation monitor requires a validation sequence."""
    with pytest.raises(ValueError, match="validation_sequence"):
        trainable_model.fit(
            scaling_sequence,
            epochs=2,
            early_stopping_monitor="val",
        )


def test_fit_rejects_short_validation_sequence(
    trainable_model: GraphKoopmanModel,
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Verify validation sequences need at least two snapshots."""
    with pytest.raises(ValueError, match="validation_sequence"):
        trainable_model.fit(
            scaling_sequence,
            validation_sequence=GraphSnapshotSequence([scaling_sequence[0]]),
            epochs=2,
        )


def _latent_rollout_norm(
    model: GraphKoopmanModel,
    sequence: GraphSnapshotSequence,
    steps: int,
) -> float:
    """Return the final latent norm after an autoregressive rollout."""
    model.eval()
    with torch.no_grad():
        edge_index = sequence[0].edge_index
        z = model.encoder(sequence[0], edge_index)
        for _ in range(steps):
            z = model.koopman(z)
        return float(z.norm().item())


def test_odo_operator_stays_bounded_under_long_rollout(
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Verify ODO parameterization avoids latent blow-up on long rollouts."""
    encoder = GNNEncoder(in_channels=3, hidden_channels=16, latent_dim=8)
    decoder = GNNDecoder(latent_dim=8, hidden_channels=16, out_channels=3)
    stable_model = GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=8,
        time_step=0.1,
        koopman_parameterization="odo",
        koopman_max_spectral_radius=0.95,
    )
    torch.manual_seed(0)
    stable_model.fit(
        scaling_sequence,
        epochs=5,
        lr=1e-2,
        loss_weights=constant_loss_weights(eigenvalue=0.1),
    )
    stable_norm = _latent_rollout_norm(stable_model, scaling_sequence, steps=40)

    unstable_encoder = GNNEncoder(in_channels=3, hidden_channels=16, latent_dim=8)
    unstable_decoder = GNNDecoder(latent_dim=8, hidden_channels=16, out_channels=3)
    unstable_model = GraphKoopmanModel(
        encoder=unstable_encoder,
        decoder=unstable_decoder,
        latent_dim=8,
        time_step=0.1,
    )
    with torch.no_grad():
        unstable_eigs = torch.tensor([1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7])
        unstable_model.koopman.K.copy_(torch.diag(unstable_eigs))
    unstable_norm = _latent_rollout_norm(unstable_model, scaling_sequence, steps=40)

    assert stable_norm < unstable_norm
    assert stable_norm < 1e3


def _alternating_topology_sequence(
    num_nodes: int = 5,
    num_timesteps: int = 8,
) -> GraphSnapshotSequence:
    """Build a dynamic-topology sequence with alternating edge sets."""
    edges_a = torch.tensor(
        [[i for i in range(num_nodes - 1)], [i + 1 for i in range(num_nodes - 1)]],
        dtype=torch.long,
    )
    edges_b = torch.tensor(
        [[0, 1, 2, 3], [1, 2, 3, 4]],
        dtype=torch.long,
    )
    snapshots = []
    for t in range(num_timesteps):
        edges = edges_a if t % 2 == 0 else edges_b
        x = torch.ones(num_nodes, 3) * (0.9**t)
        snapshots.append(Data(x=x, edge_index=edges))
    return GraphSnapshotSequence(snapshots, allow_dynamic_topology=True)


def test_fit_on_dynamic_topology_sequence(trainable_model: GraphKoopmanModel) -> None:
    """Verify training succeeds and loss decreases on dynamic topology."""
    sequence = _alternating_topology_sequence()
    torch.manual_seed(0)
    history = trainable_model.fit(sequence, epochs=15, lr=1e-2)
    assert len(history.loss) == 15
    assert history.loss[-1] < history.loss[0]


def test_fit_records_per_term_loss_history(
    trainable_model: GraphKoopmanModel,
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Verify FitHistory records unweighted per-term training losses."""
    history = trainable_model.fit(
        scaling_sequence,
        epochs=2,
        loss_weights=constant_loss_weights(forward=0.5, backward=0.25),
    )
    assert len(history.reconstruction_loss) == 2
    assert len(history.forward_loss) == 2
    assert len(history.backward_loss) == 2
    assert len(history.rollout_loss) == 2
    assert len(history.eigenvalue_loss) == 2
    assert all(value >= 0.0 for value in history.forward_loss)


def test_fit_records_per_term_validation_history(
    trainable_model: GraphKoopmanModel,
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Verify validation history includes all five unweighted loss terms."""
    snapshots = [
        Data(x=torch.ones(5, 3) * (0.9**t), edge_index=synthetic_edge_index)
        for t in range(20)
    ]
    split = temporal_split(GraphSnapshotSequence(snapshots))
    history = trainable_model.fit(
        split.train,
        validation_sequence=split.val,
        epochs=2,
        loss_weights=constant_loss_weights(forward=0.5),
    )
    assert history.val_reconstruction_loss is not None
    assert history.val_forward_loss is not None
    assert history.val_backward_loss is not None
    assert history.val_rollout_loss is not None
    assert history.val_eigenvalue_loss is not None
    assert len(history.val_reconstruction_loss) == 2
    assert len(history.val_forward_loss) == 2


def test_fit_with_lr_scheduler_factory(
    trainable_model: GraphKoopmanModel,
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Verify fit accepts an LR scheduler factory without error."""
    trainable_model.fit(
        scaling_sequence,
        epochs=2,
        lr_scheduler=lambda optim: torch.optim.lr_scheduler.StepLR(
            optim,
            step_size=1,
            gamma=0.5,
        ),
    )


def test_resolve_lr_scheduler_factory_steps_learning_rate() -> None:
    """Verify scheduler factories reduce the optimizer learning rate."""
    model = torch.nn.Linear(2, 1)
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = resolve_lr_scheduler(
        lambda opt: torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5),
        optim,
    )
    assert scheduler is not None
    optim.step()
    scheduler.step()
    assert optim.param_groups[0]["lr"] == pytest.approx(5e-3)


def test_fit_rollout_start_indices_all_differs_from_default(
    trainable_model: GraphKoopmanModel,
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Verify multi-origin rollout loss can be enabled via fit kwargs."""
    snapshots = [
        Data(x=torch.ones(5, 3) * (0.9**t), edge_index=synthetic_edge_index)
        for t in range(8)
    ]
    sequence = GraphSnapshotSequence(snapshots)
    weights = constant_loss_weights(reconstruction=0.0, rollout=1.0)
    default_breakdown = compute_training_loss(
        trainable_model,
        sequence,
        weights,
        rollout_horizon=2,
        rollout_start_indices=[0],
    )
    all_breakdown = compute_training_loss(
        trainable_model,
        sequence,
        weights,
        rollout_horizon=2,
        rollout_start_indices=[0, 1, 2, 3, 4],
    )
    assert not torch.allclose(default_breakdown.rollout, all_breakdown.rollout)


def test_rollout_start_seed_makes_random_origins_reproducible(
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Verify rollout_start_seed fixes random origin sampling per epoch."""
    from koopman_graph.training import resolve_rollout_start_indices

    kwargs = {
        "horizon": 2,
        "rollout_starts_per_epoch": 2,
        "rollout_start_seed": 7,
        "epoch": 0,
    }
    first = resolve_rollout_start_indices(scaling_sequence, **kwargs)
    second = resolve_rollout_start_indices(scaling_sequence, **kwargs)
    assert first == second


def test_fit_accepts_multiple_training_sequences(
    trainable_model: GraphKoopmanModel,
    scaling_sequence: GraphSnapshotSequence,
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Verify fit accepts a list of trajectories and trains successfully."""
    second = GraphSnapshotSequence(
        [
            Data(x=torch.ones(5, 3) * (1.1**t), edge_index=synthetic_edge_index)
            for t in range(scaling_sequence.num_timesteps)
        ]
    )
    history = trainable_model.fit(
        [scaling_sequence, second],
        epochs=3,
        lr=1e-2,
    )
    assert len(history.loss) == 3


def test_fit_rejects_mismatched_validation_sequence_list(
    trainable_model: GraphKoopmanModel,
    scaling_sequence: GraphSnapshotSequence,
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Verify validation list length must match training trajectories."""
    second = GraphSnapshotSequence(
        [
            Data(x=torch.ones(5, 3) * (1.1**t), edge_index=synthetic_edge_index)
            for t in range(scaling_sequence.num_timesteps)
        ]
    )
    with pytest.raises(ValueError, match="validation_sequence list length"):
        trainable_model.fit(
            [scaling_sequence, second],
            validation_sequence=[scaling_sequence],
            epochs=1,
        )


def test_windowed_fit_takes_multiple_optimizer_steps(
    trainable_model: GraphKoopmanModel,
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Verify window batches trigger multiple optimizer updates per epoch."""

    class CountingAdam(torch.optim.Adam):
        steps = 0

        def step(self, closure=None):
            type(self).steps += 1
            return super().step(closure)

    CountingAdam.steps = 0
    trainable_model.fit(
        scaling_sequence,
        epochs=2,
        optimizer=CountingAdam,
        window_length=3,
        batch_size=2,
        window_seed=0,
    )

    assert CountingAdam.steps == 4


def test_windowed_and_full_sequence_training_both_converge(
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Verify both training modes reduce loss on deterministic dynamics."""
    torch.manual_seed(11)
    full_model = _make_trainable_model()
    torch.manual_seed(11)
    windowed_model = _make_trainable_model()

    torch.manual_seed(0)
    full_history = full_model.fit(scaling_sequence, epochs=20, lr=5e-3)
    torch.manual_seed(0)
    windowed_history = windowed_model.fit(
        scaling_sequence,
        epochs=20,
        lr=5e-3,
        window_length=3,
        batch_size=2,
        window_seed=7,
    )

    assert full_history.loss[-1] < full_history.loss[0]
    assert windowed_history.loss[-1] < windowed_history.loss[0]


def test_windowed_fit_accepts_multiple_sequences(
    trainable_model: GraphKoopmanModel,
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Verify window sampling pools multiple training trajectories."""
    history = trainable_model.fit(
        [scaling_sequence, scaling_sequence],
        epochs=2,
        window_length=3,
        batch_size=4,
        windows_per_epoch=4,
        window_seed=3,
    )

    assert history.epochs == 2
    assert len(history.reconstruction_loss) == 2


def test_windowed_fit_rejects_rollout_horizon_longer_than_window(
    trainable_model: GraphKoopmanModel,
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Verify rollout loss must fit inside each sampled window."""
    with pytest.raises(ValueError, match="needs more than 3"):
        trainable_model.fit(
            scaling_sequence,
            epochs=1,
            window_length=3,
            rollout_horizon=3,
            loss_weights=LossWeights(reconstruction=0.0, rollout=1.0),
        )
