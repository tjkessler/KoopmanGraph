"""Tests for forward/backward consistency and related loss functions."""

import pytest
import torch
from torch_geometric.data import Data

from koopman_graph import (
    BackwardConsistencyLoss,
    ForwardConsistencyLoss,
    GNNDecoder,
    GNNEncoder,
    GraphKoopmanModel,
)
from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.losses import rollout_sequence_loss
from koopman_graph.operator import KoopmanOperator
from koopman_graph.training import (
    compute_backward_consistency_sequence_loss,
    compute_forward_consistency_sequence_loss,
    compute_training_loss,
    constant_loss_weights,
)


@pytest.fixture
def trainable_model() -> GraphKoopmanModel:
    """Provide a GraphKoopmanModel sized for loss tests."""
    encoder = GNNEncoder(in_channels=3, hidden_channels=16, latent_dim=8)
    decoder = GNNDecoder(latent_dim=8, hidden_channels=16, out_channels=3)
    return GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=8,
        time_step=0.1,
    )


def test_forward_consistency_loss_zero_for_linear_latent_dynamics() -> None:
    """Verify loss is zero when z_{t+1} equals Koopman advance of z_t."""
    latent_dim = 4
    koopman = KoopmanOperator(latent_dim, init_mode="identity")
    loss_fn = ForwardConsistencyLoss()

    z_t = torch.randn(5, latent_dim)
    z_t1 = koopman(z_t)

    loss = loss_fn(z_t, z_t1, koopman)
    assert loss.item() == pytest.approx(0.0)


def test_forward_consistency_loss_nonzero_for_inconsistent_latents() -> None:
    """Verify loss is positive when latent states are inconsistent."""
    latent_dim = 4
    koopman = KoopmanOperator(latent_dim, init_mode="identity")
    loss_fn = ForwardConsistencyLoss()

    z_t = torch.randn(5, latent_dim)
    z_t1 = torch.randn(5, latent_dim)

    loss = loss_fn(z_t, z_t1, koopman)
    assert loss.item() > 0.0


def test_forward_consistency_loss_gradient_flow() -> None:
    """Verify gradients reach the Koopman matrix."""
    latent_dim = 4
    koopman = KoopmanOperator(latent_dim, init_mode="identity_noise")
    loss_fn = ForwardConsistencyLoss()

    z_t = torch.randn(5, latent_dim, requires_grad=True)
    z_t1 = torch.randn(5, latent_dim)

    loss = loss_fn(z_t, z_t1, koopman)
    loss.backward()

    assert koopman.K.grad is not None
    assert z_t.grad is not None


def test_forward_consistency_loss_on_model_pair(
    trainable_model: GraphKoopmanModel,
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Verify forward consistency loss returns a positive scalar for a pair."""
    loss_fn = ForwardConsistencyLoss()
    edge_index = scaling_sequence[0].edge_index
    z_t = trainable_model.encoder(scaling_sequence[0], edge_index)
    z_t1 = trainable_model.encoder(scaling_sequence[1], edge_index)
    loss = loss_fn(z_t, z_t1, trainable_model.koopman)
    assert loss.ndim == 0
    assert loss.item() > 0.0


def test_forward_consistency_loss_is_differentiable_on_model_pair(
    trainable_model: GraphKoopmanModel,
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Verify forward consistency loss backpropagates through encoder and Koopman."""
    loss_fn = ForwardConsistencyLoss()
    edge_index = scaling_sequence[0].edge_index
    z_t = trainable_model.encoder(scaling_sequence[0], edge_index)
    z_t1 = trainable_model.encoder(scaling_sequence[1], edge_index)
    loss = loss_fn(z_t, z_t1, trainable_model.koopman)
    loss.backward()
    for param in trainable_model.encoder.parameters():
        assert param.grad is not None
    assert trainable_model.koopman.K.grad is not None


def test_compute_forward_consistency_sequence_loss(
    trainable_model: GraphKoopmanModel,
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Verify sequence-level forward consistency loss averages pairs."""
    loss = compute_forward_consistency_sequence_loss(trainable_model, scaling_sequence)
    assert loss.ndim == 0
    assert loss.item() > 0.0


def test_compute_training_loss_without_forward_term(
    trainable_model: GraphKoopmanModel,
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Verify weight=0 returns reconstruction loss only."""
    from koopman_graph.training import compute_sequence_loss

    breakdown = compute_training_loss(
        trainable_model,
        scaling_sequence,
        constant_loss_weights(),
    )
    recon = compute_sequence_loss(trainable_model, scaling_sequence)
    assert torch.allclose(breakdown.total, recon)


def test_compute_training_loss_with_forward_term(
    trainable_model: GraphKoopmanModel,
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Verify combined loss exceeds reconstruction loss when weight > 0."""
    from koopman_graph.training import compute_sequence_loss

    weight = 1.0
    breakdown = compute_training_loss(
        trainable_model,
        scaling_sequence,
        constant_loss_weights(forward=weight),
    )
    recon = compute_sequence_loss(trainable_model, scaling_sequence)
    fc = compute_forward_consistency_sequence_loss(trainable_model, scaling_sequence)
    expected = recon + weight * fc
    assert torch.allclose(breakdown.total, expected)


def test_fit_with_forward_consistency_weight(
    trainable_model: GraphKoopmanModel,
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Verify fit accepts forward consistency weights and trains."""
    history = trainable_model.fit(
        scaling_sequence,
        epochs=3,
        lr=1e-2,
        loss_weights=constant_loss_weights(forward=1.0),
    )
    assert len(history.loss) == 3
    assert all(isinstance(value, float) for value in history.loss)


def test_forward_consistency_sequence_loss_requires_two_snapshots(
    trainable_model: GraphKoopmanModel,
    synthetic_graph: Data,
) -> None:
    """Verify sequence forward consistency loss requires two snapshots."""
    sequence = GraphSnapshotSequence([synthetic_graph])
    with pytest.raises(ValueError, match="at least 2 snapshots"):
        compute_forward_consistency_sequence_loss(trainable_model, sequence)


def test_backward_consistency_loss_zero_for_linear_latent_dynamics() -> None:
    """Verify loss is zero when z_t equals inverse Koopman step of z_{t+1}."""
    latent_dim = 4
    koopman = KoopmanOperator(latent_dim, init_mode="identity")
    loss_fn = BackwardConsistencyLoss()

    z_t = torch.randn(5, latent_dim)
    z_t1 = koopman(z_t)

    loss = loss_fn(z_t, z_t1, koopman)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_backward_consistency_loss_nonzero_for_inconsistent_latents() -> None:
    """Verify loss is positive when latent states are inconsistent."""
    latent_dim = 4
    koopman = KoopmanOperator(latent_dim, init_mode="identity")
    loss_fn = BackwardConsistencyLoss()

    z_t = torch.randn(5, latent_dim)
    z_t1 = torch.randn(5, latent_dim)

    loss = loss_fn(z_t, z_t1, koopman)
    assert loss.item() > 0.0


def test_backward_consistency_loss_gradient_flow() -> None:
    """Verify gradients reach the Koopman matrix."""
    latent_dim = 4
    koopman = KoopmanOperator(latent_dim, init_mode="identity_noise")
    loss_fn = BackwardConsistencyLoss()

    z_t = torch.randn(5, latent_dim)
    z_t1 = torch.randn(5, latent_dim, requires_grad=True)

    loss = loss_fn(z_t, z_t1, koopman)
    loss.backward()

    assert koopman.K.grad is not None
    assert z_t1.grad is not None


def test_backward_consistency_loss_on_model_pair(
    trainable_model: GraphKoopmanModel,
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Verify backward consistency loss returns a positive scalar for a pair."""
    loss_fn = BackwardConsistencyLoss()
    edge_index = scaling_sequence[0].edge_index
    z_t = trainable_model.encoder(scaling_sequence[0], edge_index)
    z_t1 = trainable_model.encoder(scaling_sequence[1], edge_index)
    loss = loss_fn(z_t, z_t1, trainable_model.koopman)
    assert loss.ndim == 0
    assert loss.item() > 0.0


def test_backward_consistency_loss_is_differentiable_on_model_pair(
    trainable_model: GraphKoopmanModel,
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Verify backward consistency loss backpropagates through encoder and Koopman."""
    loss_fn = BackwardConsistencyLoss()
    edge_index = scaling_sequence[0].edge_index
    z_t = trainable_model.encoder(scaling_sequence[0], edge_index)
    z_t1 = trainable_model.encoder(scaling_sequence[1], edge_index)
    loss = loss_fn(z_t, z_t1, trainable_model.koopman)
    loss.backward()
    for param in trainable_model.encoder.parameters():
        assert param.grad is not None
    assert trainable_model.koopman.K.grad is not None


def test_compute_backward_consistency_sequence_loss(
    trainable_model: GraphKoopmanModel,
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Verify sequence-level backward consistency loss averages pairs."""
    loss = compute_backward_consistency_sequence_loss(trainable_model, scaling_sequence)
    assert loss.ndim == 0
    assert loss.item() > 0.0


def test_compute_training_loss_with_backward_term(
    trainable_model: GraphKoopmanModel,
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Verify combined loss includes backward term when weight > 0."""
    from koopman_graph.training import compute_sequence_loss

    weight = 1.0
    breakdown = compute_training_loss(
        trainable_model,
        scaling_sequence,
        constant_loss_weights(backward=weight),
    )
    recon = compute_sequence_loss(trainable_model, scaling_sequence)
    bc = compute_backward_consistency_sequence_loss(trainable_model, scaling_sequence)
    expected = recon + weight * bc
    assert torch.allclose(breakdown.total, expected)


def test_compute_training_loss_with_forward_and_backward_terms(
    trainable_model: GraphKoopmanModel,
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Verify combined loss sums reconstruction, forward, and backward terms."""
    from koopman_graph.training import compute_sequence_loss

    fw, bw = 0.5, 1.0
    breakdown = compute_training_loss(
        trainable_model,
        scaling_sequence,
        constant_loss_weights(forward=fw, backward=bw),
    )
    recon = compute_sequence_loss(trainable_model, scaling_sequence)
    fc = compute_forward_consistency_sequence_loss(trainable_model, scaling_sequence)
    bc = compute_backward_consistency_sequence_loss(trainable_model, scaling_sequence)
    expected = recon + fw * fc + bw * bc
    assert torch.allclose(breakdown.total, expected)


def test_compute_training_loss_with_reconstruction_weight(
    trainable_model: GraphKoopmanModel,
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Verify reconstruction weight scales the reconstruction term."""
    from koopman_graph.training import compute_sequence_loss

    weight = 0.5
    breakdown = compute_training_loss(
        trainable_model,
        scaling_sequence,
        constant_loss_weights(reconstruction=weight),
    )
    recon = compute_sequence_loss(trainable_model, scaling_sequence)
    assert torch.allclose(breakdown.total, weight * recon)


def test_compute_training_loss_requires_two_snapshots(
    trainable_model: GraphKoopmanModel,
    synthetic_graph: Data,
) -> None:
    """Verify combined training loss requires two snapshots."""
    sequence = GraphSnapshotSequence([synthetic_graph])
    with pytest.raises(ValueError, match="at least 2 snapshots"):
        compute_training_loss(
            trainable_model,
            sequence,
            constant_loss_weights(forward=1.0, backward=1.0),
        )


def test_rollout_sequence_loss_is_differentiable(
    trainable_model: GraphKoopmanModel,
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Verify rollout loss supports backpropagation."""
    loss = rollout_sequence_loss(trainable_model, scaling_sequence, horizon=2)
    loss.backward()
    for param in trainable_model.parameters():
        assert param.grad is not None


def test_rollout_sequence_loss_requires_valid_horizon(
    trainable_model: GraphKoopmanModel,
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Verify invalid rollout horizons raise clear errors."""
    with pytest.raises(ValueError, match="horizon"):
        rollout_sequence_loss(trainable_model, scaling_sequence, horizon=0)
    with pytest.raises(ValueError, match="too short"):
        rollout_sequence_loss(trainable_model, scaling_sequence, horizon=10)


def test_rollout_sequence_loss_rejects_negative_start(
    trainable_model: GraphKoopmanModel,
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Verify rollout loss rejects negative start indices."""
    with pytest.raises(ValueError, match="start"):
        rollout_sequence_loss(
            trainable_model,
            scaling_sequence,
            horizon=2,
            start=-1,
        )


def test_rollout_multi_start_loss_averages_origins(
    trainable_model: GraphKoopmanModel,
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Verify multi-start rollout loss averages valid origins."""
    from koopman_graph.losses import rollout_multi_start_loss, rollout_sequence_loss

    snapshots = [
        Data(x=torch.ones(5, 3) * (0.9**t), edge_index=synthetic_edge_index)
        for t in range(8)
    ]
    sequence = GraphSnapshotSequence(snapshots)
    multi = rollout_multi_start_loss(
        trainable_model,
        sequence,
        horizon=2,
        start_indices=[0, 2, 4],
    )
    expected = (
        rollout_sequence_loss(trainable_model, sequence, horizon=2, start=0)
        + rollout_sequence_loss(trainable_model, sequence, horizon=2, start=2)
        + rollout_sequence_loss(trainable_model, sequence, horizon=2, start=4)
    ) / 3
    assert torch.allclose(multi, expected)


def test_compute_training_loss_with_rollout_weight(
    trainable_model: GraphKoopmanModel,
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Verify rollout weight scales the rollout term."""
    weight = 0.25
    breakdown = compute_training_loss(
        trainable_model,
        scaling_sequence,
        constant_loss_weights(reconstruction=0.0, rollout=weight),
        rollout_horizon=2,
    )
    rollout = rollout_sequence_loss(trainable_model, scaling_sequence, horizon=2)
    assert torch.allclose(breakdown.total, weight * rollout)


def test_fit_with_backward_consistency_weight(
    trainable_model: GraphKoopmanModel,
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Verify fit accepts backward consistency weights and trains."""
    history = trainable_model.fit(
        scaling_sequence,
        epochs=3,
        lr=1e-2,
        loss_weights=constant_loss_weights(backward=1.0),
    )
    assert len(history.loss) == 3
    assert all(isinstance(value, float) for value in history.loss)


def test_backward_consistency_sequence_loss_requires_two_snapshots(
    trainable_model: GraphKoopmanModel,
    synthetic_graph: Data,
) -> None:
    """Verify sequence backward consistency loss requires two snapshots."""
    sequence = GraphSnapshotSequence([synthetic_graph])
    with pytest.raises(ValueError, match="at least 2 snapshots"):
        compute_backward_consistency_sequence_loss(trainable_model, sequence)


def test_eigenvalue_loss_zero_on_unit_circle_dense() -> None:
    """Verify eigenvalue hinge loss is zero for a unit-circle diagonal operator."""
    from koopman_graph.losses import EigenvalueRegularizationLoss

    koopman = KoopmanOperator(3, init_mode="identity")
    with torch.no_grad():
        koopman.K.copy_(torch.diag(torch.tensor([0.8, 1.0, -0.5])))
    loss_fn = EigenvalueRegularizationLoss()
    assert loss_fn(koopman).item() == pytest.approx(0.0, abs=1e-6)


def test_eigenvalue_loss_positive_outside_unit_circle_dense() -> None:
    """Verify eigenvalue hinge loss penalizes unstable eigenvalues."""
    from koopman_graph.losses import EigenvalueRegularizationLoss

    koopman = KoopmanOperator(2, init_mode="identity")
    with torch.no_grad():
        koopman.K.copy_(torch.diag(torch.tensor([1.5, 0.5])))
    loss_fn = EigenvalueRegularizationLoss()
    assert loss_fn(koopman).item() > 0.0


def test_eigenvalue_loss_zero_for_odo_within_bound() -> None:
    """Verify ODO eigenvalue loss is zero when diagonal stays within radius."""
    from koopman_graph.losses import EigenvalueRegularizationLoss

    koopman = KoopmanOperator(4, parameterization="odo", max_spectral_radius=0.8)
    loss_fn = EigenvalueRegularizationLoss()
    assert loss_fn(koopman).item() == pytest.approx(0.0, abs=1e-6)


def test_backward_consistency_dense_precomputes_inverse(
    trainable_model: GraphKoopmanModel,
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Verify dense backward sequence loss reuses one inverse matrix."""
    from unittest.mock import patch

    with patch.object(
        trainable_model.koopman,
        "dense_inverse_matrix",
        wraps=trainable_model.koopman.dense_inverse_matrix,
    ) as inverse_mock:
        loss = compute_backward_consistency_sequence_loss(
            trainable_model,
            scaling_sequence,
        )
    assert loss.ndim == 0
    assert inverse_mock.call_count == 1


def test_compute_training_loss_with_eigenvalue_term(
    trainable_model: GraphKoopmanModel,
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Verify combined loss includes eigenvalue term when weight > 0."""
    weights = constant_loss_weights(eigenvalue=0.5)
    breakdown = compute_training_loss(trainable_model, scaling_sequence, weights)
    assert breakdown.total.ndim == 0
    assert torch.isfinite(breakdown.total).item()
