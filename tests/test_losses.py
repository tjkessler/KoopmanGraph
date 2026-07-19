"""Tests for forward/backward consistency and related loss functions."""

import pytest
import torch
from torch_geometric.data import Data

from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.graph_utils import (
    autoregressive_latent_rollout,
    snapshot_topology_at,
)
from koopman_graph.losses import (
    BackwardConsistencyLoss,
    ForwardConsistencyLoss,
    KoopmanSparsityLoss,
    LieConsistencyLoss,
    PDEResidualLoss,
    WorstCaseReconstructionLoss,
    rollout_sequence_loss,
)
from koopman_graph.operators import (
    ContinuousKoopmanOperator,
    GraphKoopmanOperator,
    KoopmanOperator,
)
from koopman_graph.training import (
    ExtraLosses,
    LossWeights,
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


def test_masked_mse_loss_matches_hand_computed() -> None:
    """Masked MSE averages squared error over observed nodes × features."""
    from koopman_graph.losses import masked_mse_loss

    prediction = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )
    target = torch.tensor(
        [
            [1.0, 2.0],
            [0.0, 0.0],
            [7.0, 8.0],
        ]
    )
    mask = torch.tensor([False, True, True])
    # Observed nodes 1 and 2: diffs (3,4) and (-2,-2); sum sq = 9+16+4+4 = 33
    # denom = 2 nodes * 2 features = 4 → 33/4
    assert masked_mse_loss(prediction, target, mask).item() == pytest.approx(
        33.0 / 4.0,
        abs=1e-6,
    )


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


def test_eigenvalue_loss_discrete_dense_exact_hinge_value() -> None:
    """Discrete hinge mean(relu(|λ|-1)^2) is 0.125 for diag(1.5, 0.5)."""
    from koopman_graph.losses import EigenvalueRegularizationLoss

    koopman = KoopmanOperator(2, init_mode="identity")
    with torch.no_grad():
        koopman.K.copy_(torch.diag(torch.tensor([1.5, 0.5])))
    loss_fn = EigenvalueRegularizationLoss()
    assert loss_fn(koopman).item() == pytest.approx(0.125, abs=1e-6)


def test_eigenvalue_loss_zero_for_odo_within_bound() -> None:
    """Verify discrete ODO eigenvalue loss is zero when ρ(K) ≤ 1."""
    from koopman_graph.losses import EigenvalueRegularizationLoss

    koopman = KoopmanOperator(4, parameterization="odo", max_spectral_radius=0.8)
    loss_fn = EigenvalueRegularizationLoss()
    assert loss_fn(koopman).item() == pytest.approx(0.0, abs=1e-6)


def test_eigenvalue_loss_continuous_odo_uses_true_spectrum() -> None:
    """Continuous ODO eigenloss must penalize unstable assembled generators."""
    from koopman_graph.losses import EigenvalueRegularizationLoss
    from koopman_graph.operators import ContinuousKoopmanOperator

    torch.manual_seed(0)
    koopman = ContinuousKoopmanOperator(8, parameterization="odo")
    with torch.no_grad():
        koopman.cayley_O1.copy_(torch.randn(8, 8) * 2)
        koopman.cayley_O2.copy_(torch.randn(8, 8) * 2)
        koopman.diag_raw.copy_(torch.randn(8) * 3)

    assert koopman.max_real_part().item() > 0.0
    assert koopman.bound_metric().item() <= 0.0
    loss_fn = EigenvalueRegularizationLoss()
    penalty = loss_fn(koopman, dynamics_mode="continuous")
    assert penalty.item() > 0.0


def test_eigenvalue_loss_continuous_dense_uses_matrix_real_parts() -> None:
    """Continuous hinge uses eigvals(matrix).real, not concrete .L aliases."""
    from koopman_graph.losses import EigenvalueRegularizationLoss
    from koopman_graph.operators import ContinuousKoopmanOperator

    koopman = ContinuousKoopmanOperator(2, init_mode="identity")
    with torch.no_grad():
        koopman.L.copy_(torch.diag(torch.tensor([0.5, -1.0])))
    loss_fn = EigenvalueRegularizationLoss()
    assert loss_fn(koopman, dynamics_mode="continuous").item() > 0.0
    assert loss_fn(koopman, dynamics_mode="continuous").item() == pytest.approx(
        0.125,
        abs=1e-5,
    )


def test_eigenvalue_loss_dissipative_zero_continuous() -> None:
    """Dissipative continuous operators keep a zero eigenvalue penalty."""
    from koopman_graph.losses import EigenvalueRegularizationLoss
    from koopman_graph.operators import ContinuousKoopmanOperator

    koopman = ContinuousKoopmanOperator(3, parameterization="dissipative")
    loss_fn = EigenvalueRegularizationLoss()
    assert loss_fn(koopman, dynamics_mode="continuous").item() == pytest.approx(
        0.0,
        abs=1e-8,
    )


def test_eigenvalue_loss_graph_dense_requires_topology() -> None:
    """Graph dense/ODO eigenvalue loss never falls back to K_self alone."""
    from koopman_graph.losses import EigenvalueRegularizationLoss
    from koopman_graph.operators import GraphKoopmanOperator

    op = GraphKoopmanOperator(2, init_mode="identity")
    loss_fn = EigenvalueRegularizationLoss()
    with pytest.raises(ValueError, match="edge_index and num_nodes are required"):
        loss_fn(op)


def test_eigenvalue_loss_graph_neighbor_coupling_affects_penalty() -> None:
    """Fixed K_self with larger K_nbr increases the effective-operator hinge."""
    from koopman_graph.losses import EigenvalueRegularizationLoss
    from koopman_graph.operators import GraphKoopmanOperator

    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    k_self = 0.5 * torch.eye(2)
    loss_fn = EigenvalueRegularizationLoss()

    mild = GraphKoopmanOperator(2, init_mode="identity")
    mild.set_dense_matrices(k_self, 0.1 * torch.eye(2))
    strong = GraphKoopmanOperator(2, init_mode="identity")
    strong.set_dense_matrices(k_self.clone(), 2.0 * torch.eye(2))

    mild_loss = loss_fn(mild, edge_index=edge_index, num_nodes=2)
    strong_loss = loss_fn(strong, edge_index=edge_index, num_nodes=2)
    assert mild_loss.item() == pytest.approx(0.0, abs=1e-6)
    assert strong_loss.item() > mild_loss.item()


def test_eigenvalue_loss_graph_edge_weight_affects_penalty() -> None:
    """Edge weights change graph dense regularization with fixed K factors."""
    from koopman_graph.losses import EigenvalueRegularizationLoss
    from koopman_graph.operators import GraphKoopmanOperator

    # Triangle: unequal weights change Â eigenvalues (unlike a weighted path).
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]],
        dtype=torch.long,
    )
    op = GraphKoopmanOperator(2, init_mode="identity")
    op.set_dense_matrices(0.2 * torch.eye(2), 1.5 * torch.eye(2))
    loss_fn = EigenvalueRegularizationLoss()

    equal = loss_fn(
        op,
        edge_index=edge_index,
        num_nodes=3,
        edge_weight=torch.ones(6),
    )
    unequal = loss_fn(
        op,
        edge_index=edge_index,
        num_nodes=3,
        edge_weight=torch.tensor([1.0, 1.0, 1.0, 1.0, 0.01, 0.01]),
    )
    assert equal.item() != pytest.approx(unequal.item(), abs=1e-4)


def test_eigenvalue_loss_graph_structural_uses_factor_bound() -> None:
    """Graph structural modes keep factor bound_metric without topology."""
    from koopman_graph.losses import EigenvalueRegularizationLoss
    from koopman_graph.operators import GraphKoopmanOperator

    op = GraphKoopmanOperator(3, parameterization="dissipative")
    loss_fn = EigenvalueRegularizationLoss()
    assert loss_fn(op).item() == pytest.approx(0.0, abs=1e-8)


def test_compute_eigenvalue_regularization_graph_requires_sequence() -> None:
    """Training helper refuses graph dense regularization without a sequence."""
    from koopman_graph.operators import GraphKoopmanOperator
    from koopman_graph.training import compute_eigenvalue_regularization_loss

    encoder = GNNEncoder(in_channels=2, hidden_channels=4, latent_dim=2, num_layers=1)
    decoder = GNNDecoder(latent_dim=2, hidden_channels=4, out_channels=2, num_layers=1)
    model = GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=2,
        time_step=0.1,
        koopman="graph",
    )
    assert isinstance(model.koopman, GraphKoopmanOperator)
    with pytest.raises(ValueError, match="sequence is required"):
        compute_eigenvalue_regularization_loss(model)


def test_backward_consistency_graph_bilinear_global_and_per_node(
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Backward consistency accepts global and per-node graph bilinear controls."""
    encoder = GNNEncoder(in_channels=3, hidden_channels=4, latent_dim=2, num_layers=1)
    decoder = GNNDecoder(latent_dim=2, hidden_channels=4, out_channels=3, num_layers=1)
    model = GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=2,
        time_step=0.1,
        koopman="graph",
        control_dim=1,
        control_mode="bilinear",
    )
    snapshots = [
        Data(x=torch.randn(5, 3), edge_index=synthetic_edge_index) for _ in range(3)
    ]
    global_seq = GraphSnapshotSequence(
        snapshots, control_inputs=torch.randn(3, 1)
    )
    per_node_seq = GraphSnapshotSequence(
        snapshots, control_inputs=torch.randn(3, 5, 1)
    )
    for sequence in (global_seq, per_node_seq):
        loss = compute_backward_consistency_sequence_loss(model, sequence)
        assert loss.ndim == 0
        assert torch.isfinite(loss).item()
        loss.backward()
        model.zero_grad(set_to_none=True)


def test_compute_training_loss_graph_eigenvalue_uses_topology(
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Training loss threads sequence topology into graph eigenvalue hinge."""
    from koopman_graph.operators import GraphKoopmanOperator
    from koopman_graph.training import compute_eigenvalue_regularization_loss

    encoder = GNNEncoder(in_channels=3, hidden_channels=4, latent_dim=2, num_layers=1)
    decoder = GNNDecoder(latent_dim=2, hidden_channels=4, out_channels=3, num_layers=1)
    model = GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=2,
        time_step=0.1,
        koopman="graph",
    )
    assert isinstance(model.koopman, GraphKoopmanOperator)
    model.koopman.set_dense_matrices(0.5 * torch.eye(2), 2.0 * torch.eye(2))
    snapshots = [
        Data(x=torch.randn(5, 3), edge_index=synthetic_edge_index) for _ in range(3)
    ]
    sequence = GraphSnapshotSequence(snapshots)
    eig = compute_eigenvalue_regularization_loss(model, sequence)
    assert eig.item() > 0.0
    breakdown = compute_training_loss(
        model,
        sequence,
        constant_loss_weights(eigenvalue=1.0),
    )
    assert breakdown.eigenvalue.item() == pytest.approx(eig.item(), abs=1e-6)


def test_compute_eigenvalue_regularization_dynamic_pair_mean(
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Dynamic sequences average pair-target topology hinges."""
    from koopman_graph.losses import EigenvalueRegularizationLoss
    from koopman_graph.operators import GraphKoopmanOperator
    from koopman_graph.training import compute_eigenvalue_regularization_loss

    encoder = GNNEncoder(in_channels=3, hidden_channels=4, latent_dim=2, num_layers=1)
    decoder = GNNDecoder(latent_dim=2, hidden_channels=4, out_channels=3, num_layers=1)
    model = GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=2,
        time_step=0.1,
        koopman="graph",
    )
    assert isinstance(model.koopman, GraphKoopmanOperator)
    model.koopman.set_dense_matrices(0.2 * torch.eye(2), 1.5 * torch.eye(2))

    alt_edges = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]],
        dtype=torch.long,
    )
    snapshots = [
        Data(x=torch.randn(5, 3), edge_index=synthetic_edge_index),
        Data(x=torch.randn(5, 3), edge_index=synthetic_edge_index),
        Data(x=torch.randn(5, 3), edge_index=alt_edges),
    ]
    sequence = GraphSnapshotSequence(snapshots, allow_dynamic_topology=True)
    loss_fn = EigenvalueRegularizationLoss()
    expected = 0.5 * (
        loss_fn(
            model.koopman,
            edge_index=synthetic_edge_index,
            num_nodes=5,
        )
        + loss_fn(model.koopman, edge_index=alt_edges, num_nodes=5)
    )
    got = compute_eigenvalue_regularization_loss(model, sequence)
    assert got.item() == pytest.approx(expected.item(), abs=1e-5)


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


def test_backward_consistency_without_dense_inverse_helper(
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Verify Protocol operators without ``dense_inverse_matrix`` still train."""
    from torch import nn

    from koopman_graph.model import GraphKoopmanModel
    from koopman_graph.nn import GNNDecoder, GNNEncoder

    class _ContractOnlyOperator(nn.Module):
        def __init__(self, latent_dim: int) -> None:
            super().__init__()
            self.latent_dim = latent_dim
            self.control_dim = 0
            self.parameterization = "dense"
            self._matrix = nn.Parameter(torch.eye(latent_dim))

        @property
        def matrix(self) -> torch.Tensor:
            return self._matrix

        def advance(
            self,
            z: torch.Tensor,
            delta_t: float | torch.Tensor | None = None,
            *,
            control: torch.Tensor | None = None,
        ) -> torch.Tensor:
            del delta_t, control
            return z @ self._matrix.T

        def inverse_advance(
            self,
            z: torch.Tensor,
            delta_t: float | torch.Tensor | None = None,
            *,
            control: torch.Tensor | None = None,
            inverse_matrix: torch.Tensor | None = None,
        ) -> torch.Tensor:
            del delta_t, control
            inv = (
                inverse_matrix
                if inverse_matrix is not None
                else torch.linalg.pinv(self._matrix)
            )
            return z @ inv.T

        def bound_metric(self) -> torch.Tensor:
            return torch.linalg.eigvals(self._matrix).abs().max()

    encoder = GNNEncoder(in_channels=3, hidden_channels=4, latent_dim=4, num_layers=1)
    decoder = GNNDecoder(latent_dim=4, hidden_channels=4, out_channels=3, num_layers=1)
    model = GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=4,
        time_step=0.1,
        koopman=_ContractOnlyOperator(4),
    )
    assert not hasattr(model.koopman, "dense_inverse_matrix")
    loss = compute_backward_consistency_sequence_loss(model, scaling_sequence)
    assert loss.ndim == 0
    assert torch.isfinite(loss).item()


def test_compute_training_loss_with_eigenvalue_term(
    trainable_model: GraphKoopmanModel,
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Verify combined loss includes eigenvalue term when weight > 0."""
    weights = constant_loss_weights(eigenvalue=0.5)
    breakdown = compute_training_loss(trainable_model, scaling_sequence, weights)
    assert breakdown.total.ndim == 0
    assert torch.isfinite(breakdown.total).item()


def test_rollout_multi_start_loss_rejects_empty_origins(
    trainable_model: GraphKoopmanModel,
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Verify multi-start rollout loss requires at least one origin."""
    from koopman_graph.losses import rollout_multi_start_loss

    with pytest.raises(ValueError, match="at least one origin"):
        rollout_multi_start_loss(
            trainable_model,
            scaling_sequence,
            horizon=2,
            start_indices=[],
        )


def test_rollout_sequence_loss_uses_sequence_controls(
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Verify controlled models draw rollout controls from the sequence."""
    encoder = GNNEncoder(in_channels=3, hidden_channels=8, latent_dim=4)
    decoder = GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=3)
    model = GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=4,
        time_step=0.1,
        control_dim=1,
    )
    snapshots = [
        Data(x=torch.randn(5, 3), edge_index=synthetic_edge_index) for _ in range(4)
    ]
    sequence = GraphSnapshotSequence(snapshots, control_inputs=torch.randn(4, 1))

    loss = rollout_sequence_loss(model, sequence, horizon=2)

    assert loss.ndim == 0
    assert torch.isfinite(loss).item()


def test_predict_and_rollout_loss_agree_on_static_topology(
    trainable_model: GraphKoopmanModel,
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Verify predict and training rollout share predictions when topologies align.

    On a static-topology sequence, hold-last inference topology matches teacher
    target edges, so decoded rollouts must agree within numerical tolerance.
    """
    horizon = 2
    start = 0
    trainable_model.eval()
    with torch.no_grad():
        predicted = trainable_model.predict(scaling_sequence[start], steps=horizon)
        z = trainable_model.encode(scaling_sequence[start])
        targets = [scaling_sequence[start + step] for step in range(1, horizon + 1)]
        rollout = autoregressive_latent_rollout(
            trainable_model.koopman,
            trainable_model.decoder,
            z,
            steps=horizon,
            topology_at=snapshot_topology_at(targets),
            default_delta_t=trainable_model.time_step,
        )

        total = torch.zeros(())
        for step, (prediction, _, _) in enumerate(rollout):
            assert torch.allclose(prediction, predicted[step].x, atol=1e-6)
            total = total + torch.nn.functional.mse_loss(
                prediction,
                targets[step].x,
            )
        expected_loss = total / horizon
        loss = rollout_sequence_loss(
            trainable_model,
            scaling_sequence,
            horizon=horizon,
            start=start,
        )
        assert torch.allclose(loss, expected_loss, atol=1e-6)


def test_lie_consistency_is_zero_for_exact_linear_vector_field() -> None:
    """An identity observable exactly matches its continuous linear generator."""
    generator = torch.tensor([[-0.2, 1.0], [-1.0, -0.2]])
    operator = ContinuousKoopmanOperator(2, init_mode="identity")
    with torch.no_grad():
        operator.L.copy_(generator)
    state = torch.randn(5, 2, requires_grad=True)

    loss = LieConsistencyLoss()(
        state,
        observable_fn=lambda value: value,
        dynamics_fn=lambda value: value @ generator.T,
        koopman=operator,
    )

    assert loss.item() == pytest.approx(0.0, abs=1e-10)


def test_lie_consistency_rejects_wrong_vector_field_shape() -> None:
    """The known vector field must preserve the physical-state shape."""
    operator = ContinuousKoopmanOperator(2)
    with pytest.raises(ValueError, match="dynamics_fn output"):
        LieConsistencyLoss()(
            torch.randn(3, 2),
            observable_fn=lambda value: value,
            dynamics_fn=lambda value: value[:, :1],
            koopman=operator,
        )


def test_lie_consistency_rejects_controlled_operator() -> None:
    """Autonomous Lie consistency must not silently discard control terms."""
    operator = ContinuousKoopmanOperator(2, control_dim=1)
    with pytest.raises(ValueError, match="uncontrolled"):
        LieConsistencyLoss()(
            torch.randn(3, 2),
            observable_fn=lambda value: value,
            dynamics_fn=lambda value: -value,
            koopman=operator,
        )


def test_pde_residual_loss_distinguishes_exact_and_wrong_residuals() -> None:
    """A satisfied residual is zero while an incorrect PDE remains positive."""
    decoded = torch.ones(4, 1)
    snapshot = Data(
        x=decoded.clone(),
        edge_index=torch.empty((2, 0), dtype=torch.long),
    )
    loss_fn = PDEResidualLoss()

    exact = loss_fn(
        decoded,
        snapshot,
        pde_fn=lambda prediction, context: prediction - context.x,
    )
    wrong = loss_fn(
        decoded,
        snapshot,
        pde_fn=lambda prediction, context: prediction + context.x,
    )

    assert exact.item() == pytest.approx(0.0)
    assert wrong.item() == pytest.approx(4.0)


def test_lie_consistency_is_wired_into_training_loss() -> None:
    """Continuous training composes the fit-time vector field and Lie weight."""
    encoder = GNNEncoder(in_channels=2, hidden_channels=4, latent_dim=2)
    decoder = GNNDecoder(latent_dim=2, hidden_channels=4, out_channels=2)
    model = GraphKoopmanModel(
        encoder,
        decoder,
        latent_dim=2,
        dynamics_mode="continuous",
        time_step=0.1,
    )
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    sequence = GraphSnapshotSequence(
        [
            Data(x=torch.randn(2, 2), edge_index=edge_index)
            for _ in range(2)
        ]
    )

    breakdown = compute_training_loss(
        model,
        sequence,
        LossWeights(reconstruction=0.0, lie=1.0),
        extra_losses=ExtraLosses(
            lie_dynamics_fn=lambda snapshot: -snapshot.x,
        ),
    )
    breakdown.total.backward()

    assert breakdown.lie.ndim == 0
    assert torch.isfinite(breakdown.lie).item()
    assert any(parameter.grad is not None for parameter in model.parameters())


def _sparsity_fraction(matrix: torch.Tensor, *, threshold: float = 1e-2) -> float:
    """Return the fraction of entries with absolute value below ``threshold``."""
    entries = matrix.detach().reshape(-1).abs()
    return float((entries < threshold).float().mean().item())


def test_koopman_sparsity_loss_l1_mean_absolute() -> None:
    """Pure L1 sparsity equals the mean absolute entry of ``K``."""
    koopman = KoopmanOperator(3, init_mode="identity")
    with torch.no_grad():
        koopman.K.copy_(
            torch.tensor(
                [[1.0, -2.0, 0.0], [0.5, 0.0, -0.5], [0.0, 0.0, 0.25]],
            )
        )
    loss = KoopmanSparsityLoss()(koopman)
    expected = koopman.matrix.abs().mean()
    assert loss.item() == pytest.approx(expected.item(), abs=1e-6)


def test_koopman_sparsity_loss_smoothed_lp_positive() -> None:
    """Smoothed Lp (p<1) is finite and positive for a nonzero dense matrix."""
    koopman = KoopmanOperator(2, init_mode="identity_noise")
    loss = KoopmanSparsityLoss(p=0.5, eps=1e-6)(koopman)
    assert loss.ndim == 0
    assert loss.item() > 0.0
    assert torch.isfinite(loss).item()


def test_koopman_sparsity_loss_rejects_invalid_p() -> None:
    """Sparsity exponent must lie in (0, 1]."""
    with pytest.raises(ValueError, match="p must be in"):
        KoopmanSparsityLoss(p=1.5)


def test_koopman_sparsity_loss_graph_targets_self_and_nbr_not_effective() -> None:
    """Graph sparsity penalizes ``K_self``/``K_nbr``, not ``effective_matrix``."""
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    other_edges = torch.tensor([[0, 1, 0], [1, 0, 0]], dtype=torch.long)
    koopman = GraphKoopmanOperator(2, init_mode="identity")
    with torch.no_grad():
        koopman.set_dense_matrices(
            torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
            torch.tensor([[0.0, 2.0], [0.0, 0.0]]),
        )
    loss = KoopmanSparsityLoss()(koopman)
    factor_entries = torch.cat(
        (koopman.K_self.reshape(-1), koopman.K_nbr.reshape(-1)),
        dim=0,
    )
    expected = factor_entries.abs().mean()
    effective = koopman.effective_matrix(edge_index, num_nodes=2)
    assert loss.item() == pytest.approx(expected.item(), abs=1e-6)
    # Factor stack is 2 d² entries; effective is (N d)² — different tensors.
    assert factor_entries.numel() == 8
    assert effective.numel() == 16
    assert factor_entries.abs().sum().item() != pytest.approx(
        effective.abs().sum().item(),
        abs=1e-6,
    )
    # Topology must not affect the sparsity target (parameter-level only).
    assert KoopmanSparsityLoss()(koopman).item() == pytest.approx(loss.item(), abs=1e-6)
    assert not torch.allclose(
        effective,
        koopman.effective_matrix(other_edges, num_nodes=2),
    )


def test_worst_case_reconstruction_loss_is_max_node_mse() -> None:
    """Worst-case loss matches the max over per-node mean squared errors."""
    prediction = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    target = torch.zeros_like(prediction)
    loss = WorstCaseReconstructionLoss()(prediction, target)
    node_mse = (prediction - target).square().mean(dim=-1)
    assert loss.item() == pytest.approx(node_mse.max().item(), abs=1e-6)


def test_worst_case_reconstruction_loss_respects_mask() -> None:
    """Masked worst-case ignores unobserved nodes."""
    prediction = torch.tensor([[10.0], [0.5], [0.25]])
    target = torch.zeros_like(prediction)
    mask = torch.tensor([False, True, True])
    loss = WorstCaseReconstructionLoss()(prediction, target, mask)
    assert loss.item() == pytest.approx(0.25, abs=1e-6)


def test_compute_training_loss_includes_sparsity_and_worst_case(
    trainable_model: GraphKoopmanModel,
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Non-zero sparsity / worst-case weights appear in the breakdown total."""
    weights = LossWeights(reconstruction=1.0, sparsity=0.5, worst_case=0.25)
    breakdown = compute_training_loss(trainable_model, scaling_sequence, weights)
    assert breakdown.sparsity.item() > 0.0
    assert breakdown.worst_case.item() > 0.0
    expected_total = (
        weights.reconstruction * breakdown.reconstruction
        + weights.sparsity * breakdown.sparsity
        + weights.worst_case * breakdown.worst_case
    )
    assert breakdown.total.item() == pytest.approx(expected_total.item(), abs=1e-5)


def test_sparsity_fraction_increases_with_weight(
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Seeded sweep: larger ``LossWeights.sparsity`` yields a sparser ``K``."""
    fractions: list[float] = []
    for sparsity_weight in (0.0, 0.5, 2.0):
        torch.manual_seed(0)
        encoder = GNNEncoder(in_channels=3, hidden_channels=16, latent_dim=8)
        decoder = GNNDecoder(latent_dim=8, hidden_channels=16, out_channels=3)
        model = GraphKoopmanModel(
            encoder=encoder,
            decoder=decoder,
            latent_dim=8,
            time_step=0.1,
        )
        model.fit(
            scaling_sequence,
            epochs=40,
            lr=5e-2,
            loss_weights=LossWeights(reconstruction=1.0, sparsity=sparsity_weight),
        )
        fractions.append(_sparsity_fraction(model.koopman.matrix, threshold=1e-2))

    assert fractions[0] < fractions[1] < fractions[2]


def test_worst_case_is_robust_training_term_not_generalization_claim() -> None:
    """Docstring states worst-case loss is not a generalization certificate."""
    doc = WorstCaseReconstructionLoss.__doc__ or ""
    assert "not" in doc.lower()
    assert "generalization" in doc.lower()
