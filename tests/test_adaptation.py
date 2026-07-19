"""Tests for recursive Koopman online adaptation."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data

from koopman_graph import (
    GNNDecoder,
    GNNEncoder,
    GraphKoopmanModel,
    GraphSnapshotSequence,
)
from koopman_graph.adaptation import AdaptationStepResult, RecursiveKoopmanAdapter
from koopman_graph.operators import (
    VAN_LOAN_WRITEBACK_ATOL,
    ContinuousKoopmanOperator,
    KoopmanOperator,
)
from koopman_graph.serialization import snapshot_state_dict


def _latent_rollout(
    operator: torch.Tensor,
    input_matrix: torch.Tensor | None,
    controls: list[torch.Tensor] | None,
    initial: torch.Tensor,
    steps: int,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]]:
    """Generate latent transition triples for a linear system."""
    pairs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]] = []
    state = initial
    for step in range(steps):
        control = None if controls is None else controls[step]
        if control is None:
            nxt = state @ operator.T
        else:
            nxt = state @ operator.T + control @ input_matrix
        pairs.append((state.clone(), nxt.clone(), control))
        state = nxt
    return pairs


def _state_dicts_equal(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
) -> bool:
    """Return whether two state dictionaries contain identical tensors."""
    if left.keys() != right.keys():
        return False
    return all(torch.equal(left[key], right[key]) for key in left)


def test_rls_recovers_known_discrete_operator() -> None:
    """RLS should converge to a known dense K from streaming latent pairs."""
    torch.manual_seed(0)
    latent_dim = 4
    true_k = torch.tensor(
        [
            [0.85, 0.05, 0.0, 0.0],
            [0.02, 0.80, 0.03, 0.0],
            [0.0, 0.04, 0.78, 0.02],
            [0.0, 0.0, 0.05, 0.82],
        ]
    )
    adapter = RecursiveKoopmanAdapter(
        latent_dim,
        forgetting_factor=1.0,
        regularization=10.0,
        initial_k=torch.eye(latent_dim),
    )
    for _ in range(400):
        z_t = torch.randn(latent_dim)
        z_tp1 = z_t @ true_k.T
        adapter.update(z_t, z_tp1)

    error = torch.linalg.norm(adapter.discrete_matrix - true_k)
    assert error < 0.1


def test_rls_recovers_known_controlled_operator() -> None:
    """RLS should recover both K and B for controlled latent dynamics."""
    torch.manual_seed(1)
    latent_dim = 3
    true_k = torch.diag(torch.tensor([0.8, 0.75, 0.7]))
    true_b = torch.tensor([[0.2, -0.1, 0.05]])
    adapter = RecursiveKoopmanAdapter(
        latent_dim,
        control_dim=1,
        forgetting_factor=1.0,
        regularization=10.0,
        initial_k=torch.eye(latent_dim),
        initial_b=torch.zeros(1, latent_dim),
    )
    for step in range(400):
        z_t = torch.randn(latent_dim)
        control = torch.tensor([0.5 * ((-1) ** step)])
        z_tp1 = z_t @ true_k.T + control @ true_b
        adapter.update(z_t, z_tp1, control=control)

    k_error = torch.linalg.norm(adapter.discrete_matrix - true_k)
    b_error = torch.linalg.norm(adapter.control_matrix - true_b)  # type: ignore[arg-type]
    assert k_error < 0.15
    assert b_error < 0.15


def test_forgetting_factor_tracks_drifting_operator() -> None:
    """A lower forgetting factor should track an abrupt K change faster."""
    latent_dim = 3
    k_before = torch.diag(torch.tensor([0.8, 0.75, 0.7]))
    k_after = torch.diag(torch.tensor([0.55, 0.5, 0.45]))
    initial = torch.tensor([1.0, 0.5, -0.2])

    stale_adapter = RecursiveKoopmanAdapter(
        latent_dim,
        forgetting_factor=1.0,
        regularization=10.0,
        initial_k=k_before.clone(),
    )
    adaptive_adapter = RecursiveKoopmanAdapter(
        latent_dim,
        forgetting_factor=0.9,
        regularization=10.0,
        initial_k=k_before.clone(),
    )

    for z_t, z_tp1, _ in _latent_rollout(k_before, None, None, initial, steps=120):
        stale_adapter.update(z_t, z_tp1)
        adaptive_adapter.update(z_t, z_tp1)

    for z_t, z_tp1, _ in _latent_rollout(k_after, None, None, initial, steps=120):
        adaptive_adapter.update(z_t, z_tp1)

    stale_error = torch.linalg.norm(stale_adapter.discrete_matrix - k_after)
    adaptive_error = torch.linalg.norm(adaptive_adapter.discrete_matrix - k_after)
    assert adaptive_error < stale_error


def test_continuous_adapter_recovers_generator() -> None:
    """Continuous-mode RLS should recover a known Hurwitz generator."""
    torch.manual_seed(2)
    latent_dim = 3
    true_l = torch.diag(torch.tensor([-0.12, -0.18, -0.15]))
    delta_t = 0.1
    true_k = torch.linalg.matrix_exp(true_l * delta_t)
    adapter = RecursiveKoopmanAdapter(
        latent_dim,
        mode="continuous",
        forgetting_factor=1.0,
        regularization=10.0,
        initial_l=torch.zeros(latent_dim, latent_dim),
    )
    for _ in range(400):
        z_t = torch.randn(latent_dim)
        z_tp1 = z_t @ true_k.T
        adapter.update(z_t, z_tp1, delta_t=delta_t)

    l_error = torch.linalg.norm(adapter.generator_matrix - true_l)
    assert l_error < 0.2


def test_apply_to_writes_dense_operator() -> None:
    """apply_to should update dense Koopman parameters in place."""
    operator = KoopmanOperator(3, parameterization="dense", init_mode="identity")
    adapter = RecursiveKoopmanAdapter(
        3,
        initial_k=torch.diag(torch.tensor([0.5, 0.6, 0.7])),
    )
    adapter.apply_to(operator)
    assert torch.allclose(operator.K, adapter.discrete_matrix, atol=1e-6)


def test_apply_to_writes_continuous_generator() -> None:
    """apply_to should update dense generator parameters in place."""
    operator = ContinuousKoopmanOperator(
        3,
        parameterization="dense",
        init_mode="identity",
    )
    target_l = torch.diag(torch.tensor([-0.1, -0.2, -0.3]))
    adapter = RecursiveKoopmanAdapter(
        3,
        mode="continuous",
        initial_l=target_l,
    )
    adapter.apply_to(operator)
    assert torch.linalg.norm(operator.L - target_l) < 1e-5


def test_continuous_uncontrolled_writeback_matches_advance() -> None:
    """Uncontrolled continuous write-back should match matrix-exp advance."""
    torch.manual_seed(3)
    latent_dim = 3
    delta_t = 0.2
    true_l = torch.tensor(
        [
            [-0.4, 0.1, 0.0],
            [0.05, -0.35, 0.08],
            [0.0, 0.02, -0.3],
        ]
    )
    source = ContinuousKoopmanOperator(latent_dim, parameterization="dense")
    source.set_dense_matrix(true_l)

    adapter = RecursiveKoopmanAdapter(
        latent_dim,
        mode="continuous",
        forgetting_factor=1.0,
        regularization=1.0,
        initial_l=true_l,
    )
    # Exact discrete seed at the reference interval, then write back.
    adapter._set_from_generator(true_l, None, delta_t=delta_t)
    target = ContinuousKoopmanOperator(latent_dim, parameterization="dense")
    adapter.apply_to(target)

    assert torch.allclose(target.L, true_l, atol=VAN_LOAN_WRITEBACK_ATOL)
    z = torch.randn(4, latent_dim)
    assert torch.allclose(
        source.advance(z, delta_t),
        target.advance(z, delta_t),
        atol=VAN_LOAN_WRITEBACK_ATOL,
    )


def test_continuous_controlled_writeback_matches_van_loan_advance() -> None:
    """Controlled continuous write-back should match Van Loan advance."""
    torch.manual_seed(4)
    latent_dim = 3
    control_dim = 2
    delta_t = 0.25
    true_l = torch.tensor(
        [
            [-0.5, 0.1, 0.0],
            [0.05, -0.4, 0.08],
            [0.0, 0.02, -0.35],
        ]
    )
    true_b = torch.tensor(
        [
            [0.2, -0.1, 0.05],
            [0.0, 0.15, -0.05],
        ]
    )
    source = ContinuousKoopmanOperator(
        latent_dim,
        control_dim=control_dim,
        parameterization="dense",
    )
    source.set_dense_matrix(true_l, control_matrix=true_b)

    adapter = RecursiveKoopmanAdapter(
        latent_dim,
        control_dim=control_dim,
        mode="continuous",
        forgetting_factor=1.0,
        regularization=1.0,
        initial_l=true_l,
        initial_b=true_b,
    )
    adapter._set_from_generator(true_l, true_b, delta_t=delta_t)
    target = ContinuousKoopmanOperator(
        latent_dim,
        control_dim=control_dim,
        parameterization="dense",
    )
    adapter.apply_to(target)

    assert torch.allclose(target.L, true_l, atol=VAN_LOAN_WRITEBACK_ATOL)
    assert torch.allclose(target.B, true_b, atol=VAN_LOAN_WRITEBACK_ATOL)

    z = torch.randn(5, latent_dim)
    control = torch.randn(5, control_dim)
    assert torch.allclose(
        source.advance(z, delta_t, control=control),
        target.advance(z, delta_t, control=control),
        atol=VAN_LOAN_WRITEBACK_ATOL,
    )

    # Fitted discrete blocks must reproduce the same one-step map.
    z_disc = z @ adapter.discrete_matrix.T + control @ adapter.control_matrix
    assert torch.allclose(
        source.advance(z, delta_t, control=control),
        z_disc,
        atol=VAN_LOAN_WRITEBACK_ATOL,
    )


def test_set_dense_matrix_rejects_structured_parameterization() -> None:
    """set_dense_matrix should require dense parameterization."""
    operator = KoopmanOperator(3, parameterization="schur")
    try:
        operator.set_dense_matrix(torch.eye(3))
    except ValueError as exc:
        assert "dense" in str(exc)
    else:
        raise AssertionError("expected ValueError for structured parameterization")


def test_structured_parameterization_rejected() -> None:
    """Online adaptation should require dense Koopman parameterization."""
    operator = KoopmanOperator(3, parameterization="lyapunov")
    try:
        RecursiveKoopmanAdapter.from_operator(
            operator,
            mode="discrete",
        )
    except ValueError as exc:
        assert "dense" in str(exc)
    else:
        raise AssertionError("expected ValueError for structured parameterization")


def _two_node_edge_index() -> torch.Tensor:
    """Return a minimal two-node bidirectional edge index."""
    return torch.tensor([[0, 1], [1, 0]], dtype=torch.long)


def test_adaptation_step_result_is_frozen() -> None:
    """Verify ``AdaptationStepResult`` is a frozen dataclass."""
    result = AdaptationStepResult(operator_change_norm=torch.tensor(0.25))
    assert float(result.operator_change_norm.item()) == pytest.approx(0.25)
    with pytest.raises(AttributeError):
        result.operator_change_norm = torch.tensor(1.0)  # type: ignore[misc]


def test_model_adapt_step_preserves_encoder() -> None:
    """adapt_step should not mutate encoder/decoder weights or gradients."""
    edge_index = _two_node_edge_index()
    encoder = GNNEncoder(in_channels=2, hidden_channels=8, latent_dim=4, num_layers=2)
    decoder = GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=2, num_layers=2)
    model = GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=4,
        time_step=0.1,
        koopman_parameterization="dense",
    )

    snapshots = [
        Data(x=torch.randn(2, 2), edge_index=edge_index),
        Data(x=torch.randn(2, 2), edge_index=edge_index),
    ]
    model.enable_online_adaptation(forgetting_factor=0.95, regularization=50.0)
    encoder_snapshot = snapshot_state_dict(model.encoder)
    decoder_snapshot = snapshot_state_dict(model.decoder)

    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(not parameter.requires_grad for parameter in model.decoder.parameters())

    step = model.adapt_step(snapshots[0], snapshots[1])
    assert isinstance(step, AdaptationStepResult)
    assert torch.isfinite(step.operator_change_norm).item()
    with pytest.raises(AttributeError):
        step.operator_change_norm = torch.tensor(0.0)  # type: ignore[misc]

    assert _state_dicts_equal(snapshot_state_dict(model.encoder), encoder_snapshot)
    assert _state_dicts_equal(snapshot_state_dict(model.decoder), decoder_snapshot)
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())

    model.encoder.train()
    loss = model(snapshots[0]).sum()
    loss.backward()
    assert all(
        parameter.grad is None or torch.all(parameter.grad == 0)
        for parameter in model.encoder.parameters()
    )


def test_model_enable_online_adaptation_rejects_structured_operator() -> None:
    """GraphKoopmanModel should reject online adaptation for structured K."""
    encoder = GNNEncoder(in_channels=2, hidden_channels=8, latent_dim=4, num_layers=2)
    decoder = GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=2, num_layers=2)
    model = GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=4,
        time_step=0.1,
        koopman_parameterization="schur",
    )
    try:
        model.enable_online_adaptation()
    except ValueError as exc:
        assert "dense" in str(exc)
    else:
        raise AssertionError("expected ValueError for structured parameterization")


def _identity_encode(model, x_or_data, edge_index=None, edge_weight=None):
    """Treat node features as latent states (synthetic linear identity lifting)."""
    from koopman_graph.graph_utils import resolve_edge_index

    _ = edge_weight
    _ = resolve_edge_index(x_or_data, edge_index)
    if isinstance(x_or_data, Data):
        return x_or_data.x
    return x_or_data


def _identity_decode(z, edge_index, edge_weight=None):
    """Identity decoder for synthetic linear tests.

    Assigned to ``module.forward`` (instance attribute); PyTorch ``__call__``
    invokes it without re-binding ``self``.
    """
    _ = edge_index
    _ = edge_weight
    return z


def _patch_identity_io(model: GraphKoopmanModel) -> None:
    """Make encode/decode identity maps for synthetic linear KF tests."""
    model.encode = (  # type: ignore[method-assign]
        lambda *args, **kwargs: _identity_encode(model, *args, **kwargs)
    )
    model.decoder.forward = _identity_decode  # type: ignore[method-assign]


def _linear_latent_sequence(
    true_k: torch.Tensor,
    *,
    num_nodes: int = 2,
    steps: int = 12,
    process_std: float = 0.0,
    seed: int = 0,
) -> tuple[GraphSnapshotSequence, torch.Tensor]:
    """Simulate z_{t+1}=z_t @ K.T with features equal to latents."""
    torch.manual_seed(seed)
    d = true_k.shape[0]
    edge_index = (
        _two_node_edge_index()
        if num_nodes == 2
        else torch.tensor(
            [
                [i % num_nodes for i in range(num_nodes)],
                [(i + 1) % num_nodes for i in range(num_nodes)],
            ],
            dtype=torch.long,
        )
    )
    z = torch.randn(num_nodes, d)
    latents = [z.clone()]
    for _ in range(steps - 1):
        z = z @ true_k.T
        if process_std > 0:
            z = z + process_std * torch.randn_like(z)
        latents.append(z.clone())
    snapshots = [Data(x=z_t.clone(), edge_index=edge_index) for z_t in latents]
    stacked = torch.stack(latents, dim=0)
    return GraphSnapshotSequence(snapshots), stacked


def test_koopman_observer_matches_reference_kf_fully_observed() -> None:
    """Fully observed linear latent system should match a textbook KF."""
    from koopman_graph.adaptation import KoopmanObserver
    from koopman_graph.adaptation.kalman import reference_kalman_filter

    true_k = torch.diag(torch.tensor([0.9, 0.8]))
    sequence, true_latents = _linear_latent_sequence(true_k, steps=10, seed=3)
    encoder = GNNEncoder(in_channels=2, hidden_channels=4, latent_dim=2, num_layers=1)
    decoder = GNNDecoder(latent_dim=2, hidden_channels=4, out_channels=2, num_layers=1)
    model = GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=2,
        time_step=1.0,
        koopman_parameterization="dense",
    )
    model.koopman.set_dense_matrix(true_k)
    _patch_identity_io(model)

    process_noise = 1e-4
    observation_noise = 1e-3
    observer = KoopmanObserver(
        model,
        process_noise=process_noise,
        observation_noise=observation_noise,
        observation_model="latent_encode",
        initial_covariance=1.0,
    )
    result = observer.filter(sequence)

    n_nodes, d = 2, 2
    state_dim = n_nodes * d
    a_mat = torch.kron(torch.eye(n_nodes), true_k)
    h_mat = torch.eye(state_dim)
    q_mat = process_noise * torch.eye(state_dim)
    r_mat = observation_noise * torch.eye(state_dim)
    measurements = true_latents.reshape(len(sequence), -1)
    x0 = measurements[0]
    p0 = torch.eye(state_dim)
    ref_means, _, _, _ = reference_kalman_filter(
        transition=a_mat,
        process_cov=q_mat,
        observation=h_mat,
        observation_cov=r_mat,
        measurements=measurements,
        x0=x0,
        p0=p0,
    )
    assert torch.allclose(
        result.latents.reshape(len(sequence), -1), ref_means, atol=1e-5
    )


def test_koopman_observer_decoder_jacobian_linear_identity() -> None:
    """Decoder-Jacobian mode with identity decode recovers H = I on a tiny graph."""
    from koopman_graph.adaptation import KoopmanObserver

    true_k = torch.diag(torch.tensor([0.85, 0.75]))
    sequence, _ = _linear_latent_sequence(true_k, steps=6, seed=5)
    encoder = GNNEncoder(in_channels=2, hidden_channels=4, latent_dim=2, num_layers=1)
    decoder = GNNDecoder(latent_dim=2, hidden_channels=4, out_channels=2, num_layers=1)
    model = GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=2,
        time_step=1.0,
        koopman_parameterization="dense",
    )
    model.koopman.set_dense_matrix(true_k)
    _patch_identity_io(model)

    observer = KoopmanObserver(
        model,
        process_noise=1e-4,
        observation_noise=1e-3,
        observation_model="decoder_jacobian",
    )
    filtered = observer.filter(sequence)
    assert filtered.latents.shape == (6, 2, 2)
    assert torch.isfinite(filtered.latents).all()


def test_imputation_rmse_soft_monotonic_in_drop_fraction() -> None:
    """Imputation RMSE should not improve as more nodes are dropped (soft)."""
    from koopman_graph.adaptation import KoopmanObserver

    true_k = torch.diag(torch.tensor([0.9, 0.85, 0.8]))
    sequence, true_latents = _linear_latent_sequence(
        true_k,
        num_nodes=3,
        steps=16,
        seed=7,
    )
    # Rebuild edge index for 3 nodes inside helper already.
    encoder = GNNEncoder(in_channels=3, hidden_channels=4, latent_dim=3, num_layers=1)
    decoder = GNNDecoder(latent_dim=3, hidden_channels=4, out_channels=3, num_layers=1)
    model = GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=3,
        time_step=1.0,
        koopman_parameterization="dense",
    )
    model.koopman.set_dense_matrix(true_k)
    _patch_identity_io(model)
    observer = KoopmanObserver(model, process_noise=1e-4, observation_noise=1e-2)

    def masked_rmse(drop_fraction: float, seed: int) -> float:
        torch.manual_seed(seed)
        masks = torch.rand(len(sequence), sequence.num_nodes) > drop_fraction
        # Ensure at least one observed node per timestep.
        masks[:, 0] = True
        from koopman_graph import GraphSnapshotSequence

        masked = GraphSnapshotSequence(
            list(sequence),
            observation_masks=masks,
        )
        imputed = observer.impute(masked, use_smoother=True)
        errs = []
        for t in range(len(sequence)):
            miss = ~masks[t]
            if not bool(miss.any()):
                continue
            pred = imputed[t].x[miss]
            truth = true_latents[t][miss]
            errs.append(torch.mean((pred - truth) ** 2).sqrt())
        return float(torch.stack(errs).mean().item()) if errs else 0.0

    rmse_lo = masked_rmse(0.2, seed=11)
    rmse_hi = masked_rmse(0.6, seed=11)
    assert rmse_hi + 1e-5 >= rmse_lo * 0.95


def test_smoother_not_worse_than_filter_on_average() -> None:
    """RTS smooth should not exceed filter RMSE vs ground-truth latents."""
    from koopman_graph.adaptation import KoopmanObserver

    true_k = torch.diag(torch.tensor([0.88, 0.82]))
    sequence, true_latents = _linear_latent_sequence(
        true_k,
        steps=14,
        process_std=0.02,
        seed=9,
    )
    encoder = GNNEncoder(in_channels=2, hidden_channels=4, latent_dim=2, num_layers=1)
    decoder = GNNDecoder(latent_dim=2, hidden_channels=4, out_channels=2, num_layers=1)
    model = GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=2,
        time_step=1.0,
        koopman_parameterization="dense",
    )
    model.koopman.set_dense_matrix(true_k)
    _patch_identity_io(model)

    torch.manual_seed(0)
    masks = torch.ones(len(sequence), sequence.num_nodes, dtype=torch.bool)
    masks[::2, 1] = False
    from koopman_graph import GraphSnapshotSequence

    masked = GraphSnapshotSequence(list(sequence), observation_masks=masks)
    observer = KoopmanObserver(model, process_noise=1e-3, observation_noise=1e-2)
    filt = observer.filter(masked)
    smth = observer.smooth(masked)
    filt_err = torch.mean((filt.latents - true_latents) ** 2).sqrt()
    smth_err = torch.mean((smth.latents - true_latents) ** 2).sqrt()
    assert float(smth_err.item()) <= float(filt_err.item()) + 1e-5


def test_koopman_observer_rejects_bilinear() -> None:
    """Observer should refuse bilinear control_mode."""
    from koopman_graph.adaptation import KoopmanObserver

    encoder = GNNEncoder(in_channels=2, hidden_channels=4, latent_dim=2, num_layers=1)
    decoder = GNNDecoder(latent_dim=2, hidden_channels=4, out_channels=2, num_layers=1)
    model = GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=2,
        time_step=1.0,
        control_dim=1,
        control_mode="bilinear",
        koopman_parameterization="dense",
    )
    with pytest.raises(ValueError, match="bilinear"):
        KoopmanObserver(model)


def test_filter_result_is_frozen() -> None:
    """Verify ``FilterResult`` is a frozen dataclass."""
    from koopman_graph.adaptation import FilterResult

    result = FilterResult(
        latents=torch.zeros(2, 2, 2),
        covariances=torch.eye(4).unsqueeze(0).repeat(2, 1, 1),
    )
    with pytest.raises(AttributeError):
        result.latents = torch.ones(2, 2, 2)  # type: ignore[misc]


def test_graph_diffuse_impute_and_validation() -> None:
    """graph_diffuse_impute fills holes and rejects non-positive iterations."""
    from koopman_graph.adaptation.impute import graph_diffuse_impute

    edge_index = _two_node_edge_index()
    x = torch.tensor([[1.0, 2.0], [0.0, 0.0]])
    mask = torch.tensor([True, False])
    with pytest.raises(ValueError, match="iterations"):
        graph_diffuse_impute(x, mask, edge_index, iterations=0)
    filled = graph_diffuse_impute(x, mask, edge_index, iterations=4)
    assert torch.allclose(filled[0], x[0])
    assert not torch.allclose(filled[1], torch.zeros(2))
    unchanged = graph_diffuse_impute(x, torch.ones(2, dtype=torch.bool), edge_index)
    assert torch.equal(unchanged, x)


def test_koopman_observer_constructor_validation() -> None:
    """Observer rejects non-positive noise scales and invalid observation models."""
    from koopman_graph.adaptation import KoopmanObserver

    encoder = GNNEncoder(in_channels=2, hidden_channels=4, latent_dim=2, num_layers=1)
    decoder = GNNDecoder(latent_dim=2, hidden_channels=4, out_channels=2, num_layers=1)
    model = GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=2,
        time_step=1.0,
        koopman_parameterization="dense",
    )
    with pytest.raises(ValueError, match="process_noise"):
        KoopmanObserver(model, process_noise=0.0)
    with pytest.raises(ValueError, match="observation_noise"):
        KoopmanObserver(model, observation_noise=-1.0)
    with pytest.raises(ValueError, match="initial_covariance"):
        KoopmanObserver(model, initial_covariance=0.0)
    with pytest.raises(ValueError, match="observation_model"):
        KoopmanObserver(model, observation_model="not_a_mode")  # type: ignore[arg-type]


def test_rts_smooth_helper_runs() -> None:
    """Exported RTS helper should refine a short filtered trajectory."""
    from koopman_graph.adaptation.kalman import reference_kalman_filter, rts_smooth

    transition = torch.diag(torch.tensor([0.9, 0.8]))
    process_cov = 1e-3 * torch.eye(2)
    observation = torch.eye(2)
    observation_cov = 1e-2 * torch.eye(2)
    measurements = torch.randn(5, 2)
    means, covs, pred_means, pred_covs = reference_kalman_filter(
        transition=transition,
        process_cov=process_cov,
        observation=observation,
        observation_cov=observation_cov,
        measurements=measurements,
        x0=measurements[0],
        p0=torch.eye(2),
    )
    sm_means, sm_covs = rts_smooth(
        transition=transition,
        filtered_means=means,
        filtered_covs=covs,
        pred_means=pred_means,
        pred_covs=pred_covs,
    )
    assert sm_means.shape == means.shape
    assert sm_covs.shape == covs.shape
    assert torch.isfinite(sm_means).all()


def test_observer_graph_and_continuous_transitions() -> None:
    """Filter should support graph and continuous Koopman operators."""
    from koopman_graph.adaptation import KoopmanObserver

    true_k = torch.diag(torch.tensor([0.9, 0.8]))
    sequence, _ = _linear_latent_sequence(true_k, steps=5, seed=11)

    graph_model = GraphKoopmanModel(
        GNNEncoder(2, 4, 2, num_layers=1),
        GNNDecoder(2, 4, 2, num_layers=1),
        latent_dim=2,
        time_step=1.0,
        koopman="graph",
        koopman_parameterization="dense",
    )
    graph_model.koopman.set_dense_matrices(true_k, torch.zeros_like(true_k))
    _patch_identity_io(graph_model)
    graph_result = KoopmanObserver(graph_model).filter(sequence)
    assert graph_result.latents.shape == (5, 2, 2)

    cont_model = GraphKoopmanModel(
        GNNEncoder(2, 4, 2, num_layers=1),
        GNNDecoder(2, 4, 2, num_layers=1),
        latent_dim=2,
        time_step=0.5,
        dynamics_mode="continuous",
        koopman_parameterization="dense",
    )
    with torch.no_grad():
        cont_model.koopman.set_dense_matrix(torch.diag(torch.tensor([-0.2, -0.3])))
    _patch_identity_io(cont_model)
    timestamps = torch.arange(5, dtype=torch.float32) * 0.5
    cont_seq = GraphSnapshotSequence(list(sequence), timestamps=timestamps)
    cont_result = KoopmanObserver(cont_model).filter(cont_seq)
    assert cont_result.latents.shape == (5, 2, 2)


def test_observer_protocol_injection_and_controls() -> None:
    """Protocol-only operators use finite-diff A; controls add process bias."""
    from koopman_graph.adaptation import KoopmanObserver

    class _ProtocolOp(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.latent_dim = 2
            self.control_dim = 1
            self.parameterization = "dense"
            self._k = torch.nn.Parameter(torch.diag(torch.tensor([0.9, 0.8])))
            self.B = torch.nn.Parameter(torch.tensor([[0.1, -0.05]]))

        @property
        def matrix(self) -> torch.Tensor:
            return self._k

        @property
        def K(self) -> torch.Tensor:
            return self._k

        def advance(
            self,
            z: torch.Tensor,
            delta_t: float | torch.Tensor | None = None,
            *,
            control: torch.Tensor | None = None,
            edge_index: torch.Tensor | None = None,
            edge_weight: torch.Tensor | None = None,
        ) -> torch.Tensor:
            _ = delta_t, edge_index, edge_weight
            out = z @ self._k.T
            if control is not None:
                out = out + control @ self.B
            return out

        def inverse_advance(
            self,
            z: torch.Tensor,
            delta_t: float | torch.Tensor | None = None,
            *,
            control: torch.Tensor | None = None,
            inverse_matrix: torch.Tensor | None = None,
            edge_index: torch.Tensor | None = None,
            edge_weight: torch.Tensor | None = None,
        ) -> torch.Tensor:
            _ = delta_t, edge_index, edge_weight, inverse_matrix
            recovered = z
            if control is not None:
                recovered = recovered - control @ self.B
            return recovered @ torch.linalg.pinv(self._k).T

        def bound_metric(self) -> torch.Tensor:
            return torch.linalg.eigvals(self._k).abs().max()

    true_k = torch.diag(torch.tensor([0.9, 0.8]))
    sequence, _ = _linear_latent_sequence(true_k, steps=4, seed=13)
    controls = torch.zeros(4, 1)
    controls[1:, 0] = 0.2
    controlled = GraphSnapshotSequence(list(sequence), control_inputs=controls)
    model = GraphKoopmanModel(
        GNNEncoder(2, 4, 2, num_layers=1),
        GNNDecoder(2, 4, 2, num_layers=1),
        latent_dim=2,
        time_step=1.0,
        control_dim=1,
        koopman=_ProtocolOp(),
    )
    _patch_identity_io(model)
    result = KoopmanObserver(model).filter(controlled)
    assert result.latents.shape == (4, 2, 2)


def test_observer_diffusion_warm_start_edge_weight_and_impute_paths() -> None:
    """Warm-start, edge weights, filter-only impute, and empty-mask errors."""
    from koopman_graph.adaptation import KoopmanObserver

    true_k = torch.diag(torch.tensor([0.9, 0.8]))
    sequence, _ = _linear_latent_sequence(true_k, steps=5, seed=17)
    weighted_snaps = []
    for snap in sequence:
        data = Data(
            x=snap.x.clone(),
            edge_index=snap.edge_index.clone(),
            edge_weight=torch.ones(snap.edge_index.shape[1]),
        )
        weighted_snaps.append(data)
    masks = torch.ones(5, 2, dtype=torch.bool)
    masks[:, 1] = False
    weighted = GraphSnapshotSequence(weighted_snaps, observation_masks=masks)

    model = GraphKoopmanModel(
        GNNEncoder(2, 4, 2, num_layers=1),
        GNNDecoder(2, 4, 2, num_layers=1),
        latent_dim=2,
        time_step=1.0,
        koopman_parameterization="dense",
    )
    model.koopman.set_dense_matrix(true_k)
    _patch_identity_io(model)
    observer = KoopmanObserver(
        model,
        graph_diffusion_warm_start=True,
        diffusion_iterations=3,
    )
    filtered = observer.filter(weighted)
    assert filtered.latents.shape == (5, 2, 2)
    imputed = observer.impute(weighted, use_smoother=False)
    assert imputed.has_observation_masks
    assert hasattr(imputed[0], "edge_weight")

    full = GraphSnapshotSequence(weighted_snaps)
    filled = observer.impute(full, use_smoother=False)
    assert not filled.has_observation_masks

    with pytest.raises(ValueError, match="no observed"):
        observer._select_observed(
            torch.zeros(4),
            torch.eye(4),
            torch.zeros(4, dtype=torch.bool),
        )


def test_observer_decoder_jacobian_with_edge_weight() -> None:
    """decoder_jacobian path should accept edge_weight on snapshots."""
    from koopman_graph.adaptation import KoopmanObserver

    true_k = torch.diag(torch.tensor([0.85, 0.75]))
    sequence, _ = _linear_latent_sequence(true_k, steps=4, seed=19)
    snaps = [
        Data(
            x=snap.x.clone(),
            edge_index=snap.edge_index.clone(),
            edge_weight=torch.ones(snap.edge_index.shape[1]),
        )
        for snap in sequence
    ]
    weighted = GraphSnapshotSequence(snaps)
    model = GraphKoopmanModel(
        GNNEncoder(2, 4, 2, num_layers=1),
        GNNDecoder(2, 4, 2, num_layers=1),
        latent_dim=2,
        time_step=1.0,
        koopman_parameterization="dense",
    )
    model.koopman.set_dense_matrix(true_k)
    _patch_identity_io(model)
    observer = KoopmanObserver(model, observation_model="decoder_jacobian")
    result = observer.filter(weighted)
    assert result.latents.shape == (4, 2, 2)


def test_observer_peers_do_not_call_private_advance_latent() -> None:
    """Production peers must not invoke ``GraphKoopmanModel._advance_latent``."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src" / "koopman_graph"
    offenders: list[str] = []
    for path in root.rglob("*.py"):
        # Same-named model package: private advance lives on the estimator only.
        if path.name == "estimator.py" and path.parent.name == "model":
            continue
        text = path.read_text(encoding="utf-8")
        if "_advance_latent" in text:
            offenders.append(str(path.relative_to(root.parent.parent)))
    assert offenders == []


def test_observer_control_bias_parity_discrete_continuous_graph() -> None:
    """Control biases match ``model._advance_latent`` under shared ``delta_t``."""
    from koopman_graph.adaptation import KoopmanObserver
    from koopman_graph.data import resolve_pair_delta_t
    from koopman_graph.graph_utils import snapshot_edge_weight

    true_k = torch.diag(torch.tensor([0.9, 0.8]))
    sequence, _ = _linear_latent_sequence(true_k, steps=4, seed=21)
    controls = torch.zeros(4, 1)
    controls[:, 0] = torch.tensor([0.1, -0.2, 0.3, 0.0])
    controlled = GraphSnapshotSequence(list(sequence), control_inputs=controls)

    cases: list[tuple[str, GraphKoopmanModel, GraphSnapshotSequence]] = []

    discrete = GraphKoopmanModel(
        GNNEncoder(2, 4, 2, num_layers=1),
        GNNDecoder(2, 4, 2, num_layers=1),
        latent_dim=2,
        time_step=1.0,
        control_dim=1,
        koopman_parameterization="dense",
    )
    discrete.koopman.set_dense_matrix(
        true_k,
        control_matrix=torch.tensor([[0.15, -0.05]]),
    )
    _patch_identity_io(discrete)
    cases.append(("discrete", discrete, controlled))

    continuous = GraphKoopmanModel(
        GNNEncoder(2, 4, 2, num_layers=1),
        GNNDecoder(2, 4, 2, num_layers=1),
        latent_dim=2,
        time_step=0.25,
        dynamics_mode="continuous",
        control_dim=1,
        koopman_parameterization="dense",
    )
    continuous.koopman.set_dense_matrix(
        torch.diag(torch.tensor([-0.2, -0.4])),
        control_matrix=torch.tensor([[0.1, 0.05]]),
    )
    _patch_identity_io(continuous)
    timestamps = torch.arange(4, dtype=torch.float32) * 0.25
    cont_seq = GraphSnapshotSequence(
        list(sequence),
        timestamps=timestamps,
        control_inputs=controls,
    )
    cases.append(("continuous", continuous, cont_seq))

    graph = GraphKoopmanModel(
        GNNEncoder(2, 4, 2, num_layers=1),
        GNNDecoder(2, 4, 2, num_layers=1),
        latent_dim=2,
        time_step=1.0,
        control_dim=1,
        koopman="graph",
        koopman_parameterization="dense",
    )
    graph.koopman.set_dense_matrices(
        true_k,
        0.1 * true_k,
        control_matrix=torch.tensor([[0.2, -0.1]]),
    )
    _patch_identity_io(graph)
    cases.append(("graph", graph, controlled))

    for _label, model, seq in cases:
        observer = KoopmanObserver(model)
        state_dim = seq.num_nodes * model.latent_dim
        biases = observer._control_biases(
            seq,
            state_dim=state_dim,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        assert len(biases) == len(seq) - 1
        for t, bias in enumerate(biases):
            snap = seq[t]
            control = seq.control_at(t)
            delta = resolve_pair_delta_t(
                seq,
                t,
                default_time_step=float(model.time_step),
            )
            z0 = torch.zeros(seq.num_nodes, model.latent_dim)
            expected = model._advance_latent(
                z0,
                control=control,
                delta_t=delta,
                edge_index=snap.edge_index,
                edge_weight=snapshot_edge_weight(snap),
            ).reshape(-1)
            assert torch.allclose(bias, expected, atol=1e-6)


def test_observer_finite_diff_matches_advance_jacobian() -> None:
    """Finite-difference A recovers the dense Kronecker map for discrete K."""
    from koopman_graph.adaptation import KoopmanObserver

    true_k = torch.diag(torch.tensor([0.9, 0.7]))
    sequence, _ = _linear_latent_sequence(true_k, steps=3, seed=23)
    model = GraphKoopmanModel(
        GNNEncoder(2, 4, 2, num_layers=1),
        GNNDecoder(2, 4, 2, num_layers=1),
        latent_dim=2,
        time_step=1.0,
        koopman_parameterization="dense",
    )
    model.koopman.set_dense_matrix(true_k)
    _patch_identity_io(model)
    observer = KoopmanObserver(model)
    a_mat = observer._finite_diff_transition(
        sequence,
        0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    expected = torch.kron(torch.eye(2), true_k)
    assert torch.allclose(a_mat, expected, atol=1e-3)
