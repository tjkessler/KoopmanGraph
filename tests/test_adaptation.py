"""Tests for recursive Koopman online adaptation."""

from __future__ import annotations

import torch
from torch_geometric.data import Data

from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
from koopman_graph.adaptation import RecursiveKoopmanAdapter
from koopman_graph.continuous import ContinuousKoopmanOperator
from koopman_graph.operator import KoopmanOperator
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

    model.adapt_step(snapshots[0], snapshots[1])

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
