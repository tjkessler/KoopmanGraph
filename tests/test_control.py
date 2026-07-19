"""Tests for Koopman-with-control support."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from koopman_graph import (
    GNNDecoder,
    GNNEncoder,
    GraphKoopmanModel,
    GraphSnapshotSequence,
)
from koopman_graph.baselines import DMDcBaseline
from koopman_graph.data import temporal_split
from koopman_graph.datasets.ieee118 import IEEE118DynamicBenchmark
from koopman_graph.operators import KoopmanOperator
from koopman_graph.serialization import (
    build_model_config,
    load_checkpoint,
)


def _two_node_edge_index() -> torch.Tensor:
    """Return a minimal two-node bidirectional edge index."""
    return torch.tensor([[0, 1], [1, 0]], dtype=torch.long)


def _controlled_linear_states(
    operator: torch.Tensor,
    input_matrix: torch.Tensor,
    control_values: list[float],
    initial_state: torch.Tensor,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Generate states following ``x_next = x @ K.T + u @ B``."""
    states = [initial_state]
    controls: list[torch.Tensor] = []
    state = initial_state
    for value in control_values:
        control = torch.tensor([value], dtype=state.dtype)
        controls.append(control)
        state = state @ operator.T + control @ input_matrix
        states.append(state)
    return states, controls


def _sequence_from_states(
    states: list[torch.Tensor],
    controls: list[torch.Tensor],
    edge_index: torch.Tensor,
    *,
    num_nodes: int,
    in_channels: int,
) -> GraphSnapshotSequence:
    """Build a controlled graph snapshot sequence."""
    if len(controls) == len(states) - 1:
        controls = [*controls, controls[-1].clone()]
    snapshots = [
        Data(x=state.reshape(num_nodes, in_channels), edge_index=edge_index)
        for state in states
    ]
    return GraphSnapshotSequence(
        snapshots,
        control_inputs=torch.stack(controls, dim=0),
    )


def test_koopman_operator_control_forward_and_inverse() -> None:
    """Verify controlled forward and inverse steps are consistent."""
    operator = KoopmanOperator(3, control_dim=1, init_mode="identity")
    with torch.no_grad():
        operator._parameters["K"].copy_(torch.diag(torch.tensor([0.8, 0.9, 1.0])))
        operator.B.copy_(torch.tensor([[0.2, -0.1, 0.05]]))

    z = torch.randn(4, 3)
    control = torch.tensor([0.5])
    z_next = operator(z, control=control)
    z_recovered = operator.inverse_step(z_next, control=control)
    assert torch.allclose(z_recovered, z, atol=1e-5)


def test_koopman_operator_per_node_control() -> None:
    """Verify per-node controls apply distinct latent offsets."""
    operator = KoopmanOperator(2, control_dim=1, init_mode="identity")
    with torch.no_grad():
        operator._parameters["K"].zero_()
        operator.B.copy_(torch.tensor([[1.0, 0.0]]))

    z = torch.zeros(3, 2)
    control = torch.tensor([[1.0], [2.0], [3.0]])
    z_next = operator(z, control=control)
    expected = torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    assert torch.allclose(z_next, expected)


def test_graph_snapshot_sequence_control_validation(
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Verify control tensor shape validation."""
    snapshots = [
        Data(
            x=torch.randn(3, 2),
            edge_index=synthetic_edge_index,
        )
        for _ in range(4)
    ]
    with pytest.raises(ValueError, match="control_inputs must have shape"):
        GraphSnapshotSequence(
            snapshots,
            control_inputs=torch.randn(4, 2, 1, 1),
        )


def test_temporal_split_preserves_controls() -> None:
    """Verify temporal splits slice control inputs consistently."""
    edge_index = _two_node_edge_index()
    operator = torch.diag(torch.tensor([0.9, 1.0]))
    input_matrix = torch.tensor([[0.4, 0.1]])
    states, controls = _controlled_linear_states(
        operator,
        input_matrix,
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        torch.tensor([1.0, -0.5]),
    )
    sequence = _sequence_from_states(
        states,
        controls,
        edge_index,
        num_nodes=2,
        in_channels=1,
    )
    split = temporal_split(sequence, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25)
    assert split.train.has_controls
    total_timesteps = (
        split.train.num_timesteps + split.val.num_timesteps + split.test.num_timesteps
    )
    assert total_timesteps == sequence.num_timesteps
    assert split.train.control_inputs is not None
    assert split.train.control_inputs.shape[0] == split.train.num_timesteps


def test_uncontrolled_model_rejects_controlled_sequence() -> None:
    """Verify uncontrolled models reject sequences with controls."""
    edge_index = _two_node_edge_index()
    operator = torch.diag(torch.tensor([0.9, 1.0]))
    input_matrix = torch.tensor([[0.2, 0.1]])
    states, controls = _controlled_linear_states(
        operator,
        input_matrix,
        [0.1, 0.2, 0.3],
        torch.tensor([1.0, 0.5]),
    )
    sequence = _sequence_from_states(
        states,
        controls,
        edge_index,
        num_nodes=2,
        in_channels=1,
    )
    model = GraphKoopmanModel(
        encoder=GNNEncoder(in_channels=1, hidden_channels=8, latent_dim=4),
        decoder=GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=1),
        latent_dim=4,
        time_step=0.1,
    )
    with pytest.raises(ValueError, match="control_dim is 0"):
        model.fit(sequence, epochs=1)


def test_controlled_model_requires_controls(
    synthetic_edge_index: torch.Tensor,
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Verify controlled models require control inputs in sequences."""
    model = GraphKoopmanModel(
        encoder=GNNEncoder(in_channels=3, hidden_channels=8, latent_dim=4),
        decoder=GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=3),
        latent_dim=4,
        time_step=0.1,
        control_dim=1,
    )
    with pytest.raises(ValueError, match="requires sequences with control inputs"):
        model.fit(scaling_sequence, epochs=1)


def test_controlled_model_fit_and_predict() -> None:
    """Verify a controlled model can train and predict with future controls."""
    edge_index = _two_node_edge_index()
    operator = torch.diag(torch.tensor([0.85, 0.95]))
    input_matrix = torch.tensor([[0.35, 0.15]])
    states, controls = _controlled_linear_states(
        operator,
        input_matrix,
        [0.2, -0.1, 0.3, 0.0, 0.25, 0.4, -0.2, 0.1, 0.05, 0.15],
        torch.tensor([1.0, -0.5]),
    )
    sequence = _sequence_from_states(
        states,
        controls,
        edge_index,
        num_nodes=2,
        in_channels=1,
    )
    model = GraphKoopmanModel(
        encoder=GNNEncoder(in_channels=1, hidden_channels=16, latent_dim=2),
        decoder=GNNDecoder(latent_dim=2, hidden_channels=16, out_channels=1),
        latent_dim=2,
        time_step=0.1,
        control_dim=1,
    )
    torch.manual_seed(0)
    model.fit(sequence, epochs=40, lr=5e-3)
    future_controls = sequence.rollout_controls(0, steps=3)
    predictions = model.predict(sequence[0], steps=3, controls=future_controls)
    assert len(predictions) == 3
    assert predictions[0].x.shape == sequence[0].x.shape


def test_controlled_model_serialization_round_trip(tmp_path: Path) -> None:
    """Verify controlled models round-trip through save/load."""
    edge_index = _two_node_edge_index()
    operator = torch.diag(torch.tensor([0.9, 1.0]))
    input_matrix = torch.tensor([[0.2, 0.1]])
    states, controls = _controlled_linear_states(
        operator,
        input_matrix,
        [0.1, 0.2, 0.3, 0.4],
        torch.tensor([1.0, 0.5]),
    )
    sequence = _sequence_from_states(
        states,
        controls,
        edge_index,
        num_nodes=2,
        in_channels=1,
    )
    model = GraphKoopmanModel(
        encoder=GNNEncoder(in_channels=1, hidden_channels=8, latent_dim=4),
        decoder=GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=1),
        latent_dim=4,
        time_step=0.1,
        control_dim=1,
    )
    model.fit(sequence, epochs=2, lr=1e-2)
    checkpoint = tmp_path / "controlled.pt"
    model.save(checkpoint)
    config = build_model_config(model)
    assert config["control_dim"] == 1

    restored = load_checkpoint(checkpoint)
    future_controls = sequence.rollout_controls(0, steps=2)
    original = model.predict(sequence[0], steps=2, controls=future_controls)
    loaded = restored.predict(sequence[0], steps=2, controls=future_controls)
    for orig, loaded_pred in zip(original, loaded, strict=True):
        assert torch.allclose(orig.x, loaded_pred.x)


def test_dmdc_baseline_recovers_controlled_linear_dynamics() -> None:
    """Verify DMDc recovers known controlled flattened dynamics."""
    edge_index = _two_node_edge_index()
    operator = torch.tensor(
        [[0.8, 0.1], [-0.2, 1.05]],
        dtype=torch.float64,
    )
    input_matrix = torch.tensor([[0.25, -0.1]], dtype=torch.float64)
    states, controls = _controlled_linear_states(
        operator,
        input_matrix,
        [0.1, -0.2, 0.3, 0.0, 0.15],
        torch.tensor([1.0, -0.5], dtype=torch.float64),
    )
    sequence = _sequence_from_states(
        states,
        controls,
        edge_index,
        num_nodes=2,
        in_channels=1,
    )

    baseline = DMDcBaseline(time_step=0.25).fit(sequence)
    assert baseline.K is not None
    assert baseline.B is not None
    assert torch.allclose(baseline.K, operator, atol=1e-10)
    assert torch.allclose(baseline.B, input_matrix, atol=1e-10)

    future_controls = [controls[0], controls[1], controls[2]]
    predictions = baseline.predict(sequence[0], steps=3, controls=future_controls)
    for prediction, expected in zip(predictions, states[1:4], strict=True):
        assert torch.allclose(prediction.x.reshape(-1), expected, atol=1e-10)


def test_ieee118_exposes_load_ramp_controls() -> None:
    """Verify IEEE 118 benchmark can expose ramp controls."""
    sequence = IEEE118DynamicBenchmark.generate(
        num_timesteps=8,
        expose_load_ramp_control=True,
        seed=0,
    )
    assert sequence.has_controls
    assert sequence.control_dim == 1
    assert sequence.control_inputs is not None
    assert sequence.control_inputs.shape == (8, 1)


def test_bilinear_operator_forward_inverse_and_gradients() -> None:
    """Verify bilinear forward/inverse consistency and gradient flow."""
    operator = KoopmanOperator(
        3,
        control_dim=2,
        control_mode="bilinear",
        init_mode="identity",
    )
    k = torch.diag(torch.tensor([0.8, 0.9, 1.0]))
    b = torch.tensor([[0.1, -0.05, 0.0], [0.0, 0.2, -0.1]])
    n = torch.zeros(2, 3, 3)
    n[0] = torch.tensor([[0.0, 0.2, 0.0], [0.0, 0.0, 0.1], [0.05, 0.0, 0.0]])
    n[1] = torch.tensor([[0.1, 0.0, 0.0], [0.0, -0.1, 0.0], [0.0, 0.0, 0.05]])
    operator.set_dense_matrix(k, control_matrix=b, bilinear_matrices=n)

    z = torch.randn(5, 3, requires_grad=True)
    control = torch.tensor([0.4, -0.2])
    z_next = operator(z, control=control)
    expected = z @ k.T + control @ b
    expected = expected + control[0] * (z @ n[0].T) + control[1] * (z @ n[1].T)
    assert torch.allclose(z_next, expected, atol=1e-6)

    recovered = operator.inverse_step(z_next.detach(), control=control)
    assert torch.allclose(recovered, z.detach(), atol=1e-5)

    loss = z_next.square().sum()
    loss.backward()
    assert z.grad is not None
    assert operator.N.grad is not None
    assert operator.B.grad is not None


def test_bilinear_low_rank_factors_match_full_coupling() -> None:
    """Verify low-rank P/Q assemble to an equivalent full N."""
    operator = KoopmanOperator(
        4,
        control_dim=1,
        control_mode="bilinear",
        bilinear_rank=2,
        init_mode="identity",
    )
    with torch.no_grad():
        operator.B.copy_(torch.zeros(1, 4))
        p_factors = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.25]]])
        q_factors = torch.tensor([[[0.0, 1.0], [1.0, 0.0], [0.0, 0.5], [0.25, 0.0]]])
        operator.P.copy_(p_factors)
        operator.Q.copy_(q_factors)

    coupling = operator.bilinear_matrices()
    expected = operator.P[0] @ operator.Q[0].T
    assert coupling.shape == (1, 4, 4)
    assert torch.allclose(coupling[0], expected)

    z = torch.randn(3, 4)
    control = torch.tensor([0.7])
    z_next = operator(z, control=control)
    manual = z @ operator.K.T + control[0] * (z @ expected.T)
    assert torch.allclose(z_next, manual, atol=1e-6)


def test_bilinear_recovers_synthetic_system_additive_underfits() -> None:
    """Bilinear mode fits a pure state–control coupling; additive cannot."""
    torch.manual_seed(0)
    true_k = torch.diag(torch.tensor([0.85, 0.9]))
    true_n = torch.tensor([[[0.0, 0.8], [-0.7, 0.0]]])
    true_b = torch.zeros(1, 2)

    z0 = torch.tensor([1.0, -0.5])
    controls = torch.linspace(-1.0, 1.0, 40)
    states = [z0]
    z = z0.clone()
    for value in controls[:-1]:
        u = torch.tensor([float(value)])
        z = z @ true_k.T + u @ true_b + u[0] * (z @ true_n[0].T)
        states.append(z.clone())

    def _fit_mode(mode: str) -> float:
        operator = KoopmanOperator(
            2,
            control_dim=1,
            control_mode=mode,  # type: ignore[arg-type]
            init_mode="identity",
        )
        with torch.no_grad():
            operator.set_dense_matrix(
                true_k,
                control_matrix=torch.zeros(1, 2),
                **(
                    {"bilinear_matrices": torch.zeros_like(true_n)}
                    if mode == "bilinear"
                    else {}
                ),
            )
        # Keep K fixed at the true free dynamics; only fit control factors.
        operator._parameters["K"].requires_grad_(False)
        opt = torch.optim.Adam(
            [p for p in operator.parameters() if p.requires_grad],
            lr=5e-2,
        )
        final_loss = torch.tensor(0.0)
        for _ in range(600):
            opt.zero_grad()
            loss = torch.zeros(())
            for t, value in enumerate(controls[:-1]):
                u = torch.tensor([float(value)])
                pred = operator(states[t], control=u)
                loss = loss + (pred - states[t + 1]).square().mean()
            loss.backward()
            opt.step()
            final_loss = loss.detach()
        return float(final_loss)

    additive_loss = _fit_mode("additive")
    bilinear_loss = _fit_mode("bilinear")
    assert bilinear_loss < 1e-3
    assert additive_loss > 5e-2
    assert bilinear_loss < 0.1 * additive_loss


def test_bilinear_continuous_step_matches_effective_generator() -> None:
    """Continuous bilinear advance matches Van Loan on L_eff."""
    from koopman_graph.operators import ContinuousKoopmanOperator, van_loan_factors

    operator = ContinuousKoopmanOperator(
        2,
        control_dim=1,
        control_mode="bilinear",
        init_mode="identity",
    )
    l_mat = torch.tensor([[-0.5, 0.1], [0.0, -0.4]])
    b = torch.tensor([[0.2, -0.1]])
    n = torch.tensor([[[0.0, 0.3], [-0.2, 0.0]]])
    operator.set_dense_matrix(l_mat, control_matrix=b, bilinear_matrices=n)

    z = torch.tensor([[1.0, -0.5], [0.25, 0.75]])
    control = torch.tensor([0.5])
    delta = torch.tensor(0.2)
    got = operator.advance(z, delta, control=control)

    l_eff = l_mat + control[0] * n[0]
    phi11, phi12 = van_loan_factors(l_eff, b, delta)
    expected = z @ phi11.T + control @ phi12.T
    assert torch.allclose(got, expected, atol=1e-5)


def test_bilinear_model_serialization_round_trip(tmp_path: Path) -> None:
    """Verify control_mode and bilinear factors survive save/load."""
    model = GraphKoopmanModel(
        encoder=GNNEncoder(in_channels=1, hidden_channels=8, latent_dim=4),
        decoder=GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=1),
        latent_dim=4,
        time_step=0.1,
        control_dim=1,
        control_mode="bilinear",
        bilinear_rank=2,
    )
    with torch.no_grad():
        model.koopman.B.fill_(0.1)
        model.koopman.P.fill_(0.05)
        model.koopman.Q.fill_(-0.02)

    checkpoint = tmp_path / "bilinear.pt"
    model.save(checkpoint)
    config = build_model_config(model)
    assert config["control_mode"] == "bilinear"
    assert config["bilinear_rank"] == 2

    restored = load_checkpoint(checkpoint)
    assert restored.control_mode == "bilinear"
    assert restored.bilinear_rank == 2
    assert torch.allclose(restored.koopman.P, model.koopman.P)
    assert torch.allclose(restored.koopman.Q, model.koopman.Q)
    assert torch.allclose(restored.koopman.B, model.koopman.B)


def test_control_helper_validation_and_bilinear_terms() -> None:
    """Shared control helpers cover validation and bilinear assembly paths."""
    from koopman_graph.operators.control import (
        allocate_bilinear_parameters,
        bilinear_coupling_tensor,
        bilinear_state_control_term,
        broadcast_control_term,
        effective_bilinear_matrix,
        per_node_effective_bilinear_matrices,
        reset_bilinear_parameters,
        validate_control_mode,
    )

    validate_control_mode(
        control_dim=2, control_mode="bilinear", bilinear_rank=1, latent_dim=3
    )
    with pytest.raises(ValueError, match="control_mode must be"):
        validate_control_mode(
            control_dim=1, control_mode="quadratic", bilinear_rank=None, latent_dim=2  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="requires control_dim"):
        validate_control_mode(
            control_dim=0, control_mode="bilinear", bilinear_rank=None, latent_dim=2
        )
    with pytest.raises(ValueError, match="bilinear_rank requires"):
        validate_control_mode(
            control_dim=1, control_mode="additive", bilinear_rank=1, latent_dim=2
        )
    with pytest.raises(ValueError, match="bilinear_rank must be"):
        validate_control_mode(
            control_dim=1, control_mode="bilinear", bilinear_rank=0, latent_dim=2
        )
    with pytest.raises(ValueError, match="cannot exceed"):
        validate_control_mode(
            control_dim=1, control_mode="bilinear", bilinear_rank=5, latent_dim=2
        )

    module = torch.nn.Module()
    allocate_bilinear_parameters(
        module, control_dim=2, latent_dim=3, bilinear_rank=1
    )
    reset_bilinear_parameters(module)
    coupling = bilinear_coupling_tensor(module)
    assert coupling.shape == (2, 3, 3)

    empty = torch.nn.Module()
    with pytest.raises(AttributeError, match="no bilinear factors"):
        bilinear_coupling_tensor(empty)

    z = torch.randn(4, 3)
    global_u = torch.tensor([0.5, -0.25])
    term = bilinear_state_control_term(z, global_u, coupling)
    assert term.shape == z.shape
    per_node = torch.randn(4, 2)
    assert bilinear_state_control_term(z, per_node, coupling).shape == z.shape
    with pytest.raises(ValueError, match="node axis"):
        bilinear_state_control_term(torch.randn(3), per_node, coupling)
    with pytest.raises(ValueError, match="rows"):
        bilinear_state_control_term(z, torch.randn(5, 2), coupling)
    with pytest.raises(ValueError, match="control input must have shape"):
        bilinear_state_control_term(z, torch.randn(2, 2, 2), coupling)

    offset = torch.tensor([1.0, -0.5, 0.25])
    broadcast = broadcast_control_term(z, offset, latent_dim=3)
    assert broadcast.shape == z.shape
    assert torch.allclose(broadcast[0], offset)
    assert torch.allclose(broadcast[-1], offset)
    batched = torch.randn(2, 4, 3)
    assert broadcast_control_term(batched, offset, latent_dim=3).shape == batched.shape

    base = torch.eye(3)
    effective = effective_bilinear_matrix(base, global_u, coupling)
    assert effective.shape == (3, 3)
    with pytest.raises(ValueError, match="global control"):
        effective_bilinear_matrix(base, per_node, coupling)

    blocks = per_node_effective_bilinear_matrices(base, per_node, coupling)
    assert blocks.shape == (4, 3, 3)
    assert torch.allclose(
        blocks[0],
        effective_bilinear_matrix(base, per_node[0], coupling),
    )
    with pytest.raises(ValueError, match="per-node control"):
        per_node_effective_bilinear_matrices(base, global_u, coupling)
