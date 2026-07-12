"""Tests for Koopman-with-control support."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from koopman_graph import (
    DMDcBaseline,
    GNNDecoder,
    GNNEncoder,
    GraphKoopmanModel,
    GraphSnapshotSequence,
)
from koopman_graph.data import temporal_split
from koopman_graph.datasets.ieee118 import IEEE118DynamicBenchmark
from koopman_graph.operator import KoopmanOperator
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
