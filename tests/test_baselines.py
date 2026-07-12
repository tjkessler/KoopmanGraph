"""Tests for classical Koopman baselines."""

import pytest
import torch
from torch_geometric.data import Data

from koopman_graph import DMDBaseline, EDMDBaseline, GraphSnapshotSequence


def _linear_sequence(
    operator: torch.Tensor,
    initial_state: torch.Tensor,
) -> list[torch.Tensor]:
    """Generate flattened states following ``x_next = x @ K.T``."""
    states = [initial_state]
    for _ in range(5):
        states.append(states[-1] @ operator.T)
    return states


def _sequence_from_states(
    states: list[torch.Tensor],
    edge_index: torch.Tensor,
    *,
    num_nodes: int,
    in_channels: int,
    edge_weight: torch.Tensor | None = None,
) -> GraphSnapshotSequence:
    """Build a graph snapshot sequence from flattened states."""
    snapshots = []
    for state in states:
        fields = {
            "x": state.reshape(num_nodes, in_channels),
            "edge_index": edge_index,
        }
        if edge_weight is not None:
            fields["edge_weight"] = edge_weight
        snapshots.append(Data(**fields))
    return GraphSnapshotSequence(snapshots)


def test_dmd_baseline_exactly_recovers_linear_dynamics(
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Verify full-rank DMD recovers a known flattened linear system."""
    operator = torch.tensor(
        [[0.8, 0.1], [-0.2, 1.05]],
        dtype=torch.float64,
    )
    states = _linear_sequence(
        operator,
        torch.tensor([1.0, -0.5], dtype=torch.float64),
    )
    sequence = _sequence_from_states(
        states,
        synthetic_edge_index,
        num_nodes=2,
        in_channels=1,
    )

    baseline = DMDBaseline(time_step=0.25).fit(sequence)

    assert baseline.K is not None
    assert torch.allclose(baseline.K, operator, atol=1e-10)
    predictions = baseline.predict(sequence[0], steps=3)
    for prediction, expected in zip(predictions, states[1:4], strict=True):
        assert torch.allclose(prediction.x.reshape(-1), expected, atol=1e-10)


def test_dmd_baseline_preserves_prediction_topology(
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Verify predictions keep graph shape and optional edge weights."""
    operator = torch.diag(torch.tensor([0.9, 1.1], dtype=torch.float64))
    edge_weight = torch.arange(synthetic_edge_index.shape[1], dtype=torch.float64)
    sequence = _sequence_from_states(
        _linear_sequence(operator, torch.tensor([1.0, 2.0], dtype=torch.float64)),
        synthetic_edge_index,
        num_nodes=2,
        in_channels=1,
        edge_weight=edge_weight,
    )

    baseline = DMDBaseline().fit(sequence)
    prediction = baseline.predict(sequence[0], steps=1)[0]

    assert prediction.x.shape == (2, 1)
    assert torch.equal(prediction.edge_index, synthetic_edge_index)
    assert torch.equal(prediction.edge_weight, edge_weight)


def test_dmd_baseline_spectrum_uses_analysis_api(
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Verify DMD exposes continuous-time spectral analysis."""
    operator = torch.diag(torch.tensor([0.5, 0.25], dtype=torch.float64))
    sequence = _sequence_from_states(
        _linear_sequence(operator, torch.tensor([2.0, 4.0], dtype=torch.float64)),
        synthetic_edge_index,
        num_nodes=2,
        in_channels=1,
    )

    spectrum = DMDBaseline(time_step=0.5).fit(sequence).spectrum()

    assert spectrum.time_step == 0.5
    assert torch.allclose(
        spectrum.eigenvalues.real,
        torch.tensor([0.5, 0.25], dtype=torch.float64),
        atol=1e-10,
    )


def test_edmd_baseline_lifts_polynomial_observables(
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Verify EDMD fits linear dynamics in identity-plus-square observables."""
    scale = 0.7
    states = [torch.tensor([1.3 * (scale**t)], dtype=torch.float64) for t in range(6)]
    sequence = _sequence_from_states(
        states,
        synthetic_edge_index,
        num_nodes=1,
        in_channels=1,
    )

    baseline = EDMDBaseline(polynomial_degree=2).fit(sequence)

    assert baseline.K is not None
    assert baseline.K.shape == (2, 2)
    prediction = baseline.predict(sequence[0], steps=3)[-1]
    assert torch.allclose(prediction.x.reshape(-1), states[3], atol=1e-10)


def test_edmd_baseline_spectrum_is_observable_space(
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Verify EDMD spectrum reflects the observable-space operator."""
    scale = 0.6
    states = [torch.tensor([2.0 * (scale**t)], dtype=torch.float64) for t in range(6)]
    sequence = _sequence_from_states(
        states,
        synthetic_edge_index,
        num_nodes=1,
        in_channels=1,
    )

    spectrum = EDMDBaseline(time_step=2.0, polynomial_degree=2).fit(sequence).spectrum()

    assert spectrum.time_step == 2.0
    assert torch.allclose(
        spectrum.eigenvalues.real,
        torch.tensor([scale, scale**2], dtype=torch.float64),
        atol=1e-10,
    )


@pytest.mark.parametrize("baseline_cls", [DMDBaseline, EDMDBaseline])
def test_baselines_reject_single_snapshot(
    baseline_cls: type[DMDBaseline] | type[EDMDBaseline],
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Verify fitting requires at least one transition."""
    sequence = GraphSnapshotSequence(
        [Data(x=torch.ones(2, 1), edge_index=synthetic_edge_index)]
    )

    with pytest.raises(ValueError, match="at least two snapshots"):
        baseline_cls().fit(sequence)


@pytest.mark.parametrize("baseline_cls", [DMDBaseline, EDMDBaseline])
def test_baselines_reject_prediction_before_fit(
    baseline_cls: type[DMDBaseline] | type[EDMDBaseline],
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Verify prediction requires a fitted operator."""
    graph = Data(x=torch.ones(2, 1), edge_index=synthetic_edge_index)

    with pytest.raises(RuntimeError, match="must be fit"):
        baseline_cls().predict(graph, steps=1)
