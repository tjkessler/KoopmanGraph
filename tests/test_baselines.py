"""Tests for classical Koopman baselines."""

import pytest
import torch
from torch_geometric.data import Data

from koopman_graph import DMDBaseline, DMDcBaseline, EDMDBaseline, GraphSnapshotSequence
from koopman_graph.baselines import ClassicalBaseline
from koopman_graph.baselines.base import (
    _fit_controlled_row_operator,
    _flatten_snapshots,
    _transition_controls,
)


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
    assert baseline.reconstruction_matrix is not None
    assert baseline.K.shape == (2, 2)
    assert baseline.reconstruction_matrix.shape == (1, 2)
    assert "decoder" not in baseline.__dict__
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


def _dynamic_topology_sequence(
    *,
    num_timesteps: int = 4,
    with_controls: bool = False,
) -> GraphSnapshotSequence:
    """Build a short sequence with alternating edge sets."""
    edge_a = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_b = torch.tensor([[0, 0], [1, 1]], dtype=torch.long)
    snapshots = [
        Data(
            x=torch.randn(2, 1),
            edge_index=edge_a if i % 2 == 0 else edge_b,
        )
        for i in range(num_timesteps)
    ]
    controls = torch.randn(num_timesteps, 1) if with_controls else None
    return GraphSnapshotSequence(
        snapshots,
        control_inputs=controls,
        allow_dynamic_topology=True,
    )


@pytest.mark.parametrize("baseline_cls", [DMDBaseline, EDMDBaseline, DMDcBaseline])
def test_baselines_reject_dynamic_topology(
    baseline_cls: type[DMDBaseline] | type[EDMDBaseline] | type[DMDcBaseline],
) -> None:
    """Verify classical baselines reject dynamic-topology sequences at fit."""
    sequence = _dynamic_topology_sequence(
        with_controls=baseline_cls is DMDcBaseline,
    )
    assert sequence.is_dynamic_topology

    with pytest.raises(ValueError, match="is_dynamic_topology"):
        baseline_cls().fit(sequence)


def test_baselines_accept_static_topology_with_dynamic_flag(
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Verify allow_dynamic_topology alone does not reject identical edges."""
    sequence = GraphSnapshotSequence(
        [Data(x=torch.randn(2, 1), edge_index=synthetic_edge_index) for _ in range(4)],
        allow_dynamic_topology=True,
    )
    assert not sequence.is_dynamic_topology
    baseline = DMDBaseline().fit(sequence)
    assert baseline.K is not None


@pytest.mark.parametrize("baseline_cls", [DMDBaseline, EDMDBaseline])
def test_baselines_reject_prediction_before_fit(
    baseline_cls: type[DMDBaseline] | type[EDMDBaseline],
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Verify prediction requires a fitted operator."""
    graph = Data(x=torch.ones(2, 1), edge_index=synthetic_edge_index)

    with pytest.raises(RuntimeError, match="must be fit"):
        baseline_cls().predict(graph, steps=1)


def _linear_fit_sequence(
    synthetic_edge_index: torch.Tensor,
) -> GraphSnapshotSequence:
    """Build a small deterministic linear sequence for baseline fitting."""
    operator = torch.diag(torch.tensor([0.9, 1.1], dtype=torch.float64))
    states = _linear_sequence(operator, torch.tensor([1.0, 2.0], dtype=torch.float64))
    return _sequence_from_states(
        states,
        synthetic_edge_index,
        num_nodes=2,
        in_channels=1,
    )


def _controlled_sequence(
    synthetic_edge_index: torch.Tensor,
    *,
    per_node: bool = False,
    num_timesteps: int = 6,
) -> GraphSnapshotSequence:
    """Build a controlled sequence with global or per-node controls."""
    torch.manual_seed(0)
    snapshots = [
        Data(
            x=torch.randn(2, 1, dtype=torch.float64),
            edge_index=synthetic_edge_index,
        )
        for _ in range(num_timesteps)
    ]
    if per_node:
        controls = torch.randn(num_timesteps, 2, 1, dtype=torch.float64)
    else:
        controls = torch.randn(num_timesteps, 1, dtype=torch.float64)
    return GraphSnapshotSequence(snapshots, control_inputs=controls)


def test_flatten_snapshots_rejects_empty_sequence() -> None:
    """Verify flattening an empty snapshot list raises ``ValueError``."""

    class _EmptySequence:
        def __iter__(self):
            return iter([])

    with pytest.raises(ValueError, match="at least one snapshot"):
        _flatten_snapshots(_EmptySequence())


def test_flatten_snapshots_rejects_integer_features(
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Verify integer node features raise ``TypeError``."""
    snapshots = [
        Data(x=torch.ones(2, 1, dtype=torch.long), edge_index=synthetic_edge_index)
        for _ in range(2)
    ]

    with pytest.raises(TypeError, match="must be floating-point"):
        _flatten_snapshots(GraphSnapshotSequence(snapshots))


def test_dmd_baseline_truncated_rank_recovers_dynamics(
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Verify rank-truncated DMD still recovers full-rank linear dynamics."""
    operator = torch.diag(torch.tensor([0.9, 1.1], dtype=torch.float64))
    states = _linear_sequence(operator, torch.tensor([1.0, 2.0], dtype=torch.float64))
    sequence = _sequence_from_states(
        states,
        synthetic_edge_index,
        num_nodes=2,
        in_channels=1,
    )

    baseline = DMDBaseline(rank=2).fit(sequence)

    assert baseline.K is not None
    assert torch.allclose(baseline.K, operator, atol=1e-8)


def test_dmd_baseline_truncated_rank_on_low_rank_data(
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Verify truncated SVD recovers dynamics embedded in a higher-dim state."""
    embedded = torch.diag(torch.tensor([0.9, 0.7], dtype=torch.float64))
    operator = torch.zeros(4, 4, dtype=torch.float64)
    operator[:2, :2] = embedded
    initial = torch.tensor([1.0, -0.5, 0.0, 0.0], dtype=torch.float64)
    states = _linear_sequence(operator, initial)
    sequence = _sequence_from_states(
        states,
        synthetic_edge_index,
        num_nodes=2,
        in_channels=2,
    )

    baseline = DMDBaseline(rank=2).fit(sequence)

    assert baseline.K is not None
    left = torch.stack(states[:-1])
    right = torch.stack(states[1:])
    predicted = left @ baseline.K.T
    assert torch.allclose(predicted, right, atol=1e-8)
    assert torch.linalg.matrix_rank(baseline.K, atol=1e-8).item() <= 2


def test_dmd_baseline_rejects_invalid_rank(
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Verify rank bounds are enforced during fitting."""
    sequence = _linear_fit_sequence(synthetic_edge_index)

    with pytest.raises(ValueError, match="rank must be >= 1"):
        DMDBaseline(rank=0).fit(sequence)
    with pytest.raises(ValueError, match="rank must be <="):
        DMDBaseline(rank=99).fit(sequence)


@pytest.mark.parametrize("baseline_cls", [DMDBaseline, DMDcBaseline, EDMDBaseline])
def test_baselines_share_classical_baseline_scaffold(
    baseline_cls: type[ClassicalBaseline],
) -> None:
    """Verify DMD-family baselines inherit shared ClassicalBaseline scaffolding."""
    baseline = baseline_cls(time_step=0.5)
    assert isinstance(baseline, ClassicalBaseline)
    assert baseline.time_step == 0.5
    assert baseline.K is None


@pytest.mark.parametrize("baseline_cls", [DMDBaseline, DMDcBaseline, EDMDBaseline])
def test_baselines_reject_non_positive_time_step(
    baseline_cls: type,
) -> None:
    """Verify all baselines reject non-positive ``time_step``."""
    with pytest.raises(ValueError, match="time_step must be positive"):
        baseline_cls(time_step=0.0)


def test_dmd_baseline_rejects_invalid_steps(
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Verify prediction rejects ``steps < 1``."""
    sequence = _linear_fit_sequence(synthetic_edge_index)
    baseline = DMDBaseline().fit(sequence)

    with pytest.raises(ValueError, match="steps must be >= 1"):
        baseline.predict(sequence[0], steps=0)


def test_dmd_baseline_rejects_mismatched_initial_graph(
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Verify initial graph shape validation against fitted metadata."""
    sequence = _linear_fit_sequence(synthetic_edge_index)
    baseline = DMDBaseline().fit(sequence)

    wrong_nodes = Data(
        x=torch.ones(3, 1, dtype=torch.float64),
        edge_index=synthetic_edge_index,
    )
    with pytest.raises(ValueError, match="nodes, expected"):
        baseline.predict(wrong_nodes, steps=1)

    wrong_channels = Data(
        x=torch.ones(2, 2, dtype=torch.float64),
        edge_index=synthetic_edge_index,
    )
    with pytest.raises(ValueError, match="feature dimension"):
        baseline.predict(wrong_channels, steps=1)


def test_fit_controlled_row_operator_validates_controls() -> None:
    """Verify control shape and sample-count validation."""
    left = torch.randn(4, 2, dtype=torch.float64)
    right = torch.randn(4, 2, dtype=torch.float64)

    with pytest.raises(ValueError, match="controls must have shape"):
        _fit_controlled_row_operator(
            left,
            right,
            torch.randn(4, dtype=torch.float64),
            None,
        )
    with pytest.raises(ValueError, match="samples, expected"):
        _fit_controlled_row_operator(
            left,
            right,
            torch.randn(3, 1, dtype=torch.float64),
            None,
        )


def test_transition_controls_requires_controls(
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Verify transition control extraction requires control inputs."""
    sequence = _linear_fit_sequence(synthetic_edge_index)

    with pytest.raises(ValueError, match="does not contain control inputs"):
        _transition_controls(sequence)


def test_transition_controls_rejects_per_node_inputs(
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Verify per-node controls are rejected (no silent flatten)."""
    sequence = _controlled_sequence(synthetic_edge_index, per_node=True)

    with pytest.raises(ValueError, match="does not support per-node"):
        _transition_controls(sequence)


def test_dmdc_baseline_rejects_single_snapshot(
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Verify DMDc fitting requires at least one transition."""
    sequence = GraphSnapshotSequence(
        [Data(x=torch.ones(2, 1), edge_index=synthetic_edge_index)],
        control_inputs=torch.ones(1, 1),
    )

    with pytest.raises(ValueError, match="at least two snapshots"):
        DMDcBaseline().fit(sequence)


def test_dmdc_baseline_rejects_uncontrolled_sequence(
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Verify DMDc fitting requires control inputs."""
    sequence = _linear_fit_sequence(synthetic_edge_index)

    with pytest.raises(ValueError, match="requires sequences with control inputs"):
        DMDcBaseline().fit(sequence)


def test_dmdc_baseline_rejects_per_node_controls(
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Verify DMDc rejects per-node (3-D) control inputs at fit."""
    sequence = _controlled_sequence(synthetic_edge_index, per_node=True)

    with pytest.raises(ValueError, match="does not support per-node"):
        DMDcBaseline().fit(sequence)


def test_dmdc_baseline_rejects_invalid_prediction_arguments(
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Verify DMDc prediction argument validation."""
    sequence = _controlled_sequence(synthetic_edge_index)
    baseline = DMDcBaseline().fit(sequence)
    control = torch.zeros(1, dtype=torch.float64)

    with pytest.raises(ValueError, match="steps must be >= 1"):
        baseline.predict(sequence[0], steps=0, controls=[])
    with pytest.raises(ValueError, match="expected 2 control inputs"):
        baseline.predict(sequence[0], steps=2, controls=[control])


def test_dmdc_baseline_rejects_invalid_control_shapes(
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Verify global control shape validation at prediction."""
    global_baseline = DMDcBaseline().fit(_controlled_sequence(synthetic_edge_index))
    with pytest.raises(ValueError, match="global controls must have shape"):
        global_baseline.predict(
            _controlled_sequence(synthetic_edge_index)[0],
            steps=1,
            controls=[torch.zeros(2, 1, dtype=torch.float64)],
        )
    with pytest.raises(ValueError, match="global controls must have shape"):
        global_baseline.predict(
            _controlled_sequence(synthetic_edge_index)[0],
            steps=1,
            controls=[torch.zeros(2, dtype=torch.float64)],
        )


def test_dmdc_baseline_spectrum_and_unfitted_errors(
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Verify DMDc spectrum access and pre-fit error handling."""
    unfitted = DMDcBaseline()
    with pytest.raises(RuntimeError, match="must be fit"):
        unfitted.spectrum()

    baseline = DMDcBaseline(time_step=0.5).fit(
        _controlled_sequence(synthetic_edge_index)
    )
    spectrum = baseline.spectrum()
    assert spectrum.time_step == 0.5
    assert spectrum.eigenvalues.shape == (2,)


def test_edmd_baseline_rejects_invalid_polynomial_degree() -> None:
    """Verify unsupported polynomial degrees raise ``ValueError``."""
    with pytest.raises(ValueError, match="polynomial_degree must be 1 or 2"):
        EDMDBaseline(polynomial_degree=3)  # type: ignore[arg-type]


def test_edmd_baseline_degree_one_matches_dmd(
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Verify degree-1 EDMD reduces to identity observables."""
    sequence = _linear_fit_sequence(synthetic_edge_index)

    baseline = EDMDBaseline(polynomial_degree=1).fit(sequence)

    assert baseline.observable_dim == baseline.state_dim
    dmd = DMDBaseline().fit(sequence)
    assert baseline.K is not None
    assert dmd.K is not None
    assert torch.allclose(baseline.K, dmd.K, atol=1e-8)


def test_edmd_baseline_rejects_invalid_steps(
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Verify EDMD prediction rejects ``steps < 1``."""
    baseline = EDMDBaseline().fit(_linear_fit_sequence(synthetic_edge_index))

    with pytest.raises(ValueError, match="steps must be >= 1"):
        baseline.predict(_linear_fit_sequence(synthetic_edge_index)[0], steps=0)
