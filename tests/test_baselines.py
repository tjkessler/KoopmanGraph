"""Tests for classical Koopman baselines."""

import pytest
import torch
from torch_geometric.data import Data

from koopman_graph import GraphSnapshotSequence
from koopman_graph.baselines import (
    ClassicalBaseline,
    DMDBaseline,
    DMDcBaseline,
    EDMDBaseline,
)
from koopman_graph.baselines.base import (
    fit_controlled_row_operator,
    flatten_snapshots,
    gavish_donoho_omega,
    optimal_hard_threshold_rank,
    transition_controls,
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
    """DMD predictions keep graph shape; fitted K matrix advances flattened states."""
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
        flatten_snapshots(_EmptySequence())


def test_flatten_snapshots_rejects_integer_features(
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Verify integer node features raise ``TypeError``."""
    snapshots = [
        Data(x=torch.ones(2, 1, dtype=torch.long), edge_index=synthetic_edge_index)
        for _ in range(2)
    ]

    with pytest.raises(TypeError, match="must be floating-point"):
        flatten_snapshots(GraphSnapshotSequence(snapshots))


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
    """Verify rank bounds are enforced during construction / fitting."""
    sequence = _linear_fit_sequence(synthetic_edge_index)

    with pytest.raises(ValueError, match="rank must be >= 1"):
        DMDBaseline(rank=0)
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
        fit_controlled_row_operator(
            left,
            right,
            torch.randn(4, dtype=torch.float64),
            None,
        )
    with pytest.raises(ValueError, match="samples, expected"):
        fit_controlled_row_operator(
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
        transition_controls(sequence)


def test_transition_controls_rejects_per_node_inputs(
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Verify per-node controls are rejected (no silent flatten)."""
    sequence = _controlled_sequence(synthetic_edge_index, per_node=True)

    with pytest.raises(ValueError, match="does not support per-node"):
        transition_controls(sequence)


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
    """Degree-1 EDMD dictionary is identity; fitted K matches DMD matrix."""
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


def test_edmd_baseline_rejects_invalid_dictionary_knobs() -> None:
    """Verify unsupported dictionary / kernel knobs raise ``ValueError``."""
    with pytest.raises(ValueError, match="dictionary must be"):
        EDMDBaseline(dictionary="wavelet")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="length_scale must be positive"):
        EDMDBaseline(dictionary="rbf", length_scale=0.0)
    with pytest.raises(ValueError, match="kernel must be"):
        EDMDBaseline(dictionary="kernel", kernel="laplacian")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="num_centers must be >= 1"):
        EDMDBaseline(dictionary="rbf", num_centers=0)
    with pytest.raises(ValueError, match="kernel_approximation must be"):
        EDMDBaseline(
            dictionary="kernel",
            kernel_approximation="foo",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="dictionary='kernel'"):
        EDMDBaseline(
            dictionary="polynomial",
            kernel_approximation="nystrom",
        )
    with pytest.raises(ValueError, match="non-linear kernel"):
        EDMDBaseline(
            dictionary="kernel",
            kernel="linear",
            kernel_approximation="random_features",
        )
    with pytest.raises(ValueError, match="kernel_degree must be"):
        EDMDBaseline(dictionary="kernel", kernel="polynomial", kernel_degree=0)
    with pytest.raises(ValueError, match="kernel_gamma must be positive"):
        EDMDBaseline(
            dictionary="kernel",
            kernel="polynomial",
            kernel_gamma=0.0,
        )
    with pytest.raises(ValueError, match="num_features must be"):
        EDMDBaseline(
            dictionary="kernel",
            kernel="gaussian",
            kernel_approximation="random_features",
            num_features=0,
        )


def test_edmd_require_reconstruction_matrix_when_cleared(
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Clearing reconstruction_matrix after fit still fails the guard."""
    baseline = EDMDBaseline().fit(_linear_fit_sequence(synthetic_edge_index))
    baseline.reconstruction_matrix = None
    with pytest.raises(RuntimeError, match="must be fit"):
        baseline._require_reconstruction_matrix()


def test_edmd_observables_before_fit_raise(
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Observable lifts require fit state for RFF / centers / Nyström."""
    states = flatten_snapshots(_linear_fit_sequence(synthetic_edge_index))
    rff = EDMDBaseline(
        dictionary="kernel",
        kernel="gaussian",
        kernel_approximation="random_features",
    )
    with pytest.raises(RuntimeError, match="random features are not available"):
        rff._observables(states)

    rbf = EDMDBaseline(dictionary="rbf")
    with pytest.raises(RuntimeError, match="dictionary centers are not available"):
        rbf._observables(states)

    nystrom = EDMDBaseline(
        dictionary="kernel",
        kernel="gaussian",
        kernel_approximation="nystrom",
    )
    nystrom.centers = states[:3].detach().clone()
    with pytest.raises(RuntimeError, match="Nyström whitener is not available"):
        nystrom._observables(states)


def test_edmd_random_features_rejects_non_gaussian_kernel() -> None:
    """RFF initialization supports Gaussian kernels only."""
    baseline = EDMDBaseline(
        dictionary="kernel",
        kernel="polynomial",
        kernel_approximation="random_features",
    )
    with pytest.raises(ValueError, match="random_features currently supports"):
        baseline._init_random_features(torch.randn(5, 2, dtype=torch.float64))


def test_edmd_init_nystrom_whitener_requires_centers() -> None:
    """Nyström whitener build requires landmark centers."""
    baseline = EDMDBaseline(
        dictionary="kernel",
        kernel="gaussian",
        kernel_approximation="nystrom",
    )
    with pytest.raises(RuntimeError, match="landmark centers"):
        baseline._init_nystrom_whitener()


def test_edmd_select_centers_paths(
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Center selection covers Nyström defaults and RFF early return."""
    states = flatten_snapshots(_linear_fit_sequence(synthetic_edge_index))
    rff_baseline = EDMDBaseline(
        dictionary="kernel",
        kernel="gaussian",
        kernel_approximation="random_features",
    )
    assert rff_baseline._select_centers(states) is None

    nystrom_centers = EDMDBaseline(
        dictionary="kernel",
        kernel="gaussian",
        kernel_approximation="nystrom",
        num_centers=3,
    )._select_centers(states)
    assert nystrom_centers is not None
    assert nystrom_centers.shape[0] == 3

    nystrom_features = EDMDBaseline(
        dictionary="kernel",
        kernel="gaussian",
        kernel_approximation="nystrom",
        num_features=4,
    )._select_centers(states)
    assert nystrom_features is not None
    assert nystrom_features.shape[0] == 4


def test_edmd_feature_width_resolution() -> None:
    """RFF width honors num_features or defaults to min(32, T)."""
    explicit = EDMDBaseline(
        dictionary="kernel",
        kernel="gaussian",
        kernel_approximation="random_features",
        num_features=6,
    )
    assert explicit._feature_width(8) == 6
    default = EDMDBaseline(
        dictionary="kernel",
        kernel="gaussian",
        kernel_approximation="random_features",
    )
    assert default._feature_width(8) == 8
    assert default._feature_width(100) == 32


def test_edmd_polynomial_kernel_smoke(
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Polynomial kernel sections fit and roll out on a short sequence."""
    states = [
        torch.tensor([0.5 + 0.1 * t, 1.0 - 0.05 * t], dtype=torch.float64)
        for t in range(6)
    ]
    sequence = _sequence_from_states(
        states,
        synthetic_edge_index,
        num_nodes=2,
        in_channels=1,
    )
    baseline = EDMDBaseline(
        dictionary="kernel",
        kernel="polynomial",
        kernel_degree=2,
        kernel_gamma=1.0,
    ).fit(sequence)
    assert baseline.K is not None
    preds = baseline.predict(sequence[0], steps=1)
    assert preds[0].x.shape == (2, 1)


def test_edmd_nystrom_smoke(synthetic_edge_index: torch.Tensor) -> None:
    """Nyström kernel EDMD fits and rolls out on a short sequence."""
    states = [
        torch.tensor([0.5 + 0.1 * t, 1.0 - 0.05 * t], dtype=torch.float64)
        for t in range(8)
    ]
    sequence = _sequence_from_states(
        states,
        synthetic_edge_index,
        num_nodes=2,
        in_channels=1,
    )
    baseline = EDMDBaseline(
        dictionary="kernel",
        kernel="gaussian",
        kernel_approximation="nystrom",
        num_features=4,
        length_scale=2.0,
    ).fit(sequence)
    assert baseline.centers is not None
    assert baseline.centers.shape[0] == 4
    assert baseline.observable_dim == 4
    assert baseline.K is not None
    preds = baseline.predict(sequence[0], steps=1)
    assert preds[0].x.shape == (2, 1)


def test_edmd_random_features_seeded_reproducible(
    synthetic_edge_index: torch.Tensor,
) -> None:
    """RFF kernel EDMD is deterministic for a fixed ``feature_seed``."""
    states = [
        torch.tensor([0.5 + 0.1 * t, 1.0 - 0.05 * t], dtype=torch.float64)
        for t in range(8)
    ]
    sequence = _sequence_from_states(
        states,
        synthetic_edge_index,
        num_nodes=2,
        in_channels=1,
    )
    kwargs = {
        "dictionary": "kernel",
        "kernel": "gaussian",
        "kernel_approximation": "random_features",
        "num_features": 8,
        "feature_seed": 42,
        "length_scale": 1.5,
    }
    left = EDMDBaseline(**kwargs).fit(sequence)  # type: ignore[arg-type]
    right = EDMDBaseline(**kwargs).fit(sequence)  # type: ignore[arg-type]
    assert left._rff_weight is not None and right._rff_weight is not None
    assert left._rff_bias is not None and right._rff_bias is not None
    assert torch.equal(left._rff_weight, right._rff_weight)
    assert torch.equal(left._rff_bias, right._rff_bias)
    assert left.K is not None and right.K is not None
    # lstsq can differ by tiny ULPs across BLAS builds / xdist workers.
    assert torch.allclose(left.K, right.K, rtol=1e-7, atol=1e-8)
    assert left.observable_dim == 8
    preds = left.predict(sequence[0], steps=1)
    assert preds[0].x.shape == (2, 1)


def test_edmd_rbf_dictionary_smoke(
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Verify RBF EDMD fits and rolls out on a small nonlinear sequence."""
    scale = 0.85
    states = [
        torch.tensor([1.2 * (scale**t), 0.4 * (scale ** (2 * t))], dtype=torch.float64)
        for t in range(8)
    ]
    sequence = _sequence_from_states(
        states,
        synthetic_edge_index,
        num_nodes=2,
        in_channels=1,
    )

    baseline = EDMDBaseline(
        dictionary="rbf",
        num_centers=4,
        length_scale=1.5,
    ).fit(sequence)

    assert baseline.centers is not None
    assert baseline.centers.shape == (4, 2)
    assert baseline.observable_dim == 4
    assert baseline.K is not None
    assert baseline.K.shape == (4, 4)
    preds = baseline.predict(sequence[0], steps=2)
    assert len(preds) == 2
    assert preds[-1].x.shape == (2, 1)


def test_edmd_kernel_gaussian_smoke(
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Verify Gaussian kernel-section EDMD fits on a short sequence."""
    states = [
        torch.tensor([0.5 + 0.1 * t, 1.0 - 0.05 * t], dtype=torch.float64)
        for t in range(6)
    ]
    sequence = _sequence_from_states(
        states,
        synthetic_edge_index,
        num_nodes=2,
        in_channels=1,
    )

    baseline = EDMDBaseline(
        dictionary="kernel",
        kernel="gaussian",
        length_scale=2.0,
    ).fit(sequence)

    assert baseline.centers is not None
    assert baseline.centers.shape[0] == 6
    assert baseline.observable_dim == 6
    preds = baseline.predict(sequence[0], steps=1)
    assert preds[0].x.shape == (2, 1)


def test_edmd_linear_kernel_matches_dmd(
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Verify linear-kernel EDMD reduces to DMD on a linear synthetic system."""
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

    dmd = DMDBaseline().fit(sequence)
    edmd = EDMDBaseline(dictionary="kernel", kernel="linear").fit(sequence)

    assert dmd.K is not None
    assert edmd.K is not None
    assert torch.allclose(edmd.K, dmd.K, atol=1e-10)
    dmd_preds = dmd.predict(sequence[0], steps=3)
    edmd_preds = edmd.predict(sequence[0], steps=3)
    for left, right in zip(dmd_preds, edmd_preds, strict=True):
        assert torch.allclose(left.x, right.x, atol=1e-10)


def test_edmd_kernel_rejects_oversized_num_centers(
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Verify requesting more centers than snapshots fails clearly."""
    sequence = _linear_fit_sequence(synthetic_edge_index)
    with pytest.raises(ValueError, match="num_centers=.*exceeds"):
        EDMDBaseline(dictionary="rbf", num_centers=100).fit(sequence)


def test_baseline_peers_have_no_private_cross_module_imports() -> None:
    """Baseline peers must not import leading-``_`` symbols across modules."""
    import ast
    from pathlib import Path

    baselines_root = (
        Path(__file__).resolve().parents[1] / "src" / "koopman_graph" / "baselines"
    )
    peer_paths = [
        baselines_root / "dmd.py",
        baselines_root / "dmdc.py",
        baselines_root / "edmd.py",
        baselines_root / "gnn" / "base.py",
        baselines_root / "gnn" / "stgcn.py",
        baselines_root / "gnn" / "dcrnn.py",
        baselines_root / "gnn" / "wavenet.py",
    ]
    private_imports: list[str] = []
    for path in peer_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            if not node.module.startswith("koopman_graph.baselines"):
                continue
            for alias in node.names:
                if alias.name.startswith("_"):
                    private_imports.append(f"{path.name}:{node.module}.{alias.name}")
    assert private_imports == []


def test_gavish_donoho_omega_square_matches_published_constant() -> None:
    """ω(1) recovers the published unknown-σ square coefficient ≈ 2.858."""
    assert gavish_donoho_omega(1.0) == pytest.approx(2.858, abs=5e-3)


def test_optimal_hard_threshold_square_recovers_rank() -> None:
    """Seeded square low-rank-plus-noise matrix recovers the true rank."""
    torch.manual_seed(0)
    n = 80
    rank = 3
    u, _ = torch.linalg.qr(torch.randn(n, rank))
    v, _ = torch.linalg.qr(torch.randn(n, rank))
    signal = u @ torch.diag(torch.tensor([50.0, 40.0, 30.0])) @ v.T
    noisy = signal + 0.5 * torch.randn(n, n)
    singular_values = torch.linalg.svdvals(noisy)
    assert optimal_hard_threshold_rank(singular_values, num_rows=n, num_cols=n) == rank


def test_optimal_hard_threshold_nonsquare_and_square_shortcut_fails() -> None:
    """Non-square β-correction recovers rank; square constant under-selects."""
    torch.manual_seed(0)
    num_rows, num_cols, rank = 40, 100, 3
    beta = min(num_rows, num_cols) / max(num_rows, num_cols)
    u, _ = torch.linalg.qr(torch.randn(num_rows, rank))
    v, _ = torch.linalg.qr(torch.randn(num_cols, rank))
    amps = torch.tensor([30.0, 24.0, 18.0])
    signal = u @ torch.diag(amps) @ v.T
    noisy = signal + 1.0 * torch.randn(num_rows, num_cols)
    singular_values = torch.linalg.svdvals(noisy)

    selected = optimal_hard_threshold_rank(
        singular_values, num_rows=num_rows, num_cols=num_cols
    )
    assert selected == rank

    # Square-case shortcut ω≡2.858 is incorrect for β≠1.
    square_threshold = 2.858 * float(torch.median(singular_values).item())
    square_shortcut_rank = int((singular_values > square_threshold).sum().item())
    assert square_shortcut_rank != rank
    assert gavish_donoho_omega(beta) != pytest.approx(2.858, abs=1e-3)


def test_rank_auto_on_dmd_family_exposes_selected_rank(
    synthetic_edge_index: torch.Tensor,
) -> None:
    """rank='auto' is accepted by DMD/EDMD/DMDc and sets selected_rank."""
    operator = torch.diag(torch.tensor([0.9, 0.7, 0.5, 0.3], dtype=torch.float64))
    states = _linear_sequence(
        operator, torch.tensor([1.0, -0.5, 0.25, -0.125], dtype=torch.float64)
    )
    # Longer trajectory so the data matrix is well-conditioned for SVHT.
    for _ in range(40):
        states.append(states[-1] @ operator.T)
    sequence = _sequence_from_states(
        states,
        synthetic_edge_index,
        num_nodes=2,
        in_channels=2,
    )

    dmd = DMDBaseline(rank="auto").fit(sequence)
    assert dmd.selected_rank is not None
    assert dmd.selected_rank >= 1

    edmd = EDMDBaseline(rank="auto", polynomial_degree=1).fit(sequence)
    assert edmd.selected_rank is not None
    assert edmd.selected_rank >= 1

    controls = torch.zeros(len(states), 1, dtype=torch.float64)
    controlled = GraphSnapshotSequence(
        [
            Data(
                x=state.reshape(2, 2),
                edge_index=synthetic_edge_index,
            )
            for state in states
        ],
        control_inputs=controls,
    )
    dmdc = DMDcBaseline(rank="auto").fit(controlled)
    assert dmdc.selected_rank is not None
    assert dmdc.selected_rank >= 1


def test_rank_none_and_int_selected_rank_semantics(
    synthetic_edge_index: torch.Tensor,
) -> None:
    """None keeps selected_rank=None; int rank is recorded after fit."""
    sequence = _linear_fit_sequence(synthetic_edge_index)
    full = DMDBaseline(rank=None).fit(sequence)
    assert full.selected_rank is None
    truncated = DMDBaseline(rank=2).fit(sequence)
    assert truncated.selected_rank == 2


def test_rank_auto_rejects_all_zero_data_matrix() -> None:
    """All-zero left matrix yields rank-0 auto selection → ValueError."""
    left = torch.zeros(20, 10)
    right = torch.zeros(20, 10)
    from koopman_graph.baselines.base import fit_row_operator

    with pytest.raises(ValueError, match="rank='auto' selected rank 0"):
        fit_row_operator(left, right, "auto")
