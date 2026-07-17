"""Tests for Koopman spectral analysis utilities."""

import math

import pytest
import torch
from torch_geometric.data import Data

import koopman_graph.analysis as analysis
from koopman_graph import (
    GNNDecoder,
    GNNEncoder,
    GraphKoopmanModel,
    GraphSnapshotSequence,
    KoopmanSpectrum,
    compute_spectrum,
)
from koopman_graph.analysis import (
    calibrate_anomaly_threshold,
    decode_mode_shapes,
    detect_anomaly,
    dynamical_similarity,
    koopman_std,
    spectrum_distance,
)
from koopman_graph.baselines import DMDBaseline


@pytest.fixture
def graph_koopman_model() -> GraphKoopmanModel:
    """Provide a small model with deterministic Koopman dynamics."""
    model = GraphKoopmanModel(
        encoder=GNNEncoder(3, 8, 3),
        decoder=GNNDecoder(3, 8, 3),
        latent_dim=3,
        time_step=0.25,
        koopman_init_mode="identity",
    )
    with torch.no_grad():
        model.koopman.K.copy_(torch.diag(torch.tensor([0.5, 0.8, 1.0])))
    return model


def test_known_linear_system_spectrum() -> None:
    """Recover analytic eigenvalues and time scales of decay plus rotation."""
    radius = 0.9
    angle = math.pi / 4
    operator = torch.tensor(
        [
            [0.5, 0.0, 0.0],
            [0.0, radius * math.cos(angle), -radius * math.sin(angle)],
            [0.0, radius * math.sin(angle), radius * math.cos(angle)],
        ],
        dtype=torch.float64,
    )

    spectrum = compute_spectrum(operator, time_step=0.5)

    assert isinstance(spectrum, KoopmanSpectrum)
    assert spectrum.eigenvalues.dtype == torch.complex128
    assert torch.allclose(
        spectrum.magnitudes,
        torch.tensor([radius, radius, 0.5], dtype=torch.float64),
    )
    assert torch.allclose(
        spectrum.growth_rates[:2],
        torch.full((2,), math.log(radius) / 0.5, dtype=torch.float64),
    )
    assert spectrum.growth_rates[2].item() == pytest.approx(math.log(0.5) / 0.5)
    assert sorted(spectrum.frequencies[:2].tolist()) == pytest.approx([-0.25, 0.25])
    assert spectrum.frequencies[2].item() == pytest.approx(0.0)

    residual = (
        operator.to(spectrum.eigenvectors.dtype) @ spectrum.eigenvectors
        - spectrum.eigenvectors * spectrum.eigenvalues
    )
    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-12)


def test_spectrum_sorts_modes_by_descending_magnitude() -> None:
    """Sort eigenpairs consistently by decreasing magnitude."""
    operator = torch.diag(torch.tensor([0.2, 1.2, 0.7]))
    spectrum = compute_spectrum(operator, time_step=1.0)

    assert torch.allclose(spectrum.magnitudes, torch.tensor([1.2, 0.7, 0.2]))
    assert torch.allclose(spectrum.eigenvalues.real, torch.tensor([1.2, 0.7, 0.2]))


def test_zero_eigenvalue_has_negative_infinite_growth_rate() -> None:
    """Represent a zero eigenvalue as instantaneous asymptotic decay."""
    spectrum = compute_spectrum(torch.diag(torch.tensor([1.0, 0.0])), 1.0)

    assert spectrum.growth_rates[0] == 0
    assert torch.isneginf(spectrum.growth_rates[1])


@pytest.mark.parametrize(
    ("operator", "message"),
    [
        (torch.ones(2, 3), "square matrix"),
        (torch.empty(0, 0), "non-empty"),
    ],
)
def test_compute_spectrum_rejects_invalid_shape(
    operator: torch.Tensor,
    message: str,
) -> None:
    """Reject nonsquare and empty operator matrices."""
    with pytest.raises(ValueError, match=message):
        compute_spectrum(operator, 1.0)


def test_compute_spectrum_rejects_nonpositive_time_step() -> None:
    """Require a positive conversion interval."""
    with pytest.raises(ValueError, match="time_step must be positive"):
        compute_spectrum(torch.eye(2), 0.0)


def test_compute_spectrum_rejects_integer_operator() -> None:
    """Reject dtypes unsupported by eigendecomposition."""
    with pytest.raises(TypeError, match="floating-point or complex"):
        compute_spectrum(torch.eye(2, dtype=torch.int64), 1.0)


def test_mode_amplitudes_reconstruct_latent_states() -> None:
    """Project batched states into modes without losing leading dimensions."""
    operator = torch.tensor([[0.8, -0.2], [0.2, 0.8]])
    spectrum = compute_spectrum(operator, 0.1)
    states = torch.randn(2, 4, 2)

    amplitudes = spectrum.mode_amplitudes(states)
    reconstructed = (spectrum.eigenvectors @ amplitudes.reshape(-1, 2).T).T.reshape(
        states.shape
    )

    assert amplitudes.shape == states.shape
    assert amplitudes.is_complex()
    assert torch.allclose(reconstructed, states.to(torch.complex64), atol=1e-5)


@pytest.mark.parametrize("latent_states", [torch.tensor(1.0), torch.ones(3, 4)])
def test_mode_amplitudes_reject_wrong_shape(latent_states: torch.Tensor) -> None:
    """Reject scalar or mismatched latent inputs."""
    spectrum = compute_spectrum(torch.eye(2), 1.0)
    with pytest.raises(ValueError, match="trailing dimension 2"):
        spectrum.mode_amplitudes(latent_states)


def test_model_spectrum_uses_model_time_step(
    graph_koopman_model: GraphKoopmanModel,
) -> None:
    """Use model ``time_step`` for continuous-time conversion."""
    spectrum = graph_koopman_model.spectrum()

    assert spectrum.time_step == 0.25
    assert torch.allclose(
        spectrum.growth_rates,
        torch.log(torch.tensor([1.0, 0.8, 0.5])) / 0.25,
    )


def test_decode_mode_shapes_contract_and_mode_restoration(
    graph_koopman_model: GraphKoopmanModel,
    synthetic_graph: Data,
) -> None:
    """Decode selected modes without gradients and restore training mode."""
    graph_koopman_model.train()
    shapes = decode_mode_shapes(
        graph_koopman_model,
        synthetic_graph,
        mode_indices=[0, 2],
        perturbation=1e-2,
    )

    assert shapes.shape == (2, synthetic_graph.num_nodes, 3)
    assert shapes.is_complex()
    assert shapes.grad_fn is None
    assert torch.isfinite(shapes).all()
    assert graph_koopman_model.training


def test_decode_mode_shapes_matches_linear_decoder_jacobian(
    synthetic_graph: Data,
) -> None:
    """Centered FD mode shapes match the exact Jacobian of a linear decoder."""
    latent_dim = 3
    out_channels = 2
    model = GraphKoopmanModel(
        encoder=GNNEncoder(3, 8, latent_dim),
        decoder=GNNDecoder(latent_dim, 8, out_channels),
        latent_dim=latent_dim,
        time_step=0.25,
        koopman_init_mode="identity",
    )
    weight = torch.tensor(
        [[1.0, -0.5], [0.25, 2.0], [-1.0, 0.5]],
        dtype=torch.float32,
    )

    class _LinearDecoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.latent_dim = latent_dim
            self.out_channels = out_channels
            self.weight = torch.nn.Parameter(weight.clone())

        def forward(
            self,
            latent: torch.Tensor,
            edge_index: torch.Tensor,
            edge_weight: torch.Tensor | None = None,
        ) -> torch.Tensor:
            del edge_index, edge_weight
            return latent @ self.weight

    model.decoder = _LinearDecoder()
    with torch.no_grad():
        model.koopman.K.copy_(torch.diag(torch.tensor([0.5, 0.8, 1.0])))

    shapes = decode_mode_shapes(
        model,
        synthetic_graph,
        mode_indices=[0, 1],
        perturbation=1e-3,
    )
    spectrum = model.spectrum()
    for shape_index, mode_index in enumerate([0, 1]):
        direction = spectrum.eigenvectors[:, mode_index]
        direction = direction / direction.norm()
        expected = torch.complex(
            direction.real.to(weight.dtype) @ weight,
            direction.imag.to(weight.dtype) @ weight,
        )
        # Same latent direction is applied at every node for the FD probe.
        expected_nodes = expected.unsqueeze(0).expand(synthetic_graph.num_nodes, -1)
        # float32 centered FD has O(eps_mach / perturbation) cancellation noise.
        assert torch.allclose(shapes[shape_index], expected_nodes, atol=1e-4)


def test_decode_mode_shapes_accepts_tensor_input(
    graph_koopman_model: GraphKoopmanModel,
    synthetic_graph: Data,
) -> None:
    """Accept separate feature and edge tensors and default to all modes."""
    graph_koopman_model.eval()
    shapes = decode_mode_shapes(
        graph_koopman_model,
        synthetic_graph.x,
        edge_index=synthetic_graph.edge_index,
    )

    assert shapes.shape == (3, synthetic_graph.num_nodes, 3)
    assert not graph_koopman_model.training


def test_decode_mode_shapes_accepts_empty_mode_selection(
    graph_koopman_model: GraphKoopmanModel,
    synthetic_graph: Data,
) -> None:
    """Return a correctly shaped empty result for no selected modes."""
    shapes = decode_mode_shapes(graph_koopman_model, synthetic_graph, [])
    assert shapes.shape == (0, synthetic_graph.num_nodes, 3)
    assert shapes.dtype == torch.complex64


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"mode_indices": [3]}, "mode_indices"),
        ({"perturbation": 0.0}, "perturbation must be positive"),
    ],
)
def test_decode_mode_shapes_rejects_invalid_arguments(
    graph_koopman_model: GraphKoopmanModel,
    synthetic_graph: Data,
    kwargs: dict[str, object],
    message: str,
) -> None:
    """Reject invalid finite-difference and mode-selection arguments."""
    with pytest.raises(ValueError, match=message):
        decode_mode_shapes(graph_koopman_model, synthetic_graph, **kwargs)


def test_public_analysis_exports() -> None:
    """Expose spectral analysis from the package root."""
    assert analysis.KoopmanSpectrum is KoopmanSpectrum
    assert analysis.compute_spectrum is compute_spectrum
    assert analysis.decode_mode_shapes is decode_mode_shapes
    assert analysis.spectrum_distance is spectrum_distance
    assert analysis.koopman_std is koopman_std
    assert analysis.dynamical_similarity is dynamical_similarity
    assert analysis.detect_anomaly is detect_anomaly
    assert analysis.calibrate_anomaly_threshold is calibrate_anomaly_threshold


def _make_spectrum(magnitudes: list[float], time_step: float = 1.0) -> KoopmanSpectrum:
    operator = torch.diag(torch.tensor(magnitudes, dtype=torch.float64))
    return compute_spectrum(operator, time_step)


def test_spectrum_distance_zero_for_identical_spectra() -> None:
    """Identical spectra yield zero distance for both methods."""
    spectrum = _make_spectrum([0.9, 0.7, 0.5])

    wasserstein = spectrum_distance(spectrum, spectrum, "wasserstein").item()
    subspace = spectrum_distance(spectrum, spectrum, "subspace_angle").item()
    assert wasserstein == pytest.approx(0.0)
    assert subspace == pytest.approx(0.0)


def test_spectrum_distance_wasserstein_known_scaled_magnitudes() -> None:
    """Wasserstein distance matches mean absolute gap after zero-padding."""
    spectrum_a = _make_spectrum([0.9, 0.5])
    spectrum_b = _make_spectrum([0.7, 0.3, 0.1])

    distance = spectrum_distance(spectrum_a, spectrum_b, "wasserstein")
    expected = torch.mean(torch.abs(torch.tensor([0.9 - 0.7, 0.5 - 0.3, 0.0 - 0.1])))
    assert distance.item() == pytest.approx(expected.item())


def test_spectrum_distance_subspace_angle_detects_rotation() -> None:
    """Dominant eigenvector directions at right angles yield a large principal angle."""
    spectrum_a = compute_spectrum(torch.diag(torch.tensor([0.9, 0.5])), 1.0)
    spectrum_b = compute_spectrum(torch.diag(torch.tensor([0.5, 0.9])), 1.0)

    distance = spectrum_distance(spectrum_a, spectrum_b, "subspace_angle", num_modes=1)
    assert distance.item() == pytest.approx(math.pi / 2, abs=1e-5)


def test_spectrum_distance_rejects_unknown_method() -> None:
    """Reject unsupported distance methods."""
    spectrum = _make_spectrum([0.8])
    with pytest.raises(ValueError, match="method must be"):
        spectrum_distance(spectrum, spectrum, "invalid")  # type: ignore[arg-type]


def test_spectrum_distance_num_modes_limits_subspace_comparison() -> None:
    """Compare only the requested number of leading modes."""
    spectrum_a = compute_spectrum(torch.diag(torch.tensor([0.9, 0.7, 0.5])), 1.0)
    spectrum_b = compute_spectrum(torch.diag(torch.tensor([0.5, 0.9, 0.7])), 1.0)

    full = spectrum_distance(spectrum_a, spectrum_b, "subspace_angle")
    partial = spectrum_distance(spectrum_a, spectrum_b, "subspace_angle", num_modes=1)
    assert partial.item() == pytest.approx(math.pi / 2, abs=1e-5)
    assert full.item() < partial.item()


def test_koopman_std_symmetry_and_zero_diagonal() -> None:
    """Pairwise matrix is symmetric with a zero diagonal."""
    spectra = [
        _make_spectrum([0.9, 0.7]),
        _make_spectrum([0.8, 0.6]),
        _make_spectrum([0.5, 0.4]),
    ]
    matrix = koopman_std(spectra, "wasserstein")

    assert matrix.shape == (3, 3)
    assert torch.allclose(matrix, matrix.T)
    assert torch.allclose(torch.diag(matrix), torch.zeros(3, dtype=torch.float64))


def test_spectrum_distance_detaches_from_autograd(
    graph_koopman_model: GraphKoopmanModel,
) -> None:
    """Distance helpers return plain tensors safe for ``.numpy()``."""
    spectrum = graph_koopman_model.spectrum()
    assert spectrum.magnitudes.requires_grad

    distance = spectrum_distance(spectrum, spectrum, "wasserstein")
    matrix = koopman_std([spectrum, spectrum], "wasserstein")

    assert not distance.requires_grad
    assert not matrix.requires_grad
    assert matrix.numpy().shape == (2, 2)


def test_dynamical_similarity_between_models(
    graph_koopman_model: GraphKoopmanModel,
) -> None:
    """Compare two models via their learned operator spectra."""
    other = GraphKoopmanModel(
        encoder=GNNEncoder(3, 8, 3),
        decoder=GNNDecoder(3, 8, 3),
        latent_dim=3,
        time_step=0.25,
        koopman_init_mode="identity",
    )
    with torch.no_grad():
        other.koopman.K.copy_(torch.diag(torch.tensor([0.4, 0.6, 0.9])))

    distance = dynamical_similarity(graph_koopman_model, other, "wasserstein")
    assert distance.item() > 0.0


def test_dynamical_similarity_accepts_precomputed_spectra(
    graph_koopman_model: GraphKoopmanModel,
) -> None:
    """Precomputed spectra match spectrum_distance and model-provider calls."""
    other = GraphKoopmanModel(
        encoder=GNNEncoder(3, 8, 3),
        decoder=GNNDecoder(3, 8, 3),
        latent_dim=3,
        time_step=0.25,
        koopman_init_mode="identity",
    )
    with torch.no_grad():
        other.koopman.K.copy_(torch.diag(torch.tensor([0.4, 0.6, 0.9])))

    spectrum_a = graph_koopman_model.spectrum()
    spectrum_b = other.spectrum()
    expected = spectrum_distance(spectrum_a, spectrum_b, "wasserstein")

    assert dynamical_similarity(spectrum_a, spectrum_b).item() == pytest.approx(
        expected.item()
    )
    mixed = dynamical_similarity(graph_koopman_model, spectrum_b)
    assert mixed.item() == pytest.approx(expected.item())


def test_dynamical_similarity_baseline_and_model_peers(
    graph_koopman_model: GraphKoopmanModel,
) -> None:
    """Baselines compare to each other and to GraphKoopmanModel via spectra."""
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    operator_a = torch.diag(torch.tensor([0.5, 0.8], dtype=torch.float64))
    operator_b = torch.diag(torch.tensor([0.3, 0.6], dtype=torch.float64))
    sequence_a = GraphSnapshotSequence(
        [
            Data(x=state.reshape(2, 1), edge_index=edge_index)
            for state in _linear_flattened_states(
                operator_a,
                torch.tensor([1.0, -0.5], dtype=torch.float64),
            )
        ]
    )
    sequence_b = GraphSnapshotSequence(
        [
            Data(x=state.reshape(2, 1), edge_index=edge_index)
            for state in _linear_flattened_states(
                operator_b,
                torch.tensor([1.0, -0.5], dtype=torch.float64),
            )
        ]
    )
    dmd_a = DMDBaseline(time_step=0.25).fit(sequence_a)
    dmd_b = DMDBaseline(time_step=0.25).fit(sequence_b)

    baseline_distance = dynamical_similarity(dmd_a, dmd_b, "wasserstein")
    expected = spectrum_distance(dmd_a.spectrum(), dmd_b.spectrum(), "wasserstein")
    assert baseline_distance.item() == pytest.approx(expected.item())
    assert baseline_distance.item() > 0.0

    # delta_t must not break baselines that reject spectrum kwargs.
    mixed = dynamical_similarity(dmd_a, graph_koopman_model, delta_t=0.1)
    assert mixed.item() >= 0.0


def test_resolve_spectrum_rejects_unknown_source() -> None:
    """Reject objects that are neither spectra nor spectrum providers."""
    with pytest.raises(TypeError, match="KoopmanSpectrum or SpectrumProvider"):
        analysis.resolve_spectrum(object())  # type: ignore[arg-type]


def _linear_flattened_states(
    operator: torch.Tensor,
    initial_state: torch.Tensor,
) -> list[torch.Tensor]:
    """Generate flattened states following ``x_next = x @ K.T``."""
    states = [initial_state]
    for _ in range(5):
        states.append(states[-1] @ operator.T)
    return states


def test_detect_anomaly_flags_shifted_eigenvalues() -> None:
    """Injected eigenvalue shift is flagged above a tight threshold."""
    references = [_make_spectrum([0.9, 0.7]), _make_spectrum([0.88, 0.72])]
    normal = _make_spectrum([0.89, 0.71])
    shifted = _make_spectrum([0.5, 0.3])

    normal_result = detect_anomaly(references, normal, threshold=0.05)
    shifted_result = detect_anomaly(references, shifted, threshold=0.05)

    assert not normal_result.is_anomaly
    assert shifted_result.is_anomaly
    assert shifted_result.distance > normal_result.distance
    assert shifted_result.reference_mean_distance >= 0.0


def test_detect_anomaly_identical_references_mean_distance_is_zero() -> None:
    """Identical references must report finite zero pairwise mean distance."""
    spectrum = _make_spectrum([0.9, 0.7])
    result = detect_anomaly([spectrum, spectrum], spectrum, threshold=0.1)
    assert result.reference_mean_distance == pytest.approx(0.0)
    assert result.distance == pytest.approx(0.0)
    assert not result.is_anomaly


def test_detect_anomaly_requires_nonempty_references() -> None:
    """Validate reference set and threshold arguments."""
    spectrum = _make_spectrum([0.8])
    with pytest.raises(ValueError, match="reference_spectra must be non-empty"):
        detect_anomaly([], spectrum, threshold=0.1)
    with pytest.raises(ValueError, match="threshold must be >= 0"):
        detect_anomaly([spectrum], spectrum, threshold=-0.1)


def test_calibrate_anomaly_threshold_percentile_and_mean_std() -> None:
    """Percentile and mean_std summaries match hand-computed pairwise stats."""
    references = [
        _make_spectrum([0.90, 0.70]),
        _make_spectrum([0.88, 0.72]),
        _make_spectrum([0.89, 0.71]),
    ]
    matrix = koopman_std(references, "wasserstein")
    pairwise = matrix[torch.triu(torch.ones_like(matrix, dtype=torch.bool), diagonal=1)]

    percentile = calibrate_anomaly_threshold(references, "percentile", q=95)
    mean_std = calibrate_anomaly_threshold(references, "mean_std", k=2.0)

    assert percentile == pytest.approx(torch.quantile(pairwise, 0.95).item())
    expected_mean_std = pairwise.mean() + 2.0 * pairwise.std(unbiased=False)
    assert mean_std == pytest.approx(expected_mean_std.item())


def test_calibrate_anomaly_threshold_flags_shift_and_spares_nominal() -> None:
    """Calibrated threshold flags a shift while held-out nominal stays clear."""
    references = [
        _make_spectrum([0.90, 0.70]),
        _make_spectrum([0.88, 0.72]),
        _make_spectrum([0.89, 0.71]),
    ]
    held_out_nominal = _make_spectrum([0.895, 0.705])
    shifted = _make_spectrum([0.50, 0.30])

    threshold = calibrate_anomaly_threshold(references, "percentile", q=95)
    nominal_result = detect_anomaly(references, held_out_nominal, threshold=threshold)
    shifted_result = detect_anomaly(references, shifted, threshold=threshold)

    assert not nominal_result.is_anomaly
    assert shifted_result.is_anomaly
    assert shifted_result.distance > nominal_result.distance


def test_calibrate_anomaly_threshold_rejects_invalid_arguments() -> None:
    """Reject undersized reference sets and invalid method kwargs."""
    spectrum = _make_spectrum([0.8, 0.6])
    with pytest.raises(ValueError, match="at least two spectra"):
        calibrate_anomaly_threshold([spectrum], "percentile")
    with pytest.raises(ValueError, match="q must be in"):
        calibrate_anomaly_threshold([spectrum, spectrum], "percentile", q=101)
    with pytest.raises(ValueError, match="k must be >= 0"):
        calibrate_anomaly_threshold([spectrum, spectrum], "mean_std", k=-1)
    with pytest.raises(ValueError, match="method must be"):
        calibrate_anomaly_threshold([spectrum, spectrum], "max_margin")  # type: ignore[arg-type]


def test_koopman_std_rejects_empty_input() -> None:
    """Reject empty spectrum collections."""
    with pytest.raises(ValueError, match="spectra must be non-empty"):
        koopman_std([])
