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
    KoopmanSpectrum,
    compute_spectrum,
    decode_mode_shapes,
)


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
