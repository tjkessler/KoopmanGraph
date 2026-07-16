"""Spectral analysis utilities for finite-dimensional Koopman operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
from torch import Tensor
from torch_geometric.data import Data

if TYPE_CHECKING:
    from collections.abc import Sequence

    from koopman_graph.model import GraphKoopmanModel

SpectrumDistanceMethod = Literal["wasserstein", "subspace_angle"]
AnomalyThresholdMethod = Literal["percentile", "mean_std"]


@dataclass(frozen=True)
class KoopmanSpectrum:
    """Eigendecomposition and time scales of a discrete Koopman operator.

    Eigenpairs are sorted by descending eigenvalue magnitude. Frequencies are
    reported in cycles per unit time; multiply by ``2 * pi`` for angular
    frequency.

    Attributes
    ----------
    eigenvalues : Tensor
        Complex eigenvalues with shape ``(latent_dim,)``.
    eigenvectors : Tensor
        Complex right eigenvectors stored as columns, with shape
        ``(latent_dim, latent_dim)``.
    magnitudes : Tensor
        Eigenvalue magnitudes with shape ``(latent_dim,)``.
    growth_rates : Tensor
        Continuous-time exponential growth rates ``log(|lambda|) / time_step``.
    frequencies : Tensor
        Signed continuous-time frequencies
        ``angle(lambda) / (2 * pi * time_step)`` in cycles per unit time.
    time_step : float
        Physical duration represented by one discrete Koopman step.
    """

    eigenvalues: Tensor
    eigenvectors: Tensor
    magnitudes: Tensor
    growth_rates: Tensor
    frequencies: Tensor
    time_step: float

    def mode_amplitudes(self, latent_states: Tensor) -> Tensor:
        """Project latent states onto the Koopman eigenvector basis.

        For a latent row vector ``z``, the returned amplitudes ``a`` satisfy
        ``z.T = eigenvectors @ a``. Any leading dimensions are preserved.

        Parameters
        ----------
        latent_states : Tensor
            Latent states with shape ``(..., latent_dim)``.

        Returns
        -------
        Tensor
            Complex mode amplitudes with the same shape as ``latent_states``.

        Raises
        ------
        ValueError
            If the trailing latent dimension does not match the spectrum.
        RuntimeError
            If the eigenvector matrix is singular.
        """
        latent_dim = self.eigenvectors.shape[0]
        if latent_states.ndim == 0 or latent_states.shape[-1] != latent_dim:
            msg = (
                f"Expected trailing dimension {latent_dim}, "
                f"got shape {tuple(latent_states.shape)}"
            )
            raise ValueError(msg)

        vectors = self.eigenvectors.to(device=latent_states.device)
        states = latent_states.to(dtype=vectors.dtype)
        flat_states = states.reshape(-1, latent_dim)
        amplitudes = torch.linalg.solve(vectors, flat_states.T).T
        return amplitudes.reshape(latent_states.shape)


def compute_spectrum(operator: Tensor, time_step: float) -> KoopmanSpectrum:
    """Compute the sorted spectrum and continuous-time mode characteristics.

    Parameters
    ----------
    operator : Tensor
        Square discrete-time Koopman matrix with shape
        ``(latent_dim, latent_dim)``.
    time_step : float
        Positive physical duration represented by one operator step.

    Returns
    -------
    KoopmanSpectrum
        Eigenpairs sorted by descending magnitude, plus growth rates and
        frequencies converted using ``time_step``.

    Raises
    ------
    ValueError
        If ``operator`` is not a non-empty square matrix or ``time_step`` is
        not positive.
    TypeError
        If ``operator`` is not floating-point or complex.
    """
    if operator.ndim != 2 or operator.shape[0] != operator.shape[1]:
        msg = f"operator must be a square matrix, got shape {tuple(operator.shape)}"
        raise ValueError(msg)
    if operator.shape[0] == 0:
        raise ValueError("operator must be non-empty")
    if time_step <= 0:
        msg = f"time_step must be positive, got {time_step}"
        raise ValueError(msg)
    if not (operator.is_floating_point() or operator.is_complex()):
        msg = f"operator must be floating-point or complex, got {operator.dtype}"
        raise TypeError(msg)

    eigenvalues, eigenvectors = torch.linalg.eig(operator)
    magnitudes = eigenvalues.abs()
    order = torch.argsort(magnitudes, descending=True)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    magnitudes = magnitudes[order]

    growth_rates = torch.log(magnitudes) / time_step
    frequencies = torch.angle(eigenvalues) / (2 * torch.pi * time_step)
    return KoopmanSpectrum(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        magnitudes=magnitudes,
        growth_rates=growth_rates,
        frequencies=frequencies,
        time_step=float(time_step),
    )


def compute_generator_spectrum(generator: Tensor) -> KoopmanSpectrum:
    """Compute the sorted spectrum of a continuous-time Koopman generator.

    Growth rates are the real parts of the eigenvalues; frequencies are the
    imaginary parts scaled to cycles per unit time.

    Parameters
    ----------
    generator : Tensor
        Square generator matrix with shape ``(latent_dim, latent_dim)``.

    Returns
    -------
    KoopmanSpectrum
        Eigenpairs sorted by descending magnitude with native continuous-time
        growth rates and frequencies.

    Raises
    ------
    ValueError
        If ``generator`` is not a non-empty square matrix.
    TypeError
        If ``generator`` is not floating-point or complex.
    """
    if generator.ndim != 2 or generator.shape[0] != generator.shape[1]:
        msg = f"generator must be a square matrix, got shape {tuple(generator.shape)}"
        raise ValueError(msg)
    if generator.shape[0] == 0:
        raise ValueError("generator must be non-empty")
    if not (generator.is_floating_point() or generator.is_complex()):
        msg = f"generator must be floating-point or complex, got {generator.dtype}"
        raise TypeError(msg)

    eigenvalues, eigenvectors = torch.linalg.eig(generator)
    magnitudes = eigenvalues.abs()
    order = torch.argsort(magnitudes, descending=True)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    magnitudes = magnitudes[order]

    growth_rates = eigenvalues.real
    frequencies = eigenvalues.imag / (2 * torch.pi)
    return KoopmanSpectrum(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        magnitudes=magnitudes,
        growth_rates=growth_rates,
        frequencies=frequencies,
        time_step=1.0,
    )


def discrete_spectrum_at_delta_t(
    generator: Tensor,
    delta_t: float,
) -> KoopmanSpectrum:
    """Compute the spectrum of ``exp(L · Δt)`` for a generator ``L``.

    Parameters
    ----------
    generator : Tensor
        Continuous-time generator matrix.
    delta_t : float
        Integration interval.

    Returns
    -------
    KoopmanSpectrum
        Discrete-time spectrum at horizon ``delta_t``.

    Raises
    ------
    ValueError
        If ``delta_t`` is not positive.
    """
    if delta_t <= 0:
        msg = f"delta_t must be positive, got {delta_t}"
        raise ValueError(msg)
    transition = torch.linalg.matrix_exp(generator * delta_t)
    return compute_spectrum(transition, delta_t)


def decode_mode_shapes(
    model: GraphKoopmanModel,
    x_or_data: Tensor | Data,
    mode_indices: Sequence[int] | None = None,
    *,
    edge_index: Tensor | None = None,
    perturbation: float = 1e-3,
) -> Tensor:
    """Decode latent Koopman directions into spatial node-feature mode shapes.

    The decoder is generally nonlinear, so mode shapes are estimated with a
    centered finite-difference directional derivative around the encoded graph.
    Real and imaginary parts of complex eigenvectors are probed separately and
    combined into a complex-valued mode shape.

    Parameters
    ----------
    model : GraphKoopmanModel
        Model whose operator spectrum and decoder are analyzed.
    x_or_data : Tensor or Data
        Reference graph used as the decoder linearization point.
    mode_indices : sequence of int or None, optional
        Indices into the magnitude-sorted spectrum. Defaults to every mode.
    edge_index : Tensor or None, optional
        Graph edges, required when ``x_or_data`` is a feature tensor.
    perturbation : float, optional
        Positive centered finite-difference step. Default is ``1e-3``.

    Returns
    -------
    Tensor
        Complex mode shapes with shape
        ``(num_modes, num_nodes, out_channels)``.

    Raises
    ------
    ValueError
        If ``perturbation`` is not positive or a mode index is out of range.
    """
    if perturbation <= 0:
        msg = f"perturbation must be positive, got {perturbation}"
        raise ValueError(msg)

    spectrum = model.spectrum()
    latent_dim = spectrum.eigenvalues.numel()
    indices = list(range(latent_dim)) if mode_indices is None else list(mode_indices)
    if any(index < 0 or index >= latent_dim for index in indices):
        msg = f"mode_indices must be between 0 and {latent_dim - 1}, got {indices}"
        raise ValueError(msg)

    edges = model._resolve_edge_index(x_or_data, edge_index)
    edge_weight = model._resolve_edge_weight(x_or_data, None)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            latent = model.encode(x_or_data, edges, edge_weight)
            mode_shapes = [
                _decode_complex_direction(
                    model,
                    latent,
                    edges,
                    edge_weight,
                    spectrum.eigenvectors[:, index],
                    perturbation,
                )
                for index in indices
            ]
    finally:
        model.train(was_training)

    if mode_shapes:
        return torch.stack(mode_shapes)
    output_shape = (0, latent.shape[0], model.decoder.out_channels)
    return torch.empty(
        output_shape,
        dtype=spectrum.eigenvalues.dtype,
        device=latent.device,
    )


def _decode_complex_direction(
    model: GraphKoopmanModel,
    latent: Tensor,
    edge_index: Tensor,
    edge_weight: Tensor | None,
    direction: Tensor,
    perturbation: float,
) -> Tensor:
    """Estimate decoder response to one complex latent direction.

    Parameters
    ----------
    model : GraphKoopmanModel
        Model providing the decoder.
    latent : Tensor
        Encoded reference state.
    edge_index : Tensor
        Graph connectivity.
    direction : Tensor
        Complex latent eigenvector.
    perturbation : float
        Centered finite-difference step.

    Returns
    -------
    Tensor
        Complex node-feature response.
    """
    direction = direction.to(device=latent.device)
    minimum_norm = torch.finfo(direction.real.dtype).eps
    direction = direction / direction.norm().clamp_min(minimum_norm)
    real_shape = _decode_real_direction(
        model,
        latent,
        edge_index,
        edge_weight,
        direction.real.to(latent.dtype),
        perturbation,
    )
    imag_shape = _decode_real_direction(
        model,
        latent,
        edge_index,
        edge_weight,
        direction.imag.to(latent.dtype),
        perturbation,
    )
    return torch.complex(real_shape, imag_shape)


def _decode_real_direction(
    model: GraphKoopmanModel,
    latent: Tensor,
    edge_index: Tensor,
    edge_weight: Tensor | None,
    direction: Tensor,
    perturbation: float,
) -> Tensor:
    """Estimate decoder response to one real latent direction.

    Parameters
    ----------
    model : GraphKoopmanModel
        Model providing the decoder.
    latent : Tensor
        Encoded reference state.
    edge_index : Tensor
        Graph connectivity.
    direction : Tensor
        Real latent direction.
    perturbation : float
        Centered finite-difference step.

    Returns
    -------
    Tensor
        Real node-feature response.
    """
    if not torch.count_nonzero(direction):
        return torch.zeros(
            (latent.shape[0], model.decoder.out_channels),
            dtype=latent.dtype,
            device=latent.device,
        )
    plus = model.decoder(latent + perturbation * direction, edge_index, edge_weight)
    minus = model.decoder(latent - perturbation * direction, edge_index, edge_weight)
    return (plus - minus) / (2 * perturbation)


@dataclass(frozen=True)
class AnomalyDetectionResult:
    """Outcome of comparing a test spectrum against reference spectra.

    Attributes
    ----------
    is_anomaly : bool
        ``True`` when ``distance`` exceeds ``threshold``.
    distance : float
        Mean spectral distance from ``test_spectrum`` to each reference.
    reference_mean_distance : float
        Mean pairwise distance among reference spectra (baseline spread).
        Useful for calibrating ``threshold`` from in-distribution variation.
    """

    is_anomaly: bool
    distance: float
    reference_mean_distance: float


def spectrum_distance(
    spectrum_a: KoopmanSpectrum,
    spectrum_b: KoopmanSpectrum,
    method: SpectrumDistanceMethod = "wasserstein",
    *,
    num_modes: int | None = None,
) -> Tensor:
    """Measure dynamical dissimilarity between two Koopman spectra.

    Parameters
    ----------
    spectrum_a, spectrum_b : KoopmanSpectrum
        Spectra to compare. May differ in ``latent_dim``; shorter spectra are
        zero-padded when comparing magnitudes or subspaces.
    method : {"wasserstein", "subspace_angle"}, optional
        Distance definition. ``"wasserstein"`` applies 1D Wasserstein-1 to
        sorted eigenvalue magnitudes. ``"subspace_angle"`` returns the mean
        principal angle (radians) between dominant eigenvector subspaces.
        Default is ``"wasserstein"``.
    num_modes : int or None, optional
        Number of leading modes to compare for ``"subspace_angle"``. Defaults
        to ``min(latent_dim_a, latent_dim_b)``. Ignored by ``"wasserstein"``.

    Returns
    -------
    Tensor
        Scalar distance (magnitude units for Wasserstein, radians for subspace
        angle). The result is detached from the autograd graph.

    Raises
    ------
    ValueError
        If ``method`` is unknown or ``num_modes`` is out of range.
    """
    if method == "wasserstein":
        return _wasserstein_magnitude_distance(
            spectrum_a.magnitudes,
            spectrum_b.magnitudes,
        )
    if method == "subspace_angle":
        modes = _resolve_num_modes(
            spectrum_a.eigenvectors.shape[0],
            spectrum_b.eigenvectors.shape[0],
            num_modes,
        )
        return _subspace_angle_distance(
            spectrum_a.eigenvectors[:, :modes],
            spectrum_b.eigenvectors[:, :modes],
        )
    msg = f"method must be 'wasserstein' or 'subspace_angle', got {method!r}"
    raise ValueError(msg)


def koopman_std(
    spectra: Sequence[KoopmanSpectrum],
    method: SpectrumDistanceMethod = "wasserstein",
    *,
    num_modes: int | None = None,
) -> Tensor:
    """Build a pairwise dynamical-similarity distance matrix (KoopSTD).

    Parameters
    ----------
    spectra : sequence of KoopmanSpectrum
        Spectra to compare, one per trajectory, model, or regime.
    method : {"wasserstein", "subspace_angle"}, optional
        Distance passed to :func:`spectrum_distance`. Default is
        ``"wasserstein"``.
    num_modes : int or None, optional
        Leading modes for subspace-angle comparisons. Default is the minimum
        latent dimension across all spectra.

    Returns
    -------
    Tensor
        Symmetric matrix with shape ``(len(spectra), len(spectra))`` and zero
        diagonal.

    Raises
    ------
    ValueError
        If ``spectra`` is empty.
    """
    if not spectra:
        raise ValueError("spectra must be non-empty")

    count = len(spectra)
    matrix = torch.zeros((count, count), dtype=torch.float64)
    for row in range(count):
        for col in range(row + 1, count):
            distance = spectrum_distance(
                spectra[row],
                spectra[col],
                method,
                num_modes=num_modes,
            )
            value = distance.to(dtype=matrix.dtype, device=matrix.device)
            matrix[row, col] = value
            matrix[col, row] = value
    return matrix


def dynamical_similarity(
    model_a: GraphKoopmanModel,
    model_b: GraphKoopmanModel,
    method: SpectrumDistanceMethod = "wasserstein",
    *,
    num_modes: int | None = None,
    delta_t: float | None = None,
) -> Tensor:
    """Compare learned dynamics of two GraphKoopmanModel instances.

    Parameters
    ----------
    model_a, model_b : GraphKoopmanModel
        Models whose Koopman operators are compared via
        :meth:`~koopman_graph.model.GraphKoopmanModel.spectrum`.
    method : {"wasserstein", "subspace_angle"}, optional
        Distance definition. Default is ``"wasserstein"``.
    num_modes : int or None, optional
        Leading modes for subspace-angle comparisons.
    delta_t : float or None, optional
        Integration horizon forwarded to
        :meth:`~koopman_graph.model.GraphKoopmanModel.spectrum` for
        continuous-time models.

    Returns
    -------
    Tensor
        Scalar spectral distance between the two models.
    """
    return spectrum_distance(
        model_a.spectrum(delta_t=delta_t),
        model_b.spectrum(delta_t=delta_t),
        method,
        num_modes=num_modes,
    )


def calibrate_anomaly_threshold(
    reference_spectra: Sequence[KoopmanSpectrum],
    method: AnomalyThresholdMethod = "percentile",
    *,
    distance_method: SpectrumDistanceMethod = "wasserstein",
    num_modes: int | None = None,
    q: float = 95.0,
    k: float = 2.0,
) -> float:
    """Derive an anomaly threshold from pairwise reference-spectrum distances.

    Builds the upper triangle of :func:`koopman_std` among
    ``reference_spectra`` and summarizes those in-distribution distances.
    Pass the returned scalar to :func:`detect_anomaly` as ``threshold``.

    Calibration uses only reference replicates. It does not validate operating
    limits: few replicates, metric choice, and trajectory length all affect the
    scale. Prefer held-out nominal checks before deploying a threshold.

    Parameters
    ----------
    reference_spectra : sequence of KoopmanSpectrum
        At least two in-distribution reference spectra.
    method : {"percentile", "mean_std"}, optional
        Summary rule. ``"percentile"`` returns the ``q``-th percentile of
        pairwise distances. ``"mean_std"`` returns ``mean + k * std`` (sample
        standard deviation over the pairwise distances). Default is
        ``"percentile"``.
    distance_method : {"wasserstein", "subspace_angle"}, optional
        Spectral distance forwarded to :func:`koopman_std`. Default is
        ``"wasserstein"``.
    num_modes : int or None, optional
        Leading modes for subspace-angle comparisons.
    q : float, optional
        Percentile in ``[0, 100]`` when ``method="percentile"``. Default is
        ``95``.
    k : float, optional
        Non-negative multiplier for the standard-deviation term when
        ``method="mean_std"``. Default is ``2``.

    Returns
    -------
    float
        Scalar threshold in the same units as :func:`spectrum_distance`.

    Raises
    ------
    ValueError
        If fewer than two references are provided, ``method`` is unknown, or
        ``q`` / ``k`` are out of range.
    """
    if len(reference_spectra) < 2:
        raise ValueError("reference_spectra must contain at least two spectra")

    matrix = koopman_std(reference_spectra, distance_method, num_modes=num_modes)
    mask = torch.triu(torch.ones_like(matrix, dtype=torch.bool), diagonal=1)
    pairwise = matrix[mask]

    if method == "percentile":
        if not 0.0 <= q <= 100.0:
            msg = f"q must be in [0, 100], got {q}"
            raise ValueError(msg)
        return torch.quantile(pairwise, q / 100.0).item()

    if method == "mean_std":
        if k < 0.0:
            msg = f"k must be >= 0, got {k}"
            raise ValueError(msg)
        mean = pairwise.mean()
        if pairwise.numel() > 1:
            std = pairwise.std(unbiased=False)
        else:
            std = mean.new_zeros(())
        return (mean + k * std).item()

    msg = f"method must be 'percentile' or 'mean_std', got {method!r}"
    raise ValueError(msg)


def detect_anomaly(
    reference_spectra: Sequence[KoopmanSpectrum],
    test_spectrum: KoopmanSpectrum,
    threshold: float,
    method: SpectrumDistanceMethod = "wasserstein",
    *,
    num_modes: int | None = None,
) -> AnomalyDetectionResult:
    """Flag a test spectrum as anomalous when it is far from reference dynamics.

    ``threshold`` is required. Derive it from in-distribution reference
    spectra with :func:`calibrate_anomaly_threshold` (percentile or
    mean-plus-``k``-std of pairwise :func:`koopman_std` distances), or supply a
    domain-specific operating limit.

    Parameters
    ----------
    reference_spectra : sequence of KoopmanSpectrum
        In-distribution reference spectra (healthy baselines or known regimes).
    test_spectrum : KoopmanSpectrum
        Spectrum to evaluate.
    threshold : float
        Maximum acceptable mean distance from ``test_spectrum`` to the
        references. Distances above this value are flagged as anomalies.
    method : {"wasserstein", "subspace_angle"}, optional
        Distance definition. Default is ``"wasserstein"``.
    num_modes : int or None, optional
        Leading modes for subspace-angle comparisons.

    Returns
    -------
    AnomalyDetectionResult
        ``is_anomaly`` is ``True`` when the mean reference distance exceeds
        ``threshold``.

    Raises
    ------
    ValueError
        If ``reference_spectra`` is empty or ``threshold`` is negative.
    """
    if not reference_spectra:
        raise ValueError("reference_spectra must be non-empty")
    if threshold < 0.0:
        msg = f"threshold must be >= 0, got {threshold}"
        raise ValueError(msg)

    test_distances = [
        spectrum_distance(test_spectrum, reference, method, num_modes=num_modes)
        for reference in reference_spectra
    ]
    distance = torch.stack(test_distances).mean().item()

    if len(reference_spectra) == 1:
        reference_mean_distance = 0.0
    else:
        reference_matrix = koopman_std(reference_spectra, method, num_modes=num_modes)
        upper = reference_matrix.triu(diagonal=1)
        reference_mean_distance = upper[upper > 0].mean().item()

    return AnomalyDetectionResult(
        is_anomaly=distance > threshold,
        distance=distance,
        reference_mean_distance=reference_mean_distance,
    )


def _resolve_num_modes(
    latent_dim_a: int,
    latent_dim_b: int,
    num_modes: int | None,
) -> int:
    """Resolve the number of eigenvector columns to compare.

    Parameters
    ----------
    latent_dim_a, latent_dim_b : int
        Latent dimensions of the two spectra.
    num_modes : int or None
        Requested mode count. ``None`` uses ``min(latent_dim_a, latent_dim_b)``.

    Returns
    -------
    int
        Validated number of leading modes to compare.

    Raises
    ------
    ValueError
        If ``num_modes`` is outside ``[1, min(latent_dim_a, latent_dim_b)]``.
    """
    maximum = min(latent_dim_a, latent_dim_b)
    if num_modes is None:
        return maximum
    if num_modes < 1 or num_modes > maximum:
        msg = f"num_modes must be in [1, {maximum}], got {num_modes}"
        raise ValueError(msg)
    return num_modes


def _pad_magnitudes(magnitudes: Tensor, length: int) -> Tensor:
    """Zero-pad a magnitude vector to a target length.

    Parameters
    ----------
    magnitudes : Tensor
        Sorted eigenvalue magnitudes.
    length : int
        Target vector length.

    Returns
    -------
    Tensor
        Magnitudes truncated or zero-padded to ``length``.
    """
    if magnitudes.numel() >= length:
        return magnitudes[:length]
    padding = torch.zeros(
        length - magnitudes.numel(),
        dtype=magnitudes.dtype,
        device=magnitudes.device,
    )
    return torch.cat([magnitudes, padding])


def _wasserstein_magnitude_distance(
    magnitudes_a: Tensor,
    magnitudes_b: Tensor,
) -> Tensor:
    """Compute 1D Wasserstein-1 distance between eigenvalue magnitudes.

    Parameters
    ----------
    magnitudes_a, magnitudes_b : Tensor
        Sorted eigenvalue magnitudes, possibly of different lengths.

    Returns
    -------
    Tensor
        Scalar mean absolute difference after zero-padding.
    """
    length = max(magnitudes_a.numel(), magnitudes_b.numel())
    if length == 0:
        return torch.tensor(0.0, dtype=torch.float64)
    a = _pad_magnitudes(magnitudes_a.detach().to(torch.float64), length)
    b = _pad_magnitudes(magnitudes_b.detach().to(torch.float64), length)
    return torch.mean(torch.abs(a - b))


def _subspace_angle_distance(vectors_a: Tensor, vectors_b: Tensor) -> Tensor:
    """Compute the mean principal angle between eigenvector subspaces.

    Parameters
    ----------
    vectors_a, vectors_b : Tensor
        Eigenvector columns with shape ``(latent_dim, num_modes)``.

    Returns
    -------
    Tensor
        Mean principal angle in radians.
    """
    if vectors_a.numel() == 0:
        return torch.tensor(0.0, dtype=torch.float64)

    dtype = torch.complex128 if vectors_a.is_complex() else torch.float64
    basis_a, _ = torch.linalg.qr(vectors_a.detach().to(dtype))
    basis_b, _ = torch.linalg.qr(vectors_b.detach().to(dtype))
    cosines = torch.linalg.svdvals(basis_a.conj().T @ basis_b).clamp(0.0, 1.0)
    return torch.arccos(cosines).mean().to(torch.float64)
