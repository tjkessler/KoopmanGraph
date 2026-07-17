"""Spectrum computation and spatial mode-shape decoding."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch_geometric.data import Data

from koopman_graph.graph_utils import resolve_edge_index, resolve_edge_weight
from koopman_graph.spectrum_types import KoopmanSpectrum

if TYPE_CHECKING:
    from collections.abc import Sequence

    from koopman_graph.model import GraphKoopmanModel


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

    Hard-typed to :class:`~koopman_graph.model.GraphKoopmanModel` because it
    needs ``encode`` / ``decode`` and a GNN decoder. Spectrum-only comparisons
    use :func:`~koopman_graph.analysis.dynamical_similarity` instead.

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

    edges = resolve_edge_index(x_or_data, edge_index)
    edge_weight = resolve_edge_weight(x_or_data, None)
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
