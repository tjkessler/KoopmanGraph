"""Spectrum computation and spatial mode-shape decoding."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor
from torch_geometric.data import Data

from koopman_graph.graph_utils.topology import resolve_edge_index, resolve_edge_weight
from koopman_graph.protocols import ModeShapeModel
from koopman_graph.spectrum_types import (
    compute_generator_spectrum,
    compute_spectrum,
    discrete_spectrum_at_delta_t,
)

# Discrete / generator / Δt spectrum assembly lives on the neutral
# ``spectrum_types`` leaf so operators and the model façade can call it
# without importing this analysis package. Re-export for the public
# ``koopman_graph.analysis`` surface.
__all__ = [
    "compute_generator_spectrum",
    "compute_spectrum",
    "decode_mode_shapes",
    "discrete_spectrum_at_delta_t",
]


def decode_mode_shapes(
    model: ModeShapeModel,
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

    Typed against :class:`~koopman_graph.protocols.ModeShapeModel` (satisfied by
    :class:`~koopman_graph.model.GraphKoopmanModel`) so analysis does not import
    the estimator package. Spectrum-only comparisons use
    :func:`~koopman_graph.analysis.dynamical_similarity` instead.
    For ``koopman="graph"``, topology is taken from the reference graph and
    forwarded into :meth:`~koopman_graph.model.GraphKoopmanModel.spectrum`.

    Parameters
    ----------
    model : ModeShapeModel
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
        If ``perturbation`` is not positive, a mode index is out of range, or
        a graph model is missing resolvable topology.
    """
    if perturbation <= 0:
        msg = f"perturbation must be positive, got {perturbation}"
        raise ValueError(msg)

    edges = resolve_edge_index(x_or_data, edge_index)
    edge_weight = resolve_edge_weight(x_or_data, None)
    if model.uses_graph_koopman or model.uses_continuous_graph_koopman:
        if isinstance(x_or_data, Data):
            num_nodes = (
                int(x_or_data.num_nodes)
                if x_or_data.num_nodes is not None
                else int(x_or_data.x.size(0))
            )
        else:
            # Feature tensor ``(N, F)`` or delay window ``(..., N, F)``.
            num_nodes = int(x_or_data.shape[-2])
        spectrum = model.spectrum(
            edge_index=edges,
            num_nodes=num_nodes,
            edge_weight=edge_weight,
        )
    else:
        spectrum = model.spectrum()
    latent_dim = spectrum.eigenvalues.numel()
    indices = list(range(latent_dim)) if mode_indices is None else list(mode_indices)
    if any(index < 0 or index >= latent_dim for index in indices):
        msg = f"mode_indices must be between 0 and {latent_dim - 1}, got {indices}"
        raise ValueError(msg)

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
    model: ModeShapeModel,
    latent: Tensor,
    edge_index: Tensor,
    edge_weight: Tensor | None,
    direction: Tensor,
    perturbation: float,
) -> Tensor:
    """Estimate decoder response to one complex latent direction.

    Parameters
    ----------

    model : ModeShapeModel
        Model providing the decoder.
    latent : Tensor
        Encoded reference state.
    edge_index : Tensor
        Graph connectivity.
    direction : Tensor
        Complex latent eigenvector.
    perturbation : float
        Centered finite-difference step.
    edge_weight : Tensor | None
        See the function signature / summary for ``edge_weight``.

    Returns
    -------

    Tensor
        Complex node-feature response."""
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
    model: ModeShapeModel,
    latent: Tensor,
    edge_index: Tensor,
    edge_weight: Tensor | None,
    direction: Tensor,
    perturbation: float,
) -> Tensor:
    """Estimate decoder response to one real latent direction.

    Parameters
    ----------

    model : ModeShapeModel
        Model providing the decoder.
    latent : Tensor
        Encoded reference state.
    edge_index : Tensor
        Graph connectivity.
    direction : Tensor
        Real latent direction.
    perturbation : float
        Centered finite-difference step.
    edge_weight : Tensor | None
        See the function signature / summary for ``edge_weight``.

    Returns
    -------

    Tensor
        Real node-feature response."""
    if direction.numel() == latent.numel() and direction.shape != latent.shape:
        # Networked spectrum eigenvectors are ``(N·d,)``; reshape to node layout.
        direction = direction.reshape_as(latent)
    if not torch.count_nonzero(direction):
        return torch.zeros(
            (latent.shape[0], model.decoder.out_channels),
            dtype=latent.dtype,
            device=latent.device,
        )
    plus = model.decoder(latent + perturbation * direction, edge_index, edge_weight)
    minus = model.decoder(latent - perturbation * direction, edge_index, edge_weight)
    return (plus - minus) / (2 * perturbation)
