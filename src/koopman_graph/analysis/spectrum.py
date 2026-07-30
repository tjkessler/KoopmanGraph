"""Spectrum computation, mode-shape decoding, and mode-energy attribution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor
from torch_geometric.data import Data

from koopman_graph.data.hetero_layout import latent_type_slices
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
    "ModeEnergyAttribution",
    "attribute_mode_energy",
    "compute_generator_spectrum",
    "compute_spectrum",
    "decode_mode_shapes",
    "discrete_spectrum_at_delta_t",
]


@dataclass(frozen=True)
class ModeEnergyAttribution:
    """Per-mode energy fractions on type self-blocks and relation couplings.

    Honesty contract
    ----------------
    Fractions are an **interpretive diagnostic** of eigenvector mass /
    operator-action concentration on declared index / coupling blocks of an
    assembled ``K_eff``. They are **not** causal attribution, interventional
    importance, or a ResDMD residual bound on relation-attributed modes.

    Type fractions use projected mass on flat latent slices and sum to 1
    when the supplied type slices tile ``[0, N·d)``. Relation fractions use
    coupling-action mass and may **not** sum to 1 when relation supports
    overlap or leave residual mass outside the supplied blocks.

    Attributes
    ----------
    mode_indices : tuple of int
        Selected mode indices into the eigenvector columns.
    type_fractions : dict of str to Tensor
        Per-type projected-mass fractions with shape ``(num_modes,)``.
    relation_fractions : dict of str to Tensor
        Per-relation action-mass fractions with shape ``(num_modes,)``.
    latent_dim : int
        Shared latent width ``d`` used to expand node slices.
    """

    mode_indices: tuple[int, ...]
    type_fractions: dict[str, Tensor]
    relation_fractions: dict[str, Tensor]
    latent_dim: int


def attribute_mode_energy(
    k_eff: Tensor,
    eigenvectors: Tensor,
    *,
    latent_dim: int,
    node_type_slices: Mapping[str, slice] | None = None,
    relation_blocks: Mapping[str, Tensor] | None = None,
    mode_indices: Sequence[int] | None = None,
) -> ModeEnergyAttribution:
    """Attribute Koopman modes to type self-blocks and relation couplings.

    Honesty contract
    ----------------
    This helper reports **interpretive** energy fractions on declared blocks
    of an assembled effective operator ``K_eff``. It does **not** claim
    causal / interventional importance and is **not** a ResDMD-certified
    residual on relation-attributed modes.

    For each selected eigenvector column ``v`` of ``eigenvectors``:

    * **Type fraction** (projected mass)::

          ‖P_τ v‖² / ‖v‖²

      where ``P_τ`` keeps the flat indices of type ``τ`` (node-row slice
      expanded by ``latent_dim``).

    * **Relation fraction** (coupling action mass)::

          ‖C_r v‖² / ‖K_eff v‖²

      where ``C_r`` is the caller-supplied coupling block
      (typically ``Â_r ⊗ K_r``). Relation fractions are raw and need not
      sum to one under overlapping supports.

    Parameters
    ----------
    k_eff : Tensor
        Assembled effective operator with shape ``(N·d, N·d)``.
    eigenvectors : Tensor
        Eigenvector matrix with shape ``(N·d, num_modes_total)`` (columns
        are modes), typically from
        :class:`~koopman_graph.spectrum_types.KoopmanSpectrum`.
    latent_dim : int
        Shared latent width ``d``.
    node_type_slices : mapping of str to slice or None, optional
        Node-row slices (from :func:`~koopman_graph.data.node_type_slices`).
        When ``None``, type fractions are empty.
    relation_blocks : mapping of str to Tensor or None, optional
        Per-relation coupling matrices ``C_r`` with shape ``(N·d, N·d)``.
        When ``None``, relation fractions are empty.
    mode_indices : sequence of int or None, optional
        Columns of ``eigenvectors`` to attribute. Defaults to every column.

    Returns
    -------
    ModeEnergyAttribution
        Per-mode type and relation fraction dictionaries.

    Raises
    ------
    ValueError
        If shapes are inconsistent, ``latent_dim`` is invalid, a mode index
        is out of range, or a relation block has the wrong shape.
    """
    if k_eff.ndim != 2 or k_eff.shape[0] != k_eff.shape[1]:
        msg = (
            "k_eff must be a square matrix with shape (N*d, N*d); "
            f"got {tuple(k_eff.shape)}"
        )
        raise ValueError(msg)
    if eigenvectors.ndim != 2:
        msg = (
            "eigenvectors must have shape (N*d, num_modes); "
            f"got {tuple(eigenvectors.shape)}"
        )
        raise ValueError(msg)
    if eigenvectors.shape[0] != k_eff.shape[0]:
        msg = (
            "eigenvectors rows must match k_eff dimension "
            f"({k_eff.shape[0]}); got {eigenvectors.shape[0]}"
        )
        raise ValueError(msg)
    if latent_dim < 1:
        msg = f"latent_dim must be positive, got {latent_dim}"
        raise ValueError(msg)
    if k_eff.shape[0] % latent_dim != 0:
        msg = (
            f"k_eff dimension {k_eff.shape[0]} is not divisible by "
            f"latent_dim={latent_dim}"
        )
        raise ValueError(msg)

    num_modes_total = eigenvectors.shape[1]
    indices = (
        list(range(num_modes_total)) if mode_indices is None else list(mode_indices)
    )
    if any(index < 0 or index >= num_modes_total for index in indices):
        msg = f"mode_indices must be between 0 and {num_modes_total - 1}, got {indices}"
        raise ValueError(msg)

    selected = eigenvectors[:, indices]
    # Prefer real arithmetic when modes are real; otherwise use |·|² via conj.
    mode_norms_sq = _column_energy(selected)
    type_fractions: dict[str, Tensor] = {}
    if node_type_slices:
        flat_slices = latent_type_slices(node_type_slices, latent_dim)
        for name, flat_slice in flat_slices.items():
            if flat_slice.stop > k_eff.shape[0] or flat_slice.start < 0:
                msg = (
                    f"latent slice for type {name!r} {flat_slice!r} is outside "
                    f"[0, {k_eff.shape[0]})"
                )
                raise ValueError(msg)
            projected = selected[flat_slice, :]
            type_fractions[name] = _column_energy(projected) / mode_norms_sq.clamp_min(
                torch.finfo(mode_norms_sq.dtype).tiny
            )

    relation_fractions: dict[str, Tensor] = {}
    if relation_blocks:
        action = k_eff.to(dtype=selected.dtype, device=selected.device) @ selected
        action_norms_sq = _column_energy(action).clamp_min(
            torch.finfo(mode_norms_sq.dtype).tiny
        )
        for name, coupling in relation_blocks.items():
            if coupling.shape != k_eff.shape:
                msg = (
                    f"relation_blocks[{name!r}] must have shape "
                    f"{tuple(k_eff.shape)}, got {tuple(coupling.shape)}"
                )
                raise ValueError(msg)
            coupling_mat = coupling.to(
                dtype=selected.dtype,
                device=selected.device,
            )
            coupled = coupling_mat @ selected
            relation_fractions[str(name)] = _column_energy(coupled) / action_norms_sq

    return ModeEnergyAttribution(
        mode_indices=tuple(indices),
        type_fractions=type_fractions,
        relation_fractions=relation_fractions,
        latent_dim=int(latent_dim),
    )


def _column_energy(matrix: Tensor) -> Tensor:
    """Return per-column squared Euclidean energy ``‖col‖²``.

    Parameters
    ----------
    matrix : Tensor
        Matrix with shape ``(dim, num_modes)`` (real or complex).

    Returns
    -------
    Tensor
        Real non-negative energies with shape ``(num_modes,)``.
    """
    if torch.is_complex(matrix):
        return (matrix.real.square() + matrix.imag.square()).sum(dim=0)
    return matrix.square().sum(dim=0)


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
