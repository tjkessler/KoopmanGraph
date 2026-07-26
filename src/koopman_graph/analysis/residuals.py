"""Data-driven spectral residuals for learned Koopman eigenpairs.

Honesty contract
----------------
This is an a posteriori residual computed in the empirical norm of the
**learned** observable space. It flags eigenpairs that fail to propagate as
claimed on held-out data. It is **not** the certified ResDMD residual bound,
which requires Galerkin matrices for :math:`\\mathcal{K}^*\\mathcal{K}` in a
fixed dictionary and yields error control on the infinite-dimensional
spectrum.

Do **not** replace this diagnostic with the matrix residual
:math:`\\|Kv - \\lambda v\\| / \\|v\\|` on the assembled operator: that quantity
is vacuous after ``torch.linalg.eig`` and returns machine epsilon for every
mode.

References
----------
Colbrook, M. J. and Townsend, A. (2023/2024). Rigorous data-driven computation
of spectral properties of Koopman operators for dynamical systems.
*Communications on Pure and Applied Mathematics*, 77(1), 221–283.
https://doi.org/10.1002/cpa.22125 (``ColbrookTownsend2023ResDMD``)

Colbrook, M. J., Ayton, L. J., and Szőke, M. (2023). Residual dynamic mode
decomposition: robust and verified Koopmanism. *Journal of Fluid Mechanics*,
955, A21. https://doi.org/10.1017/jfm.2022.1052
(``Colbrook2023ResidualDMD``)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor
from torch_geometric.data import Data

from koopman_graph.data import GraphSnapshotSequence, resolve_sequence
from koopman_graph.graph_utils.topology import (
    resolve_edge_index,
    resolve_edge_weight,
    snapshot_hyperedge_index,
    snapshot_hyperedge_weight,
)
from koopman_graph.protocols import ModeShapeModel
from koopman_graph.spectrum_types import KoopmanSpectrum

_EPS = 1e-12


@dataclass(frozen=True)
class SpectralResidualReport:
    """Per-mode data-driven residuals on a snapshot sequence.

    Attributes
    ----------
    eigenvalues : Tensor
        Complex eigenvalues with shape ``(latent_dim,)``, magnitude-sorted to
        match the spectrum used for the residual.
    residuals : Tensor
        Non-negative real residuals with shape ``(latent_dim,)``.
    num_pairs : int
        Number of consecutive transition pairs used in the sums.
    tolerance : float
        Trustworthiness threshold applied by :meth:`trustworthy_mask`.
    """

    eigenvalues: Tensor
    residuals: Tensor
    num_pairs: int
    tolerance: float

    def trustworthy_mask(self) -> Tensor:
        """Return a boolean mask of modes with residual at most ``tolerance``.

        Returns
        -------
        Tensor
            Boolean tensor with shape ``(latent_dim,)``.
        """
        return self.residuals <= self.tolerance


def spectral_residuals(
    model: ModeShapeModel,
    sequence: GraphSnapshotSequence | Sequence[Data],
    *,
    spectrum: KoopmanSpectrum | None = None,
    tolerance: float = 1e-2,
    edge_index: Tensor | None = None,
    edge_weight: Tensor | None = None,
    delta_t: float | None = None,
) -> SpectralResidualReport:
    """Compute data-driven residuals for learned Koopman eigenpairs.

    Honesty contract
    ----------------
    This is an a posteriori residual computed in the empirical norm of the
    **learned** observable space. It flags eigenpairs that fail to propagate as
    claimed on held-out data. It is **not** the certified ResDMD residual
    bound, which requires Galerkin matrices for
    :math:`\\mathcal{K}^*\\mathcal{K}` in a fixed dictionary and yields error
    control on the infinite-dimensional spectrum.

    For each eigenpair :math:`(\\lambda_i, v_i)`, mode amplitudes
    :math:`a(t) = \\mathrm{mode\\_amplitudes}(z_t)` are the approximate
    eigenfunction values (left-eigenvector projections). The residual is

    .. math::

        \\mathrm{res}_i^2 =
        \\frac{\\sum_t |a_i(t+1) - \\lambda_i a_i(t)|^2}
             {\\sum_t |a_i(t)|^2}

    where the sum also runs over nodes when amplitudes are per-node. Using the
    training sequence is permitted but then reports a fit statistic, not
    validation.

    Parameters
    ----------
    model : ModeShapeModel
        Model providing ``encode`` / ``spectrum`` (satisfied by
        :class:`~koopman_graph.model.GraphKoopmanModel`).
    sequence : GraphSnapshotSequence or sequence of Data
        Snapshot trajectory used to evaluate residuals.
    spectrum : KoopmanSpectrum or None, optional
        Precomputed spectrum. When ``None``, computed from ``model`` with
        topology taken from the first snapshot when required. For continuous
        dynamics the discrete spectrum at ``delta_t`` (or
        ``model.time_step``) is used.
    tolerance : float, optional
        Threshold for :meth:`SpectralResidualReport.trustworthy_mask`.
        Default is ``1e-2``.
    edge_index : Tensor or None, optional
        Optional override topology for networked spectra and encoding.
    edge_weight : Tensor or None, optional
        Optional edge weights matching ``edge_index``.
    delta_t : float or None, optional
        Discrete horizon for continuous generators. When ``None`` and the
        model is continuous, uses ``model.time_step``.

    Returns
    -------
    SpectralResidualReport
        Eigenvalues, residuals, pair count, and tolerance.

    Raises
    ------
    ValueError
        If ``tolerance`` is not positive, the sequence has fewer than two
        snapshots, spectrum / latent dimensions are incompatible, or
        networked topology cannot be resolved.
    """
    if tolerance <= 0:
        msg = f"tolerance must be positive, got {tolerance}"
        raise ValueError(msg)

    snapshots = resolve_sequence(sequence)
    if snapshots.num_timesteps < 2:
        msg = (
            "spectral_residuals requires at least two snapshots, "
            f"got {snapshots.num_timesteps}"
        )
        raise ValueError(msg)

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            resolved_spectrum = (
                spectrum
                if spectrum is not None
                else _resolve_spectrum(
                    model,
                    snapshots,
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                    delta_t=delta_t,
                )
            )
            latents = _encode_latent_trajectory(
                model,
                snapshots,
                edge_index=edge_index,
                edge_weight=edge_weight,
            )
            residuals, num_pairs = _compute_mode_residuals(
                resolved_spectrum,
                latents,
            )
    finally:
        model.train(was_training)

    return SpectralResidualReport(
        eigenvalues=resolved_spectrum.eigenvalues.detach().clone(),
        residuals=residuals,
        num_pairs=num_pairs,
        tolerance=float(tolerance),
    )


def _resolve_spectrum(
    model: ModeShapeModel,
    sequence: GraphSnapshotSequence,
    *,
    edge_index: Tensor | None,
    edge_weight: Tensor | None,
    delta_t: float | None,
) -> KoopmanSpectrum:
    """Resolve discrete-time spectrum from the model and sequence topology.

    Parameters
    ----------
    model : ModeShapeModel
        Model providing ``spectrum``.
    sequence : GraphSnapshotSequence
        Trajectory used to resolve reference topology.
    edge_index : Tensor or None
        Optional topology override.
    edge_weight : Tensor or None
        Optional edge weights matching ``edge_index``.
    delta_t : float or None
        Discrete horizon for continuous generators.

    Returns
    -------
    KoopmanSpectrum
        Discrete-time spectrum used for residual evaluation.

    Raises
    ------
    ValueError
        If hypergraph topology is missing or ``delta_t`` is non-positive.
    """
    reference = sequence[0]
    edges = resolve_edge_index(reference, edge_index)
    weights = resolve_edge_weight(reference, edge_weight)
    num_nodes = (
        int(reference.num_nodes)
        if reference.num_nodes is not None
        else int(reference.x.size(0))
    )

    uses_graph = bool(getattr(model, "uses_graph_koopman", False))
    uses_continuous_graph = bool(getattr(model, "uses_continuous_graph_koopman", False))
    uses_hypergraph = bool(getattr(model, "uses_hypergraph_koopman", False))
    is_continuous = bool(getattr(model, "is_continuous", False))

    kwargs: dict[str, object] = {}
    if uses_graph or uses_continuous_graph:
        kwargs["edge_index"] = edges
        kwargs["num_nodes"] = num_nodes
        kwargs["edge_weight"] = weights
    if uses_hypergraph:
        hyperedge_index = snapshot_hyperedge_index(reference)
        if hyperedge_index is None:
            msg = (
                "hypergraph spectrum requires hyperedge_index on the reference snapshot"
            )
            raise ValueError(msg)
        kwargs["hyperedge_index"] = hyperedge_index
        kwargs["hyperedge_weight"] = snapshot_hyperedge_weight(reference)
        kwargs["num_nodes"] = num_nodes

    if is_continuous:
        resolved_delta = (
            float(delta_t)
            if delta_t is not None
            else float(model.resolve_delta_t(None))
        )
        if resolved_delta <= 0:
            msg = f"delta_t must be positive, got {resolved_delta}"
            raise ValueError(msg)
        kwargs["delta_t"] = resolved_delta

    return model.spectrum(**kwargs)  # type: ignore[arg-type]


def _encode_latent_trajectory(
    model: ModeShapeModel,
    sequence: GraphSnapshotSequence,
    *,
    edge_index: Tensor | None,
    edge_weight: Tensor | None,
) -> list[Tensor]:
    """Encode each snapshot into latents ``(N, d)``.

    Parameters
    ----------
    model : ModeShapeModel
        Model providing ``encode`` (and optionally ``encode_at``).
    sequence : GraphSnapshotSequence
        Snapshot trajectory.
    edge_index : Tensor or None
        Optional topology override for tensor-style encode.
    edge_weight : Tensor or None
        Optional edge weights matching ``edge_index``.

    Returns
    -------
    list of Tensor
        Per-timestep latent tensors with shape ``(num_nodes, latent_dim)``.
    """
    encode_at = getattr(model, "encode_at", None)
    latents: list[Tensor] = []
    for index in range(sequence.num_timesteps):
        if callable(encode_at) and edge_index is None and edge_weight is None:
            latent = encode_at(sequence, index)
        else:
            snapshot = sequence[index]
            edges = resolve_edge_index(snapshot, edge_index)
            weights = resolve_edge_weight(snapshot, edge_weight)
            latent = model.encode(snapshot, edges, weights)
        latents.append(latent.detach())
    return latents


def _compute_mode_residuals(
    spectrum: KoopmanSpectrum,
    latents: Sequence[Tensor],
) -> tuple[Tensor, int]:
    """Accumulate per-mode data-driven residuals over latent transitions.

    Parameters
    ----------
    spectrum : KoopmanSpectrum
        Discrete-time spectrum providing eigenvalues and ``mode_amplitudes``.
    latents : sequence of Tensor
        Encoded latent frames.

    Returns
    -------
    residuals : Tensor
        Non-negative real residuals with shape ``(latent_dim,)``.
    num_pairs : int
        Number of consecutive transition pairs used.

    Raises
    ------
    ValueError
        If fewer than two latent frames are provided.
    """
    if len(latents) < 2:
        msg = "residual computation requires at least two latent frames"
        raise ValueError(msg)

    spectral_dim = int(spectrum.eigenvalues.numel())
    device = spectrum.eigenvalues.device
    numerators = torch.zeros(spectral_dim, dtype=torch.float64, device=device)
    denominators = torch.zeros(spectral_dim, dtype=torch.float64, device=device)
    num_pairs = 0

    for index in range(len(latents) - 1):
        state = _prepare_amplitude_state(latents[index], spectral_dim)
        next_state = _prepare_amplitude_state(latents[index + 1], spectral_dim)
        amplitudes = spectrum.mode_amplitudes(state)
        next_amplitudes = spectrum.mode_amplitudes(next_state)
        predicted = amplitudes * spectrum.eigenvalues.to(
            dtype=amplitudes.dtype,
            device=amplitudes.device,
        )
        error = next_amplitudes - predicted
        # Sum over any leading batch / node dimensions; keep mode axis.
        reduce_dims = tuple(range(error.ndim - 1))
        if reduce_dims:
            numerators = numerators + error.abs().square().sum(dim=reduce_dims).to(
                dtype=torch.float64,
                device=device,
            )
            denominators = denominators + amplitudes.abs().square().sum(
                dim=reduce_dims
            ).to(dtype=torch.float64, device=device)
        else:
            numerators = numerators + error.abs().square().to(
                dtype=torch.float64,
                device=device,
            )
            denominators = denominators + amplitudes.abs().square().to(
                dtype=torch.float64,
                device=device,
            )
        num_pairs += 1

    residuals = torch.sqrt(numerators / denominators.clamp_min(_EPS))
    return residuals.to(dtype=torch.float64), num_pairs


def _prepare_amplitude_state(latent: Tensor, spectral_dim: int) -> Tensor:
    """Reshape a latent frame to match the spectrum trailing dimension.

    Per-node operators use trailing dim ``d`` with shape ``(N, d)``. Networked
    ``N·d`` spectra require a flattened ``(N·d,)`` state.

    Parameters
    ----------
    latent : Tensor
        Encoded latent frame.
    spectral_dim : int
        Trailing dimension expected by ``mode_amplitudes``.

    Returns
    -------
    Tensor
        Latent state with trailing dimension ``spectral_dim``.

    Raises
    ------
    ValueError
        If the latent shape is incompatible with ``spectral_dim``.
    """
    if latent.ndim == 0:
        msg = f"latent state must be at least 1-D, got shape {tuple(latent.shape)}"
        raise ValueError(msg)
    if latent.shape[-1] == spectral_dim:
        return latent
    flat = latent.reshape(-1)
    if flat.numel() != spectral_dim:
        msg = (
            "latent state is incompatible with spectrum dimension "
            f"{spectral_dim}: got shape {tuple(latent.shape)} "
            f"(flattened length {flat.numel()})"
        )
        raise ValueError(msg)
    return flat
