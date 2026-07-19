"""Spectral distance and dynamical-similarity helpers."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Literal

import torch
from torch import Tensor

from koopman_graph.protocols import SpectrumProvider
from koopman_graph.spectrum_types import KoopmanSpectrum

if TYPE_CHECKING:
    from collections.abc import Sequence

SpectrumDistanceMethod = Literal["wasserstein", "subspace_angle"]
SpectrumSource = KoopmanSpectrum | SpectrumProvider


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
        Distance definition. ``"wasserstein"`` compares sorted eigenvalue
        magnitudes with mean absolute difference after zero-padding the
        shorter list to equal length (exact 1D Wasserstein-1 when lengths
        already match; padded L1 otherwise). ``"subspace_angle"`` returns the
        mean principal angle (radians) between dominant eigenvector subspaces.
        Default is ``"wasserstein"``.
    num_modes : int or None, optional
        Number of leading modes to compare for ``"subspace_angle"``. Defaults
        to ``min(latent_dim_a, latent_dim_b)``. Ignored by ``"wasserstein"``.

    Returns
    -------
    Tensor
        Scalar distance (magnitude units for the padded magnitude metric,
        radians for subspace angle). The result is detached from the
        autograd graph.

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


def resolve_spectrum(
    source: SpectrumSource,
    *,
    delta_t: float | None = None,
    edge_index: Tensor | None = None,
    num_nodes: int | None = None,
    edge_weight: Tensor | None = None,
) -> KoopmanSpectrum:
    """Resolve a :class:`KoopmanSpectrum` from a value or spectrum provider.

    Parameters
    ----------
    source : KoopmanSpectrum or SpectrumProvider
        Precomputed spectrum, or any object with a ``spectrum`` method
        (classical baselines, :class:`~koopman_graph.model.GraphKoopmanModel`).
    delta_t : float or None, optional
        Continuous integration horizon. Forwarded only when ``source.spectrum``
        accepts a ``delta_t`` parameter (or ``**kwargs``). Ignored for
        precomputed spectra and for providers whose ``spectrum`` takes no
        kwargs (classical baselines).
    edge_index : Tensor or None, optional
        Topology for networked ``koopman="graph"`` models. Forwarded when
        accepted by ``source.spectrum``.
    num_nodes : int or None, optional
        Node count for the effective networked operator. Forwarded when
        accepted by ``source.spectrum``.
    edge_weight : Tensor or None, optional
        Optional edge weights for networked spectrum. Forwarded when accepted.

    Returns
    -------
    KoopmanSpectrum
        Resolved spectrum value.

    Raises
    ------
    TypeError
        If ``source`` is neither a :class:`KoopmanSpectrum` nor a spectrum
        provider.
    """
    if isinstance(source, KoopmanSpectrum):
        return source
    if not isinstance(source, SpectrumProvider):
        msg = (
            "source must be a KoopmanSpectrum or SpectrumProvider, "
            f"got {type(source).__name__}"
        )
        raise TypeError(msg)

    spectrum_fn = source.spectrum
    call_kwargs: dict[str, object] = {}
    if delta_t is not None:
        call_kwargs["delta_t"] = delta_t
    if edge_index is not None:
        call_kwargs["edge_index"] = edge_index
    if num_nodes is not None:
        call_kwargs["num_nodes"] = num_nodes
    if edge_weight is not None:
        call_kwargs["edge_weight"] = edge_weight

    if not call_kwargs:
        return spectrum_fn()

    try:
        signature = inspect.signature(spectrum_fn)
    except (TypeError, ValueError):
        return spectrum_fn()

    parameters = signature.parameters
    accepts_var_keyword = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    filtered = {
        key: value
        for key, value in call_kwargs.items()
        if key in parameters or accepts_var_keyword
    }
    if not filtered:
        return spectrum_fn()
    return spectrum_fn(**filtered)


def dynamical_similarity(
    model_a: SpectrumSource,
    model_b: SpectrumSource,
    method: SpectrumDistanceMethod = "wasserstein",
    *,
    num_modes: int | None = None,
    delta_t: float | None = None,
    edge_index: Tensor | None = None,
    num_nodes: int | None = None,
    edge_weight: Tensor | None = None,
) -> Tensor:
    """Compare learned dynamics via Koopman spectra.

    Accepts precomputed :class:`KoopmanSpectrum` values and/or spectrum
    providers (classical baselines or
    :class:`~koopman_graph.model.GraphKoopmanModel`). Optional ``delta_t`` and
    topology kwargs are forwarded only to providers whose ``spectrum`` accepts
    them.

    Call patterns::

        dynamical_similarity(spectrum_a, spectrum_b)
        dynamical_similarity(dmd_a, dmd_b)
        dynamical_similarity(dmd, neural_model)
        dynamical_similarity(neural_a, neural_b, delta_t=0.1)
        dynamical_similarity(
            graph_a, graph_b, edge_index=edges, num_nodes=n
        )

    Decoder-specific spatial mode analysis remains on
    :func:`~koopman_graph.analysis.decode_mode_shapes` (hard-typed to
    ``GraphKoopmanModel``).

    Parameters
    ----------
    model_a, model_b : KoopmanSpectrum or SpectrumProvider
        Spectra or fitted forecasting façades compared via
        :func:`spectrum_distance`.
    method : {"wasserstein", "subspace_angle"}, optional
        Distance definition. Default is ``"wasserstein"``.
    num_modes : int or None, optional
        Leading modes for subspace-angle comparisons.
    delta_t : float or None, optional
        Integration horizon for continuous-time neural ``spectrum`` calls.
        Ignored for precomputed spectra and classical baselines.
    edge_index : Tensor or None, optional
        Topology for networked graph-model spectrum resolution.
    num_nodes : int or None, optional
        Node count for networked graph-model spectrum resolution.
    edge_weight : Tensor or None, optional
        Optional edge weights for networked spectrum resolution.

    Returns
    -------
    Tensor
        Scalar spectral distance between the two sources.
    """
    return spectrum_distance(
        resolve_spectrum(
            model_a,
            delta_t=delta_t,
            edge_index=edge_index,
            num_nodes=num_nodes,
            edge_weight=edge_weight,
        ),
        resolve_spectrum(
            model_b,
            delta_t=delta_t,
            edge_index=edge_index,
            num_nodes=num_nodes,
            edge_weight=edge_weight,
        ),
        method,
        num_modes=num_modes,
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
    """Compare sorted eigenvalue magnitudes with padded mean absolute error.

    When both vectors have length ``n``, this equals the 1D Wasserstein-1
    distance between empirical uniform measures on the sorted samples.
    When lengths differ, the shorter vector is zero-padded to the longer
    length before averaging ``|a_i - b_i|`` (not optimal transport between
    unequally sized supports).

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
