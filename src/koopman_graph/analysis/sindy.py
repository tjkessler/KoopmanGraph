"""Sparse identification of nonlinear dynamics (SINDy) on learned latents.

Fits a sparse library expansion of the model's **encoded** latent
trajectories via sequentially thresholded least squares (STLSQ). This is an
interpretability / diagnostics tool for the learned latent map — it does
**not** recover physical governing equations in the original coordinates.

References
----------
Brunton, S. L., Proctor, J. L., and Kutz, J. N. (2016). Discovering governing
equations from data by sparse identification of nonlinear dynamical systems.
*Proceedings of the National Academy of Sciences*, 113(15), 3932–3937.
https://doi.org/10.1073/pnas.1517384113
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import Literal

import torch
from torch import Tensor
from torch_geometric.data import Data

from koopman_graph.baselines.base import require_static_topology
from koopman_graph.data import GraphSnapshotSequence, resolve_sequence
from koopman_graph.graph_utils import (
    resolve_edge_index,
    resolve_edge_weight,
    symmetric_normalized_laplacian_matvec,
)
from koopman_graph.protocols import ModeShapeModel

SINDyLibrary = Literal["poly", "graph"]
SINDyMode = Literal["discrete", "derivative"]


@dataclass(frozen=True)
class SINDyReport:
    """Sparse library fit of learned latent dynamics.

    Attributes
    ----------
    coefficients : Tensor
        Sparse coefficient matrix ``Ξ`` with shape
        ``(n_features, latent_dim)``. Inactive entries are exactly zero.
    active_mask : Tensor
        Boolean mask with the same shape as ``coefficients``.
    term_names : tuple of str
        Library column names aligned with ``coefficients`` rows.
    residual : float
        Mean squared residual of the final sparse fit.
    threshold_history : tuple of float
        Absolute threshold applied at each STLSQ iteration (length
        ``≤ max_iter``).
    mode : {"discrete", "derivative"}
        Target construction used for the fit.
    library : {"poly", "graph"}
        Candidate library family.
    """

    coefficients: Tensor
    active_mask: Tensor
    term_names: tuple[str, ...]
    residual: float
    threshold_history: tuple[float, ...]
    mode: SINDyMode
    library: SINDyLibrary


def _monomial_name(indices: tuple[int, ...]) -> str:
    """Format a multi-index monomial name such as ``z0^2*z1``.

    Parameters
    ----------

    indices : tuple[int, ...]
        See the function signature / summary for ``indices``.

    Returns
    -------

    str
        See summary line."""
    if not indices:
        return "1"
    counts: dict[int, int] = {}
    for idx in indices:
        counts[idx] = counts.get(idx, 0) + 1
    parts: list[str] = []
    for idx in sorted(counts):
        power = counts[idx]
        if power == 1:
            parts.append(f"z{idx}")
        else:
            parts.append(f"z{idx}^{power}")
    return "*".join(parts)


def _poly_multi_indices(latent_dim: int, degree: int) -> list[tuple[int, ...]]:
    """Return multi-indices for monomials of total degree ``1..degree``.

    Parameters
    ----------

    latent_dim : int
        See the function signature / summary for ``latent_dim``.
    degree : int
        See the function signature / summary for ``degree``.

    Returns
    -------

    list[tuple[int, ...]]
        See summary line."""
    indices: list[tuple[int, ...]] = []
    for total_degree in range(1, degree + 1):
        indices.extend(combinations_with_replacement(range(latent_dim), total_degree))
    return indices


def _build_poly_library(
    z: Tensor,
    degree: int,
    *,
    include_constant: bool = True,
    name_prefix: str = "",
) -> tuple[Tensor, list[str]]:
    """Build a polynomial feature matrix for rows of ``z``.

    Parameters
    ----------
    z : Tensor
        Latent samples with shape ``(num_samples, latent_dim)``.
    degree : int
        Maximum total monomial degree (``≥ 1``).
    include_constant : bool, optional
        When ``True``, prepend a column of ones named ``1`` (or
        ``{prefix}1``).
    name_prefix : str, optional
        Prefix for all term names (used for ``L_sym`` blocks).

    Returns
    -------
    Theta : Tensor
        Library matrix with shape ``(num_samples, n_features)``.
    term_names : list of str
        Column names aligned with ``Theta``.
    """
    if z.ndim != 2:
        msg = f"z must have shape (num_samples, latent_dim), got {tuple(z.shape)}"
        raise ValueError(msg)
    if degree < 1:
        msg = f"degree must be >= 1, got {degree}"
        raise ValueError(msg)

    num_samples, latent_dim = z.shape
    columns: list[Tensor] = []
    names: list[str] = []
    if include_constant:
        columns.append(torch.ones(num_samples, 1, dtype=z.dtype, device=z.device))
        names.append(f"{name_prefix}1" if name_prefix else "1")

    for multi in _poly_multi_indices(latent_dim, degree):
        term = torch.ones(num_samples, dtype=z.dtype, device=z.device)
        for idx in multi:
            term = term * z[:, idx]
        columns.append(term.unsqueeze(-1))
        base = _monomial_name(multi)
        names.append(f"{name_prefix}{base}" if name_prefix else base)

    return torch.cat(columns, dim=-1), names


def _build_graph_library(
    z_nodes: Tensor,
    edge_index: Tensor,
    edge_weight: Tensor | None,
    degree: int,
) -> tuple[Tensor, list[str]]:
    """Polynomial library on ``z`` augmented with poly features of ``L_sym z``.

    Parameters
    ----------

    z_nodes : Tensor
        See the function signature / summary for ``z_nodes``.
    edge_index : Tensor
        See the function signature / summary for ``edge_index``.
    edge_weight : Tensor | None
        See the function signature / summary for ``edge_weight``.
    degree : int
        See the function signature / summary for ``degree``.

    Returns
    -------

    tuple[Tensor, list[str]]
        See summary line.

    Raises
    ------

    ValueError
        Raised when inputs are invalid."""
    if z_nodes.ndim != 2:
        msg = (
            "z_nodes must have shape (num_nodes, latent_dim), "
            f"got {tuple(z_nodes.shape)}"
        )
        raise ValueError(msg)
    theta_z, names_z = _build_poly_library(z_nodes, degree, include_constant=True)
    lap_z = symmetric_normalized_laplacian_matvec(
        edge_index,
        z_nodes,
        edge_weight=edge_weight,
        num_nodes=z_nodes.shape[0],
    )
    theta_l, names_l = _build_poly_library(
        lap_z,
        degree,
        include_constant=False,
        name_prefix="Lsym:",
    )
    return torch.cat([theta_z, theta_l], dim=-1), names_z + names_l


def _stlsq(
    theta: Tensor,
    target: Tensor,
    *,
    threshold: float,
    max_iter: int,
) -> tuple[Tensor, Tensor, tuple[float, ...]]:
    """Sequentially thresholded least squares (column-wise STLSQ).

    Parameters
    ----------

    theta : Tensor
        See the function signature / summary for ``theta``.
    target : Tensor
        See the function signature / summary for ``target``.
    threshold : float
        See the function signature / summary for ``threshold``.
    max_iter : int
        See the function signature / summary for ``max_iter``.

    Returns
    -------

    tuple[Tensor, Tensor, tuple[float, ...]]
        See summary line.

    Raises
    ------

    ValueError
        Raised when inputs are invalid."""
    if threshold < 0:
        msg = f"threshold must be non-negative, got {threshold}"
        raise ValueError(msg)
    if max_iter < 1:
        msg = f"max_iter must be >= 1, got {max_iter}"
        raise ValueError(msg)
    if theta.ndim != 2 or target.ndim != 2:
        msg = "theta and target must be 2D"
        raise ValueError(msg)
    if theta.shape[0] != target.shape[0]:
        msg = (
            "theta and target must share the sample axis, got "
            f"{theta.shape[0]} vs {target.shape[0]}"
        )
        raise ValueError(msg)
    if theta.shape[0] < 1:
        msg = "STLSQ requires at least one sample"
        raise ValueError(msg)

    latent_dim = target.shape[1]
    coefficients = torch.linalg.lstsq(theta, target, rcond=None).solution
    if coefficients.ndim == 1:
        coefficients = coefficients.unsqueeze(-1)
    history: list[float] = []

    for _ in range(max_iter):
        history.append(float(threshold))
        previous = coefficients.clone()
        coefficients = coefficients.clone()
        coefficients[coefficients.abs() < threshold] = 0
        for col in range(latent_dim):
            active = coefficients[:, col].abs() > 0
            if not torch.any(active):
                continue
            sol = torch.linalg.lstsq(
                theta[:, active],
                target[:, col],
                rcond=None,
            ).solution
            coefficients[:, col] = 0
            coefficients[active, col] = sol
            small = coefficients[:, col].abs() < threshold
            coefficients[small, col] = 0
        if torch.allclose(coefficients, previous):
            break

    active_mask = coefficients.abs() > 0
    return coefficients, active_mask, tuple(history)


def _encode_latent_trajectory(
    model: ModeShapeModel,
    sequence: GraphSnapshotSequence,
) -> list[Tensor]:
    """Encode each snapshot into latents ``(N, d)``.

    Parameters
    ----------

    model : GraphKoopmanModel
        See the function signature / summary for ``model``.
    sequence : GraphSnapshotSequence
        See the function signature / summary for ``sequence``.

    Returns
    -------

    list[Tensor]
        See summary line."""
    latents: list[Tensor] = []
    for index in range(sequence.num_timesteps):
        with torch.no_grad():
            latents.append(model.encode_at(sequence, index).detach())
    return latents


def _finite_difference_targets(
    latents: Sequence[Tensor],
    *,
    timestamps: Tensor | None,
    time_step: float,
) -> tuple[Tensor, Tensor]:
    r"""Build stacked ``(Z_t, \dot Z_t)`` pairs via forward differences.

    Parameters
    ----------

    latents : Sequence[Tensor]
        See the function signature / summary for ``latents``.
    timestamps : Tensor | None
        See the function signature / summary for ``timestamps``.
    time_step : float
        See the function signature / summary for ``time_step``.

    Returns
    -------

    states : Tensor
        Stacked node states at times ``0 .. T-2`` with shape ``(S, d)``.
    derivatives : Tensor
        Matching finite-difference targets with shape ``(S, d)``."""
    if len(latents) < 2:
        msg = "derivative mode requires at least two latent frames"
        raise ValueError(msg)

    state_rows: list[Tensor] = []
    deriv_rows: list[Tensor] = []
    for index in range(len(latents) - 1):
        if timestamps is not None:
            delta = float((timestamps[index + 1] - timestamps[index]).item())
        else:
            delta = float(time_step)
        if delta <= 0:
            msg = f"non-positive time increment at index {index}: {delta}"
            raise ValueError(msg)
        z_t = latents[index]
        z_next = latents[index + 1]
        state_rows.append(z_t)
        deriv_rows.append((z_next - z_t) / delta)

    return torch.cat(state_rows, dim=0), torch.cat(deriv_rows, dim=0)


def _discrete_pairs(
    latents: Sequence[Tensor],
) -> tuple[Tensor, Tensor]:
    """Stack ``(Z_t, Z_{t+1})`` node rows for discrete SINDy.

    Parameters
    ----------

    latents : Sequence[Tensor]
        See the function signature / summary for ``latents``.

    Returns
    -------

    tuple[Tensor, Tensor]
        See summary line.

    Raises
    ------

    ValueError
        Raised when inputs are invalid."""
    if len(latents) < 2:
        msg = "discrete mode requires at least two latent frames"
        raise ValueError(msg)
    states = torch.cat(list(latents[:-1]), dim=0)
    targets = torch.cat(list(latents[1:]), dim=0)
    return states, targets


def _library_for_frames(
    latents: Sequence[Tensor],
    *,
    library: SINDyLibrary,
    degree: int,
    edge_index: Tensor | None,
    edge_weight: Tensor | None,
    frame_indices: Sequence[int],
) -> tuple[Tensor, tuple[str, ...]]:
    """Build stacked Θ for selected frames (node rows concatenated).

    Parameters
    ----------

    latents : Sequence[Tensor]
        See the function signature / summary for ``latents``.
    library : SINDyLibrary
        See the function signature / summary for ``library``.
    degree : int
        See the function signature / summary for ``degree``.
    edge_index : Tensor | None
        See the function signature / summary for ``edge_index``.
    edge_weight : Tensor | None
        See the function signature / summary for ``edge_weight``.
    frame_indices : Sequence[int]
        See the function signature / summary for ``frame_indices``.

    Returns
    -------

    tuple[Tensor, tuple[str, ...]]
        See summary line.

    Raises
    ------

    ValueError
        Raised when inputs are invalid."""
    blocks: list[Tensor] = []
    term_names: list[str] | None = None
    for index in frame_indices:
        z = latents[index]
        if library == "poly":
            theta, names = _build_poly_library(z, degree)
        else:
            if edge_index is None:
                msg = "graph library requires edge_index"
                raise ValueError(msg)
            theta, names = _build_graph_library(z, edge_index, edge_weight, degree)
        blocks.append(theta)
        if term_names is None:
            term_names = names
    if term_names is None:
        msg = "no frames provided for library construction"
        raise ValueError(msg)
    return torch.cat(blocks, dim=0), tuple(term_names)


def identify_sparse_dynamics(
    model: ModeShapeModel,
    sequence: GraphSnapshotSequence | Sequence[Data],
    *,
    library: SINDyLibrary = "poly",
    degree: int = 2,
    threshold: float,
    mode: SINDyMode = "discrete",
    max_iter: int = 10,
) -> SINDyReport:
    """Identify sparse latent dynamics via STLSQ on a candidate library.

    Encodes the snapshot sequence with ``model``, builds a polynomial (or
    graph-augmented) feature library over **per-node** latent rows, and fits

    - discrete: ``Z_{t+1} ≈ Θ(Z_t) Ξ``
    - derivative: ``\\dot Z_t ≈ Θ(Z_t) Ξ`` with forward differences

    .. warning::

       The fit describes the model's **learned latent** dynamics, not
       physical governing equations in the original node features.

    Parameters
    ----------
    model : GraphKoopmanModel
        Fitted (or otherwise configured) model used only for encoding.
    sequence : GraphSnapshotSequence or sequence of Data
        Snapshot trajectory. Graph library requires static pairwise topology.
    library : {"poly", "graph"}, optional
        ``"poly"`` uses constant + monomials up to ``degree``. ``"graph"``
        appends the same monomials of ``L_sym Z`` (symmetric normalized
        Laplacian). Default ``"poly"``.
    degree : int, optional
        Maximum total polynomial degree. Default ``2``.
    threshold : float
        Absolute STLSQ threshold. Required (no silent default).
    mode : {"discrete", "derivative"}, optional
        Target construction. Derivative mode uses ``sequence.timestamps``
        when present, otherwise ``model.time_step``. Default ``"discrete"``.
    max_iter : int, optional
        Maximum STLSQ iterations. Default ``10``.

    Returns
    -------
    SINDyReport
        Sparse coefficients, active mask, term names, residual, and
        threshold history.

    Raises
    ------
    ValueError
        If arguments are invalid, the sequence is too short, or graph-library
        topology is missing / dynamic.
    """
    if library not in {"poly", "graph"}:
        msg = f"library must be 'poly' or 'graph', got {library!r}"
        raise ValueError(msg)
    if mode not in {"discrete", "derivative"}:
        msg = f"mode must be 'discrete' or 'derivative', got {mode!r}"
        raise ValueError(msg)
    if degree < 1:
        msg = f"degree must be >= 1, got {degree}"
        raise ValueError(msg)

    resolved = resolve_sequence(sequence)
    if resolved.num_timesteps < 2:
        msg = "identify_sparse_dynamics requires at least two snapshots"
        raise ValueError(msg)

    edge_index: Tensor | None = None
    edge_weight: Tensor | None = None
    if library == "graph":
        require_static_topology(resolved)
        edge_index = resolve_edge_index(resolved[0], None)
        edge_weight = resolve_edge_weight(resolved[0], None)

    latents = _encode_latent_trajectory(model, resolved)
    latent_dim = latents[0].shape[-1]
    if any(frame.shape[-1] != latent_dim for frame in latents):
        msg = "encoded latent dimension is inconsistent across timesteps"
        raise ValueError(msg)

    frame_indices = range(resolved.num_timesteps - 1)
    theta, term_names = _library_for_frames(
        latents,
        library=library,
        degree=degree,
        edge_index=edge_index,
        edge_weight=edge_weight,
        frame_indices=frame_indices,
    )

    if mode == "discrete":
        _, target = _discrete_pairs(latents)
    else:
        _, target = _finite_difference_targets(
            latents,
            timestamps=resolved.timestamps,
            time_step=model.time_step,
        )

    if theta.shape[0] != target.shape[0]:
        msg = (
            "library and target sample counts disagree: "
            f"{theta.shape[0]} vs {target.shape[0]}"
        )
        raise ValueError(msg)

    coefficients, active_mask, history = _stlsq(
        theta,
        target,
        threshold=threshold,
        max_iter=max_iter,
    )
    residual = float(torch.mean((theta @ coefficients - target) ** 2).item())
    return SINDyReport(
        coefficients=coefficients,
        active_mask=active_mask,
        term_names=term_names,
        residual=residual,
        threshold_history=history,
        mode=mode,
        library=library,
    )
