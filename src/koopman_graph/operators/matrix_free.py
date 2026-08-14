"""Matrix-free matvecs / inverses / spectra for networked Koopman operators.

These helpers apply ``K_eff``, solve ``K_eff x = b``, and approximate leading
eigenvalues without assembling the dense ``(N·d, N·d)`` matrix. They are
**operator-math** primitives for ``sparsity="distributed"`` — **not**
trainer DDP / ``[distributed]`` process-group extras, and they do **not**
enable multi-GPU training.

Layout
------
Flat vectors use C-order ``Z.reshape(-1)`` with shape ``(N·d,)``, matching
:meth:`~koopman_graph.operators.GraphKoopmanOperator.effective_matrix` and
``vec(Z)`` for row-state latents ``Z ∈ R^{N×d}``.

Inverse (TASK-1828)
-------------------
:func:`invert_k_eff_graph` / :func:`invert_k_eff_hetero` use a Richardson /
Neumann-style fixed-point iteration with self-dominated left preconditioner
``M = I_N ⊗ K_self``. Defaults: ``max_iters=64``, ``tol=1e-6``. Divergence
raises ``ValueError``.

Spectrum (TASK-1829)
--------------------
:func:`spectrum_k_eff_graph` / :func:`spectrum_k_eff_hetero` use **Arnoldi**
(not Lanczos): networked ``K_eff`` is generally nonsymmetric. Returns the
``num_modes`` largest-modulus Ritz values whose Ritz residual
``||K v - λ v|| / ||v||`` is at most ``tol`` (default ``1e-6``, matching the
inverse residual scale). Krylov size defaults to
``ncv = min(dim, max(2 * num_modes + 2, 20))``. If fewer than ``num_modes``
pairs meet ``tol``, a ``ValueError`` is raised.

References
----------
Design §3.7 / G9 (v0.10.0): matrix-free forward, inverse, and spectrum
primitives for ``sparsity="distributed"``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from koopman_graph.graph_utils import (
    RELATION_NORMALIZATION_MODES,
    RelationNormalization,
    relation_normalized_adjacency_matvec,
)
from koopman_graph.graph_utils.topology import (
    hyperedge_normalized_adjacency_matvec,
    random_walk_normalized_adjacency_matvec,
    symmetric_normalized_adjacency_matvec,
)
from koopman_graph.operators.graph_types import (
    GRAPH_ADJACENCY_MODES,
    GraphAdjacency,
)

__all__ = [
    "DEFAULT_DISTRIBUTED_EIGREG_NUM_MODES",
    "DEFAULT_DISTRIBUTED_SPECTRUM_NUM_MODES",
    "DEFAULT_MATRIX_FREE_INVERSE_MAX_ITERS",
    "DEFAULT_MATRIX_FREE_INVERSE_TOL",
    "DEFAULT_MATRIX_FREE_SPECTRUM_TOL",
    "MatrixFreeInverseResult",
    "MatrixFreeSpectrumResult",
    "apply_k_eff_graph",
    "apply_k_eff_hetero",
    "apply_k_eff_hypergraph",
    "flatten_node_latents",
    "invert_k_eff_graph",
    "invert_k_eff_hetero",
    "invert_k_eff_hypergraph",
    "spectrum_k_eff_graph",
    "spectrum_k_eff_hetero",
    "spectrum_k_eff_hypergraph",
    "unflatten_node_latents",
]

DEFAULT_MATRIX_FREE_INVERSE_MAX_ITERS = 64
DEFAULT_MATRIX_FREE_INVERSE_TOL = 1e-6
DEFAULT_MATRIX_FREE_SPECTRUM_TOL = 1e-6
# Leading-modulus modes for distributed spectrum / eig-reg surrogates.
DEFAULT_DISTRIBUTED_SPECTRUM_NUM_MODES = 8
DEFAULT_DISTRIBUTED_EIGREG_NUM_MODES = DEFAULT_DISTRIBUTED_SPECTRUM_NUM_MODES


@dataclass(frozen=True)
class MatrixFreeInverseResult:
    """Result of a matrix-free Richardson / Neumann inverse solve.

    Attributes
    ----------
    solution : Tensor
        Flat solution ``x`` with the same shape as the right-hand side.
    residual_norm : Tensor
        Final absolute residual norm ``||b - K_eff x||``.
    iterations : int
        Number of Richardson updates performed.
    converged : bool
        ``True`` when the relative (or absolute) residual met ``tol``.
    """

    solution: Tensor
    residual_norm: Tensor
    iterations: int
    converged: bool


@dataclass(frozen=True)
class MatrixFreeSpectrumResult:
    """Result of a matrix-free Arnoldi spectrum estimate.

    Attributes
    ----------
    eigenvalues : Tensor
        Complex eigenvalues with shape ``(num_modes,)``, sorted by
        descending magnitude.
    num_modes : int
        Number of requested (and returned) modes.
    residual_norms : Tensor
        Per-mode Ritz residuals ``||K v - λ v|| / ||v||`` with shape
        ``(num_modes,)``.
    converged : bool
        ``True`` when all returned modes met ``tol``.
    """

    eigenvalues: Tensor
    num_modes: int
    residual_norms: Tensor
    converged: bool


def flatten_node_latents(z: Tensor) -> Tensor:
    """Flatten row-state latents ``(N, d)`` to ``(N·d,)`` in C-order.

    Parameters
    ----------
    z : Tensor
        Latent node states with shape ``(num_nodes, latent_dim)``.

    Returns
    -------
    Tensor
        Flat vector with shape ``(num_nodes * latent_dim,)``.

    Raises
    ------
    ValueError
        If ``z`` is not 2-D.
    """
    if z.ndim != 2:
        msg = (
            "flatten_node_latents expects z with shape "
            f"(num_nodes, latent_dim), got {tuple(z.shape)}"
        )
        raise ValueError(msg)
    return z.reshape(-1)


def unflatten_node_latents(
    flat: Tensor,
    *,
    num_nodes: int,
    latent_dim: int,
) -> Tensor:
    """Reshape a flat ``(N·d,)`` vector to row-state latents ``(N, d)``.

    Parameters
    ----------
    flat : Tensor
        Flat latent vector with shape ``(num_nodes * latent_dim,)``.
    num_nodes : int
        Node count ``N``.
    latent_dim : int
        Latent width ``d``.

    Returns
    -------
    Tensor
        Latents with shape ``(num_nodes, latent_dim)``.

    Raises
    ------
    ValueError
        If sizes are non-positive or ``flat`` has the wrong length.
    """
    if num_nodes < 1:
        msg = f"num_nodes must be positive, got {num_nodes}"
        raise ValueError(msg)
    if latent_dim < 1:
        msg = f"latent_dim must be positive, got {latent_dim}"
        raise ValueError(msg)
    expected = num_nodes * latent_dim
    if flat.ndim != 1 or int(flat.numel()) != expected:
        msg = (
            "unflatten_node_latents expects flat with shape "
            f"({expected},), got {tuple(flat.shape)}"
        )
        raise ValueError(msg)
    return flat.reshape(num_nodes, latent_dim)


def _validate_square_factor(matrix: Tensor, *, name: str, latent_dim: int) -> None:
    """Require a square ``(d, d)`` factor tensor.

    Parameters
    ----------
    matrix : Tensor
        Candidate factor.
    name : str
        Parameter name for error messages.
    latent_dim : int
        Expected width ``d``.

    Raises
    ------
    ValueError
        If ``matrix`` is not ``(latent_dim, latent_dim)``.
    """
    expected = (latent_dim, latent_dim)
    if matrix.shape != expected:
        msg = f"{name} must have shape {expected}, got {tuple(matrix.shape)}"
        raise ValueError(msg)


def apply_k_eff_graph(
    flat: Tensor,
    *,
    k_self: Tensor,
    k_nbr: Tensor,
    edge_index: Tensor,
    num_nodes: int,
    adjacency: GraphAdjacency = "symmetric",
    edge_weight: Tensor | None = None,
    k_bwd: Tensor | None = None,
) -> Tensor:
    """Apply graph ``K_eff`` to a flat latent without assembling the dense map.

    Implements the uncontrolled sparse forward matching
    :class:`~koopman_graph.operators.GraphKoopmanOperator`::

        Z_next = Z @ K_self.T + Â Z @ K_nbr.T

    (plus a backward dual term when ``adjacency="dual_random_walk"``).

    Parameters
    ----------
    flat : Tensor
        Flat latents with shape ``(N·d,)``.
    k_self : Tensor
        Self-coupling factor ``(d, d)``.
    k_nbr : Tensor
        Neighbor (or forward) coupling factor ``(d, d)``.
    edge_index : Tensor
        Edge index ``(2, E)``.
    num_nodes : int
        Node count ``N``.
    adjacency : {"symmetric", "random_walk", "dual_random_walk"}, optional
        Neighbor normalization mode. Default ``"symmetric"``.
    edge_weight : Tensor or None, optional
        Optional edge weights ``(E,)``.
    k_bwd : Tensor or None, optional
        Backward factor required when ``adjacency="dual_random_walk"``.

    Returns
    -------
    Tensor
        Flat result with shape ``(N·d,)``.

    Raises
    ------
    ValueError
        If shapes, ``adjacency``, or ``k_bwd`` are inconsistent.
    """
    if adjacency not in GRAPH_ADJACENCY_MODES:
        msg = (
            "adjacency must be one of "
            f"{sorted(GRAPH_ADJACENCY_MODES)}, got {adjacency!r}"
        )
        raise ValueError(msg)
    latent_dim = int(k_self.shape[0])
    _validate_square_factor(k_self, name="k_self", latent_dim=latent_dim)
    _validate_square_factor(k_nbr, name="k_nbr", latent_dim=latent_dim)
    z = unflatten_node_latents(flat, num_nodes=num_nodes, latent_dim=latent_dim)
    self_term = z @ k_self.transpose(-2, -1)

    if adjacency == "symmetric":
        neighbor = symmetric_normalized_adjacency_matvec(
            edge_index,
            z,
            edge_weight=edge_weight,
            num_nodes=num_nodes,
        )
        z_next = self_term + neighbor @ k_nbr.transpose(-2, -1)
        return flatten_node_latents(z_next)

    neighbor_fwd = random_walk_normalized_adjacency_matvec(
        edge_index,
        z,
        edge_weight=edge_weight,
        num_nodes=num_nodes,
        direction="forward",
    )
    z_next = self_term + neighbor_fwd @ k_nbr.transpose(-2, -1)
    if adjacency == "random_walk":
        return flatten_node_latents(z_next)
    if k_bwd is None:
        msg = "k_bwd is required when adjacency='dual_random_walk'"
        raise ValueError(msg)
    _validate_square_factor(k_bwd, name="k_bwd", latent_dim=latent_dim)
    neighbor_bwd = random_walk_normalized_adjacency_matvec(
        edge_index,
        z,
        edge_weight=edge_weight,
        num_nodes=num_nodes,
        direction="backward",
    )
    z_next = z_next + neighbor_bwd @ k_bwd.transpose(-2, -1)
    return flatten_node_latents(z_next)


def apply_k_eff_hetero(
    flat: Tensor,
    *,
    k_self: Tensor,
    k_relations: Sequence[Tensor],
    edge_indices: Sequence[Tensor],
    num_nodes: int,
    normalization: RelationNormalization = "rgcn_in_degree",
    edge_weights: Sequence[Tensor | None] | None = None,
) -> Tensor:
    """Apply multiplex hetero ``K_eff`` without assembling the dense map.

    Implements the uncontrolled shared-d forward matching
    :class:`~koopman_graph.operators.HeteroGraphKoopmanOperator`::

        Z_next = Z @ K_self.T + sum_r (Â_r Z) @ K_r.T

    Parameters
    ----------
    flat : Tensor
        Flat latents with shape ``(N·d,)``.
    k_self : Tensor
        Shared self-coupling factor ``(d, d)``.
    k_relations : sequence of Tensor
        Per-relation factors, each ``(d, d)``, length ``|R|``.
    edge_indices : sequence of Tensor
        Per-relation edge indices aligned with ``k_relations``.
    num_nodes : int
        Stacked node count ``N``.
    normalization : {"rgcn_in_degree", "random_walk"}, optional
        Per-relation adjacency normalization. Default ``"rgcn_in_degree"``.
    edge_weights : sequence of Tensor or None, optional
        Optional per-relation weights (same length as ``edge_indices``).

    Returns
    -------
    Tensor
        Flat result with shape ``(N·d,)``.

    Raises
    ------
    ValueError
        If shapes, bank counts, or ``normalization`` are inconsistent.
    """
    if normalization not in RELATION_NORMALIZATION_MODES:
        msg = (
            "normalization must be one of "
            f"{sorted(RELATION_NORMALIZATION_MODES)}, got {normalization!r}"
        )
        raise ValueError(msg)
    latent_dim = int(k_self.shape[0])
    _validate_square_factor(k_self, name="k_self", latent_dim=latent_dim)
    num_relations = len(k_relations)
    if num_relations < 1:
        msg = "k_relations must contain at least one relation factor"
        raise ValueError(msg)
    if len(edge_indices) != num_relations:
        msg = (
            f"edge_indices length ({len(edge_indices)}) must match "
            f"k_relations length ({num_relations})"
        )
        raise ValueError(msg)
    if edge_weights is None:
        weights: list[Tensor | None] = [None] * num_relations
    else:
        if len(edge_weights) != num_relations:
            msg = (
                f"edge_weights length ({len(edge_weights)}) must match "
                f"k_relations length ({num_relations})"
            )
            raise ValueError(msg)
        weights = list(edge_weights)

    for idx, k_rel in enumerate(k_relations):
        _validate_square_factor(
            k_rel,
            name=f"k_relations[{idx}]",
            latent_dim=latent_dim,
        )

    z = unflatten_node_latents(flat, num_nodes=num_nodes, latent_dim=latent_dim)
    z_next = z @ k_self.transpose(-2, -1)
    for relation_idx, edge_index in enumerate(edge_indices):
        aggregated = relation_normalized_adjacency_matvec(
            edge_index,
            z,
            edge_weight=weights[relation_idx],
            num_nodes=num_nodes,
            normalization=normalization,
        )
        z_next = z_next + aggregated @ k_relations[relation_idx].transpose(-2, -1)
    return flatten_node_latents(z_next)


def _invert_square_factor(matrix: Tensor) -> Tensor:
    """Invert a small square factor, raising if the factor is singular.

    Parameters
    ----------
    matrix : Tensor
        Square factor ``(d, d)``.

    Returns
    -------
    Tensor
        Inverse with the same shape.

    Raises
    ------
    ValueError
        If ``matrix`` is not square or is numerically singular.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        msg = (
            "_invert_square_factor expects a square 2-D matrix, "
            f"got shape {tuple(matrix.shape)}"
        )
        raise ValueError(msg)
    try:
        return torch.linalg.inv(matrix)
    except RuntimeError as exc:
        msg = "k_self is singular; matrix-free inverse preconditioner failed"
        raise ValueError(msg) from exc


def _apply_self_preconditioner(
    flat: Tensor,
    k_self_inv: Tensor,
    *,
    num_nodes: int,
    latent_dim: int,
) -> Tensor:
    """Apply ``M^{-1} = I_N ⊗ K_self^{-1}`` in row-state layout.

    Parameters
    ----------
    flat : Tensor
        Flat residual / vector ``(N·d,)``.
    k_self_inv : Tensor
        Inverse self factor ``(d, d)``.
    num_nodes : int
        Node count ``N``.
    latent_dim : int
        Latent width ``d``.

    Returns
    -------
    Tensor
        Flat preconditioned vector ``(N·d,)``.
    """
    z = unflatten_node_latents(flat, num_nodes=num_nodes, latent_dim=latent_dim)
    return flatten_node_latents(z @ k_self_inv.transpose(-2, -1))


def _richardson_invert(
    apply_k: Callable[[Tensor], Tensor],
    apply_m_inv: Callable[[Tensor], Tensor],
    rhs: Tensor,
    *,
    max_iters: int,
    tol: float,
    x0: Tensor | None,
) -> MatrixFreeInverseResult:
    """Richardson iteration ``x ← x + M^{-1}(b - K x)``.

    Parameters
    ----------
    apply_k : callable
        Matrix-free ``K_eff`` matvec on flat vectors.
    apply_m_inv : callable
        Left preconditioner ``M^{-1}`` on flat vectors.
    rhs : Tensor
        Flat right-hand side ``b``.
    max_iters : int
        Maximum Richardson updates.
    tol : float
        Relative residual tolerance (absolute when ``||b|| = 0``).
    x0 : Tensor or None
        Optional initial guess; defaults to zeros.

    Returns
    -------
    MatrixFreeInverseResult
        Converged solve result.

    Raises
    ------
    ValueError
        If ``max_iters`` / ``tol`` are invalid, shapes mismatch, or the
        iteration fails to meet ``tol`` within ``max_iters``.
    """
    if max_iters < 1:
        msg = f"max_iters must be positive, got {max_iters}"
        raise ValueError(msg)
    if tol <= 0.0:
        msg = f"tol must be positive, got {tol}"
        raise ValueError(msg)
    if rhs.ndim != 1:
        msg = f"rhs must be a 1-D flat vector, got shape {tuple(rhs.shape)}"
        raise ValueError(msg)

    x = torch.zeros_like(rhs) if x0 is None else x0
    if x.shape != rhs.shape:
        msg = f"x0 must have shape {tuple(rhs.shape)}, got {tuple(x.shape)}"
        raise ValueError(msg)

    rhs_norm = float(torch.linalg.vector_norm(rhs).detach())
    residual = rhs - apply_k(x)
    residual_norm = torch.linalg.vector_norm(residual)
    denom = rhs_norm if rhs_norm > 0.0 else 1.0
    if float(residual_norm.detach()) / denom <= tol:
        return MatrixFreeInverseResult(
            solution=x,
            residual_norm=residual_norm,
            iterations=0,
            converged=True,
        )

    for iteration in range(1, max_iters + 1):
        x = x + apply_m_inv(residual)
        residual = rhs - apply_k(x)
        residual_norm = torch.linalg.vector_norm(residual)
        if float(residual_norm.detach()) / denom <= tol:
            return MatrixFreeInverseResult(
                solution=x,
                residual_norm=residual_norm,
                iterations=iteration,
                converged=True,
            )

    msg = (
        "matrix-free Richardson inverse failed to converge within "
        f"max_iters={max_iters} (tol={tol}, "
        f"residual_norm={float(residual_norm.detach())}, "
        f"rhs_norm={rhs_norm})"
    )
    raise ValueError(msg)


def invert_k_eff_graph(
    rhs: Tensor,
    *,
    k_self: Tensor,
    k_nbr: Tensor,
    edge_index: Tensor,
    num_nodes: int,
    adjacency: GraphAdjacency = "symmetric",
    edge_weight: Tensor | None = None,
    k_bwd: Tensor | None = None,
    max_iters: int = DEFAULT_MATRIX_FREE_INVERSE_MAX_ITERS,
    tol: float = DEFAULT_MATRIX_FREE_INVERSE_TOL,
    x0: Tensor | None = None,
) -> MatrixFreeInverseResult:
    """Solve ``K_eff x = b`` for a graph operator via Richardson iteration.

    Uses :func:`apply_k_eff_graph` for the residual matvec and
    ``M^{-1} = I_N ⊗ K_self^{-1}`` as a left preconditioner.

    Parameters
    ----------
    rhs : Tensor
        Flat right-hand side ``b`` with shape ``(N·d,)``.
    k_self, k_nbr : Tensor
        Self / neighbor factors ``(d, d)``.
    edge_index : Tensor
        Edge index ``(2, E)``.
    num_nodes : int
        Node count ``N``.
    adjacency : {"symmetric", "random_walk", "dual_random_walk"}, optional
        Neighbor normalization mode.
    edge_weight : Tensor or None, optional
        Optional edge weights.
    k_bwd : Tensor or None, optional
        Backward factor when ``adjacency="dual_random_walk"``.
    max_iters : int, optional
        Maximum Richardson updates. Default ``64``.
    tol : float, optional
        Relative residual tolerance. Default ``1e-6``.
    x0 : Tensor or None, optional
        Optional initial guess (zeros when omitted).

    Returns
    -------
    MatrixFreeInverseResult
        Converged flat solution and residual diagnostics.

    Raises
    ------
    ValueError
        If inputs are invalid, ``K_self`` is singular, or the iteration
        diverges / stalls past ``max_iters``.
    """
    latent_dim = int(k_self.shape[0])
    _validate_square_factor(k_self, name="k_self", latent_dim=latent_dim)
    k_self_inv = _invert_square_factor(k_self)

    def _apply_k(flat: Tensor) -> Tensor:
        """Internal helper: apply k.

        Parameters
        ----------
        flat
            Value for ``flat``.

        Returns
        -------
        object
            Function result.
        """
        return apply_k_eff_graph(
            flat,
            k_self=k_self,
            k_nbr=k_nbr,
            edge_index=edge_index,
            num_nodes=num_nodes,
            adjacency=adjacency,
            edge_weight=edge_weight,
            k_bwd=k_bwd,
        )

    def _apply_m_inv(flat: Tensor) -> Tensor:
        """Internal helper: apply m inv.

        Parameters
        ----------
        flat
            Value for ``flat``.

        Returns
        -------
        object
            Function result.
        """
        return _apply_self_preconditioner(
            flat,
            k_self_inv,
            num_nodes=num_nodes,
            latent_dim=latent_dim,
        )

    return _richardson_invert(
        _apply_k,
        _apply_m_inv,
        rhs,
        max_iters=max_iters,
        tol=tol,
        x0=x0,
    )


def invert_k_eff_hetero(
    rhs: Tensor,
    *,
    k_self: Tensor,
    k_relations: Sequence[Tensor],
    edge_indices: Sequence[Tensor],
    num_nodes: int,
    normalization: RelationNormalization = "rgcn_in_degree",
    edge_weights: Sequence[Tensor | None] | None = None,
    max_iters: int = DEFAULT_MATRIX_FREE_INVERSE_MAX_ITERS,
    tol: float = DEFAULT_MATRIX_FREE_INVERSE_TOL,
    x0: Tensor | None = None,
) -> MatrixFreeInverseResult:
    """Solve multiplex hetero ``K_eff x = b`` via Richardson iteration.

    Uses :func:`apply_k_eff_hetero` for the residual matvec and
    ``M^{-1} = I_N ⊗ K_self^{-1}`` as a left preconditioner.

    Parameters
    ----------
    rhs : Tensor
        Flat right-hand side ``b`` with shape ``(N·d,)``.
    k_self : Tensor
        Shared self-coupling factor ``(d, d)``.
    k_relations : sequence of Tensor
        Per-relation factors ``(d, d)``.
    edge_indices : sequence of Tensor
        Per-relation edge indices.
    num_nodes : int
        Stacked node count ``N``.
    normalization : {"rgcn_in_degree", "random_walk"}, optional
        Per-relation adjacency normalization.
    edge_weights : sequence of Tensor or None, optional
        Optional per-relation weights.
    max_iters : int, optional
        Maximum Richardson updates. Default ``64``.
    tol : float, optional
        Relative residual tolerance. Default ``1e-6``.
    x0 : Tensor or None, optional
        Optional initial guess (zeros when omitted).

    Returns
    -------
    MatrixFreeInverseResult
        Converged flat solution and residual diagnostics.

    Raises
    ------
    ValueError
        If inputs are invalid, ``K_self`` is singular, or the iteration
        diverges / stalls past ``max_iters``.
    """
    latent_dim = int(k_self.shape[0])
    _validate_square_factor(k_self, name="k_self", latent_dim=latent_dim)
    k_self_inv = _invert_square_factor(k_self)

    def _apply_k(flat: Tensor) -> Tensor:
        """Internal helper: apply k.

        Parameters
        ----------
        flat
            Value for ``flat``.

        Returns
        -------
        object
            Function result.
        """
        return apply_k_eff_hetero(
            flat,
            k_self=k_self,
            k_relations=k_relations,
            edge_indices=edge_indices,
            num_nodes=num_nodes,
            normalization=normalization,
            edge_weights=edge_weights,
        )

    def _apply_m_inv(flat: Tensor) -> Tensor:
        """Internal helper: apply m inv.

        Parameters
        ----------
        flat
            Value for ``flat``.

        Returns
        -------
        object
            Function result.
        """
        return _apply_self_preconditioner(
            flat,
            k_self_inv,
            num_nodes=num_nodes,
            latent_dim=latent_dim,
        )

    return _richardson_invert(
        _apply_k,
        _apply_m_inv,
        rhs,
        max_iters=max_iters,
        tol=tol,
        x0=x0,
    )


def _default_ncv(dim: int, num_modes: int) -> int:
    """Return the default Arnoldi Krylov subspace size.

    Parameters
    ----------
    dim : int
        Operator dimension ``N·d``.
    num_modes : int
        Requested number of eigenvalues.

    Returns
    -------
    int
        ``min(dim, max(2 * num_modes + 2, 20))``.
    """
    return min(dim, max(2 * num_modes + 2, 20))


def _arnoldi_eigs(
    apply_k: Callable[[Tensor], Tensor],
    dim: int,
    *,
    num_modes: int,
    tol: float,
    ncv: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> MatrixFreeSpectrumResult:
    """Compute largest-modulus Ritz values via classical Arnoldi.

    Parameters
    ----------
    apply_k : callable
        Matrix-free ``K_eff`` matvec on real flat vectors of length ``dim``.
    dim : int
        Operator dimension.
    num_modes : int
        Number of eigenvalues to return.
    tol : float
        Ritz residual tolerance.
    ncv : int
        Krylov subspace dimension (``num_modes <= ncv <= dim``).
    seed : int
        RNG seed for the starting vector.
    device : torch.device
        Device for Arnoldi factors.
    dtype : torch.dtype
        Real floating dtype for Arnoldi factors.

    Returns
    -------
    MatrixFreeSpectrumResult
        Top ``num_modes`` Ritz values with residuals.

    Raises
    ------
    ValueError
        If parameters are invalid or fewer than ``num_modes`` pairs meet
        ``tol``.
    """
    if dim < 1:
        msg = f"dim must be positive, got {dim}"
        raise ValueError(msg)
    if num_modes < 1 or num_modes > dim:
        msg = f"num_modes must be in [1, {dim}], got {num_modes}"
        raise ValueError(msg)
    if tol <= 0.0:
        msg = f"tol must be positive, got {tol}"
        raise ValueError(msg)
    if ncv < num_modes or ncv > dim:
        msg = f"ncv must be in [{num_modes}, {dim}], got {ncv}"
        raise ValueError(msg)

    # Accumulate Arnoldi factors in float64 for stability (esp. repeated
    # spectra); matvecs still run in the caller's working dtype.
    work_dtype = torch.float64
    complex_dtype = torch.complex128
    out_complex = torch.complex128 if dtype == torch.float64 else torch.complex64

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    v0 = torch.randn(dim, generator=generator, dtype=work_dtype)
    v0 = v0.to(device=device)
    v0 = v0 / torch.linalg.vector_norm(v0)

    def _apply_work(flat64: Tensor) -> Tensor:
        """Internal helper: apply work.

        Parameters
        ----------
        flat64
            Value for ``flat64``.

        Returns
        -------
        object
            Function result.
        """
        return apply_k(flat64.to(dtype=dtype)).to(dtype=work_dtype)

    # V[:, j] is the j-th Arnoldi basis vector; H is (ncv+1) x ncv Hessenberg.
    basis = torch.zeros(dim, ncv + 1, dtype=work_dtype, device=device)
    hessenberg = torch.zeros(ncv + 1, ncv, dtype=work_dtype, device=device)
    basis[:, 0] = v0

    krylov = ncv
    for col in range(ncv):
        w = _apply_work(basis[:, col])
        for row in range(col + 1):
            coeff = torch.dot(basis[:, row], w)
            hessenberg[row, col] = coeff
            w = w - coeff * basis[:, row]
        # Modified Gram-Schmidt reorthogonalization pass.
        for row in range(col + 1):
            coeff = torch.dot(basis[:, row], w)
            hessenberg[row, col] = hessenberg[row, col] + coeff
            w = w - coeff * basis[:, row]
        beta = torch.linalg.vector_norm(w)
        hessenberg[col + 1, col] = beta
        if float(beta.detach()) <= tol * max(
            1.0, float(torch.linalg.vector_norm(basis[:, col]).detach())
        ):
            krylov = col + 1
            break
        if col + 1 < ncv:
            basis[:, col + 1] = w / beta

    h_square = hessenberg[:krylov, :krylov]
    eigvals_h, eigvecs_h = torch.linalg.eig(h_square.to(complex_dtype))
    # Ritz vectors in ambient space: V_m @ y
    v_m = basis[:, :krylov].to(complex_dtype)
    ritz_vectors = v_m @ eigvecs_h

    residuals: list[Tensor] = []
    for index in range(krylov):
        lam = eigvals_h[index]
        ritz_vec = ritz_vectors[:, index]
        real_part = _apply_work(ritz_vec.real)
        imag_part = _apply_work(ritz_vec.imag)
        k_ritz = torch.complex(real_part, imag_part)
        resid = torch.linalg.vector_norm(k_ritz - lam * ritz_vec)
        scale = torch.linalg.vector_norm(ritz_vec)
        residuals.append(resid / scale.clamp_min(1e-30))

    residual_tensor = torch.stack(residuals)
    # Deterministic order: |λ| ↓ (rounded), Re λ ↓, Im λ ↓.
    # Rounding |λ| keeps conjugate pairs tied under float noise.
    order = sorted(
        range(krylov),
        key=lambda i: (
            -round(float(eigvals_h[i].abs().detach()), 10),
            -float(eigvals_h[i].real.detach()),
            -float(eigvals_h[i].imag.detach()),
        ),
    )
    order_t = torch.tensor(order, device=eigvals_h.device, dtype=torch.long)
    eigvals_sorted = eigvals_h[order_t]
    resid_sorted = residual_tensor[order_t]

    accepted_vals: list[Tensor] = []
    accepted_resids: list[Tensor] = []
    for index in range(krylov):
        if float(resid_sorted[index].detach()) <= tol:
            accepted_vals.append(eigvals_sorted[index])
            accepted_resids.append(resid_sorted[index])
        if len(accepted_vals) >= num_modes:
            break

    if len(accepted_vals) < num_modes:
        msg = (
            "matrix-free Arnoldi spectrum found only "
            f"{len(accepted_vals)} of {num_modes} modes within tol={tol} "
            f"(ncv={ncv}, krylov={krylov})"
        )
        raise ValueError(msg)

    eigenvalues = torch.stack(accepted_vals[:num_modes]).to(dtype=out_complex)
    residual_norms = torch.stack(accepted_resids[:num_modes]).to(dtype=work_dtype)
    if dtype != torch.float64:
        residual_norms = residual_norms.to(dtype=torch.float32)
    return MatrixFreeSpectrumResult(
        eigenvalues=eigenvalues,
        num_modes=num_modes,
        residual_norms=residual_norms,
        converged=True,
    )


def spectrum_k_eff_graph(
    *,
    k_self: Tensor,
    k_nbr: Tensor,
    edge_index: Tensor,
    num_nodes: int,
    num_modes: int,
    adjacency: GraphAdjacency = "symmetric",
    edge_weight: Tensor | None = None,
    k_bwd: Tensor | None = None,
    tol: float = DEFAULT_MATRIX_FREE_SPECTRUM_TOL,
    ncv: int | None = None,
    max_arnoldi: int | None = None,
    seed: int = 0,
) -> MatrixFreeSpectrumResult:
    """Approximate leading graph ``K_eff`` eigenvalues via Arnoldi.

    Parameters
    ----------
    k_self, k_nbr : Tensor
        Self / neighbor factors ``(d, d)``.
    edge_index : Tensor
        Edge index ``(2, E)``.
    num_nodes : int
        Node count ``N``.
    num_modes : int
        Number of largest-modulus eigenvalues to return.
    adjacency : {"symmetric", "random_walk", "dual_random_walk"}, optional
        Neighbor normalization mode.
    edge_weight : Tensor or None, optional
        Optional edge weights.
    k_bwd : Tensor or None, optional
        Backward factor when ``adjacency="dual_random_walk"``.
    tol : float, optional
        Ritz residual tolerance. Default ``1e-6``.
    ncv : int or None, optional
        Krylov subspace size. Default ``min(N·d, max(2*num_modes+2, 20))``.
    max_arnoldi : int or None, optional
        Alias for ``ncv`` (single restart-free factorization). When both are
        set they must agree.
    seed : int, optional
        Starting-vector RNG seed. Default ``0``.

    Returns
    -------
    MatrixFreeSpectrumResult
        Top ``num_modes`` eigenvalues and Ritz residuals.

    Raises
    ------
    ValueError
        If parameters are inconsistent or Arnoldi fails to accept
        ``num_modes`` Ritz pairs.
    """
    latent_dim = int(k_self.shape[0])
    dim = num_nodes * latent_dim
    resolved_ncv = _default_ncv(dim, num_modes) if ncv is None else int(ncv)
    if max_arnoldi is not None:
        if ncv is not None and int(max_arnoldi) != resolved_ncv:
            msg = (
                "max_arnoldi and ncv must agree when both are provided; "
                f"got max_arnoldi={max_arnoldi}, ncv={ncv}"
            )
            raise ValueError(msg)
        resolved_ncv = int(max_arnoldi)

    def _apply_k(flat: Tensor) -> Tensor:
        """Internal helper: apply k.

        Parameters
        ----------
        flat
            Value for ``flat``.

        Returns
        -------
        object
            Function result.
        """
        return apply_k_eff_graph(
            flat,
            k_self=k_self,
            k_nbr=k_nbr,
            edge_index=edge_index,
            num_nodes=num_nodes,
            adjacency=adjacency,
            edge_weight=edge_weight,
            k_bwd=k_bwd,
        )

    return _arnoldi_eigs(
        _apply_k,
        dim,
        num_modes=num_modes,
        tol=tol,
        ncv=resolved_ncv,
        seed=seed,
        device=k_self.device,
        dtype=k_self.dtype if k_self.is_floating_point() else torch.float32,
    )


def spectrum_k_eff_hetero(
    *,
    k_self: Tensor,
    k_relations: Sequence[Tensor],
    edge_indices: Sequence[Tensor],
    num_nodes: int,
    num_modes: int,
    normalization: RelationNormalization = "rgcn_in_degree",
    edge_weights: Sequence[Tensor | None] | None = None,
    tol: float = DEFAULT_MATRIX_FREE_SPECTRUM_TOL,
    ncv: int | None = None,
    max_arnoldi: int | None = None,
    seed: int = 0,
) -> MatrixFreeSpectrumResult:
    """Approximate leading multiplex hetero ``K_eff`` eigenvalues via Arnoldi.

    Parameters
    ----------
    k_self : Tensor
        Shared self-coupling factor ``(d, d)``.
    k_relations : sequence of Tensor
        Per-relation factors ``(d, d)``.
    edge_indices : sequence of Tensor
        Per-relation edge indices.
    num_nodes : int
        Stacked node count ``N``.
    num_modes : int
        Number of largest-modulus eigenvalues to return.
    normalization : {"rgcn_in_degree", "random_walk"}, optional
        Per-relation adjacency normalization.
    edge_weights : sequence of Tensor or None, optional
        Optional per-relation weights.
    tol : float, optional
        Ritz residual tolerance. Default ``1e-6``.
    ncv : int or None, optional
        Krylov subspace size. Default ``min(N·d, max(2*num_modes+2, 20))``.
    max_arnoldi : int or None, optional
        Alias for ``ncv`` (single restart-free factorization).
    seed : int, optional
        Starting-vector RNG seed. Default ``0``.

    Returns
    -------
    MatrixFreeSpectrumResult
        Top ``num_modes`` eigenvalues and Ritz residuals.

    Raises
    ------
    ValueError
        If parameters are inconsistent or Arnoldi fails to accept
        ``num_modes`` Ritz pairs.
    """
    latent_dim = int(k_self.shape[0])
    dim = num_nodes * latent_dim
    resolved_ncv = _default_ncv(dim, num_modes) if ncv is None else int(ncv)
    if max_arnoldi is not None:
        if ncv is not None and int(max_arnoldi) != resolved_ncv:
            msg = (
                "max_arnoldi and ncv must agree when both are provided; "
                f"got max_arnoldi={max_arnoldi}, ncv={ncv}"
            )
            raise ValueError(msg)
        resolved_ncv = int(max_arnoldi)

    def _apply_k(flat: Tensor) -> Tensor:
        """Internal helper: apply k.

        Parameters
        ----------
        flat
            Value for ``flat``.

        Returns
        -------
        object
            Function result.
        """
        return apply_k_eff_hetero(
            flat,
            k_self=k_self,
            k_relations=k_relations,
            edge_indices=edge_indices,
            num_nodes=num_nodes,
            normalization=normalization,
            edge_weights=edge_weights,
        )

    return _arnoldi_eigs(
        _apply_k,
        dim,
        num_modes=num_modes,
        tol=tol,
        ncv=resolved_ncv,
        seed=seed,
        device=k_self.device,
        dtype=k_self.dtype if k_self.is_floating_point() else torch.float32,
    )


def apply_k_eff_hypergraph(
    flat: Tensor,
    *,
    k_self: Tensor,
    k_hedge: Tensor,
    hyperedge_index: Tensor,
    num_nodes: int,
    hyperedge_weight: Tensor | None = None,
) -> Tensor:
    """Apply Zhou-symmetric hypergraph ``K_eff`` without a dense assembly.

    Parameters
    ----------
    flat : Tensor
        Flat latents ``(N·d,)``.
    k_self, k_hedge : Tensor
        Self and hyperedge factors ``(d, d)``.
    hyperedge_index : Tensor
        Bipartite incidence.
    num_nodes : int
        Node count.
    hyperedge_weight : Tensor or None, optional
        Optional weights.

    Returns
    -------
    Tensor
        Flat ``K_eff z``.
    """
    latent_dim = int(k_self.shape[0])
    _validate_square_factor(k_self, name="k_self", latent_dim=latent_dim)
    _validate_square_factor(k_hedge, name="k_hedge", latent_dim=latent_dim)
    z = unflatten_node_latents(flat, num_nodes=num_nodes, latent_dim=latent_dim)
    coupled = hyperedge_normalized_adjacency_matvec(
        hyperedge_index,
        z,
        hyperedge_weight=hyperedge_weight,
        num_nodes=num_nodes,
    )
    z_next = z @ k_self.transpose(-2, -1) + coupled @ k_hedge.transpose(-2, -1)
    return flatten_node_latents(z_next)


def invert_k_eff_hypergraph(
    rhs: Tensor,
    *,
    k_self: Tensor,
    k_hedge: Tensor,
    hyperedge_index: Tensor,
    num_nodes: int,
    hyperedge_weight: Tensor | None = None,
    max_iters: int = DEFAULT_MATRIX_FREE_INVERSE_MAX_ITERS,
    tol: float = DEFAULT_MATRIX_FREE_INVERSE_TOL,
    x0: Tensor | None = None,
) -> MatrixFreeInverseResult:
    """Solve hypergraph ``K_eff x = b`` via Richardson iteration.

    Parameters
    ----------
    rhs : Tensor
        Flat right-hand side.
    k_self, k_hedge : Tensor
        Factor matrices.
    hyperedge_index : Tensor
        Incidence.
    num_nodes : int
        Node count.
    hyperedge_weight : Tensor or None, optional
        Optional weights.
    max_iters, tol, x0
        Richardson controls matching :func:`invert_k_eff_graph`.

    Returns
    -------
    MatrixFreeInverseResult
        Approximate solution and residual.
    """
    latent_dim = int(k_self.shape[0])
    k_self_inv = _invert_square_factor(k_self)

    def _apply_k(flat: Tensor) -> Tensor:
        """Internal helper: apply hypergraph ``K_eff``.

        Parameters
        ----------
        flat : Tensor
            Flat latents.

        Returns
        -------
        Tensor
            Flat image.
        """
        return apply_k_eff_hypergraph(
            flat,
            k_self=k_self,
            k_hedge=k_hedge,
            hyperedge_index=hyperedge_index,
            num_nodes=num_nodes,
            hyperedge_weight=hyperedge_weight,
        )

    def _apply_m_inv(flat: Tensor) -> Tensor:
        """Internal helper: apply the self-factor preconditioner.

        Parameters
        ----------
        flat : Tensor
            Flat residual.

        Returns
        -------
        Tensor
            Preconditioned residual.
        """
        return _apply_self_preconditioner(
            flat,
            k_self_inv,
            num_nodes=num_nodes,
            latent_dim=latent_dim,
        )

    return _richardson_invert(
        _apply_k,
        _apply_m_inv,
        rhs,
        max_iters=max_iters,
        tol=tol,
        x0=x0,
    )


def spectrum_k_eff_hypergraph(
    *,
    k_self: Tensor,
    k_hedge: Tensor,
    hyperedge_index: Tensor,
    num_nodes: int,
    num_modes: int,
    hyperedge_weight: Tensor | None = None,
    tol: float = DEFAULT_MATRIX_FREE_SPECTRUM_TOL,
    ncv: int | None = None,
    seed: int = 0,
) -> MatrixFreeSpectrumResult:
    """Arnoldi leading-modulus spectrum of hypergraph ``K_eff``.

    Parameters
    ----------
    k_self, k_hedge : Tensor
        Factor matrices.
    hyperedge_index : Tensor
        Incidence.
    num_nodes : int
        Node count.
    num_modes : int
        Number of modes.
    hyperedge_weight : Tensor or None, optional
        Optional weights.
    tol, ncv, seed
        Arnoldi controls matching :func:`spectrum_k_eff_graph`.

    Returns
    -------
    MatrixFreeSpectrumResult
        Leading-modulus eigenvalues.
    """
    latent_dim = int(k_self.shape[0])
    dim = num_nodes * latent_dim
    resolved_ncv = _default_ncv(dim, num_modes) if ncv is None else int(ncv)

    def _apply_k(flat: Tensor) -> Tensor:
        """Internal helper: apply hypergraph ``K_eff``.

        Parameters
        ----------
        flat : Tensor
            Flat latents.

        Returns
        -------
        Tensor
            Flat image.
        """
        return apply_k_eff_hypergraph(
            flat,
            k_self=k_self,
            k_hedge=k_hedge,
            hyperedge_index=hyperedge_index,
            num_nodes=num_nodes,
            hyperedge_weight=hyperedge_weight,
        )

    return _arnoldi_eigs(
        _apply_k,
        dim,
        num_modes=num_modes,
        tol=tol,
        ncv=resolved_ncv,
        seed=seed,
        device=k_self.device,
        dtype=k_self.dtype if k_self.is_floating_point() else torch.float32,
    )
