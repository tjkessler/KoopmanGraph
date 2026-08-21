"""Closed-form sparse identification of graph Koopman factors.

Fits shared :math:`K_{\\mathrm{self}}` / :math:`K_{\\mathrm{nbr}}` on frozen
node latents for the one-tap map

.. math::

    Z_{t+1}
    \\approx
    Z_t K_{\\mathrm{self}}^{\\top}
    +
    (\\hat A Z_t) K_{\\mathrm{nbr}}^{\\top}.

STLSQ (Brunton et al., 2016) or a proximal group-threshold iteration
selects support; an unpenalized least-squares refit follows. This is
**not** :func:`~koopman_graph.analysis.identify_sparse_dynamics`
(library :math:`\\Theta(z)`, not factor :math:`K`), **not**
:class:`~koopman_graph.losses.KoopmanSparsityLoss` (training penalty),
and **not** Pan et al. (2021) multi-task EDMD dictionary pruning.

This module must not import :mod:`koopman_graph.training` or
:mod:`koopman_graph.model`. Dual random-walk and polynomial
:math:`P>1` maps are out of scope.

References
----------
Brunton, S. L., Proctor, J. L. & Kutz, J. N. (2016). Discovering
governing equations from data by sparse identification of nonlinear
dynamical systems. *Proceedings of the National Academy of Sciences*,
113(15), 3932–3937. https://doi.org/10.1073/pnas.1517384113
(``Brunton2016SINDy``)

Pan, S., Arnold-Medabalimi, N. & Duraisamy, K. (2021). Sparsity-promoting
algorithms for the discovery of informative Koopman-invariant subspaces.
*Journal of Fluid Mechanics*, 917, A18. https://doi.org/10.1017/jfm.2021.271
(``Pan2021SparseSubspace``)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from koopman_graph.graph_utils.topology import (
    dense_random_walk_normalized_adjacency,
    dense_symmetric_normalized_adjacency,
)

__all__ = [
    "DEFAULT_GROUP_LASSO_MAX_ITER",
    "DEFAULT_STLSQ_MAX_ITER",
    "SparseFactorGroup",
    "SparseFactorMethod",
    "SparseFactorReport",
    "identify_sparse_graph_factors",
]

SparseFactorGroup = Literal["none", "self_nbr", "orbit"]
SparseFactorMethod = Literal["stlsq", "group_lasso"]
SparseFactorAdjacency = Literal["symmetric", "random_walk"]

DEFAULT_STLSQ_MAX_ITER = 10
DEFAULT_GROUP_LASSO_MAX_ITER = 200
_ISTA_REL_TOL = 1e-8
_GROUPS = frozenset({"none", "self_nbr", "orbit"})
_METHODS = frozenset({"stlsq", "group_lasso"})
_ADJACENCIES = frozenset({"symmetric", "random_walk"})


@dataclass(frozen=True)
class SparseFactorReport:
    """Sparse one-tap graph-factor fit on frozen encodings.

    ``K_self`` / ``K_nbr`` follow the package row convention
    ``Z_next = Z @ K_self.T + (Â Z) @ K_nbr.T``. Inactive entries are
    exactly zero after the unpenalized refit.

    Attributes
    ----------
    K_self, K_nbr : Tensor
        Identified factors with shape ``(d, d)``.
    active_mask_self, active_mask_nbr : Tensor
        Boolean masks with the same shape as the factors.
    residual : float
        Mean squared residual of the refit map on the stacked pairs.
    nnz : int
        Number of nonzero entries in both factors.
    n_samples : int
        Number of node×transition rows in the stacked design.
    method : {"stlsq", "group_lasso"}
        Support selector.
    group : {"none", "self_nbr", "orbit"}
        Grouping used for thresholding.
    threshold : float
        Absolute (STLSQ) or group-norm (group-lasso) cutoff.
    """

    K_self: Tensor
    K_nbr: Tensor
    active_mask_self: Tensor
    active_mask_nbr: Tensor
    residual: float
    nnz: int
    n_samples: int
    method: SparseFactorMethod
    group: SparseFactorGroup
    threshold: float

    def __post_init__(self) -> None:
        """Validate factor shapes, residual, and counts.

        Raises
        ------
        ValueError
            If shapes disagree, ``residual`` is non-finite, or counts
            are invalid.
        """
        if self.K_self.shape != self.K_nbr.shape:
            msg = (
                "K_self and K_nbr must share shape, "
                f"got {tuple(self.K_self.shape)} vs {tuple(self.K_nbr.shape)}"
            )
            raise ValueError(msg)
        if self.K_self.ndim != 2 or self.K_self.shape[0] != self.K_self.shape[1]:
            msg = f"K_self must be square 2-D, got shape {tuple(self.K_self.shape)}"
            raise ValueError(msg)
        if self.active_mask_self.shape != self.K_self.shape:
            msg = "active_mask_self must match K_self"
            raise ValueError(msg)
        if self.active_mask_nbr.shape != self.K_nbr.shape:
            msg = "active_mask_nbr must match K_nbr"
            raise ValueError(msg)
        if not _finite_float(self.residual) or self.residual < 0.0:
            msg = f"residual must be a finite non-negative float, got {self.residual}"
            raise ValueError(msg)
        if self.nnz < 0 or self.n_samples < 1:
            msg = (
                "nnz must be non-negative and n_samples >= 1, "
                f"got nnz={self.nnz}, n_samples={self.n_samples}"
            )
            raise ValueError(msg)
        if self.threshold < 0.0:
            msg = f"threshold must be non-negative, got {self.threshold}"
            raise ValueError(msg)


def _finite_float(value: float) -> bool:
    """Return whether ``value`` is a finite Python float.

    Parameters
    ----------
    value : float
        Scalar to test.

    Returns
    -------
    bool
        ``True`` when ``value`` is finite.
    """
    return value == value and value not in (float("inf"), float("-inf"))


def _assemble_adjacency(
    edge_index: Tensor,
    *,
    num_nodes: int,
    edge_weight: Tensor | None,
    dtype: torch.dtype,
    adjacency: SparseFactorAdjacency,
) -> Tensor:
    """Build dense :math:`\\hat A` matching the one-tap graph operator.

    Parameters
    ----------
    edge_index : Tensor
        COO edges with shape ``(2, E)``.
    num_nodes : int
        Node count ``N``.
    edge_weight : Tensor or None
        Optional non-negative weights with shape ``(E,)``.
    dtype : torch.dtype
        Floating dtype of the encodings.
    adjacency : {"symmetric", "random_walk"}
        Normalization.

    Returns
    -------
    Tensor
        Dense ``(N, N)`` shift.

    Raises
    ------
    ValueError
        If ``edge_index`` is not ``(2, E)`` or indices exceed ``N``.
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        msg = (
            f"edge_index must have shape (2, num_edges), got {tuple(edge_index.shape)}"
        )
        raise ValueError(msg)
    if int(edge_index.numel()) > 0:
        max_index = int(edge_index.max().item())
        min_index = int(edge_index.min().item())
        if min_index < 0 or max_index >= num_nodes:
            msg = (
                f"edge_index entries must lie in [0, {num_nodes}), "
                f"got min={min_index}, max={max_index}"
            )
            raise ValueError(msg)
    if adjacency == "symmetric":
        return dense_symmetric_normalized_adjacency(
            edge_index,
            num_nodes,
            edge_weight=edge_weight,
            dtype=dtype,
        )
    return dense_random_walk_normalized_adjacency(
        edge_index,
        num_nodes,
        edge_weight=edge_weight,
        dtype=dtype,
        direction="forward",
    )


def _stacked_design(
    z_pairs: Tensor,
    adjacency: Tensor,
) -> tuple[Tensor, Tensor]:
    """Stack consecutive pairs into ``X = [Z | ÂZ]`` and ``Y = Z_next``.

    Parameters
    ----------
    z_pairs : Tensor
        Encodings with shape ``(T, N, d)``.
    adjacency : Tensor
        Dense ``Â`` with shape ``(N, N)``.

    Returns
    -------
    tuple of Tensor
        ``X`` with shape ``((T-1) N, 2d)`` and ``Y`` with shape
        ``((T-1) N, d)``.
    """
    current = z_pairs[:-1]
    target = z_pairs[1:]
    shift = adjacency.to(device=current.device, dtype=current.dtype)
    neighbor = shift @ current
    left = current.reshape(-1, current.shape[-1])
    right = neighbor.reshape(-1, neighbor.shape[-1])
    design = torch.cat([left, right], dim=1)
    return design, target.reshape(-1, target.shape[-1])


def _group_masks(
    group: SparseFactorGroup,
    latent_dim: int,
    *,
    device: torch.device,
) -> tuple[Tensor, ...]:
    """Boolean masks over the stacked solution ``(2d, d)``.

    Parameters
    ----------
    group : {"none", "self_nbr", "orbit"}
        Grouping.
    latent_dim : int
        Factor width ``d``.
    device : torch.device
        Device of the encodings.

    Returns
    -------
    tuple of Tensor
        Masks with shape ``(2d, d)``.
    """
    rows = 2 * latent_dim
    cols = latent_dim
    if group == "none":
        masks = []
        for row in range(rows):
            for col in range(cols):
                mask = torch.zeros(rows, cols, dtype=torch.bool, device=device)
                mask[row, col] = True
                masks.append(mask)
        return tuple(masks)
    if group == "self_nbr":
        self_mask = torch.zeros(rows, cols, dtype=torch.bool, device=device)
        self_mask[:latent_dim] = True
        neighbor_mask = torch.zeros(rows, cols, dtype=torch.bool, device=device)
        neighbor_mask[latent_dim:] = True
        return (self_mask, neighbor_mask)
    masks = []
    for col in range(cols):
        mask = torch.zeros(rows, cols, dtype=torch.bool, device=device)
        mask[:, col] = True
        masks.append(mask)
    return tuple(masks)


def _threshold_groups(
    solution: Tensor,
    masks: tuple[Tensor, ...],
    threshold: float,
) -> Tensor:
    """Zero groups whose Euclidean norm is below ``threshold``.

    Parameters
    ----------
    solution : Tensor
        Stacked ``[K_self.T; K_nbr.T]`` with shape ``(2d, d)``.
    masks : tuple of Tensor
        Group membership masks.
    threshold : float
        Group-norm cutoff (absolute value when a group is one entry).

    Returns
    -------
    Tensor
        Thresholded copy of ``solution``.
    """
    out = solution.clone()
    for mask in masks:
        values = out[mask]
        if float(values.norm().item()) < threshold:
            out[mask] = 0
    return out


def _refit_on_support(
    design: Tensor,
    target: Tensor,
    support: Tensor,
) -> Tensor:
    """Unpenalized least squares on the remaining entries of ``S``.

    Parameters
    ----------
    design : Tensor
        Stacked ``X`` with shape ``(M, 2d)``.
    target : Tensor
        Stacked ``Y`` with shape ``(M, d)``.
    support : Tensor
        Boolean mask with shape ``(2d, d)``.

    Returns
    -------
    Tensor
        Refit ``S`` with inactive entries exactly zero.
    """
    latent_dim = target.shape[1]
    solution = torch.zeros(
        design.shape[1],
        latent_dim,
        dtype=design.dtype,
        device=design.device,
    )
    shared_rows = torch.equal(support, support[:, :1].expand_as(support))
    if shared_rows:
        active = support[:, 0]
        if not bool(active.any().item()):
            return solution
        fitted = torch.linalg.lstsq(design[:, active], target).solution
        solution[active] = fitted
        return solution
    for col in range(latent_dim):
        active = support[:, col]
        if not bool(active.any().item()):
            continue
        fitted = torch.linalg.lstsq(design[:, active], target[:, col]).solution
        solution[active, col] = fitted
    return solution


def _stlsq_factors(
    design: Tensor,
    target: Tensor,
    masks: tuple[Tensor, ...],
    *,
    threshold: float,
    max_iter: int,
) -> Tensor:
    """Sequentially thresholded least squares on stacked factor columns.

    Parameters
    ----------
    design, target : Tensor
        Stacked regression pair.
    masks : tuple of Tensor
        Group masks.
    threshold : float
        Group-norm cutoff.
    max_iter : int
        Maximum STLSQ iterations.

    Returns
    -------
    Tensor
        Sparse stacked solution.
    """
    solution = torch.linalg.lstsq(design, target).solution
    for _ in range(max_iter):
        previous = solution.clone()
        solution = _threshold_groups(solution, masks, threshold)
        support = solution.abs() > 0
        solution = _refit_on_support(design, target, support)
        solution = _threshold_groups(solution, masks, threshold)
        if torch.allclose(solution, previous):
            break
    return solution


def _group_soft_threshold(
    solution: Tensor,
    masks: tuple[Tensor, ...],
    tau: float,
) -> Tensor:
    """Proximal group soft-threshold with scale ``tau``.

    Parameters
    ----------
    solution : Tensor
        Current stacked map.
    masks : tuple of Tensor
        Group masks.
    tau : float
        ``step * lambda`` shrinkage.

    Returns
    -------
    Tensor
        Shrinkage of each group.
    """
    out = solution.clone()
    for mask in masks:
        values = out[mask]
        norm = float(values.norm().item())
        if norm == 0.0:
            continue
        scale = max(0.0, 1.0 - tau / norm)
        out[mask] = values * scale
    return out


def _ista_group_lasso(
    design: Tensor,
    target: Tensor,
    masks: tuple[Tensor, ...],
    *,
    threshold: float,
    max_iter: int,
) -> Tensor:
    """ISTA group-lasso followed by a hard group cutoff and unpenalized refit.

    Parameters
    ----------
    design, target : Tensor
        Stacked regression pair.
    masks : tuple of Tensor
        Group masks.
    threshold : float
        Penalty ``λ`` and post-ISTA group-norm cutoff.
    max_iter : int
        Maximum ISTA steps.

    Returns
    -------
    Tensor
        Sparse stacked solution after refit.
    """
    gram = design.T @ design
    lipschitz = torch.linalg.eigvalsh(gram).amax()
    step = 1.0 / float(lipschitz.clamp(min=1e-12).item())
    solution = torch.linalg.lstsq(design, target).solution
    for _ in range(max_iter):
        previous = solution.clone()
        residual = design @ solution - target
        solution = solution - step * (design.T @ residual)
        solution = _group_soft_threshold(solution, masks, step * threshold)
        delta = (solution - previous).norm()
        scale = previous.norm().clamp(min=1e-12)
        if float((delta / scale).item()) < _ISTA_REL_TOL:
            break
    solution = _threshold_groups(solution, masks, threshold)
    support = solution.abs() > 0
    solution = _refit_on_support(design, target, support)
    return _threshold_groups(solution, masks, threshold)


def identify_sparse_graph_factors(
    z_pairs: Tensor,
    edge_index: Tensor,
    *,
    group: SparseFactorGroup = "self_nbr",
    method: SparseFactorMethod = "stlsq",
    threshold: float,
    edge_weight: Tensor | None = None,
    adjacency: SparseFactorAdjacency = "symmetric",
    max_iter: int | None = None,
) -> SparseFactorReport:
    """Identify sparse :math:`K_{\\mathrm{self}}` / :math:`K_{\\mathrm{nbr}}`.

    Distinct from :func:`~koopman_graph.analysis.identify_sparse_dynamics`
    and :class:`~koopman_graph.losses.KoopmanSparsityLoss`. Existing L1
    training and latent SINDy paths still ship. This is not Pan et al.
    (2021) multi-task EDMD dictionary pruning.

    Parameters
    ----------
    z_pairs : Tensor
        Consecutive frozen encodings with shape ``(T, N, d)``, ``T >= 2``.
    edge_index : Tensor
        COO topology with shape ``(2, E)``.
    group : {"none", "self_nbr", "orbit"}, optional
        ``"none"`` is elementwise; ``"self_nbr"`` treats each factor as
        one group; ``"orbit"`` groups latent **rows** (row ``i`` of
        both factors). Default is ``"self_nbr"``.
    method : {"stlsq", "group_lasso"}, optional
        ``"stlsq"`` is sequentially thresholded least squares.
        ``"group_lasso"`` is ISTA group soft-threshold (teaching-thin),
        then the same unpenalized refit. Default is ``"stlsq"``.
    threshold : float
        Non-negative cutoff. STLSQ uses group Euclidean norms (absolute
        value for ``group="none"``). Group-lasso uses the same value as
        ``λ`` and as the post-ISTA cutoff.
    edge_weight : Tensor or None, optional
        Optional edge weights with shape ``(E,)``. Default is ``None``.
    adjacency : {"symmetric", "random_walk"}, optional
        Shift used to form ``Â Z``. Dual random-walk is refused.
        Default is ``"symmetric"``.
    max_iter : int or None, optional
        Iteration cap. Default is ``10`` for STLSQ and ``200`` for
        group-lasso.

    Returns
    -------
    SparseFactorReport
        Factors, masks, residual, and metadata.

    Raises
    ------
    ValueError
        If shapes, ``group``, ``method``, ``adjacency``, or
        ``threshold`` are invalid, or ``T < 2``.
    """
    if group not in _GROUPS:
        msg = f"group must be one of {sorted(_GROUPS)}, got {group!r}"
        raise ValueError(msg)
    if method not in _METHODS:
        msg = f"method must be one of {sorted(_METHODS)}, got {method!r}"
        raise ValueError(msg)
    if adjacency not in _ADJACENCIES:
        msg = (
            "adjacency must be 'symmetric' or 'random_walk' "
            f"(dual random-walk is out of scope), got {adjacency!r}"
        )
        raise ValueError(msg)
    if threshold < 0.0 or not _finite_float(float(threshold)):
        msg = f"threshold must be a finite non-negative float, got {threshold}"
        raise ValueError(msg)
    if z_pairs.ndim != 3:
        msg = f"z_pairs must have shape (T, N, d), got {tuple(z_pairs.shape)}"
        raise ValueError(msg)
    num_times, num_nodes, latent_dim = (int(dim) for dim in z_pairs.shape)
    if num_times < 2:
        msg = f"z_pairs requires T >= 2 consecutive encodings, got T={num_times}"
        raise ValueError(msg)
    if num_nodes < 1 or latent_dim < 1:
        msg = f"z_pairs requires N >= 1 and d >= 1, got N={num_nodes}, d={latent_dim}"
        raise ValueError(msg)
    if not z_pairs.is_floating_point():
        msg = f"z_pairs must be floating-point, got {z_pairs.dtype}"
        raise ValueError(msg)
    if max_iter is None:
        resolved_iter = (
            DEFAULT_STLSQ_MAX_ITER
            if method == "stlsq"
            else DEFAULT_GROUP_LASSO_MAX_ITER
        )
    else:
        resolved_iter = int(max_iter)
    if resolved_iter < 1:
        msg = f"max_iter must be >= 1, got {resolved_iter}"
        raise ValueError(msg)

    shift = _assemble_adjacency(
        edge_index,
        num_nodes=num_nodes,
        edge_weight=edge_weight,
        dtype=z_pairs.dtype,
        adjacency=adjacency,
    )
    design, target = _stacked_design(z_pairs, shift)
    masks = _group_masks(group, latent_dim, device=z_pairs.device)
    if method == "stlsq":
        stacked = _stlsq_factors(
            design,
            target,
            masks,
            threshold=float(threshold),
            max_iter=resolved_iter,
        )
    else:
        stacked = _ista_group_lasso(
            design,
            target,
            masks,
            threshold=float(threshold),
            max_iter=resolved_iter,
        )
    k_self = stacked[:latent_dim].T.contiguous()
    k_nbr = stacked[latent_dim:].T.contiguous()
    residual = float((design @ stacked - target).square().mean().item())
    mask_self = k_self.abs() > 0
    mask_nbr = k_nbr.abs() > 0
    nnz = int(mask_self.sum().item() + mask_nbr.sum().item())
    return SparseFactorReport(
        K_self=k_self,
        K_nbr=k_nbr,
        active_mask_self=mask_self,
        active_mask_nbr=mask_nbr,
        residual=residual,
        nnz=nnz,
        n_samples=int(design.shape[0]),
        method=method,
        group=group,
        threshold=float(threshold),
    )
