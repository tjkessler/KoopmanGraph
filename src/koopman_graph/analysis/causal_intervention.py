"""Labeled synthetic SCM interventions (not Granger discovery).

:func:`teaching_three_node_scm` is a three-node acyclic linear Gaussian
fixture with a known edge ``0 → 1`` and an independent node ``2``.
:func:`recover_synthetic_interventional_edges` scores
``|E[X_j | do(X_i = v_hi)] - E[X_j | do(X_i = v_lo)]|`` on that
generator. The protocol is **synthetic** and **interventional** on
the fixture only.

:func:`~koopman_graph.analysis.granger_latent_influence` remains
**non-interventional**. This module does not change that helper and
does not claim observational causal discovery on field data.

This module must not import :mod:`koopman_graph.model`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

DEFAULT_ATE_THRESHOLD = 0.25
DEFAULT_INTERVENTION_HIGH = 1.0
DEFAULT_INTERVENTION_LOW = -1.0
DEFAULT_SCM_EDGE_WEIGHT = 0.8
DEFAULT_SCM_NOISE_SCALE = 0.05
DEFAULT_SCM_SAMPLES = 512
DEFAULT_SCM_SEED = 0
TEACHING_SCM_N_NODES = 3

__all__ = [
    "DEFAULT_ATE_THRESHOLD",
    "DEFAULT_INTERVENTION_HIGH",
    "DEFAULT_INTERVENTION_LOW",
    "DEFAULT_SCM_EDGE_WEIGHT",
    "DEFAULT_SCM_NOISE_SCALE",
    "DEFAULT_SCM_SAMPLES",
    "DEFAULT_SCM_SEED",
    "SyntheticInterventionReport",
    "SyntheticSCM",
    "TEACHING_SCM_N_NODES",
    "recover_synthetic_interventional_edges",
    "sample_synthetic_intervention",
    "sample_synthetic_observational",
    "teaching_three_node_scm",
]


def _require_square_weights(weights: Tensor) -> Tensor:
    """Cast and validate a square contemporaneous weight matrix.

    Parameters
    ----------
    weights : Tensor
        Candidate ``(N, N)`` edge weights. Entry ``(i, j)`` is the
        coefficient on edge ``i → j``.

    Returns
    -------
    Tensor
        ``float64`` clone on CPU.

    Raises
    ------
    ValueError
        If the rank or shape is invalid, the diagonal is nonzero, or
        an entry is non-finite.
    """
    if weights.ndim != 2 or int(weights.shape[0]) != int(weights.shape[1]):
        msg = f"weights must have shape (N, N), got {tuple(weights.shape)}"
        raise ValueError(msg)
    if int(weights.shape[0]) < 2:
        msg = f"synthetic SCM requires N >= 2, got N={int(weights.shape[0])}"
        raise ValueError(msg)
    table = weights.detach().to(dtype=torch.float64, device="cpu").clone()
    if not bool(torch.isfinite(table).all()):
        msg = "weights must be finite"
        raise ValueError(msg)
    if bool(torch.any(table.diag().abs() > 1e-12)):
        msg = "synthetic SCM weights must have a zero diagonal"
        raise ValueError(msg)
    return table


def _topological_order(weights: Tensor) -> tuple[int, ...]:
    """Kahn topological order for an acyclic weight matrix.

    Parameters
    ----------
    weights : Tensor
        Square ``(N, N)`` matrix with ``weights[i, j]`` = edge ``i → j``.

    Returns
    -------
    tuple of int
        A parent-before-child order.

    Raises
    ------
    ValueError
        If the directed graph contains a cycle.
    """
    n_nodes = int(weights.shape[0])
    remaining = {
        (int(source), int(target))
        for source in range(n_nodes)
        for target in range(n_nodes)
        if float(weights[source, target].abs()) > 1e-12
    }
    indegree = [0] * n_nodes
    for _source, target in remaining:
        indegree[target] += 1
    ready = [node for node, degree in enumerate(indegree) if degree == 0]
    order: list[int] = []
    while ready:
        node = ready.pop(0)
        order.append(node)
        outgoing = [edge for edge in remaining if edge[0] == node]
        for edge in outgoing:
            remaining.remove(edge)
            child = edge[1]
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if remaining:
        msg = "synthetic SCM weights must be acyclic"
        raise ValueError(msg)
    return tuple(order)


def _true_edges(weights: Tensor, *, atol: float = 1e-12) -> tuple[tuple[int, int], ...]:
    """Return directed edges with absolute weight above ``atol``.

    Parameters
    ----------
    weights : Tensor
        Square ``(N, N)`` matrix with ``weights[i, j]`` = edge ``i → j``.
    atol : float, optional
        Absolute cutoff for a zero coefficient.

    Returns
    -------
    tuple of tuple of int
        Sorted ``(source, target)`` pairs.
    """
    n_nodes = int(weights.shape[0])
    edges = [
        (source, target)
        for source in range(n_nodes)
        for target in range(n_nodes)
        if source != target and float(weights[source, target].abs()) > atol
    ]
    return tuple(sorted(edges))


def _draw_assignment(
    weights: Tensor,
    noise: Tensor,
    order: tuple[int, ...],
    *,
    intervene_source: int | None,
    intervene_value: float | None,
) -> Tensor:
    """Solve one contemporaneous assignment, optionally under a do-set.

    Parameters
    ----------
    weights : Tensor
        Square ``(N, N)`` matrix with ``weights[i, j]`` = edge ``i → j``.
    noise : Tensor
        Exogenous noise with shape ``(N,)``.
    order : tuple of int
        Topological order.
    intervene_source : int or None
        Node whose structural equation is replaced. ``None`` is
        observational.
    intervene_value : float or None
        Value assigned under ``do(X_i = v)``.

    Returns
    -------
    Tensor
        One draw with shape ``(N,)``.
    """
    n_nodes = int(weights.shape[0])
    assigned = torch.zeros(n_nodes, dtype=weights.dtype)
    for node in order:
        if intervene_source is not None and node == int(intervene_source):
            assigned[node] = float(intervene_value)
            continue
        parents = weights[:, node] * assigned
        assigned[node] = parents.sum() + noise[node]
    return assigned


def _sample_rows(
    scm: SyntheticSCM,
    n_samples: int,
    *,
    generator: torch.Generator,
    intervene_source: int | None,
    intervene_value: float | None,
) -> Tensor:
    """Draw independent rows from the observational or do-law.

    Parameters
    ----------
    scm : SyntheticSCM
        Labeled synthetic fixture.
    n_samples : int
        Number of independent draws.
    generator : torch.Generator
        CPU generator for the exogenous noise.
    intervene_source : int or None
        Intervened node, or ``None`` for the observational law.
    intervene_value : float or None
        do-value when ``intervene_source`` is set.

    Returns
    -------
    Tensor
        Table with shape ``(n_samples, N)``.

    Raises
    ------
    ValueError
        If ``n_samples`` or the intervention index is invalid.
    """
    if n_samples < 1:
        msg = f"n_samples must be >= 1, got {n_samples}"
        raise ValueError(msg)
    n_nodes = int(scm.weights.shape[0])
    if intervene_source is not None:
        index = int(intervene_source)
        if index < 0 or index >= n_nodes:
            msg = f"intervention source {index} is outside [0, {n_nodes})"
            raise ValueError(msg)
        if intervene_value is None:
            msg = "intervene_value is required when intervene_source is set"
            raise ValueError(msg)
    noise = scm.noise_scale * torch.randn(
        n_samples,
        n_nodes,
        dtype=scm.weights.dtype,
        generator=generator,
    )
    rows = [
        _draw_assignment(
            scm.weights,
            noise[row],
            scm.order,
            intervene_source=intervene_source,
            intervene_value=intervene_value,
        )
        for row in range(n_samples)
    ]
    return torch.stack(rows, dim=0)


@dataclass(frozen=True)
class SyntheticSCM:
    """Acyclic linear Gaussian teaching SCM (labeled synthetic).

    Attributes
    ----------
    weights : Tensor
        Shape ``(N, N)``. Entry ``(i, j)`` is the coefficient on the
        directed edge ``i → j``. The diagonal is zero.
    noise_scale : float
        Standard deviation of independent Gaussian exogenous noise.
    seed : int
        Default seed used by the sampling helpers.
    labeled_synthetic : bool
        Always ``True``. This fixture is not a field-data SCM.
    order : tuple of int
        Topological order filled at construction.
    """

    weights: Tensor
    noise_scale: float
    seed: int
    labeled_synthetic: bool = True
    order: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """Validate the labeled synthetic fixture.

        Raises
        ------
        ValueError
            If the weights, noise scale, seed, or synthetic flag are
            invalid.
        """
        table = _require_square_weights(self.weights)
        object.__setattr__(self, "weights", table)
        if float(self.noise_scale) <= 0.0:
            msg = f"noise_scale must be > 0, got {self.noise_scale}"
            raise ValueError(msg)
        object.__setattr__(self, "noise_scale", float(self.noise_scale))
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "order", _topological_order(table))
        if self.labeled_synthetic is not True:
            msg = "SyntheticSCM.labeled_synthetic must be True"
            raise ValueError(msg)

    @property
    def true_edges(self) -> tuple[tuple[int, int], ...]:
        """Directed edges with a nonzero coefficient.

        Returns
        -------
        tuple of tuple of int
            Sorted ``(source, target)`` pairs.
        """
        return _true_edges(self.weights)


@dataclass(frozen=True)
class SyntheticInterventionReport:
    """Recovered do-edges on a labeled synthetic SCM.

    Attributes
    ----------
    scores : Tensor
        Shape ``(N, N)``. Entry ``(i, j)`` is the absolute mean shift
        of node ``j`` under ``do(X_i = v_hi)`` versus
        ``do(X_i = v_lo)``. The diagonal is zero.
    recovered_edges : tuple of tuple of int
        Pairs with ``scores[i, j] > threshold``.
    true_edges : tuple of tuple of int
        Ground-truth edges of the synthetic fixture.
    threshold : float
        Absolute ATE cutoff used to form ``recovered_edges``.
    labeled_synthetic : bool
        Always ``True``. This report is not observational Granger.
    """

    scores: Tensor
    recovered_edges: tuple[tuple[int, int], ...]
    true_edges: tuple[tuple[int, int], ...]
    threshold: float
    labeled_synthetic: bool = True

    def __post_init__(self) -> None:
        """Validate the interventional recovery record.

        Raises
        ------
        ValueError
            If shapes, flags, or the threshold are invalid.
        """
        if self.scores.ndim != 2 or int(self.scores.shape[0]) != int(
            self.scores.shape[1]
        ):
            msg = f"scores must have shape (N, N), got {tuple(self.scores.shape)}"
            raise ValueError(msg)
        object.__setattr__(
            self,
            "scores",
            self.scores.detach().to(dtype=torch.float64, device="cpu").clone(),
        )
        object.__setattr__(self, "recovered_edges", tuple(sorted(self.recovered_edges)))
        object.__setattr__(self, "true_edges", tuple(sorted(self.true_edges)))
        if float(self.threshold) <= 0.0:
            msg = f"threshold must be > 0, got {self.threshold}"
            raise ValueError(msg)
        object.__setattr__(self, "threshold", float(self.threshold))
        if self.labeled_synthetic is not True:
            msg = "SyntheticInterventionReport.labeled_synthetic must be True"
            raise ValueError(msg)


def teaching_three_node_scm(
    *,
    seed: int = DEFAULT_SCM_SEED,
    edge_weight: float = DEFAULT_SCM_EDGE_WEIGHT,
    noise_scale: float = DEFAULT_SCM_NOISE_SCALE,
) -> SyntheticSCM:
    """Build the three-node teaching SCM with a known edge ``0 → 1``.

    Parameters
    ----------
    seed : int, optional
        Default sampling seed.
    edge_weight : float, optional
        Coefficient on ``0 → 1``. Must be nonzero.
    noise_scale : float, optional
        Exogenous Gaussian scale.

    Returns
    -------
    SyntheticSCM
        Labeled synthetic fixture. Node ``2`` is independent.

    Raises
    ------
    ValueError
        If ``edge_weight`` is zero.
    """
    if abs(float(edge_weight)) <= 1e-12:
        msg = "teaching SCM edge_weight must be nonzero"
        raise ValueError(msg)
    weights = torch.zeros(TEACHING_SCM_N_NODES, TEACHING_SCM_N_NODES)
    weights[0, 1] = float(edge_weight)
    table = _require_square_weights(weights)
    return SyntheticSCM(
        weights=table,
        noise_scale=float(noise_scale),
        seed=int(seed),
        labeled_synthetic=True,
    )


def sample_synthetic_observational(
    scm: SyntheticSCM,
    n_samples: int,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Draw observational samples from a labeled synthetic SCM.

    Parameters
    ----------
    scm : SyntheticSCM
        Labeled synthetic fixture.
    n_samples : int
        Number of independent draws.
    generator : torch.Generator or None, optional
        CPU generator. Default seeds from ``scm.seed``.

    Returns
    -------
    Tensor
        Table with shape ``(n_samples, N)``.
    """
    rng = torch.Generator()
    if generator is None:
        rng.manual_seed(int(scm.seed))
    else:
        rng = generator
    return _sample_rows(
        scm,
        n_samples,
        generator=rng,
        intervene_source=None,
        intervene_value=None,
    )


def sample_synthetic_intervention(
    scm: SyntheticSCM,
    *,
    source: int,
    value: float,
    n_samples: int,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Draw samples from ``do(X_source = value)`` on a synthetic SCM.

    Parameters
    ----------
    scm : SyntheticSCM
        Labeled synthetic fixture.
    source : int
        Intervened node. Its structural equation is replaced.
    value : float
        Assigned value.
    n_samples : int
        Number of independent draws.
    generator : torch.Generator or None, optional
        CPU generator. Default seeds from ``scm.seed + 1 + source``.

    Returns
    -------
    Tensor
        Table with shape ``(n_samples, N)``.
    """
    rng = torch.Generator()
    if generator is None:
        rng.manual_seed(int(scm.seed) + 1 + int(source))
    else:
        rng = generator
    return _sample_rows(
        scm,
        n_samples,
        generator=rng,
        intervene_source=int(source),
        intervene_value=float(value),
    )


def recover_synthetic_interventional_edges(
    scm: SyntheticSCM,
    *,
    n_samples: int = DEFAULT_SCM_SAMPLES,
    intervention_low: float = DEFAULT_INTERVENTION_LOW,
    intervention_high: float = DEFAULT_INTERVENTION_HIGH,
    threshold: float = DEFAULT_ATE_THRESHOLD,
    seed: int | None = None,
) -> SyntheticInterventionReport:
    """Recover directed edges from paired do-interventions on a toy SCM.

    For each source ``i`` the helper compares mean node features under
    ``do(X_i = intervention_high)`` and ``do(X_i = intervention_low)``.
    This is a labeled synthetic protocol, not
    :func:`~koopman_graph.analysis.granger_latent_influence`.

    Parameters
    ----------
    scm : SyntheticSCM
        Labeled synthetic fixture.
    n_samples : int, optional
        Independent draws per do-value.
    intervention_low, intervention_high : float, optional
        Pair of do-values. They must differ.
    threshold : float, optional
        Absolute mean-shift cutoff that marks a recovered edge.
    seed : int or None, optional
        Base seed for the paired interventions. Default is ``scm.seed``.

    Returns
    -------
    SyntheticInterventionReport
        Absolute mean-shift scores and the recovered edge set.

    Raises
    ------
    ValueError
        If the do-values coincide or ``threshold`` is not positive.
    """
    if abs(float(intervention_high) - float(intervention_low)) <= 1e-12:
        msg = "intervention_high and intervention_low must differ"
        raise ValueError(msg)
    if float(threshold) <= 0.0:
        msg = f"threshold must be > 0, got {threshold}"
        raise ValueError(msg)
    base_seed = int(scm.seed if seed is None else seed)
    n_nodes = int(scm.weights.shape[0])
    scores = torch.zeros(n_nodes, n_nodes, dtype=torch.float64)
    for source in range(n_nodes):
        low = sample_synthetic_intervention(
            scm,
            source=source,
            value=float(intervention_low),
            n_samples=n_samples,
            generator=torch.Generator().manual_seed(base_seed + 10 * source),
        )
        high = sample_synthetic_intervention(
            scm,
            source=source,
            value=float(intervention_high),
            n_samples=n_samples,
            generator=torch.Generator().manual_seed(base_seed + 10 * source + 1),
        )
        shift = (high.mean(dim=0) - low.mean(dim=0)).abs()
        scores[source] = shift
        scores[source, source] = 0.0
    recovered = tuple(
        (source, target)
        for source in range(n_nodes)
        for target in range(n_nodes)
        if source != target and float(scores[source, target]) > float(threshold)
    )
    return SyntheticInterventionReport(
        scores=scores,
        recovered_edges=recovered,
        true_edges=scm.true_edges,
        threshold=float(threshold),
        labeled_synthetic=True,
    )
