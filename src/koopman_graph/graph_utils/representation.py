"""Exact-automorphism isotypic decomposition of the permutation representation.

Computes isotypic projectors for the action of :math:`\\mathrm{Aut}(G)` on
:math:`\\mathbb{R}^{N}` via the orbital (commutant) algebra of the exact
automorphism group (pynauty). This is **representation theory**, not
Weisfeiler–Lehman orbit coloring: WL-labeled methods are refused (R9).

MVP ceiling: ``N <= 20``. Large symmetric groups (complete / empty graphs)
use the closed-form trivial ⊕ standard decomposition without enumerating
``N!`` elements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

IsotypicMethod = Literal["automorphism"]

MAX_ISOTYPIC_NODES = 20
_MAX_ENUMERATED_GROUP_ORDER = 100_000
_CLUSTER_ATOL = 1e-7
_PROJ_ATOL = 1e-6

_PYNAUTY_INSTALL_HINT = (
    "exact automorphism isotypic decomposition requires pynauty. "
    "Install pynauty separately (the [symmetry] extra provides networkx "
    "for WL orbits only and is not sufficient for isotypic projectors)"
)


@dataclass(frozen=True)
class IsotypicDecomposition:
    """Isotypic projectors for the permutation representation of ``Aut(G)``.

    Attributes
    ----------
    projectors : tuple of Tensor
        Symmetric ``(num_nodes, num_nodes)`` projectors, one per isotypic
        component. Each satisfies ``P @ P ≈ P``; distinct projectors are
        mutually orthogonal and sum to ``I``.
    dimensions : tuple of int
        ``trace(P_k)`` (component dimension) for each projector.
    multiplicities : tuple of int
        Reported as the component dimensions for this MVP (the permutation
        representation on vertices is typically multiplicity-free on the
        textbook fixtures; full irrep multiplicity recovery is deferred).
    method : str
        Decomposition method (``\"automorphism\"``).
    num_nodes : int
        Graph order ``N``.
    group_order : int
        Order of ``Aut(G)`` used to build the projectors.
    """

    projectors: tuple[Tensor, ...]
    dimensions: tuple[int, ...]
    multiplicities: tuple[int, ...]
    method: str
    num_nodes: int
    group_order: int


def compute_isotypic_decomposition(
    edge_index: Tensor,
    num_nodes: int,
    *,
    method: str = "automorphism",
) -> IsotypicDecomposition:
    """Compute isotypic projectors from the exact automorphism group.

    Parameters
    ----------
    edge_index : Tensor
        COO edges ``(2, E)``. Undirected graphs may list one or both
        directions; self-loops are ignored.
    num_nodes : int
        Node count ``N`` (MVP requires ``1 <= N <= 20``).
    method : str, optional
        Must be ``\"automorphism\"``. Values such as ``\"wl\"`` raise
        ``ValueError`` — WL orbit refinement is not an isotypic
        decomposition.

    Returns
    -------
    IsotypicDecomposition
        Projectors, dimensions, and group metadata.

    Raises
    ------
    ValueError
        If ``method`` is not exact automorphism, ``N`` exceeds the MVP
        ceiling, or inputs are invalid.
    ImportError
        If ``pynauty`` is not installed.
    """
    if method != "automorphism":
        msg = (
            f"method={method!r} is refused for isotypic decomposition. "
            f"Weisfeiler–Lehman / approximate orbit coloring is not "
            f"representation theory; use method='automorphism' with pynauty"
        )
        raise ValueError(msg)
    if num_nodes < 1:
        msg = f"num_nodes must be positive, got {num_nodes}"
        raise ValueError(msg)
    if num_nodes > MAX_ISOTYPIC_NODES:
        msg = (
            f"isotypic decomposition MVP supports num_nodes <= "
            f"{MAX_ISOTYPIC_NODES}, got {num_nodes}"
        )
        raise ValueError(msg)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        msg = f"edge_index must have shape (2, E), got {tuple(edge_index.shape)}"
        raise ValueError(msg)

    if _is_symmetric_group_graph(edge_index, num_nodes):
        return _symmetric_group_decomposition(num_nodes)

    generators, group_order = _automorphism_generators(edge_index, num_nodes)
    if group_order > _MAX_ENUMERATED_GROUP_ORDER:
        msg = (
            f"automorphism group order {group_order} exceeds the "
            f"enumeration ceiling {_MAX_ENUMERATED_GROUP_ORDER} for "
            f"isotypic projectors at N={num_nodes}"
        )
        raise ValueError(msg)

    group = _enumerate_group(generators, num_nodes, max_order=group_order)
    if len(group) != group_order:
        msg = f"enumerated |Aut(G)|={len(group)} != reported order {group_order}"
        raise RuntimeError(msg)

    projectors = _projectors_from_orbital_algebra(group, num_nodes)
    return _pack_decomposition(
        projectors,
        num_nodes=num_nodes,
        group_order=group_order,
    )


def _require_pynauty():
    """Import pynauty or raise a guided ``ImportError``.

    Returns
    -------
    module
        Imported ``pynauty`` module.

    Raises
    ------
    ImportError
        If ``pynauty`` is not installed.
    """
    try:
        import pynauty
    except ImportError as exc:  # pragma: no cover - exercised via mock
        raise ImportError(_PYNAUTY_INSTALL_HINT) from exc
    return pynauty


def _automorphism_generators(
    edge_index: Tensor,
    num_nodes: int,
) -> tuple[list[tuple[int, ...]], int]:
    """Return pynauty generators and the automorphism group order.

    Parameters
    ----------
    edge_index
        See signature.
    num_nodes
        See signature.

    Returns
    -------
        See signature."""
    pynauty = _require_pynauty()
    adjacency = _adjacency_dict(edge_index, num_nodes)
    graph = pynauty.Graph(
        number_of_vertices=num_nodes,
        directed=False,
        adjacency_dict=adjacency,
    )
    # pynauty exposes ``autgrp`` (nauty); some docs historically said ``autgroup``.
    aut_fn = getattr(pynauty, "autgrp", None) or getattr(pynauty, "autgroup", None)
    if aut_fn is None:  # pragma: no cover - defensive
        msg = "pynauty.autgrp is unavailable"
        raise ImportError(msg)
    generators, grpsize1, grpsize2, _orbits, _numorbits = aut_fn(graph)
    group_order = int(round(float(grpsize1) * (10.0 ** float(grpsize2))))
    if group_order < 1:
        msg = f"pynauty reported non-positive group order {group_order}"
        raise RuntimeError(msg)
    normalized = [tuple(int(v) for v in gen) for gen in generators]
    return normalized, group_order


def _adjacency_dict(edge_index: Tensor, num_nodes: int) -> dict[int, list[int]]:
    """Build an undirected adjacency dict for pynauty.

    Parameters
    ----------
    edge_index
        See signature.
    num_nodes
        See signature.

    Returns
    -------
        See signature."""
    adjacency: dict[int, list[int]] = {i: [] for i in range(num_nodes)}
    if edge_index.numel() == 0:
        return adjacency
    for src, dst in edge_index.t().tolist():
        src_i, dst_i = int(src), int(dst)
        if src_i == dst_i:
            continue
        if src_i < 0 or src_i >= num_nodes or dst_i < 0 or dst_i >= num_nodes:
            msg = (
                f"edge ({src_i}, {dst_i}) outside [0, {num_nodes - 1}] "
                f"for num_nodes={num_nodes}"
            )
            raise ValueError(msg)
        if dst_i not in adjacency[src_i]:
            adjacency[src_i].append(dst_i)
        if src_i not in adjacency[dst_i]:
            adjacency[dst_i].append(src_i)
    return adjacency


def _is_symmetric_group_graph(edge_index: Tensor, num_nodes: int) -> bool:
    """Return True for empty or complete undirected graphs (``Aut = S_N``).

    Parameters
    ----------
    edge_index
        See signature.
    num_nodes
        See signature.

    Returns
    -------
        See signature."""
    if num_nodes <= 1:
        return True
    undirected: set[tuple[int, int]] = set()
    if edge_index.numel() > 0:
        for src, dst in edge_index.t().tolist():
            src_i, dst_i = int(src), int(dst)
            if src_i == dst_i:
                continue
            undirected.add((min(src_i, dst_i), max(src_i, dst_i)))
    expected = num_nodes * (num_nodes - 1) // 2
    return len(undirected) in {0, expected}


def _symmetric_group_decomposition(num_nodes: int) -> IsotypicDecomposition:
    """Closed-form trivial ⊕ standard projectors for ``S_N``.

    Parameters
    ----------
    num_nodes
        See signature.

    Returns
    -------
        See signature."""
    eye = torch.eye(num_nodes, dtype=torch.float64)
    trivial = torch.full(
        (num_nodes, num_nodes),
        1.0 / float(num_nodes),
        dtype=torch.float64,
    )
    standard = eye - trivial
    projectors = [trivial, standard] if num_nodes > 1 else [trivial]
    return _pack_decomposition(
        projectors,
        num_nodes=num_nodes,
        group_order=int(math.factorial(num_nodes)),
    )


def _pack_decomposition(
    projectors: list[Tensor],
    *,
    num_nodes: int,
    group_order: int,
) -> IsotypicDecomposition:
    """Sort projectors by descending dimension and build the result object.

    Parameters
    ----------
    projectors
        See signature.
    num_nodes
        See signature.
    group_order
        See signature.

    Returns
    -------
        See signature."""
    ordered = sorted(
        projectors,
        key=lambda p: (
            -int(round(float(torch.trace(p).item()))),
            float(p.reshape(-1)[0].item()),
        ),
    )
    dimensions = tuple(int(round(float(torch.trace(p).item()))) for p in ordered)
    return IsotypicDecomposition(
        projectors=tuple(ordered),
        dimensions=dimensions,
        multiplicities=dimensions,
        method="automorphism",
        num_nodes=int(num_nodes),
        group_order=int(group_order),
    )


def _enumerate_group(
    generators: list[tuple[int, ...]],
    num_nodes: int,
    *,
    max_order: int,
) -> list[tuple[int, ...]]:
    """Enumerate the group generated by permutations (image tuples).

    Parameters
    ----------
    generators
        See signature.
    num_nodes
        See signature.
    max_order
        See signature.

    Returns
    -------
        See signature."""
    identity = tuple(range(num_nodes))
    if not generators:
        return [identity]
    seen: set[tuple[int, ...]] = {identity}
    queue: list[tuple[int, ...]] = [identity]
    while queue:
        current = queue.pop()
        for gen in generators:
            nxt = tuple(gen[current[i]] for i in range(num_nodes))
            if nxt in seen:
                continue
            seen.add(nxt)
            if len(seen) > max_order:
                msg = (
                    f"group enumeration exceeded max_order={max_order} at N={num_nodes}"
                )
                raise ValueError(msg)
            queue.append(nxt)
    return list(seen)


def _projectors_from_orbital_algebra(
    group: list[tuple[int, ...]],
    num_nodes: int,
) -> list[Tensor]:
    """Build isotypic projectors from a generic commutant element.

    Parameters
    ----------
    group
        See signature.
    num_nodes
        See signature.

    Returns
    -------
        See signature."""
    orbitals = _pair_orbitals(group, num_nodes)
    matrices = [_orbital_matrix(orbit, num_nodes) for orbit in orbitals]
    # Deterministic generic combination (avoids accidental eigenvalue ties).
    combo = torch.zeros(num_nodes, num_nodes, dtype=torch.float64)
    for index, matrix in enumerate(matrices):
        combo = combo + math.sqrt(2.0 + index) * matrix
    combo = 0.5 * (combo + combo.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(combo)
    scale = max(1.0, float(combo.abs().max().item()))
    atol = _CLUSTER_ATOL * scale
    clusters = _cluster_eigenvalues(eigenvalues, atol=atol)
    projectors: list[Tensor] = []
    for start, stop in clusters:
        basis = eigenvectors[:, start:stop]
        projector = basis @ basis.mT
        projector = 0.5 * (projector + projector.T)
        projectors.append(projector)
    _validate_projectors(projectors, num_nodes)
    return projectors


def _pair_orbitals(
    group: list[tuple[int, ...]],
    num_nodes: int,
) -> list[frozenset[tuple[int, int]]]:
    """Orbits of ``Aut(G)`` on ordered pairs ``(i, j)``.

    Parameters
    ----------
    group
        See signature.
    num_nodes
        See signature.

    Returns
    -------
        See signature."""
    seen: set[tuple[int, int]] = set()
    orbitals: list[frozenset[tuple[int, int]]] = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if (i, j) in seen:
                continue
            orbit = {(g[i], g[j]) for g in group}
            seen.update(orbit)
            orbitals.append(frozenset(orbit))
    return orbitals


def _orbital_matrix(
    orbit: frozenset[tuple[int, int]],
    num_nodes: int,
) -> Tensor:
    """Symmetric matrix indicator of an ordered-pair orbital.

    Parameters
    ----------
    orbit
        See signature.
    num_nodes
        See signature.

    Returns
    -------
        See signature."""
    matrix = torch.zeros(num_nodes, num_nodes, dtype=torch.float64)
    for i, j in orbit:
        matrix[i, j] = 1.0
    return 0.5 * (matrix + matrix.T)


def _cluster_eigenvalues(
    eigenvalues: Tensor,
    *,
    atol: float,
) -> list[tuple[int, int]]:
    """Return half-open index ranges of numerically equal eigenvalues.

    Parameters
    ----------
    eigenvalues
        See signature.
    atol
        See signature.

    Returns
    -------
        See signature."""
    num = int(eigenvalues.numel())
    if num == 0:
        return []
    clusters: list[tuple[int, int]] = []
    start = 0
    for index in range(1, num + 1):
        if index == num or abs(float(eigenvalues[index] - eigenvalues[start])) > atol:
            clusters.append((start, index))
            start = index
    return clusters


def _validate_projectors(projectors: list[Tensor], num_nodes: int) -> None:
    """Check idempotence, mutual orthogonality, and partition of identity.

    Parameters
    ----------
    projectors
        See signature.
    num_nodes
        See signature."""
    eye = torch.eye(num_nodes, dtype=torch.float64)
    total = torch.zeros_like(eye)
    for index, projector in enumerate(projectors):
        idem = torch.linalg.norm(projector @ projector - projector)
        if float(idem) > _PROJ_ATOL:
            msg = f"projector {index} failed idempotence: ‖P² - P‖_F={float(idem)}"
            raise RuntimeError(msg)
        total = total + projector
    for i, left in enumerate(projectors):
        for j, right in enumerate(projectors):
            if i >= j:
                continue
            orth = torch.linalg.norm(left @ right)
            if float(orth) > _PROJ_ATOL:
                msg = (
                    f"projectors {i} and {j} failed orthogonality: "
                    f"‖P_i P_j‖_F={float(orth)}"
                )
                raise RuntimeError(msg)
    completeness = torch.linalg.norm(total - eye)
    if float(completeness) > _PROJ_ATOL:
        msg = f"projectors do not sum to identity: ‖ΣP - I‖_F={float(completeness)}"
        raise RuntimeError(msg)
