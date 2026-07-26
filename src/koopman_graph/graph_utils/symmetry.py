"""Node-orbit partitions for symmetry-adapted networked operators.

Orbit finding uses the ``[symmetry]`` optional extra (``networkx``) for an
approximate Weisfeiler–Lehman color refinement, or soft-optional ``pynauty``
for exact automorphism orbits. When neither backend is available,
:func:`node_orbit_partition` returns the identity partition and warns.

Symmetry is an inductive bias: approximate orbits can mis-tie operator blocks.
An explicit ``orbit_partition`` always wins over auto-detection.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Literal

import torch
from torch import Tensor

OrbitMethod = Literal["auto", "exact"]
OrbitPartition = tuple[tuple[int, ...], ...]


def validate_orbit_partition(
    partition: Sequence[Sequence[int]],
    num_nodes: int,
) -> OrbitPartition:
    """Validate and normalize a node-orbit partition.

    Parameters
    ----------
    partition : sequence of sequence of int
        Candidate orbits. Must cover ``{0, …, N-1}`` exactly once.
    num_nodes : int
        Expected node count ``N``.

    Returns
    -------
    tuple of tuple of int
        Frozen orbits sorted by minimum node index; nodes within each orbit
        sorted ascending.

    Raises
    ------
    ValueError
        If the partition is empty, has out-of-range indices, or is not a
        partition of ``range(num_nodes)``.
    """
    if num_nodes < 1:
        msg = f"num_nodes must be positive, got {num_nodes}"
        raise ValueError(msg)
    if not partition:
        msg = "orbit_partition must contain at least one orbit"
        raise ValueError(msg)

    seen: set[int] = set()
    normalized: list[tuple[int, ...]] = []
    for orbit in partition:
        if not orbit:
            msg = "orbit_partition orbits must be non-empty"
            raise ValueError(msg)
        nodes = tuple(sorted(int(n) for n in orbit))
        for node in nodes:
            if node < 0 or node >= num_nodes:
                msg = (
                    f"orbit node {node} outside [0, {num_nodes - 1}] "
                    f"for num_nodes={num_nodes}"
                )
                raise ValueError(msg)
            if node in seen:
                msg = f"orbit_partition repeats node {node}"
                raise ValueError(msg)
            seen.add(node)
        normalized.append(nodes)

    if seen != set(range(num_nodes)):
        missing = sorted(set(range(num_nodes)) - seen)
        msg = f"orbit_partition missing nodes {missing}"
        raise ValueError(msg)

    normalized.sort(key=lambda orbit: orbit[0])
    return tuple(normalized)


def identity_orbit_partition(num_nodes: int) -> OrbitPartition:
    """Return the singleton partition ``((0,), (1,), …, (N-1,))``.

    Parameters
    ----------

    num_nodes : int
        See the function signature / summary for ``num_nodes``.

    Returns
    -------

    OrbitPartition
        See summary line.

    Raises
    ------

    ValueError
        Raised when inputs are invalid."""
    if num_nodes < 1:
        msg = f"num_nodes must be positive, got {num_nodes}"
        raise ValueError(msg)
    return tuple((i,) for i in range(num_nodes))


def node_orbit_index(partition: OrbitPartition, num_nodes: int) -> Tensor:
    """Map each node to its orbit index in ``partition``.

    Parameters
    ----------

    partition : OrbitPartition
        See the function signature / summary for ``partition``.
    num_nodes : int
        See the function signature / summary for ``num_nodes``.

    Returns
    -------

    Tensor
        See summary line."""
    validated = validate_orbit_partition(partition, num_nodes)
    mapping = torch.empty(num_nodes, dtype=torch.long)
    for orbit_id, orbit in enumerate(validated):
        for node in orbit:
            mapping[node] = orbit_id
    return mapping


def hyperedge_two_section(
    hyperedge_index: Tensor,
    num_nodes: int,
) -> Tensor:
    """Build an undirected 2-section edge index from hyperedge incidence.

    Parameters
    ----------

    hyperedge_index : Tensor
        See the function signature / summary for ``hyperedge_index``.
    num_nodes : int
        See the function signature / summary for ``num_nodes``.

    Returns
    -------

    Tensor
        See summary line.

    Raises
    ------

    ValueError
        Raised when inputs are invalid.

    Notes
    -----

    Nodes that co-appear in any hyperedge become pairwise undirected edges
    (both directions). Used for ``auto_orbits`` on hypergraph operators."""
    if hyperedge_index.ndim != 2 or hyperedge_index.shape[0] != 2:
        msg = (
            "hyperedge_index must have shape (2, nnz), "
            f"got {tuple(hyperedge_index.shape)}"
        )
        raise ValueError(msg)
    if num_nodes < 1:
        msg = f"num_nodes must be positive, got {num_nodes}"
        raise ValueError(msg)

    nodes = hyperedge_index[0].tolist()
    hedges = hyperedge_index[1].tolist()
    members: dict[int, list[int]] = {}
    for node, hedge in zip(nodes, hedges, strict=True):
        members.setdefault(int(hedge), []).append(int(node))

    undirected: set[tuple[int, int]] = set()
    for group in members.values():
        unique = sorted(set(group))
        for i, src in enumerate(unique):
            for dst in unique[i + 1 :]:
                undirected.add((src, dst))
                undirected.add((dst, src))
    if not undirected:
        return torch.empty((2, 0), dtype=torch.long)
    pairs = sorted(undirected)
    return torch.tensor(pairs, dtype=torch.long).t().contiguous()


def _edge_index_to_networkx(edge_index: Tensor, num_nodes: int):
    """Build an undirected ``networkx.Graph`` (call-site import).

    Parameters
    ----------

    edge_index : Tensor
        See the function signature / summary for ``edge_index``.
    num_nodes : int
        See the function signature / summary for ``num_nodes``.

    Returns
    -------

    object
        See summary line.

    Raises
    ------

    ImportError
        Raised when inputs are invalid."""
    try:
        import networkx as nx
    except ImportError as exc:
        msg = (
            "Orbit computation requires the [symmetry] extra "
            "(networkx). Install with: pip install 'koopman-graph[symmetry]'"
        )
        raise ImportError(msg) from exc

    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    if edge_index.numel() == 0:
        return graph
    edges = edge_index.t().tolist()
    for src, dst in edges:
        if src == dst:
            continue
        graph.add_edge(int(src), int(dst))
    return graph


def _orbits_from_wl(edge_index: Tensor, num_nodes: int) -> OrbitPartition:
    """Approximate orbits via Weisfeiler–Lehman color refinement (networkx).

    Parameters
    ----------

    edge_index : Tensor
        See the function signature / summary for ``edge_index``.
    num_nodes : int
        See the function signature / summary for ``num_nodes``.

    Returns
    -------

    OrbitPartition
        See summary line."""
    graph = _edge_index_to_networkx(edge_index, num_nodes)
    colors = {node: graph.degree(node) for node in graph.nodes()}
    for _ in range(max(num_nodes, 1)):
        refined: dict[int, tuple] = {}
        for node in graph.nodes():
            neighbor_colors = tuple(
                sorted(colors[nbr] for nbr in graph.neighbors(node))
            )
            refined[node] = (colors[node], neighbor_colors)
        # Relabel colors by sorted unique signatures for stable IDs.
        signatures = sorted(set(refined.values()))
        mapping = {sig: idx for idx, sig in enumerate(signatures)}
        new_colors = {node: mapping[refined[node]] for node in graph.nodes()}
        if new_colors == colors:
            break
        colors = new_colors

    buckets: dict[int, list[int]] = {}
    for node, color in colors.items():
        buckets.setdefault(color, []).append(int(node))
    partition = [tuple(sorted(nodes)) for nodes in buckets.values()]
    return validate_orbit_partition(partition, num_nodes)


def _orbits_from_pynauty(edge_index: Tensor, num_nodes: int) -> OrbitPartition:
    """Exact automorphism orbits via ``pynauty`` (soft-optional).

    Parameters
    ----------

    edge_index : Tensor
        See the function signature / summary for ``edge_index``.
    num_nodes : int
        See the function signature / summary for ``num_nodes``.

    Returns
    -------

    OrbitPartition
        See summary line.

    Raises
    ------

    ImportError
        Raised when inputs are invalid."""
    try:
        import pynauty
    except ImportError as exc:
        msg = (
            "method='exact' requires pynauty. Install pynauty separately "
            "(the [symmetry] extra provides networkx for method='auto')"
        )
        raise ImportError(msg) from exc

    adjacency: dict[int, list[int]] = {i: [] for i in range(num_nodes)}
    if edge_index.numel() > 0:
        for src, dst in edge_index.t().tolist():
            src_i, dst_i = int(src), int(dst)
            if src_i == dst_i:
                continue
            if dst_i not in adjacency[src_i]:
                adjacency[src_i].append(dst_i)
            if src_i not in adjacency[dst_i]:
                adjacency[dst_i].append(src_i)

    graph = pynauty.Graph(
        number_of_vertices=num_nodes,
        directed=False,
        adjacency_dict=adjacency,
    )
    _generators, _grp1, _grp2, orbit_of_vertex, _numorbits = pynauty.autgroup(graph)
    buckets: dict[int, list[int]] = {}
    for node, orbit_id in enumerate(orbit_of_vertex):
        buckets.setdefault(int(orbit_id), []).append(node)
    partition = [tuple(sorted(nodes)) for nodes in buckets.values()]
    return validate_orbit_partition(partition, num_nodes)


def node_orbit_partition(
    edge_index: Tensor,
    num_nodes: int,
    method: OrbitMethod = "auto",
) -> OrbitPartition:
    """Compute a node-orbit partition for an undirected graph.

    Parameters
    ----------
    edge_index : Tensor
        Edge index ``(2, E)`` (undirected graphs may list both directions).
    num_nodes : int
        Node count ``N``.
    method : {"auto", "exact"}, optional
        ``"exact"`` requires ``pynauty``. ``"auto"`` prefers ``pynauty`` when
        installed, otherwise Weisfeiler–Lehman via ``networkx``, otherwise the
        identity partition with a warning.

    Returns
    -------
    tuple of tuple of int
        Orbit partition of ``{0, …, N-1}``.
    """
    if method not in {"auto", "exact"}:
        msg = f"method must be 'auto' or 'exact', got {method!r}"
        raise ValueError(msg)
    if num_nodes < 1:
        msg = f"num_nodes must be positive, got {num_nodes}"
        raise ValueError(msg)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        msg = f"edge_index must have shape (2, E), got {tuple(edge_index.shape)}"
        raise ValueError(msg)

    if method == "exact":
        return _orbits_from_pynauty(edge_index, num_nodes)

    try:
        import pynauty  # noqa: F401
    except ImportError:
        pynauty = None  # type: ignore[assignment]
    if pynauty is not None:
        return _orbits_from_pynauty(edge_index, num_nodes)

    try:
        return _orbits_from_wl(edge_index, num_nodes)
    except ImportError:
        warnings.warn(
            "Neither pynauty nor networkx is available; using identity "
            "orbit partition. Install with: pip install 'koopman-graph[symmetry]'",
            UserWarning,
            stacklevel=2,
        )
        return identity_orbit_partition(num_nodes)


def apply_orbit_self(
    z: Tensor,
    orbit_matrices: Sequence[Tensor],
    node_orbit: Tensor,
) -> Tensor:
    """Apply per-orbit self maps: ``z_next[i] = z[i] @ K_{orbit(i)}.T``.

    Parameters
    ----------

    z : Tensor
        See the function signature / summary for ``z``.
    orbit_matrices : Sequence[Tensor]
        See the function signature / summary for ``orbit_matrices``.
    node_orbit : Tensor
        See the function signature / summary for ``node_orbit``.

    Returns
    -------

    Tensor
        See summary line.

    Raises
    ------

    ValueError
        Raised when inputs are invalid."""
    if z.ndim != 2:
        msg = f"z must have shape (num_nodes, latent_dim), got {tuple(z.shape)}"
        raise ValueError(msg)
    num_nodes = z.shape[0]
    if node_orbit.shape != (num_nodes,):
        msg = (
            f"node_orbit must have shape ({num_nodes},), got {tuple(node_orbit.shape)}"
        )
        raise ValueError(msg)
    z_next = torch.empty_like(z)
    for orbit_id, matrix in enumerate(orbit_matrices):
        mask = node_orbit == orbit_id
        if not bool(mask.any()):
            continue
        z_next[mask] = z[mask] @ matrix.T
    return z_next


def assemble_orbit_self_blocks(
    orbit_matrices: Sequence[Tensor],
    node_orbit: Tensor,
    num_nodes: int,
) -> Tensor:
    """Stack per-node self blocks ``(N, d, d)`` from orbit matrices.

    Parameters
    ----------

    orbit_matrices : Sequence[Tensor]
        See the function signature / summary for ``orbit_matrices``.
    node_orbit : Tensor
        See the function signature / summary for ``node_orbit``.
    num_nodes : int
        See the function signature / summary for ``num_nodes``.

    Returns
    -------

    Tensor
        See summary line.

    Raises
    ------

    ValueError
        Raised when inputs are invalid."""
    if not orbit_matrices:
        msg = "orbit_matrices must be non-empty"
        raise ValueError(msg)
    latent_dim = orbit_matrices[0].shape[0]
    blocks = torch.empty(
        num_nodes,
        latent_dim,
        latent_dim,
        dtype=orbit_matrices[0].dtype,
        device=orbit_matrices[0].device,
    )
    for node in range(num_nodes):
        blocks[node] = orbit_matrices[int(node_orbit[node].item())]
    return blocks
