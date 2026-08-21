"""Opt-in next-step topology heads (link formation / dissolution).

Distinct from :class:`~koopman_graph.nn.AdaptiveAdjacency`, which is a
static Graph WaveNet self-adaptive adjacency. The default graph-state
path is :class:`SparseCandidateTopologyHead` (at most ``candidate_k``
destinations per node). :class:`PredictedTopologyHead` remains the
power-user dense :math:`N\\times N` MLP behind ``dense_mlp``, with an
:math:`N` ceiling.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

import torch
from torch import Tensor, nn
from torch_geometric.data import Data

from koopman_graph.graph_utils.propagation import hold_last_topology_at

DENSE_TOPOLOGY_MAX_NODES = 64
DEFAULT_TOPOLOGY_HIDDEN_DIM = 32
DEFAULT_CANDIDATE_K = 8
TopologyPolicy = Literal["auto", "recursive", "hold_last"]
TOPOLOGY_POLICIES: frozenset[str] = frozenset({"auto", "recursive", "hold_last"})

__all__ = [
    "DEFAULT_CANDIDATE_K",
    "DEFAULT_TOPOLOGY_HIDDEN_DIM",
    "DENSE_TOPOLOGY_MAX_NODES",
    "TOPOLOGY_POLICIES",
    "PresenceHead",
    "PredictedTopologyHead",
    "SparseCandidateTopologyHead",
    "TopologyPolicy",
    "build_candidate_index",
    "build_supervision_index",
    "candidate_edge_labels",
    "decode_weighted_topology",
    "make_recursive_topology_at",
    "recursive_training_enabled",
    "resolve_rollout_topology_at",
    "resolve_topology_policy",
]


def _pair_mlp(latent_dim: int, hidden_dim: int) -> nn.Sequential:
    """Return the shared two-layer pair scorer.

    Parameters
    ----------
    latent_dim : int
        Node latent width.
    hidden_dim : int
        Hidden width.

    Returns
    -------
    nn.Sequential
        ``(2 d) → hidden → 1`` MLP.
    """
    return nn.Sequential(
        nn.Linear(2 * latent_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    )


def build_candidate_index(
    num_nodes: int,
    candidate_k: int,
    edge_index: Tensor | None = None,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """Build a COO candidate set with at most ``candidate_k`` destinations each.

    Existing ``edge_index`` destinations (excluding self-loops) are kept
    first, then filled with random other nodes. The helper never
    materializes an :math:`N\\times N` adjacency.

    Parameters
    ----------
    num_nodes : int
        Node count :math:`N`.
    candidate_k : int
        Maximum destinations per source (positive).
    edge_index : Tensor or None, optional
        Optional COO ``(2, E)`` of current edges.
    generator : torch.Generator or None, optional
        RNG for fill samples. ``None`` uses the global generator.
    device : torch.device or str or None, optional
        Output device. Defaults to ``edge_index.device`` when given.

    Returns
    -------
    Tensor
        COO index ``(2, E)`` with ``E <= N * min(k, N-1)``.

    Raises
    ------
    ValueError
        If ``num_nodes < 2`` or ``candidate_k < 1``.
    """
    if int(num_nodes) < 2:
        msg = f"num_nodes must be at least 2, got {num_nodes}"
        raise ValueError(msg)
    if int(candidate_k) < 1:
        msg = f"candidate_k must be positive, got {candidate_k}"
        raise ValueError(msg)
    resolved_device: torch.device
    if device is not None:
        resolved_device = torch.device(device)
    elif edge_index is not None:
        resolved_device = edge_index.device
    else:
        resolved_device = torch.device("cpu")
    cap = min(int(candidate_k), int(num_nodes) - 1)
    src_parts: list[Tensor] = []
    dst_parts: list[Tensor] = []
    existing: dict[int, list[int]] = {index: [] for index in range(int(num_nodes))}
    if edge_index is not None:
        if edge_index.ndim != 2 or int(edge_index.shape[0]) != 2:
            msg = f"edge_index must have shape (2, E), got {tuple(edge_index.shape)}"
            raise ValueError(msg)
        for source, dest in zip(
            edge_index[0].tolist(), edge_index[1].tolist(), strict=True
        ):
            src_i = int(source)
            dst_i = int(dest)
            if src_i == dst_i:
                continue
            if 0 <= src_i < int(num_nodes) and 0 <= dst_i < int(num_nodes):
                bucket = existing[src_i]
                if dst_i not in bucket and len(bucket) < cap:
                    bucket.append(dst_i)
    for source in range(int(num_nodes)):
        chosen = list(existing[source])
        if len(chosen) < cap:
            banned = {source, *chosen}
            pool = [node for node in range(int(num_nodes)) if node not in banned]
            need = cap - len(chosen)
            if pool:
                order = torch.randperm(
                    len(pool), device=resolved_device, generator=generator
                )
                for offset in order.tolist()[:need]:
                    chosen.append(pool[int(offset)])
        if not chosen:
            continue
        src_parts.append(
            torch.full((len(chosen),), source, dtype=torch.long, device=resolved_device)
        )
        dst_parts.append(torch.tensor(chosen, dtype=torch.long, device=resolved_device))
    if not src_parts:
        msg = "build_candidate_index produced no candidate edges"
        raise ValueError(msg)
    return torch.stack([torch.cat(src_parts), torch.cat(dst_parts)], dim=0)


class PredictedTopologyHead(nn.Module):
    """Predict dense pairwise edge logits from node latents.

    Power-user ``dense_mlp`` path. Prefer
    :class:`SparseCandidateTopologyHead` for graph-state closure.
    Distinct from :class:`~koopman_graph.nn.AdaptiveAdjacency`.

    Parameters
    ----------
    latent_dim : int
        Node latent width.
    hidden_dim : int, optional
        MLP hidden width. Default is 32.
    max_nodes : int, optional
        Dense :math:`N\\times N` ceiling. Default
        :data:`DENSE_TOPOLOGY_MAX_NODES` (64). Keyword-only.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = DEFAULT_TOPOLOGY_HIDDEN_DIM,
        *,
        max_nodes: int = DENSE_TOPOLOGY_MAX_NODES,
    ) -> None:
        """Initialize the pairwise MLP.

        Parameters
        ----------
        latent_dim : int
            Node latent width.
        hidden_dim : int, optional
            MLP hidden width.
        max_nodes : int, optional
            Dense node-count ceiling.
        """
        super().__init__()
        if latent_dim < 1:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if max_nodes < 2:
            raise ValueError(f"max_nodes must be at least 2, got {max_nodes}")
        self.latent_dim = int(latent_dim)
        self.max_nodes = int(max_nodes)
        self.mlp = _pair_mlp(int(latent_dim), int(hidden_dim))

    def pairwise_logits(self, z: Tensor) -> Tensor:
        """Return dense ``(N, N)`` logits.

        Parameters
        ----------
        z : Tensor
            Node latents ``(N, d)``.

        Returns
        -------
        Tensor
            Dense logits ``(N, N)``.

        Raises
        ------
        ValueError
            If ``z`` is not rank-2 or :math:`N` exceeds ``max_nodes``.
        """
        if z.ndim != 2:
            raise ValueError(f"z must have shape (N, d), got {tuple(z.shape)}")
        num_nodes = z.shape[0]
        if int(num_nodes) > self.max_nodes:
            msg = (
                "PredictedTopologyHead dense path refuses N="
                f"{int(num_nodes)} > max_nodes={self.max_nodes}. "
                "Use SparseCandidateTopologyHead / topology_head="
                "'sparse_candidate'."
            )
            raise ValueError(msg)
        left = z.unsqueeze(1).expand(num_nodes, num_nodes, self.latent_dim)
        right = z.unsqueeze(0).expand(num_nodes, num_nodes, self.latent_dim)
        pairs = torch.cat([left, right], dim=-1)
        logits = self.mlp(pairs).squeeze(-1)
        logits = logits.clone()
        logits.fill_diagonal_(-1e9)
        return logits

    def edge_index(
        self,
        z: Tensor,
        *,
        threshold: float = 0.0,
        top_k: int | None = None,
    ) -> Tensor:
        """Threshold or top-k the logits into a COO ``edge_index``.

        Parameters
        ----------
        z : Tensor
            Node latents.
        threshold : float, optional
            Logit threshold when ``top_k`` is None.
        top_k : int or None, optional
            If set, keep this many outgoing edges per node.

        Returns
        -------
        Tensor
            COO ``edge_index`` of shape ``(2, E)``.
        """
        logits = self.pairwise_logits(z)
        num_nodes = logits.shape[0]
        if top_k is not None:
            k = min(int(top_k), max(num_nodes - 1, 1))
            _, indices = torch.topk(logits, k=k, dim=-1)
            src = (
                torch.arange(num_nodes, device=z.device).unsqueeze(1).expand_as(indices)
            )
            return torch.stack([src.reshape(-1), indices.reshape(-1)], dim=0)
        src, dst = (logits > threshold).nonzero(as_tuple=True)
        if src.numel() == 0:
            eye = torch.arange(num_nodes, device=z.device)
            return torch.stack([eye, (eye + 1) % num_nodes], dim=0)
        return torch.stack([src, dst], dim=0)

    def forward(self, z: Tensor) -> Tensor:
        """Return pairwise logits.

        Parameters
        ----------
        z : Tensor
            Node latents.

        Returns
        -------
        Tensor
            Dense logits.
        """
        return self.pairwise_logits(z)


class SparseCandidateTopologyHead(nn.Module):
    """Score edge logits only on a sparse candidate COO index.

    Default graph-state topology head. Logits have shape ``(E,)`` with
    ``E <= N * candidate_k``, not dense ``(N, N)``.

    Parameters
    ----------
    latent_dim : int
        Node latent width.
    hidden_dim : int, optional
        MLP hidden width. Default is 32.
    candidate_k : int, optional
        Maximum destinations per source when building candidates.
        Default is 8.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = DEFAULT_TOPOLOGY_HIDDEN_DIM,
        *,
        candidate_k: int = DEFAULT_CANDIDATE_K,
    ) -> None:
        """Initialize the sparse pair scorer.

        Parameters
        ----------
        latent_dim : int
            Node latent width.
        hidden_dim : int, optional
            MLP hidden width.
        candidate_k : int, optional
            Candidate cap per source.
        """
        super().__init__()
        if latent_dim < 1:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if candidate_k < 1:
            raise ValueError(f"candidate_k must be positive, got {candidate_k}")
        self.latent_dim = int(latent_dim)
        self.candidate_k = int(candidate_k)
        self.mlp = _pair_mlp(int(latent_dim), int(hidden_dim))

    def pair_logits(self, z: Tensor, candidate_index: Tensor) -> Tensor:
        """Return one logit per candidate edge.

        Parameters
        ----------
        z : Tensor
            Node latents ``(N, d)``.
        candidate_index : Tensor
            COO ``(2, E)``.

        Returns
        -------
        Tensor
            Logits ``(E,)``.

        Raises
        ------
        ValueError
            If ranks are wrong, ``E`` exceeds ``N (N-1)``, or
            indices are out of range.
        """
        if z.ndim != 2:
            raise ValueError(f"z must have shape (N, d), got {tuple(z.shape)}")
        if candidate_index.ndim != 2 or int(candidate_index.shape[0]) != 2:
            msg = (
                "candidate_index must have shape (2, E), "
                f"got {tuple(candidate_index.shape)}"
            )
            raise ValueError(msg)
        num_nodes = int(z.shape[0])
        num_edges = int(candidate_index.shape[1])
        max_edges = num_nodes * max(num_nodes - 1, 0)
        if num_edges > max_edges:
            msg = f"candidate_index has E={num_edges} > N*(N-1)={max_edges}"
            raise ValueError(msg)
        src = candidate_index[0]
        dst = candidate_index[1]
        if src.numel() and (
            int(src.min()) < 0
            or int(dst.min()) < 0
            or int(src.max()) >= num_nodes
            or int(dst.max()) >= num_nodes
        ):
            msg = "candidate_index contains node ids outside [0, N)"
            raise ValueError(msg)
        pairs = torch.cat([z[src], z[dst]], dim=-1)
        return self.mlp(pairs).squeeze(-1)

    def edge_index(
        self,
        z: Tensor,
        candidate_index: Tensor,
        *,
        threshold: float = 0.0,
    ) -> Tensor:
        """Keep candidate edges whose logits exceed ``threshold``.

        Parameters
        ----------
        z : Tensor
            Node latents.
        candidate_index : Tensor
            COO candidates ``(2, E)``.
        threshold : float, optional
            Logit threshold. Default ``0.0``.

        Returns
        -------
        Tensor
            Filtered COO ``(2, E_keep)``. If none pass, returns a
            cycle of self-avoiding fallback edges.
        """
        logits = self.pair_logits(z, candidate_index)
        keep = logits > threshold
        if int(keep.sum()) == 0:
            num_nodes = int(z.shape[0])
            eye = torch.arange(num_nodes, device=z.device)
            return torch.stack([eye, (eye + 1) % num_nodes], dim=0)
        return candidate_index[:, keep]

    def forward(self, z: Tensor, candidate_index: Tensor) -> Tensor:
        """Return sparse candidate logits.

        Parameters
        ----------
        z : Tensor
            Node latents.
        candidate_index : Tensor
            COO candidates.

        Returns
        -------
        Tensor
            Logits ``(E,)``.
        """
        return self.pair_logits(z, candidate_index)


class PresenceHead(nn.Module):
    """Linear per-node presence logit from node latents.

    Used when :class:`~koopman_graph.data.GraphDynamicsConfig` is attached.
    BCE-with-logits targets the next-step presence mask when the sequence
    carries masks; otherwise the term is skipped.

    Parameters
    ----------
    latent_dim : int
        Node latent width.
    """

    def __init__(self, latent_dim: int) -> None:
        """Initialize the linear presence scorer.

        Parameters
        ----------
        latent_dim : int
            Node latent width.
        """
        super().__init__()
        if latent_dim < 1:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        self.latent_dim = int(latent_dim)
        self.linear = nn.Linear(int(latent_dim), 1)

    def forward(self, z: Tensor) -> Tensor:
        """Return one logit per node.

        Parameters
        ----------
        z : Tensor
            Node latents ``(N, d)``.

        Returns
        -------
        Tensor
            Logits ``(N,)``.
        """
        if z.ndim != 2:
            raise ValueError(f"z must have shape (N, d), got {tuple(z.shape)}")
        if int(z.shape[1]) != self.latent_dim:
            msg = f"z latent width must be {self.latent_dim}, got {int(z.shape[1])}"
            raise ValueError(msg)
        return self.linear(z).squeeze(-1)


def _destination_buckets(
    num_nodes: int,
    edge_index: Tensor | None,
) -> dict[int, list[int]]:
    """Collect unique off-diagonal destinations per source.

    Parameters
    ----------
    num_nodes : int
        Node count :math:`N`.
    edge_index : Tensor or None
        Optional COO ``(2, E)``.

    Returns
    -------
    dict of int to list of int
        Destinations keyed by source, without self-loops.
    """
    buckets: dict[int, list[int]] = {index: [] for index in range(int(num_nodes))}
    if edge_index is None:
        return buckets
    if edge_index.ndim != 2 or int(edge_index.shape[0]) != 2:
        msg = f"edge_index must have shape (2, E), got {tuple(edge_index.shape)}"
        raise ValueError(msg)
    for source, dest in zip(
        edge_index[0].tolist(), edge_index[1].tolist(), strict=True
    ):
        src_i = int(source)
        dst_i = int(dest)
        if src_i == dst_i:
            continue
        if 0 <= src_i < int(num_nodes) and 0 <= dst_i < int(num_nodes):
            bucket = buckets[src_i]
            if dst_i not in bucket:
                bucket.append(dst_i)
    return buckets


def _coo_from_buckets(
    buckets: dict[int, list[int]],
    *,
    device: torch.device,
) -> Tensor:
    """Stack per-source destination lists into a COO index.

    Parameters
    ----------
    buckets : dict of int to list of int
        Destinations keyed by source.
    device : torch.device
        Output device.

    Returns
    -------
    Tensor
        COO ``(2, E)``.

    Raises
    ------
    ValueError
        If every bucket is empty.
    """
    src_parts: list[Tensor] = []
    dst_parts: list[Tensor] = []
    for source, chosen in buckets.items():
        if not chosen:
            continue
        src_parts.append(
            torch.full((len(chosen),), source, dtype=torch.long, device=device)
        )
        dst_parts.append(torch.tensor(chosen, dtype=torch.long, device=device))
    if not src_parts:
        msg = "candidate construction produced no edges"
        raise ValueError(msg)
    return torch.stack([torch.cat(src_parts), torch.cat(dst_parts)], dim=0)


def build_supervision_index(
    num_nodes: int,
    candidate_k: int,
    current_edge_index: Tensor | None,
    next_edge_index: Tensor,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """Build a COO supervision set from current edges union true next edges.

    Existing current and true-next destinations (excluding self-loops) are
    kept first. Random fill then raises each source to ``candidate_k``
    destinations when the union is smaller. If the union already exceeds
    ``candidate_k``, the union is kept and fill is skipped so formation
    positives that fit in the union are not dropped. ``candidate_k`` remains
    an inference construction cap.

    Parameters
    ----------
    num_nodes : int
        Node count :math:`N`.
    candidate_k : int
        Inference construction cap per source (positive).
    current_edge_index : Tensor or None
        COO ``(2, E)`` of the source snapshot.
    next_edge_index : Tensor
        COO ``(2, E')`` of the true next snapshot.
    generator : torch.Generator or None, optional
        RNG for fill samples. ``None`` uses the global generator.
    device : torch.device or str or None, optional
        Output device. Defaults to ``next_edge_index.device``.

    Returns
    -------
    Tensor
        COO index ``(2, E)``. ``E`` may exceed ``N k`` when the union
        is larger than the inference cap.

    Raises
    ------
    ValueError
        If ``num_nodes < 2`` or ``candidate_k < 1``.
    """
    if int(num_nodes) < 2:
        msg = f"num_nodes must be at least 2, got {num_nodes}"
        raise ValueError(msg)
    if int(candidate_k) < 1:
        msg = f"candidate_k must be positive, got {candidate_k}"
        raise ValueError(msg)
    resolved_device: torch.device
    if device is not None:
        resolved_device = torch.device(device)
    else:
        resolved_device = next_edge_index.device
    cap = min(int(candidate_k), int(num_nodes) - 1)
    buckets = _destination_buckets(int(num_nodes), current_edge_index)
    next_buckets = _destination_buckets(int(num_nodes), next_edge_index)
    for source, extras in next_buckets.items():
        bucket = buckets[source]
        for dest in extras:
            if dest not in bucket:
                bucket.append(dest)
    for source in range(int(num_nodes)):
        chosen = buckets[source]
        if len(chosen) >= cap:
            continue
        banned = {source, *chosen}
        pool = [node for node in range(int(num_nodes)) if node not in banned]
        need = cap - len(chosen)
        if not pool:
            continue
        order = torch.randperm(len(pool), device=resolved_device, generator=generator)
        for offset in order.tolist()[:need]:
            chosen.append(pool[int(offset)])
    return _coo_from_buckets(buckets, device=resolved_device)


def candidate_edge_labels(
    candidate_index: Tensor,
    true_next_index: Tensor,
    num_nodes: int,
) -> Tensor:
    """Return 0/1 labels for whether each candidate appears in the next graph.

    Parameters
    ----------
    candidate_index : Tensor
        COO candidates ``(2, E)``.
    true_next_index : Tensor
        True next-step COO ``(2, E')``.
    num_nodes : int
        Node count used to hash pairs.

    Returns
    -------
    Tensor
        Float labels ``(E,)`` in ``{0, 1}``.
    """
    if candidate_index.ndim != 2 or int(candidate_index.shape[0]) != 2:
        msg = (
            "candidate_index must have shape (2, E), "
            f"got {tuple(candidate_index.shape)}"
        )
        raise ValueError(msg)
    if true_next_index.ndim != 2 or int(true_next_index.shape[0]) != 2:
        msg = (
            "true_next_index must have shape (2, E), "
            f"got {tuple(true_next_index.shape)}"
        )
        raise ValueError(msg)
    num_edges = int(candidate_index.shape[1])
    if num_edges == 0:
        return torch.zeros(0, device=candidate_index.device, dtype=torch.float32)
    width = int(num_nodes)
    cand_hash = candidate_index[0] * width + candidate_index[1]
    if int(true_next_index.shape[1]) == 0:
        return torch.zeros(
            num_edges,
            device=candidate_index.device,
            dtype=torch.float32,
        )
    true_hash = true_next_index[0] * width + true_next_index[1]
    return torch.isin(cand_hash, true_hash).to(dtype=torch.float32)


def dense_offdiag_index(num_nodes: int, *, device: torch.device) -> Tensor:
    """Return COO of all off-diagonal ordered pairs.

    Parameters
    ----------
    num_nodes : int
        Node count :math:`N` (at least 2).
    device : torch.device
        Output device.

    Returns
    -------
    Tensor
        COO ``(2, N(N-1))``.
    """
    n_nodes = int(num_nodes)
    mask = ~torch.eye(n_nodes, dtype=torch.bool, device=device)
    src, dst = mask.nonzero(as_tuple=True)
    return torch.stack([src, dst], dim=0)


def decode_weighted_topology(
    head: PredictedTopologyHead | SparseCandidateTopologyHead,
    z: Tensor,
    current_edge_index: Tensor | None,
) -> tuple[Tensor, Tensor]:
    """Score :math:`g_\\phi(z)` and return sigmoid weights on a COO index.

    Sparse heads use :func:`build_candidate_index` from ``current_edge_index``.
    Dense heads emit all off-diagonal pairs. Weights are ``sigmoid(logits)``
    so the operator step cannot collapse to an empty graph.

    Parameters
    ----------
    head : PredictedTopologyHead or SparseCandidateTopologyHead
        Attached topology head.
    z : Tensor
        Node latents ``(N, d)``.
    current_edge_index : Tensor or None
        Current COO used to seed sparse candidates.

    Returns
    -------
    edge_index : Tensor
        COO ``(2, E)``.
    edge_weight : Tensor
        Sigmoid scores ``(E,)``.
    """
    if isinstance(head, SparseCandidateTopologyHead):
        candidates = build_candidate_index(
            int(z.shape[0]),
            head.candidate_k,
            current_edge_index,
            device=z.device,
        )
        logits = head.pair_logits(z, candidates)
        return candidates, torch.sigmoid(logits)
    if isinstance(head, PredictedTopologyHead):
        logits = head.pairwise_logits(z)
        index = dense_offdiag_index(int(z.shape[0]), device=z.device)
        weights = torch.sigmoid(logits[index[0], index[1]])
        return index, weights
    msg = (
        "decode_weighted_topology expects PredictedTopologyHead or "
        f"SparseCandidateTopologyHead, got {type(head).__name__}"
    )
    raise TypeError(msg)


def make_recursive_topology_at(
    head: PredictedTopologyHead | SparseCandidateTopologyHead,
    origin_index: Tensor,
) -> Callable[[int, Tensor], tuple[Tensor, Tensor | None]]:
    """Build ``topology_at(step, latent)`` from :math:`g_\\phi(z_t)`.

    Each call scores the current latent on candidates seeded by the
    previously emitted COO (origin edges at step 0).

    Parameters
    ----------
    head : PredictedTopologyHead or SparseCandidateTopologyHead
        Topology head.
    origin_index : Tensor
        COO of the rollout origin, used as the first candidate seed.

    Returns
    -------
    callable
        ``topology_at(step, latent) -> (edge_index, edge_weight)``.
    """
    current_index = origin_index

    def topology_at(step: int, latent: Tensor) -> tuple[Tensor, Tensor | None]:
        """Return predicted weighted topology for one rollout step.

        Parameters
        ----------
        step : int
            Zero-based rollout step (unused; the latent carries state).
        latent : Tensor
            Current node latents.

        Returns
        -------
        tuple[Tensor, Tensor or None]
            Predicted COO and sigmoid weights.
        """
        nonlocal current_index
        del step
        predicted_index, predicted_weight = decode_weighted_topology(
            head,
            latent,
            current_index,
        )
        current_index = predicted_index
        return predicted_index, predicted_weight

    return topology_at


def recursive_training_enabled(model: object) -> bool:
    """Return whether predicted topology should enter the operator step.

    Parameters
    ----------
    model : object
        Model exposing ``graph_dynamics`` and ``predicted_topology``.

    Returns
    -------
    bool
        ``True`` when a head is attached and ``recursive_training`` is
        ``True``.
    """
    config = getattr(model, "graph_dynamics", None)
    head = getattr(model, "predicted_topology", None)
    return config is not None and head is not None and bool(config.recursive_training)


def resolve_topology_policy(
    model: object,
    topology_policy: TopologyPolicy | str = "auto",
) -> Literal["recursive", "hold_last"]:
    """Resolve ``auto`` / ``recursive`` / ``hold_last`` to a concrete policy.

    Parameters
    ----------
    model : object
        Model exposing ``graph_dynamics`` and ``predicted_topology``.
    topology_policy : {"auto", "recursive", "hold_last"}, optional
        Requested policy. ``auto`` is recursive when
        :func:`recursive_training_enabled` is true, else hold-last.

    Returns
    -------
    {"recursive", "hold_last"}
        Concrete rollout policy.

    Raises
    ------
    ValueError
        If ``topology_policy`` is unknown, or ``recursive`` is requested
        without a topology head.
    """
    if topology_policy not in TOPOLOGY_POLICIES:
        allowed = ", ".join(sorted(TOPOLOGY_POLICIES))
        msg = f"topology_policy must be one of {{{allowed}}}; got {topology_policy!r}"
        raise ValueError(msg)
    if topology_policy == "hold_last":
        return "hold_last"
    head = getattr(model, "predicted_topology", None)
    if topology_policy == "recursive":
        if head is None:
            msg = "topology_policy='recursive' requires a predicted topology head"
            raise ValueError(msg)
        return "recursive"
    if recursive_training_enabled(model):
        return "recursive"
    return "hold_last"


def resolve_rollout_topology_at(
    model: object,
    origin_index: Tensor,
    origin_weight: Tensor | None,
    future_topologies: Sequence[Data] | None = None,
    topology_policy: TopologyPolicy | str = "auto",
) -> Callable[..., tuple[Tensor, Tensor | None]]:
    """Select hold-last, oracle-future, or recursive predicted topology.

    Oracle ``future_topologies`` always wins. Otherwise the resolved
    policy chooses :func:`~koopman_graph.graph_utils.hold_last_topology_at`
    or :func:`make_recursive_topology_at`.

    Parameters
    ----------
    model : object
        Model exposing ``predicted_topology`` / ``graph_dynamics``.
    origin_index : Tensor
        Origin COO.
    origin_weight : Tensor or None
        Origin edge weights (hold-last / oracle only).
    future_topologies : sequence of Data or None, optional
        Oracle future graphs. When set, hold-last-with-futures is used.
    topology_policy : {"auto", "recursive", "hold_last"}, optional
        Requested policy. Default ``"auto"``.

    Returns
    -------
    callable
        ``topology_at`` accepted by
        :func:`~koopman_graph.graph_utils.autoregressive_latent_rollout`.
    """
    if future_topologies is not None:
        return hold_last_topology_at(origin_index, origin_weight, future_topologies)
    policy = resolve_topology_policy(model, topology_policy)
    if policy == "hold_last":
        return hold_last_topology_at(origin_index, origin_weight, None)
    head = getattr(model, "predicted_topology", None)
    if head is None:
        msg = "recursive topology requires a predicted topology head"
        raise ValueError(msg)
    if not isinstance(head, (PredictedTopologyHead, SparseCandidateTopologyHead)):
        msg = (
            "predicted_topology must be PredictedTopologyHead or "
            f"SparseCandidateTopologyHead, got {type(head).__name__}"
        )
        raise TypeError(msg)
    return make_recursive_topology_at(head, origin_index)
