"""Pooling / unpooling helpers for hierarchical GraphKoopman forecasting.

Uses PyG :class:`~torch_geometric.nn.TopKPooling` (default) or
:class:`~torch_geometric.nn.SAGPooling`. Global controls pass through unchanged;
per-node controls are indexed by the pooling ``perm`` chain.

Multiplex hetero pooling scores TopK/SAG on the **union** of relation banks,
then induces each coarse relation bank with the same ``perm`` so the relation
schema is preserved. Typed hetero pooling scores **per node type** on that
type's within-type banks and induces every relation bank (including
cross-type) with the paired source/destination permutations.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import SAGPooling, TopKPooling
from torch_geometric.utils import subgraph

from koopman_graph.graph_utils import snapshot_edge_weight

PoolingKind = Literal["topk", "sag"]
PoolSchedule = Literal["per_snapshot", "hold_perm"]
EdgeTypeTriple = tuple[str, str, str]


@dataclass(frozen=True)
class PoolStep:
    """One pooling level's topology reduction metadata.

    Attributes
    ----------
    perm : Tensor
        Indices of selected nodes in the *previous* level
        (``LongTensor`` of shape ``(n_coarse,)``).
    num_fine : int
        Node count of the previous (finer) level.
    edge_index : Tensor
        Coarse scoring / homogeneous edge index after this level. For
        multiplex hetero this is the union of coarse relation banks.
    edge_weight : Tensor or None
        Optional coarse edge weights for the scoring graph.
    relation_edge_indices : tuple of Tensor or None
        Per-relation coarse banks (multiplex / typed hetero). ``None`` for
        homogeneous pooling.
    relation_edge_weights : tuple of Tensor or None, or None
        Optional per-relation coarse weights aligned with
        ``relation_edge_indices``.
    typed_node_types : tuple of str or None
        Ordered node types for typed pooling (``None`` when unused).
    typed_perms : tuple of Tensor or None
        Per-type selected fine indices aligned with ``typed_node_types``.
    typed_num_fine : tuple of int or None
        Per-type fine node counts aligned with ``typed_node_types``.
    """

    perm: Tensor
    num_fine: int
    edge_index: Tensor
    edge_weight: Tensor | None
    relation_edge_indices: tuple[Tensor, ...] | None = None
    relation_edge_weights: tuple[Tensor | None, ...] | None = None
    typed_node_types: tuple[str, ...] | None = None
    typed_perms: tuple[Tensor, ...] | None = None
    typed_num_fine: tuple[int, ...] | None = None


def build_pool_layer(
    in_channels: int,
    ratio: float,
    *,
    pooling: PoolingKind = "topk",
) -> TopKPooling | SAGPooling:
    """Construct a PyG pooling layer for one hierarchy level.

    Parameters
    ----------
    in_channels : int
        Node feature dimension used by the scoring network.
    ratio : float
        Fraction of nodes retained in ``(0, 1]``. ``1.0`` keeps all nodes
        (no-op size; scores may still reorder).
    pooling : {"topk", "sag"}, optional
        Pooling operator. Default ``"topk"``.

    Returns
    -------
    TopKPooling or SAGPooling
        Configured pooling module.

    Raises
    ------
    ValueError
        If ``ratio`` or ``pooling`` is invalid.
    """
    if not 0.0 < ratio <= 1.0:
        msg = f"pool ratio must be in (0, 1], got {ratio}"
        raise ValueError(msg)
    if pooling == "topk":
        return TopKPooling(in_channels, ratio=ratio)
    if pooling == "sag":
        return SAGPooling(in_channels, ratio=ratio)
    msg = f"pooling must be 'topk' or 'sag', got {pooling!r}"
    raise ValueError(msg)


def _is_identity_ratio(ratio: float | int) -> bool:
    """Return True when ``ratio`` should keep every node.

    PyG :class:`~torch_geometric.nn.pool.select.SelectTopK` treats the float
    ``1.0`` as an absolute count of one node. We therefore special-case full
    retention explicitly for the documented ``pool_ratios=(1.0,)`` no-op path.

    Parameters
    ----------

    ratio : float | int
        See the function signature / summary for ``ratio``.

    Returns
    -------

    bool
        Whether the ratio requests full node retention."""
    return float(ratio) >= 1.0


def apply_pool_layer(
    layer: TopKPooling | SAGPooling,
    x: Tensor,
    edge_index: Tensor,
    edge_weight: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor | None, Tensor]:
    """Run one pooling layer and return coarse features plus ``perm``.

    Parameters
    ----------
    layer : TopKPooling or SAGPooling
        Pooling module.
    x : Tensor
        Fine node features ``(N, F)``.
    edge_index : Tensor
        Fine topology ``(2, E)``.
    edge_weight : Tensor or None, optional
        Optional fine edge weights ``(E,)``.

    Returns
    -------
    tuple
        ``(x_coarse, edge_index_coarse, edge_weight_coarse, perm)``.

    Notes
    -----
    When ``layer.ratio >= 1``, returns an identity reduction (``perm = arange``)
    so ``pool_ratios=(1.0,)`` keeps all nodes. PyG would otherwise interpret
    float ``1.0`` as selecting a single node.
    """
    ratio = getattr(layer, "ratio", None)
    if ratio is not None and _is_identity_ratio(ratio):
        perm = torch.arange(x.size(0), device=x.device, dtype=torch.long)
        return x, edge_index, edge_weight, perm

    out = layer(x, edge_index, edge_attr=edge_weight)
    # TopK/SAG: (x, edge_index, edge_attr, batch, perm, score)
    x_c, edge_c, edge_attr_c, _batch, perm, _score = out
    if edge_attr_c is not None and edge_attr_c.ndim > 1:
        # Pooling may leave multi-dim edge_attr; keep scalar weights only.
        edge_attr_c = edge_attr_c.view(edge_attr_c.size(0), -1)[:, 0]
    return x_c, edge_c, edge_attr_c, perm


def union_relation_edge_index(edge_indices: Sequence[Tensor]) -> Tensor:
    """Concatenate relation banks into one scoring ``edge_index``.

    Parameters
    ----------
    edge_indices : sequence of Tensor
        Per-relation ``(2, E_r)`` banks.

    Returns
    -------
    Tensor
        Union ``edge_index`` with shape ``(2, E_union)`` (may be empty).
    """
    nonempty = [bank for bank in edge_indices if bank.numel() > 0]
    if not nonempty:
        device = edge_indices[0].device if edge_indices else torch.device("cpu")
        dtype = edge_indices[0].dtype if edge_indices else torch.long
        return torch.zeros(2, 0, dtype=dtype, device=device)
    return torch.cat(nonempty, dim=1)


def induce_relation_banks(
    perm: Tensor,
    num_fine: int,
    edge_indices: Sequence[Tensor],
    edge_weights: Sequence[Tensor | None] | None = None,
) -> tuple[tuple[Tensor, ...], tuple[Tensor | None, ...]]:
    """Induce coarse relation banks with a pooling permutation.

    Parameters
    ----------
    perm : Tensor
        Selected fine-node indices ``(n_coarse,)``.
    num_fine : int
        Fine node count.
    edge_indices : sequence of Tensor
        Fine relation banks.
    edge_weights : sequence of Tensor or None, or None, optional
        Optional fine relation weights.

    Returns
    -------
    tuple
        Coarse ``(relation_edge_indices, relation_edge_weights)``.
    """
    weights = (
        list(edge_weights) if edge_weights is not None else [None] * len(edge_indices)
    )
    coarse_indices: list[Tensor] = []
    coarse_weights: list[Tensor | None] = []
    for edge_index, edge_weight in zip(edge_indices, weights, strict=True):
        edge_c, edge_w = subgraph(
            perm,
            edge_index,
            edge_attr=edge_weight,
            relabel_nodes=True,
            num_nodes=num_fine,
        )
        coarse_indices.append(edge_c)
        coarse_weights.append(edge_w)
    return tuple(coarse_indices), tuple(coarse_weights)


def multiplex_snapshot_from_features(
    features: Tensor,
    *,
    node_type: str,
    edge_types: Sequence[EdgeTypeTriple],
    relation_edge_indices: Sequence[Tensor],
    relation_edge_weights: Sequence[Tensor | None] | None = None,
) -> HeteroData:
    """Build a multiplex ``HeteroData`` from features and relation banks.

    Parameters
    ----------
    features : Tensor
        Node features ``(N, F)`` for the single multiplex type.
    node_type : str
        Multiplex node-type name (typically ``"node"``).
    edge_types : sequence of (src, rel, dst)
        Ordered relation schema.
    relation_edge_indices : sequence of Tensor
        Relation banks aligned with ``edge_types``.
    relation_edge_weights : sequence of Tensor or None, or None, optional
        Optional per-relation weights.

    Returns
    -------
    HeteroData
        Multiplex snapshot.

    Raises
    ------
    ValueError
        If bank counts disagree with ``edge_types``.
    """
    if len(relation_edge_indices) != len(edge_types):
        msg = (
            f"expected {len(edge_types)} relation banks, "
            f"got {len(relation_edge_indices)}"
        )
        raise ValueError(msg)
    weights = (
        list(relation_edge_weights)
        if relation_edge_weights is not None
        else [None] * len(edge_types)
    )
    snap = HeteroData()
    snap[node_type].x = features
    for triple, edges, weight in zip(
        edge_types, relation_edge_indices, weights, strict=True
    ):
        snap[triple].edge_index = edges
        if weight is not None:
            snap[triple].edge_weight = weight
    return snap


def pool_features_with_steps(features: Tensor, steps: Sequence[PoolStep]) -> Data:
    """Pool node features by indexing through held :class:`PoolStep` perms.

    Reuses each step's coarse topology from the reference pool. Does **not**
    re-run TopK/SAG score nets — suitable for ``pool_schedule="hold_perm"``.

    Parameters
    ----------
    features : Tensor
        Fine node features ``(N, F)``.
    steps : sequence of PoolStep
        Fine→coarse metadata from a reference :meth:`pool_down` (or equivalent).

    Returns
    -------
    Data
        Coarse snapshot whose ``x`` is ``features`` indexed by the perm chain
        and whose topology matches the last step.

    Raises
    ------
    ValueError
        If ``steps`` is empty.
    """
    if len(steps) == 0:
        msg = "steps must contain at least one PoolStep"
        raise ValueError(msg)
    x = features
    for step in steps:
        x = x[step.perm]
    last = steps[-1]
    return snapshot_from_features(x, last.edge_index, last.edge_weight)


def pool_multiplex_features_with_steps(
    features: Tensor,
    steps: Sequence[PoolStep],
    *,
    node_type: str,
    edge_types: Sequence[EdgeTypeTriple],
) -> HeteroData:
    """Pool multiplex features with held perms and coarse relation banks.

    Parameters
    ----------
    features : Tensor
        Fine multiplex node features ``(N, F)``.
    steps : sequence of PoolStep
        Fine→coarse metadata with ``relation_edge_indices`` populated.
    node_type : str
        Multiplex node-type name.
    edge_types : sequence of (src, rel, dst)
        Relation schema.

    Returns
    -------
    HeteroData
        Coarse multiplex snapshot.

    Raises
    ------
    ValueError
        If ``steps`` is empty or lacks relation banks.
    """
    if len(steps) == 0:
        msg = "steps must contain at least one PoolStep"
        raise ValueError(msg)
    x = features
    for step in steps:
        x = x[step.perm]
    last = steps[-1]
    if last.relation_edge_indices is None:
        msg = "hold_perm multiplex pooling requires relation banks on PoolStep"
        raise ValueError(msg)
    return multiplex_snapshot_from_features(
        x,
        node_type=node_type,
        edge_types=edge_types,
        relation_edge_indices=last.relation_edge_indices,
        relation_edge_weights=last.relation_edge_weights,
    )


def within_type_scoring_edge_index(
    edge_types: Sequence[EdgeTypeTriple],
    edge_indices: Sequence[Tensor],
    *,
    node_type: str,
) -> Tensor:
    """Union within-type relation banks for TopK/SAG scoring on ``node_type``.

    Parameters
    ----------
    edge_types : sequence of (src, rel, dst)
        Ordered relation schema.
    edge_indices : sequence of Tensor
        Fine banks aligned with ``edge_types``.
    node_type : str
        Node type whose within-type edges are kept.

    Returns
    -------
    Tensor
        Scoring ``edge_index`` (may be empty when the type has only
        cross-type edges).
    """
    banks = [
        bank
        for triple, bank in zip(edge_types, edge_indices, strict=True)
        if triple[0] == node_type and triple[2] == node_type
    ]
    return union_relation_edge_index(banks)


def _fine_to_coarse_map(perm: Tensor, num_fine: int) -> Tensor:
    """Build a fine-index → coarse-index map (``-1`` = dropped).

    Parameters
    ----------
    perm : Tensor
        Selected fine indices ``(n_coarse,)``.
    num_fine : int
        Fine node count for this type.

    Returns
    -------
    Tensor
        Long map of shape ``(num_fine,)``.
    """
    mapping = torch.full(
        (num_fine,),
        -1,
        dtype=torch.long,
        device=perm.device,
    )
    mapping[perm] = torch.arange(perm.numel(), device=perm.device, dtype=torch.long)
    return mapping


def induce_typed_relation_banks(
    edge_types: Sequence[EdgeTypeTriple],
    edge_indices: Sequence[Tensor],
    edge_weights: Sequence[Tensor | None] | None,
    *,
    perms_by_type: dict[str, Tensor],
    num_fine_by_type: dict[str, int],
) -> tuple[tuple[Tensor, ...], tuple[Tensor | None, ...]]:
    """Induce coarse relation banks with per-type pooling permutations.

    Cross-type banks use the source-type ``perm`` for row indices and the
    destination-type ``perm`` for column indices.

    Parameters
    ----------
    edge_types : sequence of (src, rel, dst)
        Ordered relation schema.
    edge_indices : sequence of Tensor
        Fine relation banks.
    edge_weights : sequence of Tensor or None, or None
        Optional fine relation weights.
    perms_by_type : dict[str, Tensor]
        Selected fine indices per node type.
    num_fine_by_type : dict[str, int]
        Fine node counts per node type.

    Returns
    -------
    tuple
        Coarse ``(relation_edge_indices, relation_edge_weights)``.

    Raises
    ------
    KeyError
        If a bank endpoint type is missing from ``perms_by_type``.
    """
    weights = (
        list(edge_weights) if edge_weights is not None else [None] * len(edge_indices)
    )
    maps = {
        node_type: _fine_to_coarse_map(perm, num_fine_by_type[node_type])
        for node_type, perm in perms_by_type.items()
    }
    coarse_indices: list[Tensor] = []
    coarse_weights: list[Tensor | None] = []
    for triple, edge_index, edge_weight in zip(
        edge_types, edge_indices, weights, strict=True
    ):
        src_type, _rel, dst_type = triple
        src_map = maps[src_type]
        dst_map = maps[dst_type]
        if edge_index.numel() == 0:
            coarse_indices.append(edge_index)
            coarse_weights.append(edge_weight)
            continue
        src, dst = edge_index[0], edge_index[1]
        keep = (src_map[src] >= 0) & (dst_map[dst] >= 0)
        new_src = src_map[src[keep]]
        new_dst = dst_map[dst[keep]]
        coarse_indices.append(torch.stack([new_src, new_dst], dim=0))
        if edge_weight is None:
            coarse_weights.append(None)
        else:
            coarse_weights.append(edge_weight[keep])
    return tuple(coarse_indices), tuple(coarse_weights)


def typed_snapshot_from_features(
    features_by_type: dict[str, Tensor],
    *,
    node_types: Sequence[str],
    edge_types: Sequence[EdgeTypeTriple],
    relation_edge_indices: Sequence[Tensor],
    relation_edge_weights: Sequence[Tensor | None] | None = None,
) -> HeteroData:
    """Build a typed ``HeteroData`` from per-type features and relation banks.

    Parameters
    ----------
    features_by_type : dict[str, Tensor]
        Node features keyed by node type.
    node_types : sequence of str
        Ordered node-type schema.
    edge_types : sequence of (src, rel, dst)
        Ordered relation schema.
    relation_edge_indices : sequence of Tensor
        Relation banks aligned with ``edge_types``.
    relation_edge_weights : sequence of Tensor or None, or None, optional
        Optional per-relation weights.

    Returns
    -------
    HeteroData
        Typed snapshot.

    Raises
    ------
    ValueError
        If bank counts disagree with ``edge_types`` or a type is missing.
    KeyError
        If ``features_by_type`` omits a required node type.
    """
    if len(relation_edge_indices) != len(edge_types):
        msg = (
            f"expected {len(edge_types)} relation banks, "
            f"got {len(relation_edge_indices)}"
        )
        raise ValueError(msg)
    weights = (
        list(relation_edge_weights)
        if relation_edge_weights is not None
        else [None] * len(edge_types)
    )
    snap = HeteroData()
    for node_type in node_types:
        snap[node_type].x = features_by_type[node_type]
    for triple, edges, weight in zip(
        edge_types, relation_edge_indices, weights, strict=True
    ):
        snap[triple].edge_index = edges
        if weight is not None:
            snap[triple].edge_weight = weight
    return snap


def pool_typed_features_with_steps(
    features_by_type: dict[str, Tensor],
    steps: Sequence[PoolStep],
    *,
    node_types: Sequence[str],
    edge_types: Sequence[EdgeTypeTriple],
) -> HeteroData:
    """Pool typed features with held per-type perms and coarse relation banks.

    Parameters
    ----------
    features_by_type : dict[str, Tensor]
        Fine per-type node features.
    steps : sequence of PoolStep
        Fine→coarse metadata with typed perm fields and relation banks.
    node_types : sequence of str
        Ordered node-type schema.
    edge_types : sequence of (src, rel, dst)
        Relation schema.

    Returns
    -------
    HeteroData
        Coarse typed snapshot.

    Raises
    ------
    ValueError
        If ``steps`` is empty or lacks typed / relation metadata.
    """
    if len(steps) == 0:
        msg = "steps must contain at least one PoolStep"
        raise ValueError(msg)
    features = dict(features_by_type)
    for step in steps:
        if (
            step.typed_node_types is None
            or step.typed_perms is None
            or step.typed_num_fine is None
        ):
            msg = "hold_perm typed pooling requires typed perm fields on PoolStep"
            raise ValueError(msg)
        for node_type, perm in zip(
            step.typed_node_types, step.typed_perms, strict=True
        ):
            features[node_type] = features[node_type][perm]
    last = steps[-1]
    if last.relation_edge_indices is None:
        msg = "hold_perm typed pooling requires relation banks on PoolStep"
        raise ValueError(msg)
    return typed_snapshot_from_features(
        features,
        node_types=node_types,
        edge_types=edge_types,
        relation_edge_indices=last.relation_edge_indices,
        relation_edge_weights=last.relation_edge_weights,
    )


class ScatterUnpool(nn.Module):
    """Scatter coarse features onto fine nodes via ``perm``, optional refine.

    Non-selected fine nodes receive zeros before the optional linear refine.
    The refine layer is identity-initialized so ``pool_ratios=(1.0,)`` can
    numerically approach a flat model when the pool keeps every node.

    Attributes
    ----------
    channels : int
        Feature width at coarse and fine resolutions.
    refine : torch.nn.Linear or None
        Optional identity-initialized refinement map.
    """

    def __init__(self, channels: int, *, refine: bool = True) -> None:
        """Allocate an optional identity-initialized refine linear map.

        Parameters
        ----------
        channels : int
            Feature dimension.
        refine : bool, optional
            When ``True`` (default), apply a linear map after scatter.
        """
        super().__init__()
        self.channels = channels
        if refine:
            linear = nn.Linear(channels, channels)
            nn.init.eye_(linear.weight)
            nn.init.zeros_(linear.bias)
            self.refine: nn.Linear | None = linear
        else:
            self.refine = None

    def forward(self, coarse_x: Tensor, perm: Tensor, num_fine: int) -> Tensor:
        """Scatter ``coarse_x`` onto fine indices and optionally refine.

        Parameters
        ----------
        coarse_x : Tensor
            Coarse features ``(n_coarse, F)``.
        perm : Tensor
            Fine-level indices for each coarse row.
        num_fine : int
            Fine node count.

        Returns
        -------
        Tensor
            Fine features ``(num_fine, F)``.
        """
        fine = coarse_x.new_zeros(num_fine, coarse_x.size(-1))
        fine[perm] = coarse_x
        if self.refine is not None:
            fine = self.refine(fine)
        return fine


def pool_control(control: Tensor, perms: list[Tensor]) -> Tensor:
    """Map a control tensor through a chain of pooling permutations.

    Parameters
    ----------
    control : Tensor
        Global ``(control_dim,)`` or per-node ``(num_nodes, control_dim)``.
    perms : list of Tensor
        Pooling ``perm`` tensors from fine → coarse (applied in order).

    Returns
    -------
    Tensor
        Global control unchanged, or per-node control indexed by ``perms``.

    Raises
    ------
    ValueError
        If ``control`` rank is neither 1 nor 2.
    """
    if control.ndim == 1:
        return control
    if control.ndim != 2:
        msg = (
            "control must have shape (control_dim,) or "
            f"(num_nodes, control_dim), got {tuple(control.shape)}"
        )
        raise ValueError(msg)
    out = control
    for perm in perms:
        out = out[perm]
    return out


def pool_control_sequence(
    control_inputs: Tensor,
    perms_per_step: list[list[Tensor]],
) -> Tensor:
    """Pool a full sequence control tensor with per-timestep perm chains.

    Parameters
    ----------
    control_inputs : Tensor
        ``(T, C)`` global or ``(T, N, C)`` per-node controls.
    perms_per_step : list of list of Tensor
        Length ``T``; each entry is the fine→coarse ``perm`` chain for that
        snapshot.

    Returns
    -------
    Tensor
        Pooled controls with matching global or per-node layout.
    """
    if control_inputs.ndim == 2:
        return control_inputs
    if control_inputs.ndim != 3:
        msg = (
            "control_inputs must have shape (T, C) or (T, N, C), "
            f"got {tuple(control_inputs.shape)}"
        )
        raise ValueError(msg)
    if len(perms_per_step) != control_inputs.shape[0]:
        msg = (
            f"expected {control_inputs.shape[0]} perm chains, got {len(perms_per_step)}"
        )
        raise ValueError(msg)
    pooled = [
        pool_control(control_inputs[t], perms_per_step[t])
        for t in range(control_inputs.shape[0])
    ]
    return torch.stack(pooled, dim=0)


def snapshot_from_features(
    features: Tensor,
    edge_index: Tensor,
    edge_weight: Tensor | None = None,
) -> Data:
    """Build a ``Data`` snapshot from features and topology.

    Parameters
    ----------
    features : Tensor
        Node features.
    edge_index : Tensor
        Topology.
    edge_weight : Tensor or None, optional
        Optional edge weights.

    Returns
    -------
    Data
        Snapshot with ``x`` / ``edge_index`` / optional ``edge_weight``.
    """
    fields: dict[str, Tensor] = {"x": features, "edge_index": edge_index}
    if edge_weight is not None:
        fields["edge_weight"] = edge_weight
    return Data(**fields)


def filter_subgraph(
    edge_index: Tensor,
    node_mask: Tensor,
    edge_weight: Tensor | None = None,
) -> tuple[Tensor, Tensor | None]:
    """Induce a subgraph on selected nodes (utility for tests).

    Parameters
    ----------
    edge_index : Tensor
        Full topology.
    node_mask : Tensor
        Boolean mask over fine nodes.
    edge_weight : Tensor or None, optional
        Optional weights.

    Returns
    -------
    tuple
        Relabeled ``edge_index`` and optional weights.
    """
    edge_c, edge_w = subgraph(
        node_mask,
        edge_index,
        edge_attr=edge_weight,
        relabel_nodes=True,
        num_nodes=int(node_mask.numel()),
    )
    return edge_c, edge_w


def resolve_snapshot_inputs(
    graph: Tensor | Data,
    edge_index: Tensor | None = None,
    edge_weight: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Resolve node features and topology from tensor or ``Data`` input.

    Parameters
    ----------
    graph : Tensor or Data
        Node features or a PyG snapshot.
    edge_index : Tensor or None, optional
        Required when ``graph`` is a tensor.
    edge_weight : Tensor or None, optional
        Optional weights for tensor input.

    Returns
    -------
    tuple
        ``(x, edge_index, edge_weight)``.

    Raises
    ------
    ValueError
        If topology is missing for tensor input.
    """
    if isinstance(graph, Data):
        if graph.x is None:
            msg = "graph snapshot must define node features x"
            raise ValueError(msg)
        return graph.x, graph.edge_index, snapshot_edge_weight(graph)
    if edge_index is None:
        msg = "edge_index is required when graph is a tensor"
        raise ValueError(msg)
    return graph, edge_index, edge_weight
