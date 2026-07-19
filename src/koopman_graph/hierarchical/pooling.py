"""Pooling / unpooling helpers for hierarchical GraphKoopman forecasting.

Uses PyG :class:`~torch_geometric.nn.TopKPooling` (default) or
:class:`~torch_geometric.nn.SAGPooling`. Global controls pass through unchanged;
per-node controls are indexed by the pooling ``perm`` chain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.nn import SAGPooling, TopKPooling
from torch_geometric.utils import subgraph

from koopman_graph.graph_utils import snapshot_edge_weight

PoolingKind = Literal["topk", "sag"]


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
        Coarse edge index after this level.
    edge_weight : Tensor or None
        Optional coarse edge weights.
    """

    perm: Tensor
    num_fine: int
    edge_index: Tensor
    edge_weight: Tensor | None


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

    Returns
    -------
    bool
        Whether the ratio requests full node retention.
    """
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
            f"expected {control_inputs.shape[0]} perm chains, "
            f"got {len(perms_per_step)}"
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
