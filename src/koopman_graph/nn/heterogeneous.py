"""Relational (R-GCN-lite) and optional HGT encoder / decoder peers.

Encoder and decoder import shared activation / validation helpers from
:mod:`koopman_graph.nn.gnn` only; they do not import each other. Checkpoint
type strings ``relgraph_enc`` / ``relgraph_dec`` are registered in the
format-1 schema. Optional :class:`HGTEncoder` / :class:`HGTDecoder` peers are
``koopman_graph.nn`` power-user exports only (not factory defaults; not root
``__all__``).

Per-relation linear messages use destination in-degree normalization from
:func:`~koopman_graph.graph_utils.relation_degree_normalize` (Schlichtkrull
et al., R-GCN — normalization / per-relation message motivation only; not a
full paper reproduction). Reverse relations are not synthesized.

Optional HGT peers wrap PyTorch Geometric
:class:`~torch_geometric.nn.HGTConv` (Hu et al., Heterogeneous Graph
Transformer — attention / type-aware message motivation only; not a full
paper reproduction). They are **typed-only** and are **not** required for
hetero Koopman support; :class:`RelGraphEncoder` /
:class:`RelGraphDecoder` remain the factory-supported pair.

Multiplex vs typed
------------------
Integer ``in_channels`` / ``out_channels`` select the **multiplex** path: one
node type, a plain ``(num_nodes, …)`` tensor in and out.

A mapping ``node_type -> width`` selects the **typed** path with an explicit
``node_types`` order. Typed RelGraph encoders apply a per-type input
:class:`~torch.nn.Linear` into the shared hidden width, stack the projections
into one ``(N, hidden_channels)`` block ordered by ``node_types``, and run the
shared :class:`RelGraphConv` stack on relation banks lifted into stacked global
node numbering (see :mod:`koopman_graph.data.hetero_layout`). Typed RelGraph
decoders mirror that: shared convolutions down to ``hidden_channels``, then
per-type output heads, returning ``dict[str, Tensor]``. The latent width
``d`` is shared by every node type. HGT peers use the same stacked latent
contract and type order, with type-local PyG edge indices inside the HGT
stack.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv

from koopman_graph.data.hetero_layout import (
    global_relation_edge_indices,
    node_type_slices,
    snapshot_num_nodes_dict,
)
from koopman_graph.graph_utils.topology import (
    RELATION_NORMALIZATION_MODES,
    RelationNormalization,
    relation_normalized_adjacency_matvec,
)
from koopman_graph.nn.gnn import (
    ActivationName,
    BaseGNNModule,
    validate_positive_dims,
)

EdgeIndexBank = Sequence[Tensor] | Mapping[object, Tensor]
EdgeWeightBank = Sequence[Tensor | None] | Mapping[object, Tensor | None]
EdgeTypeTriple = tuple[str, str, str]
TypedChannels = int | Mapping[str, int]


def _validate_normalization(normalization: RelationNormalization) -> None:
    """Reject unsupported relation-normalization identifiers.

    Parameters
    ----------
    normalization : {"rgcn_in_degree", "random_walk"}
        Relation normalization mode.

    Raises
    ------
    ValueError
        If ``normalization`` is not a supported mode.
    """
    if normalization not in RELATION_NORMALIZATION_MODES:
        msg = (
            "normalization must be one of "
            f"{sorted(RELATION_NORMALIZATION_MODES)}, got {normalization!r}"
        )
        raise ValueError(msg)


def build_relgraph_convs(
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    num_layers: int,
    num_relations: int,
    *,
    normalization: RelationNormalization = "rgcn_in_degree",
    root_weight: bool = True,
) -> nn.ModuleList:
    """Build a stack of :class:`RelGraphConv` layers.

    Parameters
    ----------
    in_channels : int
        Input feature dimension for the first layer.
    hidden_channels : int
        Hidden width for intermediate layers.
    out_channels : int
        Output feature dimension for the final layer.
    num_layers : int
        Number of relational convolution layers.
    num_relations : int
        Number of relation banks (shared across layers).
    normalization : {"rgcn_in_degree", "random_walk"}, optional
        Per-relation degree normalization. Default is ``"rgcn_in_degree"``.
    root_weight : bool, optional
        If ``True``, each layer includes a self / root linear map.
        Default is ``True``.

    Returns
    -------
    nn.ModuleList
        Ordered :class:`RelGraphConv` modules.
    """
    convs: list[RelGraphConv] = []
    for layer_idx in range(num_layers):
        layer_in = in_channels if layer_idx == 0 else hidden_channels
        layer_out = out_channels if layer_idx == num_layers - 1 else hidden_channels
        convs.append(
            RelGraphConv(
                layer_in,
                layer_out,
                num_relations,
                normalization=normalization,
                root_weight=root_weight,
            )
        )
    return nn.ModuleList(convs)


class RelGraphConv(nn.Module):
    """One relational message-passing layer (R-GCN-lite).

    For each relation ``r`` applies a linear map ``W_r`` to node features,
    then aggregates with
    :func:`~koopman_graph.graph_utils.relation_normalized_adjacency_matvec`.
    An optional root transform ``W_0`` (self term) matches the classic R-GCN
    construction. This is not a full Schlichtkrull et al. reproduction
    (no basis decomposition / block-diagonal tying here).

    Attributes
    ----------
    in_channels : int
        Input feature dimension.
    out_channels : int
        Output feature dimension.
    num_relations : int
        Number of relation-specific weight banks.
    normalization : {"rgcn_in_degree", "random_walk"}
        Per-relation degree normalization mode.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_relations: int,
        *,
        normalization: RelationNormalization = "rgcn_in_degree",
        root_weight: bool = True,
        bias: bool = True,
    ) -> None:
        """Initialize per-relation and optional root transforms.

        Parameters
        ----------
        in_channels : int
            Input feature dimension.
        out_channels : int
            Output feature dimension.
        num_relations : int
            Number of relation banks (``|R| >= 1``).
        normalization : {"rgcn_in_degree", "random_walk"}, optional
            Per-relation degree normalization. Default is ``"rgcn_in_degree"``.
        root_weight : bool, optional
            If ``True``, include a self / root linear map. Default is ``True``.
        bias : bool, optional
            If ``True``, add a learnable output bias. Default is ``True``.

        Raises
        ------
        ValueError
            If dimensions are not positive or ``normalization`` is unsupported.
        """
        super().__init__()
        validate_positive_dims(
            in_channels=in_channels,
            out_channels=out_channels,
            num_relations=num_relations,
        )
        _validate_normalization(normalization)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.normalization: RelationNormalization = normalization
        self.root_weight = root_weight

        self.weight = nn.Parameter(
            torch.empty(num_relations, in_channels, out_channels)
        )
        if root_weight:
            self.root = nn.Linear(in_channels, out_channels, bias=False)
        else:
            self.root = None
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize relation banks, root weights, and bias.

        Returns
        -------
        None
        """
        for relation_idx in range(self.num_relations):
            nn.init.xavier_uniform_(self.weight[relation_idx])
        if self.root is not None:
            self.root.reset_parameters()
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        x: Tensor,
        edge_indices: Sequence[Tensor],
        edge_weights: Sequence[Tensor | None] | None = None,
    ) -> Tensor:
        """Apply relational aggregation for one layer.

        Parameters
        ----------
        x : Tensor
            Node features with shape ``(num_nodes, in_channels)``.
        edge_indices : sequence of Tensor
            Per-relation edge indices, each ``(2, num_edges_r)``, length
            ``num_relations``.
        edge_weights : sequence of Tensor or None, optional
            Optional per-relation edge weights. Defaults to unweighted ones.

        Returns
        -------
        Tensor
            Aggregated features with shape ``(num_nodes, out_channels)``.

        Raises
        ------
        ValueError
            If ``x`` rank/channels are wrong or the bank length mismatches
            ``num_relations``.
        """
        if x.ndim != 2:
            msg = (
                f"Expected x with shape (num_nodes, {self.in_channels}), "
                f"got {tuple(x.shape)}"
            )
            raise ValueError(msg)
        if x.shape[1] != self.in_channels:
            msg = (
                f"Expected in_channels={self.in_channels}, "
                f"got feature dimension {x.shape[1]}"
            )
            raise ValueError(msg)
        if len(edge_indices) != self.num_relations:
            msg = (
                f"Expected {self.num_relations} relation edge banks, "
                f"got {len(edge_indices)}"
            )
            raise ValueError(msg)
        if edge_weights is None:
            resolved_weights: list[Tensor | None] = [None] * self.num_relations
        else:
            if len(edge_weights) != self.num_relations:
                msg = (
                    f"Expected {self.num_relations} relation weight banks, "
                    f"got {len(edge_weights)}"
                )
                raise ValueError(msg)
            resolved_weights = list(edge_weights)

        if self.root is not None:
            out = self.root(x)
        else:
            out = x.new_zeros((x.size(0), self.out_channels))

        for relation_idx, edge_index in enumerate(edge_indices):
            transformed = x @ self.weight[relation_idx]
            out = out + relation_normalized_adjacency_matvec(
                edge_index,
                transformed,
                edge_weight=resolved_weights[relation_idx],
                num_nodes=x.size(0),
                normalization=self.normalization,
            )

        if self.bias is not None:
            out = out + self.bias
        return out


def _as_ordered_banks(
    bank: EdgeIndexBank | EdgeWeightBank,
    *,
    num_relations: int,
    kind: str,
) -> list:
    """Normalize a sequence or mapping of relation banks to an ordered list.

    Parameters
    ----------
    bank : sequence or mapping
        Relation-indexed tensors (or optional weights).
    num_relations : int
        Expected number of relations.
    kind : str
        Label used in error messages (``"edge_index"`` or ``"edge_weight"``).

    Returns
    -------
    list
        Ordered bank values of length ``num_relations``.

    Raises
    ------
    ValueError
        If the bank length does not match ``num_relations``.
    """
    if isinstance(bank, Mapping):
        keys = sorted(bank.keys(), key=repr)
        values = [bank[key] for key in keys]
    else:
        values = list(bank)
    if len(values) != num_relations:
        msg = f"Expected {num_relations} {kind} relation banks, got {len(values)}"
        raise ValueError(msg)
    return values


def resolve_multiplex_relation_inputs(
    x_or_data: Tensor | HeteroData,
    edge_index: EdgeIndexBank | None = None,
    edge_weight: EdgeWeightBank | None = None,
    *,
    num_relations: int,
) -> tuple[Tensor, list[Tensor], list[Tensor | None]]:
    """Resolve multiplex node features and ordered relation banks.

    When ``x_or_data`` is :class:`~torch_geometric.data.HeteroData`, the
    snapshot must be multiplex (exactly one node type). Relation banks are
    ordered by sorted edge-type keys (``repr``). Positional topology
    arguments are ignored for ``HeteroData`` input.

    Parameters
    ----------
    x_or_data : Tensor or HeteroData
        Node features or a multiplex hetero snapshot.
    edge_index : sequence/mapping of Tensor or None
        Per-relation edge indices for tensor input; ignored for HeteroData.
    edge_weight : sequence/mapping of Tensor or None, optional
        Optional per-relation weights for tensor input; ignored for HeteroData.
    num_relations : int
        Expected ``|R|`` matching the module construction.

    Returns
    -------
    tuple of (Tensor, list of Tensor, list of Tensor or None)
        Features, ordered edge indices, and optional edge weights.

    Raises
    ------
    ValueError
        If the snapshot is not multiplex, relation counts mismatch, or
        tensor input omits edge banks.
    TypeError
        If ``x_or_data`` is neither a tensor nor ``HeteroData``.
    """
    if isinstance(x_or_data, HeteroData):
        node_types = tuple(x_or_data.node_types)
        if len(node_types) != 1:
            msg = (
                "RelGraphEncoder / RelGraphDecoder accept multiplex "
                f"HeteroData with exactly one node type; got {len(node_types)} "
                f"types {node_types!r}"
            )
            raise ValueError(msg)
        node_type = node_types[0]
        x = x_or_data[node_type].x
        if x is None:
            msg = f"HeteroData node type {node_type!r} is missing feature matrix x"
            raise ValueError(msg)
        edge_types = tuple(sorted(x_or_data.edge_types, key=repr))
        if len(edge_types) != num_relations:
            msg = (
                f"Expected {num_relations} edge types on HeteroData, "
                f"got {len(edge_types)}: {edge_types!r}"
            )
            raise ValueError(msg)
        edge_indices = [x_or_data[edge_type].edge_index for edge_type in edge_types]
        edge_weights: list[Tensor | None] = []
        for edge_type in edge_types:
            store = x_or_data[edge_type]
            edge_weights.append(store.get("edge_weight", None))
        return x, edge_indices, edge_weights

    if not isinstance(x_or_data, Tensor):
        msg = (
            "RelGraphEncoder / RelGraphDecoder expect Tensor or HeteroData "
            f"input, got {type(x_or_data).__name__}"
        )
        raise TypeError(msg)
    if edge_index is None:
        msg = (
            "edge_index relation banks are required when x_or_data is a tensor "
            "for RelGraphEncoder / RelGraphDecoder"
        )
        raise ValueError(msg)
    edge_indices = _as_ordered_banks(
        edge_index,
        num_relations=num_relations,
        kind="edge_index",
    )
    if edge_weight is None:
        edge_weights = [None] * num_relations
    else:
        edge_weights = _as_ordered_banks(
            edge_weight,
            num_relations=num_relations,
            kind="edge_weight",
        )
    return x_or_data, edge_indices, edge_weights


def _normalize_typed_channels(
    channels: TypedChannels,
    node_types: Sequence[str] | None,
    *,
    argument: str,
) -> tuple[dict[str, int], tuple[str, ...]]:
    """Validate a typed ``node_type -> width`` mapping against ``node_types``.

    Parameters
    ----------
    channels : mapping of str to int
        Per-type feature width.
    node_types : sequence of str or None
        Explicit stacking order. Required for the typed path.
    argument : str
        Argument name used in error messages (``"in_channels"`` /
        ``"out_channels"``).

    Returns
    -------
    tuple of (dict of str to int, tuple of str)
        Validated widths and the ordered node-type names.

    Raises
    ------
    ValueError
        If ``node_types`` is missing, does not match the mapping keys, or a
        width is not positive.
    """
    assert isinstance(channels, Mapping)
    if node_types is None:
        msg = (
            f"node_types is required when {argument} is a mapping so the "
            "stacked type order is explicit; got "
            f"{argument} keys {sorted(channels)!r}"
        )
        raise ValueError(msg)
    resolved_types = tuple(str(name) for name in node_types)
    if len(set(resolved_types)) != len(resolved_types):
        msg = f"node_types must be unique; got {resolved_types!r}"
        raise ValueError(msg)
    if set(resolved_types) != set(channels):
        msg = (
            f"{argument} keys must match node_types "
            f"{list(resolved_types)!r}; got {sorted(channels)!r}"
        )
        raise ValueError(msg)
    widths = {name: int(channels[name]) for name in resolved_types}
    validate_positive_dims(**{f"{argument}[{name!r}]": widths[name] for name in widths})
    return widths, resolved_types


def _normalize_edge_type_order(
    edge_types: Sequence[Sequence[str]] | None,
    *,
    num_relations: int,
    required: bool = False,
) -> tuple[EdgeTypeTriple, ...] | None:
    """Validate an explicit ``(src, rel, dst)`` relation-bank order.

    Parameters
    ----------
    edge_types : sequence of sequence of str or None
        Ordered triples, or ``None`` to fall back to sorted snapshot order.
    num_relations : int
        Expected bank length ``|R|``.
    required : bool, optional
        When ``True``, ``None`` is rejected. Default is ``False``.

    Returns
    -------
    tuple of tuple of str or None
        Ordered triples, or ``None`` when no explicit order was supplied.

    Raises
    ------
    ValueError
        If a triple is malformed, empty, duplicated, the count mismatches
        ``num_relations``, or ``required`` and no order was supplied.
    """
    if edge_types is None:
        if required:
            msg = (
                "edge_types is required for typed relational modules so the "
                "relation-bank order matches the paired operator"
            )
            raise ValueError(msg)
        return None
    resolved: list[EdgeTypeTriple] = []
    for entry in edge_types:
        triple = tuple(str(part) for part in entry)
        if len(triple) != 3:
            msg = (
                "edge_types entries must be (src, rel, dst) triples; "
                f"got {tuple(entry)!r}"
            )
            raise ValueError(msg)
        if any(not part for part in triple):
            msg = "edge_types entries must use non-empty strings"
            raise ValueError(msg)
        resolved.append((triple[0], triple[1], triple[2]))
    if len(resolved) != num_relations:
        msg = (
            f"edge_types length ({len(resolved)}) must match "
            f"num_relations ({num_relations})"
        )
        raise ValueError(msg)
    if len(set(resolved)) != len(resolved):
        msg = f"edge_types must be unique; got {resolved!r}"
        raise ValueError(msg)
    return tuple(resolved)


def resolve_typed_relation_inputs(
    x_or_data: HeteroData | Mapping[str, Tensor],
    edge_index: EdgeIndexBank | None = None,
    edge_weight: EdgeWeightBank | None = None,
    *,
    node_types: Sequence[str],
    edge_types: Sequence[Sequence[str]] | None = None,
    num_relations: int,
) -> tuple[dict[str, Tensor], list[Tensor], list[Tensor | None], dict[str, int]]:
    """Resolve typed per-type features and stacked global relation banks.

    Parameters
    ----------
    x_or_data : HeteroData or mapping of str to Tensor
        Typed hetero snapshot, or per-type feature tensors. For the mapping
        form, ``edge_index`` must already use stacked global node numbering.
    edge_index : sequence/mapping of Tensor or None, optional
        Per-relation edge indices for mapping input; ignored for
        ``HeteroData``.
    edge_weight : sequence/mapping of Tensor or None, optional
        Optional per-relation weights for mapping input; ignored for
        ``HeteroData``.
    node_types : sequence of str
        Ordered node-type names defining the stacking order.
    edge_types : sequence of sequence of str or None, optional
        Explicit relation-bank order. When ``None``, ``HeteroData`` edge types
        are ordered by sorted ``repr``.
    num_relations : int
        Expected ``|R|`` matching the module construction.

    Returns
    -------
    tuple
        ``(feature_dict, edge_indices, edge_weights, num_nodes_dict)`` with
        relation banks in stacked global numbering.

    Raises
    ------
    ValueError
        If a node type or edge type is missing, relation counts mismatch, or
        mapping input omits ``edge_index``.
    TypeError
        If ``x_or_data`` is neither ``HeteroData`` nor a mapping of tensors.
    """
    resolved_types = tuple(str(name) for name in node_types)
    if isinstance(x_or_data, HeteroData):
        num_nodes_dict = snapshot_num_nodes_dict(x_or_data, resolved_types)
        present = tuple(x_or_data.node_types)
        if len(present) != len(resolved_types):
            msg = (
                f"HeteroData has node types {sorted(present)!r} but the typed "
                f"module expects exactly {list(resolved_types)!r}"
            )
            raise ValueError(msg)
        ordered_edge_types = _normalize_edge_type_order(
            edge_types,
            num_relations=num_relations,
        )
        if ordered_edge_types is None:
            ordered_edge_types = tuple(sorted(x_or_data.edge_types, key=repr))
            if len(ordered_edge_types) != num_relations:
                msg = (
                    f"Expected {num_relations} edge types on HeteroData, "
                    f"got {len(ordered_edge_types)}: {ordered_edge_types!r}"
                )
                raise ValueError(msg)
        features = {name: x_or_data[name].x for name in resolved_types}
        edge_indices = global_relation_edge_indices(
            x_or_data,
            ordered_edge_types,
            resolved_types,
            num_nodes_dict,
        )
        edge_weights: list[Tensor | None] = [
            x_or_data[edge_type].get("edge_weight", None)
            for edge_type in ordered_edge_types
        ]
        return features, edge_indices, edge_weights, num_nodes_dict

    if not isinstance(x_or_data, Mapping):
        msg = (
            "typed RelGraphEncoder / RelGraphDecoder expect HeteroData or a "
            f"mapping of per-type tensors, got {type(x_or_data).__name__}"
        )
        raise TypeError(msg)
    features = {}
    num_nodes_dict = {}
    for name in resolved_types:
        if name not in x_or_data:
            msg = (
                f"typed feature mapping is missing node type {name!r}; "
                f"expected keys {list(resolved_types)!r}, "
                f"got {sorted(x_or_data)!r}"
            )
            raise ValueError(msg)
        block = x_or_data[name]
        if block.ndim != 2:
            msg = (
                f"typed feature mapping[{name!r}] must have shape "
                f"(num_nodes, width), got {tuple(block.shape)}"
            )
            raise ValueError(msg)
        features[name] = block
        num_nodes_dict[name] = int(block.shape[0])
    if edge_index is None:
        msg = (
            "edge_index relation banks are required when typed input is a "
            "feature mapping (banks must already use stacked global numbering)"
        )
        raise ValueError(msg)
    edge_indices = _as_ordered_banks(
        edge_index,
        num_relations=num_relations,
        kind="edge_index",
    )
    if edge_weight is None:
        edge_weights = [None] * num_relations
    else:
        edge_weights = _as_ordered_banks(
            edge_weight,
            num_relations=num_relations,
            kind="edge_weight",
        )
    return features, edge_indices, edge_weights, num_nodes_dict


def _resolve_relgraph_forward_inputs(
    x_or_data: Tensor | HeteroData,
    edge_index: EdgeIndexBank | None,
    edge_weight: EdgeWeightBank | None,
    *,
    num_relations: int,
) -> tuple[Tensor, list[Tensor], list[Tensor | None]]:
    """Resolve multiplex inputs (internal alias of the public helper).

    Parameters
    ----------
    x_or_data : Tensor or HeteroData
        Node features or a multiplex hetero snapshot.
    edge_index : sequence/mapping of Tensor or None
        Per-relation edge indices for tensor input; ignored for HeteroData.
    edge_weight : sequence/mapping of Tensor or None, optional
        Optional per-relation weights for tensor input; ignored for HeteroData.
    num_relations : int
        Expected ``|R|`` matching the module construction.

    Returns
    -------
    tuple of (Tensor, list of Tensor, list of Tensor or None)
        Features, ordered edge indices, and optional edge weights.
    """
    return resolve_multiplex_relation_inputs(
        x_or_data,
        edge_index,
        edge_weight,
        num_relations=num_relations,
    )


def _relgraph_message_passing(
    module: BaseGNNModule,
    x: Tensor,
    edge_indices: Sequence[Tensor],
    edge_weights: Sequence[Tensor | None],
) -> Tensor:
    """Run stacked :class:`RelGraphConv` layers with hidden activations.

    Parameters
    ----------
    module : BaseGNNModule
        RelGraph encoder or decoder owning ``convs`` and activation.
    x : Tensor
        Node features with shape ``(num_nodes, channels)``.
    edge_indices : sequence of Tensor
        Ordered per-relation edge indices.
    edge_weights : sequence of Tensor or None
        Ordered optional per-relation edge weights.

    Returns
    -------
    Tensor
        Transformed node features after the stack.

    Raises
    ------
    ValueError
        If ``x`` has the wrong rank or channel dimension.
    TypeError
        If ``module.convs`` contains a non-``RelGraphConv`` layer.
    """
    if x.ndim != 2:
        msg = (
            f"Expected x with shape (num_nodes, {module.input_dim_name}), "
            f"got {tuple(x.shape)}"
        )
        raise ValueError(msg)
    if x.shape[1] != module.input_channels:
        msg = (
            f"Expected {module.input_dim_name}={module.input_channels}, "
            f"got feature dimension {x.shape[1]}"
        )
        raise ValueError(msg)
    for layer_idx, conv in enumerate(module.convs):
        if not isinstance(conv, RelGraphConv):
            msg = f"expected RelGraphConv in relgraph stack, got {type(conv).__name__}"
            raise TypeError(msg)
        x = conv(x, edge_indices, edge_weights)
        if layer_idx < len(module.convs) - 1:
            x = module.activation(x)
    return x


class RelGraphEncoder(BaseGNNModule):
    """Relational encoder lifting node features into shared-width latents.

    Applies stacked :class:`RelGraphConv` layers with a configurable hidden
    activation. The final layer maps to ``latent_dim`` without an activation.
    Output latents are stacked as ``(num_nodes, latent_dim)``: a single node
    type for the multiplex path, ``N = Σ_τ N_τ`` rows ordered by
    :attr:`node_types` for the typed path. Degree normalization follows
    Schlichtkrull et al. (R-GCN) in-degree convention via
    :func:`~koopman_graph.graph_utils.relation_degree_normalize` — not a full
    paper reproduction.

    Typed encoders (mapping ``in_channels``) first apply a per-type
    :class:`~torch.nn.Linear` from ``F_τ`` to ``hidden_channels`` followed by
    the hidden activation, then share one convolution stack across all types.

    Attributes
    ----------
    in_channels : int or dict of str to int
        Input node feature dimension (per type when typed).
    in_channels_dict : dict of str to int or None
        Per-type input widths for the typed path, else ``None``.
    hidden_channels : int
        Hidden relational channel width.
    latent_dim : int
        Output latent dimension per node (shared by all node types).
    num_relations : int
        Number of relation banks.
    node_types : tuple of str or None
        Ordered node-type names for the typed path, else ``None``.
    is_typed : bool
        ``True`` when ``in_channels`` was a mapping.
    edge_types : tuple of tuple of str or None
        Explicit relation-bank order, else ``None`` (sorted snapshot order).
    normalization : {"rgcn_in_degree", "random_walk"}
        Per-relation degree normalization mode.
    root_weight : bool
        Whether each layer includes a self / root linear map.
    """

    def __init__(
        self,
        in_channels: TypedChannels,
        hidden_channels: int,
        latent_dim: int,
        num_relations: int,
        *,
        num_layers: int = 2,
        activation: ActivationName = "relu",
        normalization: RelationNormalization = "rgcn_in_degree",
        root_weight: bool = True,
        node_types: Sequence[str] | None = None,
        edge_types: Sequence[Sequence[str]] | None = None,
    ) -> None:
        """Initialize the relational encoder stack.

        Parameters
        ----------
        in_channels : int or mapping of str to int
            Input node feature dimension. An integer selects the multiplex
            path; a mapping ``node_type -> F_τ`` selects the typed path and
            requires ``node_types``.
        hidden_channels : int
            Hidden channel width for intermediate layers (and for the typed
            per-type input projections).
        latent_dim : int
            Output latent dimension per node.
        num_relations : int
            Number of relation banks (``|R| >= 1``).
        num_layers : int, optional
            Number of relational convolution layers. Default is ``2``.
        activation : {"relu", "sigmoid", "tanh"}, optional
            Hidden-layer activation. Default is ``"relu"``.
        normalization : {"rgcn_in_degree", "random_walk"}, optional
            Per-relation degree normalization. Default is ``"rgcn_in_degree"``.
        root_weight : bool, optional
            If ``True``, each layer includes a self / root linear map.
            Default is ``True``.
        node_types : sequence of str or None, optional
            Ordered node-type names (stacking order). Required when
            ``in_channels`` is a mapping; must match its keys.
        edge_types : sequence of sequence of str or None, optional
            Explicit ordered ``(src, rel, dst)`` triples defining relation-bank
            order; **required** for the typed path and must match the paired
            operator's ``edge_types``. When omitted on the multiplex path,
            ``HeteroData`` edge types are ordered by sorted ``repr``.

        Raises
        ------
        ValueError
            If any dimension argument is not positive, ``normalization`` is
            unsupported, or typed metadata is missing / inconsistent.
        """
        typed = isinstance(in_channels, Mapping)
        if typed:
            in_channels_dict, resolved_node_types = _normalize_typed_channels(
                in_channels,
                node_types,
                argument="in_channels",
            )
            first_layer_in = hidden_channels
        else:
            assert isinstance(in_channels, int)
            in_channels_dict = None
            resolved_node_types = (
                None if node_types is None else tuple(str(n) for n in node_types)
            )
            validate_positive_dims(in_channels=in_channels)
            first_layer_in = int(in_channels)
        validate_positive_dims(
            hidden_channels=hidden_channels,
            latent_dim=latent_dim,
            num_relations=num_relations,
            num_layers=num_layers,
        )
        _validate_normalization(normalization)
        resolved_edge_types = _normalize_edge_type_order(
            edge_types,
            num_relations=num_relations,
            required=typed,
        )

        self.in_channels = dict(in_channels) if typed else in_channels
        self.in_channels_dict = in_channels_dict
        self.hidden_channels = hidden_channels
        self.latent_dim = latent_dim
        self.num_relations = num_relations
        self.is_typed = typed
        self.node_types = resolved_node_types
        self.edge_types = resolved_edge_types
        self.normalization: RelationNormalization = normalization
        self.root_weight = root_weight

        super().__init__(
            input_channels=first_layer_in,
            input_dim_name="hidden_channels" if typed else "in_channels",
            num_layers=num_layers,
            activation=activation,
            convs=build_relgraph_convs(
                first_layer_in,
                hidden_channels,
                latent_dim,
                num_layers,
                num_relations,
                normalization=normalization,
                root_weight=root_weight,
            ),
        )
        if in_channels_dict is not None:
            self.type_input = nn.ModuleDict(
                {
                    name: nn.Linear(width, hidden_channels)
                    for name, width in in_channels_dict.items()
                }
            )

    def forward(
        self,
        x_or_data: Tensor | HeteroData | Mapping[str, Tensor],
        edge_index: EdgeIndexBank | None = None,
        edge_weight: EdgeWeightBank | None = None,
    ) -> Tensor:
        """Encode node features with relational message passing.

        Parameters
        ----------
        x_or_data : Tensor, HeteroData, or mapping of str to Tensor
            Node features or a ``HeteroData`` snapshot. Typed encoders accept
            ``HeteroData`` or a per-type feature mapping; multiplex encoders
            accept a tensor or a single-node-type ``HeteroData``.
        edge_index : sequence/mapping of Tensor or None, optional
            Per-relation edge indices when features are passed directly.
            Ignored for ``HeteroData``. Typed banks must already use stacked
            global node numbering.
        edge_weight : sequence/mapping of Tensor or None, optional
            Optional per-relation weights; ignored for ``HeteroData``.

        Returns
        -------
        Tensor
            Latent node features with shape ``(num_nodes, latent_dim)``, rows
            ordered by :attr:`node_types` for the typed path.
        """
        if self.is_typed:
            assert self.node_types is not None
            features, edge_indices, edge_weights, _ = resolve_typed_relation_inputs(
                x_or_data,  # type: ignore[arg-type]
                edge_index,
                edge_weight,
                node_types=self.node_types,
                edge_types=self.edge_types,
                num_relations=self.num_relations,
            )
            projected = [
                self.activation(self.type_input[name](features[name]))
                for name in self.node_types
            ]
            x = torch.cat(projected, dim=0)
            return _relgraph_message_passing(self, x, edge_indices, edge_weights)

        x, edge_indices, edge_weights = _resolve_relgraph_forward_inputs(
            x_or_data,  # type: ignore[arg-type]
            edge_index,
            edge_weight,
            num_relations=self.num_relations,
        )
        return _relgraph_message_passing(self, x, edge_indices, edge_weights)


class RelGraphDecoder(BaseGNNModule):
    """Relational decoder mapping shared-width latents to physical features.

    Applies stacked :class:`RelGraphConv` layers with a configurable hidden
    activation. Normalization citations match :class:`RelGraphEncoder`.

    For the multiplex path the final convolution maps to ``out_channels``
    without an activation and :meth:`forward` returns a ``Tensor``. For the
    typed path (mapping ``out_channels``) the convolution stack ends at
    ``hidden_channels``, the hidden activation is applied, and per-type
    :class:`~torch.nn.Linear` heads produce one ``(N_τ, F_τ)`` tensor per node
    type; :meth:`forward` then returns ``dict[str, Tensor]``.

    Attributes
    ----------
    latent_dim : int
        Input latent dimension per node (shared by all node types).
    hidden_channels : int
        Hidden relational channel width.
    out_channels : int or dict of str to int
        Output physical feature dimension per node (per type when typed).
    out_channels_dict : dict of str to int or None
        Per-type output widths for the typed path, else ``None``.
    num_relations : int
        Number of relation banks.
    node_types : tuple of str or None
        Ordered node-type names for the typed path, else ``None``.
    is_typed : bool
        ``True`` when ``out_channels`` was a mapping.
    edge_types : tuple of tuple of str or None
        Explicit relation-bank order, else ``None``.
    normalization : {"rgcn_in_degree", "random_walk"}
        Per-relation degree normalization mode.
    root_weight : bool
        Whether each layer includes a self / root linear map.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_channels: int,
        out_channels: TypedChannels,
        num_relations: int,
        *,
        num_layers: int = 2,
        activation: ActivationName = "relu",
        normalization: RelationNormalization = "rgcn_in_degree",
        root_weight: bool = True,
        node_types: Sequence[str] | None = None,
        edge_types: Sequence[Sequence[str]] | None = None,
    ) -> None:
        """Initialize the relational decoder stack.

        Parameters
        ----------
        latent_dim : int
            Input latent dimension per node.
        hidden_channels : int
            Hidden channel width for intermediate layers (and the width fed to
            typed per-type output heads).
        out_channels : int or mapping of str to int
            Output physical feature dimension. An integer selects the
            multiplex path; a mapping ``node_type -> F_τ`` selects the typed
            path and requires ``node_types``.
        num_relations : int
            Number of relation banks (``|R| >= 1``).
        num_layers : int, optional
            Number of relational convolution layers. Default is ``2``.
        activation : {"relu", "sigmoid", "tanh"}, optional
            Hidden-layer activation. Default is ``"relu"``.
        normalization : {"rgcn_in_degree", "random_walk"}, optional
            Per-relation degree normalization. Default is ``"rgcn_in_degree"``.
        root_weight : bool, optional
            If ``True``, each layer includes a self / root linear map.
            Default is ``True``.
        node_types : sequence of str or None, optional
            Ordered node-type names (stacking order). Required when
            ``out_channels`` is a mapping; must match its keys.
        edge_types : sequence of sequence of str or None, optional
            Explicit ordered ``(src, rel, dst)`` triples defining relation-bank
            order; **required** for the typed path. Match the paired encoder /
            operator.

        Raises
        ------
        ValueError
            If any dimension argument is not positive, ``normalization`` is
            unsupported, or typed metadata is missing / inconsistent.
        """
        typed = isinstance(out_channels, Mapping)
        if typed:
            out_channels_dict, resolved_node_types = _normalize_typed_channels(
                out_channels,
                node_types,
                argument="out_channels",
            )
            final_layer_out = hidden_channels
        else:
            assert isinstance(out_channels, int)
            out_channels_dict = None
            resolved_node_types = (
                None if node_types is None else tuple(str(n) for n in node_types)
            )
            validate_positive_dims(out_channels=out_channels)
            final_layer_out = int(out_channels)
        validate_positive_dims(
            latent_dim=latent_dim,
            hidden_channels=hidden_channels,
            num_relations=num_relations,
            num_layers=num_layers,
        )
        _validate_normalization(normalization)
        resolved_edge_types = _normalize_edge_type_order(
            edge_types,
            num_relations=num_relations,
            required=typed,
        )

        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels
        self.out_channels = dict(out_channels) if typed else out_channels
        self.out_channels_dict = out_channels_dict
        self.num_relations = num_relations
        self.is_typed = typed
        self.node_types = resolved_node_types
        self.edge_types = resolved_edge_types
        self.normalization: RelationNormalization = normalization
        self.root_weight = root_weight

        super().__init__(
            input_channels=latent_dim,
            input_dim_name="latent_dim",
            num_layers=num_layers,
            activation=activation,
            convs=build_relgraph_convs(
                latent_dim,
                hidden_channels,
                final_layer_out,
                num_layers,
                num_relations,
                normalization=normalization,
                root_weight=root_weight,
            ),
        )
        if out_channels_dict is not None:
            self.type_output = nn.ModuleDict(
                {
                    name: nn.Linear(hidden_channels, width)
                    for name, width in out_channels_dict.items()
                }
            )

    def forward(
        self,
        x_or_data: Tensor | HeteroData,
        edge_index: EdgeIndexBank | None = None,
        edge_weight: EdgeWeightBank | None = None,
        *,
        num_nodes_dict: Mapping[str, int] | None = None,
    ) -> Tensor | dict[str, Tensor]:
        """Decode latents with relational message passing.

        Parameters
        ----------
        x_or_data : Tensor or HeteroData
            Latent features (stacked ``(num_nodes, latent_dim)``) or a
            multiplex ``HeteroData`` whose node features are treated as the
            decoder input (unusual; prefer tensor + banks).
        edge_index : sequence/mapping of Tensor or None, optional
            Per-relation edge indices when ``x_or_data`` is a tensor. Ignored
            for ``HeteroData``. Typed banks must already use stacked global
            node numbering.
        edge_weight : sequence/mapping of Tensor or None, optional
            Optional per-relation weights for tensor input; ignored for
            ``HeteroData``.
        num_nodes_dict : mapping of str to int or None, optional
            Per-type node counts. Required for the typed path so stacked rows
            can be routed to per-type output heads.

        Returns
        -------
        Tensor or dict of str to Tensor
            Reconstructed node features: ``(num_nodes, out_channels)`` for the
            multiplex path, or one ``(N_τ, F_τ)`` tensor per node type for the
            typed path.

        Raises
        ------
        ValueError
            If the typed path is used without ``num_nodes_dict``.
        """
        x, edge_indices, edge_weights = _resolve_relgraph_forward_inputs(
            x_or_data,
            edge_index,
            edge_weight,
            num_relations=self.num_relations,
        )
        hidden = _relgraph_message_passing(self, x, edge_indices, edge_weights)
        if not self.is_typed:
            return hidden
        assert self.node_types is not None
        if num_nodes_dict is None:
            msg = (
                "num_nodes_dict is required for typed RelGraphDecoder "
                f"(node_types={list(self.node_types)!r}) so stacked latent rows "
                "can be routed to per-type output heads"
            )
            raise ValueError(msg)
        slices = node_type_slices(self.node_types, num_nodes_dict)
        total = sum(int(num_nodes_dict[name]) for name in self.node_types)
        if total != int(hidden.shape[0]):
            msg = (
                f"num_nodes_dict sums to {total} but the stacked latent block "
                f"has {int(hidden.shape[0])} rows"
            )
            raise ValueError(msg)
        activated = self.activation(hidden)
        return {
            name: self.type_output[name](activated[slices[name]])
            for name in self.node_types
        }


def _resolve_hgt_activation(name: ActivationName) -> nn.Module:
    """Return a hidden activation for HGT stacks.

    Parameters
    ----------
    name : {"relu", "sigmoid", "tanh"}
        Activation identifier.

    Returns
    -------
    nn.Module
        Instantiated activation.

    Raises
    ------
    ValueError
        If ``name`` is unsupported.
    """
    if name == "relu":
        return nn.ReLU()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "tanh":
        return nn.Tanh()
    msg = f"Unknown activation: {name!r}"
    raise ValueError(msg)


def _validate_heads_divisible(channels: int, heads: int, *, argument: str) -> None:
    """Require ``channels`` to be divisible by attention ``heads``.

    Parameters
    ----------
    channels : int
        Feature width consumed or produced by :class:`~torch_geometric.nn.HGTConv`.
    heads : int
        Number of attention heads.
    argument : str
        Name used in the error message.

    Raises
    ------
    ValueError
        If ``heads`` is not positive or does not divide ``channels``.
    """
    validate_positive_dims(heads=heads)
    if channels % heads != 0:
        msg = (
            f"{argument} ({channels}) must be divisible by heads ({heads}) for HGTConv"
        )
        raise ValueError(msg)


def _validate_edge_types_cover_nodes(
    edge_types: Sequence[EdgeTypeTriple],
    node_types: Sequence[str],
) -> None:
    """Require every edge endpoint to appear in ``node_types``.

    Parameters
    ----------
    edge_types : sequence of (src, rel, dst)
        Ordered relation schema.
    node_types : sequence of str
        Declared node-type names.

    Raises
    ------
    ValueError
        If an edge endpoint is missing from ``node_types``.
    """
    node_set = set(node_types)
    for src, rel, dst in edge_types:
        if src not in node_set or dst not in node_set:
            msg = (
                f"edge type {(src, rel, dst)!r} references a node type outside "
                f"node_types={list(node_types)!r}"
            )
            raise ValueError(msg)


def build_hgt_convs(
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    num_layers: int,
    metadata: tuple[list[str], list[EdgeTypeTriple]],
    *,
    heads: int = 1,
) -> nn.ModuleList:
    """Build a stack of PyG :class:`~torch_geometric.nn.HGTConv` layers.

    Parameters
    ----------
    in_channels : int
        Shared input width (after per-type projections when used).
    hidden_channels : int
        Hidden width for intermediate layers.
    out_channels : int
        Output width for the final layer.
    num_layers : int
        Number of HGT convolution layers.
    metadata : tuple
        PyG hetero metadata ``(node_types, edge_types)``.
    heads : int, optional
        Attention heads per layer. Default is ``1``.

    Returns
    -------
    nn.ModuleList
        Ordered ``HGTConv`` modules.

    Raises
    ------
    ValueError
        If dimensions are invalid or widths are not divisible by ``heads``.
    """
    validate_positive_dims(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_layers=num_layers,
    )
    _validate_heads_divisible(in_channels, heads, argument="in_channels")
    _validate_heads_divisible(hidden_channels, heads, argument="hidden_channels")
    _validate_heads_divisible(out_channels, heads, argument="out_channels")
    convs: list[HGTConv] = []
    for layer_idx in range(num_layers):
        layer_in = in_channels if layer_idx == 0 else hidden_channels
        layer_out = out_channels if layer_idx == num_layers - 1 else hidden_channels
        convs.append(HGTConv(layer_in, layer_out, metadata, heads=heads))
    return nn.ModuleList(convs)


def resolve_hgt_typed_inputs(
    x_or_data: HeteroData | Mapping[str, Tensor],
    edge_index: Mapping[object, Tensor] | None = None,
    *,
    node_types: Sequence[str],
    edge_types: Sequence[EdgeTypeTriple],
) -> tuple[dict[str, Tensor], dict[EdgeTypeTriple, Tensor], dict[str, int]]:
    """Resolve typed features and type-local edge indices for HGT peers.

    Unlike RelGraph banks, HGT uses PyG type-local numbering (not stacked
    global offsets).

    Parameters
    ----------
    x_or_data : HeteroData or mapping of str to Tensor
        Typed hetero snapshot or per-type feature tensors.
    edge_index : mapping or None, optional
        Type-local ``(src, rel, dst) -> edge_index`` banks when features are a
        mapping. Ignored for ``HeteroData``.
    node_types : sequence of str
        Ordered node-type names (stacking order for the latent contract).
    edge_types : sequence of (src, rel, dst)
        Explicit relation schema matching the module construction.

    Returns
    -------
    tuple
        ``(feature_dict, edge_index_dict, num_nodes_dict)``.

    Raises
    ------
    ValueError
        If types / edges are missing or malformed.
    TypeError
        If ``x_or_data`` is neither ``HeteroData`` nor a feature mapping.
    """
    resolved_types = tuple(str(name) for name in node_types)
    ordered_edges = tuple(
        (str(src), str(rel), str(dst)) for src, rel, dst in edge_types
    )
    if isinstance(x_or_data, HeteroData):
        num_nodes_dict = snapshot_num_nodes_dict(x_or_data, resolved_types)
        present = tuple(x_or_data.node_types)
        if set(present) != set(resolved_types):
            msg = (
                f"HeteroData has node types {sorted(present)!r} but the HGT "
                f"module expects exactly {list(resolved_types)!r}"
            )
            raise ValueError(msg)
        features = {name: x_or_data[name].x for name in resolved_types}
        for name, block in features.items():
            if block is None:
                msg = f"HeteroData node type {name!r} is missing feature matrix x"
                raise ValueError(msg)
        edge_index_dict: dict[EdgeTypeTriple, Tensor] = {}
        for edge_type in ordered_edges:
            if edge_type not in x_or_data.edge_types:
                msg = (
                    f"HeteroData is missing edge type {edge_type!r}; "
                    f"present={sorted(x_or_data.edge_types, key=repr)!r}"
                )
                raise ValueError(msg)
            edge_index_dict[edge_type] = x_or_data[edge_type].edge_index
        return features, edge_index_dict, num_nodes_dict

    if not isinstance(x_or_data, Mapping):
        msg = (
            "typed HGTEncoder / HGTDecoder expect HeteroData or a mapping of "
            f"per-type tensors, got {type(x_or_data).__name__}"
        )
        raise TypeError(msg)
    features = {}
    num_nodes_dict = {}
    for name in resolved_types:
        if name not in x_or_data:
            msg = (
                f"typed feature mapping is missing node type {name!r}; "
                f"expected keys {list(resolved_types)!r}, "
                f"got {sorted(x_or_data)!r}"
            )
            raise ValueError(msg)
        block = x_or_data[name]
        if block.ndim != 2:
            msg = (
                f"typed feature mapping[{name!r}] must have shape "
                f"(num_nodes, width), got {tuple(block.shape)}"
            )
            raise ValueError(msg)
        features[name] = block
        num_nodes_dict[name] = int(block.shape[0])
    if edge_index is None:
        msg = (
            "edge_index relation mapping is required when typed HGT input is a "
            "feature mapping (values must use type-local numbering)"
        )
        raise ValueError(msg)
    edge_index_dict = {}
    for edge_type in ordered_edges:
        edge_index_dict[edge_type] = _lookup_typed_edge_index(
            edge_index,
            edge_type,
            expected_keys=ordered_edges,
        )
    return features, edge_index_dict, num_nodes_dict


def _lookup_typed_edge_index(
    edge_index: Mapping[object, Tensor],
    edge_type: EdgeTypeTriple,
    *,
    expected_keys: Sequence[EdgeTypeTriple],
) -> Tensor:
    """Fetch a type-local edge bank, normalizing mapping keys to triples.

    Parameters
    ----------
    edge_index : mapping
        Caller-supplied edge banks.
    edge_type : (src, rel, dst)
        Desired relation key.
    expected_keys : sequence of (src, rel, dst)
        Full schema used in error messages.

    Returns
    -------
    Tensor
        Edge index with shape ``(2, num_edges)``.

    Raises
    ------
    ValueError
        If ``edge_type`` is missing from the mapping.
    """
    if edge_type in edge_index:
        return edge_index[edge_type]
    for key, value in edge_index.items():
        try:
            key_triple = tuple(str(part) for part in key)  # type: ignore[arg-type]
        except TypeError:
            continue
        if len(key_triple) == 3 and key_triple == edge_type:
            return value
    msg = (
        f"edge_index mapping is missing edge type {edge_type!r}; "
        f"expected keys {list(expected_keys)!r}"
    )
    raise ValueError(msg)


def _stack_typed_latents(
    x_dict: Mapping[str, Tensor | None],
    node_types: Sequence[str],
) -> Tensor:
    """Stack per-type HGT outputs into the RelGraph latent contract.

    Parameters
    ----------
    x_dict : mapping of str to Tensor or None
        Per-type embeddings from ``HGTConv``.
    node_types : sequence of str
        Stacking order.

    Returns
    -------
    Tensor
        Stacked ``(N, d)`` block ordered by ``node_types``.

    Raises
    ------
    ValueError
        If any type is missing or ``None`` (no messages received).
    """
    blocks: list[Tensor] = []
    for name in node_types:
        block = x_dict.get(name)
        if block is None:
            msg = (
                f"HGTConv produced no embedding for node type {name!r}; "
                "ensure the typed schema includes edges that deliver messages "
                "to every node type"
            )
            raise ValueError(msg)
        blocks.append(block)
    return torch.cat(blocks, dim=0)


def _hgt_message_passing(
    module: nn.Module,
    x_dict: dict[str, Tensor],
    edge_index_dict: dict[EdgeTypeTriple, Tensor],
) -> dict[str, Tensor]:
    """Run stacked HGT layers with hidden activations between layers.

    Parameters
    ----------
    module : nn.Module
        Module exposing ``convs``, ``activation``, and ``num_layers``.
    x_dict : dict of str to Tensor
        Per-type features at the shared HGT width.
    edge_index_dict : dict of (src, rel, dst) to Tensor
        Type-local edge indices.

    Returns
    -------
    dict of str to Tensor
        Final per-type embeddings (no trailing activation).
    """
    hidden = x_dict
    for layer_idx, conv in enumerate(module.convs):
        hidden = conv(hidden, edge_index_dict)
        # Materialize None -> error early with a clear message.
        for name in module.node_types:
            if hidden.get(name) is None:
                msg = (
                    f"HGTConv layer {layer_idx} produced no embedding for "
                    f"node type {name!r}"
                )
                raise ValueError(msg)
        if layer_idx < module.num_layers - 1:
            hidden = {
                name: module.activation(hidden[name]) for name in module.node_types
            }
    return {name: hidden[name] for name in module.node_types}


class HGTEncoder(nn.Module):
    """Optional typed Heterogeneous Graph Transformer encoder peer.

    Wraps PyTorch Geometric :class:`~torch_geometric.nn.HGTConv` (Hu et al.,
    2020 — target-type queries, source-type keys/values, relation-specific
    priors; not a full paper reproduction). Per-type input linears map
    ``F_τ`` into a shared hidden width, HGT layers produce a shared latent
    width ``d``, and outputs are stacked as ``(N, d)`` in ``node_types``
    order — the same latent contract as typed :class:`RelGraphEncoder`.

    This peer is **optional** and is **not** required for hetero Koopman
    support. Factory ``koopman="hetero_graph"`` continues to require
    RelGraph peers; import HGT from :mod:`koopman_graph.nn` for custom
    encode/decode stacks.

    Attributes
    ----------
    in_channels : dict of str to int
        Per-type input feature widths.
    hidden_channels : int
        Shared HGT hidden width.
    latent_dim : int
        Shared output latent width ``d``.
    node_types : tuple of str
        Ordered node-type names (stacking order).
    edge_types : tuple of tuple of str
        Ordered ``(src, rel, dst)`` relation schema.
    num_relations : int
        ``|R|`` (length of ``edge_types``).
    heads : int
        Attention heads per HGT layer.
    """

    def __init__(
        self,
        in_channels: Mapping[str, int],
        hidden_channels: int,
        latent_dim: int,
        *,
        node_types: Sequence[str],
        edge_types: Sequence[Sequence[str]],
        num_layers: int = 2,
        heads: int = 1,
        activation: ActivationName = "relu",
    ) -> None:
        """Initialize the typed HGT encoder stack.

        Parameters
        ----------
        in_channels : mapping of str to int
            Per-type input widths ``F_τ``.
        hidden_channels : int
            Shared hidden width after per-type projections.
        latent_dim : int
            Shared latent width ``d``.
        node_types : sequence of str
            Ordered node-type names.
        edge_types : sequence of sequence of str
            Ordered ``(src, rel, dst)`` triples.
        num_layers : int, optional
            Number of HGT layers. Default is ``2``.
        heads : int, optional
            Attention heads (must divide ``hidden_channels`` and
            ``latent_dim``). Default is ``1``.
        activation : {"relu", "sigmoid", "tanh"}, optional
            Hidden activation between HGT layers. Default is ``"relu"``.

        Raises
        ------
        ValueError
            If typed metadata is inconsistent or widths are not divisible by
            ``heads``.
        """
        super().__init__()
        in_channels_dict, resolved_node_types = _normalize_typed_channels(
            in_channels,
            node_types,
            argument="in_channels",
        )
        validate_positive_dims(
            hidden_channels=hidden_channels,
            latent_dim=latent_dim,
            num_layers=num_layers,
        )
        resolved_edge_types = _normalize_edge_type_order(
            edge_types,
            num_relations=len(tuple(edge_types)),
            required=True,
        )
        assert resolved_edge_types is not None
        _validate_edge_types_cover_nodes(resolved_edge_types, resolved_node_types)
        _validate_heads_divisible(hidden_channels, heads, argument="hidden_channels")
        _validate_heads_divisible(latent_dim, heads, argument="latent_dim")

        self.in_channels = dict(in_channels_dict)
        self.in_channels_dict = in_channels_dict
        self.hidden_channels = hidden_channels
        self.latent_dim = latent_dim
        self.node_types = resolved_node_types
        self.edge_types = resolved_edge_types
        self.num_relations = len(resolved_edge_types)
        self.heads = heads
        self.num_layers = num_layers
        self.activation_name = activation
        self.activation = _resolve_hgt_activation(activation)
        self.is_typed = True

        metadata = (
            list(resolved_node_types),
            list(resolved_edge_types),
        )
        self.type_input = nn.ModuleDict(
            {
                name: nn.Linear(width, hidden_channels)
                for name, width in in_channels_dict.items()
            }
        )
        self.convs = build_hgt_convs(
            hidden_channels,
            hidden_channels,
            latent_dim,
            num_layers,
            metadata,
            heads=heads,
        )

    def forward(
        self,
        x_or_data: HeteroData | Mapping[str, Tensor],
        edge_index: Mapping[object, Tensor] | None = None,
        edge_weight: Mapping[object, Tensor | None] | None = None,
    ) -> Tensor:
        """Encode typed node features with HGT message passing.

        Parameters
        ----------
        x_or_data : HeteroData or mapping of str to Tensor
            Typed snapshot or per-type feature tensors.
        edge_index : mapping or None, optional
            Type-local edge banks when features are a mapping. Ignored for
            ``HeteroData``.
        edge_weight : mapping or None, optional
            Ignored (HGTConv does not consume scalar edge weights here).

        Returns
        -------
        Tensor
            Stacked latents with shape ``(N, latent_dim)`` ordered by
            :attr:`node_types`.
        """
        del edge_weight  # HGTConv path does not use scalar edge weights.
        features, edge_index_dict, _ = resolve_hgt_typed_inputs(
            x_or_data,
            edge_index,
            node_types=self.node_types,
            edge_types=self.edge_types,
        )
        projected = {
            name: self.activation(self.type_input[name](features[name]))
            for name in self.node_types
        }
        encoded = _hgt_message_passing(self, projected, edge_index_dict)
        return _stack_typed_latents(encoded, self.node_types)


class HGTDecoder(nn.Module):
    """Optional typed Heterogeneous Graph Transformer decoder peer.

    Mirrors :class:`HGTEncoder`: stacked ``(N, d)`` latents are split by
    ``node_types``, passed through :class:`~torch_geometric.nn.HGTConv`
    layers to a shared hidden width, then mapped by per-type linear heads to
    ``F_τ``. Returns ``dict[str, Tensor]`` like typed :class:`RelGraphDecoder`.

    Optional only — not required for hetero Koopman support; not a factory
    default. Citation honesty matches :class:`HGTEncoder` (Hu et al., 2020).

    Attributes
    ----------
    latent_dim : int
        Shared input latent width ``d``.
    hidden_channels : int
        Shared HGT hidden width (and typed head input width).
    out_channels : dict of str to int
        Per-type physical feature widths.
    node_types : tuple of str
        Ordered node-type names.
    edge_types : tuple of tuple of str
        Ordered ``(src, rel, dst)`` relation schema.
    num_relations : int
        ``|R|``.
    heads : int
        Attention heads per HGT layer.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_channels: int,
        out_channels: Mapping[str, int],
        *,
        node_types: Sequence[str],
        edge_types: Sequence[Sequence[str]],
        num_layers: int = 2,
        heads: int = 1,
        activation: ActivationName = "relu",
    ) -> None:
        """Initialize the typed HGT decoder stack.

        Parameters
        ----------
        latent_dim : int
            Shared latent width ``d``.
        hidden_channels : int
            Shared HGT hidden width.
        out_channels : mapping of str to int
            Per-type output widths ``F_τ``.
        node_types : sequence of str
            Ordered node-type names.
        edge_types : sequence of sequence of str
            Ordered ``(src, rel, dst)`` triples.
        num_layers : int, optional
            Number of HGT layers. Default is ``2``.
        heads : int, optional
            Attention heads (must divide ``latent_dim`` and
            ``hidden_channels``). Default is ``1``.
        activation : {"relu", "sigmoid", "tanh"}, optional
            Hidden activation between HGT layers and before typed heads.
            Default is ``"relu"``.

        Raises
        ------
        ValueError
            If typed metadata is inconsistent or widths are not divisible by
            ``heads``.
        """
        super().__init__()
        out_channels_dict, resolved_node_types = _normalize_typed_channels(
            out_channels,
            node_types,
            argument="out_channels",
        )
        validate_positive_dims(
            latent_dim=latent_dim,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
        )
        resolved_edge_types = _normalize_edge_type_order(
            edge_types,
            num_relations=len(tuple(edge_types)),
            required=True,
        )
        assert resolved_edge_types is not None
        _validate_edge_types_cover_nodes(resolved_edge_types, resolved_node_types)
        _validate_heads_divisible(latent_dim, heads, argument="latent_dim")
        _validate_heads_divisible(hidden_channels, heads, argument="hidden_channels")

        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels
        self.out_channels = dict(out_channels_dict)
        self.out_channels_dict = out_channels_dict
        self.node_types = resolved_node_types
        self.edge_types = resolved_edge_types
        self.num_relations = len(resolved_edge_types)
        self.heads = heads
        self.num_layers = num_layers
        self.activation_name = activation
        self.activation = _resolve_hgt_activation(activation)
        self.is_typed = True

        metadata = (
            list(resolved_node_types),
            list(resolved_edge_types),
        )
        self.convs = build_hgt_convs(
            latent_dim,
            hidden_channels,
            hidden_channels,
            num_layers,
            metadata,
            heads=heads,
        )
        self.type_output = nn.ModuleDict(
            {
                name: nn.Linear(hidden_channels, width)
                for name, width in out_channels_dict.items()
            }
        )

    def forward(
        self,
        z: Tensor,
        edge_index_or_data: HeteroData | Mapping[object, Tensor],
        edge_weight: Mapping[object, Tensor | None] | None = None,
        *,
        num_nodes_dict: Mapping[str, int] | None = None,
    ) -> dict[str, Tensor]:
        """Decode stacked latents with HGT message passing.

        Parameters
        ----------
        z : Tensor
            Stacked latents ``(N, latent_dim)`` ordered by :attr:`node_types`.
        edge_index_or_data : HeteroData or mapping
            Typed snapshot (topology only) or type-local
            ``(src, rel, dst) -> edge_index`` mapping.
        edge_weight : mapping or None, optional
            Ignored (HGTConv does not consume scalar edge weights here).
        num_nodes_dict : mapping of str to int or None, optional
            Per-type node counts. Required when ``edge_index_or_data`` is a
            mapping; inferred from ``HeteroData`` when omitted.

        Returns
        -------
        dict of str to Tensor
            Per-type reconstructions ``(N_τ, F_τ)``.

        Raises
        ------
        ValueError
            If ``num_nodes_dict`` is missing / inconsistent with ``z``.
        """
        del edge_weight
        if isinstance(edge_index_or_data, HeteroData):
            resolved_counts = snapshot_num_nodes_dict(
                edge_index_or_data,
                self.node_types,
            )
            if num_nodes_dict is not None and dict(num_nodes_dict) != resolved_counts:
                msg = (
                    f"num_nodes_dict {dict(num_nodes_dict)!r} does not match "
                    f"HeteroData counts {resolved_counts!r}"
                )
                raise ValueError(msg)
            num_nodes_dict = resolved_counts
            _, edge_index_dict, _ = resolve_hgt_typed_inputs(
                edge_index_or_data,
                node_types=self.node_types,
                edge_types=self.edge_types,
            )
        else:
            if num_nodes_dict is None:
                msg = (
                    "num_nodes_dict is required for typed HGTDecoder when "
                    "topology is supplied as an edge_index mapping"
                )
                raise ValueError(msg)
            num_nodes_dict = {str(k): int(v) for k, v in num_nodes_dict.items()}
            dummy_features = {
                name: torch.zeros(
                    int(num_nodes_dict[name]),
                    self.latent_dim,
                    dtype=z.dtype,
                    device=z.device,
                )
                for name in self.node_types
            }
            _, edge_index_dict, _ = resolve_hgt_typed_inputs(
                dummy_features,
                edge_index_or_data,
                node_types=self.node_types,
                edge_types=self.edge_types,
            )

        slices = node_type_slices(self.node_types, num_nodes_dict)
        total = sum(int(num_nodes_dict[name]) for name in self.node_types)
        if z.ndim != 2 or z.shape[1] != self.latent_dim:
            msg = (
                f"z must have shape (num_nodes, latent_dim={self.latent_dim}); "
                f"got {tuple(z.shape)}"
            )
            raise ValueError(msg)
        if total != int(z.shape[0]):
            msg = (
                f"num_nodes_dict sums to {total} but the stacked latent block "
                f"has {int(z.shape[0])} rows"
            )
            raise ValueError(msg)
        x_dict = {name: z[slices[name]] for name in self.node_types}
        hidden = _hgt_message_passing(self, x_dict, edge_index_dict)
        activated = {name: self.activation(hidden[name]) for name in self.node_types}
        return {
            name: self.type_output[name](activated[name]) for name in self.node_types
        }
