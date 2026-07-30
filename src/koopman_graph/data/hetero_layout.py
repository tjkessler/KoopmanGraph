"""Stacked latent / feature layout helpers for typed hetero graphs.

Layout contract
---------------
Typed heterogeneous graphs are represented in a **single stacked block**. For
ordered node types ``(τ_0, …, τ_{M-1})`` with per-type cardinalities ``N_τ``,
node-level quantities (physical features and Koopman latents alike) are stacked
row-wise into one matrix with ``N = Σ_τ N_τ`` rows::

    rows [0, N_{τ_0})                    -> type τ_0
    rows [N_{τ_0}, N_{τ_0} + N_{τ_1})    -> type τ_1
    …

so latents live in ``Z ∈ R^{N×d}`` at a **shared** latent width ``d``. Type
order is the caller-supplied ``node_type_names`` order, which for containers
built from :class:`~koopman_graph.data.HeteroGraphSnapshotSequence` is the
snapshot-0 schema order exposed by its ``node_type_names`` attribute.
Per-type latent widths ``d_τ`` are not part of this layout.

Because a stacked block has one global node numbering, per-relation
``edge_index`` tensors stored per PyG ``(src, rel, dst)`` edge type use
**type-local** indices and must be shifted into the global numbering with
:func:`offset_edge_index` (or, for a whole snapshot,
:func:`global_relation_edge_indices`) before they index stacked rows.
Destination in-degree normalization is unaffected by the shift, so a single
relation bank can be aggregated against the stacked block directly.

Dict-valued per-type features appear only at the encode / decode boundary;
:func:`stack_typed_features` and :func:`unstack_typed_features` convert between
the two views when every type shares a feature width.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor
from torch_geometric.data import HeteroData

__all__ = [
    "global_relation_edge_indices",
    "latent_type_slices",
    "mask_hetero_snapshot_features",
    "node_type_offsets",
    "node_type_slices",
    "offset_edge_index",
    "snapshot_num_nodes_dict",
    "stack_typed_features",
    "unstack_typed_features",
]

EdgeTypeTriple = tuple[str, str, str]


def _validate_node_type_names(node_type_names: Sequence[str]) -> tuple[str, ...]:
    """Validate and freeze an ordered node-type name tuple.

    Parameters
    ----------
    node_type_names : sequence of str
        Ordered node-type names defining the stacking order.

    Returns
    -------
    tuple of str
        Validated names in the supplied order.

    Raises
    ------
    ValueError
        If the sequence is empty, contains empty names, or repeats a name.
    """
    resolved = tuple(str(name) for name in node_type_names)
    if not resolved:
        msg = "node_type_names must contain at least one node type"
        raise ValueError(msg)
    if any(not name for name in resolved):
        msg = f"node_type_names entries must be non-empty strings; got {resolved!r}"
        raise ValueError(msg)
    if len(set(resolved)) != len(resolved):
        msg = f"node_type_names must be unique; got {resolved!r}"
        raise ValueError(msg)
    return resolved


def node_type_offsets(
    node_type_names: Sequence[str],
    num_nodes_dict: Mapping[str, int],
) -> dict[str, int]:
    """Return the first stacked row index of every node type.

    Parameters
    ----------
    node_type_names : sequence of str
        Ordered node-type names defining the stacking order.
    num_nodes_dict : mapping of str to int
        Node count ``N_τ`` for every name in ``node_type_names``.

    Returns
    -------
    dict of str to int
        Row offset per node type, in ``node_type_names`` order.

    Raises
    ------
    ValueError
        If names are invalid, a count is missing, or a count is not positive.
    """
    resolved = _validate_node_type_names(node_type_names)
    offsets: dict[str, int] = {}
    cursor = 0
    for name in resolved:
        if name not in num_nodes_dict:
            msg = (
                f"num_nodes_dict is missing node type {name!r}; "
                f"expected keys {list(resolved)!r}, got {sorted(num_nodes_dict)!r}"
            )
            raise ValueError(msg)
        count = int(num_nodes_dict[name])
        if count < 1:
            msg = f"num_nodes_dict[{name!r}] must be positive, got {count}"
            raise ValueError(msg)
        offsets[name] = cursor
        cursor += count
    extra = set(num_nodes_dict) - set(resolved)
    if extra:
        msg = (
            f"num_nodes_dict has node types outside node_type_names: {sorted(extra)!r}"
        )
        raise ValueError(msg)
    return offsets


def node_type_slices(
    node_type_names: Sequence[str],
    num_nodes_dict: Mapping[str, int],
) -> dict[str, slice]:
    """Return the stacked row slice of every node type.

    Parameters
    ----------
    node_type_names : sequence of str
        Ordered node-type names defining the stacking order.
    num_nodes_dict : mapping of str to int
        Node count ``N_τ`` for every name in ``node_type_names``.

    Returns
    -------
    dict of str to slice
        Half-open row slice per node type; slice lengths sum to
        ``Σ_τ N_τ`` and the slices tile ``[0, Σ_τ N_τ)`` without gaps.

    Raises
    ------
    ValueError
        If names are invalid, a count is missing, or a count is not positive.
    """
    offsets = node_type_offsets(node_type_names, num_nodes_dict)
    return {
        name: slice(offset, offset + int(num_nodes_dict[name]))
        for name, offset in offsets.items()
    }


def latent_type_slices(
    node_type_slices: Mapping[str, slice],
    latent_dim: int,
) -> dict[str, slice]:
    """Expand node-row slices into flat ``vec(Z)`` index slices.

    Under the stacked ``(N, d)`` layout with C-order ``Z.reshape(-1)``, node
    rows ``[a, b)`` occupy flat indices ``[a·d, b·d)``.

    Parameters
    ----------
    node_type_slices : mapping of str to slice
        Node-row slices from :func:`node_type_slices`.
    latent_dim : int
        Shared latent width ``d``.

    Returns
    -------
    dict of str to slice
        Flat index slices into eigenvectors / ``K_eff`` of shape ``(N·d,)``.

    Raises
    ------
    ValueError
        If ``latent_dim`` is not positive or a node slice is invalid.
    """
    if latent_dim < 1:
        msg = f"latent_dim must be positive, got {latent_dim}"
        raise ValueError(msg)
    expanded: dict[str, slice] = {}
    for name, node_slice in node_type_slices.items():
        start = node_slice.start
        stop = node_slice.stop
        if start is None or stop is None:
            msg = (
                f"node_type_slices[{name!r}] must be a bounded slice; "
                f"got {node_slice!r}"
            )
            raise ValueError(msg)
        if stop < start:
            msg = (
                f"node_type_slices[{name!r}] must satisfy stop >= start; "
                f"got {node_slice!r}"
            )
            raise ValueError(msg)
        expanded[str(name)] = slice(int(start) * latent_dim, int(stop) * latent_dim)
    return expanded


def snapshot_num_nodes_dict(
    snapshot: HeteroData,
    node_type_names: Sequence[str],
) -> dict[str, int]:
    """Return per-type node counts read from a ``HeteroData`` snapshot.

    Parameters
    ----------
    snapshot : HeteroData
        Typed (or multiplex) hetero snapshot carrying ``x`` per node type.
    node_type_names : sequence of str
        Ordered node-type names expected on the snapshot.

    Returns
    -------
    dict of str to int
        Node count ``N_τ`` per type, in ``node_type_names`` order.

    Raises
    ------
    ValueError
        If a node type is absent from the snapshot or is missing ``x``.
    """
    resolved = _validate_node_type_names(node_type_names)
    present = set(snapshot.node_types)
    counts: dict[str, int] = {}
    for name in resolved:
        if name not in present:
            msg = (
                f"HeteroData snapshot is missing node type {name!r}; "
                f"present types are {sorted(present)!r}"
            )
            raise ValueError(msg)
        features = snapshot[name].x
        if features is None:
            msg = f"HeteroData node type {name!r} is missing feature matrix x"
            raise ValueError(msg)
        counts[name] = int(features.shape[0])
    return counts


def stack_typed_features(
    feature_dict: Mapping[str, Tensor],
    node_type_names: Sequence[str],
) -> Tensor:
    """Stack equal-width per-type tensors into one ``(N, F)`` block.

    Parameters
    ----------
    feature_dict : mapping of str to Tensor
        Per-type tensors, each ``(N_τ, F)`` with a shared trailing width ``F``.
    node_type_names : sequence of str
        Ordered node-type names defining the stacking order.

    Returns
    -------
    Tensor
        Stacked block with shape ``(Σ_τ N_τ, F)``.

    Raises
    ------
    ValueError
        If a type is missing, a tensor is not 2D, or widths ``F`` disagree.
    """
    resolved = _validate_node_type_names(node_type_names)
    blocks: list[Tensor] = []
    width: int | None = None
    for name in resolved:
        if name not in feature_dict:
            msg = (
                f"feature_dict is missing node type {name!r}; "
                f"expected keys {list(resolved)!r}, got {sorted(feature_dict)!r}"
            )
            raise ValueError(msg)
        block = feature_dict[name]
        if block.ndim != 2:
            msg = (
                f"feature_dict[{name!r}] must have shape (num_nodes, width), "
                f"got {tuple(block.shape)}"
            )
            raise ValueError(msg)
        if width is None:
            width = int(block.shape[1])
        elif int(block.shape[1]) != width:
            msg = (
                "stack_typed_features requires one shared trailing width; "
                f"node type {name!r} has width {int(block.shape[1])}, "
                f"expected {width}"
            )
            raise ValueError(msg)
        blocks.append(block)
    return torch.cat(blocks, dim=0)


def unstack_typed_features(
    stacked: Tensor,
    node_type_names: Sequence[str],
    num_nodes_dict: Mapping[str, int],
) -> dict[str, Tensor]:
    """Split a stacked ``(N, F)`` block into per-type views.

    Parameters
    ----------
    stacked : Tensor
        Stacked block with shape ``(Σ_τ N_τ, F)``.
    node_type_names : sequence of str
        Ordered node-type names defining the stacking order.
    num_nodes_dict : mapping of str to int
        Node count ``N_τ`` for every name in ``node_type_names``.

    Returns
    -------
    dict of str to Tensor
        Per-type slices of ``stacked`` (views, not copies).

    Raises
    ------
    ValueError
        If ``stacked`` is not 2D or its row count differs from ``Σ_τ N_τ``.
    """
    if stacked.ndim != 2:
        msg = (
            "unstack_typed_features expects a stacked (num_nodes, width) "
            f"tensor, got {tuple(stacked.shape)}"
        )
        raise ValueError(msg)
    slices = node_type_slices(node_type_names, num_nodes_dict)
    total = sum(int(num_nodes_dict[name]) for name in slices)
    if int(stacked.shape[0]) != total:
        msg = (
            f"stacked tensor has {int(stacked.shape[0])} rows but "
            f"num_nodes_dict sums to {total}"
        )
        raise ValueError(msg)
    return {name: stacked[type_slice] for name, type_slice in slices.items()}


def offset_edge_index(
    edge_index: Tensor,
    src_offset: int,
    dst_offset: int,
) -> Tensor:
    """Shift a type-local ``edge_index`` into stacked global numbering.

    Parameters
    ----------
    edge_index : Tensor
        Type-local relation edge index with shape ``(2, num_edges)``
        (row 0 = source, row 1 = destination).
    src_offset : int
        Stacked row offset of the source node type.
    dst_offset : int
        Stacked row offset of the destination node type.

    Returns
    -------
    Tensor
        New edge index with the same shape and dtype, shifted per row. When
        both offsets are zero the input is returned unchanged.

    Raises
    ------
    ValueError
        If ``edge_index`` does not have shape ``(2, num_edges)`` or an offset
        is negative.
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        msg = (
            "offset_edge_index expects edge_index with shape (2, num_edges), "
            f"got {tuple(edge_index.shape)}"
        )
        raise ValueError(msg)
    if src_offset < 0 or dst_offset < 0:
        msg = (
            "offset_edge_index requires non-negative offsets, got "
            f"src_offset={src_offset}, dst_offset={dst_offset}"
        )
        raise ValueError(msg)
    if src_offset == 0 and dst_offset == 0:
        return edge_index
    shift = torch.tensor(
        [[int(src_offset)], [int(dst_offset)]],
        dtype=edge_index.dtype,
        device=edge_index.device,
    )
    return edge_index + shift


def mask_hetero_snapshot_features(
    snapshot: HeteroData,
    masks: Mapping[str, Tensor],
) -> HeteroData:
    """Return a copy of ``snapshot`` with unobserved node rows zeroed.

    Parameters
    ----------
    snapshot : HeteroData
        Hetero snapshot to copy. Node and edge stores are copied by reference
        except for the zeroed ``x`` matrices.
    masks : mapping of str to Tensor
        Boolean masks with shape ``(N_τ,)``. ``True`` means observed. Node
        types absent from ``masks`` keep their features unchanged.

    Returns
    -------
    HeteroData
        Snapshot with masked node features and the original schema.

    Raises
    ------
    ValueError
        If a mask shape does not match its node type's row count.
    """
    masked = HeteroData()
    for node_type in snapshot.node_types:
        for key, value in snapshot[node_type].items():
            masked[node_type][key] = value
        features = snapshot[node_type].get("x")
        mask = masks.get(node_type)
        if features is None or mask is None:
            continue
        if mask.shape != (int(features.shape[0]),):
            msg = (
                f"observation mask for node type {node_type!r} has shape "
                f"{tuple(mask.shape)}, expected ({int(features.shape[0])},)"
            )
            raise ValueError(msg)
        block = features.clone()
        block[~mask.bool()] = 0.0
        masked[node_type].x = block
    for edge_type in snapshot.edge_types:
        for key, value in snapshot[edge_type].items():
            masked[edge_type][key] = value
    return masked


def global_relation_edge_indices(
    snapshot: HeteroData,
    edge_types: Sequence[Sequence[str]],
    node_type_names: Sequence[str],
    num_nodes_dict: Mapping[str, int] | None = None,
) -> list[Tensor]:
    """Return ordered relation banks in stacked global numbering.

    Parameters
    ----------
    snapshot : HeteroData
        Typed (or multiplex) hetero snapshot carrying per-relation
        ``edge_index`` tensors.
    edge_types : sequence of sequence of str
        Ordered ``(src, rel, dst)`` triples defining relation-bank order.
    node_type_names : sequence of str
        Ordered node-type names defining the stacking order.
    num_nodes_dict : mapping of str to int or None, optional
        Per-type node counts. Read from ``snapshot`` when omitted.

    Returns
    -------
    list of Tensor
        One global ``edge_index`` per entry of ``edge_types``, same order.

    Raises
    ------
    ValueError
        If a triple is malformed, references an unknown node type, or is
        absent from ``snapshot``.
    """
    resolved_counts = (
        snapshot_num_nodes_dict(snapshot, node_type_names)
        if num_nodes_dict is None
        else dict(num_nodes_dict)
    )
    offsets = node_type_offsets(node_type_names, resolved_counts)
    present = {tuple(edge_type) for edge_type in snapshot.edge_types}
    banks: list[Tensor] = []
    for entry in edge_types:
        triple = tuple(str(part) for part in entry)
        if len(triple) != 3:
            msg = (
                "edge_types entries must be (src, rel, dst) triples; "
                f"got {tuple(entry)!r}"
            )
            raise ValueError(msg)
        src, _rel, dst = triple
        for name in (src, dst):
            if name not in offsets:
                msg = (
                    f"edge type {triple!r} references node type {name!r} "
                    f"outside node_type_names {list(offsets)!r}"
                )
                raise ValueError(msg)
        if triple not in present:
            msg = (
                f"HeteroData snapshot is missing edge type {triple!r}; "
                f"present edge types are {sorted(present)!r}"
            )
            raise ValueError(msg)
        banks.append(
            offset_edge_index(
                snapshot[triple].edge_index,
                offsets[src],
                offsets[dst],
            )
        )
    return banks
