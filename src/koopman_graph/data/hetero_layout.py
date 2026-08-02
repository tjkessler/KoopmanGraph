"""Stacked latent / feature layout helpers for typed hetero graphs.

Layout contract
---------------
Typed heterogeneous graphs use ordered node types ``(τ_0, …, τ_{M-1})`` with
per-type cardinalities ``N_τ``. Type order is the caller-supplied
``node_type_names`` order (for
:class:`~koopman_graph.data.HeteroGraphSnapshotSequence`, snapshot-0 schema
order via ``node_type_names``).

**Shared-d path (default).** When opt-in ``latent_dims`` is absent, physical
features and Koopman latents are stacked row-wise into one matrix with
``N = Σ_τ N_τ`` rows and a shared trailing width ``d``::

    rows [0, N_{τ_0})                    -> type τ_0
    rows [N_{τ_0}, N_{τ_0} + N_{τ_1})    -> type τ_1
    …

so latents live in ``Z ∈ R^{N×d}``. Flat ``vec(Z)`` slices use
:func:`latent_type_slices` (``[a·d, b·d)`` for node rows ``[a, b)``).

**Opt-in per-type widths.** :func:`validate_latent_dims` accepts
``latent_dims: Mapping[str, int]`` with exactly the keys in
``node_type_names``. Unequal ``d_τ`` latents are **block-concatenated**
(not a single ``(N, d)`` matrix): type ``τ`` contributes an ``(N_τ, d_τ)``
block flattened in C-order, so the flat length is ``Σ_τ N_τ·d_τ``. Use
:func:`latent_type_slices_from_dims` for those flat slices. Providing
``latent_dims`` never silently falls back to the shared-d path.

Because a stacked node block has one global numbering, per-relation
``edge_index`` tensors stored per PyG ``(src, rel, dst)`` edge type use
**type-local** indices and must be shifted into the global numbering with
:func:`offset_edge_index` (or, for a whole snapshot,
:func:`global_relation_edge_indices`) before they index shared-d stacked
rows. Destination in-degree normalization is unaffected by the shift.

Dict-valued per-type features appear only at the encode / decode boundary;
:func:`stack_typed_features` and :func:`unstack_typed_features` convert between
the two views when every type shares a **feature** width ``F`` (independent of
``latent_dims``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor
from torch_geometric.data import HeteroData

__all__ = [
    "global_relation_edge_indices",
    "latent_type_slices",
    "latent_type_slices_from_dims",
    "mask_hetero_snapshot_features",
    "node_type_offsets",
    "node_type_slices",
    "offset_edge_index",
    "snapshot_num_nodes_dict",
    "stack_typed_features",
    "stacked_latent_numel",
    "unstack_typed_features",
    "validate_latent_dims",
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
    """Expand node-row slices into flat ``vec(Z)`` index slices (shared ``d``).

    Under the stacked ``(N, d)`` layout with C-order ``Z.reshape(-1)``, node
    rows ``[a, b)`` occupy flat indices ``[a·d, b·d)``. For opt-in unequal
    ``d_τ``, use :func:`latent_type_slices_from_dims` instead — this helper
    never interprets a ``latent_dims`` mapping.

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


def validate_latent_dims(
    node_type_names: Sequence[str],
    latent_dims: Mapping[str, int] | None,
    *,
    shared_latent_dim: int | None = None,
) -> dict[str, int] | None:
    """Validate opt-in per-type latent widths ``d_τ``.

    Parameters
    ----------
    node_type_names : sequence of str
        Ordered node-type names defining the stacking order.
    latent_dims : mapping of str to int or None
        Opt-in ``τ -> d_τ``. ``None`` selects the shared-d path (return
        ``None``). When provided, keys must match ``node_type_names``
        exactly — missing or extra keys raise (no silent ignore).
    shared_latent_dim : int or None, optional
        Shared width ``d`` used when ``latent_dims is None``. When set, must
        be positive. Ignored when ``latent_dims`` is provided (callers that
        need ``d_τ == shared_latent_dim`` for every type should compare
        explicitly).

    Returns
    -------
    dict of str to int or None
        Frozen ``τ -> d_τ`` in ``node_type_names`` order, or ``None`` for the
        shared-d path.

    Raises
    ------
    ValueError
        If names are invalid, ``shared_latent_dim`` is invalid on the shared
        path, keys disagree with ``node_type_names``, or a width is not
        positive.
    """
    resolved = _validate_node_type_names(node_type_names)
    if latent_dims is None:
        if shared_latent_dim is not None and int(shared_latent_dim) < 1:
            msg = f"shared_latent_dim must be positive, got {shared_latent_dim}"
            raise ValueError(msg)
        return None

    missing = [name for name in resolved if name not in latent_dims]
    if missing:
        msg = (
            f"latent_dims is missing node type(s) {missing!r}; "
            f"expected keys {list(resolved)!r}, got {sorted(latent_dims)!r}"
        )
        raise ValueError(msg)
    extra = set(latent_dims) - set(resolved)
    if extra:
        msg = (
            f"latent_dims has node types outside node_type_names: "
            f"{sorted(extra)!r}; expected keys {list(resolved)!r}"
        )
        raise ValueError(msg)

    validated: dict[str, int] = {}
    for name in resolved:
        width = int(latent_dims[name])
        if width < 1:
            msg = f"latent_dims[{name!r}] must be positive, got {width}"
            raise ValueError(msg)
        validated[name] = width
    return validated


def stacked_latent_numel(
    node_type_names: Sequence[str],
    num_nodes_dict: Mapping[str, int],
    latent_dims: Mapping[str, int],
) -> int:
    """Return flat latent length ``Σ_τ N_τ · d_τ`` for opt-in ``latent_dims``.

    Parameters
    ----------
    node_type_names : sequence of str
        Ordered node-type names.
    num_nodes_dict : mapping of str to int
        Node count ``N_τ`` for every name in ``node_type_names``.
    latent_dims : mapping of str to int
        Validated (or to-be-validated) ``τ -> d_τ``.

    Returns
    -------
    int
        Total number of latent scalars in the block-concatenated layout.

    Raises
    ------
    ValueError
        If counts / widths fail validation.
    """
    dims = validate_latent_dims(node_type_names, latent_dims)
    assert dims is not None
    # Reuse count validation from node_type_offsets.
    _ = node_type_offsets(node_type_names, num_nodes_dict)
    return sum(int(num_nodes_dict[name]) * dims[name] for name in dims)


def latent_type_slices_from_dims(
    node_type_names: Sequence[str],
    num_nodes_dict: Mapping[str, int],
    latent_dims: Mapping[str, int],
) -> dict[str, slice]:
    """Return flat latent slices for opt-in unequal ``d_τ`` layouts.

    Block-concatenates per-type ``(N_τ, d_τ)`` matrices in
    ``node_type_names`` order (C-order flatten within each type). Type ``τ``
    occupies ``N_τ · d_τ`` consecutive flat indices. This is distinct from
    :func:`latent_type_slices`, which assumes a shared trailing width ``d``.

    Parameters
    ----------
    node_type_names : sequence of str
        Ordered node-type names defining the stacking order.
    num_nodes_dict : mapping of str to int
        Node count ``N_τ`` for every name in ``node_type_names``.
    latent_dims : mapping of str to int
        Opt-in ``τ -> d_τ`` (validated via :func:`validate_latent_dims`).

    Returns
    -------
    dict of str to slice
        Half-open flat slices tiling ``[0, Σ_τ N_τ·d_τ)`` without gaps.

    Raises
    ------
    ValueError
        If names, counts, or ``latent_dims`` fail validation.
    """
    dims = validate_latent_dims(node_type_names, latent_dims)
    assert dims is not None
    # Validate N_τ keys / positivity (shared with the shared-d row layout).
    _ = node_type_offsets(node_type_names, num_nodes_dict)
    slices: dict[str, slice] = {}
    cursor = 0
    for name in dims:
        width = dims[name]
        count = int(num_nodes_dict[name])
        stop = cursor + count * width
        slices[name] = slice(cursor, stop)
        cursor = stop
    return slices


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
