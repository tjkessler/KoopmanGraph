"""Growable node-universe remapping (open-world MVP).

Callers supply the index map into a finite :math:`N_{\\max}`. Silent merge
of unrelated universes is refused. This is **not** automatic entity
resolution and does **not** allow unbounded node growth.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor
from torch_geometric.data import Data

from koopman_graph.data.validation import validate_entity_ids

__all__ = [
    "EntityRemap",
    "remap_node_features",
]


def remap_node_features(
    features: Tensor,
    *,
    old_index: Tensor,
    new_capacity: int,
) -> Tensor:
    r"""Scatter node features into a larger union capacity.

    Parameters
    ----------
    features : Tensor
        Source features ``(N_old, F)``.
    old_index : Tensor
        Integer destinations in ``[0, new_capacity)`` with length ``N_old``.
    new_capacity : int
        Target union size :math:`N_{\max}`.

    Returns
    -------
    Tensor
        Features of shape ``(new_capacity, F)``. Unused rows are zeros.

    Raises
    ------
    ValueError
        If the map is not injective or indices are out of range.
    """
    if features.ndim != 2:
        raise ValueError(
            f"features must have shape (N, F), got {tuple(features.shape)}"
        )
    if old_index.ndim != 1 or int(old_index.shape[0]) != int(features.shape[0]):
        raise ValueError("old_index must be 1-D with length equal to N_old")
    if int(new_capacity) < int(features.shape[0]):
        raise ValueError("new_capacity must be at least N_old")
    if int(old_index.min()) < 0 or int(old_index.max()) >= int(new_capacity):
        raise ValueError("old_index contains IDs outside [0, new_capacity)")
    unique = torch.unique(old_index)
    if int(unique.numel()) != int(old_index.numel()):
        raise ValueError("old_index must be injective (no silent universe merge)")
    out = torch.zeros(
        int(new_capacity),
        features.shape[1],
        dtype=features.dtype,
        device=features.device,
    )
    out[old_index] = features
    return out


def _remap_edge_index(edge_index: Tensor, index: Tensor) -> Tensor:
    """Map COO endpoints through an injective node remap.

    Parameters
    ----------
    edge_index : Tensor
        Source COO with shape ``(2, E)``.
    index : Tensor
        Destinations with length ``N_old``.

    Returns
    -------
    Tensor
        Remapped COO with the same ``E``.

    Raises
    ------
    ValueError
        If an endpoint is outside ``[0, N_old)``.
    """
    if edge_index.ndim != 2 or int(edge_index.shape[0]) != 2:
        msg = f"edge_index must have shape (2, E), got {tuple(edge_index.shape)}"
        raise ValueError(msg)
    n_old = int(index.shape[0])
    if edge_index.numel() == 0:
        return edge_index.to(dtype=torch.long, device=index.device).reshape(2, 0)
    if int(edge_index.min()) < 0 or int(edge_index.max()) >= n_old:
        msg = (
            "edge_index endpoints must lie in "
            f"[0, {n_old}), got min={int(edge_index.min())}, "
            f"max={int(edge_index.max())}"
        )
        raise ValueError(msg)
    mapped = index.to(device=edge_index.device)[edge_index.long()]
    return mapped


@dataclass(frozen=True, eq=False)
class EntityRemap:
    """Injective placement of source nodes into a fixed union.

    ``entity_ids`` names the **target** rows (length :math:`N_{\\max}`).
    ``index`` places each source row into that union. Unused union rows
    stay zero / absent. This is not entity resolution across unrelated
    graphs.

    Attributes
    ----------
    entity_ids : tuple of str or int
        Unique keys for the fixed union, length :math:`N_{\\max}`.
    index : Tensor
        Long destinations ``(N_old,)`` injective into
        ``[0, N_max)``.
    """

    entity_ids: tuple[str | int, ...]
    index: Tensor

    def __post_init__(self) -> None:
        """Validate union names and the injective index.

        Raises
        ------
        ValueError
            If ``entity_ids`` is empty, ``index`` is not a 1-D injective
            long tensor into ``[0, N_max)``, or names collide.
        """
        ids = validate_entity_ids(self.entity_ids, num_nodes=len(self.entity_ids))
        if len(ids) < 1:
            msg = "EntityRemap.entity_ids must be non-empty (finite N_max)"
            raise ValueError(msg)
        object.__setattr__(self, "entity_ids", ids)
        index = self.index
        if not isinstance(index, Tensor) or index.ndim != 1:
            msg = "EntityRemap.index must be a 1-D tensor"
            raise ValueError(msg)
        if int(index.numel()) < 1:
            msg = "EntityRemap.index must contain at least one source row"
            raise ValueError(msg)
        stored = index.detach().to(dtype=torch.long).clone()
        object.__setattr__(self, "index", stored)
        remap_node_features(
            torch.zeros(
                int(stored.shape[0]),
                1,
                dtype=torch.float32,
                device=stored.device,
            ),
            old_index=stored,
            new_capacity=len(ids),
        )

    @property
    def n_max(self) -> int:
        """Return the fixed union size :math:`N_{\\max}`.

        Returns
        -------
        int
            ``len(entity_ids)``.
        """
        return len(self.entity_ids)

    @property
    def presence_mask(self) -> Tensor:
        """Return a boolean mask that is true on mapped union rows.

        Returns
        -------
        Tensor
            Shape ``(N_max,)`` on ``index.device``.
        """
        mask = torch.zeros(
            self.n_max,
            dtype=torch.bool,
            device=self.index.device,
        )
        mask[self.index] = True
        return mask

    def apply_features(self, features: Tensor) -> Tensor:
        """Scatter source features into the union.

        Parameters
        ----------
        features : Tensor
            Source features ``(N_old, F)``.

        Returns
        -------
        Tensor
            Union features ``(N_max, F)``.
        """
        return remap_node_features(
            features,
            old_index=self.index,
            new_capacity=self.n_max,
        )

    def apply_snapshot(self, snapshot: Data) -> Data:
        """Scatter one homogeneous snapshot into the union.

        Remaps ``x`` and ``edge_index`` (and ``edge_weight`` when present).
        Padded rows are zeros with no incident edges.

        Parameters
        ----------
        snapshot : Data
            Homogeneous snapshot with ``N_old`` nodes.

        Returns
        -------
        Data
            Snapshot with ``N_max`` nodes.

        Raises
        ------
        TypeError
            If ``snapshot`` is not a homogeneous ``Data``.
        ValueError
            If node count or edges do not match ``index``.
        """
        if type(snapshot) is not Data:
            msg = (
                "EntityRemap.apply_snapshot supports homogeneous Data only, "
                f"got {type(snapshot).__name__}"
            )
            raise TypeError(msg)
        if snapshot.x is None:
            msg = "EntityRemap.apply_snapshot requires Data.x"
            raise ValueError(msg)
        remapped = Data(x=self.apply_features(snapshot.x))
        edge_index = snapshot.edge_index
        if edge_index is None:
            remapped.edge_index = torch.empty(
                (2, 0),
                dtype=torch.long,
                device=self.index.device,
            )
            return remapped
        remapped.edge_index = _remap_edge_index(edge_index, self.index)
        weight = getattr(snapshot, "edge_weight", None)
        if weight is not None:
            remapped.edge_weight = weight
        return remapped

    def apply_snapshots(
        self,
        snapshots: Sequence[Data],
    ) -> tuple[list[Data], Tensor]:
        """Remap a homogeneous trajectory into the shared union.

        Parameters
        ----------
        snapshots : sequence of Data
            Snapshots that share ``N_old`` with ``index``.

        Returns
        -------
        remapped : list of Data
            Union-capacity snapshots.
        presence : Tensor
            Boolean mask ``(T, N_max)`` true on mapped rows (constant
            across time).

        Raises
        ------
        ValueError
            If ``snapshots`` is empty.
        """
        if not snapshots:
            msg = "apply_snapshots requires at least one snapshot"
            raise ValueError(msg)
        remapped = [self.apply_snapshot(snapshot) for snapshot in snapshots]
        presence = self.presence_mask.unsqueeze(0).expand(len(snapshots), -1).clone()
        return remapped, presence
