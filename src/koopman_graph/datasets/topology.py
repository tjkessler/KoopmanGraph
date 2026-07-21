"""Dataset topology helpers: shared edge builders and typed load payloads.

Provides bidirectional path/ring ``edge_index`` constructors used by synthetic
and nonlinear benchmarks, plus :class:`TopologyPayload` for cached
``load_topology`` APIs. Import builders from this module rather than
duplicating private copies in peer dataset modules.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


def path_edge_index(num_nodes: int) -> Tensor:
    """Build bidirectional path-graph edges.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the path.

    Returns
    -------
    Tensor
        Edge index with shape ``(2, num_edges)``.
    """
    if num_nodes < 2:
        return torch.zeros((2, 0), dtype=torch.long)

    src: list[int] = []
    dst: list[int] = []
    for node in range(num_nodes - 1):
        src.extend([node, node + 1])
        dst.extend([node + 1, node])
    return torch.tensor([src, dst], dtype=torch.long)


def ring_edge_index(num_nodes: int) -> Tensor:
    """Build bidirectional ring-graph edges.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the ring.

    Returns
    -------
    Tensor
        Edge index with shape ``(2, num_edges)``.
    """
    if num_nodes < 2:
        return torch.zeros((2, 0), dtype=torch.long)

    src: list[int] = []
    dst: list[int] = []
    for node in range(num_nodes):
        nxt = (node + 1) % num_nodes
        src.extend([node, nxt])
        dst.extend([nxt, node])
    return torch.tensor([src, dst], dtype=torch.long)


@dataclass(frozen=True)
class TopologyPayload(Mapping[str, Any]):
    """Frozen topology tables returned by ``load_topology`` APIs.

    Shared required fields are ``edge_index`` and ``num_nodes``. Dataset-specific
    metadata (IEEE bus tables, METR-LA sensor ids / edge weights, source URLs)
    is optional. Mapping access (``payload["edge_index"]``) exposes only fields
    that are not ``None``, so notebook and test dict-style access keeps working.

    Attributes
    ----------
    edge_index : Tensor
        COO edge index with shape ``(2, E)``.
    num_nodes : int
        Number of nodes in the graph.
    edge_weight : Tensor, optional
        Optional edge weights with shape ``(E,)``.
    initial_features : Tensor, optional
        Optional initial node features (IEEE 118).
    bus_ids : Tensor, optional
        Optional MATPOWER bus identifiers (IEEE 118).
    sensor_ids : list of str, optional
        Optional sensor identifiers (METR-LA).
    base_mva : float, optional
        System base MVA (IEEE 118).
    source_url : str, optional
        Topology source URL (IEEE 118).
    source_h5_url : str, optional
        Speed-history source URL (METR-LA).
    normalized_k : float, optional
        Gaussian-kernel threshold used for METR-LA adjacency.
    """

    edge_index: Tensor
    num_nodes: int
    edge_weight: Tensor | None = None
    initial_features: Tensor | None = None
    bus_ids: Tensor | None = None
    sensor_ids: list[str] | None = None
    base_mva: float | None = None
    source_url: str | None = None
    source_h5_url: str | None = None
    normalized_k: float | None = None

    def _present(self) -> dict[str, Any]:
        """Return non-``None`` fields as a plain dict.

        Returns
        -------
        dict
            Mapping of present field names to values.
        """
        data: dict[str, Any] = {
            "edge_index": self.edge_index,
            "num_nodes": self.num_nodes,
        }
        for name in (
            "edge_weight",
            "initial_features",
            "bus_ids",
            "sensor_ids",
            "base_mva",
            "source_url",
            "source_h5_url",
            "normalized_k",
        ):
            value = getattr(self, name)
            if value is not None:
                data[name] = value
        return data

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict of present fields (for cache serialization).

        Returns
        -------
        dict
            Mapping of present field names to values.
        """
        return self._present()

    def __getitem__(self, key: str) -> Any:
        """Return a present field by name.

        Parameters
        ----------
        key : str
            Field name (for example ``"edge_index"``).

        Returns
        -------
        object
            Value of the requested field.

        Raises
        ------
        KeyError
            If ``key`` is absent or the corresponding optional field is ``None``.
        """
        data = self._present()
        try:
            return data[key]
        except KeyError as exc:
            msg = f"{key!r}"
            raise KeyError(msg) from exc

    def __iter__(self) -> Iterator[str]:
        """Iterate present field names.

        Yields
        ------
        str
            Names of fields that are not ``None``.
        """
        return iter(self._present())

    def __len__(self) -> int:
        """Return the number of present fields.

        Returns
        -------
        int
            Count of non-``None`` fields in the payload.
        """
        return len(self._present())

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> TopologyPayload:
        """Build a payload from a mapping (e.g. a loaded cache dict).

        Parameters
        ----------
        data : Mapping
            Mapping with at least ``edge_index`` and ``num_nodes``.

        Returns
        -------
        TopologyPayload
            Frozen payload with optional fields taken from ``data`` when present.
        """
        if isinstance(data, TopologyPayload):
            return data
        return cls(
            edge_index=data["edge_index"],
            num_nodes=int(data["num_nodes"]),
            edge_weight=data.get("edge_weight"),
            initial_features=data.get("initial_features"),
            bus_ids=data.get("bus_ids"),
            sensor_ids=data.get("sensor_ids"),
            base_mva=data.get("base_mva"),
            source_url=data.get("source_url"),
            source_h5_url=data.get("source_h5_url"),
            normalized_k=data.get("normalized_k"),
        )
