"""ERA5-slice teaching mesh and dataset-card contract.

CI uses a tiny generated mesh. This is not production GraphCast / ERA5
training.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch_geometric.data import Data

from koopman_graph.data import GraphSnapshotSequence


@dataclass(frozen=True)
class ERA5SliceCard:
    """Fetch / checksum metadata for an ERA5 teaching slice.

    Attributes
    ----------
    name : str
        Slice identifier.
    source_url : str
        Documented fetch location (not downloaded in default CI).
    sha256 : str
        Expected checksum when a real slice is fetched.
    num_nodes : int
        Teaching mesh size.
    """

    name: str
    source_url: str
    sha256: str
    num_nodes: int


def era5_slice_card() -> ERA5SliceCard:
    """Return the documented ERA5 teaching-slice card.

    Returns
    -------
    ERA5SliceCard
        Fetch/checksum metadata for the tiny mesh.
    """
    return ERA5SliceCard(
        name="era5-tiny-mesh",
        source_url="https://cds.climate.copernicus.eu/",
        sha256="0" * 64,
        num_nodes=8,
    )


def generate_tiny_era5_mesh(
    *,
    num_nodes: int = 8,
    num_timesteps: int = 6,
    seed: int = 0,
) -> GraphSnapshotSequence:
    """Generate a ring-mesh teaching surrogate for ERA5 / GraphCast tests.

    Parameters
    ----------
    num_nodes : int, optional
        Mesh size. Default is 8.
    num_timesteps : int, optional
        Snapshot count.
    seed : int, optional
        RNG seed.

    Returns
    -------
    GraphSnapshotSequence
        Ring-mesh snapshots.
    """
    generator = torch.Generator().manual_seed(int(seed))
    src = torch.arange(num_nodes)
    dst = (src + 1) % num_nodes
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    snapshots: list[Data] = []
    for _ in range(int(num_timesteps)):
        features = torch.randn(num_nodes, 2, generator=generator)
        snapshots.append(Data(x=features, edge_index=edge_index))
    return GraphSnapshotSequence(snapshots)
