r"""Graphon-sampled adjacency helper for transfer experiments.

Samples a dense adjacency from a kernel \(W:[0,1]^2\to[0,1]\) at \(N\)
uniform nodes. Theory bounds are cited, not proved in-repo.
"""

from __future__ import annotations

import torch
from torch import Tensor


def sample_graphon_adjacency(
    num_nodes: int,
    *,
    kernel: str = "constant",
    density: float = 0.3,
    generator: torch.Generator | None = None,
) -> Tensor:
    r"""Sample a symmetric 0/1 adjacency from a simple graphon.

    Parameters
    ----------
    num_nodes : int
        Node count \(N\).
    kernel : {"constant", "product"}, optional
        ``constant`` uses edge probability ``density``; ``product`` uses
        \(W(u,v)=u v\) after sorting uniform latent positions.
    density : float, optional
        Edge probability for the constant graphon.
    generator : torch.Generator or None, optional
        Optional RNG.

    Returns
    -------
    Tensor
        Integer ``edge_index`` with shape ``(2, E)`` (undirected, both
        orientations).

    Notes
    -----
    Dense-graph limit viewpoint of Lovász and Szegedy (2006;
    ``LovaszSzegedy2006``). In-repo transfer experiments cite those
    bounds; they are not proved here.
    """
    if num_nodes < 2:
        raise ValueError(f"num_nodes must be >= 2, got {num_nodes}")
    positions = torch.rand(num_nodes, generator=generator)
    if kernel == "constant":
        probs = torch.full((num_nodes, num_nodes), float(density))
    elif kernel == "product":
        probs = positions.unsqueeze(1) * positions.unsqueeze(0)
    else:
        raise ValueError(f"unknown graphon kernel {kernel!r}")
    probs = torch.triu(probs, diagonal=1)
    samples = torch.rand((num_nodes, num_nodes), generator=generator) < probs
    src, dst = samples.nonzero(as_tuple=True)
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    return edge_index
