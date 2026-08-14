r"""Laplacian-structured (Hodge \(L_0\)) networked discrete Koopman operator.

Uses combinatorial graph Laplacian message passing in place of normalized
adjacency: ``Z_next = Z K_selfᵀ + (L_0 Z) K_hodgeᵀ``. Select via
``koopman="hodge"``.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch_geometric.utils import get_laplacian

from koopman_graph.operators.graph import GraphKoopmanOperator


class HodgeKoopmanOperator(GraphKoopmanOperator):
    """Graph operator whose neighbor factor multiplies a Laplacian matvec.

    Parameters
    ----------
    latent_dim : int
        Latent width. Remaining kwargs match
        :class:`~koopman_graph.operators.GraphKoopmanOperator` except
        ``adjacency`` is unused (Laplacian replaces normalized adjacency).
    """

    def _sparse_neighbor_term(
        self,
        z: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor | None,
    ) -> Tensor:
        """Apply combinatorial Laplacian then ``K_nbr``.

        Parameters
        ----------
        z : Tensor
            Node latents ``(N, d)``.
        edge_index : Tensor
            COO edges.
        edge_weight : Tensor or None
            Optional weights.

        Returns
        -------
        Tensor
            Neighbor contribution ``(N, d)``.
        """
        lap_index, lap_weight = get_laplacian(
            edge_index,
            edge_weight=edge_weight,
            num_nodes=z.shape[0],
        )
        laplacian_z = torch.zeros_like(z)
        src, dst = lap_index[0], lap_index[1]
        laplacian_z.index_add_(0, dst, lap_weight.unsqueeze(-1) * z[src])
        return self.apply_tied_neighbor(laplacian_z)
