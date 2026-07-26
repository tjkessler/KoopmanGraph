"""Self-adaptive pairwise adjacency (Graph WaveNet construction).

Learns node embeddings ``E_1, E_2 ∈ R^{N×k}`` and forms the row-stochastic
adjacency::

    Â_adp = softmax(ReLU(E_1 E_2ᵀ))

following Wu et al., Graph WaveNet (IJCAI 2019; ``Wu2019WaveNet``). The module
materializes a dense COO ``(edge_index, edge_weight)`` for encode / networked
advance / spectrum. This is an inductive bias for forecasting, **not** a causal
structure-discovery procedure.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

DEFAULT_TOPOLOGY_EMBEDDING_DIM = 8


class AdaptiveAdjacency(nn.Module):
    """Learned dense pairwise adjacency from source/target node embeddings.

    Parameters are allocated lazily via :meth:`set_num_nodes` (static per fit:
    changing ``N`` after the first allocation raises).

    Attributes
    ----------
    embedding_dim : int
        Node embedding width ``k``.
    num_nodes : int or None
        Bound graph size ``N``, or ``None`` before the first allocation.
    """

    def __init__(
        self,
        embedding_dim: int = DEFAULT_TOPOLOGY_EMBEDDING_DIM,
        num_nodes: int | None = None,
    ) -> None:
        """Initialize optional embeddings.

        Parameters
        ----------
        embedding_dim : int, optional
            Embedding width ``k``. Default is ``8``.
        num_nodes : int or None, optional
            Optional initial node count. When ``None``, call
            :meth:`set_num_nodes` before :meth:`dense_adjacency`.

        Raises
        ------
        ValueError
            If ``embedding_dim`` is not positive or ``num_nodes`` is invalid.
        """
        super().__init__()
        if embedding_dim < 1:
            msg = f"embedding_dim must be positive, got {embedding_dim}"
            raise ValueError(msg)
        self.embedding_dim = embedding_dim
        self.num_nodes: int | None = None
        self.register_parameter("source_embedding", None)
        self.register_parameter("target_embedding", None)
        if num_nodes is not None:
            self.set_num_nodes(num_nodes)

    def set_num_nodes(
        self,
        num_nodes: int,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        """Allocate or validate embeddings for a fixed node count.

        Parameters
        ----------
        num_nodes : int
            Graph size ``N``.
        device : torch.device, str, or None, optional
            Allocation device. Defaults to existing embedding device, else CPU.

        Raises
        ------
        ValueError
            If ``num_nodes`` is not positive, or differs from a prior binding.
        """
        if num_nodes < 1:
            msg = f"num_nodes must be positive, got {num_nodes}"
            raise ValueError(msg)
        if self.num_nodes == num_nodes and self.source_embedding is not None:
            return
        if self.num_nodes is not None and self.num_nodes != num_nodes:
            msg = (
                "AdaptiveAdjacency is static per fit: num_nodes was "
                f"{self.num_nodes}, got {num_nodes}. Rebuild the model for a "
                "different graph size"
            )
            raise ValueError(msg)

        if device is None:
            device = (
                self.source_embedding.device
                if self.source_embedding is not None
                else torch.device("cpu")
            )
        self.num_nodes = num_nodes
        self.source_embedding = nn.Parameter(
            torch.randn(num_nodes, self.embedding_dim, device=device) * 0.1
        )
        self.target_embedding = nn.Parameter(
            torch.randn(num_nodes, self.embedding_dim, device=device) * 0.1
        )

    def dense_adjacency(self) -> Tensor:
        """Return the row-stochastic dense adjacency ``(N, N)``.

        Returns
        -------
        Tensor
            ``softmax(ReLU(E_1 E_2ᵀ), dim=1)``.

        Raises
        ------
        RuntimeError
            If embeddings have not been allocated.
        """
        if self.source_embedding is None or self.target_embedding is None:
            msg = "AdaptiveAdjacency requires set_num_nodes before dense_adjacency"
            raise RuntimeError(msg)
        product = self.source_embedding @ self.target_embedding.transpose(0, 1)
        return torch.softmax(torch.relu(product), dim=1)

    def materialize(self) -> tuple[Tensor, Tensor]:
        """Convert dense ``Â`` into COO ``edge_index`` / ``edge_weight``.

        Includes every entry of the ``N×N`` matrix (self-loops included) so
        encode, networked advance, and spectrum share the same adjacency.

        Returns
        -------
        edge_index : Tensor
            Edge index with shape ``(2, N²)``.
        edge_weight : Tensor
            Edge weights with shape ``(N²,)`` (row-major flatten of ``Â``).
        """
        adjacency = self.dense_adjacency()
        num_nodes = adjacency.shape[0]
        rows = torch.arange(num_nodes, device=adjacency.device).repeat_interleave(
            num_nodes
        )
        cols = torch.arange(num_nodes, device=adjacency.device).repeat(num_nodes)
        edge_index = torch.stack([rows, cols], dim=0)
        edge_weight = adjacency.reshape(-1)
        return edge_index, edge_weight

    def forward(self) -> tuple[Tensor, Tensor]:
        """Materialize learned ``(edge_index, edge_weight)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            See summary line."""
        return self.materialize()
