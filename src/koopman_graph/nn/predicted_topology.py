"""Opt-in next-step topology head (link formation / dissolution).

Distinct from :class:`~koopman_graph.nn.AdaptiveAdjacency`, which is a
static Graph WaveNet self-adaptive adjacency.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class PredictedTopologyHead(nn.Module):
    """Predict pairwise edge logits from node latents.

    Parameters
    ----------
    latent_dim : int
        Node latent width.
    hidden_dim : int, optional
        MLP hidden width. Default is 32.
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 32) -> None:
        """Initialize the pairwise MLP.

        Parameters
        ----------
        latent_dim : int
            Node latent width.
        hidden_dim : int, optional
            MLP hidden width.
        """
        super().__init__()
        if latent_dim < 1:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        self.latent_dim = int(latent_dim)
        self.mlp = nn.Sequential(
            nn.Linear(2 * latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def pairwise_logits(self, z: Tensor) -> Tensor:
        """Return dense ``(N, N)`` logits.

        Parameters
        ----------
        z : Tensor
            Node latents ``(N, d)``.

        Returns
        -------
        Tensor
            Dense logits ``(N, N)``.
        """
        if z.ndim != 2:
            raise ValueError(f"z must have shape (N, d), got {tuple(z.shape)}")
        num_nodes = z.shape[0]
        left = z.unsqueeze(1).expand(num_nodes, num_nodes, self.latent_dim)
        right = z.unsqueeze(0).expand(num_nodes, num_nodes, self.latent_dim)
        pairs = torch.cat([left, right], dim=-1)
        logits = self.mlp(pairs).squeeze(-1)
        logits = logits.clone()
        logits.fill_diagonal_(-1e9)
        return logits

    def edge_index(
        self,
        z: Tensor,
        *,
        threshold: float = 0.0,
        top_k: int | None = None,
    ) -> Tensor:
        """Threshold or top-k the logits into a COO ``edge_index``.

        Parameters
        ----------
        z : Tensor
            Node latents.
        threshold : float, optional
            Logit threshold when ``top_k`` is None.
        top_k : int or None, optional
            If set, keep this many outgoing edges per node.

        Returns
        -------
        Tensor
            COO ``edge_index`` of shape ``(2, E)``.
        """
        logits = self.pairwise_logits(z)
        num_nodes = logits.shape[0]
        if top_k is not None:
            k = min(int(top_k), max(num_nodes - 1, 1))
            _, indices = torch.topk(logits, k=k, dim=-1)
            src = (
                torch.arange(num_nodes, device=z.device).unsqueeze(1).expand_as(indices)
            )
            return torch.stack([src.reshape(-1), indices.reshape(-1)], dim=0)
        src, dst = (logits > threshold).nonzero(as_tuple=True)
        if src.numel() == 0:
            eye = torch.arange(num_nodes, device=z.device)
            return torch.stack([eye, (eye + 1) % num_nodes], dim=0)
        return torch.stack([src, dst], dim=0)

    def forward(self, z: Tensor) -> Tensor:
        """Return pairwise logits.

        Parameters
        ----------
        z : Tensor
            Node latents.

        Returns
        -------
        Tensor
            Dense logits.
        """
        return self.pairwise_logits(z)
