r"""Deep probabilistic Koopman VAE MVP (encoder weights, linear \(K\)).

Distinct from :class:`~koopman_graph.uq.BayesianKoopmanUQ` (Laplace on
linear factors only).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch_geometric.data import Data

from koopman_graph.nn.encoder import GNNEncoder
from koopman_graph.operators.discrete import KoopmanOperator


@dataclass(frozen=True)
class VAELatent:
    """Reparameterized latent draw.

    Attributes
    ----------
    z : Tensor
        Sampled latents.
    kl : Tensor
        Scalar KL to a standard normal.
    """

    z: Tensor
    kl: Tensor


class KoopmanVAEEncoder(nn.Module):
    """GCN encoder emitting ``μ`` / ``log σ²`` then a linear Koopman map.

    Parameters
    ----------
    in_channels, hidden_channels, latent_dim : int
        Feature widths.

    Notes
    -----
    Distinct from Laplace factor UQ.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        latent_dim: int,
    ) -> None:
        """Initialize the variational encoder and linear ``K``.

        Parameters
        ----------
        in_channels, hidden_channels, latent_dim : int
            Feature widths.
        """
        super().__init__()
        self.encoder = GNNEncoder(in_channels, hidden_channels, 2 * latent_dim)
        self.latent_dim = int(latent_dim)
        self.koopman = KoopmanOperator(latent_dim, init_mode="identity")

    def encode(self, data: Data) -> VAELatent:
        """Reparameterize node latents.

        Parameters
        ----------
        data : Data
            Snapshot with ``x`` / ``edge_index``.

        Returns
        -------
        VAELatent
            Reparameterized draw and KL term.
        """
        stats = self.encoder(data)
        mu, logvar = stats.split(self.latent_dim, dim=-1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        kl = -0.5 * torch.mean(1.0 + logvar - mu.square() - logvar.exp())
        return VAELatent(z=z, kl=kl)

    def advance(self, z: Tensor) -> Tensor:
        """Linear latent advance.

        Parameters
        ----------
        z : Tensor
            Latent sample.

        Returns
        -------
        Tensor
            Advanced latents.
        """
        return self.koopman.advance(z)
