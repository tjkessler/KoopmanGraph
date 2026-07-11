"""Finite-dimensional Koopman operator for latent-state linear propagation."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

InitMode = Literal["identity", "identity_noise", "xavier"]


class KoopmanOperator(nn.Module):
    """Learnable finite-dimensional Koopman operator matrix **K**.

    Applies the same linear map to each node's latent vector. For input ``z`` with
    trailing dimension ``latent_dim``, the forward pass computes::

        z_next = z @ K.T

    where ``K`` has shape ``(latent_dim, latent_dim)``. Arbitrary leading dimensions
    are supported (e.g. ``(num_nodes, latent_dim)`` or
    ``(batch, num_nodes, latent_dim)``).

    Attributes
    ----------
    latent_dim : int
        Dimension of the latent space.
    init_mode : str
        Weight initialization strategy for ``K``.
    init_scale : float
        Noise scale used when ``init_mode="identity_noise"``.
    K : nn.Parameter
        Learnable Koopman matrix with shape ``(latent_dim, latent_dim)``.
    """

    def __init__(
        self,
        latent_dim: int,
        *,
        init_mode: InitMode = "identity_noise",
        init_scale: float = 1e-2,
    ) -> None:
        """Initialize the Koopman operator matrix.

        Parameters
        ----------
        latent_dim : int
            Dimension of the latent space (size of square matrix ``K``).
        init_mode : {"identity", "identity_noise", "xavier"}, optional
            Weight initialization strategy for ``K``. Default is
            ``"identity_noise"``.
        init_scale : float, optional
            Standard deviation of Gaussian noise added when
            ``init_mode="identity_noise"``. Default is ``1e-2``.

        Raises
        ------
        ValueError
            If ``latent_dim < 1`` or ``init_scale < 0``.
        """
        super().__init__()
        if latent_dim < 1:
            msg = f"latent_dim must be positive, got {latent_dim}"
            raise ValueError(msg)
        if init_scale < 0:
            msg = f"init_scale must be non-negative, got {init_scale}"
            raise ValueError(msg)

        self.latent_dim = latent_dim
        self.init_mode = init_mode
        self.init_scale = init_scale

        self.K = nn.Parameter(torch.empty(latent_dim, latent_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reinitialize ``K`` according to :attr:`init_mode`.

        Returns
        -------
        None
        """
        if self.init_mode == "identity":
            nn.init.eye_(self.K)
        elif self.init_mode == "identity_noise":
            nn.init.eye_(self.K)
            with torch.no_grad():
                self.K.add_(torch.randn_like(self.K) * self.init_scale)
        elif self.init_mode == "xavier":
            nn.init.xavier_uniform_(self.K)
        else:
            msg = f"Unknown init_mode: {self.init_mode!r}"
            raise ValueError(msg)

    def forward(self, z: Tensor) -> Tensor:
        """Advance latent states by one linear Koopman step.

        Parameters
        ----------
        z : Tensor
            Latent states with shape ``(..., latent_dim)``.

        Returns
        -------
        Tensor
            Advanced latent states with the same shape as ``z``.

        Raises
        ------
        ValueError
            If the trailing dimension of ``z`` does not match ``latent_dim``.
        """
        if z.shape[-1] != self.latent_dim:
            msg = (
                f"Expected trailing dimension {self.latent_dim}, "
                f"got shape {tuple(z.shape)}"
            )
            raise ValueError(msg)
        return z @ self.K.T
