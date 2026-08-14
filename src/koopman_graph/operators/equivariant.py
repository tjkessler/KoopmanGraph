r"""Block-equivariant latent Koopman operator (scalar + vector channels).

Vector blocks are learnable multiples of the identity (the unique linear
equivariant maps on \(\mathbb{R}^3\) under SO(3)). Optional ``[equivariance]``
path; not a full steerable-generator library.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from koopman_graph.operators.contract import InitMode, Parameterization
from koopman_graph.operators.discrete import KoopmanOperator
from koopman_graph.operators.discrete_propagation import advance_step


class EquivariantKoopmanOperator(nn.Module):
    """Block-diagonal scalar/vector latent operator.

    Parameters
    ----------
    n_scalars : int
        Number of invariant scalar latent channels.
    n_vectors : int, optional
        Number of 3-vector channels. Default is 0 (scalars only).
    """

    def __init__(
        self,
        n_scalars: int,
        n_vectors: int = 0,
        *,
        init_mode: InitMode = "identity_noise",
        init_scale: float = 1e-2,
        parameterization: Parameterization = "dense",
        max_spectral_radius: float = 0.99,
    ) -> None:
        """Initialize scalar and vector Koopman blocks.

        Parameters
        ----------
        n_scalars : int
            Number of invariant scalar latent channels.
        n_vectors : int, optional
            Number of 3-vector channels.
        init_mode : InitMode, optional
            Initialization for the scalar block.
        init_scale : float, optional
            Initialization scale for the scalar block.
        parameterization : Parameterization, optional
            Parameterization of the scalar block.
        max_spectral_radius : float, optional
            Spectral-radius bound for the scalar block.
        """
        super().__init__()
        if n_scalars < 0 or n_vectors < 0:
            msg = "n_scalars and n_vectors must be non-negative"
            raise ValueError(msg)
        if n_scalars == 0 and n_vectors == 0:
            raise ValueError("at least one scalar or vector channel is required")
        self.n_scalars = int(n_scalars)
        self.n_vectors = int(n_vectors)
        self.latent_dim = self.n_scalars + 3 * self.n_vectors
        self.control_dim = 0
        self.control_mode = "additive"
        self.parameterization = parameterization
        self.scalar = (
            KoopmanOperator(
                self.n_scalars,
                init_mode=init_mode,
                init_scale=init_scale,
                parameterization=parameterization,
                max_spectral_radius=max_spectral_radius,
                control_dim=0,
            )
            if self.n_scalars > 0
            else None
        )
        self.vector_scales = nn.Parameter(torch.ones(self.n_vectors))

    @property
    def matrix(self) -> Tensor:
        """Assembled block-diagonal operator.

        Returns
        -------
        Tensor
            Scalar block plus ``scale * I_3`` vector blocks.
        """
        blocks: list[Tensor] = []
        device = self.vector_scales.device
        dtype = self.vector_scales.dtype
        if self.scalar is not None:
            blocks.append(self.scalar.matrix)
        eye3 = torch.eye(3, device=device, dtype=dtype)
        for scale in self.vector_scales:
            blocks.append(scale * eye3)
        return torch.block_diag(*blocks) if blocks else torch.empty(0, 0)

    def bound_metric(self) -> Tensor:
        """Max of scalar bound and absolute vector scales.

        Returns
        -------
        Tensor
            Scalar bound.
        """
        parts: list[Tensor] = [self.vector_scales.abs().max()]
        if self.scalar is not None:
            parts.append(self.scalar.bound_metric())
        return torch.stack(parts).max()

    def advance(
        self,
        z: Tensor,
        delta_t: float | Tensor | None = None,
        *,
        control: Tensor | None = None,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        """Advance scalar and vector blocks independently.

        Parameters
        ----------
        z : Tensor
            Latents ``(..., latent_dim)``.
        delta_t : float, Tensor, or None, optional
            Ignored.
        control : Tensor or None, optional
            Ignored (uncontrolled).
        edge_index : Tensor or None, optional
            Ignored.
        edge_weight : Tensor or None, optional
            Ignored.

        Returns
        -------
        Tensor
            Advanced latents.
        """
        del delta_t, control, edge_index, edge_weight
        return advance_step(
            z,
            None,
            matrix=self.matrix,
            control_matrix=None,
            control_dim=0,
            control_mode="additive",
            latent_dim=self.latent_dim,
        )

    def inverse_advance(
        self,
        z: Tensor,
        delta_t: float | Tensor | None = None,
        *,
        control: Tensor | None = None,
        inverse_matrix: Tensor | None = None,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        """Dense inverse of the assembled block operator.

        Parameters
        ----------
        z : Tensor
            Latents.
        delta_t : float, Tensor, or None, optional
            Ignored.
        control : Tensor or None, optional
            Ignored.
        inverse_matrix : Tensor or None, optional
            Optional replacement for :attr:`matrix`.
        edge_index : Tensor or None, optional
            Ignored.
        edge_weight : Tensor or None, optional
            Ignored.

        Returns
        -------
        Tensor
            Inverse-advanced latents.
        """
        del delta_t, control, edge_index, edge_weight
        matrix = self.matrix if inverse_matrix is None else inverse_matrix
        inverse = torch.linalg.pinv(matrix)
        return z @ inverse.T

    def forward(self, z: Tensor) -> Tensor:
        """Module forward.

        Parameters
        ----------
        z : Tensor
            Latents.

        Returns
        -------
        Tensor
            Advanced latents.
        """
        return self.advance(z)
