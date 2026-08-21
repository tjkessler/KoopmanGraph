r"""Block-equivariant latent Koopman operator (scalar / vector / tensor).

Vector blocks are learnable multiples of :math:`I_3` (the unique linear
equivariant maps on \(\mathbb{R}^3\) under SO(3)). One additional irrep
is shipped: :math:`l=2` tensor blocks are multiples of :math:`I_5`
(:math:`2l+1=5`). Mixing across irreps is refused (Schur). Default
encoders may still project to invariant scalars. Optional
``[equivariance]`` extra is required only for the rotation test and
:class:`~koopman_graph.nn.E3EquivariantEncoder` — this module does not
import ``e3nn``.

Not a factory kind. Not a molecular MD production stack
(``Thomas2018TFN``, ``Geiger2022e3nn``).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from koopman_graph.operators.contract import InitMode, Parameterization
from koopman_graph.operators.discrete import KoopmanOperator
from koopman_graph.operators.discrete_propagation import advance_step

VECTOR_IRREP_DIM = 3
TENSOR_IRREP_L = 2
TENSOR_IRREP_DIM = 5

__all__ = [
    "TENSOR_IRREP_DIM",
    "TENSOR_IRREP_L",
    "VECTOR_IRREP_DIM",
    "EquivariantKoopmanOperator",
]


class EquivariantKoopmanOperator(nn.Module):
    """Block-diagonal scalar / vector / :math:`l=2` tensor operator.

    Parameters
    ----------
    n_scalars : int
        Number of invariant scalar latent channels.
    n_vectors : int, optional
        Number of 3-vector channels. Default is 0.
    n_tensors : int, optional
        Number of :math:`l=2` channels (width 5 each). Default is 0.
    """

    def __init__(
        self,
        n_scalars: int,
        n_vectors: int = 0,
        n_tensors: int = 0,
        *,
        init_mode: InitMode = "identity_noise",
        init_scale: float = 1e-2,
        parameterization: Parameterization = "dense",
        max_spectral_radius: float = 0.99,
    ) -> None:
        """Initialize scalar, vector, and optional tensor Koopman blocks.

        Parameters
        ----------
        n_scalars : int
            Number of invariant scalar latent channels.
        n_vectors : int, optional
            Number of 3-vector channels.
        n_tensors : int, optional
            Number of :math:`l=2` tensor channels.
        init_mode : InitMode, optional
            Initialization for the scalar block.
        init_scale : float, optional
            Initialization scale for the scalar block.
        parameterization : Parameterization, optional
            Parameterization of the scalar block.
        max_spectral_radius : float, optional
            Spectral-radius bound for the scalar block.

        Raises
        ------
        ValueError
            If a count is negative or every count is zero.
        """
        super().__init__()
        if isinstance(n_scalars, bool) or isinstance(n_vectors, bool):
            raise ValueError("n_scalars and n_vectors must be non-negative ints")
        if isinstance(n_tensors, bool):
            raise ValueError("n_tensors must be a non-negative int")
        if n_scalars < 0 or n_vectors < 0 or n_tensors < 0:
            msg = "n_scalars, n_vectors, and n_tensors must be non-negative"
            raise ValueError(msg)
        if n_scalars == 0 and n_vectors == 0 and n_tensors == 0:
            raise ValueError(
                "at least one scalar, vector, or tensor channel is required"
            )
        self.n_scalars = int(n_scalars)
        self.n_vectors = int(n_vectors)
        self.n_tensors = int(n_tensors)
        self.latent_dim = (
            self.n_scalars
            + VECTOR_IRREP_DIM * self.n_vectors
            + TENSOR_IRREP_DIM * self.n_tensors
        )
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
        self.tensor_scales = nn.Parameter(torch.ones(self.n_tensors))

    def _block_device_dtype(self) -> tuple[torch.device, torch.dtype]:
        """Device and dtype for assembled identity blocks.

        Returns
        -------
        tuple[torch.device, torch.dtype]
            Device and floating dtype of a live parameter.
        """
        if self.n_vectors > 0:
            return self.vector_scales.device, self.vector_scales.dtype
        if self.n_tensors > 0:
            return self.tensor_scales.device, self.tensor_scales.dtype
        assert self.scalar is not None
        matrix = self.scalar.matrix
        return matrix.device, matrix.dtype

    @property
    def matrix(self) -> Tensor:
        """Assembled block-diagonal operator.

        Returns
        -------
        Tensor
            Scalar block plus ``scale * I_3`` vector blocks and
            ``scale * I_5`` :math:`l=2` tensor blocks.
        """
        blocks: list[Tensor] = []
        device, dtype = self._block_device_dtype()
        if self.scalar is not None:
            blocks.append(self.scalar.matrix)
        eye3 = torch.eye(VECTOR_IRREP_DIM, device=device, dtype=dtype)
        for scale in self.vector_scales:
            blocks.append(scale * eye3)
        eye5 = torch.eye(TENSOR_IRREP_DIM, device=device, dtype=dtype)
        for scale in self.tensor_scales:
            blocks.append(scale * eye5)
        return torch.block_diag(*blocks) if blocks else torch.empty(0, 0)

    def bound_metric(self) -> Tensor:
        """Max of scalar bound and absolute vector / tensor scales.

        Returns
        -------
        Tensor
            Scalar bound.
        """
        parts: list[Tensor] = []
        if self.n_vectors > 0:
            parts.append(self.vector_scales.abs().max())
        if self.n_tensors > 0:
            parts.append(self.tensor_scales.abs().max())
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
        """Advance scalar, vector, and tensor blocks independently.

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
