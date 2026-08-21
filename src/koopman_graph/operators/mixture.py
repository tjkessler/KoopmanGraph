"""Convex mixture of discrete LTI Koopman operators.

Softmax weights over a latent window mix a bank of LTI maps. Distinct from
:class:`~koopman_graph.operators.GlobalLocalKoopmanOperator` (low-rank
correction of a single backbone) and from a parameter interpolant
:math:`K(\\mu)` (``Macesic2018Nonautonomous``). Carry regime coordinates
on :attr:`~koopman_graph.data.GraphSnapshotSequence.parameter_trajectory`.
Select via ``koopman="mixture"``.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from koopman_graph.operators.contract import InitMode, Parameterization
from koopman_graph.operators.control import ControlMode
from koopman_graph.operators.discrete import KoopmanOperator
from koopman_graph.operators.discrete_propagation import advance_step
from koopman_graph.operators.global_local import DEFAULT_LOCAL_WINDOW
from koopman_graph.operators.switched import DEFAULT_NUM_MODES


class MixtureKoopmanOperator(nn.Module):
    """Softmax mixture of discrete LTI maps.

    Parameters
    ----------
    latent_dim : int
        Latent width.
    num_modes : int, optional
        Mixture components. Default is 2.
    local_window : int, optional
        Gate lookback. Default matches global/local window.
    """

    def __init__(
        self,
        latent_dim: int,
        num_modes: int = DEFAULT_NUM_MODES,
        *,
        init_mode: InitMode = "identity_noise",
        init_scale: float = 1e-2,
        parameterization: Parameterization = "dense",
        max_spectral_radius: float = 0.99,
        control_dim: int = 0,
        control_mode: ControlMode = "additive",
        bilinear_rank: int | None = None,
        local_window: int = DEFAULT_LOCAL_WINDOW,
    ) -> None:
        """Initialize the softmax mixture of LTI maps.

        Parameters
        ----------
        latent_dim : int
            Latent width of each component.
        num_modes : int, optional
            Mixture components. Default is 2.
        init_mode : InitMode, optional
            Weight initialization forwarded to each component.
        init_scale : float, optional
            Initialization scale forwarded to each component.
        parameterization : Parameterization, optional
            Shared discrete parameterization.
        max_spectral_radius : float, optional
            Shared spectral-radius bound.
        control_dim : int, optional
            Shared control dimension.
        control_mode : {"additive", "bilinear"}, optional
            Shared control coupling.
        bilinear_rank : int or None, optional
            Shared bilinear rank.
        local_window : int, optional
            Gate lookback stored for factory round-trip.
        """
        super().__init__()
        if num_modes < 1:
            msg = f"num_modes must be positive, got {num_modes}"
            raise ValueError(msg)
        self.latent_dim = latent_dim
        self.num_modes = int(num_modes)
        self.control_dim = control_dim
        self.control_mode = control_mode
        self.parameterization = parameterization
        self.init_mode = init_mode
        self.init_scale = init_scale
        self.max_spectral_radius = max_spectral_radius
        self.bilinear_rank = bilinear_rank
        self.local_window = int(local_window)
        self.modes = nn.ModuleList(
            [
                KoopmanOperator(
                    latent_dim,
                    init_mode=init_mode,
                    init_scale=init_scale,
                    parameterization=parameterization,
                    max_spectral_radius=max_spectral_radius,
                    control_dim=control_dim,
                    control_mode=control_mode,
                    bilinear_rank=bilinear_rank,
                )
                for _ in range(self.num_modes)
            ]
        )
        self.gate = nn.Linear(latent_dim, self.num_modes)

    def mixture_weights(self, z: Tensor) -> Tensor:
        """Return softmax mixture weights from pooled latents.

        Parameters
        ----------
        z : Tensor
            Latents ``(..., latent_dim)``.

        Returns
        -------
        Tensor
            Weights with shape ``(num_modes,)``.
        """
        pooled = z.reshape(-1, self.latent_dim).mean(dim=0)
        return torch.softmax(self.gate(pooled), dim=-1)

    @property
    def matrix(self) -> Tensor:
        """Uniform mixture of component matrices (state-independent view).

        Returns
        -------
        Tensor
            Equal-weight mixture of mode matrices.
        """
        stacked = torch.stack([mode.matrix for mode in self.modes], dim=0)
        weights = torch.full(
            (self.num_modes,),
            1.0 / self.num_modes,
            dtype=stacked.dtype,
            device=stacked.device,
        )
        return torch.einsum("m,mij->ij", weights, stacked)

    def bound_metric(self) -> Tensor:
        """Maximum component bound metric.

        Returns
        -------
        Tensor
            Scalar bound.
        """
        return torch.stack([mode.bound_metric() for mode in self.modes]).max()

    def effective_matrix(self, z: Tensor) -> Tensor:
        """State-dependent mixture matrix.

        Parameters
        ----------
        z : Tensor
            Latents used to form the gate.

        Returns
        -------
        Tensor
            Convex combination of mode matrices.
        """
        weights = self.mixture_weights(z)
        stacked = torch.stack([mode.matrix for mode in self.modes], dim=0)
        return torch.einsum("m,mij->ij", weights, stacked)

    def advance(
        self,
        z: Tensor,
        delta_t: float | Tensor | None = None,
        *,
        control: Tensor | None = None,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        """Advance with the state-dependent mixture matrix.

        Parameters
        ----------
        z : Tensor
            Latent states.
        delta_t : float, Tensor, or None, optional
            Ignored.
        control : Tensor or None, optional
            Control input.
        edge_index : Tensor or None, optional
            Ignored.
        edge_weight : Tensor or None, optional
            Ignored.

        Returns
        -------
        Tensor
            Advanced latents.
        """
        del delta_t, edge_index, edge_weight
        matrix = self.effective_matrix(z)
        active = self.modes[0]
        coupling = (
            active.bilinear_matrices() if self.control_mode == "bilinear" else None
        )
        return advance_step(
            z,
            control,
            matrix=matrix,
            control_matrix=getattr(active, "B", None) if self.control_dim > 0 else None,
            control_dim=self.control_dim,
            control_mode=self.control_mode,
            latent_dim=self.latent_dim,
            coupling=coupling,
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
        """Inverse using mode 0 (not the gated map).

        Parameters
        ----------
        z : Tensor
            Latent states.
        delta_t : float, Tensor, or None, optional
            Ignored.
        control : Tensor or None, optional
            Control input.
        inverse_matrix : Tensor or None, optional
            Optional precomputed inverse.
        edge_index : Tensor or None, optional
            Ignored.
        edge_weight : Tensor or None, optional
            Ignored.

        Returns
        -------
        Tensor
            Inverse-advanced latents.
        """
        del delta_t, edge_index, edge_weight
        return self.modes[0].inverse_advance(
            z, control=control, inverse_matrix=inverse_matrix
        )

    def forward(self, z: Tensor, control: Tensor | None = None) -> Tensor:
        """Module forward.

        Parameters
        ----------
        z : Tensor
            Latent states.
        control : Tensor or None, optional
            Control input.

        Returns
        -------
        Tensor
            Advanced latents.
        """
        return self.advance(z, control=control)
