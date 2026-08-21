"""Piecewise-linear switched discrete Koopman operator.

A finite bank of LTI :class:`~koopman_graph.operators.KoopmanOperator`
maps. The active mode is an integer ``mode_index`` (default 0) or the
argmax of a latent gate. Each mode remains linear; this is not a nonlinear
latent operator and is not a parameter interpolant :math:`K(\\mu)`
(``Macesic2018Nonautonomous``). Carry regime coordinates on
:attr:`~koopman_graph.data.GraphSnapshotSequence.parameter_trajectory`.

Select via ``koopman="switched"``. Optional per-step ``phase_index``
selects a mode without mutating :attr:`mode_index` (time-of-day bins
from :func:`~koopman_graph.data.diurnal_phase_index`).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from koopman_graph.operators.contract import (
    InitMode,
    Parameterization,
    StabilityCertificate,
)
from koopman_graph.operators.control import ControlMode
from koopman_graph.operators.discrete import KoopmanOperator

DEFAULT_NUM_MODES = 2


class SwitchedKoopmanOperator(nn.Module):
    """Bank of discrete LTI maps with a discrete active mode.

    Parameters
    ----------
    latent_dim : int
        Latent width.
    num_modes : int, optional
        Number of LTI maps. Default is 2.
    init_mode, init_scale, parameterization, max_spectral_radius
        Forwarded to each mode operator.
    control_dim : int, optional
        Shared control dimension.
    control_mode : {"additive", "bilinear"}, optional
        Shared control coupling.
    bilinear_rank : int or None, optional
        Shared bilinear rank.
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
    ) -> None:
        """Initialize the switched LTI bank.

        Parameters
        ----------
        latent_dim : int
            Latent width of each mode.
        num_modes : int, optional
            Number of LTI maps. Default is 2.
        init_mode : InitMode, optional
            Weight initialization forwarded to each mode.
        init_scale : float, optional
            Initialization scale forwarded to each mode.
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
        self.mode_index = 0
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

    def set_mode(self, mode_index: int) -> None:
        """Select the active LTI map.

        Parameters
        ----------
        mode_index : int
            Index in ``[0, num_modes)``.
        """
        if not 0 <= int(mode_index) < self.num_modes:
            msg = f"mode_index must be in [0, {self.num_modes}), got {mode_index}"
            raise ValueError(msg)
        self.mode_index = int(mode_index)

    def _active(self) -> KoopmanOperator:
        """Return the selected LTI mode.

        Returns
        -------
        KoopmanOperator
            Mode at :attr:`mode_index`.
        """
        return self.modes[self.mode_index]  # type: ignore[return-value]

    def _mode_for(self, phase_index: int | None) -> KoopmanOperator:
        """Return the LTI mode for a step, optionally overriding the latch.

        Parameters
        ----------
        phase_index : int or None
            When set, select that bank index without writing
            :attr:`mode_index`. ``None`` uses the latched mode.

        Returns
        -------
        KoopmanOperator
            Mode used for this step.

        Raises
        ------
        ValueError
            If ``phase_index`` is outside ``[0, num_modes)``.
        """
        if phase_index is None:
            return self._active()
        index = int(phase_index)
        if not 0 <= index < self.num_modes:
            msg = f"phase_index must be in [0, {self.num_modes}), got {phase_index}"
            raise ValueError(msg)
        return self.modes[index]  # type: ignore[return-value]

    def infer_mode(self, z: Tensor) -> int:
        """Return the argmax gate mode from mean latents.

        Parameters
        ----------
        z : Tensor
            Latents with trailing dimension ``latent_dim``.

        Returns
        -------
        int
            Mode index in ``[0, num_modes)``.
        """
        pooled = z.reshape(-1, self.latent_dim).mean(dim=0)
        return int(torch.argmax(self.gate(pooled)).item())

    @property
    def matrix(self) -> Tensor:
        """Assembled ``K`` of the active mode.

        Returns
        -------
        Tensor
            Active-mode discrete map.
        """
        return self._active().matrix

    def bound_metric(self) -> Tensor:
        """Bound metric of the active mode.

        Returns
        -------
        Tensor
            Scalar bound.
        """
        return self._active().bound_metric()

    def stability_certificate(self) -> StabilityCertificate | None:
        """Certificate of the active mode, if any.

        Returns
        -------
        StabilityCertificate or None
            Factor certificate when the parameterization provides one.
        """
        return self._active().stability_certificate()

    def advance(
        self,
        z: Tensor,
        delta_t: float | Tensor | None = None,
        *,
        control: Tensor | None = None,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
        phase_index: int | None = None,
    ) -> Tensor:
        """Advance with the active LTI map.

        Parameters
        ----------
        z : Tensor
            Latent states.
        delta_t : float, Tensor, or None, optional
            Ignored (discrete operator).
        control : Tensor or None, optional
            Control input.
        edge_index : Tensor or None, optional
            Ignored (per-node modes).
        edge_weight : Tensor or None, optional
            Ignored.
        phase_index : int or None, optional
            Per-step mode override. Does not write :attr:`mode_index`.

        Returns
        -------
        Tensor
            Advanced latents.
        """
        del delta_t, edge_index, edge_weight
        return self._mode_for(phase_index).advance(z, control=control)

    def inverse_advance(
        self,
        z: Tensor,
        delta_t: float | Tensor | None = None,
        *,
        control: Tensor | None = None,
        inverse_matrix: Tensor | None = None,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
        phase_index: int | None = None,
    ) -> Tensor:
        """Inverse step of the active LTI map.

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
        phase_index : int or None, optional
            Per-step mode override. Does not write :attr:`mode_index`.

        Returns
        -------
        Tensor
            Inverse-advanced latents.
        """
        del delta_t, edge_index, edge_weight
        return self._mode_for(phase_index).inverse_advance(
            z, control=control, inverse_matrix=inverse_matrix
        )

    def forward(
        self,
        z: Tensor,
        control: Tensor | None = None,
        *,
        phase_index: int | None = None,
    ) -> Tensor:
        """Module forward: active-mode advance.

        Parameters
        ----------
        z : Tensor
            Latent states.
        control : Tensor or None, optional
            Control input.
        phase_index : int or None, optional
            Per-step mode override. Does not write :attr:`mode_index`.

        Returns
        -------
        Tensor
            Advanced latents.
        """
        return self.advance(z, control=control, phase_index=phase_index)
