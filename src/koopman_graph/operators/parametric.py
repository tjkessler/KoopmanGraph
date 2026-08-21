"""Parameter interpolant of discrete LTI Koopman operators.

Convex combination :math:`K(\\mu)=\\sum_j \\alpha_j(\\mu) K_j` with simplex
or RBF weights on a bank of per-node maps. Distinct from latent-gated
:class:`~koopman_graph.operators.mixture.MixtureKoopmanOperator` and
piecewise :class:`~koopman_graph.operators.switched.SwitchedKoopmanOperator`.
Select via ``koopman="parametric"``. Carry :math:`\\mu_t` on
:attr:`~koopman_graph.data.GraphSnapshotSequence.parameter_trajectory`.

This module must not import :mod:`koopman_graph.model` or
:mod:`koopman_graph.data`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from koopman_graph.operators.contract import InitMode, Parameterization
from koopman_graph.operators.control import ControlMode
from koopman_graph.operators.discrete import KoopmanOperator
from koopman_graph.operators.discrete_propagation import advance_step
from koopman_graph.operators.switched import DEFAULT_NUM_MODES

WeightKind = Literal["rbf", "simplex"]
DEFAULT_PARAMETER_DIM = 1
DEFAULT_WEIGHT_KIND: WeightKind = "rbf"
LENGTHSCALE_FLOOR = 1e-6
_RIDGE_FLOOR = 0.0

#: Parameterizations whose convex combination keeps the declared constraint.
INTERPOLANT_SAFE_PARAMETERIZATIONS: frozenset[str] = frozenset(
    {"dense", "row_stochastic", "doubly_stochastic"}
)

__all__ = [
    "DEFAULT_PARAMETER_DIM",
    "DEFAULT_WEIGHT_KIND",
    "INTERPOLANT_SAFE_PARAMETERIZATIONS",
    "LeaveOneRegimeOutReport",
    "ParametricKoopmanOperator",
    "WeightKind",
    "leave_one_regime_out",
]


def _reject_interpolant_parameterization(parameterization: str) -> None:
    """Refuse mixes that would silently drop a structural constraint.

    Parameters
    ----------
    parameterization : str
        Shared discrete parameterization of each mode.

    Raises
    ------
    ValueError
        If a convex combination does not preserve the constraint.
    """
    if parameterization in INTERPOLANT_SAFE_PARAMETERIZATIONS:
        return
    msg = (
        "parametric interpolant refuses parameterization="
        f"{parameterization!r}: a convex combination of mode matrices "
        "does not preserve that constraint. Use 'dense', "
        "'row_stochastic', or 'doubly_stochastic'"
    )
    raise ValueError(msg)


def _ridge_edmd(z: Tensor, z_next: Tensor, *, ridge: float) -> Tensor:
    """Ridge EDMD for :math:`z_{+} \\approx z K^{\\top}`.

    Parameters
    ----------
    z : Tensor
        Latents with shape ``(n, d)``.
    z_next : Tensor
        Successor latents with the same shape.
    ridge : float
        Tikhonov weight on ``K`` (must be ``>= 0``).

    Returns
    -------
    Tensor
        Estimated ``K`` with shape ``(d, d)``.

    Raises
    ------
    ValueError
        If shapes disagree or ``ridge`` is negative.
    """
    if z.ndim != 2 or z_next.ndim != 2:
        msg = (
            "ridge EDMD requires 2-D latents (n, d), "
            f"got z={tuple(z.shape)} z_next={tuple(z_next.shape)}"
        )
        raise ValueError(msg)
    if z.shape != z_next.shape:
        msg = (
            "z and z_next must share shape (n, d), "
            f"got z={tuple(z.shape)} z_next={tuple(z_next.shape)}"
        )
        raise ValueError(msg)
    if ridge < _RIDGE_FLOOR:
        msg = f"ridge must be >= 0, got {ridge}"
        raise ValueError(msg)
    n_pairs, dim = z.shape
    if n_pairs < 1:
        msg = "ridge EDMD requires at least one pair"
        raise ValueError(msg)
    gram = z.T @ z
    if ridge > 0.0:
        gram = gram + ridge * torch.eye(dim, dtype=z.dtype, device=z.device)
    k_t = torch.linalg.solve(gram, z.T @ z_next)
    return k_t.T


def _one_step_mse(z: Tensor, z_next: Tensor, matrix: Tensor) -> Tensor:
    """Mean squared residual of :math:`z_{+} - z K^{\\top}`.

    Parameters
    ----------
    z, z_next : Tensor
        Latent pairs with shape ``(n, d)``.
    matrix : Tensor
        Discrete map ``K`` with shape ``(d, d)``.

    Returns
    -------
    Tensor
        Scalar MSE.
    """
    residual = z_next - z @ matrix.T
    return residual.square().mean()


class ParametricKoopmanOperator(nn.Module):
    """Discrete per-node interpolant :math:`K(\\mu)=\\sum_j \\alpha_j(\\mu) K_j`.

    ``weight_kind="rbf"`` uses L1-normalized Gaussian kernels at learnable
    anchors. ``weight_kind="simplex"`` uses softmax of a learned affine map
    of :math:`\\mu`. Both produce simplex weights, so the mix is a convex
    combination of the mode matrices.

    Parameters
    ----------
    latent_dim : int
        Latent width of each mode.
    num_modes : int, optional
        Number of LTI maps. Default is 2.
    parameter_dim : int, optional
        Width :math:`d_\\mu` of the regime coordinate. Default is 1.
    weight_kind : {"rbf", "simplex"}, optional
        Interpolation weights. Default is ``"rbf"``.
    init_mode, init_scale, parameterization, max_spectral_radius
        Forwarded to each mode operator.
    control_dim : int, optional
        Shared control dimension.
    control_mode : {"additive", "bilinear"}, optional
        Shared control coupling.
    bilinear_rank : int or None, optional
        Shared bilinear rank.

    Raises
    ------
    ValueError
        If ``num_modes`` / ``parameter_dim`` are invalid, ``weight_kind``
        is unknown, or ``parameterization`` is not preserved by a convex
        combination.

    Notes
    -----
    This is not a cocycle :math:`K(t)` and not a latent-gated mixture
    :math:`\\sum_i w_i(z) K_i`. A single stationary spectrum of
    :attr:`matrix` (equal-weight mix) is generally inappropriate for
    nonautonomous families (``Macesic2018Nonautonomous``). Export of
    conditioned maps is refused until a discrete homogeneous subset is
    traceable.

    References
    ----------
    Maćešić, S., Črnjarić-Žic, N. and Mezić, I. (2018). Koopman operator
    family spectrum for nonautonomous systems. *SIAM Journal on Applied
    Dynamical Systems* 17:2478–2515. doi:10.1137/17M1133610
    (``Macesic2018Nonautonomous``).
    """

    def __init__(
        self,
        latent_dim: int,
        num_modes: int = DEFAULT_NUM_MODES,
        *,
        parameter_dim: int = DEFAULT_PARAMETER_DIM,
        weight_kind: WeightKind = DEFAULT_WEIGHT_KIND,
        init_mode: InitMode = "identity_noise",
        init_scale: float = 1e-2,
        parameterization: Parameterization = "dense",
        max_spectral_radius: float = 0.99,
        control_dim: int = 0,
        control_mode: ControlMode = "additive",
        bilinear_rank: int | None = None,
    ) -> None:
        """Initialize the interpolant bank and weight map.

        Parameters
        ----------
        latent_dim : int
            Latent width of each component.
        num_modes : int, optional
            Number of LTI maps. Default is 2.
        parameter_dim : int, optional
            Regime-coordinate width. Default is 1.
        weight_kind : {"rbf", "simplex"}, optional
            Interpolation kind. Default is ``"rbf"``.
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
        """
        super().__init__()
        if num_modes < 1:
            msg = f"num_modes must be positive, got {num_modes}"
            raise ValueError(msg)
        if parameter_dim < 1:
            msg = f"parameter_dim must be >= 1, got {parameter_dim}"
            raise ValueError(msg)
        if weight_kind not in {"rbf", "simplex"}:
            msg = f"weight_kind must be 'rbf' or 'simplex', got {weight_kind!r}"
            raise ValueError(msg)
        _reject_interpolant_parameterization(parameterization)
        self.latent_dim = latent_dim
        self.num_modes = int(num_modes)
        self.parameter_dim = int(parameter_dim)
        self.weight_kind: WeightKind = weight_kind
        self.control_dim = control_dim
        self.control_mode = control_mode
        self.parameterization = parameterization
        self.init_mode = init_mode
        self.init_scale = init_scale
        self.max_spectral_radius = max_spectral_radius
        self.bilinear_rank = bilinear_rank
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
        anchors = torch.zeros(self.num_modes, self.parameter_dim)
        if self.num_modes > 1:
            anchors[:, 0] = torch.linspace(0.0, 1.0, self.num_modes)
        self.anchors = nn.Parameter(anchors)
        self._raw_lengthscale = nn.Parameter(torch.zeros(()))
        self.gate = nn.Linear(self.parameter_dim, self.num_modes)
        self._stored_parameters: Tensor | None = None

    @property
    def lengthscale(self) -> Tensor:
        """Positive RBF length-scale (softplus of a raw parameter).

        Returns
        -------
        Tensor
            Scalar length-scale ``> 0``.
        """
        return torch.nn.functional.softplus(self._raw_lengthscale) + LENGTHSCALE_FLOOR

    def set_lengthscale(self, value: float | Tensor) -> None:
        """Write a positive length-scale into the unconstrained parameter.

        Parameters
        ----------
        value : float or Tensor
            Target length-scale (must be ``> 0``).

        Raises
        ------
        ValueError
            If ``value`` is not positive and finite.
        """
        scale = float(value)
        if not math.isfinite(scale) or scale <= 0.0:
            msg = f"lengthscale must be a positive finite float, got {value!r}"
            raise ValueError(msg)
        # Inverse softplus: log(exp(y) - 1) with y = scale - floor.
        shifted = scale - LENGTHSCALE_FLOOR
        if shifted <= 0.0:
            msg = (
                "lengthscale must exceed "
                f"{LENGTHSCALE_FLOOR:g} after the positivity floor, got {scale}"
            )
            raise ValueError(msg)
        inverse = math.log(math.expm1(shifted))
        with torch.no_grad():
            self._raw_lengthscale.fill_(inverse)

    def set_anchors(self, anchors: Tensor) -> None:
        """Copy regime anchors in place.

        Parameters
        ----------
        anchors : Tensor
            Shape ``(num_modes, parameter_dim)``.

        Raises
        ------
        ValueError
            If the shape is wrong or values are non-finite.
        """
        if anchors.shape != (self.num_modes, self.parameter_dim):
            msg = (
                "anchors must have shape "
                f"({self.num_modes}, {self.parameter_dim}), "
                f"got {tuple(anchors.shape)}"
            )
            raise ValueError(msg)
        if anchors.is_floating_point() and not torch.all(torch.isfinite(anchors)):
            msg = "anchors must be finite"
            raise ValueError(msg)
        with torch.no_grad():
            self.anchors.copy_(anchors.to(dtype=self.anchors.dtype))

    def set_parameters(self, parameters: Tensor) -> None:
        """Store a default :math:`\\mu` used when ``advance`` omits it.

        Parameters
        ----------
        parameters : Tensor
            Regime coordinate with shape ``(parameter_dim,)``.
        """
        resolved = self._resolve_parameters(parameters)
        self._stored_parameters = resolved.detach()

    def _resolve_parameters(self, parameters: Tensor | None) -> Tensor:
        """Validate :math:`\\mu` or fall back to the stored coordinate.

        Parameters
        ----------
        parameters : Tensor or None
            Caller :math:`\\mu`, or ``None`` to use :meth:`set_parameters`.

        Returns
        -------
        Tensor
            Coordinate with shape ``(parameter_dim,)`` on the module device.

        Raises
        ------
        ValueError
            If :math:`\\mu` is missing, has the wrong shape, or is non-finite.
        """
        if parameters is None:
            stored = self._stored_parameters
            if stored is None:
                msg = (
                    "ParametricKoopmanOperator.advance requires parameters "
                    "with shape (parameter_dim,) or a prior set_parameters "
                    "call; got None"
                )
                raise ValueError(msg)
            return stored
        if parameters.ndim != 1 or int(parameters.shape[0]) != self.parameter_dim:
            msg = (
                "parameters must have shape "
                f"({self.parameter_dim},), got {tuple(parameters.shape)}"
            )
            raise ValueError(msg)
        finite = torch.isfinite(parameters)
        if parameters.is_floating_point() and not bool(torch.all(finite)):
            msg = "parameters must be finite"
            raise ValueError(msg)
        return parameters.to(device=self.anchors.device, dtype=self.anchors.dtype)

    def interpolation_weights(self, parameters: Tensor | None = None) -> Tensor:
        """Return simplex weights :math:`\\alpha(\\mu)`.

        Parameters
        ----------
        parameters : Tensor or None, optional
            Regime coordinate with shape ``(parameter_dim,)``. When omitted,
            uses :meth:`set_parameters`.

        Returns
        -------
        Tensor
            Nonnegative weights with shape ``(num_modes,)`` summing to 1.
        """
        mu = self._resolve_parameters(parameters)
        if self.weight_kind == "simplex":
            return torch.softmax(self.gate(mu), dim=-1)
        dist_sq = torch.sum((self.anchors - mu) ** 2, dim=-1)
        scale = self.lengthscale * self.lengthscale
        logits = -dist_sq / scale
        return torch.softmax(logits, dim=-1)

    @property
    def matrix(self) -> Tensor:
        """Equal-weight mix of mode matrices (state-independent view).

        Returns
        -------
        Tensor
            Uniform convex combination of the :math:`K_j`. This is not
            :math:`K(\\mu)` at a stored coordinate.
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

    def effective_matrix(self, parameters: Tensor | None = None) -> Tensor:
        """Interpolant matrix :math:`K(\\mu)`.

        Parameters
        ----------
        parameters : Tensor or None, optional
            Regime coordinate. When omitted, uses :meth:`set_parameters`.

        Returns
        -------
        Tensor
            Convex combination of mode matrices.
        """
        weights = self.interpolation_weights(parameters)
        stacked = torch.stack([mode.matrix for mode in self.modes], dim=0)
        return torch.einsum("m,mij->ij", weights, stacked)

    def _controlled_step(
        self,
        z: Tensor,
        *,
        matrix: Tensor,
        control: Tensor | None,
    ) -> Tensor:
        """Advance with a frozen interpolant matrix and mode-0 control.

        Parameters
        ----------
        z : Tensor
            Latent states.
        matrix : Tensor
            Effective ``K``.
        control : Tensor or None
            Control input.

        Returns
        -------
        Tensor
            Advanced latents.
        """
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

    def advance(
        self,
        z: Tensor,
        delta_t: float | Tensor | None = None,
        *,
        control: Tensor | None = None,
        parameters: Tensor | None = None,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        """Advance with :math:`K(\\mu)`.

        Parameters
        ----------
        z : Tensor
            Latent states.
        delta_t : float, Tensor, or None, optional
            Ignored.
        control : Tensor or None, optional
            Control input.
        parameters : Tensor or None, optional
            Regime coordinate with shape ``(parameter_dim,)``.
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
        matrix = self.effective_matrix(parameters)
        return self._controlled_step(z, matrix=matrix, control=control)

    def inverse_advance(
        self,
        z: Tensor,
        delta_t: float | Tensor | None = None,
        *,
        control: Tensor | None = None,
        inverse_matrix: Tensor | None = None,
        parameters: Tensor | None = None,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        """Inverse of the interpolant at :math:`\\mu` (not a mix of inverses).

        Parameters
        ----------
        z : Tensor
            Latent states.
        delta_t : float, Tensor, or None, optional
            Ignored.
        control : Tensor or None, optional
            Control input.
        inverse_matrix : Tensor or None, optional
            Optional precomputed inverse of :math:`K(\\mu)`.
        parameters : Tensor or None, optional
            Regime coordinate.
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
        if inverse_matrix is None:
            inverse_matrix = torch.linalg.inv(self.effective_matrix(parameters))
        return self.modes[0].inverse_advance(
            z, control=control, inverse_matrix=inverse_matrix
        )

    def forward(
        self,
        z: Tensor,
        control: Tensor | None = None,
        *,
        parameters: Tensor | None = None,
    ) -> Tensor:
        """Module forward.

        Parameters
        ----------
        z : Tensor
            Latent states.
        control : Tensor or None, optional
            Control input.
        parameters : Tensor or None, optional
            Regime coordinate.

        Returns
        -------
        Tensor
            Advanced latents.
        """
        return self.advance(z, control=control, parameters=parameters)


@dataclass(frozen=True, eq=False)
class LeaveOneRegimeOutReport:
    """One-step comparison of an interpolant against pooled LTI EDMD.

    Attributes
    ----------
    held_out_mu : Tensor
        Regime coordinate that was held out, shape ``(d_mu,)``.
    interpolant_mse : float
        Hold-out one-step MSE of the RBF interpolant.
    pooled_lti_mse : float
        Hold-out one-step MSE of ridge EDMD on concatenated train pairs.
    n_train_pairs : int
        Number of latent pairs used to fit the interpolant and the pooled map.
    n_holdout_pairs : int
        Number of hold-out pairs.
    ridge : float
        Tikhonov weight used for both EDMD fits.
    """

    held_out_mu: Tensor
    interpolant_mse: float
    pooled_lti_mse: float
    n_train_pairs: int
    n_holdout_pairs: int
    ridge: float


def leave_one_regime_out(
    regimes: Sequence[tuple[Tensor, Tensor, Tensor]],
    *,
    hold_out: int,
    ridge: float = 1e-4,
) -> LeaveOneRegimeOutReport:
    """Fit an RBF interpolant on all but one :math:`\\mu` and compare to pooled LTI.

    Each regime is ``(mu, z, z_next)`` with ``mu`` shape ``(d_mu,)`` and
    latent pairs shape ``(n_j, d)``. Train regimes receive per-regime ridge
    EDMD maps placed at those :math:`\\mu` as RBF anchors. The pooled map is
    ridge EDMD on the concatenated train pairs. Both are scored by one-step
    MSE on the held-out regime. This is an identification helper on latents,
    not a GNN forecast claim.

    Parameters
    ----------
    regimes : sequence of (Tensor, Tensor, Tensor)
        Per-regime ``(mu, z, z_next)``. At least two regimes are required.
    hold_out : int
        Index of the held-out regime.
    ridge : float, optional
        Tikhonov weight. Default is ``1e-4``.

    Returns
    -------
    LeaveOneRegimeOutReport
        Hold-out MSEs and pair counts.

    Raises
    ------
    ValueError
        If there are fewer than two regimes, ``hold_out`` is out of range,
        widths disagree, or a train regime is empty.
    """
    n_regimes = len(regimes)
    if n_regimes < 2:
        msg = f"leave_one_regime_out requires at least 2 regimes, got {n_regimes}"
        raise ValueError(msg)
    if not 0 <= int(hold_out) < n_regimes:
        msg = f"hold_out must be in [0, {n_regimes}), got {hold_out}"
        raise ValueError(msg)
    hold = int(hold_out)
    mu_h, z_h, z_next_h = regimes[hold]
    if mu_h.ndim != 1 or int(mu_h.shape[0]) < 1:
        msg = (
            "held-out mu must have shape (d_mu,) with d_mu >= 1, "
            f"got {tuple(mu_h.shape)}"
        )
        raise ValueError(msg)
    parameter_dim = int(mu_h.shape[0])
    train_mus: list[Tensor] = []
    train_maps: list[Tensor] = []
    train_z: list[Tensor] = []
    train_z_next: list[Tensor] = []
    latent_dim: int | None = None
    for index, (mu, z, z_next) in enumerate(regimes):
        if index == hold:
            continue
        if mu.shape != mu_h.shape:
            msg = (
                f"regime {index} mu shape {tuple(mu.shape)} must match "
                f"held-out {tuple(mu_h.shape)}"
            )
            raise ValueError(msg)
        k_j = _ridge_edmd(z, z_next, ridge=ridge)
        if latent_dim is None:
            latent_dim = int(k_j.shape[0])
        train_mus.append(mu)
        train_maps.append(k_j)
        train_z.append(z)
        train_z_next.append(z_next)
    if latent_dim is None:
        msg = "leave_one_regime_out requires at least one train regime"
        raise ValueError(msg)
    interpolant = ParametricKoopmanOperator(
        latent_dim,
        num_modes=len(train_maps),
        parameter_dim=parameter_dim,
        weight_kind="rbf",
        parameterization="dense",
        init_mode="identity",
        control_dim=0,
    )
    interpolant.set_anchors(torch.stack(train_mus, dim=0))
    pairwise = torch.cdist(interpolant.anchors, interpolant.anchors)
    off = pairwise[pairwise > 0]
    if off.numel() > 0:
        interpolant.set_lengthscale(float(off.median().item()))
    for mode, matrix in zip(interpolant.modes, train_maps, strict=True):
        mode.set_dense_matrix(matrix)
    k_hold = interpolant.effective_matrix(mu_h)
    interpolant_mse = float(_one_step_mse(z_h, z_next_h, k_hold).item())
    pooled = _ridge_edmd(
        torch.cat(train_z, dim=0),
        torch.cat(train_z_next, dim=0),
        ridge=ridge,
    )
    pooled_mse = float(_one_step_mse(z_h, z_next_h, pooled).item())
    n_train = int(sum(int(block.shape[0]) for block in train_z))
    return LeaveOneRegimeOutReport(
        held_out_mu=mu_h.detach().clone(),
        interpolant_mse=interpolant_mse,
        pooled_lti_mse=pooled_mse,
        n_train_pairs=n_train,
        n_holdout_pairs=int(z_h.shape[0]),
        ridge=float(ridge),
    )
