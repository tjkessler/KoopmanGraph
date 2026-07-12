"""Finite-dimensional Koopman operator for latent-state linear propagation."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

InitMode = Literal["identity", "identity_noise", "xavier"]
Parameterization = Literal["dense", "odo"]


def _cayley_orthogonal(skew_params: Tensor) -> Tensor:
    """Build an orthogonal matrix via the Cayley transform.

    Parameters
    ----------
    skew_params : Tensor
        Square parameter matrix; only its skew-symmetric part is used.

    Returns
    -------
    Tensor
        Orthogonal matrix with the same shape as ``skew_params``.
    """
    skew = 0.5 * (skew_params - skew_params.T)
    identity = torch.eye(skew.shape[0], device=skew.device, dtype=skew.dtype)
    return torch.linalg.solve(identity - skew, identity + skew)


def _bounded_diagonal(raw: Tensor, max_radius: float) -> Tensor:
    """Map unconstrained parameters to a bounded diagonal matrix.

    Parameters
    ----------
    raw : Tensor
        Unconstrained diagonal parameters with shape ``(latent_dim,)``.
    max_radius : float
        Maximum absolute value on the diagonal.

    Returns
    -------
    Tensor
        Diagonal matrix with spectral values in ``[-max_radius, max_radius]``.
    """
    values = torch.tanh(raw) * max_radius
    return torch.diag(values)


class KoopmanOperator(nn.Module):
    """Learnable finite-dimensional Koopman operator matrix **K**.

    Applies the same linear map to each node's latent vector. For input ``z`` with
    trailing dimension ``latent_dim``, the uncontrolled forward pass computes::

        z_next = z @ K.T

    When :attr:`control_dim` is positive, exogenous inputs drive the transition::

        z_next = z @ K.T + u @ B

    where ``K`` has shape ``(latent_dim, latent_dim)`` and ``B`` has shape
    ``(control_dim, latent_dim)``. Global controls ``u`` with shape
    ``(control_dim,)`` are broadcast to every node; per-node controls use shape
    ``(num_nodes, control_dim)``. Arbitrary leading dimensions are supported
    (e.g. ``(num_nodes, latent_dim)`` or ``(batch, num_nodes, latent_dim)``).

    Attributes
    ----------
    latent_dim : int
        Dimension of the latent space.
    control_dim : int
        Dimension of exogenous control inputs. Zero disables control.
    init_mode : str
        Weight initialization strategy for ``K``.
    init_scale : float
        Noise scale used when ``init_mode="identity_noise"``.
    parameterization : str
        Parameterization used for ``K`` (``"dense"`` or ``"odo"``).
    max_spectral_radius : float
        Upper bound on the spectral radius when ``parameterization="odo"``.
    """

    def __init__(
        self,
        latent_dim: int,
        *,
        init_mode: InitMode = "identity_noise",
        init_scale: float = 1e-2,
        parameterization: Parameterization = "dense",
        max_spectral_radius: float = 1.0,
        control_dim: int = 0,
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
        parameterization : {"dense", "odo"}, optional
            Matrix parameterization. ``"dense"`` stores ``K`` directly.
            ``"odo"`` factorizes ``K = O_1 D O_2^\\top`` with orthogonal
            factors via Cayley transforms and a bounded diagonal ``D``.
            Default is ``"dense"``.
        max_spectral_radius : float, optional
            Maximum absolute eigenvalue magnitude enforced by the ``"odo"``
            parameterization. Default is ``1.0``.
        control_dim : int, optional
            Dimension of exogenous control inputs. When ``0``, the operator
            is uncontrolled. When positive, a learnable input matrix ``B``
            with shape ``(control_dim, latent_dim)`` is added. Default is
            ``0``.

        Raises
        ------
        ValueError
            If ``latent_dim < 1``, ``init_scale < 0``,
            ``max_spectral_radius <= 0``, or ``control_dim < 0``.
        """
        super().__init__()
        if latent_dim < 1:
            msg = f"latent_dim must be positive, got {latent_dim}"
            raise ValueError(msg)
        if init_scale < 0:
            msg = f"init_scale must be non-negative, got {init_scale}"
            raise ValueError(msg)
        if max_spectral_radius <= 0:
            msg = f"max_spectral_radius must be positive, got {max_spectral_radius}"
            raise ValueError(msg)
        if control_dim < 0:
            msg = f"control_dim must be non-negative, got {control_dim}"
            raise ValueError(msg)

        self.latent_dim = latent_dim
        self.init_mode = init_mode
        self.init_scale = init_scale
        self.parameterization = parameterization
        self.max_spectral_radius = max_spectral_radius
        self.control_dim = control_dim

        if parameterization == "dense":
            self.register_parameter(
                "K",
                nn.Parameter(torch.empty(latent_dim, latent_dim)),
            )
            self.reset_parameters()
        elif parameterization == "odo":
            self.cayley_O1 = nn.Parameter(torch.zeros(latent_dim, latent_dim))
            self.cayley_O2 = nn.Parameter(torch.zeros(latent_dim, latent_dim))
            self.diag_raw = nn.Parameter(torch.zeros(latent_dim))
            self.reset_parameters()
        else:
            msg = f"Unknown parameterization: {parameterization!r}"
            raise ValueError(msg)

        if control_dim > 0:
            self.B = nn.Parameter(torch.zeros(control_dim, latent_dim))
            self.reset_control_parameters()

    def reset_control_parameters(self) -> None:
        """Reinitialize the control input matrix ``B``.

        Returns
        -------
        None
        """
        if self.control_dim <= 0:
            return
        nn.init.zeros_(self.B)

    def control_term(self, u: Tensor, *, num_nodes: int | None = None) -> Tensor:
        """Map control inputs to a latent-space offset ``u @ B``.

        Parameters
        ----------
        u : Tensor
            Global control with shape ``(control_dim,)`` or per-node control
            with shape ``(num_nodes, control_dim)``.
        num_nodes : int or None, optional
            Expected node count when ``u`` is per-node. Used for validation
            only.

        Returns
        -------
        Tensor
            Latent offset with shape ``(latent_dim,)`` for global control or
            ``(num_nodes, latent_dim)`` for per-node control.

        Raises
        ------
        ValueError
            If :attr:`control_dim` is zero, ``u`` has invalid shape, or
            per-node ``u`` does not match ``num_nodes``.
        """
        if self.control_dim == 0:
            msg = "control_term requires control_dim > 0"
            raise ValueError(msg)
        if u.ndim == 1:
            if u.shape[0] != self.control_dim:
                msg = (
                    f"Expected global control shape ({self.control_dim},), "
                    f"got {tuple(u.shape)}"
                )
                raise ValueError(msg)
            return u @ self.B
        if u.ndim == 2:
            if u.shape[1] != self.control_dim:
                msg = (
                    f"Expected per-node control shape (num_nodes, {self.control_dim}), "
                    f"got {tuple(u.shape)}"
                )
                raise ValueError(msg)
            if num_nodes is not None and u.shape[0] != num_nodes:
                msg = f"Per-node control has {u.shape[0]} rows, expected {num_nodes}"
                raise ValueError(msg)
            return u @ self.B
        msg = (
            "control input must have shape (control_dim,) for global control "
            f"or (num_nodes, control_dim) for per-node control, got {tuple(u.shape)}"
        )
        raise ValueError(msg)

    def _broadcast_control_term(self, z: Tensor, control_term: Tensor) -> Tensor:
        """Broadcast a global control offset to match latent state shape.

        Parameters
        ----------
        z : Tensor
            Latent states with shape ``(..., latent_dim)``.
        control_term : Tensor
            Global control offset with shape ``(latent_dim,)``.

        Returns
        -------
        Tensor
            Broadcast control offset with the same shape as ``z``.
        """
        view_shape = (1,) * (z.ndim - 1) + (self.latent_dim,)
        return control_term.view(view_shape).expand_as(z)

    @property
    def K(self) -> Tensor:
        """Assembled Koopman matrix with shape ``(latent_dim, latent_dim)``.

        For ``parameterization="dense"`` this is the learnable parameter.
        For ``parameterization="odo"`` it is assembled from orthogonal factors
        and a bounded diagonal matrix.

        Returns
        -------
        Tensor
            Current operator matrix ``K``.
        """
        dense_k = self._parameters.get("K")
        if dense_k is not None:
            return dense_k
        return self._assemble_odo_matrix()

    def reset_parameters(self) -> None:
        """Reinitialize operator parameters according to :attr:`init_mode`.

        Returns
        -------
        None
        """
        if self.parameterization == "dense":
            self._reset_dense_parameters()
        else:
            self._reset_odo_parameters()

    def _reset_dense_parameters(self) -> None:
        """Reinitialize the dense learnable matrix ``K``.

        Returns
        -------
        None
        """
        matrix = self._parameters["K"]
        if self.init_mode == "identity":
            nn.init.eye_(matrix)
        elif self.init_mode == "identity_noise":
            nn.init.eye_(matrix)
            with torch.no_grad():
                matrix.add_(torch.randn_like(matrix) * self.init_scale)
        elif self.init_mode == "xavier":
            nn.init.xavier_uniform_(matrix)
        else:
            msg = f"Unknown init_mode: {self.init_mode!r}"
            raise ValueError(msg)

    def _identity_diag_raw(self) -> float:
        """Return raw diagonal parameters for a near-identity ODO operator.

        Returns
        -------
        float
            Unconstrained diagonal parameter mapped near unit eigenvalues.
        """
        target = min(1.0, self.max_spectral_radius) * (1.0 - 1e-6)
        ratio = target / self.max_spectral_radius
        return float(torch.atanh(torch.tensor(ratio)).item())

    def _reset_odo_parameters(self) -> None:
        """Reinitialize Cayley and diagonal ODO parameters.

        Returns
        -------
        None
        """
        nn.init.zeros_(self.cayley_O1)
        nn.init.zeros_(self.cayley_O2)
        if self.init_mode == "identity":
            nn.init.constant_(self.diag_raw, self._identity_diag_raw())
        elif self.init_mode == "identity_noise":
            nn.init.constant_(self.diag_raw, self._identity_diag_raw())
            with torch.no_grad():
                noise = torch.randn_like(self.diag_raw) * self.init_scale
                current = torch.tanh(self.diag_raw) * self.max_spectral_radius
                updated = (current + noise).clamp(
                    min=-self.max_spectral_radius + 1e-6,
                    max=self.max_spectral_radius - 1e-6,
                )
                self.diag_raw.copy_(torch.atanh(updated / self.max_spectral_radius))
        elif self.init_mode == "xavier":
            nn.init.xavier_uniform_(self.cayley_O1)
            nn.init.xavier_uniform_(self.cayley_O2)
            nn.init.uniform_(self.diag_raw, -0.5, 0.5)
        else:
            msg = f"Unknown init_mode: {self.init_mode!r}"
            raise ValueError(msg)

    def _odo_orthogonal_factors(self) -> tuple[Tensor, Tensor]:
        """Build orthogonal factors for the ODO parameterization.

        Returns
        -------
        tuple of Tensor
            Orthogonal matrices ``(O_1, O_2)``.
        """
        return _cayley_orthogonal(self.cayley_O1), _cayley_orthogonal(self.cayley_O2)

    def _odo_diagonal(self) -> Tensor:
        """Build the bounded diagonal factor for the ODO parameterization.

        Returns
        -------
        Tensor
            Diagonal matrix with bounded eigenvalues.
        """
        return _bounded_diagonal(self.diag_raw, self.max_spectral_radius)

    def _assemble_odo_matrix(self) -> Tensor:
        """Assemble ``K = O_1 D O_2^T`` from ODO factors.

        Returns
        -------
        Tensor
            Assembled operator matrix.
        """
        o1, o2 = self._odo_orthogonal_factors()
        diagonal = self._odo_diagonal()
        return o1 @ diagonal @ o2.T

    def spectral_radius(self) -> Tensor:
        """Return the spectral radius of the assembled operator matrix.

        Returns
        -------
        Tensor
            Scalar tensor with the maximum eigenvalue magnitude.
        """
        if self.parameterization == "odo":
            return torch.tanh(self.diag_raw).abs().max() * self.max_spectral_radius
        eigenvalues = torch.linalg.eigvals(self.K)
        return eigenvalues.abs().max()

    def forward(self, z: Tensor, control: Tensor | None = None) -> Tensor:
        """Advance latent states by one linear Koopman step.

        When :attr:`control_dim` is positive, the controlled update is::

            z_next = z @ K.T + control_effect

        where ``control_effect`` is ``u @ B`` broadcast for global control
        ``u`` with shape ``(control_dim,)`` or applied per node when ``u`` has
        shape ``(num_nodes, control_dim)``.

        Parameters
        ----------
        z : Tensor
            Latent states with shape ``(..., latent_dim)``.
        control : Tensor or None, optional
            Exogenous control input applied during this step. Required when
            :attr:`control_dim` is positive.

        Returns
        -------
        Tensor
            Advanced latent states with the same shape as ``z``.

        Raises
        ------
        ValueError
            If the trailing dimension of ``z`` does not match ``latent_dim``,
            controls are missing for a controlled operator, or ``control`` has
            an invalid shape.
        """
        if z.shape[-1] != self.latent_dim:
            msg = (
                f"Expected trailing dimension {self.latent_dim}, "
                f"got shape {tuple(z.shape)}"
            )
            raise ValueError(msg)
        z_next = z @ self.K.T
        if self.control_dim == 0:
            if control is not None:
                msg = "control input provided to an uncontrolled operator"
                raise ValueError(msg)
            return z_next
        if control is None:
            msg = "control input is required when control_dim > 0"
            raise ValueError(msg)
        offset = self.control_term(
            control,
            num_nodes=z.shape[-2] if z.ndim >= 2 else None,
        )
        if control.ndim == 1:
            offset = self._broadcast_control_term(z, offset)
        return z_next + offset

    def inverse_step(
        self,
        z: Tensor,
        *,
        control: Tensor | None = None,
        inverse_matrix: Tensor | None = None,
    ) -> Tensor:
        """Apply one inverse Koopman step to recover the previous latent state.

        For forward dynamics ``z_{t+1} = z_t @ K.T + u_t @ B``, this returns
        an estimate of ``z_t`` from ``z_{t+1}`` and the control ``u_t`` that
        drove the transition.

        Parameters
        ----------
        z : Tensor
            Latent states at time ``t+1`` with shape ``(..., latent_dim)``.
        control : Tensor or None, optional
            Control input that drove the forward transition. Required when
            :attr:`control_dim` is positive.
        inverse_matrix : Tensor or None, optional
            Precomputed ``K^{-1}`` for dense parameterization. When omitted, the
            inverse is computed on demand.

        Returns
        -------
        Tensor
            Recovered latent states at time ``t``, same shape as ``z``.
        """
        adjusted = z
        if self.control_dim > 0:
            if control is None:
                msg = "control input is required when control_dim > 0"
                raise ValueError(msg)
            offset = self.control_term(
                control,
                num_nodes=z.shape[-2] if z.ndim >= 2 else None,
            )
            if control.ndim == 1:
                offset = self._broadcast_control_term(z, offset)
            adjusted = z - offset
        if self.parameterization == "odo":
            o1, o2 = self._odo_orthogonal_factors()
            diagonal = self._odo_diagonal()
            diag_values = torch.diag(diagonal)
            eps = torch.finfo(diag_values.dtype).eps
            inverse_diag = torch.diag(1.0 / diag_values.clamp_min(eps))
            inverse_k = o2 @ inverse_diag @ o1.T
            return adjusted @ inverse_k.T
        matrix = inverse_matrix
        if matrix is None:
            matrix = self.dense_inverse_matrix()
        return adjusted @ matrix.T

    def dense_inverse_matrix(self) -> Tensor:
        """Return the inverse (or pseudo-inverse) of the assembled dense matrix.

        Intended for reuse across multiple backward-consistency pair evaluations
        within one training step.

        Returns
        -------
        Tensor
            Matrix ``K^{-1}`` (or ``K^{\\dagger}``) with shape
            ``(latent_dim, latent_dim)``.

        Raises
        ------
        ValueError
            If :attr:`parameterization` is not ``"dense"``.
        """
        if self.parameterization != "dense":
            msg = "dense_inverse_matrix is only available for dense parameterization"
            raise ValueError(msg)
        matrix = self.K
        try:
            return torch.linalg.inv(matrix)
        except RuntimeError:
            return torch.linalg.pinv(matrix)
