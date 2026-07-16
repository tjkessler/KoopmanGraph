"""Continuous-time Koopman generator and matrix-exponential propagation."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

from koopman_graph.operator import (
    DISSIPATIVE_MIN_EIGENVALUE,
    InitMode,
    StabilityCertificate,
    _cayley_orthogonal,
    _strict_spectral_bound,
)

GeneratorParameterization = Literal["dense", "odo", "schur", "dissipative", "lyapunov"]


def _negative_strict_diagonal_values(
    raw: Tensor,
    max_real_eigenvalue: float,
) -> Tensor:
    """Map raw parameters to strictly negative diagonal generator eigenvalues.

    Parameters
    ----------
    raw : Tensor
        Unconstrained diagonal parameters with shape ``(latent_dim,)``.
    max_real_eigenvalue : float
        Magnitude scale for stable real parts.

    Returns
    -------
    Tensor
        Negative diagonal eigenvalues in ``(-bound, 0)``.
    """
    bound = _strict_spectral_bound(max_real_eigenvalue)
    return -torch.tanh(raw).abs() * bound


class ContinuousKoopmanOperator(nn.Module):
    """Learnable continuous-time Koopman generator **L**.

    Discrete propagation over an interval ``Δt`` follows::

        K(Δt) = exp(L · Δt),    z(t+Δt) = z(t) @ K(Δt).T + control_integral

    When :attr:`control_dim` is positive, the controlled generator dynamics are
    ``ż = z L + u B̃`` (row-vector convention). Piecewise-constant controls over
    ``[0, Δt]`` are integrated exactly via a Van Loan block-matrix exponential.

    Opt-in Hurwitz-stable parameterizations mirror the discrete structural modes
    from :class:`~koopman_graph.operator.KoopmanOperator` but enforce
    ``Re(λ) < 0`` for generator eigenvalues rather than ``|λ| < 1`` for
    discrete steps.

    Stability modes
    ---------------
    **Soft (no mathematical guarantee on assembled *L*):**

    - ``"dense"`` — unconstrained learnable generator. Pair with eigenvalue
      regularization during training for empirical Hurwitz penalization.
    - ``"odo"`` — ODO factorization on the generator. Bounds diagonal factors
      only; assembled ``L`` is **not** guaranteed Hurwitz-stable.
      :meth:`max_real_part` reports a diagonal-factor bound, not the true
      maximum real eigenvalue of ``L``.

    **Structural (generator eigenvalues forced into the left half-plane):**

    - ``"schur"``, ``"dissipative"``, ``"lyapunov"`` — same philosophy as the
      discrete structural modes; certified via :meth:`stability_certificate`.

    Attributes
    ----------
    latent_dim : int
        Dimension of the latent space.
    control_dim : int
        Dimension of exogenous control inputs. Zero disables control.
    init_mode : str
        Weight initialization strategy for ``L``.
    init_scale : float
        Noise scale used when ``init_mode="identity_noise"``.
    parameterization : str
        Generator parameterization.
    max_real_eigenvalue : float
        Scale for structurally stable negative real parts.
    """

    def __init__(
        self,
        latent_dim: int,
        *,
        init_mode: InitMode = "identity_noise",
        init_scale: float = 1e-2,
        parameterization: GeneratorParameterization = "dense",
        max_real_eigenvalue: float = 1.0,
        control_dim: int = 0,
    ) -> None:
        """Initialize the continuous-time Koopman generator.

        Parameters
        ----------
        latent_dim : int
            Dimension of the latent space.
        init_mode : {"identity", "identity_noise", "xavier"}, optional
            Initialization strategy. Default is ``"identity_noise"``.
        init_scale : float, optional
            Noise scale for ``init_mode="identity_noise"``. Default is ``1e-2``.
        parameterization : {"dense", "odo", "schur", "dissipative", "lyapunov"},
            optional
            Generator parameterization. ``"dense"`` and ``"odo"`` are **soft**
            modes with no structural Hurwitz guarantee (``"odo"`` bounds
            diagonal factors only). ``"schur"``, ``"dissipative"``, and
            ``"lyapunov"`` enforce **structural** Hurwitz stability. Default is
            ``"dense"``.
        max_real_eigenvalue : float, optional
            Magnitude scale for structurally stable negative eigenvalues.
            Default is ``1.0``.
        control_dim : int, optional
            Control input dimension. Default is ``0``.
        """
        super().__init__()
        if latent_dim < 1:
            msg = f"latent_dim must be positive, got {latent_dim}"
            raise ValueError(msg)
        if init_scale < 0:
            msg = f"init_scale must be non-negative, got {init_scale}"
            raise ValueError(msg)
        if max_real_eigenvalue <= 0:
            msg = f"max_real_eigenvalue must be positive, got {max_real_eigenvalue}"
            raise ValueError(msg)
        if control_dim < 0:
            msg = f"control_dim must be non-negative, got {control_dim}"
            raise ValueError(msg)

        self.latent_dim = latent_dim
        self.init_mode = init_mode
        self.init_scale = init_scale
        self.parameterization = parameterization
        self.max_real_eigenvalue = max_real_eigenvalue
        self.control_dim = control_dim

        if parameterization == "dense":
            self.register_parameter(
                "L", nn.Parameter(torch.empty(latent_dim, latent_dim))
            )
        elif parameterization == "odo":
            self.cayley_O1 = nn.Parameter(torch.zeros(latent_dim, latent_dim))
            self.cayley_O2 = nn.Parameter(torch.zeros(latent_dim, latent_dim))
            self.diag_raw = nn.Parameter(torch.zeros(latent_dim))
        elif parameterization == "schur":
            self.cayley_Q = nn.Parameter(torch.zeros(latent_dim, latent_dim))
            self.schur_diag_raw = nn.Parameter(torch.zeros(latent_dim))
            self.schur_off_raw = nn.Parameter(torch.zeros(latent_dim, latent_dim))
        elif parameterization == "dissipative":
            self.dissipative_L = nn.Parameter(torch.zeros(latent_dim, latent_dim))
        elif parameterization == "lyapunov":
            self.cayley_Q = nn.Parameter(torch.zeros(latent_dim, latent_dim))
            self.lyap_diag_raw = nn.Parameter(torch.zeros(latent_dim))
            self.lyap_p_raw = nn.Parameter(torch.zeros(latent_dim))
        else:
            msg = f"Unknown parameterization: {parameterization!r}"
            raise ValueError(msg)

        self.reset_parameters()

        if control_dim > 0:
            self.B = nn.Parameter(torch.empty(control_dim, latent_dim))
            self.reset_control_parameters()

    def reset_control_parameters(self) -> None:
        """Reinitialize the continuous control input matrix ``B``.

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
            Global or per-node control input.
        num_nodes : int or None, optional
            Expected node count for per-node controls.

        Returns
        -------
        Tensor
            Latent control offset.
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

        Returns
        -------
        Tensor
            Broadcast control offset.
        """
        view_shape = (1,) * (z.ndim - 1) + (self.latent_dim,)
        return control_term.view(view_shape).expand_as(z)

    @property
    def L(self) -> Tensor:
        """Assembled generator matrix with shape ``(latent_dim, latent_dim)``.

        Returns
        -------
        Tensor
            Current generator matrix ``L``.
        """
        if self.parameterization == "dense":
            dense_l = self._parameters.get("L")
            if dense_l is None:
                raise AttributeError("L")
            return dense_l
        return self._assemble_generator()

    def transition_matrix(self, delta_t: float | Tensor) -> Tensor:
        """Return the discrete propagator ``K(Δt) = exp(L · Δt)``.

        Parameters
        ----------
        delta_t : float or Tensor
            Integration interval.

        Returns
        -------
        Tensor
            Discrete propagator matrix.
        """
        delta = torch.as_tensor(delta_t, dtype=self.L.dtype, device=self.L.device)
        return torch.linalg.matrix_exp(self.L * delta)

    def advance(
        self,
        z: Tensor,
        delta_t: float | Tensor,
        *,
        control: Tensor | None = None,
    ) -> Tensor:
        """Advance latent states over a continuous-time interval ``Δt``.

        Parameters
        ----------
        z : Tensor
            Latent states with shape ``(..., latent_dim)``.
        delta_t : float or Tensor
            Integration interval. ``0`` returns ``z`` unchanged.
        control : Tensor or None, optional
            Piecewise-constant control over ``[0, Δt]``.

        Returns
        -------
        Tensor
            Advanced latent states with the same shape as ``z``.
        """
        if z.shape[-1] != self.latent_dim:
            msg = (
                f"Expected trailing dimension {self.latent_dim}, "
                f"got shape {tuple(z.shape)}"
            )
            raise ValueError(msg)

        delta = torch.as_tensor(delta_t, dtype=z.dtype, device=z.device)
        if torch.isclose(delta, torch.zeros((), device=z.device, dtype=z.dtype)).item():
            if control is not None and self.control_dim > 0:
                msg = "control input is ignored when delta_t is zero"
                raise ValueError(msg)
            return z

        if self.control_dim == 0:
            if control is not None:
                msg = "control input provided to an uncontrolled operator"
                raise ValueError(msg)
            transition = self.transition_matrix(delta)
            return z @ transition.T

        if control is None:
            msg = "control input is required when control_dim > 0"
            raise ValueError(msg)

        return self._advance_controlled(z, delta, control)

    def _advance_controlled(
        self,
        z: Tensor,
        delta_t: Tensor,
        control: Tensor,
    ) -> Tensor:
        """Advance with Van Loan block-matrix exponential integration.

        Returns
        -------
        Tensor
            Controlled advanced latent states.
        """
        if control.ndim == 1:
            phi11, phi12 = self._van_loan_factors(delta_t)
            offset = control @ phi12.T
            if z.ndim > 1:
                offset = self._broadcast_control_term(z, offset)
            return z @ phi11.T + offset

        if control.ndim == 2:
            phi11, phi12 = self._van_loan_factors(delta_t)
            return z @ phi11.T + control @ phi12.T

        msg = (
            "control input must have shape (control_dim,) or "
            f"(num_nodes, control_dim), got {tuple(control.shape)}"
        )
        raise ValueError(msg)

    def _van_loan_factors(self, delta_t: Tensor) -> tuple[Tensor, Tensor]:
        """Return Van Loan factors ``Phi11`` and ``Phi12`` for interval ``Δt``.

        Returns
        -------
        tuple[Tensor, Tensor]
            State and control transition factors.
        """
        latent_dim = self.latent_dim
        control_dim = self.control_dim
        generator = self.L
        dtype = generator.dtype
        device = generator.device

        block = torch.zeros(
            (latent_dim + control_dim, latent_dim + control_dim),
            dtype=dtype,
            device=device,
        )
        block[:latent_dim, :latent_dim] = generator.T
        block[:latent_dim, latent_dim:] = self.B.T
        exponential = torch.linalg.matrix_exp(block * delta_t)
        phi11 = exponential[:latent_dim, :latent_dim]
        phi12 = exponential[:latent_dim, latent_dim:]
        return phi11, phi12

    def inverse_advance(
        self,
        z: Tensor,
        delta_t: float | Tensor,
        *,
        control: Tensor | None = None,
    ) -> Tensor:
        """Recover the previous latent state before advancing over ``Δt``.

        Returns
        -------
        Tensor
            Recovered latent states.
        """
        if self.control_dim > 0 and control is None:
            msg = "control input is required when control_dim > 0"
            raise ValueError(msg)

        adjusted = z
        if self.control_dim > 0:
            assert control is not None
            if control.ndim == 1:
                _, phi12 = self._van_loan_factors(
                    torch.as_tensor(delta_t, dtype=z.dtype, device=z.device)
                )
                offset = control @ phi12.T
                if z.ndim > 1:
                    offset = self._broadcast_control_term(z, offset)
                adjusted = z - offset
            else:
                _, phi12 = self._van_loan_factors(
                    torch.as_tensor(delta_t, dtype=z.dtype, device=z.device)
                )
                adjusted = z - control @ phi12.T

        inverse_transition = self.transition_matrix(
            -torch.as_tensor(delta_t, dtype=self.L.dtype, device=self.L.device)
        )
        return adjusted @ inverse_transition.T

    def forward(
        self,
        z: Tensor,
        control: Tensor | None = None,
        *,
        delta_t: float | Tensor = 1.0,
    ) -> Tensor:
        """Advance latent states by ``delta_t`` (defaults to ``1.0``).

        Returns
        -------
        Tensor
            Advanced latent states.
        """
        return self.advance(z, delta_t, control=control)

    def reset_parameters(self) -> None:
        """Reinitialize generator parameters according to :attr:`init_mode`.

        Returns
        -------
        None
        """
        resetters = {
            "dense": self._reset_dense_parameters,
            "odo": self._reset_odo_parameters,
            "schur": self._reset_schur_parameters,
            "dissipative": self._reset_dissipative_parameters,
            "lyapunov": self._reset_lyapunov_parameters,
        }
        resetters[self.parameterization]()

    def _reset_dense_parameters(self) -> None:
        """Reinitialize the dense learnable matrix ``L``.

        Returns
        -------
        None
        """
        matrix = self._parameters["L"]
        if self.init_mode == "identity":
            nn.init.zeros_(matrix)
        elif self.init_mode == "identity_noise":
            nn.init.zeros_(matrix)
            with torch.no_grad():
                matrix.add_(torch.randn_like(matrix) * self.init_scale)
        elif self.init_mode == "xavier":
            nn.init.xavier_uniform_(matrix)
        else:
            msg = f"Unknown init_mode: {self.init_mode!r}"
            raise ValueError(msg)

    def _identity_negative_diag_raw(self) -> float:
        """Return raw diagonal init for a near-zero stable generator.

        Returns
        -------
        float
        """
        bound = _strict_spectral_bound(self.max_real_eigenvalue)
        target = -bound * 1e-2
        ratio = abs(target) / bound
        return float(torch.atanh(torch.tensor(ratio)).item())

    def _reset_odo_parameters(self) -> None:
        """Reinitialize ODO generator parameters.

        Returns
        -------
        None
        """
        nn.init.zeros_(self.cayley_O1)
        nn.init.zeros_(self.cayley_O2)
        if self.init_mode in {"identity", "identity_noise"}:
            nn.init.constant_(self.diag_raw, self._identity_negative_diag_raw())
            if self.init_mode == "identity_noise":
                with torch.no_grad():
                    self.diag_raw.add_(
                        torch.randn_like(self.diag_raw) * self.init_scale
                    )
        elif self.init_mode == "xavier":
            nn.init.xavier_uniform_(self.cayley_O1)
            nn.init.xavier_uniform_(self.cayley_O2)
            nn.init.normal_(self.diag_raw)
        else:
            msg = f"Unknown init_mode: {self.init_mode!r}"
            raise ValueError(msg)

    def _reset_schur_parameters(self) -> None:
        """Reinitialize Schur generator parameters.

        Returns
        -------
        None
        """
        nn.init.zeros_(self.cayley_Q)
        nn.init.zeros_(self.schur_off_raw)
        if self.init_mode in {"identity", "identity_noise"}:
            nn.init.constant_(self.schur_diag_raw, self._identity_negative_diag_raw())
            if self.init_mode == "identity_noise":
                with torch.no_grad():
                    self.schur_off_raw.add_(
                        torch.randn_like(self.schur_off_raw) * self.init_scale
                    )
        elif self.init_mode == "xavier":
            nn.init.xavier_uniform_(self.cayley_Q)
            nn.init.xavier_uniform_(self.schur_off_raw)
            nn.init.normal_(self.schur_diag_raw)
        else:
            msg = f"Unknown init_mode: {self.init_mode!r}"
            raise ValueError(msg)

    def _reset_dissipative_parameters(self) -> None:
        """Reinitialize dissipative generator parameters.

        Returns
        -------
        None
        """
        nn.init.zeros_(self.dissipative_L)
        if self.init_mode == "identity_noise":
            with torch.no_grad():
                self.dissipative_L.add_(
                    torch.randn_like(self.dissipative_L) * self.init_scale
                )
        elif self.init_mode == "xavier":
            nn.init.xavier_uniform_(self.dissipative_L)

    def _reset_lyapunov_parameters(self) -> None:
        """Reinitialize Lyapunov generator parameters.

        Returns
        -------
        None
        """
        nn.init.zeros_(self.cayley_Q)
        if self.init_mode in {"identity", "identity_noise"}:
            nn.init.constant_(self.lyap_diag_raw, self._identity_negative_diag_raw())
            nn.init.constant_(self.lyap_p_raw, 0.0)
            if self.init_mode == "identity_noise":
                with torch.no_grad():
                    self.lyap_diag_raw.add_(
                        torch.randn_like(self.lyap_diag_raw) * self.init_scale
                    )
        elif self.init_mode == "xavier":
            nn.init.xavier_uniform_(self.cayley_Q)
            nn.init.normal_(self.lyap_diag_raw)
            nn.init.normal_(self.lyap_p_raw)
        else:
            msg = f"Unknown init_mode: {self.init_mode!r}"
            raise ValueError(msg)

    def _odo_diagonal(self) -> Tensor:
        """Build the negative diagonal factor for ODO generators.

        Returns
        -------
        Tensor
        """
        values = _negative_strict_diagonal_values(
            self.diag_raw,
            self.max_real_eigenvalue,
        )
        return torch.diag(values)

    def _assemble_odo_generator(self) -> Tensor:
        """Assemble the ODO generator matrix.

        Returns
        -------
        Tensor
        """
        o1, o2 = _cayley_orthogonal(self.cayley_O1), _cayley_orthogonal(self.cayley_O2)
        return o1 @ self._odo_diagonal() @ o2.T

    def _schur_triangular(self) -> Tensor:
        """Build the upper-triangular Schur factor.

        Returns
        -------
        Tensor
        """
        diag_vals = _negative_strict_diagonal_values(
            self.schur_diag_raw,
            self.max_real_eigenvalue,
        )
        triangular = torch.triu(self.schur_off_raw, diagonal=1)
        return triangular + torch.diag(diag_vals)

    def _assemble_schur_generator(self) -> Tensor:
        """Assemble the Schur-form generator matrix.

        Returns
        -------
        Tensor
        """
        q = _cayley_orthogonal(self.cayley_Q)
        return q @ self._schur_triangular() @ q.T

    def _dissipative_factor(self) -> Tensor:
        """Build the lower-triangular dissipative factor.

        Returns
        -------
        Tensor
        """
        lower = torch.tril(self.dissipative_L)
        diag_index = torch.arange(self.latent_dim, device=lower.device)
        lower[diag_index, diag_index] = (
            torch.nn.functional.softplus(lower[diag_index, diag_index])
            + DISSIPATIVE_MIN_EIGENVALUE
        )
        return lower

    def _dissipative_generator(self) -> Tensor:
        """Build the dissipative generator matrix.

        Returns
        -------
        Tensor
        """
        factor = self._dissipative_factor()
        identity = torch.eye(
            self.latent_dim,
            device=factor.device,
            dtype=factor.dtype,
        )
        spd = factor @ factor.T + DISSIPATIVE_MIN_EIGENVALUE * identity
        return -spd

    def _lyapunov_diagonal(self) -> Tensor:
        """Return negative diagonal eigenvalues.

        Returns
        -------
        Tensor
            Negative diagonal eigenvalues.
        """
        return _negative_strict_diagonal_values(
            self.lyap_diag_raw,
            self.max_real_eigenvalue,
        )

    def _lyapunov_matrix(self) -> Tensor:
        """Return the Lyapunov certificate matrix.

        Returns
        -------
        Tensor
        """
        q = _cayley_orthogonal(self.cayley_Q)
        p = torch.nn.functional.softplus(self.lyap_p_raw) + 1e-6
        return q @ torch.diag(p) @ q.T

    def _assemble_lyapunov_generator(self) -> Tensor:
        """Assemble the Lyapunov generator matrix.

        Returns
        -------
        Tensor
        """
        q = _cayley_orthogonal(self.cayley_Q)
        return q @ torch.diag(self._lyapunov_diagonal()) @ q.T

    def _assemble_generator(self) -> Tensor:
        """Assemble ``L`` for the active parameterization.

        Returns
        -------
        Tensor
        """
        assemblers = {
            "odo": self._assemble_odo_generator,
            "schur": self._assemble_schur_generator,
            "dissipative": self._dissipative_generator,
            "lyapunov": self._assemble_lyapunov_generator,
        }
        return assemblers[self.parameterization]()

    def max_real_part(self) -> Tensor:
        """Return an upper bound on the generator's maximum real eigenvalue.

        For structurally stable modes, this is a certified bound on
        ``\\max \\operatorname{Re}(\\lambda_i(L))``. For ``"odo"``, the
        returned value is a diagonal-factor bound only — **not** the true
        maximum real eigenvalue of assembled ``L``.

        Returns
        -------
        Tensor
            Scalar upper bound on the maximum real eigenvalue (or diagonal
            factor bound for ``"odo"``).
        """
        if self.parameterization in {"odo", "schur", "lyapunov"}:
            if self.parameterization == "odo":
                raw = self.diag_raw
            elif self.parameterization == "schur":
                raw = self.schur_diag_raw
            else:
                raw = self.lyap_diag_raw
            diagonal = _negative_strict_diagonal_values(raw, self.max_real_eigenvalue)
            return diagonal.max()
        if self.parameterization == "dissipative":
            generator = self._dissipative_generator()
            return torch.linalg.eigvalsh(generator).max()
        eigenvalues = torch.linalg.eigvals(self.L)
        return eigenvalues.real.max()

    def stability_certificate(self) -> StabilityCertificate | None:
        """Return a Hurwitz stability certificate when available.

        Returns
        -------
        StabilityCertificate or None
            Certificate dictionary when available.
        """
        if self.parameterization == "lyapunov":
            diagonal = self._lyapunov_diagonal()
            margin = -diagonal.max()
            return {
                "lyapunov_matrix": self._lyapunov_matrix(),
                "margin": margin,
            }
        if self.parameterization in {"schur", "odo"}:
            margin = -self.max_real_part()
            return {"margin": margin}
        if self.parameterization == "dissipative":
            margin = -self.max_real_part()
            return {"margin": margin}
        return None
