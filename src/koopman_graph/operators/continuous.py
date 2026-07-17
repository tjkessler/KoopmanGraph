"""Continuous-time Koopman generator and matrix-exponential propagation."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from koopman_graph.operators.contract import (
    DISSIPATIVE_MIN_EIGENVALUE,
    STABILITY_EPS_MARGIN,
    InitMode,
    Parameterization,
    StabilityCertificate,
    cayley_orthogonal,
    strict_spectral_bound,
)
from koopman_graph.operators.control import (
    ControlMode,
    allocate_bilinear_parameters,
    bilinear_coupling_tensor,
    effective_bilinear_matrix,
    reset_bilinear_parameters,
    validate_control_mode,
)

# Thin alias for older call sites / docs that used the continuous-specific name.
GeneratorParameterization = Parameterization

# Default absolute tolerance for Van Loan discrete↔generator round-trips in tests
# and documented adaptation fidelity checks (float32 matrix-exp / logm residual).
VAN_LOAN_WRITEBACK_ATOL = 1e-5


def matrix_log(matrix: Tensor) -> Tensor:
    """Return the principal matrix logarithm via complex eigendecomposition.

    For diagonalizable ``M = V \\operatorname{diag}(\\lambda) V^{-1}``,

    .. math::

        \\log M = V \\operatorname{diag}(\\log \\lambda_i) V^{-1}

    with the principal branch of the scalar logarithm. Real inputs return
    ``result.real`` (callers should keep spectra away from the negative-real
    branch cut when a real logarithm is required).

    Limitations
    -----------
    - Non-diagonalizable matrices are not handled (Jordan blocks need a
      different formula).
    - Eigenvalues on or near the negative real axis can make the principal
      log complex; discarding the imaginary part is then approximate.
    - Used by Van Loan inversion and continuous RLS write-back; prefer
      well-conditioned generators with moderate ``Δt``.

    Parameters
    ----------
    matrix : Tensor
        Square matrix with shape ``(d, d)``.

    Returns
    -------
    Tensor
        Matrix logarithm. Real for real ``matrix`` when the imaginary part
        of the eigendecomposition path is negligible.
    """
    eigenvalues, eigenvectors = torch.linalg.eig(matrix)
    log_eigenvalues = torch.log(eigenvalues)
    result = eigenvectors @ torch.diag(log_eigenvalues) @ torch.linalg.inv(eigenvectors)
    if matrix.is_complex():
        return result
    return result.real


def van_loan_factors(
    generator: Tensor,
    control_matrix: Tensor,
    delta_t: float | Tensor,
) -> tuple[Tensor, Tensor]:
    """Return Van Loan factors ``Phi11`` and ``Phi12`` for interval ``Δt``.

    Matches uncontrolled advance ``z @ exp(L · Δt).T`` and the discrete
    row convention ``z @ K.T + u @ B``. Column form is
    ``ẋ = L x + B^T u`` with Van Loan block::

        block = [[L, B.T], [0, 0]]
        exp(block · Δt) = [[Phi11, Phi12], [0, I]]

    so ``Phi11 = exp(L · Δt)`` and
    ``z_{t+Δt} = z @ Phi11.T + u @ Phi12.T``.

    Parameters
    ----------
    generator : Tensor
        Continuous generator ``L`` with shape ``(latent_dim, latent_dim)``.
    control_matrix : Tensor
        Continuous control matrix ``B`` with shape
        ``(control_dim, latent_dim)``.
    delta_t : float or Tensor
        Integration interval.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(Phi11, Phi12)`` with shapes ``(latent_dim, latent_dim)`` and
        ``(latent_dim, control_dim)``.
    """
    latent_dim = generator.shape[0]
    control_dim = control_matrix.shape[0]
    delta = torch.as_tensor(delta_t, dtype=generator.dtype, device=generator.device)
    block = torch.zeros(
        (latent_dim + control_dim, latent_dim + control_dim),
        dtype=generator.dtype,
        device=generator.device,
    )
    block[:latent_dim, :latent_dim] = generator
    block[:latent_dim, latent_dim:] = control_matrix.T
    exponential = torch.linalg.matrix_exp(block * delta)
    phi11 = exponential[:latent_dim, :latent_dim]
    phi12 = exponential[:latent_dim, latent_dim:]
    return phi11, phi12


def van_loan_generator_from_discrete(
    discrete_k: Tensor,
    discrete_b: Tensor,
    delta_t: float | Tensor,
) -> tuple[Tensor, Tensor]:
    """Recover continuous ``(L, B)`` from discrete Van Loan propagator blocks.

    Inverts::

        [[K, Phi12], [0, I]] = exp([[L, B.T], [0, 0]] · Δt)

    where ``K = Phi11 = exp(L · Δt)`` and ``B_disc = Phi12.T`` (library row
    convention ``z @ K.T + u @ B_disc``).

    Parameters
    ----------
    discrete_k : Tensor
        Discrete state propagator ``K(Δt)`` with shape
        ``(latent_dim, latent_dim)``.
    discrete_b : Tensor
        Discrete control matrix with shape ``(control_dim, latent_dim)``.
    delta_t : float or Tensor
        Integration interval used to form the discrete blocks.

    Returns
    -------
    tuple[Tensor, Tensor]
        Continuous generator ``L`` and control ``B``.

    Notes
    -----
    Round-trip fidelity is typically within :data:`VAN_LOAN_WRITEBACK_ATOL`
    for moderate ``Δt`` when ``K(Δt)`` stays away from matrix-logarithm branch
    cuts. Large or highly oscillatory intervals can degrade recovery.
    """
    latent_dim = discrete_k.shape[0]
    control_dim = discrete_b.shape[0]
    delta = float(torch.as_tensor(delta_t).item())
    if delta <= 0.0:
        msg = f"delta_t must be positive, got {delta}"
        raise ValueError(msg)

    identity = torch.eye(
        control_dim,
        dtype=discrete_k.dtype,
        device=discrete_k.device,
    )
    block = torch.zeros(
        (latent_dim + control_dim, latent_dim + control_dim),
        dtype=discrete_k.dtype,
        device=discrete_k.device,
    )
    block[:latent_dim, :latent_dim] = discrete_k
    block[:latent_dim, latent_dim:] = discrete_b.T
    block[latent_dim:, latent_dim:] = identity
    generator_block = matrix_log(block) / delta
    generator = generator_block[:latent_dim, :latent_dim]
    control_matrix = generator_block[:latent_dim, latent_dim:].T
    return generator, control_matrix


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
        Strictly negative diagonal eigenvalues in ``(-bound, 0)``. A floor of
        :data:`~koopman_graph.operators.STABILITY_EPS_MARGIN` (capped at half
        the bound) keeps ``raw = 0`` strictly left of the imaginary axis, so
        structural ``schur`` / ``lyapunov`` generators remain Hurwitz even at
        the origin of parameter space.
    """
    bound = strict_spectral_bound(max_real_eigenvalue)
    eps = min(STABILITY_EPS_MARGIN, 0.5 * bound)
    return -eps - torch.tanh(raw).abs() * (bound - eps)


class ContinuousKoopmanOperator(nn.Module):
    """Learnable continuous-time Koopman generator **L**.

    Discrete propagation over an interval ``Δt`` follows::

        K(Δt) = exp(L · Δt),    z(t+Δt) = z(t) @ K(Δt).T + control_integral

    This matches the discrete row convention and the column ODE
    ``ẋ = L x + B^{\\top} u`` (equivalently ``ż = z L^{\\top} + u B``).
    With ``control_mode="bilinear"``, piecewise-constant controls make the
    state map linear time-invariant over ``[0, Δt]`` with effective generator
    ``L_eff = L + sum_i u_i N_i``, which is integrated exactly via Van Loan
    (same additive ``B`` integral). This is exact for fixed ``u`` over the
    step; it is not a closed-form bilinear matrix exponential for
    time-varying ``u(t)``.

    Opt-in Hurwitz-stable parameterizations mirror the discrete structural modes
    from :class:`~koopman_graph.operators.KoopmanOperator` but enforce
    ``Re(λ) < 0`` for generator eigenvalues rather than ``|λ| < 1`` for
    discrete steps.

    Stability modes
    ---------------
    **Soft (no mathematical guarantee on assembled *L*):**

    - ``"dense"`` — unconstrained learnable generator. Pair with eigenvalue
      regularization during training for empirical Hurwitz penalization.
    - ``"odo"`` — ODO factorization on the generator. Bounds diagonal factors
      only; assembled ``L`` is **not** guaranteed Hurwitz-stable (unlike
      discrete ODO, where orthogonal factors imply an operator-norm bound on
      ``\\rho(K)``). :meth:`bound_metric` reports the diagonal-factor bound;
      :meth:`max_real_part` always returns the true maximum real eigenvalue of
      ``L`` via ``eigvals``. Pair with
      :class:`~koopman_graph.losses.EigenvalueRegularizationLoss` (true-spectrum
      path) for empirical Hurwitz penalization.

    **Structural (generator eigenvalues forced into the left half-plane):**

    - ``"schur"``, ``"dissipative"``, ``"lyapunov"`` — same philosophy as the
      discrete structural modes (Schur uses a real upper-triangular factor,
      forcing real generator eigenvalues); certified via
      :meth:`stability_certificate`.

    Attributes
    ----------
    latent_dim : int
        Dimension of the latent space.
    control_dim : int
        Dimension of exogenous control inputs. Zero disables control.
    control_mode : {"additive", "bilinear"}
        Additive ``B`` only, or bilinear state–control couplings ``N_i``.
    bilinear_rank : int or None
        Optional low-rank size for bilinear factors.
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
        parameterization: Parameterization = "dense",
        max_real_eigenvalue: float = 1.0,
        control_dim: int = 0,
        control_mode: ControlMode = "additive",
        bilinear_rank: int | None = None,
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
        control_mode : {"additive", "bilinear"}, optional
            Control coupling mode. Default is ``"additive"``.
        bilinear_rank : int or None, optional
            Low-rank bilinear size when ``control_mode="bilinear"``.
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
        validate_control_mode(
            control_dim=control_dim,
            control_mode=control_mode,
            bilinear_rank=bilinear_rank,
            latent_dim=latent_dim,
        )

        self.latent_dim = latent_dim
        self.init_mode = init_mode
        self.init_scale = init_scale
        self.parameterization = parameterization
        self.max_real_eigenvalue = max_real_eigenvalue
        self.control_dim = control_dim
        self.control_mode = control_mode
        self.bilinear_rank = bilinear_rank

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
            if control_mode == "bilinear":
                allocate_bilinear_parameters(
                    self,
                    control_dim=control_dim,
                    latent_dim=latent_dim,
                    bilinear_rank=bilinear_rank,
                )
            self.reset_control_parameters()

    def reset_control_parameters(self) -> None:
        """Reinitialize the continuous control input matrix ``B`` (and bilinear).

        Returns
        -------
        None
        """
        if self.control_dim <= 0:
            return
        nn.init.zeros_(self.B)
        if self.control_mode == "bilinear":
            reset_bilinear_parameters(self)

    def bilinear_matrices(self) -> Tensor:
        """Return assembled bilinear couplings ``N`` with shape ``(C, D, D)``.

        Returns
        -------
        Tensor
            Full-rank or low-rank-assembled ``N_i`` stack.
        """
        if self.control_mode != "bilinear":
            msg = "bilinear_matrices requires control_mode='bilinear'"
            raise ValueError(msg)
        return bilinear_coupling_tensor(self)

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

        Prefer :attr:`matrix` when writing code against
        :class:`~koopman_graph.operators.KoopmanOperatorContract`.

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

    @property
    def matrix(self) -> Tensor:
        """Assembled generator matrix (alias of :attr:`L`).

        Returns
        -------
        Tensor
            Current continuous-time generator ``L``.
        """
        return self.L

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
        delta_t: float | Tensor | None = None,
        *,
        control: Tensor | None = None,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        """Advance latent states over a continuous-time interval ``Δt``.

        Parameters
        ----------
        z : Tensor
            Latent states with shape ``(..., latent_dim)``.
        delta_t : float, Tensor, or None
            Integration interval. ``0`` returns ``z`` unchanged. Required
            (must not be ``None``) for continuous operators.
        control : Tensor or None, optional
            Piecewise-constant control over ``[0, Δt]``.
        edge_index : Tensor or None, optional
            Ignored (per-node continuous operator).
        edge_weight : Tensor or None, optional
            Ignored (per-node continuous operator).

        Returns
        -------
        Tensor
            Advanced latent states with the same shape as ``z``.

        Raises
        ------
        ValueError
            If ``delta_t`` is ``None``, the trailing dimension of ``z`` does
            not match ``latent_dim``, or controls are invalid.
        """
        _ = edge_index, edge_weight
        if delta_t is None:
            msg = "delta_t is required for ContinuousKoopmanOperator.advance"
            raise ValueError(msg)
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

        For bilinear mode, uses ``L_eff = L + sum_i u_i N_i`` (global) or a
        per-node ``L_eff`` when ``control`` is per-node.

        Returns
        -------
        Tensor
            Controlled advanced latent states.
        """
        if self.control_mode == "additive":
            return self._advance_van_loan(z, delta_t, control, generator=self.L)

        coupling = self.bilinear_matrices()
        if control.ndim == 1:
            l_eff = effective_bilinear_matrix(self.L, control, coupling)
            return self._advance_van_loan(z, delta_t, control, generator=l_eff)

        if control.ndim == 2:
            if z.ndim < 2 or z.shape[-2] != control.shape[0]:
                msg = (
                    "per-node bilinear control requires z with a matching "
                    f"node axis, got z={tuple(z.shape)}, u={tuple(control.shape)}"
                )
                raise ValueError(msg)
            advanced = torch.empty_like(z)
            for node_idx in range(control.shape[0]):
                l_eff = effective_bilinear_matrix(
                    self.L,
                    control[node_idx],
                    coupling,
                )
                node_z = z[..., node_idx : node_idx + 1, :]
                node_u = control[node_idx]
                node_next = self._advance_van_loan(
                    node_z,
                    delta_t,
                    node_u,
                    generator=l_eff,
                )
                advanced[..., node_idx : node_idx + 1, :] = node_next
            return advanced

        msg = (
            "control input must have shape (control_dim,) or "
            f"(num_nodes, control_dim), got {tuple(control.shape)}"
        )
        raise ValueError(msg)

    def _advance_van_loan(
        self,
        z: Tensor,
        delta_t: Tensor,
        control: Tensor,
        *,
        generator: Tensor,
    ) -> Tensor:
        """Van Loan advance for a fixed generator over ``Δt``.

        Returns
        -------
        Tensor
            Advanced latents.
        """
        phi11, phi12 = van_loan_factors(generator, self.B, delta_t)
        if control.ndim == 1:
            offset = control @ phi12.T
            if z.ndim > 1:
                offset = self._broadcast_control_term(z, offset)
            return z @ phi11.T + offset
        if control.ndim == 2:
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
        return van_loan_factors(self.L, self.B, delta_t)

    def set_dense_matrix(
        self,
        matrix: Tensor,
        *,
        control_matrix: Tensor | None = None,
        bilinear_matrices: Tensor | None = None,
    ) -> None:
        """Write dense generator parameters in place.

        Parameters
        ----------
        matrix : Tensor
            Dense generator ``L`` with shape ``(latent_dim, latent_dim)``.
        control_matrix : Tensor or None, optional
            Dense control matrix ``B`` with shape
            ``(control_dim, latent_dim)``. Required when ``control_dim > 0``.
        bilinear_matrices : Tensor or None, optional
            Full-rank bilinear stack when ``control_mode="bilinear"`` and
            ``bilinear_rank is None``.

        Raises
        ------
        ValueError
            If the operator is not densely parameterized or control shapes
            are invalid.
        """
        if self.parameterization != "dense":
            msg = (
                "set_dense_matrix requires parameterization='dense', "
                f"got {self.parameterization!r}"
            )
            raise ValueError(msg)
        if matrix.shape != (self.latent_dim, self.latent_dim):
            msg = (
                f"Expected generator shape ({self.latent_dim}, {self.latent_dim}), "
                f"got {tuple(matrix.shape)}"
            )
            raise ValueError(msg)

        dense_l = self._parameters.get("L")
        if dense_l is None:
            raise AttributeError("L")
        with torch.no_grad():
            dense_l.copy_(matrix.to(device=dense_l.device, dtype=dense_l.dtype))
            if self.control_dim > 0:
                if control_matrix is None:
                    msg = "control_matrix is required when control_dim > 0"
                    raise ValueError(msg)
                expected = (self.control_dim, self.latent_dim)
                if control_matrix.shape != expected:
                    msg = (
                        f"Expected control_matrix shape {expected}, "
                        f"got {tuple(control_matrix.shape)}"
                    )
                    raise ValueError(msg)
                self.B.copy_(
                    control_matrix.to(device=self.B.device, dtype=self.B.dtype)
                )
                if self.control_mode == "bilinear":
                    if self.bilinear_rank is not None:
                        msg = (
                            "set_dense_matrix bilinear_matrices writeback "
                            "requires bilinear_rank=None (full-rank N)"
                        )
                        raise ValueError(msg)
                    if bilinear_matrices is None:
                        msg = (
                            "bilinear_matrices is required when control_mode='bilinear'"
                        )
                        raise ValueError(msg)
                    expected_n = (
                        self.control_dim,
                        self.latent_dim,
                        self.latent_dim,
                    )
                    if bilinear_matrices.shape != expected_n:
                        msg = (
                            f"Expected bilinear_matrices shape {expected_n}, "
                            f"got {tuple(bilinear_matrices.shape)}"
                        )
                        raise ValueError(msg)
                    self.N.copy_(
                        bilinear_matrices.to(device=self.N.device, dtype=self.N.dtype)
                    )
                elif bilinear_matrices is not None:
                    msg = "bilinear_matrices provided to an additive-control operator"
                    raise ValueError(msg)
            elif control_matrix is not None or bilinear_matrices is not None:
                msg = "control_matrix provided to an uncontrolled operator"
                raise ValueError(msg)

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
        """Recover the previous latent state before advancing over ``Δt``.

        Parameters
        ----------
        z : Tensor
            Latent states after advancing over ``Δt``.
        delta_t : float, Tensor, or None
            Integration interval. Required (must not be ``None``).
        control : Tensor or None, optional
            Control applied during the forward interval.
        inverse_matrix : Tensor or None, optional
            Accepted for :class:`~koopman_graph.operators.KoopmanOperatorContract`
            symmetry with discrete operators; ignored for continuous dynamics.
        edge_index : Tensor or None, optional
            Ignored (per-node continuous operator).
        edge_weight : Tensor or None, optional
            Ignored (per-node continuous operator).

        Returns
        -------
        Tensor
            Recovered latent states.

        Raises
        ------
        ValueError
            If ``delta_t`` is ``None`` or controls are missing when required.
        """
        _ = inverse_matrix, edge_index, edge_weight
        if delta_t is None:
            msg = "delta_t is required for ContinuousKoopmanOperator.inverse_advance"
            raise ValueError(msg)
        if self.control_dim > 0 and control is None:
            msg = "control input is required when control_dim > 0"
            raise ValueError(msg)

        adjusted = z
        delta = torch.as_tensor(delta_t, dtype=z.dtype, device=z.device)
        if self.control_dim > 0:
            assert control is not None
            if self.control_mode == "bilinear":
                return self._inverse_advance_bilinear(z, delta, control)

            if control.ndim == 1:
                _, phi12 = self._van_loan_factors(delta)
                offset = control @ phi12.T
                if z.ndim > 1:
                    offset = self._broadcast_control_term(z, offset)
                adjusted = z - offset
            else:
                _, phi12 = self._van_loan_factors(delta)
                adjusted = z - control @ phi12.T

        inverse_transition = self.transition_matrix(-delta)
        return adjusted @ inverse_transition.T

    def _inverse_advance_bilinear(
        self,
        z: Tensor,
        delta_t: Tensor,
        control: Tensor,
    ) -> Tensor:
        """Invert a bilinear continuous step under piecewise-constant ``u``.

        Returns
        -------
        Tensor
            Recovered latents.
        """
        coupling = self.bilinear_matrices()

        def _invert_one(state: Tensor, u: Tensor, generator: Tensor) -> Tensor:
            """Invert one Van Loan step for a fixed effective generator.

            Parameters
            ----------
            state : Tensor
                Latents after the forward interval.
            u : Tensor
                Control applied during the interval.
            generator : Tensor
                Effective generator ``L_eff``.

            Returns
            -------
            Tensor
                Recovered latents before the interval.
            """
            phi11, phi12 = van_loan_factors(generator, self.B, delta_t)
            if u.ndim == 1:
                offset = u @ phi12.T
                if state.ndim > 1:
                    offset = self._broadcast_control_term(state, offset)
                adjusted = state - offset
            else:
                adjusted = state - u @ phi12.T
            try:
                inverse_phi = torch.linalg.inv(phi11)
            except RuntimeError:
                inverse_phi = torch.linalg.pinv(phi11)
            return adjusted @ inverse_phi.T

        if control.ndim == 1:
            l_eff = effective_bilinear_matrix(self.L, control, coupling)
            return _invert_one(z, control, l_eff)

        if control.ndim == 2:
            if z.ndim < 2 or z.shape[-2] != control.shape[0]:
                msg = (
                    "per-node bilinear inverse requires matching node axes, "
                    f"got z={tuple(z.shape)}, u={tuple(control.shape)}"
                )
                raise ValueError(msg)
            recovered = torch.empty_like(z)
            for node_idx in range(control.shape[0]):
                l_eff = effective_bilinear_matrix(
                    self.L,
                    control[node_idx],
                    coupling,
                )
                recovered[..., node_idx : node_idx + 1, :] = _invert_one(
                    z[..., node_idx : node_idx + 1, :],
                    control[node_idx],
                    l_eff,
                )
            return recovered

        msg = (
            "control input must have shape (control_dim,) or "
            f"(num_nodes, control_dim), got {tuple(control.shape)}"
        )
        raise ValueError(msg)

    def forward(
        self,
        z: Tensor,
        control: Tensor | None = None,
        *,
        delta_t: float | Tensor = 1.0,
    ) -> Tensor:
        """Advance latent states by ``delta_t``.

        Standalone soft default is ``1.0`` when callers omit an interval.
        Prefer :meth:`advance` with an explicit ``delta_t``, or model-backed
        paths that resolve missing intervals to
        :attr:`~koopman_graph.model.GraphKoopmanModel.time_step` via
        :meth:`~koopman_graph.model.GraphKoopmanModel.resolve_delta_t`.

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
        bound = strict_spectral_bound(self.max_real_eigenvalue)
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
        o1, o2 = cayley_orthogonal(self.cayley_O1), cayley_orthogonal(self.cayley_O2)
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
        q = cayley_orthogonal(self.cayley_Q)
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
        q = cayley_orthogonal(self.cayley_Q)
        p = torch.nn.functional.softplus(self.lyap_p_raw) + 1e-6
        return q @ torch.diag(p) @ q.T

    def _assemble_lyapunov_generator(self) -> Tensor:
        """Assemble the Lyapunov generator matrix.

        Returns
        -------
        Tensor
        """
        q = cayley_orthogonal(self.cayley_Q)
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

    def bound_metric(self) -> Tensor:
        """Return the cheap soft/structural monitoring bound.

        For ``"odo"``, this is a diagonal-factor bound — **not** the true
        maximum real eigenvalue of assembled ``L``. For structurally stable
        modes, this is a closed-form certified upper bound on
        ``\\max \\operatorname{Re}(\\lambda_i(L))``. For ``"dense"``, this
        equals :meth:`max_real_part`. Prefer :meth:`bound_metric` when writing
        code against :class:`~koopman_graph.operators.KoopmanOperatorContract`;
        use :meth:`max_real_part` for the true spectrum via ``eigvals``.

        Returns
        -------
        Tensor
            Scalar bound metric for the active parameterization.
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
        return self.max_real_part()

    def max_real_part(self) -> Tensor:
        """Return the true max real eigenvalue of assembled ``L``.

        Always computed from the assembled generator via ``eigvals``. For the
        cheap soft/structural monitoring bound (diagonal-factor bound for
        ``"odo"``, closed-form certificates for structural modes), use
        :meth:`bound_metric`.

        Returns
        -------
        Tensor
            Scalar tensor ``\\max_i \\operatorname{Re}(\\lambda_i(L))``.
        """
        eigenvalues = torch.linalg.eigvals(self.L)
        return eigenvalues.real.max()

    def stability_certificate(self) -> StabilityCertificate | None:
        """Return a Hurwitz stability certificate when available.

        For ``"lyapunov"``, returns the Lyapunov matrix ``P`` and a positive
        margin. For ``"schur"`` and ``"dissipative"``, returns the margin from
        :meth:`bound_metric`. Returns ``None`` for ``"dense"`` and ``"odo"``
        (soft modes have no structural certificate).

        Returns
        -------
        StabilityCertificate or None
            Frozen certificate with ``margin`` and optional ``lyapunov_matrix``.
        """
        if self.parameterization == "lyapunov":
            diagonal = self._lyapunov_diagonal()
            margin = -diagonal.max()
            return StabilityCertificate(
                margin=margin,
                lyapunov_matrix=self._lyapunov_matrix(),
            )
        if self.parameterization == "schur":
            return StabilityCertificate(margin=-self.bound_metric())
        if self.parameterization == "dissipative":
            return StabilityCertificate(margin=-self.bound_metric())
        return None
