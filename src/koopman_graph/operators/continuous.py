"""Continuous-time Koopman generator and matrix-exponential propagation.

Thin string-mode orchestrator for dense / ODO / Schur / dissipative /
Lyapunov / ``auxiliary_spectral`` generators. Cohesive Van Loan factor
construction, structural parameterization, and advance / inverse execution
live in shallow sibling modules
(:mod:`~koopman_graph.operators.continuous_van_loan`,
:mod:`~koopman_graph.operators.continuous_parameterizations`,
:mod:`~koopman_graph.operators.continuous_propagation`) and are re-exported
here so deep imports keep working.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from koopman_graph.operators.auxiliary_spectral import (
    DEFAULT_AUXILIARY_HIDDEN_DIMS,
    AuxiliarySpectralNetwork,
    normalize_auxiliary_hidden_dims,
    reset_auxiliary_network,
)
from koopman_graph.operators.continuous_parameterizations import (
    assemble_dissipative_generator,
    assemble_lyapunov_generator,
    assemble_odo_generator,
    assemble_schur_generator,
    continuous_bound_metric,
    continuous_stability_certificate,
    lyapunov_certificate_matrix,
    lyapunov_diagonal,
    max_real_part_of_generator,
    negative_strict_diagonal_values,
    reset_dense_generator,
    reset_dissipative_generator,
    reset_lyapunov_generator,
    reset_odo_generator,
    reset_schur_generator,
)
from koopman_graph.operators.continuous_propagation import (
    advance_interval,
    inverse_advance_interval,
)
from koopman_graph.operators.continuous_van_loan import (
    VAN_LOAN_WRITEBACK_ATOL,
    matrix_log,
    van_loan_factors,
    van_loan_generator_from_discrete,
)
from koopman_graph.operators.contract import (
    InitMode,
    Parameterization,
    StabilityCertificate,
)
from koopman_graph.operators.control import (
    ControlMode,
    allocate_bilinear_parameters,
    bilinear_coupling_tensor,
    map_control_term,
    reset_bilinear_parameters,
    validate_control_mode,
    write_dense_operator_parameters,
)

# Thin alias for older call sites / docs that used the continuous-specific name.
GeneratorParameterization = Parameterization

__all__ = [
    "VAN_LOAN_WRITEBACK_ATOL",
    "ContinuousKoopmanOperator",
    "GeneratorParameterization",
    "matrix_log",
    "negative_strict_diagonal_values",
    "van_loan_factors",
    "van_loan_generator_from_discrete",
]


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

    **Parametric / locally linear (state-dependent spectrum):**

    - ``"auxiliary_spectral"`` — an auxiliary MLP maps ``z`` to instantaneous
      eigenvalues and assembles a block-diagonal rotation–scaling generator
      ``L(z)`` (Lusch et al., 2018). Prefer :meth:`generator_at` /
      :meth:`instantaneous_spectrum`; fixed :attr:`matrix` / :attr:`L` are
      unavailable. State dependence weakens global spectral-radius and Hurwitz
      certificates — this mode is complementary to delay embeddings for
      continuous-spectrum phenomenology, not a claim about the
      infinite-dimensional Koopman operator continuum.

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
    auxiliary_hidden_dims : tuple of int
        Hidden widths for ``"auxiliary_spectral"`` (empty tuple otherwise).
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
        auxiliary_hidden_dims: Sequence[int] | None = None,
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
        parameterization : {"dense", "odo", "schur", "dissipative", "lyapunov",
            "auxiliary_spectral"}, optional
            Generator parameterization. ``"dense"`` and ``"odo"`` are **soft**
            modes with no structural Hurwitz guarantee (``"odo"`` bounds
            diagonal factors only). ``"schur"``, ``"dissipative"``, and
            ``"lyapunov"`` enforce **structural** Hurwitz stability.
            ``"auxiliary_spectral"`` uses a state-dependent auxiliary network
            (see class docs). Default is ``"dense"``.
        max_real_eigenvalue : float, optional
            Magnitude scale for structurally stable negative eigenvalues.
            Default is ``1.0``.
        control_dim : int, optional
            Control input dimension. Default is ``0``.
        control_mode : {"additive", "bilinear"}, optional
            Control coupling mode. Default is ``"additive"``.
        bilinear_rank : int or None, optional
            Low-rank bilinear size when ``control_mode="bilinear"``.
        auxiliary_hidden_dims : sequence of int or None, optional
            Hidden layer widths for ``"auxiliary_spectral"``. Default is
            ``(64, 64)``. Ignored for other parameterizations unless a
            non-default value is passed (then raises).
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
        if (
            parameterization != "auxiliary_spectral"
            and auxiliary_hidden_dims is not None
            and tuple(auxiliary_hidden_dims) != DEFAULT_AUXILIARY_HIDDEN_DIMS
        ):
            msg = (
                "auxiliary_hidden_dims is only valid with "
                "parameterization='auxiliary_spectral'"
            )
            raise ValueError(msg)

        self.latent_dim = latent_dim
        self.init_mode = init_mode
        self.init_scale = init_scale
        self.parameterization = parameterization
        self.max_real_eigenvalue = max_real_eigenvalue
        self.control_dim = control_dim
        self.control_mode = control_mode
        self.bilinear_rank = bilinear_rank
        self.auxiliary_hidden_dims: tuple[int, ...] = ()

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
        elif parameterization == "auxiliary_spectral":
            dims = normalize_auxiliary_hidden_dims(auxiliary_hidden_dims)
            self.auxiliary_hidden_dims = dims
            self.auxiliary_net = AuxiliarySpectralNetwork(
                latent_dim,
                hidden_dims=dims,
            )
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
        return map_control_term(
            u,
            getattr(self, "B", None),
            control_dim=self.control_dim,
            num_nodes=num_nodes,
        )

    @property
    def L(self) -> Tensor:
        """Assembled generator matrix with shape ``(latent_dim, latent_dim)``.

        Prefer :attr:`matrix` when writing code against
        :class:`~koopman_graph.operators.KoopmanOperatorContract`.

        Returns
        -------
        Tensor
            Current generator matrix ``L``.

        Raises
        ------
        ValueError
            If ``parameterization="auxiliary_spectral"`` (use
            :meth:`generator_at` for the state-dependent generator).
        """
        if self.parameterization == "auxiliary_spectral":
            msg = (
                "ContinuousKoopmanOperator.L is unavailable for "
                "parameterization='auxiliary_spectral'; use generator_at(z) "
                "for the state-dependent generator"
            )
            raise ValueError(msg)
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

        Raises
        ------
        ValueError
            If ``parameterization="auxiliary_spectral"``.
        """
        return self.L

    def generator_at(self, z: Tensor) -> Tensor:
        """Return the instantaneous generator ``L(z)`` (auxiliary mode).

        For fixed-parameterization modes this returns the global :attr:`L`,
        broadcast to match leading dimensions of ``z`` when needed.

        Parameters
        ----------
        z : Tensor
            Latent states with shape ``(..., latent_dim)``.

        Returns
        -------
        Tensor
            Generator(s) with shape ``(..., latent_dim, latent_dim)``.
        """
        if z.shape[-1] != self.latent_dim:
            msg = (
                f"Expected trailing dimension {self.latent_dim}, "
                f"got shape {tuple(z.shape)}"
            )
            raise ValueError(msg)
        if self.parameterization == "auxiliary_spectral":
            return self.auxiliary_net.generator_at(z)
        generator = self.L
        if z.ndim == 1:
            return generator
        leading = z.shape[:-1]
        return generator.expand(*leading, self.latent_dim, self.latent_dim)

    def instantaneous_spectrum(self, z: Tensor) -> Tensor:
        """Return instantaneous eigenvalues of ``L(z)``.

        Parameters
        ----------
        z : Tensor
            Latent states with shape ``(..., latent_dim)``.

        Returns
        -------
        Tensor
            Complex eigenvalues with shape ``(..., latent_dim)``, unsorted.
        """
        generator = self.generator_at(z)
        if generator.ndim == 2:
            return torch.linalg.eigvals(generator)
        flat = generator.reshape(-1, self.latent_dim, self.latent_dim)
        values = torch.linalg.eigvals(flat)
        return values.reshape(*z.shape[:-1], self.latent_dim)

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

        Raises
        ------
        ValueError
            If ``parameterization="auxiliary_spectral"`` (state-dependent;
            use :meth:`advance` or ``exp(generator_at(z) * delta_t)``).
        """
        if self.parameterization == "auxiliary_spectral":
            msg = (
                "transition_matrix requires a fixed generator; for "
                "auxiliary_spectral use advance(...) or "
                "torch.linalg.matrix_exp(generator_at(z) * delta_t)"
            )
            raise ValueError(msg)
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
        generator = (
            self.generator_at(z)
            if self.parameterization == "auxiliary_spectral"
            else self.L
            if delta_t is not None
            else z
        )
        coupling = self.bilinear_matrices() if self.control_mode == "bilinear" else None
        return advance_interval(
            z,
            delta_t,
            control,
            latent_dim=self.latent_dim,
            control_dim=self.control_dim,
            control_mode=self.control_mode,
            parameterization=self.parameterization,
            generator=generator,
            control_matrix=self.B if self.control_dim > 0 else None,
            coupling=coupling,
        )

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

        dense_l = self._parameters.get("L")
        if dense_l is None:
            raise AttributeError("L")
        write_dense_operator_parameters(
            dense_l,
            matrix,
            control_dim=self.control_dim,
            latent_dim=self.latent_dim,
            control_mode=self.control_mode,
            bilinear_rank=self.bilinear_rank,
            control_parameter=self.B if self.control_dim > 0 else None,
            bilinear_parameter=(
                self.N
                if (
                    self.control_dim > 0
                    and self.control_mode == "bilinear"
                    and self.bilinear_rank is None
                )
                else None
            ),
            control_matrix=control_matrix,
            bilinear_matrices=bilinear_matrices,
            matrix_label="generator",
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
        generator = (
            self.generator_at(z)
            if self.parameterization == "auxiliary_spectral"
            else self.L
            if delta_t is not None
            else z
        )
        coupling = self.bilinear_matrices() if self.control_mode == "bilinear" else None
        return inverse_advance_interval(
            z,
            delta_t,
            control,
            latent_dim=self.latent_dim,
            control_dim=self.control_dim,
            control_mode=self.control_mode,
            parameterization=self.parameterization,
            generator=generator,
            control_matrix=self.B if self.control_dim > 0 else None,
            coupling=coupling,
        )

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
            "auxiliary_spectral": self._reset_auxiliary_parameters,
        }
        resetters[self.parameterization]()

    def _reset_auxiliary_parameters(self) -> None:
        """Reinitialize the auxiliary spectral network.

        Returns
        -------
        None
        """
        reset_auxiliary_network(
            self.auxiliary_net,
            init_mode=self.init_mode,
            init_scale=self.init_scale,
        )

    def _reset_dense_parameters(self) -> None:
        """Reinitialize the dense learnable matrix ``L``.

        Notes
        -----
        Thin wrapper that delegates to the shared continuous parameterization
        helpers for the active mode.
        """
        reset_dense_generator(
            self._parameters["L"],
            init_mode=self.init_mode,
            init_scale=self.init_scale,
        )

    def _reset_odo_parameters(self) -> None:
        """Reinitialize ODO generator parameters.

        Notes
        -----
        Thin wrapper that delegates to the shared continuous parameterization
        helpers for the active mode.
        """
        reset_odo_generator(
            self.cayley_O1,
            self.cayley_O2,
            self.diag_raw,
            init_mode=self.init_mode,
            init_scale=self.init_scale,
            max_real_eigenvalue=self.max_real_eigenvalue,
        )

    def _reset_schur_parameters(self) -> None:
        """Reinitialize Schur generator parameters.

        Notes
        -----
        Thin wrapper that delegates to the shared continuous parameterization
        helpers for the active mode.
        """
        reset_schur_generator(
            self.cayley_Q,
            self.schur_diag_raw,
            self.schur_off_raw,
            init_mode=self.init_mode,
            init_scale=self.init_scale,
            max_real_eigenvalue=self.max_real_eigenvalue,
        )

    def _reset_dissipative_parameters(self) -> None:
        """Reinitialize dissipative generator parameters.

        Notes
        -----
        Thin wrapper that delegates to the shared continuous parameterization
        helpers for the active mode.
        """
        reset_dissipative_generator(
            self.dissipative_L,
            init_mode=self.init_mode,
            init_scale=self.init_scale,
        )

    def _reset_lyapunov_parameters(self) -> None:
        """Reinitialize Lyapunov generator parameters.

        Notes
        -----
        Thin wrapper that delegates to the shared continuous parameterization
        helpers for the active mode.
        """
        reset_lyapunov_generator(
            self.cayley_Q,
            self.lyap_diag_raw,
            self.lyap_p_raw,
            init_mode=self.init_mode,
            init_scale=self.init_scale,
            max_real_eigenvalue=self.max_real_eigenvalue,
        )

    def _assemble_odo_generator(self) -> Tensor:
        """Assemble the ODO generator matrix.

        Returns
        -------
        Tensor
            Assembled matrix or factor for the active parameterization.
        """
        return assemble_odo_generator(
            self.cayley_O1,
            self.cayley_O2,
            self.diag_raw,
            self.max_real_eigenvalue,
        )

    def _assemble_schur_generator(self) -> Tensor:
        """Assemble the Schur-form generator matrix.

        Returns
        -------
        Tensor
            Assembled matrix or factor for the active parameterization.
        """
        return assemble_schur_generator(
            self.cayley_Q,
            self.schur_diag_raw,
            self.schur_off_raw,
            self.max_real_eigenvalue,
        )

    def _dissipative_generator(self) -> Tensor:
        """Build the dissipative generator matrix.

        Returns
        -------
        Tensor
            Assembled matrix or factor for the active parameterization.
        """
        return assemble_dissipative_generator(self.dissipative_L, self.latent_dim)

    def _lyapunov_diagonal(self) -> Tensor:
        """Return negative diagonal eigenvalues.

        Returns
        -------
        Tensor
            Assembled matrix or factor for the active parameterization.
        """
        return lyapunov_diagonal(self.lyap_diag_raw, self.max_real_eigenvalue)

    def _lyapunov_matrix(self) -> Tensor:
        """Return the Lyapunov certificate matrix.

        Returns
        -------
        Tensor
            Assembled matrix or factor for the active parameterization.
        """
        return lyapunov_certificate_matrix(self.cayley_Q, self.lyap_p_raw)

    def _assemble_lyapunov_generator(self) -> Tensor:
        """Assemble the Lyapunov generator matrix.

        Returns
        -------
        Tensor
            Assembled matrix or factor for the active parameterization.
        """
        return assemble_lyapunov_generator(
            self.cayley_Q,
            self.lyap_diag_raw,
            self.max_real_eigenvalue,
        )

    def _assemble_generator(self) -> Tensor:
        """Assemble ``L`` for the active parameterization.

        Notes
        -----
        Thin wrapper that delegates to the shared continuous parameterization
        helpers for the active mode.
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
        return continuous_bound_metric(
            self.parameterization,
            max_real_eigenvalue=self.max_real_eigenvalue,
            diag_raw=getattr(self, "diag_raw", None),
            schur_diag_raw=getattr(self, "schur_diag_raw", None),
            lyap_diag_raw=getattr(self, "lyap_diag_raw", None),
            dissipative_generator=(
                self._dissipative_generator()
                if self.parameterization == "dissipative"
                else None
            ),
            assembled_generator=(self.L if self.parameterization == "dense" else None),
        )

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

        Raises
        ------
        ValueError
            If ``parameterization="auxiliary_spectral"``.
        """
        if self.parameterization == "auxiliary_spectral":
            msg = (
                "max_real_part is unavailable for parameterization="
                "'auxiliary_spectral'; use instantaneous_spectrum(z).real.max()"
            )
            raise ValueError(msg)
        return max_real_part_of_generator(self.L)

    def stability_certificate(self) -> StabilityCertificate | None:
        """Return a Hurwitz stability certificate when available.

        For ``"lyapunov"``, returns the Lyapunov matrix ``P`` and a positive
        margin. For ``"schur"`` and ``"dissipative"``, returns the margin from
        :meth:`bound_metric`. Returns ``None`` for ``"dense"``, ``"odo"``, and
        ``"auxiliary_spectral"`` (no structural global certificate).

        Returns
        -------
        StabilityCertificate or None
            Frozen certificate with ``margin`` and optional ``lyapunov_matrix``.
        """
        return continuous_stability_certificate(
            self.parameterization,
            bound_metric=(
                self.bound_metric()
                if self.parameterization in {"schur", "dissipative"}
                else None
            ),
            lyapunov_diagonal=(
                self._lyapunov_diagonal()
                if self.parameterization == "lyapunov"
                else None
            ),
            lyapunov_matrix=(
                self._lyapunov_matrix() if self.parameterization == "lyapunov" else None
            ),
        )
