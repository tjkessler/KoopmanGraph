"""Discrete finite-dimensional Koopman operator matrix **K**.

Thin string-mode orchestrator for dense / ODO / Schur / dissipative /
Lyapunov parameterizations. Cohesive assembly / reset and advance / inverse
execution live in shallow sibling modules
(:mod:`~koopman_graph.operators.discrete_parameterizations`,
:mod:`~koopman_graph.operators.discrete_propagation`) and are re-exported
here so deep imports keep working.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from koopman_graph.operators.contract import (
    InitMode,
    Parameterization,
    StabilityCertificate,
    build_stability_certificate,
    strict_diagonal_values,
)
from koopman_graph.operators.control import (
    ControlMode,
    allocate_bilinear_parameters,
    bilinear_coupling_tensor,
    effective_bilinear_matrix,
    map_control_term,
    reset_bilinear_parameters,
    validate_control_mode,
    write_dense_operator_parameters,
)
from koopman_graph.operators.discrete_parameterizations import (
    assemble_dissipative_matrix,
    assemble_lyapunov_matrix,
    assemble_odo_matrix,
    assemble_schur_matrix,
    dissipative_generator,
    identity_diag_raw,
    identity_strict_diag_raw,
    lyapunov_certificate_matrix,
    lyapunov_diagonal,
    odo_diagonal,
    odo_orthogonal_factors,
    reset_dense_matrix,
    reset_dissipative_matrix,
    reset_lyapunov_matrix,
    reset_odo_matrix,
    reset_schur_matrix,
    schur_triangular,
)
from koopman_graph.operators.discrete_propagation import (
    advance_step,
    dense_inverse_or_pinv,
    inverse_matrix_for_parameterization,
)
from koopman_graph.operators.discrete_propagation import (
    inverse_step as propagate_inverse_step,
)

__all__ = [
    "KoopmanOperator",
    "identity_diag_raw",
    "identity_strict_diag_raw",
]


class KoopmanOperator(nn.Module):
    """Learnable finite-dimensional Koopman operator matrix **K**.

    Applies the same linear map to each node's latent vector. For input ``z`` with
    trailing dimension ``latent_dim``, the uncontrolled forward pass computes::

        z_next = z @ K.T

    When :attr:`control_dim` is positive, exogenous inputs drive the transition.
    With default ``control_mode="additive"``::

        z_next = z @ K.T + u @ B

    With ``control_mode="bilinear"`` (control-affine / bilinear Koopman)::

        z_next = z @ K.T + u @ B + sum_i u[..., i] * (z @ N_i.T)

    where ``K`` has shape ``(latent_dim, latent_dim)``, ``B`` has shape
    ``(control_dim, latent_dim)``, and each ``N_i`` is either a full
    ``(latent_dim, latent_dim)`` matrix or a low-rank factor
    ``N_i = P_i Q_i^T`` when ``bilinear_rank`` is set. Global controls ``u``
    with shape ``(control_dim,)`` are broadcast to every node; per-node
    controls use shape ``(num_nodes, control_dim)``. Arbitrary leading
    dimensions are supported (e.g. ``(num_nodes, latent_dim)`` or
    ``(batch, num_nodes, latent_dim)``).

    Beyond unconstrained ``"dense"`` storage, KoopmanGraph offers **soft**
    regularization and **structural** stability parameterizations. See
    **Stability modes** below before choosing ``parameterization``.

    Stability modes
    ---------------
    **Soft (no strict structural certificate):**

    - ``"dense"`` — unconstrained learnable matrix. Pair with
      :class:`~koopman_graph.losses.EigenvalueRegularizationLoss` during training
      for empirical unit-circle penalization.
    - ``"odo"`` — orthogonal–diagonal–orthogonal factorization
      (DeepKoopFormer-style). Cayley factors are orthogonal, so
      ``\\|K\\|_2 = \\|D\\|_2 = \\max |d_i|`` and therefore
      ``\\rho(K) \\leq \\max |d_i| \\leq max_spectral_radius``. This is an
      operator-norm bound, not a structural certificate: there is no strict
      ε-interior margin, no Lyapunov certificate, and non-normal transient
      growth remains possible. :meth:`bound_metric` reports the diagonal-factor
      bound; :meth:`spectral_radius` returns ``\\max |\\lambda_i(K)|`` via
      ``eigvals``. Prefer structural modes for long-horizon guarantees.

    **Structural (eigenvalues forced strictly inside the unit disk):**

    Opt-in modes force eigenvalues strictly inside the unit disk for
    long-horizon stability (Mallada, 2025; stability-constrained Deep Koopman
    literature). For these modes ``max_spectral_radius`` must lie in
    ``(0, 1]``:

    - ``"schur"`` — orthogonally similar upper-triangular form
      ``K = Q T Q^T`` with real diagonal magnitudes strictly inside
      ``(-max_spectral_radius, max_spectral_radius)`` via a ``tanh`` map
      (forces real eigenvalues; unlike classical real Schur, there are no
      2×2 blocks).
    - ``"dissipative"`` — symmetric contraction ``K = exp(-S)`` with
      ``S = L L^T + \\varepsilon I`` positive definite. The contraction is
      fixed by :data:`DISSIPATIVE_MIN_EIGENVALUE`; ``max_spectral_radius`` is
      accepted for API consistency but does not tighten the bound.
    - ``"lyapunov"`` — ``K = Q \\operatorname{diag}(d) Q^T`` with certified
      Lyapunov matrix ``P = Q \\operatorname{diag}(p) Q^T`` and
      ``|d_i| < max_spectral_radius \\leq 1``.

    Attributes
    ----------
    latent_dim : int
        Dimension of the latent space.
    control_dim : int
        Dimension of exogenous control inputs. Zero disables control.
    control_mode : {"additive", "bilinear"}
        How controls enter the latent map. ``"additive"`` uses ``u @ B`` only;
        ``"bilinear"`` adds state–control couplings ``N_i``.
    bilinear_rank : int or None
        When ``control_mode="bilinear"``, optional low-rank size for
        ``N_i = P_i Q_i^T``. ``None`` stores full-rank ``N_i``.
    init_mode : str
        Weight initialization strategy for ``K``.
    init_scale : float
        Noise scale used when ``init_mode="identity_noise"``.
    parameterization : str
        Parameterization used for ``K``.
    max_spectral_radius : float
        Target spectral bound. For ``"odo"`` this may exceed ``1``. For
        structural modes it must be in ``(0, 1]``; Schur/Lyapunov use
        ``max_spectral_radius - STABILITY_EPS_MARGIN`` internally, while
        dissipative ignores the value for matrix assembly.
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
        control_mode: ControlMode = "additive",
        bilinear_rank: int | None = None,
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
        parameterization : {"dense", "odo", "schur", "dissipative", "lyapunov"},
            optional
            Matrix parameterization. ``"dense"`` stores ``K`` directly with no
            stability guarantee (optional eigenvalue loss during training).
            ``"odo"`` factorizes ``K = O_1 D O_2^\\top`` with orthogonal Cayley
            factors and a bounded diagonal ``D``, which implies
            ``\\rho(K) \\leq \\max |d_i| \\leq max_spectral_radius`` via the
            operator 2-norm, but without a strict ε-interior certificate.
            ``"schur"``, ``"dissipative"``, and ``"lyapunov"`` embed
            **structural** stability guarantees (strict unit-disk eigenvalues)
            and require ``max_spectral_radius`` in ``(0, 1]``. Dissipative
            ignores the numeric value when assembling ``K``. Continuous-only
            ``"auxiliary_spectral"`` is rejected here. Default is ``"dense"``.
        max_spectral_radius : float, optional
            Target spectral bound for ``"odo"`` diagonal factors and for
            Schur/Lyapunov diagonals. Soft modes (``"odo"``) may use values
            greater than ``1``. Structural modes require ``(0, 1]``; Schur and
            Lyapunov enforce a strict interior margin of
            :data:`STABILITY_EPS_MARGIN` below this value, while dissipative
            always contracts via :data:`DISSIPATIVE_MIN_EIGENVALUE`. Default is
            ``1.0``.
        control_dim : int, optional
            Dimension of exogenous control inputs. When ``0``, the operator
            is uncontrolled. When positive, a learnable input matrix ``B``
            with shape ``(control_dim, latent_dim)`` is added. Default is
            ``0``.
        control_mode : {"additive", "bilinear"}, optional
            Control coupling. ``"additive"`` (default) uses ``u @ B`` only.
            ``"bilinear"`` adds state–control terms for control-affine systems.
        bilinear_rank : int or None, optional
            Low-rank size for bilinear factors when ``control_mode="bilinear"``.
            ``None`` (default) stores full-rank ``N_i``.

        Raises
        ------
        ValueError
            If ``latent_dim < 1``, ``init_scale < 0``,
            ``max_spectral_radius <= 0``, structural modes receive
            ``max_spectral_radius > 1``, ``control_dim < 0``, or control-mode
            settings are invalid.
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
        if (
            parameterization in {"schur", "dissipative", "lyapunov"}
            and max_spectral_radius > 1.0
        ):
            msg = (
                "structural parameterizations require max_spectral_radius <= 1 "
                f"for unit-disk certificates, got {max_spectral_radius}"
            )
            raise ValueError(msg)
        if control_dim < 0:
            msg = f"control_dim must be non-negative, got {control_dim}"
            raise ValueError(msg)
        if parameterization == "auxiliary_spectral":
            msg = (
                "parameterization='auxiliary_spectral' is continuous-only; "
                "use ContinuousKoopmanOperator (dynamics_mode='continuous')"
            )
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
        self.max_spectral_radius = max_spectral_radius
        self.control_dim = control_dim
        self.control_mode = control_mode
        self.bilinear_rank = bilinear_rank

        if parameterization == "dense":
            self.register_parameter(
                "K",
                nn.Parameter(torch.empty(latent_dim, latent_dim)),
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
        """Reinitialize the control input matrix ``B`` (and bilinear factors).

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

        Raises
        ------
        ValueError
            If ``control_mode`` is not ``"bilinear"``.
        """
        if self.control_mode != "bilinear":
            msg = "bilinear_matrices requires control_mode='bilinear'"
            raise ValueError(msg)
        return bilinear_coupling_tensor(self)

    def effective_state_matrix(self, control: Tensor) -> Tensor:
        """Return ``K + sum_i u_i N_i`` for a global bilinear control.

        Parameters
        ----------
        control : Tensor
            Global control with shape ``(control_dim,)``.

        Returns
        -------
        Tensor
            Effective discrete map with shape ``(latent_dim, latent_dim)``.
        """
        if self.control_mode != "bilinear":
            return self.K
        return effective_bilinear_matrix(self.K, control, self.bilinear_matrices())

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
        return map_control_term(
            u,
            getattr(self, "B", None),
            control_dim=self.control_dim,
            num_nodes=num_nodes,
        )

    @property
    def K(self) -> Tensor:
        """Assembled Koopman matrix with shape ``(latent_dim, latent_dim)``.

        For ``parameterization="dense"`` this is the learnable parameter.
        Other modes assemble ``K`` from their factorized parameters.
        Prefer :attr:`matrix` when writing code against
        :class:`KoopmanOperatorContract`.

        Returns
        -------
        Tensor
            Current operator matrix ``K``.
        """
        if self.parameterization == "dense":
            dense_k = self._parameters.get("K")
            if dense_k is None:
                raise AttributeError("K")
            return dense_k
        return self._assemble_matrix()

    @property
    def matrix(self) -> Tensor:
        """Assembled operator matrix (alias of :attr:`K`).

        Returns
        -------
        Tensor
            Current discrete Koopman matrix ``K``.
        """
        return self.K

    def set_dense_matrix(
        self,
        matrix: Tensor,
        *,
        control_matrix: Tensor | None = None,
        bilinear_matrices: Tensor | None = None,
    ) -> None:
        """Write dense Koopman parameters in place.

        Parameters
        ----------
        matrix : Tensor
            Dense operator ``K`` with shape ``(latent_dim, latent_dim)``.
        control_matrix : Tensor or None, optional
            Dense control matrix ``B`` with shape
            ``(control_dim, latent_dim)``. Required when ``control_dim > 0``.
        bilinear_matrices : Tensor or None, optional
            Full-rank bilinear stack ``N`` with shape
            ``(control_dim, latent_dim, latent_dim)``. Required when
            ``control_mode="bilinear"`` and ``bilinear_rank is None``.

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

        dense_k = self._parameters.get("K")
        if dense_k is None:
            raise AttributeError("K")
        write_dense_operator_parameters(
            dense_k,
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
            matrix_label="matrix",
        )

    def reset_parameters(self) -> None:
        """Reinitialize operator parameters according to :attr:`init_mode`.

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
        """Reinitialize the dense learnable matrix ``K``.

        Notes
        -----
        Thin wrapper that delegates to the shared discrete parameterization
        helpers for the active mode.
        """
        reset_dense_matrix(
            self._parameters["K"],
            init_mode=self.init_mode,
            init_scale=self.init_scale,
        )

    def _reset_odo_parameters(self) -> None:
        """Reinitialize Cayley and diagonal ODO parameters.

        Notes
        -----
        Thin wrapper that delegates to the shared discrete parameterization
        helpers for the active mode.
        """
        reset_odo_matrix(
            self.cayley_O1,
            self.cayley_O2,
            self.diag_raw,
            init_mode=self.init_mode,
            init_scale=self.init_scale,
            max_spectral_radius=self.max_spectral_radius,
        )

    def _reset_schur_parameters(self) -> None:
        """Reinitialize Schur-form parameters.

        Notes
        -----
        Thin wrapper that delegates to the shared discrete parameterization
        helpers for the active mode.
        """
        reset_schur_matrix(
            self.cayley_Q,
            self.schur_diag_raw,
            self.schur_off_raw,
            init_mode=self.init_mode,
            init_scale=self.init_scale,
            max_spectral_radius=self.max_spectral_radius,
        )

    def _reset_dissipative_parameters(self) -> None:
        """Reinitialize dissipative generator parameters.

        Notes
        -----
        Thin wrapper that delegates to the shared discrete parameterization
        helpers for the active mode.
        """
        reset_dissipative_matrix(
            self.dissipative_L,
            init_mode=self.init_mode,
            init_scale=self.init_scale,
        )

    def _reset_lyapunov_parameters(self) -> None:
        """Reinitialize Lyapunov-certified symmetric parameters.

        Notes
        -----
        Thin wrapper that delegates to the shared discrete parameterization
        helpers for the active mode.
        """
        reset_lyapunov_matrix(
            self.cayley_Q,
            self.lyap_diag_raw,
            self.lyap_p_raw,
            init_mode=self.init_mode,
            init_scale=self.init_scale,
            max_spectral_radius=self.max_spectral_radius,
        )

    def _odo_orthogonal_factors(self) -> tuple[Tensor, Tensor]:
        """Build orthogonal factors for the ODO parameterization.

        Returns
        -------
        Tensor
            Assembled matrix or factor for the active parameterization.
        """
        return odo_orthogonal_factors(self.cayley_O1, self.cayley_O2)

    def _odo_diagonal(self) -> Tensor:
        """Build the bounded diagonal factor for the ODO parameterization.

        Returns
        -------
        Tensor
            Assembled matrix or factor for the active parameterization.
        """
        return odo_diagonal(self.diag_raw, self.max_spectral_radius)

    def _assemble_odo_matrix(self) -> Tensor:
        """Assemble ``K = O_1 D O_2^T`` from ODO factors.

        Returns
        -------
        Tensor
            Assembled matrix or factor for the active parameterization.
        """
        return assemble_odo_matrix(
            self.cayley_O1,
            self.cayley_O2,
            self.diag_raw,
            self.max_spectral_radius,
        )

    def _schur_triangular(self) -> Tensor:
        """Build the upper-triangular Schur factor ``T``.

        Returns
        -------
        Tensor
            Assembled matrix or factor for the active parameterization.
        """
        return schur_triangular(
            self.schur_diag_raw,
            self.schur_off_raw,
            self.max_spectral_radius,
        )

    def _assemble_schur_matrix(self) -> Tensor:
        """Assemble ``K = Q T Q^T`` from Schur factors.

        Returns
        -------
        Tensor
            Assembled matrix or factor for the active parameterization.
        """
        return assemble_schur_matrix(
            self.cayley_Q,
            self.schur_diag_raw,
            self.schur_off_raw,
            self.max_spectral_radius,
        )

    def _dissipative_generator(self) -> Tensor:
        """Build the SPD generator ``S = L L^T + \\varepsilon I``.

        Returns
        -------
        Tensor
            Symmetric positive-definite dissipative generator.
        """
        return dissipative_generator(self.dissipative_L, self.latent_dim)

    def _assemble_dissipative_matrix(self) -> Tensor:
        """Assemble ``K = exp(-S)`` from the dissipative generator.

        Returns
        -------
        Tensor
            Assembled matrix or factor for the active parameterization.
        """
        return assemble_dissipative_matrix(self.dissipative_L, self.latent_dim)

    def _lyapunov_diagonal(self) -> Tensor:
        """Return strict stable eigenvalues for the Lyapunov parameterization.

        Returns
        -------
        Tensor
            Assembled matrix or factor for the active parameterization.
        """
        return lyapunov_diagonal(self.lyap_diag_raw, self.max_spectral_radius)

    def _lyapunov_matrix(self) -> Tensor:
        """Return the Lyapunov certificate matrix ``P = Q diag(p) Q^T``.

        Returns
        -------
        Tensor
            Assembled matrix or factor for the active parameterization.
        """
        return lyapunov_certificate_matrix(self.cayley_Q, self.lyap_p_raw)

    def _assemble_lyapunov_matrix(self) -> Tensor:
        """Assemble ``K = Q diag(d) Q^T`` with Lyapunov certificate ``P``.

        Returns
        -------
        Tensor
            Assembled matrix or factor for the active parameterization.
        """
        return assemble_lyapunov_matrix(
            self.cayley_Q,
            self.lyap_diag_raw,
            self.max_spectral_radius,
        )

    def _assemble_matrix(self) -> Tensor:
        """Assemble ``K`` for the active non-dense parameterization.

        Notes
        -----
        Thin wrapper that delegates to the shared discrete parameterization
        helpers for the active mode.
        """
        assemblers = {
            "odo": self._assemble_odo_matrix,
            "schur": self._assemble_schur_matrix,
            "dissipative": self._assemble_dissipative_matrix,
            "lyapunov": self._assemble_lyapunov_matrix,
        }
        return assemblers[self.parameterization]()

    def bound_metric(self) -> Tensor:
        """Return the cheap soft/structural monitoring bound.

        For ``"odo"``, this is the maximum bounded diagonal entry of ``D`` —
        **not** the spectral radius of assembled ``K``. For structurally
        stable modes, this is a closed-form certified upper bound on
        ``\\max |\\lambda_i(K)|``. For ``"dense"``, this equals
        :meth:`spectral_radius`. Prefer :meth:`bound_metric` when writing
        code against :class:`KoopmanOperatorContract`; use
        :meth:`spectral_radius` for the true spectrum via ``eigvals``.

        Returns
        -------
        Tensor
            Scalar bound metric for the active parameterization.
        """
        if self.parameterization == "odo":
            return torch.tanh(self.diag_raw).abs().max() * self.max_spectral_radius
        if self.parameterization in {"schur", "lyapunov"}:
            raw = (
                self.schur_diag_raw
                if self.parameterization == "schur"
                else self.lyap_diag_raw
            )
            return strict_diagonal_values(raw, self.max_spectral_radius).abs().max()
        if self.parameterization == "dissipative":
            generator = self._dissipative_generator()
            min_eval = torch.linalg.eigvalsh(generator).min()
            return torch.exp(-min_eval)
        return self.spectral_radius()

    def spectral_radius(self) -> Tensor:
        """Return the true spectral radius ``\\max |\\lambda_i(K)|``.

        Always computed from the assembled operator via ``eigvals``. For the
        cheap soft/structural monitoring bound (diagonal-factor bound for
        ``"odo"``, closed-form certificates for structural modes), use
        :meth:`bound_metric`.

        Returns
        -------
        Tensor
            Scalar tensor ``\\max_i |\\lambda_i(K)|``.
        """
        eigenvalues = torch.linalg.eigvals(self.K)
        return eigenvalues.abs().max()

    def stability_certificate(self) -> StabilityCertificate | None:
        """Return a stability certificate when the parameterization provides one.

        Discrete structural modes (``"schur"``, ``"dissipative"``,
        ``"lyapunov"``) all report the unit-disk gap ``1 - bound_metric``.
        Lyapunov also returns the certificate matrix ``P``. Returns ``None``
        for ``"dense"`` and ``"odo"``.

        Returns
        -------
        StabilityCertificate or None
            Frozen certificate with ``margin`` and optional ``lyapunov_matrix``.
        """
        if self.parameterization == "lyapunov":
            radius = self.bound_metric()
            return build_stability_certificate(
                torch.as_tensor(1.0 - radius),
                lyapunov_matrix=self._lyapunov_matrix(),
            )
        if self.parameterization in {"schur", "dissipative"}:
            radius = self.bound_metric()
            return build_stability_certificate(torch.as_tensor(1.0 - radius))
        return None

    def forward(self, z: Tensor, control: Tensor | None = None) -> Tensor:
        """Advance latent states by one linear Koopman step.

        When :attr:`control_dim` is positive and ``control_mode="additive"``::

            z_next = z @ K.T + u @ B

        When ``control_mode="bilinear"``::

            z_next = z @ K.T + u @ B + sum_i u[..., i] * (z @ N_i.T)

        Global control ``u`` has shape ``(control_dim,)``; per-node control
        has shape ``(num_nodes, control_dim)``. Thin wrapper around
        :func:`~koopman_graph.operators.discrete_propagation.advance_step`.

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
        coupling = self.bilinear_matrices() if self.control_mode == "bilinear" else None
        return advance_step(
            z,
            control,
            matrix=self.K,
            control_matrix=getattr(self, "B", None) if self.control_dim > 0 else None,
            control_dim=self.control_dim,
            control_mode=self.control_mode,
            latent_dim=self.latent_dim,
            coupling=coupling,
        )

    def inverse_step(
        self,
        z: Tensor,
        *,
        control: Tensor | None = None,
        inverse_matrix: Tensor | None = None,
    ) -> Tensor:
        """Apply one inverse Koopman step to recover the previous latent state.

        For additive control ``z_{t+1} = z_t @ K.T + u_t @ B``, this returns
        an estimate of ``z_t`` from ``z_{t+1}`` and the control ``u_t`` that
        drove the transition. For bilinear control with **global** ``u``, the
        effective map ``K_eff = K + sum_i u_i N_i`` is inverted. Per-node
        bilinear inverse applies a distinct ``K_eff`` per node. Thin wrapper
        around
        :func:`~koopman_graph.operators.discrete_propagation.inverse_step`.

        Parameters
        ----------
        z : Tensor
            Latent states at time ``t+1`` with shape ``(..., latent_dim)``.
        control : Tensor or None, optional
            Control input that drove the forward transition. Required when
            :attr:`control_dim` is positive.
        inverse_matrix : Tensor or None, optional
            Precomputed ``K^{-1}`` for dense additive parameterization. When
            omitted, the inverse is computed on demand. Ignored for bilinear
            mode (effective ``K_eff`` is inverted instead).

        Returns
        -------
        Tensor
            Recovered latent states at time ``t``, same shape as ``z``.
        """
        coupling = self.bilinear_matrices() if self.control_mode == "bilinear" else None
        use_bilinear = self.control_dim > 0 and self.control_mode == "bilinear"
        inverse_k = (
            None
            if use_bilinear
            else self._inverse_matrix(inverse_matrix=inverse_matrix)
        )
        return propagate_inverse_step(
            z,
            control=control,
            matrix=self.K,
            control_matrix=getattr(self, "B", None) if self.control_dim > 0 else None,
            control_dim=self.control_dim,
            control_mode=self.control_mode,
            latent_dim=self.latent_dim,
            coupling=coupling,
            inverse_matrix=inverse_k,
        )

    def advance(
        self,
        z: Tensor,
        delta_t: float | Tensor | None = None,
        *,
        control: Tensor | None = None,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        """Advance latent states by one discrete Koopman step.

        Implements :class:`KoopmanOperatorContract`. ``delta_t`` is accepted for
        API symmetry with continuous operators and is ignored. Topology kwargs
        are accepted for API symmetry with networked operators and ignored.

        Parameters
        ----------
        z : Tensor
            Latent states with shape ``(..., latent_dim)``.
        delta_t : float, Tensor, or None, optional
            Ignored for discrete operators.
        control : Tensor or None, optional
            Exogenous control input. Required when :attr:`control_dim` is
            positive.
        edge_index : Tensor or None, optional
            Ignored for per-node operators.
        edge_weight : Tensor or None, optional
            Ignored for per-node operators.

        Returns
        -------
        Tensor
            Advanced latent states with the same shape as ``z``.
        """
        _ = delta_t, edge_index, edge_weight
        return self.forward(z, control=control)

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
        """Recover the previous latent state (contract alias of :meth:`inverse_step`).

        Parameters
        ----------
        z : Tensor
            Latent states at time ``t+1``.
        delta_t : float, Tensor, or None, optional
            Ignored for discrete operators.
        control : Tensor or None, optional
            Control that drove the forward transition.
        inverse_matrix : Tensor or None, optional
            Optional precomputed ``K^{-1}``.
        edge_index : Tensor or None, optional
            Ignored for per-node operators.
        edge_weight : Tensor or None, optional
            Ignored for per-node operators.

        Returns
        -------
        Tensor
            Recovered latent states at time ``t``.
        """
        _ = delta_t, edge_index, edge_weight
        return self.inverse_step(
            z,
            control=control,
            inverse_matrix=inverse_matrix,
        )

    def _inverse_matrix(self, *, inverse_matrix: Tensor | None = None) -> Tensor:
        """Return ``K^{-1}`` for the active parameterization.

        Thin dispatcher around
        :func:`~koopman_graph.operators.discrete_propagation.inverse_matrix_for_parameterization`.

        Parameters
        ----------
        inverse_matrix : Tensor or None, optional
            Precomputed inverse for dense parameterization.

        Returns
        -------
        Tensor
            Inverse operator matrix with shape ``(latent_dim, latent_dim)``.
        """
        if self.parameterization == "dense":
            return inverse_matrix_for_parameterization(
                self.parameterization,
                dense_matrix=self.K,
                inverse_matrix=inverse_matrix,
            )
        if self.parameterization == "odo":
            o1, o2 = self._odo_orthogonal_factors()
            return inverse_matrix_for_parameterization(
                self.parameterization,
                odo_left=o1,
                odo_right=o2,
                odo_diagonal=self._odo_diagonal(),
            )
        if self.parameterization == "schur":
            return inverse_matrix_for_parameterization(
                self.parameterization,
                schur_cayley_q=self.cayley_Q,
                schur_triangular=self._schur_triangular(),
            )
        if self.parameterization == "dissipative":
            return inverse_matrix_for_parameterization(
                self.parameterization,
                dissipative_generator=self._dissipative_generator(),
            )
        if self.parameterization == "lyapunov":
            return inverse_matrix_for_parameterization(
                self.parameterization,
                lyapunov_cayley_q=self.cayley_Q,
                lyapunov_diagonal=self._lyapunov_diagonal(),
            )
        msg = f"Unknown parameterization: {self.parameterization!r}"
        raise ValueError(msg)

    def dense_inverse_matrix(self) -> Tensor:
        """Return the inverse (or pseudo-inverse) of the assembled dense matrix.

        Intended for reuse across multiple backward-consistency pair evaluations
        within one training step. Thin wrapper around
        :func:`~koopman_graph.operators.discrete_propagation.dense_inverse_or_pinv`.

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
        return dense_inverse_or_pinv(self.K)
