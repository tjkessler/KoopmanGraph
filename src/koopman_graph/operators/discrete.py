"""Discrete finite-dimensional Koopman operator matrix **K**."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from koopman_graph.operators.contract import (
    DISSIPATIVE_MIN_EIGENVALUE,
    InitMode,
    Parameterization,
    StabilityCertificate,
    _bounded_diagonal,
    _safe_diagonal_inverse,
    _strict_diagonal_values,
    cayley_orthogonal,
    strict_spectral_bound,
)


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
            ignores the numeric value when assembling ``K``. Default is
            ``"dense"``.
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

        Raises
        ------
        ValueError
            If ``latent_dim < 1``, ``init_scale < 0``,
            ``max_spectral_radius <= 0``, structural modes receive
            ``max_spectral_radius > 1``, or ``control_dim < 0``.
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
    ) -> None:
        """Write dense Koopman parameters in place.

        Parameters
        ----------
        matrix : Tensor
            Dense operator ``K`` with shape ``(latent_dim, latent_dim)``.
        control_matrix : Tensor or None, optional
            Dense control matrix ``B`` with shape
            ``(control_dim, latent_dim)``. Required when ``control_dim > 0``.

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
                f"Expected matrix shape ({self.latent_dim}, {self.latent_dim}), "
                f"got {tuple(matrix.shape)}"
            )
            raise ValueError(msg)

        dense_k = self._parameters.get("K")
        if dense_k is None:
            raise AttributeError("K")
        with torch.no_grad():
            dense_k.copy_(matrix.to(device=dense_k.device, dtype=dense_k.dtype))
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
            elif control_matrix is not None:
                msg = "control_matrix provided to an uncontrolled operator"
                raise ValueError(msg)

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

    def _identity_strict_diag_raw(self) -> float:
        """Return raw diagonal parameters for a near-identity strict-stable mode.

        Returns
        -------
        float
            Unconstrained diagonal parameter mapped near the strict bound.
        """
        bound = strict_spectral_bound(self.max_spectral_radius)
        target = bound * (1.0 - 1e-6)
        ratio = target / bound
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

    def _reset_strict_diagonal(
        self,
        diag_param: nn.Parameter,
        *,
        cayley: nn.Parameter | None = None,
        off_param: nn.Parameter | None = None,
    ) -> None:
        """Initialize strict-stable Schur/Lyapunov diagonal and optional factors.

        Parameters
        ----------
        diag_param : nn.Parameter
            Diagonal parameter tensor to initialize.
        cayley : nn.Parameter or None, optional
            Optional Cayley parameter matrix for orthogonal factors.
        off_param : nn.Parameter or None, optional
            Optional upper-triangular off-diagonal parameters.

        Returns
        -------
        None
        """
        if cayley is not None:
            nn.init.zeros_(cayley)
        if off_param is not None:
            nn.init.zeros_(off_param)
        if self.init_mode == "identity":
            nn.init.constant_(diag_param, self._identity_strict_diag_raw())
        elif self.init_mode == "identity_noise":
            nn.init.constant_(diag_param, self._identity_strict_diag_raw())
            with torch.no_grad():
                bound = strict_spectral_bound(self.max_spectral_radius)
                noise = torch.randn_like(diag_param) * self.init_scale
                current = torch.tanh(diag_param) * bound
                updated = (current + noise).clamp(
                    min=-bound + 1e-6,
                    max=bound - 1e-6,
                )
                diag_param.copy_(torch.atanh(updated / bound))
        elif self.init_mode == "xavier":
            if cayley is not None:
                nn.init.xavier_uniform_(cayley)
            if off_param is not None:
                nn.init.xavier_uniform_(off_param)
                off_param.data.copy_(torch.triu(off_param.data, diagonal=1))
            nn.init.uniform_(diag_param, -0.5, 0.5)
        else:
            msg = f"Unknown init_mode: {self.init_mode!r}"
            raise ValueError(msg)

    def _reset_schur_parameters(self) -> None:
        """Reinitialize Schur-form parameters.

        Returns
        -------
        None
        """
        self._reset_strict_diagonal(
            self.schur_diag_raw,
            cayley=self.cayley_Q,
            off_param=self.schur_off_raw,
        )

    def _reset_dissipative_parameters(self) -> None:
        """Reinitialize dissipative generator parameters.

        Returns
        -------
        None
        """
        nn.init.zeros_(self.dissipative_L)
        if self.init_mode == "identity_noise":
            with torch.no_grad():
                noise = torch.randn_like(self.dissipative_L) * self.init_scale
                self.dissipative_L.add_(noise)
        elif self.init_mode == "xavier":
            nn.init.xavier_uniform_(self.dissipative_L)
            self.dissipative_L.data.copy_(torch.tril(self.dissipative_L.data))

    def _reset_lyapunov_parameters(self) -> None:
        """Reinitialize Lyapunov-certified symmetric parameters.

        Returns
        -------
        None
        """
        self._reset_strict_diagonal(self.lyap_diag_raw, cayley=self.cayley_Q)
        nn.init.zeros_(self.lyap_p_raw)

    def _odo_orthogonal_factors(self) -> tuple[Tensor, Tensor]:
        """Build orthogonal factors for the ODO parameterization.

        Returns
        -------
        tuple of Tensor
            Orthogonal matrices ``(O_1, O_2)``.
        """
        return cayley_orthogonal(self.cayley_O1), cayley_orthogonal(self.cayley_O2)

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

    def _schur_triangular(self) -> Tensor:
        """Build the upper-triangular Schur factor ``T``.

        Returns
        -------
        Tensor
            Upper-triangular Schur factor with bounded diagonal.
        """
        diag_vals = _strict_diagonal_values(
            self.schur_diag_raw,
            self.max_spectral_radius,
        )
        triangular = torch.triu(self.schur_off_raw, diagonal=1)
        return triangular + torch.diag(diag_vals)

    def _assemble_schur_matrix(self) -> Tensor:
        """Assemble ``K = Q T Q^T`` from Schur factors.

        Returns
        -------
        Tensor
            Assembled Schur-form operator matrix.
        """
        q = cayley_orthogonal(self.cayley_Q)
        return q @ self._schur_triangular() @ q.T

    def _dissipative_factor(self) -> Tensor:
        """Build the lower-triangular factor ``L`` for the generator ``S``.

        Returns
        -------
        Tensor
            Lower-triangular factor with positive diagonal entries.
        """
        lower = torch.tril(self.dissipative_L)
        diag_index = torch.arange(self.latent_dim, device=lower.device)
        lower[diag_index, diag_index] = (
            torch.nn.functional.softplus(lower[diag_index, diag_index])
            + DISSIPATIVE_MIN_EIGENVALUE
        )
        return lower

    def _dissipative_generator(self) -> Tensor:
        """Build the SPD generator ``S = L L^T + \\varepsilon I``.

        Returns
        -------
        Tensor
            Symmetric positive-definite generator matrix.
        """
        factor = self._dissipative_factor()
        identity = torch.eye(
            self.latent_dim,
            device=factor.device,
            dtype=factor.dtype,
        )
        return factor @ factor.T + DISSIPATIVE_MIN_EIGENVALUE * identity

    def _assemble_dissipative_matrix(self) -> Tensor:
        """Assemble ``K = exp(-S)`` from the dissipative generator.

        Returns
        -------
        Tensor
            Symmetric contractive operator matrix.
        """
        generator = self._dissipative_generator()
        return torch.linalg.matrix_exp(-generator)

    def _lyapunov_diagonal(self) -> Tensor:
        """Return strict stable eigenvalues for the Lyapunov parameterization.

        Returns
        -------
        Tensor
            Diagonal eigenvalues strictly inside
            ``(-max_spectral_radius, max_spectral_radius)`` with
            ``max_spectral_radius <= 1``, so they lie inside the unit disk.
        """
        return _strict_diagonal_values(self.lyap_diag_raw, self.max_spectral_radius)

    def _lyapunov_matrix(self) -> Tensor:
        """Return the Lyapunov certificate matrix ``P = Q diag(p) Q^T``.

        Returns
        -------
        Tensor
            Symmetric positive-definite Lyapunov matrix.
        """
        q = cayley_orthogonal(self.cayley_Q)
        p = torch.nn.functional.softplus(self.lyap_p_raw) + 1e-6
        return q @ torch.diag(p) @ q.T

    def _assemble_lyapunov_matrix(self) -> Tensor:
        """Assemble ``K = Q diag(d) Q^T`` with Lyapunov certificate ``P``.

        Returns
        -------
        Tensor
            Lyapunov-certified symmetric operator matrix.
        """
        q = cayley_orthogonal(self.cayley_Q)
        return q @ torch.diag(self._lyapunov_diagonal()) @ q.T

    def _assemble_matrix(self) -> Tensor:
        """Assemble ``K`` for the active non-dense parameterization.

        Returns
        -------
        Tensor
            Assembled operator matrix.
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
            return _strict_diagonal_values(raw, self.max_spectral_radius).abs().max()
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
            return StabilityCertificate(
                margin=torch.as_tensor(1.0 - radius),
                lyapunov_matrix=self._lyapunov_matrix(),
            )
        if self.parameterization in {"schur", "dissipative"}:
            radius = self.bound_metric()
            return StabilityCertificate(margin=torch.as_tensor(1.0 - radius))
        return None

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
        inverse_k = self._inverse_matrix(inverse_matrix=inverse_matrix)
        return adjusted @ inverse_k.T

    def advance(
        self,
        z: Tensor,
        delta_t: float | Tensor | None = None,
        *,
        control: Tensor | None = None,
    ) -> Tensor:
        """Advance latent states by one discrete Koopman step.

        Implements :class:`KoopmanOperatorContract`. ``delta_t`` is accepted for
        API symmetry with continuous operators and is ignored.

        Parameters
        ----------
        z : Tensor
            Latent states with shape ``(..., latent_dim)``.
        delta_t : float, Tensor, or None, optional
            Ignored for discrete operators.
        control : Tensor or None, optional
            Exogenous control input. Required when :attr:`control_dim` is
            positive.

        Returns
        -------
        Tensor
            Advanced latent states with the same shape as ``z``.
        """
        _ = delta_t
        return self.forward(z, control=control)

    def inverse_advance(
        self,
        z: Tensor,
        delta_t: float | Tensor | None = None,
        *,
        control: Tensor | None = None,
        inverse_matrix: Tensor | None = None,
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

        Returns
        -------
        Tensor
            Recovered latent states at time ``t``.
        """
        _ = delta_t
        return self.inverse_step(
            z,
            control=control,
            inverse_matrix=inverse_matrix,
        )

    def _inverse_matrix(self, *, inverse_matrix: Tensor | None = None) -> Tensor:
        """Return ``K^{-1}`` for the active parameterization.

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
            if inverse_matrix is not None:
                return inverse_matrix
            return self.dense_inverse_matrix()
        if self.parameterization == "odo":
            o1, o2 = self._odo_orthogonal_factors()
            diag_values = torch.diag(self._odo_diagonal())
            return o2 @ _safe_diagonal_inverse(diag_values) @ o1.T
        if self.parameterization == "schur":
            q = cayley_orthogonal(self.cayley_Q)
            triangular = self._schur_triangular()
            triangular_inv = torch.linalg.inv(triangular)
            return q @ triangular_inv @ q.T
        if self.parameterization == "dissipative":
            generator = self._dissipative_generator()
            return torch.linalg.matrix_exp(generator)
        if self.parameterization == "lyapunov":
            q = cayley_orthogonal(self.cayley_Q)
            return q @ _safe_diagonal_inverse(self._lyapunov_diagonal()) @ q.T
        msg = f"Unknown parameterization: {self.parameterization!r}"
        raise ValueError(msg)

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
