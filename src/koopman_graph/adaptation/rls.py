"""Recursive least-squares online adaptation for Koopman operators.

Part of :mod:`koopman_graph.adaptation` (peer to the Kalman observer).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from koopman_graph.graph_utils import KoopmanPropagator
from koopman_graph.operators import (
    ContinuousKoopmanOperator,
    KoopmanOperator,
    matrix_log,
    van_loan_factors,
    van_loan_generator_from_discrete,
)
from koopman_graph.protocols import DynamicsMode

#: Alias of :data:`~koopman_graph.protocols.DynamicsMode` for the RLS API.
AdaptationMode = DynamicsMode


@dataclass(frozen=True)
class AdaptationStepResult:
    """Diagnostics returned by one online adaptation step.

    Public result types in this package are frozen dataclasses with attribute
    access (not mapping/dict styles).

    Attributes
    ----------
    operator_change_norm : Tensor
        Frobenius norm of the change in the assembled operator matrix.
    """

    operator_change_norm: Tensor


class RecursiveKoopmanAdapter:
    """Recursive least squares (RLS) adapter for Koopman operator matrices.

    Online adaptation estimates a dense discrete propagator from streaming
    latent pairs using the library row convention::

        z_{t+1} = z_t @ K.T + u_t @ B

    In continuous mode the adapter tracks the effective propagator
    ``K(Δt)`` for the observed interval and writes back a dense generator
    that matches
    :meth:`~koopman_graph.operators.ContinuousKoopmanOperator.advance`:

    - Uncontrolled: ``L = logm(K(Δt)) / Δt`` (exact for
      ``K(Δt) = exp(L · Δt)``).
    - Controlled: recover ``(L, B)`` by inverting the Van Loan block-matrix
      exponential used in forward integration.

    Only ``parameterization="dense"`` operators are supported for write-back.

    Control layouts match the neural operator: global ``(control_dim,)``
    broadcasts to every latent row; per-node ``(num_nodes, control_dim)``
    keeps one control row per latent sample. Classical
    :class:`~koopman_graph.baselines.DMDcBaseline` rejects 3-D sequence
    controls (see architecture control layout capability matrix).

    **Built-in operators only.** ``from_operator`` / ``apply_to`` accept
    :class:`~koopman_graph.operators.KoopmanOperator` and
    :class:`~koopman_graph.operators.ContinuousKoopmanOperator`. Custom
    ``koopman=`` injections that only satisfy
    :class:`~koopman_graph.operators.KoopmanOperatorContract` are supported
    for train/predict/spectrum paths, but **not** for RLS seed or write-back
    (no portable dense-parameter setter on the Protocol).

    Notes
    -----
    **Discrete mode** (``mode="discrete"``) is exact for the fitted row
    convention: RLS directly estimates ``K`` (and ``B`` when controlled).

    **Continuous mode** fits a *discrete* propagator ``K(Δt)`` (and discrete
    control block when controlled) per observed interval, then maps to
    continuous parameters aligned with Van Loan / matrix-exponential forward
    dynamics. Write-back fidelity is typically within
    :data:`~koopman_graph.operators.VAN_LOAN_WRITEBACK_ATOL` for moderate
    ``Δt``.

    Recovery can still degrade when:

    - ``Δt`` is large or ``K(Δt)`` lies near a matrix-logarithm branch cut,
    - ``Δt`` varies across adaptation steps (each update overwrites a single
      reference interval).

    Prefer ``mode="discrete"`` for uniformly sampled data when a discrete
    operator is acceptable. Continuous mode is appropriate when the model uses
    ``dynamics_mode="continuous"`` and adapted parameters must match
    continuous forward propagation.

    Historical note: prior to TASK-704, continuous write-back used the
    first-order approximations ``L ≈ logm(K)/Δt`` with ``B̃ ≈ B(Δt)/Δt``
    for controlled systems, which disagreed with Van Loan integration.

    Attributes
    ----------
    latent_dim : int
        Latent dimension.
    control_dim : int
        Control dimension. Zero disables control.
    mode : str
        Adaptation mode: ``"discrete"`` or ``"continuous"``.
    forgetting_factor : float
        Exponential forgetting factor ``λ`` in ``(0, 1]``.
    regularization : float
        Initial covariance scale on the regressor covariance ``P``.
    """

    def __init__(
        self,
        latent_dim: int,
        *,
        control_dim: int = 0,
        mode: AdaptationMode = "discrete",
        forgetting_factor: float = 0.99,
        regularization: float = 1e3,
        initial_k: Tensor | None = None,
        initial_b: Tensor | None = None,
        initial_l: Tensor | None = None,
    ) -> None:
        """Initialize recursive least-squares adaptation state.

        Parameters
        ----------
        latent_dim : int
            Latent dimension.
        control_dim : int, optional
            Control dimension. Default is ``0``.
        mode : {"discrete", "continuous"}, optional
            Whether updates apply to a discrete operator or a continuous
            generator. Default is ``"discrete"``.
        forgetting_factor : float, optional
            Exponential forgetting factor ``λ`` in ``(0, 1]``. Default is
            ``0.99``.
        regularization : float, optional
            Initial covariance scale on the regressor covariance ``P``.
            Larger values mean lower initial confidence. Default is ``1e3``.
        initial_k : Tensor or None, optional
            Initial discrete propagator with shape ``(latent_dim, latent_dim)``.
        initial_b : Tensor or None, optional
            Initial control matrix with shape ``(control_dim, latent_dim)``.
        initial_l : Tensor or None, optional
            Initial generator with shape ``(latent_dim, latent_dim)`` for
            continuous mode.

        Raises
        ------
        ValueError
            If configuration values are invalid.
        """
        if latent_dim < 1:
            msg = f"latent_dim must be positive, got {latent_dim}"
            raise ValueError(msg)
        if control_dim < 0:
            msg = f"control_dim must be non-negative, got {control_dim}"
            raise ValueError(msg)
        if not 0.0 < forgetting_factor <= 1.0:
            msg = f"forgetting_factor must be in (0, 1], got {forgetting_factor}"
            raise ValueError(msg)
        if regularization <= 0.0:
            msg = f"regularization must be positive, got {regularization}"
            raise ValueError(msg)

        self.latent_dim = latent_dim
        self.control_dim = control_dim
        self.mode = mode
        self.forgetting_factor = forgetting_factor
        self.regularization = regularization
        self._reference_delta_t = 1.0

        regressor_dim = latent_dim + control_dim
        self._weights = torch.zeros(regressor_dim, latent_dim)
        self._covariance = torch.eye(regressor_dim) * regularization

        if mode == "discrete":
            if initial_k is None:
                initial_k = torch.eye(latent_dim)
            self._set_from_discrete(initial_k, initial_b)
        else:
            if initial_l is None:
                initial_l = torch.zeros(latent_dim, latent_dim)
            if initial_b is None and control_dim > 0:
                initial_b = torch.zeros(control_dim, latent_dim)
            self._set_from_generator(initial_l, initial_b)

    @classmethod
    def from_operator(
        cls,
        koopman: KoopmanPropagator,
        *,
        mode: AdaptationMode,
        forgetting_factor: float = 0.99,
        regularization: float = 1e3,
    ) -> RecursiveKoopmanAdapter:
        """Build an adapter seeded from an existing Koopman operator.

        Parameters
        ----------
        koopman : KoopmanOperator or ContinuousKoopmanOperator
            Source operator. Must use ``parameterization="dense"``.
        mode : {"discrete", "continuous"}
            Adaptation mode.
        forgetting_factor : float, optional
            RLS forgetting factor. Default is ``0.99``.
        regularization : float, optional
            Initial covariance scale. Default is ``1e3``.

        Returns
        -------
        RecursiveKoopmanAdapter
            Adapter initialized from ``koopman``.

        Raises
        ------
        ValueError
            If ``koopman`` does not use dense parameterization.
        TypeError
            If ``koopman`` is not a supported operator type.
        """
        cls._validate_dense_parameterization(koopman)
        if isinstance(koopman, KoopmanOperator):
            initial_k = koopman.K.detach().cpu()
            initial_b = koopman.B.detach().cpu() if koopman.control_dim > 0 else None
            return cls(
                koopman.latent_dim,
                control_dim=koopman.control_dim,
                mode="discrete",
                forgetting_factor=forgetting_factor,
                regularization=regularization,
                initial_k=initial_k,
                initial_b=initial_b,
            )
        if isinstance(koopman, ContinuousKoopmanOperator):
            initial_l = koopman.L.detach().cpu()
            initial_b = koopman.B.detach().cpu() if koopman.control_dim > 0 else None
            return cls(
                koopman.latent_dim,
                control_dim=koopman.control_dim,
                mode="continuous",
                forgetting_factor=forgetting_factor,
                regularization=regularization,
                initial_l=initial_l,
                initial_b=initial_b,
            )
        msg = (
            "Online adaptation seed/write-back supports built-in "
            "KoopmanOperator and ContinuousKoopmanOperator only "
            f"(got {type(koopman).__name__}). Custom koopman= injections "
            "are Protocol-capable for train/predict/spectrum but not RLS."
        )
        raise TypeError(msg)

    @staticmethod
    def _validate_dense_parameterization(koopman: KoopmanPropagator) -> None:
        """Require dense parameterization for online write-back.

        Raises
        ------
        ValueError
            If the operator is not densely parameterized.
        """
        if koopman.parameterization != "dense":
            msg = (
                "Online adaptation requires dense Koopman parameterization; "
                f"got {koopman.parameterization!r}. Train with "
                "koopman_parameterization='dense' before enabling adaptation."
            )
            raise ValueError(msg)

    def _set_from_discrete(
        self,
        k_matrix: Tensor,
        b_matrix: Tensor | None,
    ) -> None:
        """Initialize regression weights from discrete ``K`` and optional ``B``.

        Parameters
        ----------
        k_matrix : Tensor
            Discrete propagator with shape ``(latent_dim, latent_dim)``.
        b_matrix : Tensor or None
            Control matrix with shape ``(control_dim, latent_dim)``.

        Raises
        ------
        ValueError
            If ``control_dim > 0`` but ``b_matrix`` is ``None``.
        """
        k_matrix = k_matrix.detach().cpu()
        self._weights[: self.latent_dim] = k_matrix.T
        if self.control_dim > 0:
            if b_matrix is None:
                msg = "initial_b is required when control_dim > 0"
                raise ValueError(msg)
            self._weights[self.latent_dim :] = b_matrix.detach().cpu()

    def _set_from_generator(
        self,
        generator: Tensor,
        b_matrix: Tensor | None,
        *,
        delta_t: float = 1.0,
    ) -> None:
        """Initialize regression weights from a generator matrix.

        Parameters
        ----------
        generator : Tensor
            Continuous generator with shape ``(latent_dim, latent_dim)``.
        b_matrix : Tensor or None
            Generator control matrix with shape ``(control_dim, latent_dim)``.
        delta_t : float, optional
            Reference interval used to map ``L`` to a discrete propagator.
            Default is ``1.0``.
        """
        generator = generator.detach().cpu()
        if self.control_dim == 0 or b_matrix is None:
            propagator = torch.linalg.matrix_exp(generator * delta_t)
            discrete_control = None
        else:
            phi11, phi12 = van_loan_factors(
                generator,
                b_matrix.detach().cpu(),
                delta_t,
            )
            propagator = phi11
            discrete_control = phi12.T
        self._reference_delta_t = delta_t
        self._set_from_discrete(propagator, discrete_control)

    @property
    def discrete_matrix(self) -> Tensor:
        """Current dense discrete propagator ``K``.

        Returns
        -------
        Tensor
            Matrix with shape ``(latent_dim, latent_dim)``.
        """
        return self._weights[: self.latent_dim].T

    @property
    def control_matrix(self) -> Tensor | None:
        """Current discrete control matrix ``B``.

        Returns
        -------
        Tensor or None
            Matrix with shape ``(control_dim, latent_dim)`` when controlled.
        """
        if self.control_dim == 0:
            return None
        return self._weights[self.latent_dim :]

    @property
    def generator_matrix(self) -> Tensor:
        """Current dense generator ``L`` for continuous mode.

        Returns
        -------
        Tensor
            Matrix with shape ``(latent_dim, latent_dim)``.
        """
        generator, _ = self._continuous_parameters()
        return generator

    @property
    def generator_control_matrix(self) -> Tensor | None:
        """Current continuous control matrix ``B`` for continuous mode.

        Returns
        -------
        Tensor or None
            Matrix with shape ``(control_dim, latent_dim)`` when controlled.
        """
        if self.control_dim == 0:
            return None
        _, control = self._continuous_parameters()
        return control

    def _continuous_parameters(self) -> tuple[Tensor, Tensor | None]:
        """Map the current discrete RLS estimate to continuous ``(L, B)``.

        Returns
        -------
        tuple[Tensor, Tensor or None]
            Generator and optional continuous control matrix.
        """
        if self.control_dim == 0:
            return (
                matrix_log(self.discrete_matrix) / self._reference_delta_t,
                None,
            )
        control = self.control_matrix
        assert control is not None
        return van_loan_generator_from_discrete(
            self.discrete_matrix,
            control,
            self._reference_delta_t,
        )

    def update(
        self,
        z_t: Tensor,
        z_tp1: Tensor,
        *,
        control: Tensor | None = None,
        delta_t: float | Tensor | None = None,
    ) -> AdaptationStepResult:
        """Incorporate one latent transition into the RLS estimate.

        Parameters
        ----------
        z_t : Tensor
            Source latent states with shape ``(latent_dim,)`` or
            ``(num_nodes, latent_dim)``.
        z_tp1 : Tensor
            Target latent states with the same shape as ``z_t``.
        control : Tensor or None, optional
            Control input for the transition. Required when
            ``control_dim > 0``.
        delta_t : float or Tensor or None, optional
            Integration interval for continuous mode. Ignored in discrete mode.

        Returns
        -------
        AdaptationStepResult
            Diagnostics for the update.

        Notes
        -----
        In continuous mode, each update fits a discrete propagator for the
        supplied ``delta_t``; write-back via :meth:`apply_to` recovers a
        Van Loan-aligned generator (and control matrix when controlled).

        Raises
        ------
        ValueError
            If shapes are invalid, controls are missing, or continuous mode
            is used without ``delta_t``.
        """
        if self.mode == "continuous" and delta_t is None:
            msg = "delta_t is required for continuous-mode adaptation"
            raise ValueError(msg)

        z_rows = self._as_rows(z_t)
        y_rows = self._as_rows(z_tp1)
        if z_rows.shape != y_rows.shape:
            msg = (
                f"z_t and z_tp1 must have matching shapes after flattening, "
                f"got {tuple(z_rows.shape)} and {tuple(y_rows.shape)}"
            )
            raise ValueError(msg)

        previous_matrix = self.discrete_matrix.clone()
        control_row = self._resolve_control_row(control, batch_rows=z_rows.shape[0])

        if self.mode == "continuous":
            assert delta_t is not None
            delta = float(torch.as_tensor(delta_t).item())
            if delta <= 0.0:
                msg = f"delta_t must be positive, got {delta}"
                raise ValueError(msg)
            self._reference_delta_t = delta

        for index in range(z_rows.shape[0]):
            control_slice = None
            if control_row is not None:
                control_slice = (
                    control_row[index] if control_row.ndim == 2 else control_row
                )
            phi = self._build_regressor(z_rows[index], control_slice)
            self._rls_update(phi, y_rows[index])

        change = torch.linalg.norm(self.discrete_matrix - previous_matrix)
        return AdaptationStepResult(operator_change_norm=change)

    def apply_to(self, koopman: KoopmanPropagator) -> None:
        """Write the current RLS estimate into ``koopman`` parameters.

        Parameters
        ----------
        koopman : KoopmanOperator or ContinuousKoopmanOperator
            Dense operator to update in place.

        Notes
        -----
        For :class:`~koopman_graph.operators.ContinuousKoopmanOperator`,
        generator and control parameters are recovered so that
        :meth:`~koopman_graph.operators.ContinuousKoopmanOperator.advance`
        matches the fitted discrete propagator (Van Loan-aligned when
        controlled). Writes go through :meth:`set_dense_matrix`.

        Raises
        ------
        ValueError
            If ``koopman`` is not densely parameterized.
        TypeError
            If ``koopman`` is not a supported operator type.
        """
        self._validate_dense_parameterization(koopman)
        if isinstance(koopman, KoopmanOperator):
            control = self.control_matrix
            koopman.set_dense_matrix(
                self.discrete_matrix,
                control_matrix=control,
            )
            return

        if isinstance(koopman, ContinuousKoopmanOperator):
            generator, control = self._continuous_parameters()
            koopman.set_dense_matrix(generator, control_matrix=control)
            return

        msg = (
            "Online adaptation seed/write-back supports built-in "
            "KoopmanOperator and ContinuousKoopmanOperator only "
            f"(got {type(koopman).__name__}). Custom koopman= injections "
            "are Protocol-capable for train/predict/spectrum but not RLS."
        )
        raise TypeError(msg)

    def _as_rows(self, z: Tensor) -> Tensor:
        """Flatten latent tensors to ``(num_rows, latent_dim)``.

        Parameters
        ----------
        z : Tensor
            Latent tensor with shape ``(latent_dim,)`` or
            ``(num_nodes, latent_dim)``.

        Returns
        -------
        Tensor
            Two-dimensional latent tensor.

        Raises
        ------
        ValueError
            If ``z`` has an invalid shape.
        """
        if z.ndim == 1:
            if z.shape[0] != self.latent_dim:
                msg = (
                    f"Expected latent vector of length {self.latent_dim}, "
                    f"got {tuple(z.shape)}"
                )
                raise ValueError(msg)
            return z.unsqueeze(0)
        if z.ndim == 2:
            if z.shape[1] != self.latent_dim:
                msg = (
                    f"Expected trailing latent dimension {self.latent_dim}, "
                    f"got {tuple(z.shape)}"
                )
                raise ValueError(msg)
            return z
        msg = (
            "latent tensors must have shape (latent_dim,) or "
            f"(num_nodes, latent_dim), got {tuple(z.shape)}"
        )
        raise ValueError(msg)

    def _resolve_control_row(
        self,
        control: Tensor | None,
        *,
        batch_rows: int,
    ) -> Tensor | None:
        """Normalize control inputs to one row per latent sample.

        Parameters
        ----------
        control : Tensor or None
            Global or per-node control input.
        batch_rows : int
            Number of latent rows in the current update.

        Returns
        -------
        Tensor or None
            Control row or matrix aligned with latent rows.

        Raises
        ------
        ValueError
            If controls are missing or have invalid shape.
        """
        if self.control_dim == 0:
            if control is not None:
                msg = "control input provided to an uncontrolled adapter"
                raise ValueError(msg)
            return None
        if control is None:
            msg = "control input is required when control_dim > 0"
            raise ValueError(msg)

        if control.ndim == 1:
            if control.shape[0] != self.control_dim:
                msg = (
                    f"Expected global control shape ({self.control_dim},), "
                    f"got {tuple(control.shape)}"
                )
                raise ValueError(msg)
            return control

        if control.ndim == 2:
            if control.shape[1] != self.control_dim:
                msg = (
                    f"Expected per-node control shape (num_nodes, {self.control_dim}), "
                    f"got {tuple(control.shape)}"
                )
                raise ValueError(msg)
            if control.shape[0] == 1 and batch_rows > 1:
                return control.squeeze(0)
            if control.shape[0] != batch_rows:
                msg = (
                    f"Per-node control rows ({control.shape[0]}) must match "
                    f"latent rows ({batch_rows})"
                )
                raise ValueError(msg)
            return control

        msg = (
            "control input must have shape (control_dim,) or "
            f"(num_nodes, control_dim), got {tuple(control.shape)}"
        )
        raise ValueError(msg)

    def _build_regressor(
        self,
        z_row: Tensor,
        control_row: Tensor | None,
    ) -> Tensor:
        """Build an augmented regressor row ``[z, u]``.

        Parameters
        ----------
        z_row : Tensor
            Latent row with shape ``(latent_dim,)``.
        control_row : Tensor or None
            Control row with shape ``(control_dim,)``.

        Returns
        -------
        Tensor
            Augmented regressor with shape ``(latent_dim + control_dim,)``.
        """
        if self.control_dim == 0:
            return z_row
        assert control_row is not None
        return torch.cat([z_row, control_row], dim=0)

    def _rls_update(self, phi: Tensor, target: Tensor) -> None:
        """Apply one recursive least-squares update.

        Parameters
        ----------
        phi : Tensor
            Regressor vector with shape ``(latent_dim + control_dim,)``.
        target : Tensor
            Target latent vector with shape ``(latent_dim,)``.

        Returns
        -------
        None
        """
        phi = phi.to(dtype=self._weights.dtype)
        target = target.to(dtype=self._weights.dtype)
        denom = self.forgetting_factor + phi @ self._covariance @ phi
        gain = (self._covariance @ phi) / denom
        residual = target - phi @ self._weights
        self._weights = self._weights + gain.unsqueeze(1) * residual.unsqueeze(0)
        self._covariance = (
            self._covariance - gain.unsqueeze(1) * (phi @ self._covariance).unsqueeze(0)
        ) / self.forgetting_factor
