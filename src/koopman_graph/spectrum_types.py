"""Neutral value types and spectrum assembly helpers.

Power-user module: importable as ``koopman_graph.spectrum_types``, documented
in architecture docs, and **not** re-exported in package ``__all__``.
:class:`KoopmanSpectrum` and :func:`compute_spectrum` are re-exported from
:mod:`koopman_graph.analysis` and (for the type / discrete helper) the package
root public surface. Operators, the model façade, and continuous-horizon
helpers import spectrum assembly from here so they never depend on
:mod:`koopman_graph.analysis`.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import torch
from torch import Tensor

#: Argument tolerance (radians) for the Nyquist-adjacent aliasing flag.
#: Negative-real eigenvalues have :math:`\arg=\pi` and therefore flag.
ALIASING_ARG_ATOL = 1e-3

#: Dimensionless 2-norm warning threshold for :math:`\kappa(V)` after
#: column 2-normalization (same definition as
#: :attr:`SpectralDiagnostics.eigenvector_condition`).
#: :meth:`KoopmanSpectrum.mode_amplitudes` emits a ``UserWarning`` when
#: :math:`\kappa(V)` exceeds this value, then still solves :math:`Va=z`.
#: Float64 numerical singularity is near :math:`1/u\sim 10^{16}`; this
#: threshold warns well before that.
CONDITION_WARN = 1e6

_DEFECTIVE_V_MESSAGE = (
    "Eigenvector matrix is singular (defective or numerically "
    "non-diagonalizable). Do not invert V; use a Schur invariant "
    "subspace or a well-conditioned subset of modes. "
    "mode_amplitudes does not compute a Schur form."
)


class DefectiveSpectrumError(torch.linalg.LinAlgError):
    """Raised when an eigenbasis cannot be inverted because V is singular.

    Used by :meth:`KoopmanSpectrum.mode_amplitudes` and
    :func:`~koopman_graph.operators.continuous_van_loan.matrix_log`.
    The stored eigenbasis is defective or numerically singular, so
    inverting :math:`V` is not well-posed. Use a Schur invariant
    subspace, a well-conditioned subset of modes, or
    ``matrix_log(..., defective="schur")``. This exception is **not**
    an automatic Schur fallback and does **not** mean defectivity marks
    every critical regime.

    Notes
    -----
    Wilkinson (1965) treats defective eigenproblems: a matrix need not
    admit a full eigenbasis. Finite-horizon transients of a non-normal
    but diagonalizable map are a separate issue; see
    :class:`SpectralDiagnostics`.

    References
    ----------
    Wilkinson, J. H. (1965). *The Algebraic Eigenvalue Problem*. Clarendon
    Press. (``Wilkinson1965``)
    """


@dataclass(frozen=True)
class SpectralDiagnostics:
    """Eigenvector conditioning, non-normality, and sampling diagnostics.

    These quantities flag when a stored eigenbasis is a poor amplitude
    coordinate and when non-normality can produce transient growth above
    :math:`\\rho(K)^t`. Discrete spectra also report the Nyquist frequency
    :math:`1/(2\\Delta t)` in **cycles per unit time**, a per-mode aliasing
    flag when :math:`\\pi-|\\arg\\lambda|\\le` ``ALIASING_ARG_ATOL``,
    and :math:`\\operatorname{sign}(\\operatorname{Re}\\lambda)` so that
    :math:`\\log|\\lambda|` is not read as a signed growth rate. They are
    **not** a finite-horizon bound on :math:`\\|K^k\\|`, not a replacement
    for
    :meth:`~koopman_graph.operators.KoopmanOperator.stability_certificate`,
    not a resolvent-norm / pseudospectrum certificate, and not a
    sampling-theorem identification result. Resolvent grids remain opt-in
    via :func:`~koopman_graph.analysis.resolvent_norm_grid`.

    :math:`\\kappa(V)` uses the matrix 2-norm of **column-normalized** right
    eigenvectors so a normal matrix scores near 1 rather than reflecting
    arbitrary ``eig`` column scaling. Per-mode
    :math:`\\kappa_i = 1/|w_i^H v_i|` uses unit-norm left and right
    eigenvectors. Departure is
    :math:`\\|KK^H - K^H K\\|_F`; the relative variant divides by
    :math:`\\|K\\|_F^2` (dimensionless; 0 when :math:`K=0`).

    Attributes
    ----------
    eigenvector_condition : float
        :math:`\\kappa(V) = \\|V\\|_2 \\|V^{-1}\\|_2` after column
        2-normalization. ``+inf`` when :math:`V` is numerically singular.
        Dimensionless.
    eigenvalue_sensitivities : Tensor
        Per-mode Wilkinson :math:`\\kappa_i` with shape ``(latent_dim,)``,
        aligned with stored eigenpairs. Non-negative; ``+inf`` when the
        corresponding pair cannot be normalized. Dimensionless.
    departure_from_normality : float
        Frobenius departure :math:`\\|KK^H - K^H K\\|_F` (same units as
        :math:`|K|^2`). Always finite for a finite matrix.
    departure_from_normality_relative : float
        ``departure_from_normality / ||K||_F^2`` when :math:`\\|K\\|_F > 0`,
        else ``0``. Dimensionless.
    nyquist_frequency : float or None
        Discrete Nyquist frequency :math:`1/(2\\Delta t)` in cycles per
        unit time (the same time unit as ``time_step``). ``None`` on
        generator spectra (the placeholder ``time_step=1.0`` on
        :class:`KoopmanSpectrum` is not a sampling interval).
    aliasing_warning : Tensor
        Boolean mask with shape ``(latent_dim,)``. ``True`` when
        :math:`\\pi-|\\arg\\lambda|\\le` ``ALIASING_ARG_ATOL``
        (radians). All ``False`` on generator spectra. Negative-real
        eigenvalues have :math:`\\arg=\\pi` and therefore flag.
    sign_of_eigenvalue : Tensor
        :math:`\\operatorname{sign}(\\operatorname{Re}\\lambda)\\in
        \\{-1,0,1\\}` with shape ``(latent_dim,)``. Dimensionless.
        Filled for both discrete and generator spectra.

    Notes
    -----
    Finite-horizon transients of non-normal operators can exceed the
    spectral-radius envelope even when every :math:`|\\lambda|\\le 1`.
    Discrete frequencies remain principal values in cycles per unit
    time; phase unwrapping does not recover aliases. :math:`\\log|\\lambda|`
    discards sign.

    References
    ----------
    Wilkinson, J. H. (1965). *The Algebraic Eigenvalue Problem*. Clarendon
    Press. (``Wilkinson1965``)

    Trefethen, L. N. and Embree, M. (2005). *Spectra and Pseudospectra*.
    Princeton University Press. https://doi.org/10.1515/9780691213101
    (``TrefethenEmbree2005``)

    Nyquist frequency :math:`1/(2\\Delta t)` is standard sampling theory
    (cycles per unit time).

    Zeng, Z., Yue, Z., Mauroy, A., Gonçalves, J., and Yuan, Y. (2022).
    A sampling theorem for exact identification of continuous-time
    nonlinear dynamical systems. *2022 IEEE 61st Conference on Decision
    and Control (CDC)*, 6686–6692.
    https://doi.org/10.1109/CDC51059.2022.9992482
    (``Zeng2022Sampling``; related Koopman sampling literature — this
    helper does not implement that identification theorem.)
    """

    eigenvector_condition: float
    eigenvalue_sensitivities: Tensor
    departure_from_normality: float
    departure_from_normality_relative: float
    nyquist_frequency: float | None
    aliasing_warning: Tensor
    sign_of_eigenvalue: Tensor

    def __post_init__(self) -> None:
        """Validate shapes and non-negativity of diagnostic fields.

        Raises
        ------
        ValueError
            If sensitivities, aliasing flags, or signs are not aligned
            1-D tensors, contain NaN, or if any scalar field is NaN or
            negative (``+inf`` is allowed for condition numbers).
        """
        sens = self.eigenvalue_sensitivities
        if sens.ndim != 1 or int(sens.numel()) == 0:
            msg = (
                "eigenvalue_sensitivities must be a nonempty 1-D tensor, "
                f"got shape {tuple(sens.shape)}"
            )
            raise ValueError(msg)
        if bool(torch.isnan(sens).any().item()):
            raise ValueError("eigenvalue_sensitivities must not contain NaN")
        if bool((sens < 0).any().item()):
            raise ValueError("eigenvalue_sensitivities must be non-negative")
        n = int(sens.numel())
        alias = self.aliasing_warning
        if alias.ndim != 1 or int(alias.numel()) != n:
            msg = (
                "aliasing_warning must have shape (latent_dim,), "
                f"got {tuple(alias.shape)} for latent_dim={n}"
            )
            raise ValueError(msg)
        if alias.dtype != torch.bool:
            raise ValueError("aliasing_warning must be a bool tensor")
        signs = self.sign_of_eigenvalue
        if signs.ndim != 1 or int(signs.numel()) != n:
            msg = (
                "sign_of_eigenvalue must have shape (latent_dim,), "
                f"got {tuple(signs.shape)} for latent_dim={n}"
            )
            raise ValueError(msg)
        if signs.is_complex():
            raise ValueError("sign_of_eigenvalue must be real")
        if bool(torch.isnan(signs).any().item()):
            raise ValueError("sign_of_eigenvalue must not contain NaN")
        allowed = (signs == -1) | (signs == 0) | (signs == 1)
        if not bool(allowed.all().item()):
            raise ValueError("sign_of_eigenvalue values must be in {-1, 0, 1}")
        nyquist = self.nyquist_frequency
        if nyquist is not None and (
            math.isnan(nyquist) or not math.isfinite(nyquist) or nyquist <= 0
        ):
            raise ValueError(
                f"nyquist_frequency must be a finite value > 0 when set, got {nyquist}"
            )
        for name, value in (
            ("eigenvector_condition", self.eigenvector_condition),
            ("departure_from_normality", self.departure_from_normality),
            (
                "departure_from_normality_relative",
                self.departure_from_normality_relative,
            ),
        ):
            if math.isnan(value):
                raise ValueError(f"{name} must not be NaN")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class KoopmanSpectrum:
    """Eigendecomposition and time scales of a Koopman operator or generator.

    Eigenpairs are sorted by descending eigenvalue magnitude. Frequencies are
    reported in cycles per unit time; multiply by ``2 * pi`` for angular
    frequency.

    Semantics depend on how the spectrum was produced:

    - :func:`~koopman_graph.spectrum_types.compute_spectrum` (discrete ``K``;
      also re-exported as :func:`~koopman_graph.analysis.compute_spectrum`):
      ``growth_rates = log(|lambda|) / time_step`` and
      ``frequencies = angle(lambda) / (2 * pi * time_step)``, with
      ``time_step`` equal to the discrete sampling interval.
    - :func:`~koopman_graph.spectrum_types.compute_generator_spectrum`
      (continuous ``L``; also re-exported from
      :mod:`koopman_graph.analysis`): ``growth_rates = Re(mu)`` and
      ``frequencies = Im(mu) / (2 * pi)``, with ``time_step`` set to ``1.0``
      as a placeholder (native continuous-time units).

    Optional ``residuals`` carry a posteriori data-driven residuals from
    :func:`~koopman_graph.analysis.spectral_residuals`. Optional
    ``diagnostics`` carry :class:`SpectralDiagnostics` (eigenvector
    condition, per-mode Wilkinson sensitivities, Frobenius departure
    from normality, discrete Nyquist frequency in cycles per unit time,
    aliasing flags, and :math:`\\operatorname{sign}(\\operatorname{Re}
    \\lambda)`). Populate either field with ``dataclasses.replace`` —
    this dataclass is frozen. Residuals and diagnostics are analysis
    state only and are **not** part of model checkpoints. Diagnostics
    are **not** a finite-horizon bound on :math:`\\|K^k\\|`. Discrete
    frequencies remain principal values; they do not recover aliases.
    :meth:`mode_amplitudes` warns when :math:`\\kappa(V)` exceeds
    ``CONDITION_WARN`` and raises :class:`DefectiveSpectrumError` when
    :math:`V` is singular.

    Attributes
    ----------
    eigenvalues : Tensor
        Complex eigenvalues with shape ``(latent_dim,)``.
    eigenvectors : Tensor
        Complex right eigenvectors stored as columns, with shape
        ``(latent_dim, latent_dim)``.
    magnitudes : Tensor
        Eigenvalue magnitudes with shape ``(latent_dim,)``.
    growth_rates : Tensor
        Continuous-time growth rates (see semantics above).
    frequencies : Tensor
        Signed frequencies in cycles per unit time (see semantics above).
    time_step : float
        Discrete sampling interval for discrete spectra; ``1.0`` for
        generator spectra.
    residuals : Tensor or None
        Optional non-negative real residuals with shape ``(latent_dim,)``.
        Default is ``None`` (unset).
    diagnostics : SpectralDiagnostics or None
        Optional conditioning / non-normality diagnostics. Default is
        ``None`` (unset; Kronecker assembly paths may omit them).
        :func:`compute_spectrum` and :func:`compute_generator_spectrum`
        populate this field. Generator spectra set
        ``nyquist_frequency`` to ``None`` and ``aliasing_warning`` to
        all ``False``.
    """

    eigenvalues: Tensor
    eigenvectors: Tensor
    magnitudes: Tensor
    growth_rates: Tensor
    frequencies: Tensor
    time_step: float
    residuals: Tensor | None = None
    diagnostics: SpectralDiagnostics | None = None

    def __post_init__(self) -> None:
        """Validate optional residual and diagnostic tensors.

        Raises
        ------
        ValueError
            If ``residuals`` is present but not a 1-D tensor of length
            ``latent_dim``, or contains negative / non-finite values; or if
            diagnostic tensors are not length ``latent_dim``.
        """
        latent_dim = int(self.eigenvalues.numel())
        if self.residuals is not None:
            if self.residuals.ndim != 1 or int(self.residuals.numel()) != latent_dim:
                msg = (
                    "residuals must have shape (latent_dim,), "
                    f"got {tuple(self.residuals.shape)} for latent_dim={latent_dim}"
                )
                raise ValueError(msg)
            if not bool(torch.isfinite(self.residuals).all().item()):
                msg = "residuals must be finite"
                raise ValueError(msg)
            if bool((self.residuals < 0).any().item()):
                msg = "residuals must be non-negative"
                raise ValueError(msg)
        if self.diagnostics is None:
            return
        diag = self.diagnostics
        for name, tensor in (
            ("eigenvalue_sensitivities", diag.eigenvalue_sensitivities),
            ("aliasing_warning", diag.aliasing_warning),
            ("sign_of_eigenvalue", diag.sign_of_eigenvalue),
        ):
            if tensor.ndim != 1 or int(tensor.numel()) != latent_dim:
                msg = (
                    f"diagnostics.{name} must have shape (latent_dim,), "
                    f"got {tuple(tensor.shape)} for latent_dim={latent_dim}"
                )
                raise ValueError(msg)

    def mode_amplitudes(self, latent_states: Tensor) -> Tensor:
        """Project latent states onto the Koopman eigenvector basis.

        For a latent row vector ``z``, the returned amplitudes ``a`` satisfy
        ``z.T = eigenvectors @ a`` (equivalently :math:`Va=z^{\\top}`).
        Any leading dimensions are preserved. When :math:`\\kappa(V)`
        exceeds ``CONDITION_WARN`` the solve still proceeds after a
        warning. Singular :math:`V` raises
        :class:`DefectiveSpectrumError` rather than a bare
        ``RuntimeError``.

        Parameters
        ----------
        latent_states : Tensor
            Latent states with shape ``(..., latent_dim)``.

        Returns
        -------
        Tensor
            Complex mode amplitudes with the same shape as ``latent_states``.

        Raises
        ------
        ValueError
            If the trailing latent dimension does not match the spectrum.
        DefectiveSpectrumError
            If the eigenvector matrix is singular or defective. The
            message hints at a Schur invariant subspace; this method
            does not compute one.

        Warnings
        --------
        UserWarning
            Emitted when :math:`\\kappa(V)` exceeds ``CONDITION_WARN``.
            The solve still proceeds. Not a finite-horizon bound on
            :math:`\\|K^k\\|`.
        """
        latent_dim = self.eigenvectors.shape[0]
        if latent_states.ndim == 0 or latent_states.shape[-1] != latent_dim:
            msg = (
                f"Expected trailing dimension {latent_dim}, "
                f"got shape {tuple(latent_states.shape)}"
            )
            raise ValueError(msg)

        vectors = self.eigenvectors.to(device=latent_states.device)
        if self.diagnostics is not None:
            kappa = self.diagnostics.eigenvector_condition
        else:
            kappa = _column_normalized_condition(vectors)
        if not math.isfinite(kappa):
            raise DefectiveSpectrumError(_DEFECTIVE_V_MESSAGE)
        if kappa > CONDITION_WARN:
            warnings.warn(
                f"Eigenvector condition kappa(V)={kappa:.3e} exceeds "
                f"CONDITION_WARN={CONDITION_WARN:.0e}. Mode amplitudes "
                "remain the solution of V a = z but are a poor "
                "coordinate; this is not a finite-horizon bound on "
                "||K^k||.",
                UserWarning,
                stacklevel=2,
            )
        states = latent_states.to(dtype=vectors.dtype)
        flat_states = states.reshape(-1, latent_dim)
        try:
            amplitudes = torch.linalg.solve(vectors, flat_states.T).T
        except torch.linalg.LinAlgError as exc:
            raise DefectiveSpectrumError(_DEFECTIVE_V_MESSAGE) from exc
        return amplitudes.reshape(latent_states.shape)


def _departure_from_normality(matrix: Tensor) -> tuple[float, float]:
    """Return Frobenius departure and the relative (scale-free) variant.

    Parameters
    ----------
    matrix : Tensor
        Square operator or generator.

    Returns
    -------
    tuple of float
        Absolute :math:`\\|MM^H-M^H M\\|_F` and that quantity over
        :math:`\\|M\\|_F^2` (0 when the Frobenius norm is 0).
    """
    commutator = matrix @ matrix.mH - matrix.mH @ matrix
    departure = float(torch.linalg.matrix_norm(commutator, ord="fro").real.item())
    frobenius = float(torch.linalg.matrix_norm(matrix, ord="fro").real.item())
    if frobenius == 0.0:
        return departure, 0.0
    return departure, departure / (frobenius * frobenius)


def _real_dtype_for(vectors: Tensor) -> torch.dtype:
    """Return a real dtype matching ``vectors`` precision.

    Parameters
    ----------
    vectors : Tensor
        Eigenvector matrix (real or complex).

    Returns
    -------
    torch.dtype
        ``float64`` for double / complex128 storage, else ``float32``.
    """
    if vectors.dtype in {torch.complex128, torch.float64}:
        return torch.float64
    return torch.float32


def _column_unit_eigenvectors(eigenvectors: Tensor) -> Tensor | None:
    """Return column 2-normalized eigenvectors, or None if invalid.

    Parameters
    ----------
    eigenvectors : Tensor
        Right eigenvectors as columns, shape ``(n, n)``.

    Returns
    -------
    Tensor or None
        Column-normalized copy, or ``None`` when a column norm is
        non-finite or non-positive.
    """
    col_norms = torch.linalg.vector_norm(eigenvectors, dim=0)
    if not bool(torch.isfinite(col_norms).all().item()) or bool(
        (col_norms <= 0).any().item()
    ):
        return None
    return eigenvectors / col_norms.to(dtype=eigenvectors.dtype)


def _column_normalized_condition(eigenvectors: Tensor) -> float:
    """Return :math:`\\kappa(V)` after column 2-normalization, or ``+inf``.

    Parameters
    ----------
    eigenvectors : Tensor
        Right eigenvectors as columns, shape ``(n, n)``.

    Returns
    -------
    float
        Dimensionless 2-norm condition number, or ``+inf`` when :math:`V`
        is numerically singular.
    """
    vectors_unit = _column_unit_eigenvectors(eigenvectors)
    if vectors_unit is None:
        return math.inf
    try:
        condition = torch.linalg.cond(vectors_unit)
    except (RuntimeError, torch.linalg.LinAlgError):
        return math.inf
    condition_value = float(condition.real.item())
    if not math.isfinite(condition_value):
        return math.inf
    return condition_value


def aliasing_warning_mask(eigenvalues: Tensor) -> Tensor:
    """Return True where :math:`|\\arg\\lambda|` is within atol of :math:`\\pi`.

    The comparison is
    :math:`\\pi-|\\arg\\lambda|\\le` ``ALIASING_ARG_ATOL``
    (radians). Negative-real values have :math:`\\arg=\\pi` and flag.
    This mask does not recover aliases and does not change
    :math:`\\log|\\lambda|` growth rates.

    Parameters
    ----------
    eigenvalues : Tensor
        Real or complex eigenvalues (any shape).

    Returns
    -------
    Tensor
        Boolean mask with the same shape as ``eigenvalues``.
    """
    values = (
        eigenvalues
        if eigenvalues.is_complex()
        else eigenvalues.to(dtype=torch.complex128)
    )
    return (math.pi - torch.angle(values).abs()) <= ALIASING_ARG_ATOL


def _nyquist_aliasing_fields(
    eigenvalues: Tensor,
    *,
    time_step: float | None,
) -> tuple[float | None, Tensor, Tensor]:
    """Nyquist frequency, aliasing mask, and sign of Re(lambda).

    Reports :math:`1/(2\\Delta t)` in cycles per unit time,
    :math:`\\operatorname{sign}(\\operatorname{Re}\\lambda)`, and the
    Nyquist-adjacent aliasing mask.

    Parameters
    ----------
    eigenvalues : Tensor
        Eigenvalues aligned with stored eigenpairs, shape ``(n,)``.
    time_step : float or None
        Discrete sampling interval, or ``None`` for a generator spectrum.

    Returns
    -------
    nyquist_frequency : float or None
        :math:`1/(2\\Delta t)` in cycles per unit time, or ``None``.
    aliasing_warning : Tensor
        Boolean mask, shape ``(n,)``. All ``False`` when ``time_step``
        is ``None``.
    sign_of_eigenvalue : Tensor
        :math:`\\operatorname{sign}(\\operatorname{Re}\\lambda)` on the
        eigenvalue device.

    Raises
    ------
    ValueError
        If ``time_step`` is set but is not a finite value > 0.
    """
    real_part = eigenvalues.real if eigenvalues.is_complex() else eigenvalues
    signs = torch.sign(real_part)
    n = int(eigenvalues.numel())
    if time_step is None:
        alias = torch.zeros(n, dtype=torch.bool, device=eigenvalues.device)
        return None, alias, signs
    dt = float(time_step)
    if not math.isfinite(dt) or dt <= 0.0:
        msg = f"time_step must be a finite value > 0 when set, got {time_step}"
        raise ValueError(msg)
    nyquist = 1.0 / (2.0 * dt)
    return nyquist, aliasing_warning_mask(eigenvalues), signs


def compute_spectral_diagnostics(
    matrix: Tensor,
    eigenvectors: Tensor,
    *,
    eigenvalues: Tensor,
    time_step: float | None = None,
) -> SpectralDiagnostics:
    """Compute conditioning, non-normality, and sampling diagnostics.

    ``matrix`` is the discrete operator or generator whose eigenpairs are
    stored in ``eigenvectors`` (columns, already in spectrum order).
    Pass the matching ``eigenvalues`` (same order). For discrete maps,
    ``time_step`` is the sampling interval :math:`\\Delta t` and
    ``nyquist_frequency`` is :math:`1/(2\\Delta t)` in cycles per unit
    time. For generator spectra, pass ``time_step=None`` so Nyquist is
    unset and aliasing flags stay ``False``. Does not assemble a
    resolvent grid and does not implement a sampling-theorem
    identification test.

    Parameters
    ----------
    matrix : Tensor
        Square operator or generator with shape ``(n, n)``.
    eigenvectors : Tensor
        Right eigenvectors as columns, shape ``(n, n)``.
    eigenvalues : Tensor
        Eigenvalues aligned with ``eigenvectors`` columns, shape
        ``(n,)``.
    time_step : float or None, optional
        Discrete sampling interval. ``None`` (default) marks a generator
        spectrum: ``nyquist_frequency`` is ``None`` and
        ``aliasing_warning`` is all ``False``. ``sign_of_eigenvalue`` is
        still filled.

    Returns
    -------
    SpectralDiagnostics
        Condition numbers, Frobenius departure, Nyquist frequency,
        aliasing flags, and eigenvalue signs. ``+inf`` condition fields
        when :math:`V` is numerically singular; departure and sampling
        fields remain finite.

    Raises
    ------
    ValueError
        If ``matrix`` or ``eigenvectors`` is not a nonempty square pair of
        matching size, if ``eigenvalues`` is not length ``n``, or if
        ``time_step`` is set but is not a finite value > 0.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        msg = f"matrix must be square, got shape {tuple(matrix.shape)}"
        raise ValueError(msg)
    if eigenvectors.ndim != 2 or eigenvectors.shape[0] != eigenvectors.shape[1]:
        msg = f"eigenvectors must be square, got shape {tuple(eigenvectors.shape)}"
        raise ValueError(msg)
    if matrix.shape[0] != eigenvectors.shape[0]:
        msg = (
            "matrix and eigenvectors must share dimension, "
            f"got {tuple(matrix.shape)} vs {tuple(eigenvectors.shape)}"
        )
        raise ValueError(msg)
    if matrix.shape[0] == 0:
        raise ValueError("matrix must be non-empty")
    n = int(eigenvectors.shape[0])
    if eigenvalues.ndim != 1 or int(eigenvalues.numel()) != n:
        msg = (
            "eigenvalues must have shape (n,), "
            f"got {tuple(eigenvalues.shape)} for n={n}"
        )
        raise ValueError(msg)

    departure, relative = _departure_from_normality(matrix)
    nyquist, aliasing, signs = _nyquist_aliasing_fields(
        eigenvalues,
        time_step=time_step,
    )
    real_dtype = _real_dtype_for(eigenvectors)
    inf_sens = torch.full(
        (n,),
        math.inf,
        dtype=real_dtype,
        device=eigenvectors.device,
    )
    inf_diag = SpectralDiagnostics(
        eigenvector_condition=math.inf,
        eigenvalue_sensitivities=inf_sens,
        departure_from_normality=departure,
        departure_from_normality_relative=relative,
        nyquist_frequency=nyquist,
        aliasing_warning=aliasing,
        sign_of_eigenvalue=signs,
    )

    vectors_unit = _column_unit_eigenvectors(eigenvectors)
    if vectors_unit is None:
        return inf_diag
    try:
        condition = torch.linalg.cond(vectors_unit)
        inverse = torch.linalg.inv(vectors_unit)
    except (RuntimeError, torch.linalg.LinAlgError):
        return inf_diag
    condition_value = float(condition.real.item())
    if not math.isfinite(condition_value):
        return inf_diag
    sensitivities = torch.linalg.vector_norm(inverse, dim=1).real.to(dtype=real_dtype)
    return SpectralDiagnostics(
        eigenvector_condition=condition_value,
        eigenvalue_sensitivities=sensitivities,
        departure_from_normality=departure,
        departure_from_normality_relative=relative,
        nyquist_frequency=nyquist,
        aliasing_warning=aliasing,
        sign_of_eigenvalue=signs,
    )


def compute_spectrum(operator: Tensor, time_step: float) -> KoopmanSpectrum:
    """Compute the sorted spectrum and continuous-time mode characteristics.

    Neutral-leaf discrete spectrum assembly used by operators (for example
    :meth:`~koopman_graph.operators.graph.GraphKoopmanOperator.spectrum`) and
    re-exported from :mod:`koopman_graph.analysis` for the public API.

    Parameters
    ----------
    operator : Tensor
        Square discrete-time Koopman matrix with shape
        ``(latent_dim, latent_dim)``.
    time_step : float
        Positive physical duration represented by one operator step.

    Returns
    -------
    KoopmanSpectrum
        Eigenpairs sorted by descending magnitude, plus growth rates and
        frequencies converted using ``time_step``. Frequencies are
        principal values in **cycles per unit time**. ``diagnostics`` is
        always populated, including Nyquist frequency
        :math:`1/(2\\Delta t)`, aliasing flags, and
        :math:`\\operatorname{sign}(\\operatorname{Re}\\lambda)`. Emits
        one ``UserWarning`` when any mode is Nyquist-adjacent.

    Raises
    ------
    ValueError
        If ``operator`` is not a non-empty square matrix or ``time_step`` is
        not positive.
    TypeError
        If ``operator`` is not floating-point or complex.
    """
    if operator.ndim != 2 or operator.shape[0] != operator.shape[1]:
        msg = f"operator must be a square matrix, got shape {tuple(operator.shape)}"
        raise ValueError(msg)
    if operator.shape[0] == 0:
        raise ValueError("operator must be non-empty")
    if time_step <= 0:
        msg = f"time_step must be positive, got {time_step}"
        raise ValueError(msg)
    if not (operator.is_floating_point() or operator.is_complex()):
        msg = f"operator must be floating-point or complex, got {operator.dtype}"
        raise TypeError(msg)

    eigenvalues, eigenvectors = torch.linalg.eig(operator)
    magnitudes = eigenvalues.abs()
    order = torch.argsort(magnitudes, descending=True)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    magnitudes = magnitudes[order]

    growth_rates = torch.log(magnitudes) / time_step
    frequencies = torch.angle(eigenvalues) / (2 * torch.pi * time_step)
    diagnostics = compute_spectral_diagnostics(
        operator,
        eigenvectors,
        eigenvalues=eigenvalues,
        time_step=float(time_step),
    )
    if bool(diagnostics.aliasing_warning.any().item()):
        warnings.warn(
            "One or more discrete-time eigenvalues have argument within "
            f"{ALIASING_ARG_ATOL} rad of pi (Nyquist phase). Reported "
            "frequencies are principal values in cycles per unit time; "
            "phase unwrapping does not recover aliases. log(|lambda|) "
            "discards sign(Re lambda).",
            UserWarning,
            stacklevel=2,
        )
    return KoopmanSpectrum(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        magnitudes=magnitudes,
        growth_rates=growth_rates,
        frequencies=frequencies,
        time_step=float(time_step),
        diagnostics=diagnostics,
    )


def compute_generator_spectrum(generator: Tensor) -> KoopmanSpectrum:
    """Compute the sorted spectrum of a continuous-time Koopman generator.

    Growth rates are the real parts of the eigenvalues; frequencies are the
    imaginary parts scaled to cycles per unit time.

    Neutral-leaf generator spectrum assembly used by continuous networked
    operators and re-exported from :mod:`koopman_graph.analysis`.

    Parameters
    ----------
    generator : Tensor
        Square generator matrix with shape ``(latent_dim, latent_dim)``.

    Returns
    -------
    KoopmanSpectrum
        Eigenpairs sorted by descending magnitude with native continuous-time
        growth rates and frequencies in **cycles per unit time**.
        ``diagnostics`` is always populated.
        ``nyquist_frequency`` is ``None`` and ``aliasing_warning`` is
        all ``False`` (the placeholder ``time_step=1.0`` is not a
        sampling interval). ``sign_of_eigenvalue`` is still filled.

    Raises
    ------
    ValueError
        If ``generator`` is not a non-empty square matrix.
    TypeError
        If ``generator`` is not floating-point or complex.
    """
    if generator.ndim != 2 or generator.shape[0] != generator.shape[1]:
        msg = f"generator must be a square matrix, got shape {tuple(generator.shape)}"
        raise ValueError(msg)
    if generator.shape[0] == 0:
        raise ValueError("generator must be non-empty")
    if not (generator.is_floating_point() or generator.is_complex()):
        msg = f"generator must be floating-point or complex, got {generator.dtype}"
        raise TypeError(msg)

    eigenvalues, eigenvectors = torch.linalg.eig(generator)
    magnitudes = eigenvalues.abs()
    order = torch.argsort(magnitudes, descending=True)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    magnitudes = magnitudes[order]

    growth_rates = eigenvalues.real
    frequencies = eigenvalues.imag / (2 * torch.pi)
    return KoopmanSpectrum(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        magnitudes=magnitudes,
        growth_rates=growth_rates,
        frequencies=frequencies,
        time_step=1.0,
        diagnostics=compute_spectral_diagnostics(
            generator,
            eigenvectors,
            eigenvalues=eigenvalues,
            time_step=None,
        ),
    )


def discrete_spectrum_at_delta_t(
    generator: Tensor,
    delta_t: float,
) -> KoopmanSpectrum:
    """Compute the spectrum of ``exp(L · Δt)`` for a generator ``L``.

    Neutral-leaf continuous-to-discrete spectrum helper used by the model
    façade and re-exported from :mod:`koopman_graph.analysis`.

    Parameters
    ----------
    generator : Tensor
        Continuous-time generator matrix.
    delta_t : float
        Integration interval.

    Returns
    -------
    KoopmanSpectrum
        Discrete-time spectrum at horizon ``delta_t``.

    Raises
    ------
    ValueError
        If ``delta_t`` is not positive.
    """
    if delta_t <= 0:
        msg = f"delta_t must be positive, got {delta_t}"
        raise ValueError(msg)
    transition = torch.linalg.matrix_exp(generator * delta_t)
    return compute_spectrum(transition, delta_t)
