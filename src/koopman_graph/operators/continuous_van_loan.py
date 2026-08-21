"""Van Loan block-matrix helpers for continuous-time Koopman control.

Owned by the continuous capability surface: import via
``koopman_graph.operators.continuous`` (re-exports) or
``koopman_graph.operators``. This module holds cohesive matrix-log / Van Loan
factor construction;
:class:`~koopman_graph.operators.continuous.ContinuousKoopmanOperator`
remains the string-mode orchestrator.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor

from koopman_graph.spectrum_types import (
    DefectiveSpectrumError,
    _column_normalized_condition,
)

# Default absolute tolerance for Van Loan discrete↔generator round-trips in tests
# and documented adaptation fidelity checks (float32 matrix-exp / logm residual).
VAN_LOAN_WRITEBACK_ATOL = 1e-5

DefectiveLogMode = Literal["error", "schur"]

_DEFECTIVE_LOG_MESSAGE = (
    "Eigenvector matrix is singular (defective or numerically "
    "non-diagonalizable). matrix_log does not invert V; pass "
    "defective='schur' for a SciPy Schur-Pade logarithm (CPU), "
    "or use a Schur invariant subspace. This is not a claim that "
    "defectivity marks every critical regime."
)


def _resolve_log_rcond(matrix: Tensor, rcond: float | None) -> float:
    """Return a positive reciprocal-condition cutoff for the eigenbasis.

    Parameters
    ----------
    matrix : Tensor
        Square matrix whose logarithm is requested.
    rcond : float or None
        Caller cutoff, or ``None`` for ``n * eps`` of the real dtype.

    Returns
    -------
    float
        Positive cutoff. The eigendecomposition path treats
        :math:`\\kappa(V) > 1/\\texttt{rcond}` as defective.

    Raises
    ------
    ValueError
        If ``rcond`` is set but is not a finite value > 0.
    """
    if rcond is None:
        real_dtype = matrix.real.dtype if matrix.is_complex() else matrix.dtype
        return float(matrix.shape[0]) * float(torch.finfo(real_dtype).eps)
    value = float(rcond)
    if not math.isfinite(value) or value <= 0.0:
        msg = f"rcond must be a finite value > 0 when set, got {rcond}"
        raise ValueError(msg)
    return value


def _matrix_log_schur(matrix: Tensor) -> Tensor:
    """Principal matrix logarithm via SciPy Schur-Pade ``logm`` (CPU).

    Parameters
    ----------
    matrix : Tensor
        Square matrix with shape ``(d, d)``.

    Returns
    -------
    Tensor
        Matrix logarithm on ``matrix.device``. Real inputs return the
        real part, matching the eigendecomposition path.

    Raises
    ------
    ImportError
        If SciPy is not installed.
    """
    try:
        from scipy.linalg import logm
    except ImportError as exc:
        msg = (
            "matrix_log(..., defective='schur') requires SciPy. "
            "Install with: pip install scipy  or  pip install "
            "'koopman-graph[dev]' or 'koopman-graph[mpc]'"
        )
        raise ImportError(msg) from exc

    array = matrix.detach().cpu().numpy()
    logged = torch.as_tensor(logm(array), device=matrix.device)
    if matrix.is_complex():
        return logged.to(dtype=matrix.dtype)
    return logged.real.to(dtype=matrix.dtype)


def matrix_log(
    matrix: Tensor,
    *,
    defective: DefectiveLogMode = "error",
    rcond: float | None = None,
) -> Tensor:
    """Return the principal matrix logarithm of a square matrix.

    For diagonalizable ``M = V \\operatorname{diag}(\\lambda) V^{-1}``,

    .. math::

        \\log M = V \\operatorname{diag}(\\log \\lambda_i) V^{-1}

    with the principal branch of the scalar logarithm. Real inputs return
    ``result.real`` (callers should keep spectra away from the negative-real
    branch cut when a real logarithm is required).

    When the eigenbasis is numerically singular, the default
    ``defective="error"`` raises
    :class:`~koopman_graph.spectrum_types.DefectiveSpectrumError` rather
    than inverting :math:`V`. Opt-in ``defective="schur"`` uses SciPy
    ``logm`` (Al-Mohy–Higham inverse scaling and squaring / Padé on a
    Schur form; CPU NumPy round-trip). That path is **not** a claim that
    defectivity marks every critical regime.

    Limitations
    -----------
    - Eigenvalues on or near the negative real axis can make the principal
      log complex; discarding the imaginary part is then approximate.
    - ``defective="schur"`` requires SciPy and runs on CPU.
    - Used by Van Loan inversion and continuous RLS write-back; prefer
      well-conditioned generators with moderate ``Δt``.

    Parameters
    ----------
    matrix : Tensor
        Square matrix with shape ``(d, d)``.
    defective : {"error", "schur"}, optional
        ``"error"`` (default) raises when :math:`V` is singular beyond
        ``rcond``. ``"schur"`` delegates to SciPy ``logm``.
    rcond : float or None, optional
        Reciprocal-condition cutoff. ``None`` (default) uses
        ``n * eps`` of the real working dtype. Ignored when
        ``defective="schur"``.

    Returns
    -------
    Tensor
        Matrix logarithm. Real for real ``matrix`` when the imaginary part
        of the eigendecomposition path is negligible.

    Raises
    ------
    ValueError
        If ``defective`` is not ``"error"`` or ``"schur"``, or if
        ``rcond`` is set but is not a finite value > 0.
    DefectiveSpectrumError
        If ``defective="error"`` and the eigenbasis is singular or
        ill-conditioned beyond ``rcond``.
    ImportError
        If ``defective="schur"`` and SciPy is not installed.

    References
    ----------
    Al-Mohy, A. H. and Higham, N. J. (2012). Improved inverse scaling
    and squaring algorithms for the matrix logarithm. *SIAM Journal on
    Scientific Computing*, 34(4), C153–C169.
    https://doi.org/10.1137/110852553
    (``AlMohyHigham2012``; SciPy ``logm`` implements this family of
    methods. This helper does not re-implement the Padé coefficients.)
    """
    if defective == "schur":
        return _matrix_log_schur(matrix)
    if defective != "error":
        msg = f"defective must be 'error' or 'schur', got {defective!r}"
        raise ValueError(msg)

    eigenvalues, eigenvectors = torch.linalg.eig(matrix)
    rcond_value = _resolve_log_rcond(matrix, rcond)
    kappa = _column_normalized_condition(eigenvectors)
    if (not math.isfinite(kappa)) or kappa > (1.0 / rcond_value):
        raise DefectiveSpectrumError(_DEFECTIVE_LOG_MESSAGE)
    try:
        inverse = torch.linalg.inv(eigenvectors)
    except torch.linalg.LinAlgError as exc:
        raise DefectiveSpectrumError(_DEFECTIVE_LOG_MESSAGE) from exc
    log_eigenvalues = torch.log(eigenvalues)
    result = eigenvectors @ torch.diag(log_eigenvalues) @ inverse
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
        Discrete Koopman block ``K = exp(L · Δt)``.
    discrete_b : Tensor
        Discrete control block in the library row convention.
    delta_t : float or Tensor
        Integration interval.

    Returns
    -------
    tuple[Tensor, Tensor]
        Continuous generator ``L`` and control ``B``.

    Raises
    ------
    ValueError
        If ``delta_t`` is not positive.
    DefectiveSpectrumError
        If the Van Loan block is non-diagonalizable under the default
        ``matrix_log(..., defective="error")``.

    Notes
    -----
    Round-trip fidelity is typically within :data:`VAN_LOAN_WRITEBACK_ATOL`
    for moderate ``Δt`` when ``K(Δt)`` stays away from matrix-logarithm branch
    cuts. Large or highly oscillatory intervals can degrade recovery.
    ``matrix_log`` uses ``defective="error"`` by default: Jordan /
    singular eigenbases raise rather than silently inverting :math:`V`.
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
