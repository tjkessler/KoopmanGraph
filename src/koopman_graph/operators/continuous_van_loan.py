"""Van Loan block-matrix helpers for continuous-time Koopman control.

Owned by the continuous capability surface: import via
``koopman_graph.operators.continuous`` (re-exports) or
``koopman_graph.operators``. This module holds cohesive matrix-log / Van Loan
factor construction;
:class:`~koopman_graph.operators.continuous.ContinuousKoopmanOperator`
remains the string-mode orchestrator.
"""

from __future__ import annotations

import torch
from torch import Tensor

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
