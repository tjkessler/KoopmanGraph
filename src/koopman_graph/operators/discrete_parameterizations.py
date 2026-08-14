"""Structural / dense matrix assembly and reset for discrete operators.

Shallow sibling of :mod:`koopman_graph.operators.discrete`. Holds cohesive
ODO / Schur / dissipative / Lyapunov / dense mechanics used by
:class:`~koopman_graph.operators.discrete.KoopmanOperator` without introducing
a parameterization class hierarchy. Prefer importing via ``discrete``
re-exports when writing against the discrete capability surface.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from koopman_graph.operators.contract import (
    DISSIPATIVE_MIN_EIGENVALUE,
    InitMode,
    bounded_diagonal,
    cayley_orthogonal,
    strict_diagonal_values,
    strict_spectral_bound,
)


def identity_diag_raw(max_spectral_radius: float) -> float:
    """Return raw diagonal parameters for a near-identity ODO operator.

    Parameters
    ----------
    max_spectral_radius : float
        Soft spectral bound used by the ODO diagonal map.

    Returns
    -------
    float
        Unconstrained diagonal parameter mapped near unit eigenvalues.
    """
    target = min(1.0, max_spectral_radius) * (1.0 - 1e-6)
    ratio = target / max_spectral_radius
    return float(torch.atanh(torch.tensor(ratio)).item())


def identity_strict_diag_raw(max_spectral_radius: float) -> float:
    """Return raw diagonal parameters for a near-identity strict-stable mode.

    Parameters
    ----------
    max_spectral_radius : float
        Target spectral bound (must be ``<= 1`` for structural modes).

    Returns
    -------
    float
        Unconstrained diagonal parameter mapped near the strict bound.
    """
    bound = strict_spectral_bound(max_spectral_radius)
    target = bound * (1.0 - 1e-6)
    ratio = target / bound
    return float(torch.atanh(torch.tensor(ratio)).item())


def odo_orthogonal_factors(
    cayley_O1: Tensor,
    cayley_O2: Tensor,
) -> tuple[Tensor, Tensor]:
    """Build orthogonal factors for the ODO parameterization.

    Parameters
    ----------
    cayley_O1, cayley_O2 : Tensor
        Cayley factors for the two orthogonal matrices.

    Returns
    -------
    tuple of Tensor
        Orthogonal matrices ``(O_1, O_2)``.
    """
    return cayley_orthogonal(cayley_O1), cayley_orthogonal(cayley_O2)


def odo_diagonal(diag_raw: Tensor, max_spectral_radius: float) -> Tensor:
    """Build the bounded diagonal factor for the ODO parameterization.

    Parameters
    ----------
    diag_raw : Tensor
        Unconstrained diagonal parameters.
    max_spectral_radius : float
        Soft spectral bound for diagonal entries.

    Returns
    -------
    Tensor
        Diagonal matrix with bounded eigenvalues.
    """
    return bounded_diagonal(diag_raw, max_spectral_radius)


def assemble_odo_matrix(
    cayley_O1: Tensor,
    cayley_O2: Tensor,
    diag_raw: Tensor,
    max_spectral_radius: float,
) -> Tensor:
    """Assemble ``K = O_1 D O_2^T`` from ODO factors.

    Parameters
    ----------
    cayley_O1, cayley_O2 : Tensor
        Cayley factors for the two orthogonal matrices.
    diag_raw : Tensor
        Unconstrained diagonal parameters.
    max_spectral_radius : float
        Soft spectral bound for diagonal entries.

    Returns
    -------
    Tensor
        Assembled operator matrix.
    """
    o1, o2 = odo_orthogonal_factors(cayley_O1, cayley_O2)
    diagonal = odo_diagonal(diag_raw, max_spectral_radius)
    return o1 @ diagonal @ o2.T


def schur_triangular(
    schur_diag_raw: Tensor,
    schur_off_raw: Tensor,
    max_spectral_radius: float,
) -> Tensor:
    """Build the upper-triangular Schur factor ``T``.

    Parameters
    ----------
    schur_diag_raw : Tensor
        Unconstrained diagonal parameters of ``T``.
    schur_off_raw : Tensor
        Strictly upper-triangular entries of ``T`` (lower triangle ignored).
    max_spectral_radius : float
        Strict spectral bound for diagonal entries.

    Returns
    -------
    Tensor
        Upper-triangular Schur factor with bounded diagonal.
    """
    diag_vals = strict_diagonal_values(schur_diag_raw, max_spectral_radius)
    triangular = torch.triu(schur_off_raw, diagonal=1)
    return triangular + torch.diag(diag_vals)


def assemble_schur_matrix(
    cayley_Q: Tensor,
    schur_diag_raw: Tensor,
    schur_off_raw: Tensor,
    max_spectral_radius: float,
) -> Tensor:
    """Assemble ``K = Q T Q^T`` from Schur factors.

    Parameters
    ----------
    cayley_Q : Tensor
        Cayley factor for the orthogonal matrix ``Q``.
    schur_diag_raw : Tensor
        Unconstrained diagonal parameters of ``T``.
    schur_off_raw : Tensor
        Strictly upper-triangular entries of ``T``.
    max_spectral_radius : float
        Strict spectral bound for diagonal entries.

    Returns
    -------
    Tensor
        Assembled Schur-form operator matrix.
    """
    q = cayley_orthogonal(cayley_Q)
    return (
        q @ schur_triangular(schur_diag_raw, schur_off_raw, max_spectral_radius) @ q.T
    )


def dissipative_factor(dissipative_L: Tensor, latent_dim: int) -> Tensor:
    """Build the lower-triangular factor ``L`` for the generator ``S``.

    Parameters
    ----------
    dissipative_L : Tensor
        Lower-triangular factor parameters.
    latent_dim : int
        Latent dimension (square size of ``L``).

    Returns
    -------
    Tensor
        Lower-triangular factor with positive diagonal entries.
    """
    lower = torch.tril(dissipative_L)
    diag_index = torch.arange(latent_dim, device=lower.device)
    lower[diag_index, diag_index] = (
        torch.nn.functional.softplus(lower[diag_index, diag_index])
        + DISSIPATIVE_MIN_EIGENVALUE
    )
    return lower


def dissipative_generator(dissipative_L: Tensor, latent_dim: int) -> Tensor:
    """Build the SPD generator ``S = L L^T + \\varepsilon I``.

    Parameters
    ----------
    dissipative_L : Tensor
        Lower-triangular factor parameters.
    latent_dim : int
        Latent dimension (square size of ``S``).

    Returns
    -------
    Tensor
        Symmetric positive-definite generator matrix.
    """
    factor = dissipative_factor(dissipative_L, latent_dim)
    identity = torch.eye(latent_dim, device=factor.device, dtype=factor.dtype)
    return factor @ factor.T + DISSIPATIVE_MIN_EIGENVALUE * identity


def assemble_dissipative_matrix(dissipative_L: Tensor, latent_dim: int) -> Tensor:
    """Assemble ``K = exp(-S)`` from the dissipative generator.

    Parameters
    ----------
    dissipative_L : Tensor
        Lower-triangular factor parameters.
    latent_dim : int
        Latent dimension (square size of ``K``).

    Returns
    -------
    Tensor
        Symmetric contractive operator matrix.
    """
    generator = dissipative_generator(dissipative_L, latent_dim)
    return torch.linalg.matrix_exp(-generator)


def lyapunov_diagonal(lyap_diag_raw: Tensor, max_spectral_radius: float) -> Tensor:
    """Return strict stable eigenvalues for the Lyapunov parameterization.

    Parameters
    ----------
    lyap_diag_raw : Tensor
        Unconstrained diagonal parameters.
    max_spectral_radius : float
        Strict spectral bound (``<= 1``).

    Returns
    -------
    Tensor
        Diagonal eigenvalues strictly inside
        ``(-max_spectral_radius, max_spectral_radius)`` with
        ``max_spectral_radius <= 1``, so they lie inside the unit disk.
    """
    return strict_diagonal_values(lyap_diag_raw, max_spectral_radius)


def lyapunov_certificate_matrix(cayley_Q: Tensor, lyap_p_raw: Tensor) -> Tensor:
    """Return the Lyapunov certificate matrix ``P = Q diag(p) Q^T``.

    Parameters
    ----------
    cayley_Q : Tensor
        Cayley factor for the orthogonal matrix ``Q``.
    lyap_p_raw : Tensor
        Unconstrained positive-diagonal parameters of ``P``.

    Returns
    -------
    Tensor
        Symmetric positive-definite Lyapunov matrix.
    """
    q = cayley_orthogonal(cayley_Q)
    p = torch.nn.functional.softplus(lyap_p_raw) + 1e-6
    return q @ torch.diag(p) @ q.T


def assemble_lyapunov_matrix(
    cayley_Q: Tensor,
    lyap_diag_raw: Tensor,
    max_spectral_radius: float,
) -> Tensor:
    """Assemble ``K = Q diag(d) Q^T`` with Lyapunov certificate ``P``.

    Parameters
    ----------
    cayley_Q : Tensor
        Cayley factor for the orthogonal matrix ``Q``.
    lyap_diag_raw : Tensor
        Unconstrained diagonal parameters.
    max_spectral_radius : float
        Strict spectral bound for diagonal entries.

    Returns
    -------
    Tensor
        Lyapunov-certified symmetric operator matrix.
    """
    q = cayley_orthogonal(cayley_Q)
    return q @ torch.diag(lyapunov_diagonal(lyap_diag_raw, max_spectral_radius)) @ q.T


def reset_dense_matrix(
    matrix: Tensor,
    *,
    init_mode: InitMode,
    init_scale: float,
) -> None:
    """Reinitialize a dense learnable matrix ``K``.

    Parameters
    ----------
    matrix : Tensor
        Dense parameter tensor mutated in place.
    init_mode : InitMode
        Initialization strategy.
    init_scale : float
        Noise scale for ``identity_noise``.
    """
    if init_mode == "identity":
        nn.init.eye_(matrix)
    elif init_mode == "identity_noise":
        nn.init.eye_(matrix)
        with torch.no_grad():
            matrix.add_(torch.randn_like(matrix) * init_scale)
    elif init_mode == "xavier":
        nn.init.xavier_uniform_(matrix)
    else:
        msg = f"Unknown init_mode: {init_mode!r}"
        raise ValueError(msg)


def reset_odo_matrix(
    cayley_O1: Tensor,
    cayley_O2: Tensor,
    diag_raw: Tensor,
    *,
    init_mode: InitMode,
    init_scale: float,
    max_spectral_radius: float,
) -> None:
    """Reinitialize Cayley and diagonal ODO parameters in place.

    Parameters
    ----------
    cayley_O1 : Tensor
        First Cayley skew factor.
    cayley_O2 : Tensor
        Second Cayley skew factor.
    diag_raw : Tensor
        Unconstrained diagonal raw parameters.
    init_mode : InitMode
        Initialization strategy.
    init_scale : float
        Noise scale for ``identity_noise``.
    max_spectral_radius : float
        Spectral-radius bound used for identity diagonal initialization.
    """
    nn.init.zeros_(cayley_O1)
    nn.init.zeros_(cayley_O2)
    if init_mode == "identity":
        nn.init.constant_(diag_raw, identity_diag_raw(max_spectral_radius))
    elif init_mode == "identity_noise":
        nn.init.constant_(diag_raw, identity_diag_raw(max_spectral_radius))
        with torch.no_grad():
            noise = torch.randn_like(diag_raw) * init_scale
            current = torch.tanh(diag_raw) * max_spectral_radius
            updated = (current + noise).clamp(
                min=-max_spectral_radius + 1e-6,
                max=max_spectral_radius - 1e-6,
            )
            diag_raw.copy_(torch.atanh(updated / max_spectral_radius))
    elif init_mode == "xavier":
        nn.init.xavier_uniform_(cayley_O1)
        nn.init.xavier_uniform_(cayley_O2)
        nn.init.uniform_(diag_raw, -0.5, 0.5)
    else:
        msg = f"Unknown init_mode: {init_mode!r}"
        raise ValueError(msg)


def reset_strict_diagonal(
    diag_param: Tensor,
    *,
    init_mode: InitMode,
    init_scale: float,
    max_spectral_radius: float,
    cayley: Tensor | None = None,
    off_param: Tensor | None = None,
) -> None:
    """Initialize strict-stable Schur/Lyapunov diagonal and optional factors.

    Parameters
    ----------
    diag_param : Tensor
        Unconstrained strict-stable diagonal raw parameters.
    init_mode : InitMode
        Initialization strategy.
    init_scale : float
        Noise scale for ``identity_noise``.
    max_spectral_radius : float
        Spectral-radius bound for the strict diagonal map.
    cayley : Tensor or None
        Optional Cayley skew factor to reset.
    off_param : Tensor or None
        Optional upper-triangular off-diagonal factor to reset.
    """
    if cayley is not None:
        nn.init.zeros_(cayley)
    if off_param is not None:
        nn.init.zeros_(off_param)
    if init_mode == "identity":
        nn.init.constant_(diag_param, identity_strict_diag_raw(max_spectral_radius))
    elif init_mode == "identity_noise":
        nn.init.constant_(diag_param, identity_strict_diag_raw(max_spectral_radius))
        with torch.no_grad():
            bound = strict_spectral_bound(max_spectral_radius)
            noise = torch.randn_like(diag_param) * init_scale
            current = torch.tanh(diag_param) * bound
            updated = (current + noise).clamp(
                min=-bound + 1e-6,
                max=bound - 1e-6,
            )
            diag_param.copy_(torch.atanh(updated / bound))
    elif init_mode == "xavier":
        if cayley is not None:
            nn.init.xavier_uniform_(cayley)
        if off_param is not None:
            nn.init.xavier_uniform_(off_param)
            off_param.data.copy_(torch.triu(off_param.data, diagonal=1))
        nn.init.uniform_(diag_param, -0.5, 0.5)
    else:
        msg = f"Unknown init_mode: {init_mode!r}"
        raise ValueError(msg)


def reset_schur_matrix(
    cayley_Q: Tensor,
    schur_diag_raw: Tensor,
    schur_off_raw: Tensor,
    *,
    init_mode: InitMode,
    init_scale: float,
    max_spectral_radius: float,
) -> None:
    """Reinitialize Schur-form parameters in place.

    Parameters
    ----------
    cayley_Q : Tensor
        Cayley skew factor for the orthogonal Schur basis.
    schur_diag_raw : Tensor
        Unconstrained Schur diagonal raw parameters.
    schur_off_raw : Tensor
        Unconstrained upper-triangular off-diagonal parameters.
    init_mode : InitMode
        Initialization strategy.
    init_scale : float
        Noise scale for ``identity_noise``.
    max_spectral_radius : float
        Spectral-radius bound used for identity diagonal initialization.
    """
    reset_strict_diagonal(
        schur_diag_raw,
        init_mode=init_mode,
        init_scale=init_scale,
        max_spectral_radius=max_spectral_radius,
        cayley=cayley_Q,
        off_param=schur_off_raw,
    )


def reset_dissipative_matrix(
    dissipative_L: Tensor,
    *,
    init_mode: InitMode,
    init_scale: float,
) -> None:
    """Reinitialize dissipative generator parameters in place.

    Parameters
    ----------
    dissipative_L : Tensor
        Lower-triangular factor for the SPD generator.
    init_mode : InitMode
        Initialization strategy.
    init_scale : float
        Noise scale for ``identity_noise``.
    """
    nn.init.zeros_(dissipative_L)
    if init_mode == "identity_noise":
        with torch.no_grad():
            dissipative_L.add_(torch.randn_like(dissipative_L) * init_scale)
    elif init_mode == "xavier":
        nn.init.xavier_uniform_(dissipative_L)
        dissipative_L.data.copy_(torch.tril(dissipative_L.data))
    elif init_mode != "identity":
        msg = f"Unknown init_mode: {init_mode!r}"
        raise ValueError(msg)


def reset_lyapunov_matrix(
    cayley_Q: Tensor,
    lyap_diag_raw: Tensor,
    lyap_p_raw: Tensor,
    *,
    init_mode: InitMode,
    init_scale: float,
    max_spectral_radius: float,
) -> None:
    """Reinitialize Lyapunov-certified symmetric parameters in place.

    Parameters
    ----------
    cayley_Q : Tensor
        Cayley skew factor for the Lyapunov eigenbasis.
    lyap_diag_raw : Tensor
        Unconstrained Lyapunov eigenvalue raw parameters.
    lyap_p_raw : Tensor
        Unconstrained Lyapunov certificate diagonal raw parameters.
    init_mode : InitMode
        Initialization strategy.
    init_scale : float
        Noise scale for ``identity_noise``.
    max_spectral_radius : float
        Spectral-radius bound used for identity diagonal initialization.
    """
    reset_strict_diagonal(
        lyap_diag_raw,
        init_mode=init_mode,
        init_scale=init_scale,
        max_spectral_radius=max_spectral_radius,
        cayley=cayley_Q,
    )
    nn.init.zeros_(lyap_p_raw)


_SINKHORN_ITERS = 8
_CAYLEY_DAMP = 1e-6


def assemble_row_stochastic_matrix(raw: Tensor) -> Tensor:
    """Assemble a row-stochastic map via softmax on the last axis.

    Under the library advance ``z ← z Kᵀ``, row-stochastic ``K`` conserves
    the sum of latent coordinates. This does **not** certify conservation of
    decoded physical mass after a nonlinear decoder.

    Parameters
    ----------
    raw : Tensor
        Unconstrained logits with shape ``(d, d)``.

    Returns
    -------
    Tensor
        Row-stochastic matrix with the same shape.
    """
    return torch.softmax(raw, dim=-1)


def assemble_doubly_stochastic_matrix(
    raw: Tensor,
    *,
    iterations: int = _SINKHORN_ITERS,
) -> Tensor:
    """Assemble an approximately doubly-stochastic map by Sinkhorn iteration.

    Parameters
    ----------
    raw : Tensor
        Unconstrained logits with shape ``(d, d)``.
    iterations : int, optional
        Alternating row/column normalizations. Default is 8.

    Returns
    -------
    Tensor
        Non-negative matrix with approximately unit row and column sums.

    Notes
    -----
    Finite Sinkhorn scaling after Cuturi (2013; ``Cuturi2013Sinkhorn``).
    Iteration count is a teaching default, not an exact doubly-stochastic
    projection.
    """
    matrix = torch.exp(raw)
    for _ in range(int(iterations)):
        matrix = matrix / matrix.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        matrix = matrix / matrix.sum(dim=-2, keepdim=True).clamp_min(1e-12)
    return matrix


def _symmetric_from_raw(raw: Tensor) -> Tensor:
    """Build a symmetric generator from an unconstrained square factor.

    Parameters
    ----------
    raw : Tensor
        Unconstrained square tensor.

    Returns
    -------
    Tensor
        Symmetric part ``(raw + rawᵀ) / 2``.
    """
    return 0.5 * (raw + raw.transpose(-1, -2))


def assemble_symplectic_matrix(raw: Tensor) -> Tensor:
    """Assemble a symplectic map by Cayley transform of a Hamiltonian matrix.

    Requires even ``latent_dim``. The Hamiltonian is ``H = J S`` with ``S``
    symmetric and ``J`` the standard symplectic block matrix.

    Parameters
    ----------
    raw : Tensor
        Unconstrained square factor with shape ``(d, d)`` and even ``d``.

    Returns
    -------
    Tensor
        Approximate symplectic matrix ``(I - H)^{-1} (I + H)``.

    Raises
    ------
    ValueError
        If ``d`` is odd.

    Notes
    -----
    Hamiltonian neural-network lineage (Greydanus, Dzamba, and Yosinski,
    2019; ``Greydanus2019HNN``). The symplectic constraint applies to
    parameterized ``K``, not decoded ``x``.
    """
    dim = int(raw.shape[-1])
    if dim % 2 != 0:
        msg = f"symplectic parameterization requires even latent_dim, got {dim}"
        raise ValueError(msg)
    half = dim // 2
    identity_half = torch.eye(half, dtype=raw.dtype, device=raw.device)
    zeros = torch.zeros_like(identity_half)
    symplectic_j = torch.cat(
        [
            torch.cat([zeros, identity_half], dim=-1),
            torch.cat([-identity_half, zeros], dim=-1),
        ],
        dim=-2,
    )
    symmetric = _symmetric_from_raw(raw)
    hamiltonian = symplectic_j @ symmetric
    identity = torch.eye(dim, dtype=raw.dtype, device=raw.device)
    damped = identity - hamiltonian + _CAYLEY_DAMP * identity
    return torch.linalg.solve(damped, identity + hamiltonian)


def reset_logit_matrix(
    raw: Tensor,
    *,
    init_mode: InitMode,
    init_scale: float,
) -> None:
    """Initialize unconstrained logits toward a near-identity stochastic map.

    Parameters
    ----------
    raw : Tensor
        Logit parameter to reset in place.
    init_mode : InitMode
        Initialization strategy.
    init_scale : float
        Noise scale for ``identity_noise`` / ``xavier``.
    """
    with torch.no_grad():
        raw.fill_(-8.0)
        raw.fill_diagonal_(8.0)
        if init_mode == "identity_noise":
            raw.add_(torch.randn_like(raw) * init_scale)
        elif init_mode == "xavier":
            nn.init.xavier_uniform_(raw)
        elif init_mode != "identity":
            msg = f"Unknown init_mode: {init_mode!r}"
            raise ValueError(msg)


def reset_symplectic_matrix(
    raw: Tensor,
    *,
    init_mode: InitMode,
    init_scale: float,
) -> None:
    """Initialize the symplectic generator toward a near-identity Cayley map.

    Parameters
    ----------
    raw : Tensor
        Unconstrained Hamiltonian factor.
    init_mode : InitMode
        Initialization strategy.
    init_scale : float
        Noise scale for noisy modes.
    """
    with torch.no_grad():
        raw.zero_()
        if init_mode in {"identity_noise", "xavier"}:
            raw.add_(torch.randn_like(raw) * init_scale)
        elif init_mode != "identity":
            msg = f"Unknown init_mode: {init_mode!r}"
            raise ValueError(msg)
