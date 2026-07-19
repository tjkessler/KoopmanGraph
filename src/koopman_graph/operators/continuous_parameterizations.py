"""Structural / dense generator assembly and reset for continuous operators.

Shallow sibling of :mod:`koopman_graph.operators.continuous`. Holds cohesive
ODO / Schur / dissipative / Lyapunov / dense mechanics used by
:class:`~koopman_graph.operators.continuous.ContinuousKoopmanOperator` without
introducing a parameterization class hierarchy. Prefer importing via
``continuous`` re-exports when writing against the continuous capability
surface.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from koopman_graph.operators.contract import (
    DISSIPATIVE_MIN_EIGENVALUE,
    STABILITY_EPS_MARGIN,
    InitMode,
    Parameterization,
    StabilityCertificate,
    build_stability_certificate,
    cayley_orthogonal,
    strict_spectral_bound,
)


def negative_strict_diagonal_values(
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


def identity_negative_diag_raw(max_real_eigenvalue: float) -> float:
    """Return raw diagonal init for a near-zero stable generator.

    Parameters
    ----------
    max_real_eigenvalue : float
        Magnitude scale for structurally stable negative real parts.

    Returns
    -------
    float
        Unconstrained raw value whose mapped eigenvalue is near the origin
        but strictly negative.
    """
    bound = strict_spectral_bound(max_real_eigenvalue)
    target = -bound * 1e-2
    ratio = abs(target) / bound
    return float(torch.atanh(torch.tensor(ratio)).item())


def assemble_odo_generator(
    cayley_O1: Tensor,
    cayley_O2: Tensor,
    diag_raw: Tensor,
    max_real_eigenvalue: float,
) -> Tensor:
    """Assemble the ODO generator ``L = O_1 D O_2^T``.

    Parameters
    ----------
    cayley_O1, cayley_O2 : Tensor
        Cayley factors for the two orthogonal matrices.
    diag_raw : Tensor
        Unconstrained diagonal parameters.
    max_real_eigenvalue : float
        Magnitude scale for negative diagonal entries.

    Returns
    -------
    Tensor
        Assembled generator matrix.
    """
    o1, o2 = cayley_orthogonal(cayley_O1), cayley_orthogonal(cayley_O2)
    values = negative_strict_diagonal_values(diag_raw, max_real_eigenvalue)
    return o1 @ torch.diag(values) @ o2.T


def assemble_schur_generator(
    cayley_Q: Tensor,
    schur_diag_raw: Tensor,
    schur_off_raw: Tensor,
    max_real_eigenvalue: float,
) -> Tensor:
    """Assemble the Schur-form generator ``L = Q T Q^T``.

    Parameters
    ----------
    cayley_Q : Tensor
        Cayley factor for the orthogonal matrix ``Q``.
    schur_diag_raw : Tensor
        Unconstrained diagonal parameters of ``T``.
    schur_off_raw : Tensor
        Strictly upper-triangular entries of ``T`` (lower triangle ignored).
    max_real_eigenvalue : float
        Magnitude scale for negative diagonal entries.

    Returns
    -------
    Tensor
        Assembled generator matrix.
    """
    diag_vals = negative_strict_diagonal_values(schur_diag_raw, max_real_eigenvalue)
    triangular = torch.triu(schur_off_raw, diagonal=1) + torch.diag(diag_vals)
    q = cayley_orthogonal(cayley_Q)
    return q @ triangular @ q.T


def assemble_dissipative_generator(
    dissipative_L: Tensor,
    latent_dim: int,
) -> Tensor:
    """Assemble the dissipative generator ``L = -(S)`` with SPD ``S``.

    Parameters
    ----------
    dissipative_L : Tensor
        Lower-triangular factor parameters.
    latent_dim : int
        Latent dimension (square size of ``L``).

    Returns
    -------
    Tensor
        Symmetric negative-definite generator matrix.
    """
    lower = torch.tril(dissipative_L)
    diag_index = torch.arange(latent_dim, device=lower.device)
    lower[diag_index, diag_index] = (
        torch.nn.functional.softplus(lower[diag_index, diag_index])
        + DISSIPATIVE_MIN_EIGENVALUE
    )
    identity = torch.eye(latent_dim, device=lower.device, dtype=lower.dtype)
    spd = lower @ lower.T + DISSIPATIVE_MIN_EIGENVALUE * identity
    return -spd


def lyapunov_diagonal(
    lyap_diag_raw: Tensor,
    max_real_eigenvalue: float,
) -> Tensor:
    """Return negative diagonal eigenvalues for the Lyapunov parameterization.

    Parameters
    ----------
    lyap_diag_raw : Tensor
        Unconstrained diagonal parameters.
    max_real_eigenvalue : float
        Magnitude scale for stable real parts.

    Returns
    -------
    Tensor
        Strictly negative diagonal eigenvalues.
    """
    return negative_strict_diagonal_values(lyap_diag_raw, max_real_eigenvalue)


def lyapunov_certificate_matrix(
    cayley_Q: Tensor,
    lyap_p_raw: Tensor,
) -> Tensor:
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


def assemble_lyapunov_generator(
    cayley_Q: Tensor,
    lyap_diag_raw: Tensor,
    max_real_eigenvalue: float,
) -> Tensor:
    """Assemble the Lyapunov generator ``L = Q diag(d) Q^T``.

    Parameters
    ----------
    cayley_Q : Tensor
        Cayley factor for the orthogonal matrix ``Q``.
    lyap_diag_raw : Tensor
        Unconstrained diagonal parameters.
    max_real_eigenvalue : float
        Magnitude scale for negative diagonal entries.

    Returns
    -------
    Tensor
        Assembled generator matrix.
    """
    q = cayley_orthogonal(cayley_Q)
    return q @ torch.diag(lyapunov_diagonal(lyap_diag_raw, max_real_eigenvalue)) @ q.T


def reset_dense_generator(
    matrix: Tensor,
    *,
    init_mode: InitMode,
    init_scale: float,
) -> None:
    """Reinitialize a dense learnable generator matrix ``L``.

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
        nn.init.zeros_(matrix)
    elif init_mode == "identity_noise":
        nn.init.zeros_(matrix)
        with torch.no_grad():
            matrix.add_(torch.randn_like(matrix) * init_scale)
    elif init_mode == "xavier":
        nn.init.xavier_uniform_(matrix)
    else:
        msg = f"Unknown init_mode: {init_mode!r}"
        raise ValueError(msg)


def reset_odo_generator(
    cayley_O1: Tensor,
    cayley_O2: Tensor,
    diag_raw: Tensor,
    *,
    init_mode: InitMode,
    init_scale: float,
    max_real_eigenvalue: float,
) -> None:
    """Reinitialize ODO generator parameters in place.

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
    max_real_eigenvalue : float
        Upper bound used for identity diagonal initialization.
    """
    nn.init.zeros_(cayley_O1)
    nn.init.zeros_(cayley_O2)
    if init_mode in {"identity", "identity_noise"}:
        nn.init.constant_(diag_raw, identity_negative_diag_raw(max_real_eigenvalue))
        if init_mode == "identity_noise":
            with torch.no_grad():
                diag_raw.add_(torch.randn_like(diag_raw) * init_scale)
    elif init_mode == "xavier":
        nn.init.xavier_uniform_(cayley_O1)
        nn.init.xavier_uniform_(cayley_O2)
        nn.init.normal_(diag_raw)
    else:
        msg = f"Unknown init_mode: {init_mode!r}"
        raise ValueError(msg)


def reset_schur_generator(
    cayley_Q: Tensor,
    schur_diag_raw: Tensor,
    schur_off_raw: Tensor,
    *,
    init_mode: InitMode,
    init_scale: float,
    max_real_eigenvalue: float,
) -> None:
    """Reinitialize Schur generator parameters in place.

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
    max_real_eigenvalue : float
        Upper bound used for identity diagonal initialization.
    """
    nn.init.zeros_(cayley_Q)
    nn.init.zeros_(schur_off_raw)
    if init_mode in {"identity", "identity_noise"}:
        nn.init.constant_(
            schur_diag_raw, identity_negative_diag_raw(max_real_eigenvalue)
        )
        if init_mode == "identity_noise":
            with torch.no_grad():
                schur_off_raw.add_(torch.randn_like(schur_off_raw) * init_scale)
    elif init_mode == "xavier":
        nn.init.xavier_uniform_(cayley_Q)
        nn.init.xavier_uniform_(schur_off_raw)
        nn.init.normal_(schur_diag_raw)
    else:
        msg = f"Unknown init_mode: {init_mode!r}"
        raise ValueError(msg)


def reset_dissipative_generator(
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
    elif init_mode != "identity":
        msg = f"Unknown init_mode: {init_mode!r}"
        raise ValueError(msg)


def reset_lyapunov_generator(
    cayley_Q: Tensor,
    lyap_diag_raw: Tensor,
    lyap_p_raw: Tensor,
    *,
    init_mode: InitMode,
    init_scale: float,
    max_real_eigenvalue: float,
) -> None:
    """Reinitialize Lyapunov generator parameters in place.

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
    max_real_eigenvalue : float
        Upper bound used for identity diagonal initialization.
    """
    nn.init.zeros_(cayley_Q)
    if init_mode in {"identity", "identity_noise"}:
        nn.init.constant_(
            lyap_diag_raw, identity_negative_diag_raw(max_real_eigenvalue)
        )
        nn.init.constant_(lyap_p_raw, 0.0)
        if init_mode == "identity_noise":
            with torch.no_grad():
                lyap_diag_raw.add_(torch.randn_like(lyap_diag_raw) * init_scale)
    elif init_mode == "xavier":
        nn.init.xavier_uniform_(cayley_Q)
        nn.init.normal_(lyap_diag_raw)
        nn.init.normal_(lyap_p_raw)
    else:
        msg = f"Unknown init_mode: {init_mode!r}"
        raise ValueError(msg)


def max_real_part_of_generator(generator: Tensor) -> Tensor:
    """Return ``max Re(λ_i(L))`` from an assembled continuous generator.

    Parameters
    ----------
    generator : Tensor
        Assembled generator ``L`` with shape ``(latent_dim, latent_dim)``.

    Returns
    -------
    Tensor
        Scalar tensor ``\\max_i \\operatorname{Re}(\\lambda_i(L))``.
    """
    return torch.linalg.eigvals(generator).real.max()


def continuous_bound_metric(
    parameterization: Parameterization,
    *,
    max_real_eigenvalue: float,
    diag_raw: Tensor | None = None,
    schur_diag_raw: Tensor | None = None,
    lyap_diag_raw: Tensor | None = None,
    dissipative_generator: Tensor | None = None,
    assembled_generator: Tensor | None = None,
) -> Tensor:
    """Cheap soft/structural monitoring bound for continuous generators.

    For ``"odo"`` / ``"schur"`` / ``"lyapunov"``, this is a diagonal-factor
    bound (not necessarily the true max real eigenvalue of assembled ``L``).
    For ``"dissipative"``, this is the largest eigenvalue of the
    symmetric-part generator. For ``"dense"``, this equals
    :func:`max_real_part_of_generator`. ``"auxiliary_spectral"`` has no
    fixed global bound.

    Parameters
    ----------
    parameterization : Parameterization
        Active generator parameterization.
    max_real_eigenvalue : float
        Magnitude scale for structurally stable negative real parts.
    diag_raw : Tensor or None, optional
        ODO diagonal raw parameters.
    schur_diag_raw : Tensor or None, optional
        Schur diagonal raw parameters.
    lyap_diag_raw : Tensor or None, optional
        Lyapunov diagonal raw parameters.
    dissipative_generator : Tensor or None, optional
        Assembled dissipative generator.
    assembled_generator : Tensor or None, optional
        Assembled dense generator (for the dense eigvals path).

    Returns
    -------
    Tensor
        Scalar bound metric for the active parameterization.

    Raises
    ------
    ValueError
        If ``parameterization="auxiliary_spectral"`` or required tensors are
        missing for the active mode.
    """
    if parameterization in {"odo", "schur", "lyapunov"}:
        if parameterization == "odo":
            raw = diag_raw
        elif parameterization == "schur":
            raw = schur_diag_raw
        else:
            raw = lyap_diag_raw
        if raw is None:
            msg = f"{parameterization} bound_metric requires diagonal raw parameters"
            raise ValueError(msg)
        diagonal = negative_strict_diagonal_values(raw, max_real_eigenvalue)
        return diagonal.max()
    if parameterization == "dissipative":
        if dissipative_generator is None:
            msg = "dissipative bound_metric requires dissipative_generator"
            raise ValueError(msg)
        return torch.linalg.eigvalsh(dissipative_generator).max()
    if parameterization == "auxiliary_spectral":
        msg = (
            "bound_metric is unavailable for parameterization="
            "'auxiliary_spectral' (state-dependent spectrum); use "
            "instantaneous_spectrum(z) / generator_at(z)"
        )
        raise ValueError(msg)
    if assembled_generator is None:
        msg = "dense bound_metric requires assembled_generator"
        raise ValueError(msg)
    return max_real_part_of_generator(assembled_generator)


def continuous_stability_certificate(
    parameterization: Parameterization,
    *,
    bound_metric: Tensor | None = None,
    lyapunov_diagonal: Tensor | None = None,
    lyapunov_matrix: Tensor | None = None,
) -> StabilityCertificate | None:
    """Hurwitz stability certificate for continuous structural modes.

    Parameters
    ----------
    parameterization : Parameterization
        Active generator parameterization.
    bound_metric : Tensor or None, optional
        Continuous :func:`continuous_bound_metric` value (used for ``schur`` /
        ``dissipative`` margins).
    lyapunov_diagonal : Tensor or None, optional
        Lyapunov diagonal eigenvalues (``"lyapunov"`` margin path).
    lyapunov_matrix : Tensor or None, optional
        Lyapunov certificate matrix ``P``.

    Returns
    -------
    StabilityCertificate or None
        Frozen certificate for structural modes; ``None`` for soft /
        auxiliary modes.
    """
    if parameterization == "lyapunov":
        if lyapunov_diagonal is None:
            msg = "lyapunov stability_certificate requires lyapunov_diagonal"
            raise ValueError(msg)
        return build_stability_certificate(
            -lyapunov_diagonal.max(),
            lyapunov_matrix=lyapunov_matrix,
        )
    if parameterization in {"schur", "dissipative"}:
        if bound_metric is None:
            msg = f"{parameterization} stability_certificate requires bound_metric"
            raise ValueError(msg)
        return build_stability_certificate(-bound_metric)
    return None
