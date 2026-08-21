"""Closed-form Koopman identification on frozen latent encodings.

Ridge (Tikhonov) and total-least-squares maps follow the package row
convention ``z_next = z @ K.T``. Constrained least squares is the same
ridge solve followed by a spectral-radius projection onto the unit disk.
This is **not** ResKoopNet residual training, not Askham–Kutz variable
projection, and not graph-factor identification.

This module must not import :mod:`koopman_graph.training` or
:mod:`koopman_graph.model`. It does not import :mod:`koopman_graph.baselines`;
tests may cross-check against ``fit_row_operator`` / ``fit_tls_row_operator``.

References
----------
Williams, M. O., Kevrekidis, I. G. & Rowley, C. W. (2015). A data-driven
approximation of the Koopman operator: extending dynamic mode
decomposition. *Journal of Nonlinear Science*, 25(6), 1307–1346.
https://doi.org/10.1007/s00332-015-9258-5

Dawson, S. T. M., Hemati, M. S., Williams, M. O. & Rowley, C. W. (2016).
Characterizing and correcting for the effect of sensor noise in the
dynamic mode decomposition. *Experiments in Fluids*, 57(3), 42.
https://doi.org/10.1007/s00348-016-2127-7

Hemati, M. S., Rowley, C. W., Deem, E. A. & Cattafesta, L. N. (2017).
De-biasing the dynamic mode decomposition for applied Koopman spectral
analysis of noisy datasets. *Theoretical and Computational Fluid
Dynamics*, 31(4), 349–368. https://doi.org/10.1007/s00162-017-0432-2

Xu, Y., Shao, K., Logothetis, N. K. & Shen, Z. (2025). ResKoopNet:
learning Koopman representations for complex dynamics with spectral
residuals. *Proceedings of the 42nd International Conference on Machine
Learning*, PMLR 267:69647–69674.
https://proceedings.mlr.press/v267/xu25y.html
(``Xu2025ResKoopNet``)
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from koopman_graph.identification.config import IdentificationConfig
from koopman_graph.identification.protocol import LatentPairs, OperatorSnapshot
from koopman_graph.operators.discrete import KoopmanOperator

__all__ = [
    "ClosedFormBackend",
    "apply_operator_snapshot",
    "identify_operator",
]

_UNIT_DISK_RADIUS = 1.0


def _flatten_latent_pairs(encodings: LatentPairs) -> tuple[Tensor, Tensor]:
    """Reshape consecutive encodings to ``(n_samples, d)``.

    Parameters
    ----------
    encodings : LatentPairs
        ``z_t`` / ``z_next`` with trailing width :math:`d`.

    Returns
    -------
    tuple of Tensor
        Flattened source and target matrices sharing dtype and device.

    Raises
    ------
    ValueError
        If the encodings are not at least 2-D.
    """
    z_t = encodings.z_t
    z_next = encodings.z_next
    if z_t.ndim < 2:
        msg = (
            "latent encodings must be at least 2-D with trailing latent "
            f"width, got shape {tuple(z_t.shape)}"
        )
        raise ValueError(msg)
    width = z_t.shape[-1]
    return z_t.reshape(-1, width), z_next.reshape(-1, width)


def _fit_ridge_row_operator(left: Tensor, right: Tensor, ridge: float) -> Tensor:
    """Fit ``right ~= left @ K.T`` with optional Tikhonov weight.

    When ``ridge > 0``, solves the augmented system
    ``[left; sqrt(ridge) I] A = [right; 0]`` and returns ``K = A.T``.
    ``ridge`` is a dimensionless Gram weight, not a physical regularization.

    Parameters
    ----------
    left : Tensor
        Source encodings with shape ``(n_samples, d)``.
    right : Tensor
        Target encodings with shape ``(n_samples, d)``.
    ridge : float
        Non-negative finite Tikhonov weight.

    Returns
    -------
    Tensor
        Row-convention operator ``K`` with shape ``(d, d)``.

    Raises
    ------
    ValueError
        If ``ridge == 0`` and there are fewer samples than the latent width.
    """
    n_samples, dim = left.shape
    if right.shape != left.shape:
        msg = (
            "ridge left/right must share shape, "
            f"got {tuple(left.shape)} vs {tuple(right.shape)}"
        )
        raise ValueError(msg)
    if ridge == 0.0 and n_samples < dim:
        msg = (
            "ridge identification with ridge=0 needs at least latent_dim "
            f"samples; got {n_samples} samples and dim {dim}"
        )
        raise ValueError(msg)
    if ridge == 0.0:
        return torch.linalg.lstsq(left, right).solution.T

    sqrt_ridge = math.sqrt(ridge)
    identity = torch.eye(dim, dtype=left.dtype, device=left.device)
    stacked_left = torch.cat([left, sqrt_ridge * identity], dim=0)
    stacked_right = torch.cat(
        [right, torch.zeros(dim, right.shape[1], dtype=left.dtype, device=left.device)],
        dim=0,
    )
    return torch.linalg.lstsq(stacked_left, stacked_right).solution.T


def _fit_tls_row_operator(left: Tensor, right: Tensor) -> Tensor:
    """Fit a total-least-squares map in the package row convention.

    Stacks column-oriented snapshot matrices, truncates the joint SVD to
    the state dimension, and forms ``K = U_y @ pinv(U_x)``.

    Parameters
    ----------
    left : Tensor
        Source encodings with shape ``(n_samples, d)``.
    right : Tensor
        Target encodings with shape ``(n_samples, d)``.

    Returns
    -------
    Tensor
        Row-convention operator ``K`` with shape ``(d, d)``.

    Raises
    ------
    ValueError
        If shapes disagree, rank is degenerate, or ``U_x`` is rank-deficient.
    """
    if left.shape != right.shape:
        msg = (
            "TLS left/right must share shape, "
            f"got {tuple(left.shape)} vs {tuple(right.shape)}"
        )
        raise ValueError(msg)
    if left.ndim != 2:
        msg = f"TLS left/right must be 2-D, got shape {tuple(left.shape)}"
        raise ValueError(msg)

    state_dim = left.shape[1]
    stacked = torch.cat([left.T, right.T], dim=0)
    truncated_rank = min(state_dim, min(stacked.shape))
    if truncated_rank < 1:
        msg = f"TLS truncation rank must be >= 1, got {truncated_rank}"
        raise ValueError(msg)

    u, _, _ = torch.linalg.svd(stacked, full_matrices=False)
    u_r = u[:, :truncated_rank]
    u_x = u_r[:state_dim, :]
    u_y = u_r[state_dim:, :]
    try:
        return u_y @ torch.linalg.pinv(u_x)
    except RuntimeError as exc:
        msg = "TLS identification failed: truncated U_x block is rank-deficient"
        raise ValueError(msg) from exc


def _scale_spectral_radius(matrix: Tensor, max_radius: float) -> Tensor:
    """Scale ``matrix`` so that :math:`\\rho(K) \\le` ``max_radius``.

    Leaves the matrix unchanged when the spectral radius is already at
    or below the bound. This is a projection, not a Lyapunov or Schur
    certificate.

    Parameters
    ----------
    matrix : Tensor
        Square operator.
    max_radius : float
        Positive spectral-radius cap.

    Returns
    -------
    Tensor
        Scaled operator, same dtype and device as ``matrix``.

    Raises
    ------
    ValueError
        If the spectral radius is non-finite.
    """
    radius = torch.linalg.eigvals(matrix).abs().max()
    rho = float(radius.real)
    if not math.isfinite(rho):
        msg = f"constrained_ls spectral radius is non-finite, got {rho!r}"
        raise ValueError(msg)
    if rho <= max_radius:
        return matrix
    return matrix * (max_radius / rho)


def identify_operator(
    encodings: LatentPairs,
    config: IdentificationConfig,
) -> OperatorSnapshot:
    """Fit a dense per-node operator from consecutive encodings.

    ``solver="alternating"`` uses the ridge formula (the fit loop supplies
    the encoder/operator alternation). ``solver="varpro"`` is not
    implemented here; use :class:`~koopman_graph.baselines.OptDMDBaseline`
    for a teaching variable-projection baseline.

    Parameters
    ----------
    encodings : LatentPairs
        Frozen ``z_t`` / ``z_next`` pairs. Trailing width is the latent
        dimension :math:`d`; leading axes are flattened.
    config : IdentificationConfig
        Solver name and Tikhonov weight.

    Returns
    -------
    OperatorSnapshot
        Dense ``matrix`` with shape ``(d, d)``. Factor slots stay ``None``.

    Raises
    ------
    NotImplementedError
        If ``config.solver`` is ``"varpro"``.
    ValueError
        If the encodings are empty or a solver-specific assumption fails.
    """
    if config.solver == "varpro":
        msg = (
            "solver='varpro' is not implemented in identification; "
            "use koopman_graph.baselines.OptDMDBaseline for a teaching "
            "variable-projection DMD, or choose ridge / tls / "
            "constrained_ls / alternating"
        )
        raise NotImplementedError(msg)

    left, right = _flatten_latent_pairs(encodings)
    if left.shape[0] < 1:
        msg = "identification requires at least one latent pair"
        raise ValueError(msg)

    method = "ridge" if config.solver == "alternating" else config.solver
    if method == "ridge":
        matrix = _fit_ridge_row_operator(left, right, config.ridge)
    elif method == "tls":
        matrix = _fit_tls_row_operator(left, right)
    elif method == "constrained_ls":
        matrix = _fit_ridge_row_operator(left, right, config.ridge)
        matrix = _scale_spectral_radius(matrix, _UNIT_DISK_RADIUS)
    else:
        msg = f"unsupported identification solver {config.solver!r}"
        raise ValueError(msg)
    return OperatorSnapshot(matrix=matrix)


def apply_operator_snapshot(koopman: nn.Module, snapshot: OperatorSnapshot) -> None:
    """Write an identified dense matrix onto a per-node discrete operator.

    Parameters
    ----------
    koopman : nn.Module
        Must be a densely parameterized uncontrolled
        :class:`~koopman_graph.operators.KoopmanOperator`.
    snapshot : OperatorSnapshot
        Snapshot whose ``matrix`` slot is set.

    Raises
    ------
    ValueError
        If the operator is not a dense uncontrolled ``KoopmanOperator``,
        or ``snapshot.matrix`` is missing or the wrong shape.
    """
    if not isinstance(koopman, KoopmanOperator):
        msg = (
            "identification currently supports discrete per-node "
            "KoopmanOperator only, got "
            f"{type(koopman).__name__}"
        )
        raise ValueError(msg)
    if koopman.parameterization != "dense":
        msg = (
            "identification requires parameterization='dense', "
            f"got {koopman.parameterization!r}"
        )
        raise ValueError(msg)
    if koopman.control_dim != 0:
        msg = (
            "identification does not support controlled operators "
            f"(control_dim={koopman.control_dim})"
        )
        raise ValueError(msg)
    matrix = snapshot.matrix
    if matrix is None:
        msg = "OperatorSnapshot.matrix is required for per-node identification"
        raise ValueError(msg)
    koopman.set_dense_matrix(matrix)


class ClosedFormBackend:
    """Default :class:`IdentificationBackend` using :func:`identify_operator`.

    Notes
    -----
    This class is a thin dispatch; it does not store fitted state.
    """

    def fit_operator(
        self,
        encodings: LatentPairs,
        config: IdentificationConfig,
    ) -> OperatorSnapshot:
        """Fit a dense operator snapshot from frozen encodings.

        Parameters
        ----------
        encodings : LatentPairs
            Consecutive latent pairs.
        config : IdentificationConfig
            Solver name and ridge weight.

        Returns
        -------
        OperatorSnapshot
            Dense identified ``K``.
        """
        return identify_operator(encodings, config)
