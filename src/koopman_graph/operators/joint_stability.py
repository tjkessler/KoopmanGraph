"""Structural joint bounds on assembled Koopman operators (TASK-1822–1824).

Honesty contract
----------------
Gershgorin's circle theorem yields a **sufficient** upper bound on the
spectral radius of a square matrix ``A``::

    ρ(A) ≤ max_i ( |a_ii| + Σ_{j≠i} |a_ij| )

The bound is **not tight** in general: it may exceed the true ``ρ(A)``
(including cases where the bound is ``> 1`` while ``ρ(A) < 1``). It never
undercuts ``ρ(A)``.

Opt-in assembled certificates (TASK-1824) when ``N·d`` is modest:

* ``kind="schur"`` — true spectral radius ``ρ(K_eff)`` via ``eigvals``
  (labeled Schur to mirror factor structural vocabulary). Not a training
  parameterization of assembled ``K_eff``.
* ``kind="lyapunov"`` — same ``ρ``, plus a discrete Lyapunov matrix ``P``
  solving ``P - K^T P K = I`` when ``ρ < 1``; ``P`` is omitted when
  ``ρ ≥ 1``.

Distinguish four surfaces:

1. **Joint Gershgorin bound** (default
   :meth:`~koopman_graph.operators.HeteroGraphKoopmanOperator.stability_certificate`)
   — sufficient upper bound on assembled ``ρ(K_eff)``; not a tight
   certificate.
2. **Opt-in assembled Schur / Lyapunov certificates**
   (``kind="schur"|"lyapunov"``) — post-hoc on assembled ``K_eff`` under
   size ceilings; not factor-wise structural training modes.
3. **Factor structural certificates**
   (:class:`~koopman_graph.operators.StabilityCertificate` via
   :meth:`~koopman_graph.operators.HeteroGraphKoopmanOperator.factor_stability_certificate`)
   — Schur / Lyapunov / dissipative margins on individual factors; **not**
   joint ``ρ(K_eff)``.
4. **Soft assembled eigenvalue regularization**
   (:class:`~koopman_graph.losses.EigenvalueRegularizationLoss`) — training
   hinge on assembled eigenvalues; not a certificate.

References
----------
Gershgorin, S. (1931). Über die Abgrenzung der Eigenwerte einer Matrix.
*Izv. Akad. Nauk. USSR Otd. Fiz.-Mat. Nauk*, 6, 749–754.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

__all__ = [
    "JOINT_BOUND_KINDS",
    "MAX_JOINT_LYAPUNOV_SIZE",
    "MAX_JOINT_SCHUR_SIZE",
    "JointBoundKind",
    "JointStabilityCertificate",
    "build_joint_stability_certificate",
    "gershgorin_radius_bound",
    "joint_certificate_from_assembled",
    "lyapunov_joint_bound",
    "require_joint_assembled_size",
    "schur_radius_bound",
]

JointBoundKind = Literal["gershgorin", "schur", "lyapunov"]
JOINT_BOUND_KINDS: frozenset[str] = frozenset({"gershgorin", "schur", "lyapunov"})

# Match EigenvalueRegularizationLoss assembled eig ceiling (eigvals cost).
MAX_JOINT_SCHUR_SIZE = 4096
# Vectorized discrete Lyapunov is O((N·d)^6); keep modest.
MAX_JOINT_LYAPUNOV_SIZE = 64


@dataclass(frozen=True)
class JointStabilityCertificate:
    """Joint bound / certificate on assembled ``ρ(K_eff)``.

    Public result types in this package are frozen dataclasses with attribute
    access (not mapping/dict styles).

    Attributes
    ----------
    bound : Tensor
        Upper bound (or true radius) on ``ρ(K_eff)``. For
        ``kind="gershgorin"`` this is a sufficient bound that may be loose
        (DESIGN R4). For ``"schur"`` / ``"lyapunov"`` it is the true
        spectral radius.
    margin : Tensor
        Discrete unit-disk gap ``1 - bound``. For Gershgorin, a positive
        margin means the **bound** lies inside the unit disk (not a proof
        that ``ρ < 1`` when the bound is loose). For Schur / Lyapunov, the
        margin tracks true ``ρ``.
    kind : {"gershgorin", "schur", "lyapunov"}
        Bound construction used for this certificate.
    lyapunov_matrix : Tensor or None
        Discrete Lyapunov solution ``P`` when ``kind="lyapunov"`` and
        ``ρ < 1``; otherwise ``None``.
    """

    bound: Tensor
    margin: Tensor
    kind: JointBoundKind = "gershgorin"
    lyapunov_matrix: Tensor | None = None


def build_joint_stability_certificate(
    bound: Tensor,
    *,
    kind: JointBoundKind = "gershgorin",
    lyapunov_matrix: Tensor | None = None,
) -> JointStabilityCertificate:
    """Build a frozen :class:`JointStabilityCertificate` from a scalar bound.

    Parameters
    ----------
    bound : Tensor
        Scalar upper bound (or true radius) on ``ρ(K_eff)``.
    kind : {"gershgorin", "schur", "lyapunov"}, optional
        Bound construction label. Default ``"gershgorin"``.
    lyapunov_matrix : Tensor or None, optional
        Optional discrete Lyapunov matrix for ``kind="lyapunov"``.

    Returns
    -------
    JointStabilityCertificate
        Frozen joint certificate with ``margin = 1 - bound``.

    Raises
    ------
    ValueError
        If ``kind`` is not a supported joint bound kind.
    """
    if kind not in JOINT_BOUND_KINDS:
        msg = f"kind must be one of {sorted(JOINT_BOUND_KINDS)}, got {kind!r}"
        raise ValueError(msg)
    bound_scalar = bound.reshape(())
    return JointStabilityCertificate(
        bound=bound_scalar,
        margin=1.0 - bound_scalar,
        kind=kind,
        lyapunov_matrix=lyapunov_matrix,
    )


def require_joint_assembled_size(size: int, *, kind: JointBoundKind) -> None:
    """Raise if assembled size exceeds the ceiling for ``kind``.

    Parameters
    ----------
    size : int
        Assembled operator width ``N·d`` (or ``Σ N_τ·d_τ``).
    kind : {"gershgorin", "schur", "lyapunov"}
        Requested joint certificate kind. ``"gershgorin"`` has no ceiling.

    Raises
    ------
    ValueError
        If ``size`` is non-positive or exceeds the kind-specific ceiling.
    """
    if size < 1:
        msg = f"assembled joint certificate size must be positive, got {size}"
        raise ValueError(msg)
    if kind == "gershgorin":
        return
    if kind == "schur":
        ceiling = MAX_JOINT_SCHUR_SIZE
    elif kind == "lyapunov":
        ceiling = MAX_JOINT_LYAPUNOV_SIZE
    else:
        msg = f"kind must be one of {sorted(JOINT_BOUND_KINDS)}, got {kind!r}"
        raise ValueError(msg)
    if size > ceiling:
        msg = (
            f"assembled joint certificate kind={kind!r} requires "
            f"N·d ≤ {ceiling}, got size={size}"
        )
        raise ValueError(msg)


def _require_square_matrix(matrix: Tensor, *, caller: str) -> int:
    """Validate a square 2-D matrix and return its width.

    Parameters
    ----------
    matrix : Tensor
        Candidate square matrix.
    caller : str
        Name used in error messages.

    Returns
    -------
    int
        Matrix width ``n``.

    Raises
    ------
    ValueError
        If ``matrix`` is not 2-D square or is empty.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        msg = f"{caller} expects a square 2-D matrix, got shape {tuple(matrix.shape)}"
        raise ValueError(msg)
    size = int(matrix.shape[0])
    if size < 1:
        msg = f"{caller} requires n ≥ 1"
        raise ValueError(msg)
    return size


def gershgorin_radius_bound(matrix: Tensor) -> Tensor:
    """Return the Gershgorin row-sum upper bound on ``ρ(matrix)``.

    Computes ``max_i (|a_ii| + Σ_{j≠i} |a_ij|)`` for a square matrix. This is
    a sufficient bound: ``ρ(A) ≤`` the returned value, with equality not
    guaranteed (DESIGN R4 / Appendix B).

    Parameters
    ----------
    matrix : Tensor
        Square real or complex matrix with shape ``(n, n)``, ``n ≥ 1``.

    Returns
    -------
    Tensor
        Scalar (0-dim) bound with a real floating dtype compatible with
        ``matrix``.

    Raises
    ------
    ValueError
        If ``matrix`` is not 2-D square or is empty.
    """
    _require_square_matrix(matrix, caller="gershgorin_radius_bound")
    abs_matrix = matrix.abs()
    # Row sums of |A|; subtract diagonal then re-add |diag| once.
    row_off = abs_matrix.sum(dim=-1) - abs_matrix.diagonal()
    radii = abs_matrix.diagonal() + row_off
    return radii.max().real


def schur_radius_bound(matrix: Tensor) -> Tensor:
    """Return the true spectral radius ``ρ(matrix)`` via ``eigvals``.

    Labeled ``schur`` for the joint certificate API (TASK-1824); this is a
    post-hoc spectrum of assembled ``K_eff``, not a structural training
    parameterization.

    Parameters
    ----------
    matrix : Tensor
        Square real or complex matrix with shape ``(n, n)``, ``n ≥ 1``.

    Returns
    -------
    Tensor
        Scalar (0-dim) ``max |λ_i|``.

    Raises
    ------
    ValueError
        If ``matrix`` is not 2-D square, is empty, or exceeds
        :data:`MAX_JOINT_SCHUR_SIZE`.
    """
    size = _require_square_matrix(matrix, caller="schur_radius_bound")
    require_joint_assembled_size(size, kind="schur")
    return torch.linalg.eigvals(matrix).abs().max().real


def joint_certificate_from_assembled(
    matrix: Tensor,
    *,
    kind: JointBoundKind,
) -> JointStabilityCertificate:
    """Build a joint certificate from an assembled square operator.

    Parameters
    ----------
    matrix : Tensor
        Assembled ``K_eff`` with shape ``(n, n)``.
    kind : {"gershgorin", "schur", "lyapunov"}
        Certificate construction. ``"gershgorin"`` uses the row-sum bound;
        ``"schur"`` / ``"lyapunov"`` enforce size ceilings.

    Returns
    -------
    JointStabilityCertificate
        Frozen joint certificate for ``matrix``.

    Raises
    ------
    ValueError
        If ``kind`` is unsupported, the matrix is invalid, or a ceiling is
        exceeded.
    """
    if kind == "gershgorin":
        return build_joint_stability_certificate(
            gershgorin_radius_bound(matrix),
            kind="gershgorin",
        )
    if kind == "schur":
        return build_joint_stability_certificate(
            schur_radius_bound(matrix),
            kind="schur",
        )
    if kind == "lyapunov":
        rho, lyap = lyapunov_joint_bound(matrix)
        return build_joint_stability_certificate(
            rho,
            kind="lyapunov",
            lyapunov_matrix=lyap,
        )
    msg = f"kind must be one of {sorted(JOINT_BOUND_KINDS)}, got {kind!r}"
    raise ValueError(msg)


def lyapunov_joint_bound(matrix: Tensor) -> tuple[Tensor, Tensor | None]:
    """Return ``(ρ, P)`` for a discrete Lyapunov certificate on ``matrix``.

    Computes ``ρ = max|λ|``. When ``ρ < 1``, solves
    ``P - K^T P K = I`` via the vectorized system
    ``(I - K^T ⊗ K^T) vec(P) = vec(I)`` and returns the reshaped ``P``.
    When ``ρ ≥ 1``, returns ``(ρ, None)`` (no PD discrete Lyapunov solution
    for ``Q = I``).

    Parameters
    ----------
    matrix : Tensor
        Square real matrix with shape ``(n, n)``, ``n ≥ 1``.

    Returns
    -------
    tuple of (Tensor, Tensor or None)
        Spectral radius and optional Lyapunov matrix ``P``.

    Raises
    ------
    ValueError
        If ``matrix`` is not 2-D square, is empty, or exceeds
        :data:`MAX_JOINT_LYAPUNOV_SIZE`.
    """
    size = _require_square_matrix(matrix, caller="lyapunov_joint_bound")
    require_joint_assembled_size(size, kind="lyapunov")
    # Work in real float; complex assembled K_eff is out of scope here.
    k = matrix.to(dtype=torch.promote_types(matrix.dtype, torch.float32))
    if k.is_complex():
        msg = "lyapunov_joint_bound requires a real assembled matrix"
        raise ValueError(msg)
    rho = torch.linalg.eigvals(k).abs().max().real
    if float(rho.detach()) >= 1.0 - 1e-8:
        return rho, None
    eye = torch.eye(size, dtype=k.dtype, device=k.device)
    # (I - kron(K.T, K.T)) vec(P) = vec(I)
    kron = torch.kron(k.transpose(-2, -1), k.transpose(-2, -1))
    system = torch.eye(size * size, dtype=k.dtype, device=k.device) - kron
    vec_p = torch.linalg.solve(system, eye.reshape(-1))
    lyap = vec_p.reshape(size, size)
    # Symmetrize numerical noise from the linear solve.
    lyap = 0.5 * (lyap + lyap.transpose(-2, -1))
    return rho, lyap
