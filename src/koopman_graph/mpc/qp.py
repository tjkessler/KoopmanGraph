"""Condensed quadratic program for additive Koopman-MPC.

Assembles a dense QP in the stacked input sequence ``U = [u_0; …; u_{H-1}]``
for the linear plant

    x_{h+1} = A x_h + B_u u_h

with quadratic stage / terminal costs on ``y_h ≈ C x_h`` and inputs, plus
box constraints on ``u`` and optional box constraints on ``y``.

OSQP is an optional ``[mpc]`` extra imported at call time.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

_OSQP_IMPORT_ERROR = (
    "OSQP is required for KoopmanMPC. Install with: pip install 'koopman-graph[mpc]'"
)


def require_osqp() -> Any:
    """Import ``osqp`` or raise a guided ``ImportError``.

    Returns
    -------
    module
        The ``osqp`` module.

    Raises
    ------
    ImportError
        If OSQP is not installed.
    """
    try:
        import osqp
    except ImportError as exc:  # pragma: no cover - exercised via mock
        raise ImportError(_OSQP_IMPORT_ERROR) from exc
    return osqp


def _as_2d_spd(
    matrix: NDArray[np.floating],
    name: str,
    size: int,
) -> NDArray[np.float64]:
    """Validate and cast a square cost matrix.

    Parameters
    ----------

    matrix : NDArray[np.floating]
        See the function signature / summary for ``matrix``.
    name : str
        See the function signature / summary for ``name``.
    size : int
        See the function signature / summary for ``size``.

    Returns
    -------

    NDArray[np.float64]
        See summary line.

    Raises
    ------

    ValueError
        Raised when inputs are invalid."""
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.shape != (size, size):
        msg = f"{name} must have shape {(size, size)}, got {arr.shape}"
        raise ValueError(msg)
    if not np.allclose(arr, arr.T, atol=1e-8):
        msg = f"{name} must be symmetric"
        raise ValueError(msg)
    eig = np.linalg.eigvalsh(arr)
    if float(eig.min()) < -1e-10:
        msg = f"{name} must be positive semidefinite"
        raise ValueError(msg)
    return 0.5 * (arr + arr.T)


def assemble_condensed_mpc(
    *,
    a_mat: NDArray[np.floating],
    b_mat: NDArray[np.floating],
    c_mat: NDArray[np.floating],
    x0: NDArray[np.floating],
    references: NDArray[np.floating],
    q_cost: NDArray[np.floating],
    r_cost: NDArray[np.floating],
    qf_cost: NDArray[np.floating],
    u_min: NDArray[np.floating] | None,
    u_max: NDArray[np.floating] | None,
    y_min: NDArray[np.floating] | None,
    y_max: NDArray[np.floating] | None,
    stage_margins: NDArray[np.floating] | None = None,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Build dense OSQP data ``(P, q, A, l, u)`` for condensed MPC.

    Parameters
    ----------
    a_mat : ndarray
        State matrix ``A`` with shape ``(d, d)`` (column convention).
    b_mat : ndarray
        Input matrix ``B_u`` with shape ``(d, C)``.
    c_mat : ndarray
        Output map ``C`` with shape ``(F, d)``.
    x0 : ndarray
        Initial state with shape ``(d,)``.
    references : ndarray
        Output references with shape ``(H + 1, F)`` (stages + terminal).
    q_cost, r_cost, qf_cost : ndarray
        PSD stage / input / terminal weights.
    u_min, u_max : ndarray or None
        Optional input box bounds with shape ``(C,)``.
    y_min, y_max : ndarray or None
        Optional output box bounds with shape ``(F,)`` applied at every
        predicted stage (including terminal).
    stage_margins : ndarray or None, optional
        Non-negative per-stage half-widths with shape ``(H + 1,)``. When
        set, stage ``h`` uses ``y_min + m_h`` / ``y_max - m_h`` (broadcast
        over features). Input bounds are never tightened. Requires
        ``y_min`` and/or ``y_max``.

    Returns
    -------
    tuple of ndarray
        ``(P, q_vec, A_ineq, l, u)`` ready for OSQP (dense; convert to CSC
        at solve time).
    """
    a = np.asarray(a_mat, dtype=np.float64)
    b = np.asarray(b_mat, dtype=np.float64)
    c = np.asarray(c_mat, dtype=np.float64)
    x0_vec = np.asarray(x0, dtype=np.float64).reshape(-1)
    refs = np.asarray(references, dtype=np.float64)

    d = a.shape[0]
    if a.shape != (d, d):
        msg = f"a_mat must be square, got {a.shape}"
        raise ValueError(msg)
    control_dim = b.shape[1]
    if b.shape[0] != d:
        msg = f"b_mat must have shape {(d, control_dim)}, got {b.shape}"
        raise ValueError(msg)
    out_dim = c.shape[0]
    if c.shape[1] != d:
        msg = f"c_mat must have shape {(out_dim, d)}, got {c.shape}"
        raise ValueError(msg)
    if x0_vec.shape != (d,):
        msg = f"x0 must have shape {(d,)}, got {x0_vec.shape}"
        raise ValueError(msg)
    if refs.ndim != 2 or refs.shape[1] != out_dim:
        msg = f"references must have shape (H+1, {out_dim}), got {refs.shape}"
        raise ValueError(msg)
    horizon = refs.shape[0] - 1
    if horizon < 1:
        msg = "references must include at least one stage plus terminal"
        raise ValueError(msg)

    q_mat = _as_2d_spd(q_cost, "Q", out_dim)
    r_mat = _as_2d_spd(r_cost, "R", control_dim)
    qf_mat = _as_2d_spd(qf_cost, "Qf", out_dim)

    # Prediction: X = Sx x0 + Su U, with X stacking x_1..x_H (and we also
    # need x_0 for stage-0 cost). Build free response and input maps for
    # x_0..x_H.
    n_u = horizon * control_dim
    # Maps from U to each x_h (h=0..H). x_0 independent of U.
    su_blocks: list[NDArray[np.float64]] = [
        np.zeros((d, n_u), dtype=np.float64)  # h=0
    ]
    sx_blocks: list[NDArray[np.float64]] = [np.eye(d)]
    a_pow = np.eye(d)
    for h in range(1, horizon + 1):
        # x_h = A^h x0 + sum_{j=0}^{h-1} A^{h-1-j} B u_j
        a_pow = a @ a_pow
        sx_blocks.append(a_pow.copy())
        su = np.zeros((d, n_u), dtype=np.float64)
        a_b = b.copy()
        for j in range(h - 1, -1, -1):
            col = slice(j * control_dim, (j + 1) * control_dim)
            su[:, col] = a_b
            a_b = a @ a_b
        su_blocks.append(su)

    # Cost: sum_{h=0}^{H-1} ||C x_h - r_h||_Q^2 + ||u_h||_R^2
    #      + ||C x_H - r_H||_{Qf}^2
    p_mat = np.zeros((n_u, n_u), dtype=np.float64)
    q_vec = np.zeros(n_u, dtype=np.float64)

    for h in range(horizon):
        c_su = c @ su_blocks[h]
        c_sx_x0 = c @ (sx_blocks[h] @ x0_vec)
        err0 = c_sx_x0 - refs[h]
        p_mat += c_su.T @ q_mat @ c_su
        q_vec += c_su.T @ q_mat @ err0
        # Input cost for u_h
        block = slice(h * control_dim, (h + 1) * control_dim)
        p_mat[block, block] += r_mat

    c_su_t = c @ su_blocks[horizon]
    c_sx_x0_t = c @ (sx_blocks[horizon] @ x0_vec)
    err_t = c_sx_x0_t - refs[horizon]
    p_mat += c_su_t.T @ qf_mat @ c_su_t
    q_vec += c_su_t.T @ qf_mat @ err_t

    # Symmetrize numerical drift.
    p_mat = 0.5 * (p_mat + p_mat.T)

    # Inequality constraints: u bounds and optional y bounds.
    constraint_rows: list[NDArray[np.float64]] = []
    lower: list[float] = []
    upper: list[float] = []

    eye_u = np.eye(n_u)
    if u_min is not None or u_max is not None:
        u_lo = (
            np.asarray(u_min, dtype=np.float64).reshape(-1)
            if u_min is not None
            else np.full(control_dim, -np.inf)
        )
        u_hi = (
            np.asarray(u_max, dtype=np.float64).reshape(-1)
            if u_max is not None
            else np.full(control_dim, np.inf)
        )
        if u_lo.shape != (control_dim,) or u_hi.shape != (control_dim,):
            msg = f"u_min/u_max must have shape {(control_dim,)}"
            raise ValueError(msg)
        constraint_rows.append(eye_u)
        lower.extend(np.tile(u_lo, horizon).tolist())
        upper.extend(np.tile(u_hi, horizon).tolist())

    margins: NDArray[np.float64] | None = None
    if stage_margins is not None:
        if y_min is None and y_max is None:
            msg = "stage_margins require y_min and/or y_max"
            raise ValueError(msg)
        margins = np.asarray(stage_margins, dtype=np.float64).reshape(-1)
        if margins.shape != (horizon + 1,):
            msg = f"stage_margins must have shape {(horizon + 1,)}, got {margins.shape}"
            raise ValueError(msg)
        if np.any(margins < -1e-12):
            msg = "stage_margins must be non-negative"
            raise ValueError(msg)
        margins = np.maximum(margins, 0.0)

    if y_min is not None or y_max is not None:
        y_lo = (
            np.asarray(y_min, dtype=np.float64).reshape(-1)
            if y_min is not None
            else np.full(out_dim, -np.inf)
        )
        y_hi = (
            np.asarray(y_max, dtype=np.float64).reshape(-1)
            if y_max is not None
            else np.full(out_dim, np.inf)
        )
        if y_lo.shape != (out_dim,) or y_hi.shape != (out_dim,):
            msg = f"y_min/y_max must have shape {(out_dim,)}"
            raise ValueError(msg)
        for h in range(horizon + 1):
            c_su = c @ su_blocks[h]
            free = c @ (sx_blocks[h] @ x0_vec)
            margin = 0.0 if margins is None else float(margins[h])
            stage_lo = y_lo + margin
            stage_hi = y_hi - margin
            constraint_rows.append(c_su)
            # y_lo ≤ C x0_free + C Su U ≤ y_hi
            # → y_lo - free ≤ C Su U ≤ y_hi - free
            lower.extend((stage_lo - free).tolist())
            upper.extend((stage_hi - free).tolist())

    if constraint_rows:
        a_ineq = np.vstack(constraint_rows)
        l_vec = np.asarray(lower, dtype=np.float64)
        u_vec = np.asarray(upper, dtype=np.float64)
    else:
        # OSQP requires at least a dummy constraint; use 0*U = 0.
        a_ineq = np.zeros((1, n_u), dtype=np.float64)
        l_vec = np.array([0.0])
        u_vec = np.array([0.0])

    return p_mat, q_vec, a_ineq, l_vec, u_vec


def solve_dense_qp(
    p_mat: NDArray[np.floating],
    q_vec: NDArray[np.floating],
    a_ineq: NDArray[np.floating],
    l_vec: NDArray[np.floating],
    u_vec: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Solve a dense QP with OSQP and return the primal solution.

    Parameters
    ----------

    p_mat : NDArray[np.floating]
        See the function signature / summary for ``p_mat``.
    q_vec : NDArray[np.floating]
        See the function signature / summary for ``q_vec``.
    a_ineq : NDArray[np.floating]
        See the function signature / summary for ``a_ineq``.
    l_vec : NDArray[np.floating]
        See the function signature / summary for ``l_vec``.
    u_vec : NDArray[np.floating]
        See the function signature / summary for ``u_vec``.

    Returns
    -------

    NDArray[np.float64]
        See summary line.

    Raises
    ------

    ImportError
        If the ``[mpc]`` extra (OSQP) is missing.
    RuntimeError
        If the QP is infeasible or OSQP fails to return a solution."""
    osqp = require_osqp()
    from scipy import sparse

    p_csc = sparse.csc_matrix(np.asarray(p_mat, dtype=np.float64))
    a_csc = sparse.csc_matrix(np.asarray(a_ineq, dtype=np.float64))
    q = np.asarray(q_vec, dtype=np.float64).reshape(-1)
    lower = np.asarray(l_vec, dtype=np.float64).reshape(-1)
    upper = np.asarray(u_vec, dtype=np.float64).reshape(-1)

    solver = osqp.OSQP()
    solver.setup(
        P=p_csc,
        q=q,
        A=a_csc,
        l=lower,
        u=upper,
        verbose=False,
        polishing=False,
        eps_abs=1e-8,
        eps_rel=1e-8,
    )
    result = solver.solve()
    status = getattr(result.info, "status", None)
    if result.x is None or status not in {"solved", "solved inaccurate"}:
        msg = (
            f"KoopmanMPC QP solve failed (status={status!r}). "
            "Relax constraints or the horizon, or check model controllability."
        )
        raise RuntimeError(msg)
    return np.asarray(result.x, dtype=np.float64)
