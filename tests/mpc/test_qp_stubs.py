"""Coverage and error-path tests for :mod:`koopman_graph.mpc`."""

from __future__ import annotations

import builtins
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from koopman_graph.mpc.qp import (
    _as_2d_spd,
    assemble_condensed_mpc,
    require_osqp,
    solve_dense_qp,
)


def _qp_kwargs(**overrides: Any) -> dict[str, Any]:
    """Minimal condensed-QP arguments for a 2-D plant."""
    payload = {
        "a_mat": 0.9 * np.eye(2),
        "b_mat": np.array([[1.0], [0.0]]),
        "c_mat": np.eye(2),
        "x0": np.zeros(2),
        "references": np.zeros((3, 2)),
        "q_cost": np.eye(2),
        "r_cost": np.eye(1),
        "qf_cost": np.eye(2),
        "u_min": None,
        "u_max": None,
        "y_min": None,
        "y_max": None,
    }
    payload.update(overrides)
    return payload


def test_mpc_qp_assembly_and_mocked_osqp(monkeypatch: pytest.MonkeyPatch) -> None:
    """Condensed QP assembly and OSQP solve run without the ``[mpc]`` extra."""
    assembled = assemble_condensed_mpc(**_qp_kwargs())
    assert len(assembled) == 5
    bounded = assemble_condensed_mpc(
        **_qp_kwargs(
            u_min=np.array([-1.0]),
            u_max=np.array([1.0]),
            y_min=np.array([-2.0, -2.0]),
            y_max=np.array([2.0, 2.0]),
            stage_margins=np.zeros(3),
        )
    )
    assert bounded[2].shape[0] > 1
    with pytest.raises(ValueError, match="square"):
        assemble_condensed_mpc(**_qp_kwargs(a_mat=np.ones((2, 3))))
    with pytest.raises(ValueError, match="b_mat"):
        assemble_condensed_mpc(**_qp_kwargs(b_mat=np.ones((3, 1))))
    with pytest.raises(ValueError, match="c_mat"):
        assemble_condensed_mpc(**_qp_kwargs(c_mat=np.ones((2, 3))))
    with pytest.raises(ValueError, match="x0"):
        assemble_condensed_mpc(**_qp_kwargs(x0=np.zeros(3)))
    with pytest.raises(ValueError, match="references"):
        assemble_condensed_mpc(**_qp_kwargs(references=np.zeros((3, 3))))
    with pytest.raises(ValueError, match="at least one stage"):
        assemble_condensed_mpc(**_qp_kwargs(references=np.zeros((1, 2))))
    with pytest.raises(ValueError, match="stage_margins require"):
        assemble_condensed_mpc(**_qp_kwargs(stage_margins=np.zeros(3)))
    with pytest.raises(ValueError, match="stage_margins must have shape"):
        assemble_condensed_mpc(
            **_qp_kwargs(y_min=np.zeros(2), stage_margins=np.zeros(2))
        )
    with pytest.raises(ValueError, match="non-negative"):
        assemble_condensed_mpc(
            **_qp_kwargs(y_min=np.zeros(2), stage_margins=np.array([-1.0, 0.0, 0.0]))
        )
    with pytest.raises(ValueError, match="u_min/u_max"):
        assemble_condensed_mpc(**_qp_kwargs(u_min=np.zeros(2)))
    with pytest.raises(ValueError, match="y_min/y_max"):
        assemble_condensed_mpc(**_qp_kwargs(y_min=np.zeros(3), y_max=np.zeros(3)))
    with pytest.raises(ValueError, match="shape"):
        _as_2d_spd(np.eye(3), "Q", 2)
    with pytest.raises(ValueError, match="symmetric"):
        _as_2d_spd(np.array([[1.0, 2.0], [0.0, 1.0]]), "Q", 2)
    with pytest.raises(ValueError, match="positive semidefinite"):
        _as_2d_spd(np.array([[-1.0, 0.0], [0.0, -1.0]]), "Q", 2)

    class _Solved:
        x = np.zeros(2)
        info = SimpleNamespace(status="solved")

    class _OSQP:
        def setup(self, **_kwargs: Any) -> None:
            return None

        def solve(self) -> _Solved:
            return _Solved()

    fake = ModuleType("osqp")
    fake.OSQP = _OSQP  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "osqp", fake)
    p_mat, q_vec, a_ineq, l_vec, u_vec = assembled
    sol = solve_dense_qp(p_mat, q_vec, a_ineq, l_vec, u_vec)
    assert sol.shape == (2,)

    class _Failed:
        x = None
        info = SimpleNamespace(status="infeasible")

    class _BadOSQP(_OSQP):
        def solve(self) -> _Failed:
            return _Failed()

    fake.OSQP = _BadOSQP  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError, match="QP solve failed"):
        solve_dense_qp(p_mat, q_vec, a_ineq, l_vec, u_vec)
    monkeypatch.delitem(sys.modules, "osqp", raising=False)
    real_import = builtins.__import__

    def _block(
        name: str,
        globals: Any = None,
        locals: Any = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "osqp":
            raise ImportError("blocked")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _block)
    with pytest.raises(ImportError, match="koopman-graph\\[mpc\\]"):
        require_osqp()
