"""Coverage and error-path tests for :mod:`koopman_graph.analysis`."""

from __future__ import annotations

import builtins
from typing import Any

import numpy as np
import pytest
import torch

from koopman_graph.analysis import (
    discrete_lyapunov_lmi,
)
from koopman_graph.operators.joint_stability import MAX_JOINT_LYAPUNOV_SIZE


def test_lmi_oversized_import_failure_and_solve_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lyapunov helper reports infeasible on size, import, and solve failures."""
    oversized = discrete_lyapunov_lmi(torch.eye(MAX_JOINT_LYAPUNOV_SIZE + 1))
    assert oversized.feasible is False

    real_import = builtins.__import__

    def _block_scipy(
        name: str,
        globals: Any = None,
        locals: Any = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "scipy" or name.startswith("scipy."):
            raise ImportError("blocked scipy")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _block_scipy)
    missing = discrete_lyapunov_lmi(0.5 * torch.eye(3))
    assert missing.feasible is False
    monkeypatch.setattr(builtins, "__import__", real_import)

    import scipy.linalg as sla

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise np.linalg.LinAlgError("singular")

    monkeypatch.setattr(sla, "solve_discrete_lyapunov", _boom)
    failed = discrete_lyapunov_lmi(0.5 * torch.eye(3))
    assert failed.feasible is False
