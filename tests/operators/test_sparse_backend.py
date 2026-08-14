"""Coverage and error-path tests for :mod:`koopman_graph.operators`."""

from __future__ import annotations

import builtins
from typing import Any

import pytest
import torch

from koopman_graph.operators import (
    sparse_leading_eigenvalues,
)


def test_sparse_backend_scipy_eigs_path() -> None:
    """Matrices larger than 32 use SciPy ``eigs`` when available."""
    values = sparse_leading_eigenvalues(torch.eye(40), num_modes=3)
    assert values.numel() == 3


def test_sparse_backend_falls_back_without_scipy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Large-matrix eigensolve falls back to dense torch when SciPy is blocked."""
    real_import = builtins.__import__

    def _block(
        name: str,
        globals: Any = None,
        locals: Any = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "scipy.sparse.linalg" or name.startswith("scipy.sparse.linalg"):
            raise ImportError("blocked")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _block)
    values = sparse_leading_eigenvalues(torch.eye(40), num_modes=2)
    assert values.numel() == 2
