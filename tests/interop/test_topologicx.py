"""Coverage and error-path tests for :mod:`koopman_graph.interop`."""

from __future__ import annotations

import builtins
import sys
from types import ModuleType
from typing import Any

import pytest
import torch

from koopman_graph.interop.topologicx import (
    incidence_tensors_from_arrays,
    require_topologicx,
)


def test_topologicx_require_and_optional_b2(monkeypatch: pytest.MonkeyPatch) -> None:
    """TopologicX require-path is guided; ``B2`` is omitted when absent."""
    payload = incidence_tensors_from_arrays(torch.eye(2))
    assert "B2" not in payload
    monkeypatch.delitem(sys.modules, "topologicx", raising=False)
    real_import = builtins.__import__

    def _block(
        name: str,
        globals: Any = None,
        locals: Any = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "topologicx":
            raise ImportError("blocked")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _block)
    with pytest.raises(ImportError, match="koopman-graph\\[tdl\\]"):
        require_topologicx()
    fake = ModuleType("topologicx")
    monkeypatch.setattr(builtins, "__import__", real_import)
    monkeypatch.setitem(sys.modules, "topologicx", fake)
    assert require_topologicx() is fake


def test_topologicx_bridge_from_arrays() -> None:
    """Incidence bridge copies caller tensors without TopologicX."""
    b1 = torch.eye(3)
    payload = incidence_tensors_from_arrays(b1, boundary_1_2=torch.ones(2, 1))
    assert payload["B1"].shape == (3, 3)
    assert "B2" in payload
