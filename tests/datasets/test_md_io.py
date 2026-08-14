"""Coverage and error-path tests for :mod:`koopman_graph.datasets`."""

from __future__ import annotations

import builtins
import sys
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from koopman_graph.datasets.molecular.md_io import load_md_trajectory, require_mdtraj


def test_md_io_mocked_load_and_missing_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """mdtraj load is exercised via a stub module; missing extra is guided."""
    fake = ModuleType("mdtraj")

    class _Traj:
        xyz = np.zeros((3, 4, 3), dtype=np.float32)

    def _load(_path: str, **_kwargs: Any) -> _Traj:
        return _Traj()

    fake.load = _load  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mdtraj", fake)
    loaded = load_md_trajectory("unused.xtc")
    assert loaded.n_atoms == 4
    assert loaded.xyz.shape == (3, 4, 3)

    monkeypatch.delitem(sys.modules, "mdtraj", raising=False)
    real_import = builtins.__import__

    def _block_mdtraj(
        name: str,
        globals: Any = None,
        locals: Any = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "mdtraj" or name.startswith("mdtraj."):
            raise ImportError("blocked")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _block_mdtraj)
    with pytest.raises(ImportError, match="koopman-graph\\[md\\]"):
        require_mdtraj()
