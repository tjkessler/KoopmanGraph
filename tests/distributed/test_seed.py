"""Coverage and error-path tests for :mod:`koopman_graph.distributed`."""

from __future__ import annotations

import builtins
from typing import Any

import pytest
import torch

from koopman_graph.distributed.seed import seed_everything


def test_seed_everything_numpy_missing_and_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Seeding skips NumPy when blocked and still touches the CUDA RNG API."""
    real_import = builtins.__import__

    def _block_numpy(
        name: str,
        globals: Any = None,
        locals: Any = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "numpy" or name.startswith("numpy."):
            raise ImportError("blocked numpy")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _block_numpy)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda _seed: None)
    assert seed_everything(3, rank=1) == 4
