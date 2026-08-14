"""Coverage and error-path tests for :mod:`koopman_graph.distributed`."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import Tensor
from torch_geometric.data import Data

from koopman_graph.data import (
    GraphSnapshotSequence,
)


def _path_edges(num_nodes: int = 4) -> Tensor:
    """Undirected path ``edge_index``."""
    src = list(range(num_nodes - 1)) + list(range(1, num_nodes))
    dst = list(range(1, num_nodes)) + list(range(num_nodes - 1))
    return torch.tensor([src, dst], dtype=torch.long)


def _tiny_sequence(*, num_nodes: int = 4, timesteps: int = 4) -> GraphSnapshotSequence:
    """Path-graph snapshot sequence."""
    edge_index = _path_edges(num_nodes)
    snapshots = [
        Data(x=torch.randn(num_nodes, 2), edge_index=edge_index)
        for _ in range(timesteps)
    ]
    return GraphSnapshotSequence(snapshots)


def test_dask_prep_with_stub_dask(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dask prep materializes eager sequences through a local compute stub."""
    import koopman_graph.distributed.dask_prep as dask_mod

    class _StubDask:
        @staticmethod
        def compute(*items: Any, scheduler: str | None = None) -> tuple[Any, ...]:
            del scheduler
            return tuple(items)

        @staticmethod
        def delayed(function: Any) -> Any:
            def _wrapped(*args: Any, **kwargs: Any) -> Any:
                return function(*args, **kwargs)

            return _wrapped

    monkeypatch.setattr(dask_mod, "_import_dask", lambda: _StubDask())
    sequence = _tiny_sequence(timesteps=4)
    materialized = dask_mod.materialize_sequences([sequence], scheduler="threads")
    assert materialized[0] is sequence
    with pytest.raises(ValueError, match="at least one"):
        dask_mod.materialize_sequences([])
    with pytest.raises(ValueError, match="window_length"):
        dask_mod.materialize_window_index_list([sequence], window_length=1)
    origins = dask_mod.materialize_window_index_list([sequence], window_length=2)
    assert origins
    short = _tiny_sequence(timesteps=2)
    with pytest.raises(ValueError, match="at least 3 snapshots"):
        dask_mod._window_origins_for_sequence(0, short, window_length=3)
