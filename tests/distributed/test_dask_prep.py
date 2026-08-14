"""Tests for offline Dask prep helpers (TASK-1844)."""

from __future__ import annotations

import importlib

import pytest
import torch
from tests.helpers import REPO_ROOT
from torch_geometric.data import Data

from koopman_graph import distributed as distributed_pkg
from koopman_graph.data import GraphSnapshotSequence, build_window_index_list

_MODULE_PATH = REPO_ROOT / "src" / "koopman_graph" / "distributed" / "dask_prep.py"


def _tiny_sequence(*, timesteps: int = 5) -> GraphSnapshotSequence:
    """Deterministic path-graph trajectory for materialize smokes."""
    edge = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    return GraphSnapshotSequence(
        [Data(x=torch.ones(2, 1) * float(t), edge_index=edge) for t in range(timesteps)]
    )


def test_dask_prep_symbols_exported() -> None:
    """Dask prep helpers are listed on the public distributed API."""
    assert "materialize_sequences" in distributed_pkg.__all__
    assert "materialize_window_index_list" in distributed_pkg.__all__


def test_import_dask_missing_raises_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing Dask raises an actionable ``koopman-graph[dask]`` hint."""
    import koopman_graph.distributed.dask_prep as mod

    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None) -> object:
        if name == "dask" or name.startswith("dask."):
            raise ImportError("simulated missing dask")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    with pytest.raises(ImportError, match="koopman-graph\\[dask\\]"):
        mod._import_dask()


def test_lazy_getattr_does_not_require_dask_until_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolving the lazy export imports dask_prep but not Dask itself."""
    import koopman_graph.distributed.dask_prep as mod

    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None) -> object:
        if name == "dask" or name.startswith("dask."):
            raise ImportError("simulated missing dask")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    # Package surface resolves without calling Dask.
    assert callable(distributed_pkg.materialize_sequences)
    assert callable(distributed_pkg.materialize_window_index_list)
    with pytest.raises(ImportError, match="koopman-graph\\[dask\\]"):
        mod.materialize_sequences([_tiny_sequence()])


def test_materialize_sequences_round_trip() -> None:
    """Delayed sequences compute to equal in-memory trajectories."""
    dask = pytest.importorskip("dask")
    from koopman_graph.distributed.dask_prep import materialize_sequences

    seq_a = _tiny_sequence(timesteps=4)
    seq_b = _tiny_sequence(timesteps=5)
    delayed_parts = [
        dask.delayed(lambda s: s)(seq_a),
        dask.delayed(lambda s: s)(seq_b),
    ]
    materialized = materialize_sequences(delayed_parts, scheduler="threads")
    assert len(materialized) == 2
    assert materialized[0].num_timesteps == 4
    assert materialized[1].num_timesteps == 5
    assert torch.allclose(materialized[0][0].x, seq_a[0].x)
    assert torch.allclose(materialized[1][-1].x, seq_b[-1].x)


def test_materialize_window_index_matches_canonical() -> None:
    """Delayed window-index build matches ``build_window_index_list``."""
    pytest.importorskip("dask")
    from koopman_graph.distributed.dask_prep import materialize_window_index_list

    sequences = [_tiny_sequence(timesteps=5), _tiny_sequence(timesteps=4)]
    window_length = 3
    expected = build_window_index_list(sequences, window_length)
    got = materialize_window_index_list(
        sequences,
        window_length,
        scheduler="threads",
    )
    assert got == expected


def test_materialize_window_index_from_delayed_sequences() -> None:
    """Window-index path accepts delayed sequence inputs."""
    dask = pytest.importorskip("dask")
    from koopman_graph.distributed.dask_prep import materialize_window_index_list

    seq = _tiny_sequence(timesteps=5)
    delayed = [dask.delayed(lambda s: s)(seq)]
    expected = build_window_index_list([seq], window_length=3)
    got = materialize_window_index_list(delayed, window_length=3)
    assert got == expected


def test_materialize_sequences_rejects_empty() -> None:
    """Empty sequence list raises before ``dask.compute``."""
    pytest.importorskip("dask")
    from koopman_graph.distributed.dask_prep import materialize_sequences

    with pytest.raises(ValueError, match="at least one trajectory"):
        materialize_sequences([])


def test_materialize_window_index_rejects_bad_window_and_short_sequence() -> None:
    """``window_length < 2`` and short trajectories raise clearly."""
    pytest.importorskip("dask")
    from koopman_graph.distributed.dask_prep import (
        _window_origins_for_sequence,
        materialize_window_index_list,
    )

    with pytest.raises(ValueError, match="window_length must be >= 2"):
        materialize_window_index_list([_tiny_sequence()], window_length=1)

    short = _tiny_sequence(timesteps=2)
    with pytest.raises(ValueError, match="at least 3 snapshots"):
        materialize_window_index_list([short], window_length=3)

    with pytest.raises(ValueError, match="sequence 0 has 2"):
        _window_origins_for_sequence(0, short, window_length=3)


def test_window_origins_for_sequence_covers_valid_starts() -> None:
    """Per-sequence helper enumerates every valid start index."""
    from koopman_graph.distributed.dask_prep import _window_origins_for_sequence

    sequence = _tiny_sequence(timesteps=5)
    origins = _window_origins_for_sequence(2, sequence, window_length=3)
    assert [(o.sequence_index, o.start) for o in origins] == [
        (2, 0),
        (2, 1),
        (2, 2),
    ]


def test_dask_prep_honesty_docs() -> None:
    """Module docs state sole Dask API and not a training loop."""
    source = _MODULE_PATH.read_text(encoding="utf-8").lower()
    assert "sole" in source
    assert "training loop" in source
    assert "not" in source
    assert "run_fit_loop" in source or "fit" in source
