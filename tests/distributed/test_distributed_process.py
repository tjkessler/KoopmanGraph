"""Tests for ``koopman_graph.distributed`` process and seed helpers.

Single-process defaults only. Multi-process gloo smoke tests belong in
TASK-1707 (``@pytest.mark.distributed``).
"""

from __future__ import annotations

import ast

import pytest
import torch
from tests.helpers import REPO_ROOT

import koopman_graph
from koopman_graph import distributed as distributed_pkg
from koopman_graph.distributed import (
    barrier,
    get_rank,
    get_world_size,
    init_process_group_from_env,
    is_main_process,
    seed_everything,
)

_SRC = REPO_ROOT / "src" / "koopman_graph" / "distributed"


def test_package_import_and_all() -> None:
    """Capability package exports the process/seed surface.

    Lazy optional-extra symbols stay listed in ``__all__`` but may raise
    ``ImportError`` on attribute access when Lightning / Ray are absent;
    ``hasattr`` is therefore only asserted for core (non-extra) exports.
    """
    expected = {
        "DistributedWindowSampler",
        "KoopmanLightningModule",
        "all_reduce_mean",
        "barrier",
        "fit_ensemble_with_ray",
        "fit_with_fabric",
        "get_rank",
        "get_world_size",
        "init_process_group_from_env",
        "is_main_process",
        "materialize_sequences",
        "materialize_window_index_list",
        "prepare_ddp_model",
        "run_ddp_fit_loop",
        "run_ray_train_fit_loop",
        "seed_everything",
        "shard_sequences_for_rank",
        "unwrap_model",
    }
    assert set(distributed_pkg.__all__) == expected
    optional_extra_exports = frozenset(
        {
            "KoopmanLightningModule",
            "fit_ensemble_with_ray",
            "materialize_sequences",
            "materialize_window_index_list",
            "run_ray_train_fit_loop",
        }
    )
    for name in expected - optional_extra_exports:
        assert hasattr(distributed_pkg, name)
    # Fabric helper is eagerly bound; Lightning/Ray remain lazy.
    assert hasattr(distributed_pkg, "fit_with_fabric")


def test_root_all_excludes_distributed_symbols() -> None:
    """Distributed helpers stay off the root stable ``__all__``."""
    exported = set(koopman_graph.__all__)
    assert "distributed" not in exported
    assert "get_rank" not in exported
    assert "seed_everything" not in exported
    assert "init_process_group_from_env" not in exported


def test_single_process_defaults() -> None:
    """Uninitialized distributed behaves as a one-process job."""
    assert get_rank() == 0
    assert get_world_size() == 1
    assert is_main_process() is True
    barrier()  # no-op


def test_init_process_group_from_env_noop_without_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Absent ``WORLD_SIZE`` leaves the process group uninitialized."""
    for key in (
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        monkeypatch.delenv(key, raising=False)
    assert init_process_group_from_env() is None
    assert get_rank() == 0
    assert get_world_size() == 1


def test_active_process_group_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Initialized helpers forward to ``torch.distributed`` rank/world/barrier."""
    import torch.distributed as dist

    import koopman_graph.distributed.process as process_mod

    monkeypatch.setattr(process_mod, "_is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda: 3)
    monkeypatch.setattr(dist, "get_world_size", lambda: 4)
    called: list[str] = []
    monkeypatch.setattr(dist, "barrier", lambda: called.append("barrier"))

    assert get_rank() == 3
    assert get_world_size() == 4
    assert is_main_process() is False
    barrier()
    assert called == ["barrier"]


def test_env_world_size_and_default_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """``WORLD_SIZE`` parsing and nccl/gloo backend selection."""
    import koopman_graph.distributed.process as process_mod

    monkeypatch.delenv("WORLD_SIZE", raising=False)
    assert process_mod._env_world_size() is None
    monkeypatch.setenv("WORLD_SIZE", "")
    assert process_mod._env_world_size() is None
    monkeypatch.setenv("WORLD_SIZE", "2")
    assert process_mod._env_world_size() == 2

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert process_mod._default_backend(world_size=2) == "nccl"
    assert process_mod._default_backend(world_size=1) == "gloo"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert process_mod._default_backend(world_size=8) == "gloo"


def test_init_process_group_from_env_already_initialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Already-initialized groups return ``dist.group.WORLD`` without re-init."""
    import types

    import torch.distributed as dist

    import koopman_graph.distributed.process as process_mod

    sentinel = object()
    monkeypatch.setattr(process_mod, "_is_initialized", lambda: True)
    monkeypatch.setattr(dist, "group", types.SimpleNamespace(WORLD=sentinel))
    assert init_process_group_from_env() is sentinel


def test_init_process_group_from_env_initializes_and_sets_cuda_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Env-driven init resolves backend and sets CUDA device for nccl."""
    import types

    import torch.distributed as dist

    import koopman_graph.distributed.process as process_mod

    monkeypatch.setattr(process_mod, "_is_initialized", lambda: False)
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    captured: dict[str, object] = {}

    def fake_init_process_group(*, backend: str) -> None:
        captured["backend"] = backend

    def fake_set_device(index: int) -> None:
        captured["device"] = index

    sentinel = object()
    monkeypatch.setattr(dist, "init_process_group", fake_init_process_group)
    monkeypatch.setattr(torch.cuda, "set_device", fake_set_device)
    monkeypatch.setattr(dist, "group", types.SimpleNamespace(WORLD=sentinel))

    assert init_process_group_from_env() is sentinel
    assert captured["backend"] == "nccl"
    assert captured["device"] == 1

    captured.clear()
    assert init_process_group_from_env(backend="gloo") is sentinel
    assert captured["backend"] == "gloo"
    assert "device" not in captured


def test_seed_everything_reproducible_and_rank_offset() -> None:
    """Effective seed is ``seed + rank`` and reseeds torch deterministically."""
    effective = seed_everything(123, rank=2)
    assert effective == 125
    first = torch.randn(4)
    seed_everything(123, rank=2)
    second = torch.randn(4)
    assert torch.equal(first, second)

    seed_everything(123, rank=0)
    other = torch.randn(4)
    assert not torch.equal(first, other)


def test_seed_everything_uses_get_rank_when_rank_omitted() -> None:
    """Default rank follows :func:`get_rank` (``0`` when inactive)."""
    assert seed_everything(7) == 7


def test_process_and_seed_modules_have_no_optional_framework_imports() -> None:
    """``process`` / ``seed`` must not import Lightning, Ray, or Dask."""
    forbidden = {"lightning", "ray", "dask"}
    for module_name in ("process.py", "seed.py"):
        source = (_SRC / module_name).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name.split(".", maxsplit=1)[0])
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported.add(node.module.split(".", maxsplit=1)[0])
        assert imported.isdisjoint(forbidden), (module_name, imported)
