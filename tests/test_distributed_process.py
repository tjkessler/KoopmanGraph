"""Tests for ``koopman_graph.distributed`` process and seed helpers.

Single-process defaults only. Multi-process gloo smoke tests belong in
TASK-1707 (``@pytest.mark.distributed``).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

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

_SRC = Path(__file__).resolve().parents[1] / "src" / "koopman_graph" / "distributed"


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
        "prepare_ddp_model",
        "run_ddp_fit_loop",
        "seed_everything",
        "shard_sequences_for_rank",
        "unwrap_model",
    }
    assert set(distributed_pkg.__all__) == expected
    optional_extra_exports = frozenset(
        {"KoopmanLightningModule", "fit_ensemble_with_ray"}
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
