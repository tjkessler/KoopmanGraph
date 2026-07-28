"""Process-group and rank helpers for distributed training.

Framework-agnostic wrappers around :mod:`torch.distributed`. When the
process group is unavailable or not initialized, helpers behave as a
single-process job (rank ``0``, world size ``1``, no-op barrier).

Backend default for :func:`init_process_group_from_env`:

* ``nccl`` when CUDA is available and the intended world size is greater
  than ``1``
* ``gloo`` otherwise (CPU / CI)

Multi-process gloo smoke coverage lives in
``tests/test_distributed_ddp.py`` (``@pytest.mark.distributed``, opt-in via
``KOOPMAN_GRAPH_DISTRIBUTED_TESTS=1``); this module's unit tests cover the
single-process defaults only.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

__all__ = [
    "barrier",
    "get_rank",
    "get_world_size",
    "init_process_group_from_env",
    "is_main_process",
]


def _is_initialized() -> bool:
    """Return whether a default process group is active.

    Returns
    -------
    bool
        ``True`` when :mod:`torch.distributed` is available and initialized.
    """
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Return the global rank, or ``0`` when distributed is inactive.

    Returns
    -------
    int
        Process rank in ``[0, world_size)``, or ``0`` if
        :mod:`torch.distributed` is unavailable or not initialized.
    """
    if not _is_initialized():
        return 0
    return int(dist.get_rank())


def get_world_size() -> int:
    """Return the world size, or ``1`` when distributed is inactive.

    Returns
    -------
    int
        Number of processes, or ``1`` if unavailable / not initialized.
    """
    if not _is_initialized():
        return 1
    return int(dist.get_world_size())


def is_main_process() -> bool:
    """Return ``True`` when this process should own logging / checkpoints.

    Returns
    -------
    bool
        ``True`` iff :func:`get_rank` is ``0``.
    """
    return get_rank() == 0


def barrier() -> None:
    """Synchronize all ranks, or no-op when distributed is inactive.

    Notes
    -----
    When no process group is active this function returns immediately.
    """
    if not _is_initialized():
        return
    dist.barrier()


def _env_world_size() -> int | None:
    """Parse ``WORLD_SIZE`` from the environment when present.

    Returns
    -------
    int or None
        Parsed world size, or ``None`` when the variable is unset or empty.
    """
    raw = os.environ.get("WORLD_SIZE")
    if raw is None or raw == "":
        return None
    return int(raw)


def _default_backend(*, world_size: int) -> str:
    """Choose ``nccl`` or ``gloo`` from CUDA visibility and world size.

    Parameters
    ----------
    world_size : int
        Intended process-group size from the environment.

    Returns
    -------
    str
        Backend name (``"nccl"`` or ``"gloo"``).
    """
    if world_size > 1 and torch.cuda.is_available():
        return "nccl"
    return "gloo"


def init_process_group_from_env(
    *,
    backend: str | None = None,
) -> dist.ProcessGroup | None:
    """Initialize the default process group from ``torchrun``-style env vars.

    Reads ``RANK``, ``WORLD_SIZE``, ``LOCAL_RANK``, ``MASTER_ADDR``, and
    ``MASTER_PORT`` when present (as set by ``torchrun``).

    Parameters
    ----------
    backend : str or None, optional
        Distributed backend. When ``None``, uses ``nccl`` if CUDA is
        available and world size ``> 1``, otherwise ``gloo``.

    Returns
    -------
    torch.distributed.ProcessGroup or None
        The default process group when initialized (or already
        initialized). ``None`` when env vars are absent and the group was
        not previously initialized (single-process no-op).

    Notes
    -----
    If a process group is already initialized, this function returns the
    existing default group without re-initializing.
    """
    if _is_initialized():
        return dist.group.WORLD

    world_size = _env_world_size()
    if world_size is None:
        return None

    resolved_backend = (
        backend if backend is not None else _default_backend(world_size=world_size)
    )
    dist.init_process_group(backend=resolved_backend)
    local_rank = os.environ.get("LOCAL_RANK")
    if (
        local_rank is not None
        and local_rank != ""
        and resolved_backend == "nccl"
        and torch.cuda.is_available()
    ):
        torch.cuda.set_device(int(local_rank))
    return dist.group.WORLD
