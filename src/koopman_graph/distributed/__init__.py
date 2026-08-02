"""Distributed training adapters for KoopmanGraph.

Capability layout
-----------------
``process``
    Rank / world-size helpers and env-based process-group init
    (:func:`~koopman_graph.distributed.get_rank`,
    :func:`~koopman_graph.distributed.init_process_group_from_env`, …).
``seed``
    :func:`~koopman_graph.distributed.seed_everything` for per-rank RNG
    seeding.
``sampling``
    :class:`~koopman_graph.distributed.DistributedWindowSampler` for
    rank-sharded temporal windows (global ``windows_per_epoch`` cap, then
    shard); :func:`~koopman_graph.distributed.shard_sequences_for_rank` for
    full-trajectory multi-sequence sharding.
``ddp``
    :func:`~koopman_graph.distributed.prepare_ddp_model` and
    :func:`~koopman_graph.distributed.run_ddp_fit_loop` for native DDP /
    ``torchrun`` training.
``fabric``
    :func:`~koopman_graph.distributed.fit_with_fabric` (lazy Lightning
    Fabric import; shares the private epoch driver with DDP).
``lightning_module``
    :class:`~koopman_graph.distributed.KoopmanLightningModule` (optional
    Trainer sugar; lazy Lightning import).
``ray_jobs``
    :func:`~koopman_graph.distributed.fit_ensemble_with_ray` (optional
    parallel ensemble member fits; lazy Ray import).
``dask_prep``
    :func:`~koopman_graph.distributed.materialize_sequences` and
    :func:`~koopman_graph.distributed.materialize_window_index_list`
    (sole library Dask API; offline prep for fit / DDP; lazy Dask import).
    Not a Dask training loop.

Power-user module: import as ``koopman_graph.distributed``. Symbols are
intentionally omitted from root ``koopman_graph.__all__`` (see architecture
docs). ``process`` and ``seed`` stay free of Lightning, Ray, and Dask
imports.

Single-process defaults (no active process group): rank ``0``, world size
``1``, :func:`~koopman_graph.distributed.barrier` is a no-op; DDP wrapping
is skipped.
"""

from typing import Any

from koopman_graph.distributed.ddp import (
    all_reduce_mean,
    prepare_ddp_model,
    run_ddp_fit_loop,
    unwrap_model,
)
from koopman_graph.distributed.fabric import fit_with_fabric
from koopman_graph.distributed.process import (
    barrier,
    get_rank,
    get_world_size,
    init_process_group_from_env,
    is_main_process,
)
from koopman_graph.distributed.sampling import (
    DistributedWindowSampler,
    shard_sequences_for_rank,
)
from koopman_graph.distributed.seed import seed_everything

__all__ = [
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
    "seed_everything",
    "shard_sequences_for_rank",
    "unwrap_model",
]


def __getattr__(name: str) -> Any:
    """Lazy-load optional-extra symbols without importing Lightning / Ray / Dask.

    Parameters
    ----------
    name : str
        Attribute name requested on :mod:`koopman_graph.distributed`.

    Returns
    -------
    object
        Lazily imported public symbol.

    Raises
    ------
    AttributeError
        If ``name`` is not a known lazy export.
    """
    if name == "KoopmanLightningModule":
        from koopman_graph.distributed.lightning_module import KoopmanLightningModule

        return KoopmanLightningModule
    if name == "fit_ensemble_with_ray":
        from koopman_graph.distributed.ray_jobs import fit_ensemble_with_ray

        return fit_ensemble_with_ray
    if name == "materialize_sequences":
        from koopman_graph.distributed.dask_prep import materialize_sequences

        return materialize_sequences
    if name == "materialize_window_index_list":
        from koopman_graph.distributed.dask_prep import materialize_window_index_list

        return materialize_window_index_list
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
