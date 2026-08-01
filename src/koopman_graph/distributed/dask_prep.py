"""Offline Dask helpers for sequence / window-index materialization.

This module is the **sole** library Dask API (Q6). It materializes delayed
partitions into in-memory :class:`~koopman_graph.data.SnapshotSequence`
lists and window origins for hand-off to
:func:`~koopman_graph.training.run_fit_loop`,
:func:`~koopman_graph.distributed.run_ddp_fit_loop`, or
:class:`~koopman_graph.distributed.DistributedWindowSampler`.

It is **not** a Dask gradient training loop and must not replace the PyTorch
fit / DDP / Fabric paths. Dask is imported lazily so
:mod:`koopman_graph.distributed` stays importable without the ``[dask]``
extra until this submodule is used.

Example
-------
>>> import dask  # doctest: +SKIP
>>> from koopman_graph.distributed import materialize_sequences
>>> delayed_seqs = [dask.delayed(load)(p) for p in paths]  # doctest: +SKIP
>>> sequences = materialize_sequences(delayed_seqs)  # doctest: +SKIP
>>> model.fit(sequences, epochs=10)  # doctest: +SKIP
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any

from koopman_graph.data import SnapshotSequence, WindowOrigin, build_window_index_list

__all__ = [
    "materialize_sequences",
    "materialize_window_index_list",
]

_DASK_INSTALL_HINT = 'pip install "koopman-graph[dask]"'


def _import_dask() -> Any:
    """Import the ``dask`` package with an actionable install hint.

    Returns
    -------
    module
        The ``dask`` package.

    Raises
    ------
    ImportError
        If Dask is not installed.
    """
    try:
        return importlib.import_module("dask")
    except ImportError as exc:
        msg = (
            "Dask is required for materialize_sequences / "
            "materialize_window_index_list; "
            f"install with: {_DASK_INSTALL_HINT}"
        )
        raise ImportError(msg) from exc


def _window_origins_for_sequence(
    sequence_index: int,
    sequence: SnapshotSequence,
    window_length: int,
) -> list[WindowOrigin]:
    """Build window origins for one trajectory with a fixed global index.

    Parameters
    ----------
    sequence_index
        Value for ``sequence_index``.
    sequence
        Value for ``sequence``.
    window_length
        Value for ``window_length``.

    Returns
    -------
    object
        Function result.
    """
    if sequence.num_timesteps < window_length:
        msg = (
            f"every sequence must contain at least {window_length} snapshots; "
            f"sequence {sequence_index} has {sequence.num_timesteps}"
        )
        raise ValueError(msg)
    return [
        WindowOrigin(sequence_index, start)
        for start in range(sequence.num_timesteps - window_length + 1)
    ]


def materialize_sequences(
    sequences: Sequence[Any],
    *,
    scheduler: str | None = "threads",
) -> list[SnapshotSequence]:
    """Compute delayed (or eager) sequences into an in-memory list.

    Offline prep only: pass the result to ``run_fit_loop``,
    ``run_ddp_fit_loop``, or ``DistributedWindowSampler``. This is not a
    Dask training loop.

    Parameters
    ----------
    sequences : sequence
        Delayed objects and/or already-materialized
        :class:`~koopman_graph.data.SnapshotSequence` trajectories.
    scheduler : str or None, optional
        Dask scheduler name forwarded to ``dask.compute`` (default
        ``\"threads\"``).

    Returns
    -------
    list of SnapshotSequence
        Computed trajectories in input order.

    Raises
    ------
    ImportError
        If the ``[dask]`` extra is not installed.
    ValueError
        If ``sequences`` is empty.
    """
    dask = _import_dask()
    sequence_list = list(sequences)
    if not sequence_list:
        msg = "sequences must contain at least one trajectory"
        raise ValueError(msg)
    computed = dask.compute(*sequence_list, scheduler=scheduler)
    return list(computed)


def materialize_window_index_list(
    sequences: Sequence[Any],
    window_length: int,
    *,
    scheduler: str | None = "threads",
) -> list[WindowOrigin]:
    """Build window origins with a delayed per-sequence task graph.

    Materializes ``sequences`` first (if delayed), then schedules one delayed
    origin-build task per trajectory. The flattened result matches
    :func:`~koopman_graph.data.build_window_index_list` on the same inputs.

    Parameters
    ----------
    sequences : sequence
        Delayed or eager snapshot sequences.
    window_length : int
        Number of snapshots per window. Must be at least ``2``.
    scheduler : str or None, optional
        Dask scheduler name (default ``\"threads\"``).

    Returns
    -------
    list of WindowOrigin
        Every valid ``(sequence_index, start)`` pair in trajectory order.

    Raises
    ------
    ImportError
        If the ``[dask]`` extra is not installed.
    ValueError
        If ``window_length < 2``, ``sequences`` is empty, or any trajectory
        is shorter than ``window_length``.
    """
    dask = _import_dask()
    if window_length < 2:
        msg = f"window_length must be >= 2, got {window_length}"
        raise ValueError(msg)
    materialized = materialize_sequences(sequences, scheduler=scheduler)
    # Eager validation for empty / short trajectories (matches sampling errors).
    build_window_index_list(materialized, window_length)
    tasks = [
        dask.delayed(_window_origins_for_sequence)(index, sequence, window_length)
        for index, sequence in enumerate(materialized)
    ]
    parts = dask.compute(*tasks, scheduler=scheduler)
    return [origin for part in parts for origin in part]
