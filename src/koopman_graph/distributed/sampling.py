"""Rank-aware sampling helpers for data-parallel training.

``DistributedWindowSampler`` partitions the global window-origin list so each
rank trains on a disjoint shard. ``windows_per_epoch`` is applied as a
**global** cap before sharding (same meaning as
:class:`~koopman_graph.data.WindowSampler` when ``world_size == 1``).

:func:`shard_sequences_for_rank` shards full trajectories for non-windowed
multi-trajectory fit (raises when there are fewer trajectories than ranks).

Both helpers accept homogeneous
:class:`~koopman_graph.data.GraphSnapshotSequence` and multiplex
:class:`~koopman_graph.data.HeteroGraphSnapshotSequence` trajectories.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import torch

from koopman_graph.data import SnapshotSequence
from koopman_graph.data.sampling import (
    WindowOrigin,
    as_trajectory_list,
    build_window_index_list,
)
from koopman_graph.distributed.process import get_rank, get_world_size

__all__ = [
    "DistributedWindowSampler",
    "shard_sequences_for_rank",
]


def shard_sequences_for_rank(
    sequences: SnapshotSequence | Sequence[SnapshotSequence],
    *,
    rank: int | None = None,
    world_size: int | None = None,
) -> list[SnapshotSequence]:
    """Return this rank's shard of full trajectories for non-windowed fit.

    Partitions with ``sequences[rank::world_size]``. Use
    :class:`DistributedWindowSampler` when training on temporal windows
    instead of whole trajectories.

    Parameters
    ----------
    sequences : GraphSnapshotSequence or HeteroGraphSnapshotSequence or sequence
        Training trajectories to shard (including
        :class:`~koopman_graph.data.MultiTrajectory`).
    rank : int or None, optional
        Process rank. Defaults to
        :func:`~koopman_graph.distributed.get_rank`.
    world_size : int or None, optional
        Number of processes. Defaults to
        :func:`~koopman_graph.distributed.get_world_size`.

    Returns
    -------
    list of GraphSnapshotSequence or HeteroGraphSnapshotSequence
        Trajectories assigned to ``rank``.

    Raises
    ------
    ValueError
        If ``sequences`` is empty, ``rank`` / ``world_size`` are invalid, or
        ``len(sequences) < world_size`` (full-sequence DDP needs at least one
        trajectory per rank; prefer windowed sampling or fewer ranks).
    """
    sequence_list = as_trajectory_list(sequences)

    resolved_rank = get_rank() if rank is None else int(rank)
    resolved_world = get_world_size() if world_size is None else int(world_size)
    if resolved_rank < 0:
        msg = f"rank must be >= 0, got {resolved_rank}"
        raise ValueError(msg)
    if resolved_world < 1:
        msg = f"world_size must be >= 1, got {resolved_world}"
        raise ValueError(msg)
    if resolved_rank >= resolved_world:
        msg = (
            f"rank must be < world_size, got rank={resolved_rank}, "
            f"world_size={resolved_world}"
        )
        raise ValueError(msg)
    if not sequence_list:
        msg = "sequences must contain at least one trajectory"
        raise ValueError(msg)
    if len(sequence_list) < resolved_world:
        msg = (
            f"full-sequence distributed fit requires at least world_size "
            f"trajectories (got {len(sequence_list)} < {resolved_world}); "
            "use DistributedWindowSampler for windowed training, or reduce "
            "the number of ranks"
        )
        raise ValueError(msg)
    return sequence_list[resolved_rank::resolved_world]


class DistributedWindowSampler:
    """Sample fixed-length temporal windows on a per-rank shard.

    Parameters
    ----------
    sequences : GraphSnapshotSequence or HeteroGraphSnapshotSequence or sequence
        Source trajectories. Each must contain at least ``window_length``
        snapshots.
    window_length : int
        Number of snapshots per sampled window. Must be at least ``2``.
    batch_size : int, optional
        Number of windows yielded together. Default is ``8``.
    windows_per_epoch : int or None, optional
        Global cap on windows per epoch **before** rank sharding. ``None``
        uses every valid window. Each rank then receives
        ``selected[rank::world_size]``.
    shuffle : bool, optional
        Randomize the global window order each epoch (identical permutation
        on every rank when ``seed`` is set). Default is ``True``.
    seed : int or None, optional
        Base seed for the global shuffle (``seed + epoch``). Required when
        ``shuffle=True`` and ``world_size > 1`` so ranks agree on the
        permutation before sharding.
    rank : int or None, optional
        Process rank. Defaults to
        :func:`~koopman_graph.distributed.get_rank`.
    world_size : int or None, optional
        Number of processes. Defaults to
        :func:`~koopman_graph.distributed.get_world_size`.

    Notes
    -----
    Global shuffle uses ``seed + epoch`` only (not ``seed + rank``) so every
    rank builds the same ordered list before ``indices[rank::world_size]``.
    """

    def __init__(
        self,
        sequences: SnapshotSequence | Sequence[SnapshotSequence],
        *,
        window_length: int,
        batch_size: int = 8,
        windows_per_epoch: int | None = None,
        shuffle: bool = True,
        seed: int | None = None,
        rank: int | None = None,
        world_size: int | None = None,
    ) -> None:
        """Initialize a rank-sharded temporal window sampler.

        Parameters
        ----------
        sequences : GraphSnapshotSequence or HeteroGraphSnapshotSequence or sequence
            Source trajectories.
        window_length : int
            Number of snapshots per sampled window.
        batch_size : int, optional
            Windows per yielded batch. Default is ``8``.
        windows_per_epoch : int or None, optional
            Global cap before sharding.
        shuffle : bool, optional
            Whether to shuffle the global origin list. Default is ``True``.
        seed : int or None, optional
            Base seed for the global shuffle.
        rank : int or None, optional
            Override for the process rank.
        world_size : int or None, optional
            Override for the world size.

        Raises
        ------
        ValueError
            If configuration is invalid or ``shuffle=True`` with
            ``world_size > 1`` and ``seed is None``.
        """
        if batch_size < 1:
            msg = f"batch_size must be >= 1, got {batch_size}"
            raise ValueError(msg)
        if windows_per_epoch is not None and windows_per_epoch < 1:
            msg = f"windows_per_epoch must be >= 1 when set, got {windows_per_epoch}"
            raise ValueError(msg)

        sequence_list = as_trajectory_list(sequences)

        resolved_rank = get_rank() if rank is None else int(rank)
        resolved_world = get_world_size() if world_size is None else int(world_size)
        if resolved_rank < 0:
            msg = f"rank must be >= 0, got {resolved_rank}"
            raise ValueError(msg)
        if resolved_world < 1:
            msg = f"world_size must be >= 1, got {resolved_world}"
            raise ValueError(msg)
        if resolved_rank >= resolved_world:
            msg = (
                f"rank must be < world_size, got rank={resolved_rank}, "
                f"world_size={resolved_world}"
            )
            raise ValueError(msg)
        if shuffle and resolved_world > 1 and seed is None:
            msg = (
                "DistributedWindowSampler requires seed when shuffle=True "
                "and world_size > 1 so ranks share one global permutation"
            )
            raise ValueError(msg)

        self.sequences = sequence_list
        self.window_length = window_length
        self.batch_size = batch_size
        self.windows_per_epoch = windows_per_epoch
        self.shuffle = shuffle
        self.seed = seed
        self.rank = resolved_rank
        self.world_size = resolved_world
        self._origins: list[WindowOrigin] = build_window_index_list(
            sequence_list,
            window_length,
        )

    @property
    def num_windows(self) -> int:
        """Return the global number of valid windows (before sharding).

        Returns
        -------
        int
            Total valid temporal origins across all source trajectories.
        """
        return len(self._origins)

    def _global_selected_indices(self, epoch: int) -> list[int]:
        """Return the globally capped (and optionally shuffled) origin indices.

        Parameters
        ----------
        epoch : int
            Zero-based epoch index.

        Returns
        -------
        list of int
            Indices into ``_origins`` after shuffle and global cap.
        """
        indices = list(range(self.num_windows))
        if self.shuffle:
            generator = None
            if self.seed is not None:
                generator = torch.Generator()
                generator.manual_seed(self.seed + epoch)
            indices = torch.randperm(
                self.num_windows,
                generator=generator,
            ).tolist()
        limit = (
            self.num_windows
            if self.windows_per_epoch is None
            else min(self.windows_per_epoch, self.num_windows)
        )
        return indices[:limit]

    def rank_origin_indices(self, epoch: int = 0) -> list[int]:
        """Return this rank's shard of global origin indices for ``epoch``.

        Parameters
        ----------
        epoch : int, optional
            Zero-based epoch index. Default is ``0``.

        Returns
        -------
        list of int
            Indices into the global origin list assigned to this rank.
        """
        if epoch < 0:
            msg = f"epoch must be >= 0, got {epoch}"
            raise ValueError(msg)
        selected = self._global_selected_indices(epoch)
        return selected[self.rank :: self.world_size]

    def iter_epoch(
        self,
        epoch: int = 0,
    ) -> Iterator[list[SnapshotSequence]]:
        """Yield batches of windows for one epoch on this rank's shard.

        Parameters
        ----------
        epoch : int, optional
            Zero-based epoch index mixed into ``seed``. Default is ``0``.

        Yields
        ------
        list of GraphSnapshotSequence or HeteroGraphSnapshotSequence
            A batch containing at most ``batch_size`` temporal windows.
        """
        shard = self.rank_origin_indices(epoch)
        for offset in range(0, len(shard), self.batch_size):
            batch = []
            for origin_index in shard[offset : offset + self.batch_size]:
                origin = self._origins[origin_index]
                batch.append(
                    self.sequences[origin.sequence_index].slice(
                        origin.start,
                        origin.start + self.window_length,
                    )
                )
            yield batch

    def __iter__(self) -> Iterator[list[SnapshotSequence]]:
        """Yield the epoch-zero batch sequence for this rank.

        Yields
        ------
        list of GraphSnapshotSequence or HeteroGraphSnapshotSequence
            A batch of fixed-length temporal windows.
        """
        return self.iter_epoch(0)
