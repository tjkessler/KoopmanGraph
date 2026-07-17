"""Fixed-length temporal window sampling for multi-sequence training."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import torch

from koopman_graph.data.containers import GraphSnapshotSequence


class WindowSampler:
    """Sample fixed-length temporal windows from one or more trajectories.

    Parameters
    ----------
    sequences : GraphSnapshotSequence or sequence of GraphSnapshotSequence
        Source trajectories. Each must contain at least ``window_length``
        snapshots.
    window_length : int
        Number of snapshots per sampled window. Must be at least ``2``.
    batch_size : int, optional
        Number of windows yielded together. Default is ``8``.
    windows_per_epoch : int or None, optional
        Maximum number of windows sampled per epoch. ``None`` uses every valid
        window. Values larger than the available window count are capped.
    shuffle : bool, optional
        Randomize window order each epoch. Default is ``True``.
    seed : int or None, optional
        Base seed for reproducible epoch-specific shuffling.
    """

    def __init__(
        self,
        sequences: GraphSnapshotSequence | Sequence[GraphSnapshotSequence],
        *,
        window_length: int,
        batch_size: int = 8,
        windows_per_epoch: int | None = None,
        shuffle: bool = True,
        seed: int | None = None,
    ) -> None:
        """Initialize a fixed-length temporal window sampler.

        Parameters
        ----------
        sequences : GraphSnapshotSequence or sequence of GraphSnapshotSequence
            Source trajectories.
        window_length : int
            Number of snapshots per sampled window.
        batch_size : int, optional
            Number of windows yielded together. Default is ``8``.
        windows_per_epoch : int or None, optional
            Maximum sampled windows per epoch. ``None`` uses every window.
        shuffle : bool, optional
            Whether to randomize window order. Default is ``True``.
        seed : int or None, optional
            Base seed for reproducible epoch-specific shuffling.
        """
        if window_length < 2:
            msg = f"window_length must be >= 2, got {window_length}"
            raise ValueError(msg)
        if batch_size < 1:
            msg = f"batch_size must be >= 1, got {batch_size}"
            raise ValueError(msg)
        if windows_per_epoch is not None and windows_per_epoch < 1:
            msg = f"windows_per_epoch must be >= 1 when set, got {windows_per_epoch}"
            raise ValueError(msg)

        if isinstance(sequences, GraphSnapshotSequence):
            sequence_list = [sequences]
        else:
            sequence_list = list(sequences)
        if not sequence_list:
            msg = "sequences must contain at least one trajectory"
            raise ValueError(msg)

        short_lengths = [
            sequence.num_timesteps
            for sequence in sequence_list
            if sequence.num_timesteps < window_length
        ]
        if short_lengths:
            msg = (
                f"every sequence must contain at least {window_length} snapshots; "
                f"shortest has {min(short_lengths)}"
            )
            raise ValueError(msg)

        self.sequences = sequence_list
        self.window_length = window_length
        self.batch_size = batch_size
        self.windows_per_epoch = windows_per_epoch
        self.shuffle = shuffle
        self.seed = seed
        self._origins = [
            (sequence_index, start)
            for sequence_index, sequence in enumerate(sequence_list)
            for start in range(sequence.num_timesteps - window_length + 1)
        ]

    @property
    def num_windows(self) -> int:
        """Return the total number of valid windows.

        Returns
        -------
        int
            Number of valid windows across every source trajectory.
        """
        return len(self._origins)

    def iter_epoch(
        self,
        epoch: int = 0,
    ) -> Iterator[list[GraphSnapshotSequence]]:
        """Yield batches of windows for one epoch.

        Parameters
        ----------
        epoch : int, optional
            Zero-based epoch index mixed into ``seed``. Default is ``0``.

        Yields
        ------
        list of GraphSnapshotSequence
            A batch containing at most ``batch_size`` temporal windows.
        """
        if epoch < 0:
            msg = f"epoch must be >= 0, got {epoch}"
            raise ValueError(msg)

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
        selected = indices[:limit]
        for offset in range(0, len(selected), self.batch_size):
            batch = []
            for origin_index in selected[offset : offset + self.batch_size]:
                sequence_index, start = self._origins[origin_index]
                batch.append(
                    self.sequences[sequence_index].slice(
                        start,
                        start + self.window_length,
                    )
                )
            yield batch

    def __iter__(self) -> Iterator[list[GraphSnapshotSequence]]:
        """Yield the epoch-zero batch sequence.

        Yields
        ------
        list of GraphSnapshotSequence
            A batch of fixed-length temporal windows.
        """
        return self.iter_epoch(0)
