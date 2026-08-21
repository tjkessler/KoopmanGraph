"""Temporal and neighbor-subgraph window sampling for training.

``WindowSampler`` yields fixed-length temporal windows over full graphs.
``NeighborWindowSampler`` adds PyG ``k_hop_subgraph`` spatial sampling so
large static graphs can train on subgraph windows. Sampled-subgraph training
is an approximation (loss over induced subgraphs, not the full graph);
``predict`` / ``evaluate`` remain full-graph.

:func:`resolve_window_sampler` is the shared construction entry point for
single-process ``run_fit_loop`` and distributed fit adapters so sampler
policy cannot drift between paths.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator, Sequence
from typing import Any, NamedTuple

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

from koopman_graph.data.containers import (
    GraphSnapshotSequence,
    HeteroGraphSnapshotSequence,
)
from koopman_graph.data.trajectories import SnapshotSequence
from koopman_graph.graph_utils import snapshot_edge_weight


class WindowOrigin(NamedTuple):
    """Index of one valid fixed-length temporal window.

    Notes
    -----
    ``sequence_index`` indexes the source trajectory list; ``start`` is the
    inclusive start timestep of the window within that trajectory.
    """

    sequence_index: int
    start: int


def as_trajectory_list(
    sequences: SnapshotSequence | Sequence[SnapshotSequence],
) -> list[SnapshotSequence]:
    """Normalize a single trajectory or sequence of trajectories to a list.

    A lone :class:`~koopman_graph.data.GraphSnapshotSequence` or
    :class:`~koopman_graph.data.HeteroGraphSnapshotSequence` is wrapped as
    one element (not expanded via snapshot iteration).

    Parameters
    ----------
    sequences : GraphSnapshotSequence or HeteroGraphSnapshotSequence or sequence
        One trajectory or an ordered collection of trajectories (including
        :class:`~koopman_graph.data.MultiTrajectory`).

    Returns
    -------
    list of GraphSnapshotSequence or HeteroGraphSnapshotSequence
        Trajectory list suitable for window indexing / sharding.
    """
    if isinstance(sequences, (GraphSnapshotSequence, HeteroGraphSnapshotSequence)):
        return [sequences]
    return list(sequences)


def build_window_index_list(
    sequences: Sequence[SnapshotSequence],
    window_length: int,
) -> list[WindowOrigin]:
    """Build the list of valid temporal window origins for ``sequences``.

    Parameters
    ----------
    sequences : sequence of GraphSnapshotSequence or HeteroGraphSnapshotSequence
        Source trajectories. Each must contain at least ``window_length``
        snapshots.
    window_length : int
        Number of snapshots per window. Must be at least ``2``.

    Returns
    -------
    list of WindowOrigin
        Every valid ``(sequence_index, start)`` pair in trajectory order.

    Raises
    ------
    ValueError
        If ``window_length < 2``, ``sequences`` is empty, or any trajectory
        is shorter than ``window_length``.
    """
    if window_length < 2:
        msg = f"window_length must be >= 2, got {window_length}"
        raise ValueError(msg)
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
    return [
        WindowOrigin(sequence_index, start)
        for sequence_index, sequence in enumerate(sequence_list)
        for start in range(sequence.num_timesteps - window_length + 1)
    ]


class WindowSampler:
    """Sample fixed-length temporal windows from one or more trajectories.

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
        Maximum number of windows sampled per epoch. ``None`` uses every valid
        window. Values larger than the available window count are capped.
    shuffle : bool, optional
        Randomize window order each epoch. Default is ``True``.
    seed : int or None, optional
        Base seed for reproducible epoch-specific shuffling.
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
    ) -> None:
        """Initialize a fixed-length temporal window sampler.

        Parameters
        ----------
        sequences : GraphSnapshotSequence or HeteroGraphSnapshotSequence or sequence
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
        if batch_size < 1:
            msg = f"batch_size must be >= 1, got {batch_size}"
            raise ValueError(msg)
        if windows_per_epoch is not None and windows_per_epoch < 1:
            msg = f"windows_per_epoch must be >= 1 when set, got {windows_per_epoch}"
            raise ValueError(msg)

        sequence_list = as_trajectory_list(sequences)

        self.sequences = sequence_list
        self.window_length = window_length
        self.batch_size = batch_size
        self.windows_per_epoch = windows_per_epoch
        self.shuffle = shuffle
        self.seed = seed
        self._origins = build_window_index_list(sequence_list, window_length)

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
                origin = self._origins[origin_index]
                batch.append(
                    self.sequences[origin.sequence_index].slice(
                        origin.start,
                        origin.start + self.window_length,
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


def _require_static_pairwise_graph(sequence: SnapshotSequence) -> None:
    """Reject sequences unsupported by neighbor-subgraph sampling.

    Parameters
    ----------
    sequence : GraphSnapshotSequence or HeteroGraphSnapshotSequence
        Candidate source trajectory.

    Returns
    -------
    None
        Nothing; raises on unsupported inputs.

    Raises
    ------
    ValueError
        If ``sequence`` is multiplex / typed hetero, dynamic-topology, or
        hyperedge-carrying.
    """
    if isinstance(sequence, HeteroGraphSnapshotSequence):
        msg = (
            "NeighborWindowSampler does not support HeteroGraphSnapshotSequence; "
            "use homogeneous GraphSnapshotSequence trajectories"
        )
        raise ValueError(msg)
    if sequence.is_dynamic_topology:
        msg = (
            "NeighborWindowSampler requires static pairwise topology; "
            "dynamic-topology sequences are not supported"
        )
        raise ValueError(msg)
    if sequence.has_hyperedges:
        msg = "NeighborWindowSampler does not support hyperedge-carrying sequences"
        raise ValueError(msg)


def induce_neighbor_subgraph_sequence(
    sequence: GraphSnapshotSequence,
    *,
    seed_nodes: Tensor,
    num_hops: int,
) -> GraphSnapshotSequence:
    """Return a remapped k-hop subgraph sequence for a static graph window.

    Parameters
    ----------
    sequence : GraphSnapshotSequence
        Temporal window with static pairwise topology (no hyperedges).
    seed_nodes : Tensor
        Seed node indices in the original graph (1D long tensor).
    num_hops : int
        Number of hops for :func:`~torch_geometric.utils.k_hop_subgraph`.

    Returns
    -------
    GraphSnapshotSequence
        Window with remapped node indices ``0 .. n_sub - 1`` and induced edges.

    Raises
    ------
    ValueError
        If the sequence has dynamic topology or hyperedges.
    """
    _require_static_pairwise_graph(sequence)
    num_nodes = sequence.num_nodes
    edge_index = sequence.edge_index
    subset, sub_edge_index, _, edge_mask = k_hop_subgraph(
        seed_nodes,
        num_hops,
        edge_index,
        relabel_nodes=True,
        num_nodes=num_nodes,
    )

    snapshots: list[Data] = []
    for timestep in range(sequence.num_timesteps):
        snapshot = sequence[timestep]
        fields: dict[str, Tensor] = {
            "x": snapshot.x[subset],
            "edge_index": sub_edge_index,
        }
        weight = snapshot_edge_weight(snapshot)
        if weight is not None:
            fields["edge_weight"] = weight[edge_mask]
        snapshots.append(Data(**fields))

    control_inputs = sequence.control_inputs
    if control_inputs is not None and control_inputs.ndim == 3:
        control_inputs = control_inputs[:, subset, :]

    observation_masks = sequence.observation_masks
    if observation_masks is not None:
        observation_masks = observation_masks[:, subset]

    presence_masks = sequence.presence_masks
    if presence_masks is not None:
        presence_masks = presence_masks[:, subset]

    entity_ids = None
    if sequence.entity_ids is not None:
        entity_ids = tuple(sequence.entity_ids[int(i)] for i in subset.tolist())

    return GraphSnapshotSequence(
        snapshots,
        control_inputs=control_inputs,
        timestamps=sequence.timestamps,
        parameter_trajectory=sequence.parameter_trajectory,
        observation_masks=observation_masks,
        presence_masks=presence_masks,
        entity_ids=entity_ids,
        allow_node_churn=sequence.allow_node_churn,
    )


class NeighborWindowSampler:
    """Sample temporal windows on k-hop subgraphs of a static large graph.

    Combines :class:`WindowSampler`-style temporal origins with PyG
    :func:`~torch_geometric.utils.k_hop_subgraph` spatial sampling. Each
    yielded window uses a freshly sampled seed set (seeded by ``seed`` and
    epoch / window index). Training losses therefore see **sampled** topology;
    eigenvalue regularization on graph operators is likewise an approximation
    over the induced subgraph. Full-graph ``predict`` / ``evaluate`` are
    unchanged.

    Parameters
    ----------
    sequences : GraphSnapshotSequence or sequence of GraphSnapshotSequence
        Source trajectories (static pairwise topology only).
    window_length : int
        Number of snapshots per temporal window (``>= 2``).
    num_nodes : int
        Number of seed nodes sampled per window (before hop expansion).
    num_hops : int, optional
        Hop radius for subgraph expansion. Default is ``2``.
    batch_size : int, optional
        Windows per yielded batch. Default is ``8``.
    windows_per_epoch : int or None, optional
        Cap on windows per epoch. ``None`` uses every valid temporal origin.
    shuffle : bool, optional
        Shuffle temporal origins each epoch. Default is ``True``.
    seed : int or None, optional
        Base seed for origin shuffle and seed-node sampling.
    """

    def __init__(
        self,
        sequences: SnapshotSequence | Sequence[SnapshotSequence],
        *,
        window_length: int,
        num_nodes: int,
        num_hops: int = 2,
        batch_size: int = 8,
        windows_per_epoch: int | None = None,
        shuffle: bool = True,
        seed: int | None = None,
    ) -> None:
        """Initialize a neighbor-subgraph temporal window sampler.

        Parameters
        ----------
        sequences : GraphSnapshotSequence or sequence of GraphSnapshotSequence
            Homogeneous source trajectories (hetero rejected).
        window_length : int
            See signature.
        num_nodes : int
            See signature.
        num_hops : int
            See signature.
        batch_size : int
            See signature.
        windows_per_epoch : int or None
            See signature.
        shuffle : bool
            See signature.
        seed : int or None
            See signature.

        Returns
        -------
        None
            Nothing.

        Raises
        ------
        ValueError
            If configuration is invalid or any trajectory is hetero /
            dynamic / hyperedge-carrying.
        """
        if num_nodes < 1:
            msg = f"num_nodes must be >= 1, got {num_nodes}"
            raise ValueError(msg)
        if num_hops < 0:
            msg = f"num_hops must be >= 0, got {num_hops}"
            raise ValueError(msg)

        sequence_list = as_trajectory_list(sequences)
        for sequence in sequence_list:
            _require_static_pairwise_graph(sequence)

        self._temporal = WindowSampler(
            sequence_list,
            window_length=window_length,
            batch_size=batch_size,
            windows_per_epoch=windows_per_epoch,
            shuffle=shuffle,
            seed=seed,
        )

        self.num_nodes = num_nodes
        self.num_hops = num_hops

    @property
    def sequences(self) -> list[GraphSnapshotSequence]:
        """Source trajectories (shared with the inner temporal sampler).

        Returns
        -------
        list[GraphSnapshotSequence]
            See summary line."""
        return self._temporal.sequences

    @sequences.setter
    def sequences(self, value: list[GraphSnapshotSequence]) -> None:
        """Replace source trajectories after device placement.

        Parameters
        ----------

        value : list[GraphSnapshotSequence]
            See the function signature / summary for ``value``.

        Returns
        -------

        None
            See summary line."""
        self._temporal.sequences = value

    @property
    def window_length(self) -> int:
        """Temporal window length.

        Returns
        -------
        int
            See summary line."""
        return self._temporal.window_length

    @property
    def batch_size(self) -> int:
        """Windows per yielded batch.

        Returns
        -------
        int
            See summary line."""
        return self._temporal.batch_size

    @property
    def windows_per_epoch(self) -> int | None:
        """Optional cap on windows per epoch.

        Returns
        -------
        int | None
            See summary line."""
        return self._temporal.windows_per_epoch

    @property
    def shuffle(self) -> bool:
        """Whether temporal origins are shuffled each epoch.

        Returns
        -------
        bool
            See summary line."""
        return self._temporal.shuffle

    @property
    def seed(self) -> int | None:
        """Base RNG seed.

        Returns
        -------
        int | None
            See summary line."""
        return self._temporal.seed

    @property
    def num_windows(self) -> int:
        """Return the number of valid temporal window origins.

        Returns
        -------
        int
            See summary line."""
        return self._temporal.num_windows

    def _seed_generator(self, *, epoch: int, window_index: int) -> torch.Generator:
        """Build a generator for one window's seed-node draw.

        Parameters
        ----------

        epoch : int
            See the function signature / summary for ``epoch``.
        window_index : int
            See the function signature / summary for ``window_index``.

        Returns
        -------

        torch.Generator
            See summary line."""
        generator = torch.Generator()
        base = 0 if self.seed is None else self.seed
        # Mix epoch and window index; keep within signed 64-bit range.
        mixed = (base + 1_000_003 * epoch + 9_701 * window_index) % (2**63 - 1)
        generator.manual_seed(int(mixed))
        return generator

    def _subgraph_window(
        self,
        window: GraphSnapshotSequence,
        *,
        epoch: int,
        window_index: int,
    ) -> GraphSnapshotSequence:
        """Sample seeds and induce a k-hop subgraph for one temporal window.

        Parameters
        ----------

        window : GraphSnapshotSequence
            See the function signature / summary for ``window``.
        epoch : int
            See the function signature / summary for ``epoch``.
        window_index : int
            See the function signature / summary for ``window_index``.

        Returns
        -------

        GraphSnapshotSequence
            See summary line."""
        graph_nodes = window.num_nodes
        n_seeds = min(self.num_nodes, graph_nodes)
        generator = self._seed_generator(epoch=epoch, window_index=window_index)
        seed_nodes = torch.randperm(graph_nodes, generator=generator)[:n_seeds]
        return induce_neighbor_subgraph_sequence(
            window,
            seed_nodes=seed_nodes,
            num_hops=self.num_hops,
        )

    def iter_epoch(
        self,
        epoch: int = 0,
    ) -> Iterator[list[GraphSnapshotSequence]]:
        """Yield batches of neighbor-sampled temporal windows.

        Parameters
        ----------
        epoch : int, optional
            Zero-based epoch index. Default is ``0``.

        Yields
        ------
        list of GraphSnapshotSequence
            Batch of subgraph windows with remapped node indices.
        """
        window_index = 0
        for batch in self._temporal.iter_epoch(epoch):
            sampled_batch = [
                self._subgraph_window(
                    window,
                    epoch=epoch,
                    window_index=window_index + offset,
                )
                for offset, window in enumerate(batch)
            ]
            window_index += len(batch)
            yield sampled_batch

    def __iter__(self) -> Iterator[list[GraphSnapshotSequence]]:
        """Yield the epoch-zero batch sequence.

        Returns
        -------
        Iterator[list[GraphSnapshotSequence]]
            See summary line."""
        return self.iter_epoch(0)


WindowLikeSampler = WindowSampler | NeighborWindowSampler


def resolve_window_sampler(
    train_sequences: Sequence[SnapshotSequence],
    *,
    window_length: int | None,
    batch_size: int = 8,
    windows_per_epoch: int | None = None,
    window_seed: int | None = None,
    sampler: WindowLikeSampler | Any | None = None,
    distributed: bool = False,
    rank: int | None = None,
    world_size: int | None = None,
    api_name: str = "fit",
    shuffle: bool = True,
) -> WindowLikeSampler | Any | None:
    """Resolve a local or distributed temporal window sampler.

    Shared by single-process :func:`~koopman_graph.training.loop.run_fit_loop`
    and distributed fit adapters so mutual-exclusion and construction policy
    stay aligned.

    Parameters
    ----------
    train_sequences : sequence of SnapshotSequence
        Training trajectories used when constructing a sampler from
        ``window_length``.
    window_length : int or None
        Window length when building a sampler. Mutually exclusive with
        ``sampler``.
    batch_size : int, optional
        Windows per batch. Default is ``8``.
    windows_per_epoch : int or None, optional
        Cap on windows per epoch (global cap before sharding when
        ``distributed=True``).
    window_seed : int or None, optional
        Base shuffle seed passed to the constructed sampler.
    sampler : WindowSampler, NeighborWindowSampler, \
DistributedWindowSampler, or None, optional
        Pre-built sampler. When ``distributed=False``, accepted as-is.
        When ``distributed=True``, only
        :class:`~koopman_graph.distributed.DistributedWindowSampler` is
        accepted.
    distributed : bool, optional
        If ``True``, construct or accept a rank-aware
        :class:`~koopman_graph.distributed.DistributedWindowSampler`.
        Default is ``False`` (local :class:`WindowSampler` path).
    rank : int or None, optional
        Rank override for distributed construction.
    world_size : int or None, optional
        World-size override for distributed construction.
    api_name : str, optional
        Public API name used in distributed error messages. Default is
        ``"fit"``.
    shuffle : bool, optional
        Whether constructed samplers shuffle window order. Default is
        ``True``.

    Returns
    -------
    WindowSampler, NeighborWindowSampler, DistributedWindowSampler, or None
        Resolved sampler, or ``None`` when neither ``sampler`` nor
        ``window_length`` is set (full-sequence training).

    Raises
    ------
    ValueError
        If both ``sampler`` and ``window_length`` are set, or if
        ``distributed=True`` and ``sampler`` is a non-distributed window
        sampler.
    TypeError
        If ``distributed=True`` and ``sampler`` has an unsupported type.

    Notes
    -----
    **Hetero vs homogeneous legality** (documented for callers; this helper
    does not raise on hetero sequences by itself):

    - :class:`WindowSampler` and
      :class:`~koopman_graph.distributed.DistributedWindowSampler` are the
      intended temporal-window samplers for both homogeneous and hetero
      trajectories (single-process hetero windowed fit is gated separately
      in :func:`~koopman_graph.training.loop.run_fit_loop` until that guard
      is narrowed).
    - :class:`NeighborWindowSampler` remains **homogeneous-only** (static
      pairwise topology); it rejects
      :class:`~koopman_graph.data.HeteroGraphSnapshotSequence` at
      construction.
    """
    if sampler is not None and window_length is not None:
        msg = "pass sampler or window_length, not both"
        raise ValueError(msg)

    if not distributed:
        if sampler is not None:
            return sampler
        if window_length is None:
            return None
        return WindowSampler(
            train_sequences,
            window_length=window_length,
            batch_size=batch_size,
            windows_per_epoch=windows_per_epoch,
            shuffle=shuffle,
            seed=window_seed,
        )

    # importlib avoids a static ``data.sampling`` → ``distributed.sampling``
    # import edge (architecture DEP cycle with the reverse import).
    _DistributedWindowSampler = importlib.import_module(
        "koopman_graph.distributed.sampling"
    ).DistributedWindowSampler

    if sampler is None:
        if window_length is None:
            return None
        return _DistributedWindowSampler(
            train_sequences,
            window_length=window_length,
            batch_size=batch_size,
            windows_per_epoch=windows_per_epoch,
            shuffle=shuffle,
            seed=window_seed,
            rank=rank,
            world_size=world_size,
        )
    if isinstance(sampler, _DistributedWindowSampler):
        return sampler
    if isinstance(sampler, (WindowSampler, NeighborWindowSampler)):
        msg = (
            f"{api_name} requires DistributedWindowSampler or "
            "window_length=...; plain WindowSampler / NeighborWindowSampler "
            "are not rank-aware"
        )
        raise ValueError(msg)
    msg = f"unsupported sampler type: {type(sampler)!r}"
    raise TypeError(msg)
