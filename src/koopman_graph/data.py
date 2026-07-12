"""Utilities for spatiotemporal graph snapshot sequences."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data

ArrayLike = Tensor | np.ndarray


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


def _as_tensor(value: ArrayLike, *, dtype: torch.dtype | None = None) -> Tensor:
    """Convert an array-like value to a :class:`torch.Tensor`.

    Parameters
    ----------
    value : Tensor or ndarray
        Input array or tensor.
    dtype : torch.dtype, optional
        Target dtype. When ``value`` is already a tensor, conversion is applied
        only if the dtypes differ.

    Returns
    -------
    Tensor
        Tensor representation of ``value``.
    """
    if isinstance(value, Tensor):
        if dtype is not None and value.dtype != dtype:
            return value.to(dtype=dtype)
        return value
    return torch.as_tensor(value, dtype=dtype)


def _snapshot_edge_weight(snapshot: Data) -> Tensor | None:
    """Return optional scalar edge weights attached to a snapshot.

    Parameters
    ----------
    snapshot : Data
        Graph snapshot that may carry ``edge_weight``.

    Returns
    -------
    Tensor or None
        Edge weights with shape ``(num_edges,)``, or ``None`` when absent.
    """
    edge_weight = getattr(snapshot, "edge_weight", None)
    if edge_weight is None:
        return None
    return edge_weight


def _validate_control_inputs(
    control_inputs: Tensor,
    *,
    num_timesteps: int,
    num_nodes: int,
) -> None:
    """Validate optional per-timestep control inputs.

    Parameters
    ----------
    control_inputs : Tensor
        Control tensor with shape ``(num_timesteps, control_dim)`` for global
        controls or ``(num_timesteps, num_nodes, control_dim)`` for per-node
        controls.
    num_timesteps : int
        Expected number of timesteps in the sequence.
    num_nodes : int
        Expected node count for per-node controls.

    Raises
    ------
    ValueError
        If ``control_inputs`` has invalid rank or shape.
    """
    if control_inputs.ndim not in (2, 3):
        msg = (
            "control_inputs must have shape (num_timesteps, control_dim) or "
            "(num_timesteps, num_nodes, control_dim), "
            f"got {tuple(control_inputs.shape)}"
        )
        raise ValueError(msg)
    if control_inputs.shape[0] != num_timesteps:
        msg = (
            f"control_inputs has {control_inputs.shape[0]} timesteps, "
            f"expected {num_timesteps}"
        )
        raise ValueError(msg)
    if control_inputs.ndim == 3 and control_inputs.shape[1] != num_nodes:
        msg = (
            f"per-node control_inputs has {control_inputs.shape[1]} nodes, "
            f"expected {num_nodes}"
        )
        raise ValueError(msg)


def _validate_snapshot_metadata(snapshots: Sequence[Data]) -> None:
    """Verify consistent node count and feature dimension across snapshots.

    Parameters
    ----------
    snapshots : sequence of Data
        Graph snapshots to validate.

    Raises
    ------
    ValueError
        If the sequence is empty or any snapshot differs in node count or
        feature dimension from the first snapshot.
    """
    if not snapshots:
        msg = "GraphSnapshotSequence requires at least one snapshot"
        raise ValueError(msg)

    reference = snapshots[0]
    ref_num_nodes = reference.num_nodes
    ref_in_channels = reference.x.shape[1]

    for idx, snapshot in enumerate(snapshots[1:], start=1):
        if snapshot.num_nodes != ref_num_nodes:
            msg = (
                f"Snapshot {idx} has {snapshot.num_nodes} nodes, "
                f"expected {ref_num_nodes}"
            )
            raise ValueError(msg)
        if snapshot.x.shape[1] != ref_in_channels:
            msg = (
                f"Snapshot {idx} has feature dimension {snapshot.x.shape[1]}, "
                f"expected {ref_in_channels}"
            )
            raise ValueError(msg)


def _snapshots_have_dynamic_topology(snapshots: Sequence[Data]) -> bool:
    """Return whether any snapshot differs in ``edge_index`` from the first.

    Parameters
    ----------
    snapshots : sequence of Data
        Graph snapshots to inspect.

    Returns
    -------
    bool
        ``True`` when at least one snapshot uses a different ``edge_index``.
    """
    if not snapshots:
        return False
    reference = snapshots[0].edge_index
    return any(
        not torch.equal(snapshot.edge_index, reference) for snapshot in snapshots[1:]
    )


def _validate_shared_topology(snapshots: Sequence[Data]) -> None:
    """Verify that all snapshots share node count, features, and topology.

    Parameters
    ----------
    snapshots : sequence of Data
        Graph snapshots to validate.

    Raises
    ------
    ValueError
        If the sequence is empty or any snapshot differs in ``edge_index``,
        optional ``edge_weight``, node count, or feature dimension from the
        first snapshot.
    """
    _validate_snapshot_metadata(snapshots)

    reference = snapshots[0]
    ref_edge_index = reference.edge_index
    ref_edge_weight = _snapshot_edge_weight(reference)

    for idx, snapshot in enumerate(snapshots[1:], start=1):
        if not torch.equal(snapshot.edge_index, ref_edge_index):
            msg = f"Snapshot {idx} has a different edge_index than snapshot 0"
            raise ValueError(msg)
        edge_weight = _snapshot_edge_weight(snapshot)
        if (ref_edge_weight is None) != (edge_weight is None):
            msg = f"Snapshot {idx} edge_weight presence does not match snapshot 0"
            raise ValueError(msg)
        if ref_edge_weight is not None and not torch.allclose(
            edge_weight,
            ref_edge_weight,
            equal_nan=True,
        ):
            msg = f"Snapshot {idx} has a different edge_weight than snapshot 0"
            raise ValueError(msg)


@dataclass(frozen=True)
class TemporalSplit:
    """Train, validation, and test snapshot sequences from a temporal split.

    Attributes
    ----------
    train : GraphSnapshotSequence
        Earliest contiguous snapshots used for training.
    val : GraphSnapshotSequence
        Middle contiguous snapshots used for validation.
    test : GraphSnapshotSequence
        Latest contiguous snapshots held out for evaluation.
    """

    train: GraphSnapshotSequence
    val: GraphSnapshotSequence
    test: GraphSnapshotSequence


def temporal_split(
    sequence: GraphSnapshotSequence,
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    min_train_timesteps: int = 2,
    min_val_timesteps: int = 2,
    min_test_timesteps: int = 1,
) -> TemporalSplit:
    """Split a snapshot sequence into contiguous train, validation, and test sets.

    Earlier snapshots are assigned to training, later snapshots to validation and
    test. Ratios must sum to ``1.0``.

    Parameters
    ----------
    sequence : GraphSnapshotSequence
        Full time-ordered snapshot sequence to split.
    train_ratio : float, optional
        Fraction of timesteps assigned to training. Default is ``0.7``.
    val_ratio : float, optional
        Fraction assigned to validation. Default is ``0.1``.
    test_ratio : float, optional
        Fraction assigned to test. Default is ``0.2``.
    min_train_timesteps : int, optional
        Minimum training snapshots required. Default is ``2``.
    min_val_timesteps : int, optional
        Minimum validation snapshots required. Default is ``2``.
    min_test_timesteps : int, optional
        Minimum test snapshots required. Default is ``1``.

    Returns
    -------
    TemporalSplit
        Contiguous train, validation, and test sequences sharing topology.

    Raises
    ------
    ValueError
        If ratios do not sum to ``1.0``, any minimum is violated, or the
        sequence is too short for the requested split.
    """
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        msg = f"train_ratio + val_ratio + test_ratio must equal 1.0, got {ratio_sum}"
        raise ValueError(msg)
    if min_train_timesteps < 2:
        msg = f"min_train_timesteps must be >= 2, got {min_train_timesteps}"
        raise ValueError(msg)
    if min_val_timesteps < 1 or min_test_timesteps < 1:
        msg = "min_val_timesteps and min_test_timesteps must be >= 1"
        raise ValueError(msg)

    num_timesteps = sequence.num_timesteps
    min_required = min_train_timesteps + min_val_timesteps + min_test_timesteps
    if num_timesteps < min_required:
        msg = (
            f"sequence has {num_timesteps} timesteps but needs at least "
            f"{min_required} for the requested split"
        )
        raise ValueError(msg)

    train_end = int(num_timesteps * train_ratio)
    val_end = train_end + int(num_timesteps * val_ratio)
    train_end = max(train_end, min_train_timesteps)
    val_end = max(val_end, train_end + min_val_timesteps)
    if num_timesteps - val_end < min_test_timesteps:
        val_end = num_timesteps - min_test_timesteps

    train_snapshots = sequence.snapshots[:train_end]
    val_snapshots = sequence.snapshots[train_end:val_end]
    test_snapshots = sequence.snapshots[val_end:]

    if len(train_snapshots) < min_train_timesteps:  # pragma: no cover - defensive
        msg = (
            f"train split has {len(train_snapshots)} timesteps, "
            f"expected at least {min_train_timesteps}"
        )
        raise ValueError(msg)
    if len(val_snapshots) < min_val_timesteps:
        msg = (
            f"validation split has {len(val_snapshots)} timesteps, "
            f"expected at least {min_val_timesteps}"
        )
        raise ValueError(msg)
    if len(test_snapshots) < min_test_timesteps:  # pragma: no cover - defensive
        msg = (
            f"test split has {len(test_snapshots)} timesteps, "
            f"expected at least {min_test_timesteps}"
        )
        raise ValueError(msg)

    return TemporalSplit(
        train=GraphSnapshotSequence(
            train_snapshots,
            allow_dynamic_topology=sequence.allow_dynamic_topology,
            control_inputs=(
                None
                if sequence.control_inputs is None
                else sequence.control_inputs[:train_end]
            ),
        ),
        val=GraphSnapshotSequence(
            val_snapshots,
            allow_dynamic_topology=sequence.allow_dynamic_topology,
            control_inputs=(
                None
                if sequence.control_inputs is None
                else sequence.control_inputs[train_end:val_end]
            ),
        ),
        test=GraphSnapshotSequence(
            test_snapshots,
            allow_dynamic_topology=sequence.allow_dynamic_topology,
            control_inputs=(
                None
                if sequence.control_inputs is None
                else sequence.control_inputs[val_end:]
            ),
        ),
    )


class GraphSnapshotSequence:
    """Container for a time-ordered sequence of PyG ``Data`` graph snapshots.

    By default all snapshots must share the same ``edge_index``, optional
    ``edge_weight``, node count, and feature dimension. Set
    ``allow_dynamic_topology=True`` to permit per-snapshot ``edge_index`` while
    still requiring a fixed node count and feature dimension. Optional
    :attr:`control_inputs` store exogenous inputs ``u_t`` applied when
    advancing from snapshot ``t`` to ``t+1``. Downstream training APIs should
    require at least two snapshots; construction here allows a single snapshot
    for inspection or prediction-only workflows.

    Notes
    -----
    Read-only views of sequence metadata are exposed as :attr:`snapshots`,
    :attr:`edge_index`, :attr:`edge_weight`, :attr:`is_dynamic_topology`,
    :attr:`control_inputs`, :attr:`has_controls`, :attr:`control_dim`,
    :attr:`num_nodes`, :attr:`num_timesteps`, and :attr:`in_channels`. The
    :attr:`edge_index` and :attr:`edge_weight` properties are only defined for
    static-topology sequences; use ``sequence[t].edge_index`` when
    :attr:`is_dynamic_topology` is ``True``.
    """

    def __init__(
        self,
        snapshots: Sequence[Data],
        *,
        allow_dynamic_topology: bool = False,
        control_inputs: Tensor | None = None,
    ) -> None:
        """Initialize from a sequence of graph snapshots.

        Parameters
        ----------
        snapshots : sequence of Data
            Time-ordered graph snapshots. Validated for shared node count and
            feature dimension on construction. Topology is also validated to be
            shared unless ``allow_dynamic_topology=True``.
        allow_dynamic_topology : bool, optional
            When ``True``, permit per-snapshot ``edge_index`` and
            ``edge_weight``. Default is ``False``.
        control_inputs : Tensor or None, optional
            Per-timestep control inputs. Global controls use shape
            ``(num_timesteps, control_dim)``; per-node controls use
            ``(num_timesteps, num_nodes, control_dim)``. Entry ``t`` drives the
            transition from ``snapshots[t]`` to ``snapshots[t+1]``.
        """
        snapshot_list = list(snapshots)
        if allow_dynamic_topology:
            _validate_snapshot_metadata(snapshot_list)
        else:
            _validate_shared_topology(snapshot_list)
        if control_inputs is not None:
            _validate_control_inputs(
                control_inputs,
                num_timesteps=len(snapshot_list),
                num_nodes=int(snapshot_list[0].num_nodes),
            )
        self._snapshots = snapshot_list
        self._control_inputs = control_inputs
        self._allow_dynamic_topology = allow_dynamic_topology
        self._is_dynamic_topology = (
            allow_dynamic_topology and _snapshots_have_dynamic_topology(snapshot_list)
        )

    @classmethod
    def from_arrays(
        cls,
        node_features: ArrayLike,
        edge_index: ArrayLike,
        *,
        edge_weight: ArrayLike | None = None,
        control_inputs: ArrayLike | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> GraphSnapshotSequence:
        """Build a sequence from node feature arrays and a shared topology.

        Parameters
        ----------
        node_features : array-like
            Array with shape ``(num_timesteps, num_nodes, in_channels)``.
        edge_index : array-like
            Shared edge index with shape ``(2, num_edges)``.
        edge_weight : array-like, optional
            Shared scalar edge weights with shape ``(num_edges,)``. When
            provided, attached to every snapshot.
        control_inputs : array-like, optional
            Per-timestep control inputs with shape ``(num_timesteps,
            control_dim)`` or ``(num_timesteps, num_nodes, control_dim)``.
        dtype : torch.dtype, optional
            Floating dtype used when converting numpy inputs to torch tensors.
            Default is ``torch.float32``.

        Returns
        -------
        :class:`~koopman_graph.data.GraphSnapshotSequence`
            Validated snapshot sequence.

        Raises
        ------
        ValueError
            If ``node_features``, ``edge_index``, or ``edge_weight`` have
            invalid shape.
        """
        features = _as_tensor(node_features, dtype=dtype)
        edges = _as_tensor(edge_index, dtype=torch.long)
        weights = None if edge_weight is None else _as_tensor(edge_weight, dtype=dtype)
        controls = (
            None if control_inputs is None else _as_tensor(control_inputs, dtype=dtype)
        )

        if features.ndim != 3:
            msg = (
                f"node_features must have shape "
                f"(num_timesteps, num_nodes, in_channels), got {tuple(features.shape)}"
            )
            raise ValueError(msg)
        if edges.ndim != 2 or edges.shape[0] != 2:
            msg = f"edge_index must have shape (2, num_edges), got {tuple(edges.shape)}"
            raise ValueError(msg)
        if weights is not None and weights.ndim != 1:
            msg = (
                f"edge_weight must have shape (num_edges,), got {tuple(weights.shape)}"
            )
            raise ValueError(msg)
        if weights is not None and weights.shape[0] != edges.shape[1]:
            msg = (
                f"edge_weight length {weights.shape[0]} does not match "
                f"num_edges {edges.shape[1]}"
            )
            raise ValueError(msg)
        if features.shape[0] < 1:
            msg = "node_features must contain at least one timestep"
            raise ValueError(msg)

        snapshots = []
        for t in range(features.shape[0]):
            if weights is None:
                snapshots.append(Data(x=features[t], edge_index=edges))
            else:
                snapshots.append(
                    Data(
                        x=features[t],
                        edge_index=edges,
                        edge_weight=weights.clone(),
                    )
                )
        return cls(snapshots, control_inputs=controls)

    @classmethod
    def from_dynamic_arrays(
        cls,
        node_features: ArrayLike,
        edge_indices: Sequence[ArrayLike],
        *,
        edge_weights: Sequence[ArrayLike | None] | None = None,
        control_inputs: ArrayLike | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> GraphSnapshotSequence:
        """Build a dynamic-topology sequence from per-timestep edge indices.

        Parameters
        ----------
        node_features : array-like
            Array with shape ``(num_timesteps, num_nodes, in_channels)``.
        edge_indices : sequence of array-like
            One edge index per timestep, each with shape ``(2, num_edges_t)``.
        edge_weights : sequence of array-like or None, optional
            Optional per-timestep scalar edge weights aligned with
            ``edge_indices``. When provided, must have the same length as
            ``edge_indices``.
        control_inputs : array-like, optional
            Per-timestep control inputs with shape ``(num_timesteps,
            control_dim)`` or ``(num_timesteps, num_nodes, control_dim)``.
        dtype : torch.dtype, optional
            Floating dtype used when converting numpy inputs to torch tensors.
            Default is ``torch.float32``.

        Returns
        -------
        :class:`~koopman_graph.data.GraphSnapshotSequence`
            Validated snapshot sequence with ``allow_dynamic_topology=True``.

        Raises
        ------
        ValueError
            If shapes are inconsistent or ``edge_indices`` length mismatches
            ``num_timesteps``.
        """
        features = _as_tensor(node_features, dtype=dtype)
        if features.ndim != 3:
            msg = (
                f"node_features must have shape "
                f"(num_timesteps, num_nodes, in_channels), got {tuple(features.shape)}"
            )
            raise ValueError(msg)
        if features.shape[0] < 1:
            msg = "node_features must contain at least one timestep"
            raise ValueError(msg)

        num_timesteps = int(features.shape[0])
        if len(edge_indices) != num_timesteps:
            msg = (
                f"edge_indices has length {len(edge_indices)}, expected "
                f"{num_timesteps} to match node_features timesteps"
            )
            raise ValueError(msg)
        if edge_weights is not None and len(edge_weights) != num_timesteps:
            msg = (
                f"edge_weights has length {len(edge_weights)}, expected {num_timesteps}"
            )
            raise ValueError(msg)

        snapshots: list[Data] = []
        for t in range(num_timesteps):
            edges = _as_tensor(edge_indices[t], dtype=torch.long)
            if edges.ndim != 2 or edges.shape[0] != 2:
                msg = (
                    f"edge_indices[{t}] must have shape (2, num_edges), "
                    f"got {tuple(edges.shape)}"
                )
                raise ValueError(msg)
            weight = None
            if edge_weights is not None:
                weight_value = edge_weights[t]
                if weight_value is not None:
                    weight = _as_tensor(weight_value, dtype=dtype)
                    if weight.ndim != 1:
                        msg = (
                            f"edge_weights[{t}] must have shape (num_edges,), "
                            f"got {tuple(weight.shape)}"
                        )
                        raise ValueError(msg)
                    if weight.shape[0] != edges.shape[1]:
                        msg = (
                            f"edge_weights[{t}] length {weight.shape[0]} does not "
                            f"match num_edges {edges.shape[1]}"
                        )
                        raise ValueError(msg)
            if weight is None:
                snapshots.append(Data(x=features[t], edge_index=edges))
            else:
                snapshots.append(
                    Data(
                        x=features[t],
                        edge_index=edges,
                        edge_weight=weight,
                    )
                )

        controls = (
            None if control_inputs is None else _as_tensor(control_inputs, dtype=dtype)
        )
        return cls(
            snapshots,
            allow_dynamic_topology=True,
            control_inputs=controls,
        )

    @property
    def is_dynamic_topology(self) -> bool:
        """Return whether snapshots use time-varying ``edge_index``.

        Returns
        -------
        bool
            ``True`` when the sequence was constructed with
            ``allow_dynamic_topology=True`` and at least one snapshot differs in
            ``edge_index`` from the first snapshot.
        """
        return self._is_dynamic_topology

    @property
    def allow_dynamic_topology(self) -> bool:
        """Return whether dynamic topology mode was enabled at construction.

        Returns
        -------
        bool
            ``True`` when per-snapshot ``edge_index`` values are permitted.
        """
        return self._allow_dynamic_topology

    @property
    def control_inputs(self) -> Tensor | None:
        """Return per-timestep control inputs when present.

        Returns
        -------
        Tensor or None
            Control tensor with shape ``(num_timesteps, control_dim)`` or
            ``(num_timesteps, num_nodes, control_dim)``.
        """
        return self._control_inputs

    @property
    def has_controls(self) -> bool:
        """Return whether the sequence carries control inputs.

        Returns
        -------
        bool
            ``True`` when :attr:`control_inputs` is not ``None``.
        """
        return self._control_inputs is not None

    @property
    def control_dim(self) -> int:
        """Return the control feature dimension.

        Returns
        -------
        int
            Control dimension when controls are present, otherwise ``0``.
        """
        if self._control_inputs is None:
            return 0
        if self._control_inputs.ndim == 2:
            return int(self._control_inputs.shape[1])
        return int(self._control_inputs.shape[2])

    def control_at(self, index: int) -> Tensor:
        """Return the control input driving transition from snapshot ``index``.

        Parameters
        ----------
        index : int
            Timestep index in ``[0, num_timesteps - 1]``.

        Returns
        -------
        Tensor
            Control vector for the transition ``index -> index + 1``.

        Raises
        ------
        ValueError
            If controls are absent or ``index`` is out of range.
        """
        if self._control_inputs is None:
            msg = "sequence does not contain control inputs"
            raise ValueError(msg)
        if index < 0 or index >= self.num_timesteps:
            msg = (
                f"control index {index} is out of range for "
                f"{self.num_timesteps} timesteps"
            )
            raise ValueError(msg)
        return self._control_inputs[index]

    def rollout_controls(self, start: int, steps: int) -> list[Tensor]:
        """Return controls for an autoregressive rollout from a start snapshot.

        Parameters
        ----------
        start : int
            Index of the initial snapshot.
        steps : int
            Number of rollout steps.

        Returns
        -------
        list of Tensor
            Control inputs for each rollout step. Empty when the sequence has
            no controls.

        Raises
        ------
        ValueError
            If ``start`` or ``steps`` are invalid or controls are unavailable
            for the requested horizon.
        """
        if steps < 1:
            msg = f"steps must be >= 1, got {steps}"
            raise ValueError(msg)
        if start < 0 or start >= self.num_timesteps:
            msg = f"start index {start} is out of range"
            raise ValueError(msg)
        if not self.has_controls:
            return []
        end = start + steps
        if end > self.num_timesteps:
            msg = (
                f"sequence has controls for {self.num_timesteps} timesteps but "
                f"rollout from start={start} requires {steps} controls"
            )
            raise ValueError(msg)
        return [self.control_at(start + step) for step in range(steps)]

    @property
    def snapshots(self) -> list[Data]:
        """Return the underlying list of graph snapshots.

        Returns
        -------
        list of Data
            Time-ordered PyG graph snapshots.
        """
        return self._snapshots

    @property
    def edge_index(self) -> Tensor:
        """Return the shared edge index for static-topology sequences.

        Returns
        -------
        Tensor
            Edge index with shape ``(2, num_edges)``.

        Raises
        ------
        ValueError
            If :attr:`is_dynamic_topology` is ``True``.
        """
        if self._is_dynamic_topology:
            msg = (
                "edge_index is undefined for dynamic-topology sequences; "
                "use sequence[t].edge_index"
            )
            raise ValueError(msg)
        return self._snapshots[0].edge_index

    @property
    def edge_weight(self) -> Tensor | None:
        """Return the shared scalar edge weights for static-topology sequences.

        Returns
        -------
        Tensor or None
            Edge weights with shape ``(num_edges,)``, or ``None`` when the
            sequence is unweighted.

        Raises
        ------
        ValueError
            If :attr:`is_dynamic_topology` is ``True``.
        """
        if self._is_dynamic_topology:
            msg = (
                "edge_weight is undefined for dynamic-topology sequences; "
                "use sequence[t].edge_weight"
            )
            raise ValueError(msg)
        return _snapshot_edge_weight(self._snapshots[0])

    @property
    def num_nodes(self) -> int:
        """Return the number of nodes in the graph topology.

        Returns
        -------
        int
            Node count shared across all snapshots.
        """
        return int(self._snapshots[0].num_nodes)

    @property
    def num_timesteps(self) -> int:
        """Return the number of timesteps in the sequence.

        Returns
        -------
        int
            Length of the temporal sequence.
        """
        return len(self._snapshots)

    @property
    def in_channels(self) -> int:
        """Return the node feature dimension.

        Returns
        -------
        int
            Feature dimension shared across all snapshots.
        """
        return int(self._snapshots[0].x.shape[1])

    def __len__(self) -> int:
        """Return the number of timesteps in the sequence.

        Returns
        -------
        int
            Same value as :attr:`num_timesteps`.
        """
        return len(self._snapshots)

    def __getitem__(self, index: int) -> Data:
        """Return the graph snapshot at ``index``.

        Parameters
        ----------
        index : int
            Timestep index.

        Returns
        -------
        Data
            Graph snapshot at the requested timestep.
        """
        return self._snapshots[index]

    def slice(self, start: int, stop: int) -> GraphSnapshotSequence:
        """Return a contiguous temporal sub-sequence.

        Parameters
        ----------
        start : int
            Inclusive start index.
        stop : int
            Exclusive stop index.

        Returns
        -------
        GraphSnapshotSequence
            Snapshots in ``[start, stop)`` with matching controls and topology
            policy.

        Raises
        ------
        ValueError
            If the bounds are negative, empty, reversed, or exceed the
            sequence length.
        """
        if start < 0 or stop <= start or stop > self.num_timesteps:
            msg = (
                "slice bounds must satisfy "
                f"0 <= start < stop <= {self.num_timesteps}, "
                f"got start={start}, stop={stop}"
            )
            raise ValueError(msg)
        return GraphSnapshotSequence(
            self._snapshots[start:stop],
            allow_dynamic_topology=self.allow_dynamic_topology,
            control_inputs=(
                None if self.control_inputs is None else self.control_inputs[start:stop]
            ),
        )

    def __iter__(self) -> Iterator[Data]:
        """Iterate over graph snapshots in temporal order.

        Yields
        ------
        Data
            Graph snapshot at each timestep.
        """
        return iter(self._snapshots)


def resolve_sequence(
    sequence: GraphSnapshotSequence | Sequence[Data],
) -> GraphSnapshotSequence:
    """Normalize input into a validated snapshot sequence.

    Wraps a plain sequence of ``Data`` snapshots in
    :class:`GraphSnapshotSequence`; existing sequences are returned unchanged.

    Parameters
    ----------
    sequence : GraphSnapshotSequence or sequence of Data
        Raw snapshot input from a training, baseline, or inference API.

    Returns
    -------
    GraphSnapshotSequence
        Validated sequence container.
    """
    if isinstance(sequence, GraphSnapshotSequence):
        return sequence
    return GraphSnapshotSequence(sequence)
