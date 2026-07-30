"""Graph snapshot sequence containers (homogeneous and heterogeneous)."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence

import torch
from torch import Tensor
from torch_geometric.data import Data, HeteroData

from koopman_graph.data.construction import (
    build_snapshots_from_arrays,
    build_snapshots_from_dynamic_arrays,
    build_windowed_snapshots,
)
from koopman_graph.data.validation import (
    ArrayLike,
    EdgeTypeTriple,
    hetero_snapshots_have_dynamic_topology,
    snapshots_have_dynamic_topology,
    validate_control_inputs,
    validate_hetero_control_inputs,
    validate_hetero_observation_masks,
    validate_hetero_snapshot_metadata,
    validate_observation_masks,
    validate_shared_hetero_topology,
    validate_shared_topology,
    validate_snapshot_metadata,
    validate_static_hyperedges,
    validate_timestamps,
)
from koopman_graph.graph_utils import (
    snapshot_edge_weight,
    snapshot_hyperedge_index,
    snapshot_hyperedge_weight,
)


class GraphSnapshotSequence:
    """Container for a time-ordered sequence of PyG ``Data`` graph snapshots.

    By default all snapshots must share the same ``edge_index``, optional
    ``edge_weight``, node count, and feature dimension. Set
    ``allow_dynamic_topology=True`` to permit per-snapshot ``edge_index`` while
    still requiring a fixed node count and feature dimension. Optional
    ``hyperedge_index`` / ``hyperedge_weight`` on each ``Data`` snapshot carry
    a PyG bipartite hyperedge incidence (row 0 = nodes, row 1 = hyperedges);
    when present they must be identical across timesteps (static hyperedges
    only). Optional :attr:`control_inputs` store exogenous inputs ``u_t``
    applied when advancing from snapshot ``t`` to ``t+1`` (global ``(T, C)``
    or per-node ``(T, N, C)``). Not every consumer supports both layouts:
    neural model / adaptation preserve per-node rows;
    :class:`~koopman_graph.env.GraphKoopmanEnv` and
    :class:`~koopman_graph.baselines.DMDcBaseline` are global-only (see
    architecture control layout capability matrix). Optional
    :attr:`observation_masks` mark which nodes are measured at each timestep
    (``True`` = observed). When masks are present, training and evaluation
    losses average only over observed nodes; reconstruction at pair
    ``(t, t+1)`` uses ``mask[t+1]``, and consistency terms use
    ``mask[t] & mask[t+1]``. Downstream training APIs should
    require at least two snapshots; construction here allows a single snapshot
    for inspection or prediction-only workflows.

    The snapshot **collection** is logically immutable after construction:
    :attr:`snapshots` returns a ``tuple`` that cannot be appended to or
    replaced in place. Individual ``Data`` objects are **borrowed** (not
    cloned): in-place mutation of node features or topology on a returned
    ``Data`` is possible and is not prevented. Callers that need isolation
    should clone snapshots explicitly.

    Notes
    -----
    Read-only views of sequence metadata are exposed as :attr:`snapshots`,
    :attr:`edge_index`, :attr:`edge_weight`, :attr:`hyperedge_index`,
    :attr:`hyperedge_weight`, :attr:`has_hyperedges`,
    :attr:`is_dynamic_topology`, :attr:`control_inputs`, :attr:`has_controls`,
    :attr:`control_dim`, :attr:`timestamps`, :attr:`has_timestamps`,
    :attr:`observation_masks`, :attr:`has_observation_masks`, and
    :attr:`num_nodes`. :attr:`edge_index` and :attr:`edge_weight` properties
    are only defined for static-topology sequences; use
    ``sequence[t].edge_index`` when :attr:`is_dynamic_topology` is ``True``.
    """

    def __init__(
        self,
        snapshots: Sequence[Data],
        *,
        allow_dynamic_topology: bool = False,
        control_inputs: Tensor | None = None,
        timestamps: Tensor | None = None,
        observation_masks: Tensor | None = None,
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
        timestamps : Tensor or None, optional
            Strictly increasing physical timestamps with shape
            ``(num_timesteps,)``. When present, training uses per-pair
            ``Δt = timestamps[t+1] - timestamps[t]``.
        observation_masks : Tensor or None, optional
            Per-timestep node observation mask with shape
            ``(num_timesteps, num_nodes)``. ``True`` (or ``1``) marks an
            observed node measurement at that snapshot.
        """
        snapshot_list = list(snapshots)
        if allow_dynamic_topology:
            validate_snapshot_metadata(snapshot_list)
        else:
            validate_shared_topology(snapshot_list)
        validate_static_hyperedges(snapshot_list)
        if control_inputs is not None:
            validate_control_inputs(
                control_inputs,
                num_timesteps=len(snapshot_list),
                num_nodes=int(snapshot_list[0].num_nodes),
            )
        if timestamps is not None:
            validate_timestamps(
                timestamps,
                num_timesteps=len(snapshot_list),
            )
        validated_masks = None
        if observation_masks is not None:
            validated_masks = validate_observation_masks(
                observation_masks,
                num_timesteps=len(snapshot_list),
                num_nodes=int(snapshot_list[0].num_nodes),
            )
        # Tuple freezes collection length/order; Data elements remain borrowed.
        self._snapshots = tuple(snapshot_list)
        self._control_inputs = control_inputs
        self._timestamps = timestamps
        self._observation_masks = validated_masks
        self._allow_dynamic_topology = allow_dynamic_topology
        self._is_dynamic_topology = (
            allow_dynamic_topology and snapshots_have_dynamic_topology(snapshot_list)
        )

    @classmethod
    def from_arrays(
        cls,
        node_features: ArrayLike,
        edge_index: ArrayLike,
        *,
        edge_weight: ArrayLike | None = None,
        hyperedge_index: ArrayLike | None = None,
        hyperedge_weight: ArrayLike | None = None,
        control_inputs: ArrayLike | None = None,
        timestamps: ArrayLike | None = None,
        observation_masks: ArrayLike | None = None,
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
        hyperedge_index : array-like, optional
            Shared PyG bipartite hyperedge incidence with shape ``(2, nnz)``
            (row 0 = nodes, row 1 = hyperedges).
        hyperedge_weight : array-like, optional
            Shared hyperedge weights with shape ``(num_hyperedges,)``.
            Requires ``hyperedge_index``.
        control_inputs : array-like, optional
            Per-timestep control inputs with shape ``(num_timesteps,
            control_dim)`` or ``(num_timesteps, num_nodes, control_dim)``.
        timestamps : array-like, optional
            Strictly increasing physical timestamps with shape
            ``(num_timesteps,)``.
        observation_masks : array-like, optional
            Per-timestep node observation mask with shape
            ``(num_timesteps, num_nodes)``.
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
            If ``node_features``, ``edge_index``, ``edge_weight``, or hyperedge
            fields have invalid shape.
        """
        built = build_snapshots_from_arrays(
            node_features,
            edge_index,
            edge_weight=edge_weight,
            hyperedge_index=hyperedge_index,
            hyperedge_weight=hyperedge_weight,
            control_inputs=control_inputs,
            timestamps=timestamps,
            observation_masks=observation_masks,
            dtype=dtype,
        )
        return cls(
            built.snapshots,
            control_inputs=built.control_inputs,
            timestamps=built.timestamps,
            observation_masks=built.observation_masks,
        )

    @classmethod
    def from_dynamic_arrays(
        cls,
        node_features: ArrayLike,
        edge_indices: Sequence[ArrayLike],
        *,
        edge_weights: Sequence[ArrayLike | None] | None = None,
        control_inputs: ArrayLike | None = None,
        timestamps: ArrayLike | None = None,
        observation_masks: ArrayLike | None = None,
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
        timestamps : array-like, optional
            Strictly increasing physical timestamps with shape
            ``(num_timesteps,)``.
        observation_masks : array-like, optional
            Per-timestep node observation mask with shape
            ``(num_timesteps, num_nodes)``.
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
        built = build_snapshots_from_dynamic_arrays(
            node_features,
            edge_indices,
            edge_weights=edge_weights,
            control_inputs=control_inputs,
            timestamps=timestamps,
            observation_masks=observation_masks,
            dtype=dtype,
        )
        return cls(
            built.snapshots,
            allow_dynamic_topology=built.allow_dynamic_topology,
            control_inputs=built.control_inputs,
            timestamps=built.timestamps,
            observation_masks=built.observation_masks,
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

    @property
    def timestamps(self) -> Tensor | None:
        """Return per-snapshot physical timestamps when present.

        Returns
        -------
        Tensor or None
            Timestamps with shape ``(num_timesteps,)``.
        """
        return self._timestamps

    @property
    def has_timestamps(self) -> bool:
        """Return whether the sequence carries timestamps.

        Returns
        -------
        bool
            ``True`` when :attr:`timestamps` is not ``None``.
        """
        return self._timestamps is not None

    def delta_t_at(self, index: int) -> Tensor:
        """Return ``timestamps[index + 1] - timestamps[index]``.

        Parameters
        ----------

        index : int
            Source snapshot index for the transition pair.

        Returns
        -------

        Tensor
            See summary line.

        Raises
        ------

        ValueError
            If timestamps are absent or ``index`` is out of range."""
        if self._timestamps is None:
            msg = "sequence does not contain timestamps"
            raise ValueError(msg)
        if index < 0 or index >= self.num_timesteps - 1:
            msg = (
                f"delta_t index {index} is out of range for "
                f"{self.num_timesteps} timesteps"
            )
            raise ValueError(msg)
        return self._timestamps[index + 1] - self._timestamps[index]

    @property
    def observation_masks(self) -> Tensor | None:
        """Return per-timestep node observation masks when present.

        Returns
        -------
        Tensor or None
            Boolean mask with shape ``(num_timesteps, num_nodes)``.
        """
        return self._observation_masks

    @property
    def has_observation_masks(self) -> bool:
        """Return whether the sequence carries observation masks.

        Returns
        -------
        bool
            ``True`` when :attr:`observation_masks` is not ``None``.
        """
        return self._observation_masks is not None

    def observation_mask_at(self, index: int) -> Tensor:
        """Return the observation mask for snapshot ``index``.

        Parameters
        ----------
        index : int
            Timestep index in ``[0, num_timesteps - 1]``.

        Returns
        -------
        Tensor
            Boolean mask with shape ``(num_nodes,)``.

        Raises
        ------
        ValueError
            If masks are absent or ``index`` is out of range.
        """
        if self._observation_masks is None:
            msg = "sequence does not contain observation_masks"
            raise ValueError(msg)
        if index < 0 or index >= self.num_timesteps:
            msg = (
                f"observation mask index {index} is out of range for "
                f"{self.num_timesteps} timesteps"
            )
            raise ValueError(msg)
        return self._observation_masks[index]

    def pair_observation_mask(self, index: int) -> Tensor:
        """Return ``mask[index] & mask[index + 1]`` for transition pairs.

        Parameters
        ----------
        index : int
            Source snapshot index for the transition pair.

        Returns
        -------
        Tensor
            Boolean mask with shape ``(num_nodes,)``.

        Raises
        ------
        ValueError
            If masks are absent or ``index`` is out of range for a pair.
        """
        if self._observation_masks is None:
            msg = "sequence does not contain observation_masks"
            raise ValueError(msg)
        if index < 0 or index >= self.num_timesteps - 1:
            msg = (
                f"pair observation mask index {index} is out of range for "
                f"{self.num_timesteps} timesteps"
            )
            raise ValueError(msg)
        return self._observation_masks[index] & self._observation_masks[index + 1]

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
    def snapshots(self) -> tuple[Data, ...]:
        """Return the immutable sequence of graph snapshots.

        Returns
        -------
        tuple of Data
            Time-ordered PyG graph snapshots. The tuple itself cannot be
            mutated; individual ``Data`` objects are borrowed references
            (see class docstring).
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
        return snapshot_edge_weight(self._snapshots[0])

    @property
    def has_hyperedges(self) -> bool:
        """Return whether snapshots carry a hyperedge incidence.

        Returns
        -------
        bool
            ``True`` when snapshot 0 has ``hyperedge_index`` (validated as
            shared across the sequence at construction).
        """
        return snapshot_hyperedge_index(self._snapshots[0]) is not None

    @property
    def hyperedge_index(self) -> Tensor | None:
        """Return the shared hyperedge incidence when present.

        Returns
        -------
        Tensor or None
            Bipartite incidence with shape ``(2, nnz)``, or ``None`` when the
            sequence has no hyperedges.
        """
        return snapshot_hyperedge_index(self._snapshots[0])

    @property
    def hyperedge_weight(self) -> Tensor | None:
        """Return the shared hyperedge weights when present.

        Returns
        -------
        Tensor or None
            Weights with shape ``(num_hyperedges,)``, or ``None`` when
            unweighted or when hyperedges are absent.
        """
        return snapshot_hyperedge_weight(self._snapshots[0])

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
            timestamps=(
                None if self.timestamps is None else self.timestamps[start:stop]
            ),
            observation_masks=(
                None
                if self.observation_masks is None
                else self.observation_masks[start:stop]
            ),
        )

    def windowed(
        self,
        n_delays: int,
        *,
        stride: int = 1,
        pad: bool = True,
        zero_unobserved: bool = True,
    ) -> GraphSnapshotSequence:
        """Return a sequence whose features are Hankel-stacked delay windows.

        Each output snapshot at position ``i`` stores node features
        ``(num_nodes, n_delays * F)`` built from the source window ending at
        source index ``start + i * stride`` (with optional left zero-padding).
        Topology, controls, timestamps, and observation masks are taken from
        the **end** of each window. Topology changes inside a window raise
        unless :attr:`allow_dynamic_topology` is ``True``.

        Parameters
        ----------
        n_delays : int
            Delay window length (must be >= 1). ``1`` returns a feature-cloned
            copy of the (possibly strided) source sequence.
        stride : int, optional
            Step between successive window ends. Default is ``1``.
        pad : bool, optional
            Zero-pad missing history at the sequence start. When ``False``,
            the first window end is ``n_delays - 1``. Default is ``True``.
        zero_unobserved : bool, optional
            Zero unobserved node rows inside each delay slot when observation
            masks are present. Default is ``True``.

        Returns
        -------
        GraphSnapshotSequence
            Windowed trajectory with stacked channel features.

        Raises
        ------
        ValueError
            If ``n_delays < 1``, ``stride < 1``, the sequence is too short when
            ``pad=False``, or a window spans a topology change on a static
            sequence.
        """
        built = build_windowed_snapshots(
            self,
            n_delays,
            stride=stride,
            pad=pad,
            zero_unobserved=zero_unobserved,
        )
        return GraphSnapshotSequence(
            built.snapshots,
            allow_dynamic_topology=built.allow_dynamic_topology,
            control_inputs=built.control_inputs,
            timestamps=built.timestamps,
            observation_masks=built.observation_masks,
        )

    def __iter__(self) -> Iterator[Data]:
        """Iterate over graph snapshots in temporal order.

        Yields
        ------
        Data
            Graph snapshot at each timestep.
        """
        return iter(self._snapshots)


class HeteroGraphSnapshotSequence:
    """Container for time-ordered PyG ``HeteroData`` multiplex / typed snapshots.

    Snapshots use PyG ``(src_type, relation, dst_type)`` edge stores. Node
    counts ``N_τ``, feature dims ``F_τ``, and the edge-type set are fixed across
    the sequence. Per-relation ``edge_index`` values are shared unless
    ``allow_dynamic_topology=True``.

    **Multiplex** graphs are the one-node-type case (commonly ``"node"``) with
    ``|R| >= 1`` edge types. Multiple node types are validated here for the
    typed path; encode/advance consumers land in later tasks.

    There is **no** homogeneous :attr:`edge_index` property. Use
    :attr:`edge_index_dict` (static topology) or ``sequence[t][edge_type]``.

    The snapshot collection is immutable after construction; individual
    ``HeteroData`` objects are borrowed (not cloned).

    Parameters
    ----------
    snapshots : sequence of HeteroData
        Time-ordered heterogeneous snapshots.
    allow_dynamic_topology : bool, optional
        When ``True``, permit per-snapshot per-relation ``edge_index``.
        Default is ``False``.
    control_inputs : Tensor or mapping of str to Tensor or None, optional
        Global ``(T, C)`` controls, single-type ``(T, N, C)``, or a dict of
        per-type ``(T, C)`` / ``(T, N_τ, C)`` tensors.
    timestamps : Tensor or None, optional
        Strictly increasing timestamps with shape ``(T,)``.
    observation_masks : mapping of str to Tensor or None, optional
        Per-type masks with shape ``(T, N_τ)`` for every node type.
    """

    def __init__(
        self,
        snapshots: Sequence[HeteroData],
        *,
        allow_dynamic_topology: bool = False,
        control_inputs: Tensor | Mapping[str, Tensor] | None = None,
        timestamps: Tensor | None = None,
        observation_masks: Mapping[str, Tensor] | None = None,
    ) -> None:
        """Initialize from a sequence of heterogeneous graph snapshots.

        Parameters
        ----------
        snapshots : sequence of HeteroData
            Time-ordered heterogeneous snapshots. Validated for fixed
            ``N_τ``, ``F_τ``, and edge-type set. Topology is shared unless
            ``allow_dynamic_topology=True``.
        allow_dynamic_topology : bool, optional
            When ``True``, permit per-snapshot per-relation ``edge_index``.
            Default is ``False``.
        control_inputs : Tensor or mapping of str to Tensor or None, optional
            Global ``(T, C)`` controls, single-type ``(T, N, C)``, or a dict of
            per-type ``(T, C)`` / ``(T, N_τ, C)`` tensors.
        timestamps : Tensor or None, optional
            Strictly increasing timestamps with shape ``(T,)``.
        observation_masks : mapping of str to Tensor or None, optional
            Per-type masks with shape ``(T, N_τ)`` for every node type.
        """
        snapshot_list = list(snapshots)
        for idx, snapshot in enumerate(snapshot_list):
            if not isinstance(snapshot, HeteroData):
                msg = (
                    "HeteroGraphSnapshotSequence snapshots must be HeteroData; "
                    f"index {idx} has type {type(snapshot).__name__}"
                )
                raise TypeError(msg)

        if allow_dynamic_topology:
            node_feature_dims, num_nodes, edge_types = (
                validate_hetero_snapshot_metadata(snapshot_list)
            )
        else:
            node_feature_dims, num_nodes, edge_types = validate_shared_hetero_topology(
                snapshot_list
            )

        if control_inputs is not None:
            validate_hetero_control_inputs(
                control_inputs,
                num_timesteps=len(snapshot_list),
                num_nodes=num_nodes,
            )
        if timestamps is not None:
            validate_timestamps(
                timestamps,
                num_timesteps=len(snapshot_list),
            )
        validated_masks = None
        if observation_masks is not None:
            validated_masks = validate_hetero_observation_masks(
                observation_masks,
                num_timesteps=len(snapshot_list),
                num_nodes=num_nodes,
            )

        self._snapshots = tuple(snapshot_list)
        self._node_feature_dims = dict(node_feature_dims)
        self._num_nodes = dict(num_nodes)
        self._edge_types = list(edge_types)
        self._node_type_names = list(node_feature_dims.keys())
        self._control_inputs: Tensor | dict[str, Tensor] | None
        if isinstance(control_inputs, Mapping):
            self._control_inputs = dict(control_inputs)
        else:
            self._control_inputs = control_inputs
        self._timestamps = timestamps
        self._observation_masks = validated_masks
        self._allow_dynamic_topology = allow_dynamic_topology
        self._is_dynamic_topology = (
            allow_dynamic_topology
            and hetero_snapshots_have_dynamic_topology(snapshot_list, edge_types)
        )

    @property
    def node_types(self) -> dict[str, int]:
        """Return mapping from node type name to feature dimension ``F_τ``.

        Returns
        -------
        dict of str to int
            Feature dimension per node type (ordered as in snapshot 0).
        """
        return dict(self._node_feature_dims)

    @property
    def node_type_names(self) -> tuple[str, ...]:
        """Return ordered node type names from snapshot 0.

        Returns
        -------
        tuple of str
            Node type names in schema order.
        """
        return tuple(self._node_type_names)

    @property
    def edge_types(self) -> tuple[EdgeTypeTriple, ...]:
        """Return ordered ``(src_type, relation, dst_type)`` triples.

        Returns
        -------
        tuple of (str, str, str)
            Edge types shared across the sequence.
        """
        return tuple(self._edge_types)

    @property
    def num_nodes_dict(self) -> dict[str, int]:
        """Return mapping from node type to node count ``N_τ``.

        Returns
        -------
        dict of str to int
            Fixed per-type cardinalities.
        """
        return dict(self._num_nodes)

    def num_nodes_of(self, node_type: str) -> int:
        """Return the node count for ``node_type``.

        Parameters
        ----------
        node_type : str
            Node type name.

        Returns
        -------
        int
            ``N_τ`` for the requested type.

        Raises
        ------
        KeyError
            If ``node_type`` is not in the sequence schema.
        """
        if node_type not in self._num_nodes:
            msg = (
                f"unknown node type {node_type!r}; "
                f"valid types are {sorted(self._num_nodes)!r}"
            )
            raise KeyError(msg)
        return int(self._num_nodes[node_type])

    @property
    def num_nodes(self) -> int:
        """Return node count for multiplex (single-type) sequences.

        Returns
        -------
        int
            ``N`` when the sequence has exactly one node type.

        Raises
        ------
        ValueError
            If more than one node type is present; use
            :attr:`num_nodes_dict`, :meth:`num_nodes_of`, or
            :attr:`num_nodes_total` instead.
        """
        if len(self._num_nodes) != 1:
            msg = (
                "num_nodes is defined only for single-node-type (multiplex) "
                f"sequences; got types {sorted(self._num_nodes)!r}. "
                "Use num_nodes_dict, num_nodes_of(...), or num_nodes_total."
            )
            raise ValueError(msg)
        return int(next(iter(self._num_nodes.values())))

    @property
    def num_nodes_total(self) -> int:
        """Return the sum of per-type node counts.

        Returns
        -------
        int
            ``sum_τ N_τ``.
        """
        return int(sum(self._num_nodes.values()))

    @property
    def in_channels(self) -> int:
        """Return feature dim for multiplex (single-type) sequences.

        Returns
        -------
        int
            ``F`` when exactly one node type is present.

        Raises
        ------
        ValueError
            If more than one node type is present; use :attr:`node_types`.
        """
        if len(self._node_feature_dims) != 1:
            msg = (
                "in_channels is defined only for single-node-type (multiplex) "
                f"sequences; got types {sorted(self._node_feature_dims)!r}. "
                "Use node_types[type] instead."
            )
            raise ValueError(msg)
        return int(next(iter(self._node_feature_dims.values())))

    @property
    def is_dynamic_topology(self) -> bool:
        """Return whether any relation uses time-varying ``edge_index``.

        Returns
        -------
        bool
            ``True`` when dynamic topology was enabled and at least one
            relation differs from snapshot 0.
        """
        return self._is_dynamic_topology

    @property
    def allow_dynamic_topology(self) -> bool:
        """Return whether dynamic topology mode was enabled at construction.

        Returns
        -------
        bool
            ``True`` when per-snapshot per-relation ``edge_index`` is permitted.
        """
        return self._allow_dynamic_topology

    @property
    def edge_index_dict(self) -> dict[EdgeTypeTriple, Tensor]:
        """Return per-relation edge indices for static-topology sequences.

        Returns
        -------
        dict of (str, str, str) to Tensor
            Mapping from edge type to ``edge_index`` with shape ``(2, E_r)``.

        Raises
        ------
        ValueError
            If :attr:`is_dynamic_topology` is ``True``.
        """
        if self._is_dynamic_topology:
            msg = (
                "edge_index_dict is undefined for dynamic-topology sequences; "
                "use sequence[t][edge_type].edge_index"
            )
            raise ValueError(msg)
        reference = self._snapshots[0]
        return {
            edge_type: reference[edge_type].edge_index for edge_type in self._edge_types
        }

    @property
    def control_inputs(self) -> Tensor | dict[str, Tensor] | None:
        """Return control inputs when present.

        Returns
        -------
        Tensor or dict of str to Tensor or None
            Global/per-node tensor or per-type mapping.
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
            Shared control dimension when controls are present, else ``0``.
        """
        controls = self._control_inputs
        if controls is None:
            return 0
        if isinstance(controls, Mapping):
            first = next(iter(controls.values()))
            if first.ndim == 2:
                return int(first.shape[1])
            return int(first.shape[2])
        if controls.ndim == 2:
            return int(controls.shape[1])
        return int(controls.shape[2])

    @property
    def timestamps(self) -> Tensor | None:
        """Return per-snapshot physical timestamps when present.

        Returns
        -------
        Tensor or None
            Timestamps with shape ``(num_timesteps,)``.
        """
        return self._timestamps

    @property
    def has_timestamps(self) -> bool:
        """Return whether the sequence carries timestamps.

        Returns
        -------
        bool
            ``True`` when :attr:`timestamps` is not ``None``.
        """
        return self._timestamps is not None

    @property
    def observation_masks(self) -> dict[str, Tensor] | None:
        """Return per-type observation masks when present.

        Returns
        -------
        dict of str to Tensor or None
            Boolean masks with shape ``(T, N_τ)`` per node type.
        """
        if self._observation_masks is None:
            return None
        return dict(self._observation_masks)

    @property
    def has_observation_masks(self) -> bool:
        """Return whether the sequence carries observation masks.

        Returns
        -------
        bool
            ``True`` when :attr:`observation_masks` is not ``None``.
        """
        return self._observation_masks is not None

    def observation_mask_at(self, index: int) -> dict[str, Tensor]:
        """Return per-type observation masks at ``index``.

        Parameters
        ----------
        index : int
            Timestep index.

        Returns
        -------
        dict of str to Tensor
            Boolean masks with shape ``(N_τ,)`` per node type.

        Raises
        ------
        ValueError
            If the sequence has no observation masks.
        IndexError
            If ``index`` is out of range.
        """
        if self._observation_masks is None:
            msg = "sequence does not contain observation_masks"
            raise ValueError(msg)
        if index < 0 or index >= self.num_timesteps:
            msg = f"index must satisfy 0 <= index < {self.num_timesteps}, got {index}"
            raise IndexError(msg)
        return {
            node_type: mask[index]
            for node_type, mask in self._observation_masks.items()
        }

    def pair_observation_mask(self, index: int) -> dict[str, Tensor]:
        """Return per-type masks observed at both ``index`` and ``index + 1``.

        Parameters
        ----------
        index : int
            Index of the first snapshot in the consecutive pair.

        Returns
        -------
        dict of str to Tensor
            Element-wise ``AND`` of consecutive masks per node type.

        Raises
        ------
        ValueError
            If the sequence has no observation masks.
        IndexError
            If ``index`` is not a valid pair start.
        """
        if self._observation_masks is None:
            msg = "sequence does not contain observation_masks"
            raise ValueError(msg)
        if index < 0 or index >= self.num_timesteps - 1:
            msg = (
                f"index must satisfy 0 <= index < {self.num_timesteps - 1}, got {index}"
            )
            raise IndexError(msg)
        return {
            node_type: mask[index] & mask[index + 1]
            for node_type, mask in self._observation_masks.items()
        }

    @property
    def snapshots(self) -> tuple[HeteroData, ...]:
        """Return the immutable sequence of heterogeneous snapshots.

        Returns
        -------
        tuple of HeteroData
            Time-ordered snapshots (borrowed references).
        """
        return self._snapshots

    @property
    def num_timesteps(self) -> int:
        """Return the number of timesteps in the sequence.

        Returns
        -------
        int
            Length of the temporal sequence.
        """
        return len(self._snapshots)

    def __len__(self) -> int:
        """Return the number of timesteps in the sequence.

        Returns
        -------
        int
            Same value as :attr:`num_timesteps`.
        """
        return len(self._snapshots)

    def __getitem__(self, index: int) -> HeteroData:
        """Return the heterogeneous snapshot at ``index``.

        Parameters
        ----------
        index : int
            Timestep index.

        Returns
        -------
        HeteroData
            Snapshot at the requested timestep.
        """
        return self._snapshots[index]

    def __iter__(self) -> Iterator[HeteroData]:
        """Iterate over heterogeneous snapshots in temporal order.

        Yields
        ------
        HeteroData
            Snapshot at each timestep.
        """
        return iter(self._snapshots)

    def slice(self, start: int, stop: int) -> HeteroGraphSnapshotSequence:
        """Return a contiguous temporal sub-sequence.

        Parameters
        ----------
        start : int
            Inclusive start index.
        stop : int
            Exclusive stop index.

        Returns
        -------
        HeteroGraphSnapshotSequence
            Snapshots in ``[start, stop)`` with matching metadata.

        Raises
        ------
        ValueError
            If the bounds are invalid.
        """
        if start < 0 or stop <= start or stop > self.num_timesteps:
            msg = (
                "slice bounds must satisfy "
                f"0 <= start < stop <= {self.num_timesteps}, "
                f"got start={start}, stop={stop}"
            )
            raise ValueError(msg)

        controls: Tensor | dict[str, Tensor] | None
        if self._control_inputs is None:
            controls = None
        elif isinstance(self._control_inputs, Mapping):
            controls = {
                node_type: tensor[start:stop]
                for node_type, tensor in self._control_inputs.items()
            }
        else:
            controls = self._control_inputs[start:stop]

        masks = None
        if self._observation_masks is not None:
            masks = {
                node_type: mask[start:stop]
                for node_type, mask in self._observation_masks.items()
            }

        return HeteroGraphSnapshotSequence(
            self._snapshots[start:stop],
            allow_dynamic_topology=self.allow_dynamic_topology,
            control_inputs=controls,
            timestamps=(
                None if self._timestamps is None else self._timestamps[start:stop]
            ),
            observation_masks=masks,
        )
