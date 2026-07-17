"""Snapshot sequence containers and multi-trajectory helpers."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data

from koopman_graph.graph_utils import snapshot_edge_weight

ArrayLike = Tensor | np.ndarray


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


def _validate_timestamps(
    timestamps: Tensor,
    *,
    num_timesteps: int,
) -> None:
    """Validate optional per-snapshot timestamps.

    Parameters
    ----------
    timestamps : Tensor
        Monotone increasing timestamps with shape ``(num_timesteps,)``.
    num_timesteps : int
        Expected number of snapshots.

    Raises
    ------
    ValueError
        If ``timestamps`` has invalid shape or is not strictly increasing.
    """
    if timestamps.ndim != 1:
        msg = (
            "timestamps must have shape (num_timesteps,), "
            f"got {tuple(timestamps.shape)}"
        )
        raise ValueError(msg)
    if timestamps.shape[0] != num_timesteps:
        msg = f"timestamps has {timestamps.shape[0]} entries, expected {num_timesteps}"
        raise ValueError(msg)
    if num_timesteps >= 2:
        deltas = timestamps[1:] - timestamps[:-1]
        if not torch.all(deltas > 0):
            msg = "timestamps must be strictly monotone increasing"
            raise ValueError(msg)


def _validate_observation_masks(
    observation_masks: Tensor,
    *,
    num_timesteps: int,
    num_nodes: int,
) -> Tensor:
    """Validate optional per-snapshot node observation masks.

    Parameters
    ----------
    observation_masks : Tensor
        Boolean or 0/1 mask with shape ``(num_timesteps, num_nodes)``.
        ``True`` (or ``1``) means the node is observed at that timestep.
    num_timesteps : int
        Expected number of snapshots.
    num_nodes : int
        Expected node count.

    Returns
    -------
    Tensor
        Boolean mask with shape ``(num_timesteps, num_nodes)``.

    Raises
    ------
    ValueError
        If ``observation_masks`` has invalid shape, dtype, or no observed nodes
        at any timestep.
    """
    if observation_masks.ndim != 2:
        msg = (
            "observation_masks must have shape (num_timesteps, num_nodes), "
            f"got {tuple(observation_masks.shape)}"
        )
        raise ValueError(msg)
    if observation_masks.shape != (num_timesteps, num_nodes):
        msg = (
            "observation_masks shape "
            f"{tuple(observation_masks.shape)} does not match "
            f"(num_timesteps={num_timesteps}, num_nodes={num_nodes})"
        )
        raise ValueError(msg)
    if observation_masks.dtype not in (torch.bool, torch.float, torch.int, torch.long):
        msg = (
            "observation_masks must be boolean or numeric 0/1, "
            f"got dtype {observation_masks.dtype}"
        )
        raise ValueError(msg)

    if observation_masks.dtype != torch.bool:
        unique = torch.unique(observation_masks)
        if not torch.all((unique == 0) | (unique == 1)):
            msg = "numeric observation_masks must contain only 0 and 1"
            raise ValueError(msg)

    mask = observation_masks.bool()
    empty_timesteps = torch.where(~mask.any(dim=1))[0]
    if empty_timesteps.numel() > 0:
        msg = (
            "observation_masks must have at least one observed node per timestep; "
            f"timesteps with no observations: {empty_timesteps.tolist()}"
        )
        raise ValueError(msg)
    return mask


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
    ref_edge_weight = snapshot_edge_weight(reference)

    for idx, snapshot in enumerate(snapshots[1:], start=1):
        if not torch.equal(snapshot.edge_index, ref_edge_index):
            msg = f"Snapshot {idx} has a different edge_index than snapshot 0"
            raise ValueError(msg)
        edge_weight = snapshot_edge_weight(snapshot)
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


class GraphSnapshotSequence:
    """Container for a time-ordered sequence of PyG ``Data`` graph snapshots.

    By default all snapshots must share the same ``edge_index``, optional
    ``edge_weight``, node count, and feature dimension. Set
    ``allow_dynamic_topology=True`` to permit per-snapshot ``edge_index`` while
    still requiring a fixed node count and feature dimension. Optional
    :attr:`control_inputs` store exogenous inputs ``u_t`` applied when
    advancing from snapshot ``t`` to ``t+1`` (global ``(T, C)`` or per-node
    ``(T, N, C)``). Not every consumer supports both layouts: neural model /
    adaptation preserve per-node rows; :class:`~koopman_graph.env.GraphKoopmanEnv`
    and :class:`~koopman_graph.baselines.DMDcBaseline` are global-only (see
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
    :attr:`edge_index`, :attr:`edge_weight`, :attr:`is_dynamic_topology`,
    :attr:`control_inputs`, :attr:`has_controls`, :attr:`control_dim`,
    :attr:`timestamps`, :attr:`has_timestamps`, :attr:`observation_masks`,
    :attr:`has_observation_masks`, and :attr:`num_nodes`,
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
            _validate_snapshot_metadata(snapshot_list)
        else:
            _validate_shared_topology(snapshot_list)
        if control_inputs is not None:
            _validate_control_inputs(
                control_inputs,
                num_timesteps=len(snapshot_list),
                num_nodes=int(snapshot_list[0].num_nodes),
            )
        if timestamps is not None:
            _validate_timestamps(
                timestamps,
                num_timesteps=len(snapshot_list),
            )
        validated_masks = None
        if observation_masks is not None:
            validated_masks = _validate_observation_masks(
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
            If ``node_features``, ``edge_index``, or ``edge_weight`` have
            invalid shape.
        """
        features = _as_tensor(node_features, dtype=dtype)
        edges = _as_tensor(edge_index, dtype=torch.long)
        weights = None if edge_weight is None else _as_tensor(edge_weight, dtype=dtype)
        controls = (
            None if control_inputs is None else _as_tensor(control_inputs, dtype=dtype)
        )
        times = None if timestamps is None else _as_tensor(timestamps, dtype=dtype)
        masks = (
            None
            if observation_masks is None
            else _as_tensor(observation_masks, dtype=torch.bool)
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
        return cls(
            snapshots,
            control_inputs=controls,
            timestamps=times,
            observation_masks=masks,
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
        times = None if timestamps is None else _as_tensor(timestamps, dtype=dtype)
        masks = (
            None
            if observation_masks is None
            else _as_tensor(observation_masks, dtype=torch.bool)
        )
        return cls(
            snapshots,
            allow_dynamic_topology=True,
            control_inputs=controls,
            timestamps=times,
            observation_masks=masks,
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

        Raises
        ------
        ValueError
            If timestamps are absent or ``index`` is out of range.
        """
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

    def __iter__(self) -> Iterator[Data]:
        """Iterate over graph snapshots in temporal order.

        Yields
        ------
        Data
            Graph snapshot at each timestep.
        """
        return iter(self._snapshots)


@dataclass(frozen=True)
class MultiTrajectory:
    """Explicit multi-trajectory container for training and validation input.

    Prefer this over a bare ``list[GraphSnapshotSequence]`` when calling
    :meth:`~koopman_graph.model.GraphKoopmanModel.fit` so multi-trajectory
    intent cannot be confused with a single trajectory of ``Data`` snapshots.
    A plain list of :class:`GraphSnapshotSequence` remains accepted as a
    compatibility shim.

    Attributes
    ----------
    sequences : tuple of GraphSnapshotSequence
        Non-empty trajectories of the same system.
    """

    sequences: tuple[GraphSnapshotSequence, ...]

    def __post_init__(self) -> None:
        """Validate that ``sequences`` is a non-empty trajectory tuple.

        Raises
        ------
        ValueError
            If ``sequences`` is empty.
        TypeError
            If any element is not a :class:`GraphSnapshotSequence`.
        """
        if not self.sequences:
            msg = "MultiTrajectory requires at least one GraphSnapshotSequence"
            raise ValueError(msg)
        for index, sequence in enumerate(self.sequences):
            if not isinstance(sequence, GraphSnapshotSequence):
                msg = (
                    "MultiTrajectory sequences must be GraphSnapshotSequence "
                    f"instances; index {index} has type {type(sequence).__name__}"
                )
                raise TypeError(msg)

    def __len__(self) -> int:
        """Return the number of trajectories.

        Returns
        -------
        int
            Length of :attr:`sequences`.
        """
        return len(self.sequences)

    def __iter__(self) -> Iterator[GraphSnapshotSequence]:
        """Iterate over trajectories.

        Yields
        ------
        GraphSnapshotSequence
            Each trajectory in order.
        """
        return iter(self.sequences)

    def __getitem__(self, index: int) -> GraphSnapshotSequence:
        """Return the trajectory at ``index``.

        Parameters
        ----------
        index : int
            Trajectory index.

        Returns
        -------
        GraphSnapshotSequence
            Trajectory at the requested index.
        """
        return self.sequences[index]


def as_multi_trajectory(
    *trajectories: GraphSnapshotSequence | Sequence[GraphSnapshotSequence],
) -> MultiTrajectory:
    """Build a :class:`MultiTrajectory` from sequences or a sequence of sequences.

    Parameters
    ----------
    *trajectories
        Either one iterable of :class:`GraphSnapshotSequence`, or individual
        sequences passed as separate arguments.

    Returns
    -------
    MultiTrajectory
        Validated multi-trajectory container.

    Raises
    ------
    TypeError
        If arguments are not snapshot sequences.
    ValueError
        If no trajectories are provided.
    """
    if len(trajectories) == 1 and not isinstance(
        trajectories[0], GraphSnapshotSequence
    ):
        sequence_list = list(trajectories[0])
    else:
        sequence_list = list(trajectories)
    return MultiTrajectory(tuple(sequence_list))


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


def resolve_pair_delta_t(
    sequence: GraphSnapshotSequence,
    timestep: int,
    *,
    default_time_step: float,
) -> float:
    """Return the integration interval for transition ``timestep -> timestep + 1``.

    Parameters
    ----------
    sequence : GraphSnapshotSequence
        Snapshot sequence that may carry timestamps.
    timestep : int
        Source snapshot index.
    default_time_step : float
        Fallback interval when timestamps are absent.

    Returns
    -------
    float
        Positive integration interval.
    """
    if sequence.has_timestamps:
        return float(sequence.delta_t_at(timestep).item())
    return default_time_step
