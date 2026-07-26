"""Structural typing for snapshot sequences without importing containers.

Leaf module: construction / delay_windows annotate against these Protocols so
the data package import graph stays ``containers → construction/delay_windows``
(no reverse edges). Inventory still counts ``TYPE_CHECKING`` imports, so this
module must not import :mod:`koopman_graph.data.containers`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import Tensor
from torch_geometric.data import Data


@runtime_checkable
class SnapshotSequenceLike(Protocol):
    """Minimal sequence surface used by delay-window / Hankel builders.

    Notes
    -----
    Satisfied by :class:`~koopman_graph.data.GraphSnapshotSequence`. Kept free
    of ``containers`` imports so construction / delay peers stay acyclic.
    """

    @property
    def num_timesteps(self) -> int:
        """Number of snapshots in the trajectory.

        Returns
        -------
        int
            Trajectory length ``T``.
        """
        ...

    @property
    def allow_dynamic_topology(self) -> bool:
        """Whether per-snapshot topology changes are permitted.

        Returns
        -------
        bool
            ``True`` when edge topology may vary over time.
        """
        ...

    @property
    def has_observation_masks(self) -> bool:
        """Whether observation masks are attached.

        Returns
        -------
        bool
            ``True`` when masks are present for every timestep.
        """
        ...

    @property
    def control_inputs(self) -> Tensor | None:
        """Optional control tensor aligned with timesteps.

        Returns
        -------
        Tensor or None
            Controls with leading time dimension, or ``None``.
        """
        ...

    @property
    def timestamps(self) -> Tensor | None:
        """Optional timestamp tensor aligned with timesteps.

        Returns
        -------
        Tensor or None
            Timestamps with shape ``(T,)``, or ``None``.
        """
        ...

    @property
    def observation_masks(self) -> Tensor | None:
        """Optional boolean observation masks ``(T, N)``.

        Returns
        -------
        Tensor or None
            Mask stack, or ``None`` when fully observed.
        """
        ...

    def __getitem__(self, index: int) -> Data:
        """Return the snapshot at ``index``.

        Parameters
        ----------
        index : int
            Timestep index in ``[0, num_timesteps)``.

        Returns
        -------
        Data
            PyG snapshot at ``index``.
        """
        ...

    def observation_mask_at(self, index: int) -> Tensor | None:
        """Return the observation mask at ``index``, if any.

        Parameters
        ----------
        index : int
            Timestep index in ``[0, num_timesteps)``.

        Returns
        -------
        Tensor or None
            Boolean mask for that timestep, or ``None``.
        """
        ...
