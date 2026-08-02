"""Depth-2 multi-resolution DMD baseline.

Builds a coarse root DMD on the full window plus two child DMDs on
half-window residuals after a slow-mode filter. Autoregressive
``predict`` uses the root operator only (``x_next = x @ K.T``) so
noise-free linear recovery matches :class:`~koopman_graph.baselines.DMDBaseline`.
Primary-source citations are deferred to Sphinx Phase 61 verification.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from torch import Tensor
from torch_geometric.data import Data

from koopman_graph.baselines.base import (
    ClassicalBaseline,
    RankSpec,
    check_initial_graph,
    copy_topology,
    fit_row_operator,
    flatten_snapshots,
    require_static_topology,
    resolve_fit_rank,
)
from koopman_graph.data import (
    GraphSnapshotSequence,
    resolve_sequence,
)
from koopman_graph.spectrum_types import KoopmanSpectrum, compute_spectrum


@dataclass
class MRDMDNode:
    """One node in a depth-2 multi-resolution DMD tree.

    Parameters
    ----------
    level : int
        Tree depth (``0`` root, ``1`` children).
    start : int
        Inclusive snapshot index in the training window.
    stop : int
        Exclusive snapshot index.
    K : Tensor
        Local row-convention operator for this window.
    eigenvalues : Tensor
        Eigenvalues of ``K``.
    children : list of MRDMDNode
        Child nodes (empty at leaves).
    """

    level: int
    start: int
    stop: int
    K: Tensor
    eigenvalues: Tensor
    children: list[MRDMDNode] = field(default_factory=list)


def _slow_mode_mask(eigenvalues: Tensor, num_timesteps: int) -> Tensor:
    """Return a boolean mask for slow eigenvalues (``|arg(λ)| ≤ π/T``).

    Parameters
    ----------
    eigenvalues
        Value for ``eigenvalues``.
    num_timesteps
        Value for ``num_timesteps``.

    Returns
    -------
    object
        Function result.
    """
    return eigenvalues.angle().abs() <= (math.pi / float(num_timesteps))


def _slow_trajectory(states: Tensor, operator: Tensor) -> Tensor:
    """Reconstruct the slow-mode trajectory under ``operator``.

    Parameters
    ----------
    states : Tensor
        Training states ``(T, state_dim)``.
    operator : Tensor
        Root Koopman matrix ``K`` with ``x_next = x @ K.T``.

    Returns
    -------
    Tensor
        Slow reconstruction with the same shape as ``states``.
    """
    num_timesteps = int(states.shape[0])
    working = operator.to(dtype=torch.complex128)
    eigenvalues, eigenvectors = torch.linalg.eig(working)
    slow = _slow_mode_mask(eigenvalues, num_timesteps)
    if not bool(slow.any()):
        return torch.zeros_like(states)

    initial = states[0].to(dtype=torch.complex128)
    coefficients = torch.linalg.solve(eigenvectors, initial)
    slow_coefficients = torch.where(slow, coefficients, torch.zeros_like(coefficients))
    trajectory = []
    for step in range(num_timesteps):
        amplitudes = slow_coefficients * (eigenvalues**step)
        reconstructed = eigenvectors @ amplitudes
        trajectory.append(reconstructed.real.to(dtype=states.dtype))
    return torch.stack(trajectory, dim=0)


class MRDMDBaseline(ClassicalBaseline):
    """Depth-2 multi-resolution DMD on flattened node states.

    Fits a root DMD on the full sequence, subtracts a slow-mode
    reconstruction, and fits child DMDs on each residual half-window.
    Forecasts use the root ``K`` only.

    Satisfies :class:`~koopman_graph.protocols.ForecastModel` and
    :class:`~koopman_graph.protocols.UncontrolledForecastModel`.

    Parameters
    ----------
    time_step : float, optional
        Physical duration represented by one snapshot transition. Used by
        :meth:`spectrum`. Default is ``1.0``.
    rank : int or None or {"auto"}, optional
        Truncated-SVD rank for each window fit. Default is ``None``.
    """

    def __init__(self, *, time_step: float = 1.0, rank: RankSpec = None) -> None:
        """Initialize the depth-2 mrDMD baseline.

        Parameters
        ----------
        time_step
            Value for ``time_step``.
        rank
            Value for ``rank``.
        """
        super().__init__(time_step=time_step, rank=rank)
        self.root: MRDMDNode | None = None

    def _is_fitted(self) -> bool:
        """Return whether the root operator has been fit.

        Returns
        -------
        object
            Function result.
        """
        return self.K is not None and self.root is not None

    def fit(
        self,
        sequence: GraphSnapshotSequence | Sequence[Data],
    ) -> MRDMDBaseline:
        """Fit a depth-2 multi-resolution DMD tree.

        Parameters
        ----------
        sequence : GraphSnapshotSequence or sequence of Data
            Training snapshots with shared topology. Requires at least four
            snapshots so each half-window has two or more samples.

        Returns
        -------
        MRDMDBaseline
            The fitted baseline (``self``) for sklearn-style chaining.

        Raises
        ------
        ValueError
            If fewer than four snapshots are provided, topology varies, or
            rank is invalid.
        """
        resolved = resolve_sequence(sequence)
        require_static_topology(resolved)
        if resolved.num_timesteps < 4:
            msg = (
                "MRDMDBaseline.fit requires at least four snapshots for a depth-2 tree"
            )
            raise ValueError(msg)

        states = flatten_snapshots(resolved)
        left = states[:-1]
        self.selected_rank = resolve_fit_rank(left, self.rank)
        k_root = fit_row_operator(left, states[1:], self.selected_rank)
        root_eigs = torch.linalg.eigvals(k_root.to(dtype=torch.complex128))

        slow = _slow_trajectory(states, k_root)
        residual = states - slow
        midpoint = residual.shape[0] // 2
        children: list[MRDMDNode] = []
        for start, stop in ((0, midpoint), (midpoint, residual.shape[0])):
            window = residual[start:stop]
            if window.shape[0] < 2:
                continue
            child_left = window[:-1]
            child_rank = resolve_fit_rank(child_left, self.rank)
            child_k = fit_row_operator(child_left, window[1:], child_rank)
            children.append(
                MRDMDNode(
                    level=1,
                    start=start,
                    stop=stop,
                    K=child_k,
                    eigenvalues=torch.linalg.eigvals(
                        child_k.to(dtype=torch.complex128)
                    ),
                )
            )

        self.root = MRDMDNode(
            level=0,
            start=0,
            stop=int(states.shape[0]),
            K=k_root,
            eigenvalues=root_eigs,
            children=children,
        )
        self.K = k_root
        self.num_nodes = resolved.num_nodes
        self.in_channels = resolved.in_channels
        self.state_dim = states.shape[1]
        return self

    def predict(self, initial_graph: Data, steps: int) -> list[Data]:
        """Autoregressively predict using the root operator (Data-only).

        Parameters
        ----------
        initial_graph
            Value for ``initial_graph``.
        steps
            Value for ``steps``.

        Returns
        -------
        object
            Function result.
        """
        operator = self._require_operator()
        num_nodes, in_channels = self._require_graph_metadata()
        if steps < 1:
            msg = f"steps must be >= 1, got {steps}"
            raise ValueError(msg)
        check_initial_graph(
            initial_graph,
            num_nodes=num_nodes,
            in_channels=in_channels,
        )

        state = initial_graph.x.reshape(-1)
        topology = copy_topology(initial_graph)
        predictions: list[Data] = []
        for _ in range(steps):
            state = state @ operator.T
            x = state.reshape(num_nodes, in_channels)
            predictions.append(Data(x=x, **topology))
        return predictions

    def spectrum(self) -> KoopmanSpectrum:
        """Return the root mrDMD operator spectrum.

        Returns
        -------
        object
            Function result.
        """
        return compute_spectrum(self._require_operator(), self.time_step)
