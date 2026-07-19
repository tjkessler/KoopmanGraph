"""Shared scaffolding for classical DMD-family baselines.

Module-level helpers (``require_static_topology``, ``flatten_snapshots``,
``fit_row_operator``, ``fit_controlled_row_operator``,
``require_global_controls``, ``transition_controls``, ``copy_topology``,
``check_initial_graph``) are documented non-private power-user symbols for
classical and GNN baseline peers. They are not re-exported from package or
root ``__all__``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor
from torch_geometric.data import Data

from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.graph_utils import snapshot_edge_weight


def require_static_topology(sequence: GraphSnapshotSequence) -> None:
    """Reject dynamic-topology sequences for classical baselines.

    Classical DMD-family baselines flatten node states and copy only the
    initial graph topology onto predictions. Fitting on time-varying edges
    would silently ignore topology changes.

    Parameters
    ----------
    sequence : GraphSnapshotSequence
        Candidate training sequence.

    Raises
    ------
    ValueError
        If ``sequence.is_dynamic_topology`` is ``True``.
    """
    if sequence.is_dynamic_topology:
        msg = (
            "classical baselines require a fixed graph topology; "
            "got a sequence with is_dynamic_topology=True"
        )
        raise ValueError(msg)


def flatten_snapshots(sequence: GraphSnapshotSequence) -> Tensor:
    """Stack graph snapshot features into row-vector states.

    Parameters
    ----------
    sequence : GraphSnapshotSequence
        Graph snapshots to flatten.

    Returns
    -------
    Tensor
        Flattened state matrix with shape ``(num_timesteps, state_dim)``.

    Raises
    ------
    TypeError
        If node features are not floating-point tensors.
    """
    states = [snapshot.x.reshape(-1) for snapshot in sequence]
    if not states:
        msg = "sequence must contain at least one snapshot"
        raise ValueError(msg)
    if not states[0].is_floating_point():
        msg = f"snapshot features must be floating-point, got {states[0].dtype}"
        raise TypeError(msg)
    return torch.stack(states)


def fit_row_operator(left: Tensor, right: Tensor, rank: int | None) -> Tensor:
    """Fit ``right ~= left @ A`` and return ``K`` for ``x_next = x @ K.T``.

    Parameters
    ----------
    left : Tensor
        Source states or observables with shape ``(num_samples, state_dim)``.
    right : Tensor
        Target states or observables with shape ``(num_samples, state_dim)``.
    rank : int or None
        Optional truncated-SVD rank. ``None`` uses full least squares.

    Returns
    -------
    Tensor
        Row-convention Koopman matrix ``K`` with shape
        ``(state_dim, state_dim)``.

    Raises
    ------
    ValueError
        If ``rank`` is outside the valid range for the data matrix.
    """
    if rank is None:
        solution = torch.linalg.lstsq(left, right).solution
        return solution.T

    if rank < 1:
        msg = f"rank must be >= 1 when provided, got {rank}"
        raise ValueError(msg)
    max_rank = min(left.shape)
    if rank > max_rank:
        msg = f"rank must be <= {max_rank} for data matrix shape {tuple(left.shape)}"
        raise ValueError(msg)

    u, singular_values, vh = torch.linalg.svd(left, full_matrices=False)
    u_r = u[:, :rank]
    s_r = singular_values[:rank]
    vh_r = vh[:rank, :]
    solution = vh_r.T @ ((u_r.T @ right) / s_r.unsqueeze(1))
    return solution.T


def fit_controlled_row_operator(
    left: Tensor,
    right: Tensor,
    controls: Tensor,
    rank: int | None,
) -> tuple[Tensor, Tensor]:
    """Fit ``right ~= left @ K.T + controls @ B``.

    Parameters
    ----------
    left : Tensor
        Source states with shape ``(num_samples, state_dim)``.
    right : Tensor
        Target states with shape ``(num_samples, state_dim)``.
    controls : Tensor
        Control inputs with shape ``(num_samples, control_dim)``.
    rank : int or None
        Optional truncated-SVD rank for the augmented regression.

    Returns
    -------
    tuple of Tensor
        ``(K, B)`` with shapes ``(state_dim, state_dim)`` and
        ``(control_dim, state_dim)``.
    """
    if controls.ndim != 2:
        msg = (
            "controls must have shape (num_samples, control_dim), "
            f"got {tuple(controls.shape)}"
        )
        raise ValueError(msg)
    if controls.shape[0] != left.shape[0]:
        msg = f"controls has {controls.shape[0]} samples, expected {left.shape[0]}"
        raise ValueError(msg)
    augmented = torch.cat([left, controls], dim=-1)
    joint = fit_row_operator(augmented, right, rank)
    state_dim = left.shape[1]
    k_matrix = joint[:, :state_dim]
    b_matrix = joint[:, state_dim:].T
    return k_matrix, b_matrix


def require_global_controls(sequence: GraphSnapshotSequence) -> None:
    """Reject per-node (3-D) control layouts for classical DMDc.

    :class:`~koopman_graph.baselines.DMDcBaseline` fits a single global control
    vector per transition on flattened joint states. Accepting
    ``(T, N, control_dim)`` and flattening would silently encode different
    physics than neural / adaptation per-node row matching.

    Parameters
    ----------
    sequence : GraphSnapshotSequence
        Candidate training sequence with control inputs.

    Raises
    ------
    ValueError
        If controls are missing or have per-node (3-D) layout.
    """
    if not sequence.has_controls or sequence.control_inputs is None:
        msg = "sequence does not contain control inputs"
        raise ValueError(msg)
    if sequence.control_inputs.ndim == 3:
        msg = (
            "DMDcBaseline does not support per-node (3-D) control_inputs with "
            "shape (T, N, control_dim); use global controls with shape "
            "(T, control_dim). Neural GraphKoopmanModel / "
            "RecursiveKoopmanAdapter preserve per-node control rows — see the "
            "architecture control layout capability matrix"
        )
        raise ValueError(msg)


def transition_controls(sequence: GraphSnapshotSequence) -> Tensor:
    """Return global control inputs aligned with consecutive transitions.

    Parameters
    ----------
    sequence : GraphSnapshotSequence
        Sequence with global (2-D) controls.

    Returns
    -------
    Tensor
        Controls with shape ``(num_timesteps - 1, control_dim)``.

    Raises
    ------
    ValueError
        If controls are missing or have per-node (3-D) layout.
    """
    require_global_controls(sequence)
    controls = sequence.control_inputs
    if controls is None:  # pragma: no cover - guarded by require_global_controls
        msg = "sequence does not contain control inputs"
        raise ValueError(msg)
    return controls[:-1]


def copy_topology(initial_graph: Data) -> dict[str, Tensor]:
    """Copy topology tensors for a predicted PyG snapshot.

    Parameters
    ----------
    initial_graph : Data
        Graph snapshot providing ``edge_index`` and optional ``edge_weight``.

    Returns
    -------
    dict of str to Tensor
        Topology fields suitable for constructing a predicted ``Data`` object.
    """
    fields = {"edge_index": initial_graph.edge_index}
    edge_weight = snapshot_edge_weight(initial_graph)
    if edge_weight is not None:
        fields["edge_weight"] = edge_weight
    return fields


def check_initial_graph(
    initial_graph: Data,
    *,
    num_nodes: int,
    in_channels: int,
) -> None:
    """Validate an initial graph shape against fitted baseline metadata.

    Parameters
    ----------
    initial_graph : Data
        Initial graph snapshot for autoregressive prediction.
    num_nodes : int
        Node count recorded when the baseline was fit.
    in_channels : int
        Feature dimension recorded when the baseline was fit.

    Raises
    ------
    ValueError
        If node count or feature dimension does not match fitted metadata.
    """
    if initial_graph.num_nodes != num_nodes:
        msg = f"initial graph has {initial_graph.num_nodes} nodes, expected {num_nodes}"
        raise ValueError(msg)
    if initial_graph.x.shape[1] != in_channels:
        msg = (
            f"initial graph has feature dimension {initial_graph.x.shape[1]}, "
            f"expected {in_channels}"
        )
        raise ValueError(msg)


class ClassicalBaseline(ABC):
    """Shared scaffolding for classical DMD-family baselines.

    Holds common ``time_step`` / ``rank`` configuration, fitted graph metadata,
    and fitted-state guards. Concrete baselines implement :meth:`_is_fitted` and
    the :class:`~koopman_graph.protocols.ForecastModel` surface (``fit`` /
    ``predict`` / ``spectrum``). The Protocol remains the typing façade; this
    ABC is the implementation scaffold.

    Attributes
    ----------
    time_step : float
        Physical duration represented by one snapshot transition.
    rank : int or None
        Optional truncated-SVD rank. ``None`` uses full least squares.
    K : Tensor or None
        Fitted Koopman matrix, or ``None`` before :meth:`fit`.
    num_nodes : int or None
        Node count recorded at fit time.
    in_channels : int or None
        Feature dimension recorded at fit time.
    state_dim : int or None
        Flattened state dimension recorded at fit time.
    """

    def __init__(self, *, time_step: float = 1.0, rank: int | None = None) -> None:
        """Initialize shared baseline hyperparameters.

        Parameters
        ----------
        time_step : float, optional
            Physical duration represented by one snapshot transition. Default
            is ``1.0``.
        rank : int or None, optional
            Optional truncated-SVD rank. ``None`` uses full least squares.

        Raises
        ------
        ValueError
            If ``time_step`` is not positive.
        """
        if time_step <= 0:
            msg = f"time_step must be positive, got {time_step}"
            raise ValueError(msg)
        self.time_step = float(time_step)
        self.rank = rank
        self.K: Tensor | None = None
        self.num_nodes: int | None = None
        self.in_channels: int | None = None
        self.state_dim: int | None = None

    @abstractmethod
    def _is_fitted(self) -> bool:
        """Return whether fit-time operators and metadata are available.

        Returns
        -------
        bool
            ``True`` when the baseline can run ``predict`` / ``spectrum``.
        """

    def _unfitted_message(self) -> str:
        """Return the class-specific unfitted error message.

        Returns
        -------
        str
            Message used by :meth:`_check_fitted` and require helpers.
        """
        return (
            f"{type(self).__name__} must be fit before prediction or spectral analysis"
        )

    def _check_fitted(self) -> None:
        """Raise if the baseline has not been fit.

        Raises
        ------
        RuntimeError
            If required fitted state is missing.
        """
        if not self._is_fitted():
            raise RuntimeError(self._unfitted_message())

    def _require_operator(self) -> Tensor:
        """Return the fitted Koopman matrix after a fitted-state check.

        Returns
        -------
        Tensor
            Fitted operator ``K``.

        Raises
        ------
        RuntimeError
            If the baseline has not been fit.
        """
        self._check_fitted()
        if self.K is None:
            raise RuntimeError(self._unfitted_message())
        return self.K

    def _require_graph_metadata(self) -> tuple[int, int]:
        """Return ``(num_nodes, in_channels)`` after a fitted-state check.

        Returns
        -------
        tuple of int
            Fitted node count and feature dimension.

        Raises
        ------
        RuntimeError
            If the baseline has not been fit.
        """
        self._check_fitted()
        if self.num_nodes is None or self.in_channels is None:
            raise RuntimeError(self._unfitted_message())
        return self.num_nodes, self.in_channels
