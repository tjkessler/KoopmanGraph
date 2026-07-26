"""Shared scaffolding for classical DMD-family baselines.

Module-level helpers (``require_static_topology``, ``flatten_snapshots``,
``fit_row_operator``, ``fit_controlled_row_operator``,
``require_global_controls``, ``transition_controls``, ``copy_topology``,
``check_initial_graph``, ``optimal_hard_threshold_rank``) are documented
non-private power-user symbols for classical and GNN baseline peers. They
are not re-exported from package or root ``__all__``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import torch
from torch import Tensor
from torch_geometric.data import Data

from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.data.validation import require_no_hyperedges
from koopman_graph.graph_utils import snapshot_edge_weight

RankSpec = int | None | Literal["auto"]


def require_static_topology(sequence: GraphSnapshotSequence) -> None:
    """Reject dynamic-topology or hyperedge sequences for classical baselines.

    Classical DMD-family baselines flatten node states and copy only the
    initial graph topology onto predictions. Fitting on time-varying edges
    or hyperedge incidence would silently ignore that structure.

    Parameters
    ----------
    sequence : GraphSnapshotSequence
        Candidate training sequence.

    Raises
    ------
    ValueError
        If ``sequence.is_dynamic_topology`` is ``True`` or the sequence
        carries ``hyperedge_index``.
    """
    if sequence.is_dynamic_topology:
        msg = (
            "classical baselines require a fixed graph topology; "
            "got a sequence with is_dynamic_topology=True"
        )
        raise ValueError(msg)
    require_no_hyperedges(sequence)


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


def gavish_donoho_omega(beta: float) -> float:
    """Cubic approximation to the unknown-``σ`` coefficient ``ω(β)``.

    Parameters
    ----------
    beta : float
        Aspect ratio ``β = min(m, n) / max(m, n)`` in ``(0, 1]``.

    Returns
    -------
    float
        Approximate ``ω(β)``. At ``β = 1`` this recovers ``≈ 2.858``.

    Raises
    ------
    ValueError
        If ``beta`` is outside ``(0, 1]``.

    Notes
    -----
    Gavish & Donoho (2014) give ``τ̂* = ω(β) · y_med`` for unknown noise
    level. Exact ``ω(β) = λ*(β) / √μ_β`` requires the Marchenko–Pastur
    median; the published cubic
    ``0.56 β³ − 0.95 β² + 1.82 β + 1.43`` is used here.

    References
    ----------
    Gavish, M. & Donoho, D. L. (2014). The optimal hard threshold for
    singular values is ``4/√3``. *IEEE Transactions on Information Theory*,
    60(8), 5040–5053. https://doi.org/10.1109/TIT.2014.2323359
    (``GavishDonoho2014``)
    """
    if not 0.0 < beta <= 1.0:
        msg = f"beta must lie in (0, 1], got {beta}"
        raise ValueError(msg)
    return ((0.56 * beta - 0.95) * beta + 1.82) * beta + 1.43


def optimal_hard_threshold_rank(
    singular_values: Tensor,
    *,
    num_rows: int,
    num_cols: int,
) -> int:
    """Select truncated-SVD rank by the Gavish–Donoho median threshold.

    Implements the unknown-``σ`` rule ``τ̂* = ω(β) · y_med`` where
    ``y_med`` is the median empirical singular value and
    ``β = min(m, n) / max(m, n)`` (transpose convention so ``β ∈ (0, 1]``).
    Singular values strictly above ``τ̂*`` are retained.

    For the square case ``β = 1``, ``ω(1) ≈ 2.858``. The known-``σ``
    companion threshold is ``τ* = λ*(β) √n σ`` with
    ``λ*(1) = 4/√3 ≈ 2.309``; this helper implements only the
    unknown-``σ`` median form.

    Assumptions (asymptotic optimality): additive white noise, matrix
    dimensions large relative to the signal rank, and a constant
    signal-to-noise ratio. Finite-sample recovery is not guaranteed.

    Parameters
    ----------
    singular_values : Tensor
        Non-increasing singular values from ``torch.linalg.svd`` (1-D).
    num_rows : int
        Number of rows ``m`` of the data matrix.
    num_cols : int
        Number of columns ``n`` of the data matrix.

    Returns
    -------
    int
        Selected rank in ``0 .. len(singular_values)``. ``0`` means every
        singular value fell at or below the threshold (e.g. all-zero data).

    Raises
    ------
    ValueError
        If shapes / dimensions are invalid or singular values are
        non-finite.

    References
    ----------
    Gavish, M. & Donoho, D. L. (2014). The optimal hard threshold for
    singular values is ``4/√3``. *IEEE Transactions on Information Theory*,
    60(8), 5040–5053. https://doi.org/10.1109/TIT.2014.2323359
    (``GavishDonoho2014``)
    """
    if singular_values.ndim != 1:
        msg = f"singular_values must be 1-D, got shape {tuple(singular_values.shape)}"
        raise ValueError(msg)
    if num_rows < 1 or num_cols < 1:
        msg = f"num_rows and num_cols must be >= 1, got {num_rows}, {num_cols}"
        raise ValueError(msg)
    if singular_values.numel() == 0:
        return 0
    if not torch.isfinite(singular_values).all():
        msg = "singular_values must be finite (no NaN/Inf)"
        raise ValueError(msg)

    beta = min(num_rows, num_cols) / max(num_rows, num_cols)
    omega = gavish_donoho_omega(beta)
    y_med = float(torch.median(singular_values).item())
    if y_med <= 0.0:
        return 0
    threshold = omega * y_med
    return int((singular_values > threshold).sum().item())


def resolve_fit_rank(left: Tensor, rank: RankSpec) -> int | None:
    """Resolve ``rank`` to ``None`` (full LS) or a positive truncation.

    Parameters
    ----------
    left : Tensor
        Data matrix with shape ``(num_samples, feature_dim)`` whose SVD
        would be truncated.
    rank : int or None or {"auto"}
        Truncation request. ``None`` keeps full least squares. ``"auto"``
        applies :func:`optimal_hard_threshold_rank` to the singular values
        of ``left``.

    Returns
    -------
    int or None
        ``None`` for full least squares, otherwise a positive rank.

    Raises
    ------
    ValueError
        If ``rank`` is invalid or ``"auto"`` selects rank 0.
    """
    if rank is None:
        return None
    if rank == "auto":
        if left.ndim != 2:
            msg = f"left must be 2-D for rank='auto', got shape {tuple(left.shape)}"
            raise ValueError(msg)
        singular_values = torch.linalg.svdvals(left)
        selected = optimal_hard_threshold_rank(
            singular_values,
            num_rows=int(left.shape[0]),
            num_cols=int(left.shape[1]),
        )
        if selected < 1:
            msg = (
                "rank='auto' selected rank 0 (all singular values at or "
                "below the Gavish–Donoho threshold). Provide a denser / "
                "less noisy data matrix or set rank to a positive integer."
            )
            raise ValueError(msg)
        return selected
    if not isinstance(rank, int):
        msg = f"rank must be int, None, or 'auto', got {rank!r}"
        raise ValueError(msg)
    if rank < 1:
        msg = f"rank must be >= 1 when provided, got {rank}"
        raise ValueError(msg)
    max_rank = min(left.shape)
    if rank > max_rank:
        msg = f"rank must be <= {max_rank} for data matrix shape {tuple(left.shape)}"
        raise ValueError(msg)
    return rank


def fit_row_operator(left: Tensor, right: Tensor, rank: RankSpec) -> Tensor:
    """Fit ``right ~= left @ A`` and return ``K`` for ``x_next = x @ K.T``.

    Parameters
    ----------
    left : Tensor
        Source states or observables with shape ``(num_samples, state_dim)``.
    right : Tensor
        Target states or observables with shape ``(num_samples, state_dim)``.
    rank : int or None or {"auto"}
        Optional truncated-SVD rank. ``None`` uses full least squares.
        ``"auto"`` selects the rank by the Gavish–Donoho unknown-``σ``
        median hard threshold (see :func:`optimal_hard_threshold_rank`).

    Returns
    -------
    Tensor
        Row-convention Koopman matrix ``K`` with shape
        ``(state_dim, state_dim)``.

    Raises
    ------
    ValueError
        If ``rank`` is outside the valid range for the data matrix, or
        ``"auto"`` yields rank 0.

    References
    ----------
    Gavish, M. & Donoho, D. L. (2014). The optimal hard threshold for
    singular values is ``4/√3``. *IEEE Transactions on Information Theory*,
    60(8), 5040–5053. https://doi.org/10.1109/TIT.2014.2323359
    """
    resolved = resolve_fit_rank(left, rank)
    if resolved is None:
        solution = torch.linalg.lstsq(left, right).solution
        return solution.T

    u, singular_values, vh = torch.linalg.svd(left, full_matrices=False)
    u_r = u[:, :resolved]
    s_r = singular_values[:resolved]
    vh_r = vh[:resolved, :]
    solution = vh_r.T @ ((u_r.T @ right) / s_r.unsqueeze(1))
    return solution.T


def fit_controlled_row_operator(
    left: Tensor,
    right: Tensor,
    controls: Tensor,
    rank: RankSpec,
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
    rank : int or None or {"auto"}
        Optional truncated-SVD rank for the augmented regression. Same
        semantics as :func:`fit_row_operator`.

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
    rank : int or None or {"auto"}
        Truncated-SVD rank request. ``None`` uses full least squares;
        ``"auto"`` selects the Gavish–Donoho hard threshold at fit time.
    selected_rank : int or None
        Rank actually used by the last successful :meth:`fit`. ``None``
        before fit, or when ``rank is None`` (full least squares). After
        ``rank="auto"`` or an integer ``rank``, this is the positive
        truncation used.
    K : Tensor or None
        Fitted Koopman matrix, or ``None`` before :meth:`fit`.
    num_nodes : int or None
        Node count recorded at fit time.
    in_channels : int or None
        Feature dimension recorded at fit time.
    state_dim : int or None
        Flattened state dimension recorded at fit time.
    """

    def __init__(self, *, time_step: float = 1.0, rank: RankSpec = None) -> None:
        """Initialize shared baseline hyperparameters.

        Parameters
        ----------
        time_step : float, optional
            Physical duration represented by one snapshot transition. Default
            is ``1.0``.
        rank : int or None or {"auto"}, optional
            Optional truncated-SVD rank. ``None`` uses full least squares.
            ``"auto"`` applies the Gavish–Donoho unknown-``σ`` median hard
            threshold at fit time.

        Raises
        ------
        ValueError
            If ``time_step`` is not positive or ``rank`` is invalid.
        """
        if time_step <= 0:
            msg = f"time_step must be positive, got {time_step}"
            raise ValueError(msg)
        if rank is not None and rank != "auto":
            if not isinstance(rank, int):
                msg = f"rank must be int, None, or 'auto', got {rank!r}"
                raise ValueError(msg)
            if rank < 1:
                msg = f"rank must be >= 1 when provided, got {rank}"
                raise ValueError(msg)
        self.time_step = float(time_step)
        self.rank: RankSpec = rank
        self.selected_rank: int | None = None
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
