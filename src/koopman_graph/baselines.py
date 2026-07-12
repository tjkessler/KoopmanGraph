"""Classical topology-agnostic Koopman baselines."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from torch import Tensor
from torch_geometric.data import Data

from koopman_graph.analysis import KoopmanSpectrum, compute_spectrum
from koopman_graph.data import (
    GraphSnapshotSequence,
    _snapshot_edge_weight,
    resolve_sequence,
)

PolynomialDegree = Literal[1, 2]


def _flatten_snapshots(sequence: GraphSnapshotSequence) -> Tensor:
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


def _fit_row_operator(left: Tensor, right: Tensor, rank: int | None) -> Tensor:
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


def _fit_controlled_row_operator(
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
    joint = _fit_row_operator(augmented, right, rank)
    state_dim = left.shape[1]
    k_matrix = joint[:, :state_dim]
    b_matrix = joint[:, state_dim:].T
    return k_matrix, b_matrix


def _transition_controls(sequence: GraphSnapshotSequence) -> Tensor:
    """Return control inputs aligned with consecutive snapshot transitions.

    Parameters
    ----------
    sequence : GraphSnapshotSequence
        Sequence with controls.

    Returns
    -------
    Tensor
        Controls with shape ``(num_timesteps - 1, control_dim)`` for global
        controls or flattened per-node controls.
    """
    if not sequence.has_controls:
        msg = "sequence does not contain control inputs"
        raise ValueError(msg)
    controls = sequence.control_inputs
    assert controls is not None
    transition_controls = controls[:-1]
    if transition_controls.ndim == 3:
        return transition_controls.reshape(transition_controls.shape[0], -1)
    return transition_controls


def _copy_topology(initial_graph: Data) -> dict[str, Tensor]:
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
    edge_weight = _snapshot_edge_weight(initial_graph)
    if edge_weight is not None:
        fields["edge_weight"] = edge_weight
    return fields


def _check_initial_graph(
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
        msg = (
            f"initial graph has {initial_graph.num_nodes} nodes, "
            f"expected {num_nodes}"
        )
        raise ValueError(msg)
    if initial_graph.x.shape[1] != in_channels:
        msg = (
            f"initial graph has feature dimension {initial_graph.x.shape[1]}, "
            f"expected {in_channels}"
        )
        raise ValueError(msg)


class DMDBaseline:
    """Dynamic Mode Decomposition baseline on flattened node states.

    ``DMDBaseline`` ignores graph message passing: each graph snapshot is
    reshaped into one vector and a linear map is fit by least squares. The
    learned operator follows the package convention ``x_next = x @ K.T``.

    Parameters
    ----------
    time_step : float, optional
        Physical duration represented by one snapshot transition. Used by
        :meth:`spectrum`. Default is ``1.0``.
    rank : int or None, optional
        Optional truncated-SVD rank for the data matrix. ``None`` uses the full
        least-squares solution. Default is ``None``.
    """

    def __init__(self, *, time_step: float = 1.0, rank: int | None = None) -> None:
        """Initialize the DMD baseline.

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

    def fit(
        self,
        sequence: GraphSnapshotSequence | Sequence[Data],
    ) -> DMDBaseline:
        """Fit the DMD operator from consecutive graph snapshots.

        Parameters
        ----------
        sequence : GraphSnapshotSequence or sequence of Data
            Training snapshots with shared topology.

        Returns
        -------
        DMDBaseline
            The fitted baseline.

        Raises
        ------
        ValueError
            If fewer than two snapshots are provided or rank is invalid.
        """
        resolved = resolve_sequence(sequence)
        if resolved.num_timesteps < 2:
            msg = "DMDBaseline.fit requires at least two snapshots"
            raise ValueError(msg)

        states = _flatten_snapshots(resolved)
        self.K = _fit_row_operator(states[:-1], states[1:], self.rank)
        self.num_nodes = resolved.num_nodes
        self.in_channels = resolved.in_channels
        self.state_dim = states.shape[1]
        return self

    def predict(self, initial_graph: Data, steps: int) -> list[Data]:
        """Autoregressively predict future graph snapshots.

        Parameters
        ----------
        initial_graph : Data
            Initial graph snapshot. Its topology is copied to every prediction.
        steps : int
            Number of future snapshots to predict.

        Returns
        -------
        list of Data
            Predicted graph snapshots with the same node/feature shape as the
            fitted training data.
        """
        self._check_fitted()
        if steps < 1:
            msg = f"steps must be >= 1, got {steps}"
            raise ValueError(msg)
        _check_initial_graph(
            initial_graph,
            num_nodes=self.num_nodes,
            in_channels=self.in_channels,
        )

        assert self.K is not None
        assert self.num_nodes is not None
        assert self.in_channels is not None

        state = initial_graph.x.reshape(-1)
        topology = _copy_topology(initial_graph)
        predictions: list[Data] = []
        for _ in range(steps):
            state = state @ self.K.T
            x = state.reshape(self.num_nodes, self.in_channels)
            predictions.append(Data(x=x, **topology))
        return predictions

    def spectrum(self) -> KoopmanSpectrum:
        """Return the DMD operator spectrum.

        Returns
        -------
        KoopmanSpectrum
            Eigendecomposition and continuous-time mode characteristics of the
            fitted DMD operator.
        """
        self._check_fitted()
        assert self.K is not None
        return compute_spectrum(self.K, self.time_step)

    def _check_fitted(self) -> None:
        """Raise if the baseline has not been fit.

        Raises
        ------
        RuntimeError
            If no DMD operator has been fit.
        """
        if self.K is None:
            msg = "DMDBaseline must be fit before prediction or spectral analysis"
            raise RuntimeError(msg)


class DMDcBaseline:
    """Dynamic Mode Decomposition with control on flattened node states.

    ``DMDcBaseline`` extends :class:`DMDBaseline` with exogenous inputs,
    fitting ``x_{t+1} = x_t @ K.T + u_t @ B`` by least squares on flattened
    graph snapshots. Global controls use shape ``(control_dim,)``; per-node
    controls are flattened during fitting and prediction.

    Parameters
    ----------
    time_step : float, optional
        Physical duration represented by one snapshot transition. Used by
        :meth:`spectrum`. Default is ``1.0``.
    rank : int or None, optional
        Optional truncated-SVD rank for the augmented regression. ``None`` uses
        the full least-squares solution. Default is ``None``.
    """

    def __init__(self, *, time_step: float = 1.0, rank: int | None = None) -> None:
        """Initialize the DMDc baseline.

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
        self.B: Tensor | None = None
        self.num_nodes: int | None = None
        self.in_channels: int | None = None
        self.state_dim: int | None = None
        self.control_dim: int | None = None
        self.per_node_controls: bool = False
        self.num_nodes_control: int | None = None

    def fit(
        self,
        sequence: GraphSnapshotSequence | Sequence[Data],
    ) -> DMDcBaseline:
        """Fit controlled DMD operators from consecutive graph snapshots.

        Parameters
        ----------
        sequence : GraphSnapshotSequence or sequence of Data
            Training snapshots with shared topology and control inputs.

        Returns
        -------
        DMDcBaseline
            The fitted baseline.

        Raises
        ------
        ValueError
            If fewer than two snapshots are provided, controls are missing, or
            rank is invalid.
        """
        resolved = resolve_sequence(sequence)
        if resolved.num_timesteps < 2:
            msg = "DMDcBaseline.fit requires at least two snapshots"
            raise ValueError(msg)
        if not resolved.has_controls:
            msg = "DMDcBaseline.fit requires sequences with control inputs"
            raise ValueError(msg)

        states = _flatten_snapshots(resolved)
        controls = _transition_controls(resolved)
        self.per_node_controls = resolved.control_inputs is not None and (
            resolved.control_inputs.ndim == 3
        )
        if self.per_node_controls:
            assert resolved.control_inputs is not None
            self.num_nodes_control = int(resolved.control_inputs.shape[1])
            self.control_dim = int(resolved.control_inputs.shape[2])
        else:
            self.control_dim = int(controls.shape[1])

        self.K, self.B = _fit_controlled_row_operator(
            states[:-1],
            states[1:],
            controls,
            self.rank,
        )
        self.num_nodes = resolved.num_nodes
        self.in_channels = resolved.in_channels
        self.state_dim = states.shape[1]
        return self

    def predict(
        self,
        initial_graph: Data,
        steps: int,
        controls: Sequence[Tensor],
    ) -> list[Data]:
        """Autoregressively predict future graph snapshots with future controls.

        Parameters
        ----------
        initial_graph : Data
            Initial graph snapshot. Its topology is copied to every prediction.
        steps : int
            Number of future snapshots to predict.
        controls : sequence of Tensor
            Future control inputs, one per rollout step. Global controls use
            shape ``(control_dim,)``; per-node controls use
            ``(num_nodes, control_dim)``.

        Returns
        -------
        list of Data
            Predicted graph snapshots with the same node/feature shape as the
            fitted training data.
        """
        self._check_fitted()
        if steps < 1:
            msg = f"steps must be >= 1, got {steps}"
            raise ValueError(msg)
        if len(controls) != steps:
            msg = f"expected {steps} control inputs, got {len(controls)}"
            raise ValueError(msg)
        _check_initial_graph(
            initial_graph,
            num_nodes=self.num_nodes,
            in_channels=self.in_channels,
        )

        assert self.K is not None
        assert self.B is not None
        assert self.num_nodes is not None
        assert self.in_channels is not None
        assert self.control_dim is not None

        state = initial_graph.x.reshape(-1)
        topology = _copy_topology(initial_graph)
        predictions: list[Data] = []
        for control in controls:
            control_vector = self._control_vector(control)
            state = state @ self.K.T + control_vector @ self.B
            x = state.reshape(self.num_nodes, self.in_channels)
            predictions.append(Data(x=x, **topology))
        return predictions

    def spectrum(self) -> KoopmanSpectrum:
        """Return the autonomous DMD operator spectrum.

        Returns
        -------
        KoopmanSpectrum
            Eigendecomposition of the fitted state-transition operator ``K``.
        """
        self._check_fitted()
        assert self.K is not None
        return compute_spectrum(self.K, self.time_step)

    def _control_vector(self, control: Tensor) -> Tensor:
        """Flatten a global or per-node control input for prediction.

        Parameters
        ----------
        control : Tensor
            Control input for one rollout step.

        Returns
        -------
        Tensor
            Row vector with shape ``(control_feature_dim,)``.
        """
        assert self.control_dim is not None
        if self.per_node_controls:
            if control.ndim != 2:
                msg = (
                    "per-node controls must have shape "
                    f"(num_nodes, {self.control_dim}), got {tuple(control.shape)}"
                )
                raise ValueError(msg)
            return control.reshape(-1)
        if control.ndim != 1 or control.shape[0] != self.control_dim:
            msg = (
                f"global controls must have shape ({self.control_dim},), "
                f"got {tuple(control.shape)}"
            )
            raise ValueError(msg)
        return control

    def _check_fitted(self) -> None:
        """Raise if the baseline has not been fit.

        Raises
        ------
        RuntimeError
            If no DMDc operators have been fit.
        """
        if self.K is None or self.B is None:
            msg = "DMDcBaseline must be fit before prediction or spectral analysis"
            raise RuntimeError(msg)


class EDMDBaseline:
    """Extended DMD baseline with polynomial observables.

    EDMD lifts flattened graph states into a fixed observable space, fits a
    linear Koopman operator there, and learns a least-squares decoder back to
    physical node features. ``polynomial_degree=1`` is an identity observable;
    ``polynomial_degree=2`` appends elementwise squared terms.

    Parameters
    ----------
    time_step : float, optional
        Physical duration represented by one snapshot transition. Used by
        :meth:`spectrum`. Default is ``1.0``.
    rank : int or None, optional
        Optional truncated-SVD rank for the observable data matrix. ``None``
        uses the full least-squares solution. Default is ``None``.
    polynomial_degree : {1, 2}, optional
        Polynomial observable degree. Default is ``2``.
    """

    def __init__(
        self,
        *,
        time_step: float = 1.0,
        rank: int | None = None,
        polynomial_degree: PolynomialDegree = 2,
    ) -> None:
        """Initialize the EDMD baseline.

        Parameters
        ----------
        time_step : float, optional
            Physical duration represented by one snapshot transition. Default
            is ``1.0``.
        rank : int or None, optional
            Optional truncated-SVD rank in observable space. ``None`` uses full
            least squares.
        polynomial_degree : {1, 2}, optional
            Polynomial observable degree. Default is ``2``.

        Raises
        ------
        ValueError
            If ``time_step`` is not positive or ``polynomial_degree`` is not
            supported.
        """
        if time_step <= 0:
            msg = f"time_step must be positive, got {time_step}"
            raise ValueError(msg)
        if polynomial_degree not in (1, 2):
            msg = f"polynomial_degree must be 1 or 2, got {polynomial_degree}"
            raise ValueError(msg)
        self.time_step = float(time_step)
        self.rank = rank
        self.polynomial_degree = polynomial_degree
        self.K: Tensor | None = None
        self.decoder: Tensor | None = None
        self.num_nodes: int | None = None
        self.in_channels: int | None = None
        self.state_dim: int | None = None
        self.observable_dim: int | None = None

    def fit(
        self,
        sequence: GraphSnapshotSequence | Sequence[Data],
    ) -> EDMDBaseline:
        """Fit EDMD operator and linear reconstruction decoder.

        Parameters
        ----------
        sequence : GraphSnapshotSequence or sequence of Data
            Training snapshots with shared topology.

        Returns
        -------
        EDMDBaseline
            The fitted baseline.

        Raises
        ------
        ValueError
            If fewer than two snapshots are provided or rank is invalid.
        """
        resolved = resolve_sequence(sequence)
        if resolved.num_timesteps < 2:
            msg = "EDMDBaseline.fit requires at least two snapshots"
            raise ValueError(msg)

        states = _flatten_snapshots(resolved)
        observables = self._observables(states)
        self.K = _fit_row_operator(observables[:-1], observables[1:], self.rank)
        self.decoder = torch.linalg.lstsq(observables, states).solution.T
        self.num_nodes = resolved.num_nodes
        self.in_channels = resolved.in_channels
        self.state_dim = states.shape[1]
        self.observable_dim = observables.shape[1]
        return self

    def predict(self, initial_graph: Data, steps: int) -> list[Data]:
        """Autoregressively predict future graph snapshots.

        Parameters
        ----------
        initial_graph : Data
            Initial graph snapshot. Its topology is copied to every prediction.
        steps : int
            Number of future snapshots to predict.

        Returns
        -------
        list of Data
            Predicted graph snapshots with the same node/feature shape as the
            fitted training data.

        Raises
        ------
        RuntimeError
            If the baseline has not been fit.
        ValueError
            If ``steps < 1`` or graph metadata does not match the fit data.
        """
        self._check_fitted()
        if steps < 1:
            msg = f"steps must be >= 1, got {steps}"
            raise ValueError(msg)
        _check_initial_graph(
            initial_graph,
            num_nodes=self.num_nodes,
            in_channels=self.in_channels,
        )

        assert self.K is not None
        assert self.decoder is not None
        assert self.num_nodes is not None
        assert self.in_channels is not None

        observable = self._observables(initial_graph.x.reshape(1, -1)).squeeze(0)
        topology = _copy_topology(initial_graph)
        predictions: list[Data] = []
        for _ in range(steps):
            observable = observable @ self.K.T
            state = observable @ self.decoder.T
            x = state.reshape(self.num_nodes, self.in_channels)
            predictions.append(Data(x=x, **topology))
        return predictions

    def spectrum(self) -> KoopmanSpectrum:
        """Return the EDMD observable-space operator spectrum.

        Returns
        -------
        KoopmanSpectrum
            Eigendecomposition and continuous-time mode characteristics of the
            fitted observable-space operator.
        """
        self._check_fitted()
        assert self.K is not None
        return compute_spectrum(self.K, self.time_step)

    def _observables(self, states: Tensor) -> Tensor:
        """Lift flattened states into fixed polynomial observables.

        Parameters
        ----------
        states : Tensor
            Flattened physical states with shape ``(..., state_dim)``.

        Returns
        -------
        Tensor
            Observable matrix. For degree 2, identity features are concatenated
            with elementwise squared features.
        """
        if self.polynomial_degree == 1:
            return states
        return torch.cat([states, states.square()], dim=-1)

    def _check_fitted(self) -> None:
        """Raise if the baseline has not been fit.

        Raises
        ------
        RuntimeError
            If no EDMD operator or decoder has been fit.
        """
        if self.K is None or self.decoder is None:
            msg = "EDMDBaseline must be fit before prediction or spectral analysis"
            raise RuntimeError(msg)
