"""Extended DMD baseline with polynomial observables."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from torch import Tensor
from torch_geometric.data import Data

from koopman_graph.analysis import compute_spectrum
from koopman_graph.baselines.base import (
    ClassicalBaseline,
    _check_initial_graph,
    _copy_topology,
    _fit_row_operator,
    _flatten_snapshots,
    _require_static_topology,
)
from koopman_graph.data import (
    GraphSnapshotSequence,
    resolve_sequence,
)
from koopman_graph.spectrum_types import KoopmanSpectrum

PolynomialDegree = Literal[1, 2]


class EDMDBaseline(ClassicalBaseline):
    """Extended DMD baseline with polynomial observables.

    EDMD lifts flattened graph states into a fixed observable space, fits a
    linear Koopman operator there, and learns a least-squares
    ``reconstruction_matrix`` back to physical node features (not a GNN
    decoder). ``polynomial_degree=1`` is an identity observable;
    ``polynomial_degree=2`` appends elementwise squared terms.

    Satisfies :class:`~koopman_graph.protocols.ForecastModel` and the narrower
    :class:`~koopman_graph.protocols.UncontrolledForecastModel` peer set.
    ``predict`` accepts a PyG ``Data`` snapshot only (not raw feature tensors)
    and does not take controls or future topologies.

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
        super().__init__(time_step=time_step, rank=rank)
        if polynomial_degree not in (1, 2):
            msg = f"polynomial_degree must be 1 or 2, got {polynomial_degree}"
            raise ValueError(msg)
        self.polynomial_degree = polynomial_degree
        self.reconstruction_matrix: Tensor | None = None
        self.observable_dim: int | None = None

    def _is_fitted(self) -> bool:
        """Return whether the EDMD operator and reconstruction matrix are fit.

        Returns
        -------
        bool
            ``True`` when ``K`` and ``reconstruction_matrix`` are available.
        """
        return self.K is not None and self.reconstruction_matrix is not None

    def _require_reconstruction_matrix(self) -> Tensor:
        """Return the fitted reconstruction matrix after a fitted-state check.

        Returns
        -------
        Tensor
            Least-squares map from observables to flattened physical states.

        Raises
        ------
        RuntimeError
            If the baseline has not been fit.
        """
        self._check_fitted()
        if self.reconstruction_matrix is None:
            raise RuntimeError(self._unfitted_message())
        return self.reconstruction_matrix

    def fit(
        self,
        sequence: GraphSnapshotSequence | Sequence[Data],
    ) -> EDMDBaseline:
        """Fit EDMD operator and linear reconstruction matrix.

        Parameters
        ----------
        sequence : GraphSnapshotSequence or sequence of Data
            Training snapshots with shared topology.

        Returns
        -------
        EDMDBaseline
            The fitted baseline (``self``) for sklearn-style chaining. Unlike
            :meth:`~koopman_graph.model.GraphKoopmanModel.fit`, which returns
            :class:`~koopman_graph.training.FitHistory`, classical baselines
            return ``self``; see the ``ForecastModel`` call-site matrix in
            :doc:`architecture`.

        Raises
        ------
        ValueError
            If fewer than two snapshots are provided, the sequence has dynamic
            topology, or rank is invalid.
        """
        resolved = resolve_sequence(sequence)
        _require_static_topology(resolved)
        if resolved.num_timesteps < 2:
            msg = "EDMDBaseline.fit requires at least two snapshots"
            raise ValueError(msg)

        states = _flatten_snapshots(resolved)
        observables = self._observables(states)
        self.K = _fit_row_operator(observables[:-1], observables[1:], self.rank)
        self.reconstruction_matrix = torch.linalg.lstsq(observables, states).solution.T
        self.num_nodes = resolved.num_nodes
        self.in_channels = resolved.in_channels
        self.state_dim = states.shape[1]
        self.observable_dim = observables.shape[1]
        return self

    def predict(self, initial_graph: Data, steps: int) -> list[Data]:
        """Autoregressively predict future graph snapshots (Data-only).

        Uncontrolled peer call site shared with :class:`DMDBaseline` and
        :class:`~koopman_graph.model.GraphKoopmanModel` when the latter is
        invoked as ``predict(data, steps)``. Does not accept raw feature
        tensors or a ``controls`` argument.

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
        operator = self._require_operator()
        reconstruction = self._require_reconstruction_matrix()
        num_nodes, in_channels = self._require_graph_metadata()
        if steps < 1:
            msg = f"steps must be >= 1, got {steps}"
            raise ValueError(msg)
        _check_initial_graph(
            initial_graph,
            num_nodes=num_nodes,
            in_channels=in_channels,
        )

        observable = self._observables(initial_graph.x.reshape(1, -1)).squeeze(0)
        topology = _copy_topology(initial_graph)
        predictions: list[Data] = []
        for _ in range(steps):
            observable = observable @ operator.T
            state = observable @ reconstruction.T
            x = state.reshape(num_nodes, in_channels)
            predictions.append(Data(x=x, **topology))
        return predictions

    def spectrum(self) -> KoopmanSpectrum:
        """Return the EDMD observable-space operator spectrum.

        Takes no kwargs (unlike continuous
        :meth:`~koopman_graph.model.GraphKoopmanModel.spectrum`).

        Returns
        -------
        KoopmanSpectrum
            Eigendecomposition and continuous-time mode characteristics of the
            fitted observable-space operator.
        """
        return compute_spectrum(self._require_operator(), self.time_step)

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
