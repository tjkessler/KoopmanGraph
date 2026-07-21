"""Dynamic Mode Decomposition baseline on flattened node states."""

from __future__ import annotations

from collections.abc import Sequence

from torch_geometric.data import Data

from koopman_graph.baselines.base import (
    ClassicalBaseline,
    check_initial_graph,
    copy_topology,
    fit_row_operator,
    flatten_snapshots,
    require_static_topology,
)
from koopman_graph.data import (
    GraphSnapshotSequence,
    resolve_sequence,
)
from koopman_graph.spectrum_types import KoopmanSpectrum, compute_spectrum


class DMDBaseline(ClassicalBaseline):
    """Dynamic Mode Decomposition baseline on flattened node states.

    ``DMDBaseline`` ignores graph message passing: each graph snapshot is
    reshaped into one vector and a linear map is fit by least squares. The
    learned operator follows the package convention ``x_next = x @ K.T``.

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
        Optional truncated-SVD rank for the data matrix. ``None`` uses the full
        least-squares solution. Default is ``None``.
    """

    def _is_fitted(self) -> bool:
        """Return whether the DMD operator has been fit.

        Returns
        -------
        bool
            ``True`` when ``K`` is available.
        """
        return self.K is not None

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
        require_static_topology(resolved)
        if resolved.num_timesteps < 2:
            msg = "DMDBaseline.fit requires at least two snapshots"
            raise ValueError(msg)

        states = flatten_snapshots(resolved)
        self.K = fit_row_operator(states[:-1], states[1:], self.rank)
        self.num_nodes = resolved.num_nodes
        self.in_channels = resolved.in_channels
        self.state_dim = states.shape[1]
        return self

    def predict(self, initial_graph: Data, steps: int) -> list[Data]:
        """Autoregressively predict future graph snapshots (Data-only).

        Uncontrolled peer call site shared with :class:`EDMDBaseline` and
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
        """Return the DMD operator spectrum.

        Takes no kwargs (unlike continuous
        :meth:`~koopman_graph.model.GraphKoopmanModel.spectrum`).

        Returns
        -------
        KoopmanSpectrum
            Eigendecomposition and continuous-time mode characteristics of the
            fitted DMD operator.
        """
        return compute_spectrum(self._require_operator(), self.time_step)
