"""Total-least-squares Dynamic Mode Decomposition baseline.

Fits a TLS linear map on jointly SVD-truncated stacked snapshot matrices
(topology-blind). Same Data-only ``predict`` surface as
:class:`~koopman_graph.baselines.DMDBaseline`. Primary-source citations are
deferred to Sphinx Phase 61 verification.
"""

from __future__ import annotations

from collections.abc import Sequence

from torch_geometric.data import Data

from koopman_graph.baselines.base import (
    ClassicalBaseline,
    check_initial_graph,
    copy_topology,
    fit_tls_row_operator,
    flatten_snapshots,
    require_static_topology,
    resolve_fit_rank,
)
from koopman_graph.data import (
    GraphSnapshotSequence,
    resolve_sequence,
)
from koopman_graph.spectrum_types import KoopmanSpectrum, compute_spectrum


class TLSDMDBaseline(ClassicalBaseline):
    """Total-least-squares DMD baseline on flattened node states.

    Ignores graph message passing: each snapshot is reshaped into one vector
    and a TLS linear map is fit. The learned operator follows
    ``x_next = x @ K.T``.

    Satisfies :class:`~koopman_graph.protocols.ForecastModel` and
    :class:`~koopman_graph.protocols.UncontrolledForecastModel`.

    Parameters
    ----------
    time_step : float, optional
        Physical duration represented by one snapshot transition. Used by
        :meth:`spectrum`. Default is ``1.0``.
    rank : int or None or {"auto"}, optional
        Truncated-SVD rank for the joint data matrix. ``None`` / ``"auto"``
        follow the same semantics as :class:`~koopman_graph.baselines.DMDBaseline`.
        Default is ``None``.
    """

    def _is_fitted(self) -> bool:
        """Return whether the TLS DMD operator has been fit.

        Returns
        -------
        object
            Function result.
        """
        return self.K is not None

    def fit(
        self,
        sequence: GraphSnapshotSequence | Sequence[Data],
    ) -> TLSDMDBaseline:
        """Fit the TLS DMD operator from consecutive graph snapshots.

        Parameters
        ----------
        sequence : GraphSnapshotSequence or sequence of Data
            Training snapshots with shared topology.

        Returns
        -------
        TLSDMDBaseline
            The fitted baseline (``self``) for sklearn-style chaining.

        Raises
        ------
        ValueError
            If fewer than two snapshots are provided, the sequence has
            dynamic topology, or rank is invalid.
        """
        resolved = resolve_sequence(sequence)
        require_static_topology(resolved)
        if resolved.num_timesteps < 2:
            msg = "TLSDMDBaseline.fit requires at least two snapshots"
            raise ValueError(msg)

        states = flatten_snapshots(resolved)
        left = states[:-1]
        self.selected_rank = resolve_fit_rank(left, self.rank)
        self.K = fit_tls_row_operator(left, states[1:], self.selected_rank)
        self.num_nodes = resolved.num_nodes
        self.in_channels = resolved.in_channels
        self.state_dim = states.shape[1]
        return self

    def predict(self, initial_graph: Data, steps: int) -> list[Data]:
        """Autoregressively predict future graph snapshots (Data-only).

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
        """Return the TLS DMD operator spectrum.

        Returns
        -------
        KoopmanSpectrum
            Eigendecomposition and continuous-time mode characteristics of the
            fitted operator.
        """
        return compute_spectrum(self._require_operator(), self.time_step)
