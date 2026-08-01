"""Optimized Dynamic Mode Decomposition baseline (MVP variable projection).

Refines an exact-DMD initialization with a compact eigenvalue / amplitude
update on tiny sequences. Topology-blind Data-only ``predict`` matches
:class:`~koopman_graph.baselines.DMDBaseline`. Primary-source citations are
deferred to Sphinx Phase 61 verification.
"""

from __future__ import annotations

from collections.abc import Sequence

from torch_geometric.data import Data

from koopman_graph.baselines.base import (
    ClassicalBaseline,
    RankSpec,
    check_initial_graph,
    copy_topology,
    fit_opt_row_operator,
    flatten_snapshots,
    require_static_topology,
    resolve_fit_rank,
)
from koopman_graph.data import (
    GraphSnapshotSequence,
    resolve_sequence,
)
from koopman_graph.spectrum_types import KoopmanSpectrum, compute_spectrum


class OptDMDBaseline(ClassicalBaseline):
    """Optimized DMD baseline on flattened node states.

    Ignores graph message passing. Fits a linear map via light variable
    projection initialized from exact DMD. The learned operator follows
    ``x_next = x @ K.T``.

    Satisfies :class:`~koopman_graph.protocols.ForecastModel` and
    :class:`~koopman_graph.protocols.UncontrolledForecastModel`.

    Parameters
    ----------
    time_step : float, optional
        Physical duration represented by one snapshot transition. Used by
        :meth:`spectrum`. Default is ``1.0``.
    rank : int or None or {"auto"}, optional
        Truncated-SVD rank for the exact-DMD initialization. Default is
        ``None``.
    max_iter : int, optional
        Variable-projection refinement iterations. Default is ``20``.
    """

    def __init__(
        self,
        *,
        time_step: float = 1.0,
        rank: RankSpec = None,
        max_iter: int = 20,
    ) -> None:
        """Initialize the optimized DMD baseline.

        Parameters
        ----------
        time_step
            Value for ``time_step``.
        rank
            Value for ``rank``.
        max_iter
            Value for ``max_iter``.
        """
        super().__init__(time_step=time_step, rank=rank)
        if max_iter < 1:
            msg = f"max_iter must be >= 1, got {max_iter}"
            raise ValueError(msg)
        self.max_iter = int(max_iter)

    def _is_fitted(self) -> bool:
        """Return whether the optimized DMD operator has been fit.

        Returns
        -------
        object
            Function result.
        """
        return self.K is not None

    def fit(
        self,
        sequence: GraphSnapshotSequence | Sequence[Data],
    ) -> OptDMDBaseline:
        """Fit the optimized DMD operator from consecutive snapshots.

        Parameters
        ----------
        sequence : GraphSnapshotSequence or sequence of Data
            Training snapshots with shared topology.

        Returns
        -------
        OptDMDBaseline
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
            msg = "OptDMDBaseline.fit requires at least two snapshots"
            raise ValueError(msg)

        states = flatten_snapshots(resolved)
        left = states[:-1]
        self.selected_rank = resolve_fit_rank(left, self.rank)
        self.K = fit_opt_row_operator(
            left,
            states[1:],
            self.selected_rank,
            max_iter=self.max_iter,
        )
        self.num_nodes = resolved.num_nodes
        self.in_channels = resolved.in_channels
        self.state_dim = states.shape[1]
        return self

    def predict(self, initial_graph: Data, steps: int) -> list[Data]:
        """Autoregressively predict future graph snapshots (Data-only).

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
        """Return the optimized DMD operator spectrum.

        Returns
        -------
        object
            Function result.
        """
        return compute_spectrum(self._require_operator(), self.time_step)
