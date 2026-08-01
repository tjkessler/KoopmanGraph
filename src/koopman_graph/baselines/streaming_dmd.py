"""Streaming / online least-squares DMD baseline.

Accumulates Gram matrices over consecutive flattened states so the fitted
operator matches batch least squares on the same pairs. Supports
:meth:`update` for one-snapshot increments. Topology-blind Data-only
``predict`` matches :class:`~koopman_graph.baselines.DMDBaseline`.
Primary-source citations are deferred to Sphinx Phase 61 verification.
"""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor
from torch_geometric.data import Data

from koopman_graph.baselines.base import (
    ClassicalBaseline,
    check_initial_graph,
    copy_topology,
    flatten_snapshots,
    require_static_topology,
    streaming_gram_init,
    streaming_gram_solve,
    streaming_gram_update,
)
from koopman_graph.data import (
    GraphSnapshotSequence,
    resolve_sequence,
)
from koopman_graph.spectrum_types import KoopmanSpectrum, compute_spectrum


class StreamingDMDBaseline(ClassicalBaseline):
    """Online Gram least-squares DMD on flattened node states.

    ``fit`` processes an entire sequence; :meth:`update` ingests one new
    snapshot after the fitted buffer. The learned operator follows
    ``x_next = x @ K.T``.

    Satisfies :class:`~koopman_graph.protocols.ForecastModel` and
    :class:`~koopman_graph.protocols.UncontrolledForecastModel`.

    Parameters
    ----------
    time_step : float, optional
        Physical duration represented by one snapshot transition. Used by
        :meth:`spectrum`. Default is ``1.0``.
    """

    def __init__(self, *, time_step: float = 1.0) -> None:
        """Initialize the streaming DMD baseline (rank is unused).

        Parameters
        ----------
        time_step
            Value for ``time_step``.
        """
        super().__init__(time_step=time_step, rank=None)
        self._gram: Tensor | None = None
        self._cross: Tensor | None = None
        self._last_state: Tensor | None = None

    def _is_fitted(self) -> bool:
        """Return whether at least one transition has been accumulated.

        Returns
        -------
        object
            Function result.
        """
        return self.K is not None

    def fit(
        self,
        sequence: GraphSnapshotSequence | Sequence[Data],
    ) -> StreamingDMDBaseline:
        """Fit by accumulating all consecutive snapshot pairs.

        Parameters
        ----------
        sequence : GraphSnapshotSequence or sequence of Data
            Training snapshots with shared topology.

        Returns
        -------
        StreamingDMDBaseline
            The fitted baseline (``self``) for sklearn-style chaining.

        Raises
        ------
        ValueError
            If fewer than two snapshots are provided or topology varies.
        """
        resolved = resolve_sequence(sequence)
        require_static_topology(resolved)
        if resolved.num_timesteps < 2:
            msg = "StreamingDMDBaseline.fit requires at least two snapshots"
            raise ValueError(msg)

        states = flatten_snapshots(resolved)
        state_dim = int(states.shape[1])
        self._gram, self._cross = streaming_gram_init(
            state_dim,
            dtype=states.dtype,
            device=states.device,
        )
        for index in range(states.shape[0] - 1):
            self._gram, self._cross = streaming_gram_update(
                self._gram,
                self._cross,
                states[index],
                states[index + 1],
            )
        self.K = streaming_gram_solve(self._gram, self._cross)
        self.selected_rank = None
        self.num_nodes = resolved.num_nodes
        self.in_channels = resolved.in_channels
        self.state_dim = state_dim
        self._last_state = states[-1].detach().clone()
        return self

    def update(self, snapshot: Data) -> StreamingDMDBaseline:
        """Ingest one new snapshot and refresh the operator.

        Parameters
        ----------
        snapshot : Data
            Next graph snapshot after the current buffer.

        Returns
        -------
        StreamingDMDBaseline
            ``self`` after the incremental update.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        ValueError
            If graph metadata does not match the fitted sequence.
        """
        if self._gram is None or self._cross is None or self._last_state is None:
            msg = "StreamingDMDBaseline.update requires a prior fit()"
            raise RuntimeError(msg)
        num_nodes, in_channels = self._require_graph_metadata()
        check_initial_graph(
            snapshot,
            num_nodes=num_nodes,
            in_channels=in_channels,
        )
        new_state = snapshot.x.reshape(-1).to(
            dtype=self._last_state.dtype,
            device=self._last_state.device,
        )
        self._gram, self._cross = streaming_gram_update(
            self._gram,
            self._cross,
            self._last_state,
            new_state,
        )
        self.K = streaming_gram_solve(self._gram, self._cross)
        self._last_state = new_state.detach().clone()
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
        """Return the streaming DMD operator spectrum.

        Returns
        -------
        object
            Function result.
        """
        return compute_spectrum(self._require_operator(), self.time_step)
