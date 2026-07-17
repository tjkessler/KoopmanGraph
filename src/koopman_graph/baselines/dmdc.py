"""Dynamic Mode Decomposition with control on flattened node states."""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor
from torch_geometric.data import Data

from koopman_graph.analysis import compute_spectrum
from koopman_graph.baselines.base import (
    ClassicalBaseline,
    _check_initial_graph,
    _copy_topology,
    _fit_controlled_row_operator,
    _flatten_snapshots,
    _require_global_controls,
    _require_static_topology,
    _transition_controls,
)
from koopman_graph.data import (
    GraphSnapshotSequence,
    resolve_sequence,
)
from koopman_graph.spectrum_types import KoopmanSpectrum


class DMDcBaseline(ClassicalBaseline):
    """Dynamic Mode Decomposition with control on flattened node states.

    ``DMDcBaseline`` uses the same flattened-state DMD idea as
    :class:`~koopman_graph.baselines.DMDBaseline`, adding exogenous inputs and
    fitting ``x_{t+1} = x_t @ K.T + u_t @ B`` by least squares. **Global
    controls only:** sequence ``control_inputs`` must have shape
    ``(T, control_dim)`` and each ``predict`` control must have shape
    ``(control_dim,)``. Per-node ``(T, N, control_dim)`` layouts are
    **rejected** at ``fit`` (they would silently encode different physics than
    neural / adaptation per-node row matching). It shares
    :class:`~koopman_graph.baselines.ClassicalBaseline` scaffolding rather than
    subclassing :class:`~koopman_graph.baselines.DMDBaseline` (``predict``
    requires future controls).

    Satisfies loose :class:`~koopman_graph.protocols.ForecastModel` only.
    **Not** an :class:`~koopman_graph.protocols.UncontrolledForecastModel` peer:
    ``predict`` requires a ``controls`` sequence (one input per rollout step).
    Initial graph input is PyG ``Data`` only (not raw feature tensors).

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
        super().__init__(time_step=time_step, rank=rank)
        self.B: Tensor | None = None
        self.control_dim: int | None = None

    def _is_fitted(self) -> bool:
        """Return whether DMDc operators have been fit.

        Returns
        -------
        bool
            ``True`` when ``K`` and ``B`` are available.
        """
        return self.K is not None and self.B is not None

    def _require_control_matrix(self) -> Tensor:
        """Return the fitted control matrix after a fitted-state check.

        Returns
        -------
        Tensor
            Fitted control matrix ``B``.

        Raises
        ------
        RuntimeError
            If the baseline has not been fit.
        """
        self._check_fitted()
        if self.B is None:
            raise RuntimeError(self._unfitted_message())
        return self.B

    def _require_control_dim(self) -> int:
        """Return the fitted control dimension after a fitted-state check.

        Returns
        -------
        int
            Fitted control feature dimension.

        Raises
        ------
        RuntimeError
            If the baseline has not been fit.
        """
        self._check_fitted()
        if self.control_dim is None:
            raise RuntimeError(self._unfitted_message())
        return self.control_dim

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
            The fitted baseline (``self``) for sklearn-style chaining. Unlike
            :meth:`~koopman_graph.model.GraphKoopmanModel.fit`, which returns
            :class:`~koopman_graph.training.FitHistory`, classical baselines
            return ``self``; see the ``ForecastModel`` call-site matrix in
            :doc:`architecture`.

        Raises
        ------
        ValueError
            If fewer than two snapshots are provided, controls are missing,
            controls are per-node (3-D), the sequence has dynamic topology, or
            rank is invalid.
        """
        resolved = resolve_sequence(sequence)
        _require_static_topology(resolved)
        if resolved.num_timesteps < 2:
            msg = "DMDcBaseline.fit requires at least two snapshots"
            raise ValueError(msg)
        if not resolved.has_controls or resolved.control_inputs is None:
            msg = "DMDcBaseline.fit requires sequences with control inputs"
            raise ValueError(msg)
        _require_global_controls(resolved)

        states = _flatten_snapshots(resolved)
        controls = _transition_controls(resolved)
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
        """Autoregressively predict with required future controls (Data-only).

        Unlike :class:`~koopman_graph.baselines.DMDBaseline` /
        :class:`~koopman_graph.baselines.EDMDBaseline`, ``controls`` is
        required (no default). Do not treat DMDc as drop-in interchangeable
        with uncontrolled :class:`~koopman_graph.protocols.ForecastModel`
        peers; see :func:`~koopman_graph.protocols.accepts_uncontrolled_data_predict`.

        Parameters
        ----------
        initial_graph : Data
            Initial graph snapshot. Its topology is copied to every prediction.
            Raw feature tensors are not accepted.
        steps : int
            Number of future snapshots to predict.
        controls : sequence of Tensor
            Future control inputs, one per rollout step (required). Each entry
            must have global shape ``(control_dim,)``.

        Returns
        -------
        list of Data
            Predicted graph snapshots with the same node/feature shape as the
            fitted training data.
        """
        operator = self._require_operator()
        control_matrix = self._require_control_matrix()
        num_nodes, in_channels = self._require_graph_metadata()
        if steps < 1:
            msg = f"steps must be >= 1, got {steps}"
            raise ValueError(msg)
        if len(controls) != steps:
            msg = f"expected {steps} control inputs, got {len(controls)}"
            raise ValueError(msg)
        _check_initial_graph(
            initial_graph,
            num_nodes=num_nodes,
            in_channels=in_channels,
        )

        state = initial_graph.x.reshape(-1)
        topology = _copy_topology(initial_graph)
        predictions: list[Data] = []
        for control in controls:
            control_vector = self._control_vector(control)
            state = state @ operator.T + control_vector @ control_matrix
            x = state.reshape(num_nodes, in_channels)
            predictions.append(Data(x=x, **topology))
        return predictions

    def spectrum(self) -> KoopmanSpectrum:
        """Return the autonomous DMD operator spectrum.

        Takes no kwargs (unlike continuous
        :meth:`~koopman_graph.model.GraphKoopmanModel.spectrum`).

        Returns
        -------
        KoopmanSpectrum
            Eigendecomposition of the fitted state-transition operator ``K``.
        """
        return compute_spectrum(self._require_operator(), self.time_step)

    def _control_vector(self, control: Tensor) -> Tensor:
        """Validate a global control input for prediction.

        Parameters
        ----------
        control : Tensor
            Control input for one rollout step with shape ``(control_dim,)``.

        Returns
        -------
        Tensor
            Row vector with shape ``(control_dim,)``.

        Raises
        ------
        ValueError
            If ``control`` is not a 1-D global vector of the fitted dimension.
        """
        control_dim = self._require_control_dim()
        if control.ndim != 1 or control.shape[0] != control_dim:
            msg = (
                f"global controls must have shape ({control_dim},), "
                f"got {tuple(control.shape)}"
            )
            raise ValueError(msg)
        return control
