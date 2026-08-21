"""Generator EDMD (gEDMD) from supplied dictionary derivatives.

Klus et al. (2020) fit a finite-dimensional generator by least squares
on dictionary samples and their time derivatives,
:math:`\\dot\\psi \\approx \\psi L^{\\top}` in the package row convention.
This baseline does **not** form :math:`L` from snapshot finite differences
or from irregular timestamps alone. It is distinct from
:func:`~koopman_graph.analysis.identify_sparse_dynamics` (SINDy / STLSQ
on learned latents, including ``mode="derivative"``).

References
----------
Klus, S., Nüske, F., Peitz, S., Niemann, J.-H. & Schütte, C. (2020).
Data-driven approximation of the Koopman generator: Model reduction,
system identification, and control. *Physica D: Nonlinear Phenomena*,
406, 132416. https://doi.org/10.1016/j.physd.2020.132416
(``Klus2020gEDMD``)
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor
from torch_geometric.data import Data

from koopman_graph.baselines.base import (
    check_initial_graph,
    copy_topology,
    fit_row_operator,
    flatten_snapshots,
    require_static_topology,
    resolve_fit_rank,
)
from koopman_graph.baselines.edmd import EDMDBaseline
from koopman_graph.data import GraphSnapshotSequence, resolve_sequence
from koopman_graph.spectrum_types import KoopmanSpectrum, compute_generator_spectrum

__all__ = [
    "GEDMDBaseline",
]


def _missing_derivative_message() -> str:
    """Return the shared missing-derivative error.

    Returns
    -------
    str
        Actionable ``ValueError`` text.
    """
    return (
        "GEDMDBaseline.fit requires generator-action data: set Data.dx_dt "
        "on every snapshot (same shape as x) or pass derivatives= with shape "
        "(T, state_dim) or (T, N, d). Finite differences of snapshots or "
        "timestamps alone do not identify the generator."
    )


def polynomial_observable_derivatives(
    states: Tensor,
    state_derivatives: Tensor,
    *,
    polynomial_degree: int,
) -> Tensor:
    """Chain-rule the polynomial dictionary through state derivatives.

    Degree 1 is the identity lift (``dψ = dx/dt``). Degree 2 appends
    elementwise squares, so ``d(x²)/dt = 2 x ⊙ dx/dt``.

    Parameters
    ----------
    states : Tensor
        Flattened states with shape ``(num_samples, state_dim)``.
    state_derivatives : Tensor
        Matching ``dx/dt`` with the same shape.
    polynomial_degree : int
        ``1`` or ``2``.

    Returns
    -------
    Tensor
        Dictionary derivatives with shape ``(num_samples, observable_dim)``.

    Raises
    ------
    ValueError
        If shapes disagree or ``polynomial_degree`` is unsupported.
    """
    if states.shape != state_derivatives.shape:
        msg = (
            "states and state_derivatives must share shape, got "
            f"{tuple(states.shape)} and {tuple(state_derivatives.shape)}"
        )
        raise ValueError(msg)
    if polynomial_degree == 1:
        return state_derivatives
    if polynomial_degree == 2:
        return torch.cat(
            [state_derivatives, 2.0 * states * state_derivatives],
            dim=-1,
        )
    msg = f"polynomial_degree must be 1 or 2, got {polynomial_degree}"
    raise ValueError(msg)


def _as_state_derivatives(
    derivatives: Tensor,
    *,
    num_timesteps: int,
    state_dim: int,
    num_nodes: int,
    in_channels: int,
) -> Tensor:
    """Validate and flatten a stacked derivative tensor.

    Parameters
    ----------
    derivatives : Tensor
        ``(T, state_dim)`` or ``(T, N, d)``.
    num_timesteps, state_dim, num_nodes, in_channels : int
        Fitted layout.

    Returns
    -------
    Tensor
        Flattened derivatives with shape ``(T, state_dim)``.

    Raises
    ------
    ValueError
        If rank or shape does not match the snapshot layout.
    """
    if derivatives.ndim == 2:
        expected = (num_timesteps, state_dim)
        if tuple(derivatives.shape) != expected:
            msg = (
                "derivatives must have shape (T, state_dim) "
                f"{expected}, got {tuple(derivatives.shape)}"
            )
            raise ValueError(msg)
        return derivatives
    if derivatives.ndim == 3:
        expected = (num_timesteps, num_nodes, in_channels)
        if tuple(derivatives.shape) != expected:
            msg = (
                "derivatives must have shape (T, N, d) "
                f"{expected}, got {tuple(derivatives.shape)}"
            )
            raise ValueError(msg)
        return derivatives.reshape(num_timesteps, state_dim)
    msg = (
        "derivatives must have shape (T, state_dim) or (T, N, d), "
        f"got ndim={derivatives.ndim} shape={tuple(derivatives.shape)}"
    )
    raise ValueError(msg)


def resolve_state_derivatives(
    sequence: GraphSnapshotSequence,
    derivatives: Tensor | None,
    *,
    num_timesteps: int,
    state_dim: int,
    num_nodes: int,
    in_channels: int,
) -> Tensor:
    """Collect ``dx/dt`` from ``derivatives=`` or per-snapshot ``Data.dx_dt``.

    Parameters
    ----------
    sequence : GraphSnapshotSequence
        Training snapshots.
    derivatives : Tensor or None
        Optional stacked override. When set, snapshot ``dx_dt`` is ignored.
    num_timesteps, state_dim, num_nodes, in_channels : int
        Fitted layout.

    Returns
    -------
    Tensor
        Flattened derivatives with shape ``(T, state_dim)``.

    Raises
    ------
    ValueError
        If derivatives are missing, incomplete, or the wrong shape.
        Finite differences of snapshots are not substituted.
    """
    if derivatives is not None:
        return _as_state_derivatives(
            derivatives,
            num_timesteps=num_timesteps,
            state_dim=state_dim,
            num_nodes=num_nodes,
            in_channels=in_channels,
        )

    rows: list[Tensor] = []
    for index, snapshot in enumerate(sequence):
        if "dx_dt" not in snapshot or snapshot.dx_dt is None:
            raise ValueError(_missing_derivative_message())
        if snapshot.dx_dt.shape != snapshot.x.shape:
            msg = (
                f"snapshot {index} dx_dt shape {tuple(snapshot.dx_dt.shape)} "
                f"does not match x shape {tuple(snapshot.x.shape)}"
            )
            raise ValueError(msg)
        rows.append(snapshot.dx_dt.reshape(-1))
    stacked = torch.stack(rows)
    if tuple(stacked.shape) != (num_timesteps, state_dim):
        msg = (
            "stacked Data.dx_dt must have shape "
            f"{(num_timesteps, state_dim)}, got {tuple(stacked.shape)}"
        )
        raise ValueError(msg)
    return stacked


class GEDMDBaseline(EDMDBaseline):
    """Generator EDMD baseline from supplied dictionary derivatives.

    Same polynomial lift and Data-only ``predict`` surface as
    :class:`~koopman_graph.baselines.EDMDBaseline`, but ``K`` stores the
    generator :math:`L` for :math:`\\dot\\psi \\approx \\psi L^{\\top}`
    rather than a discrete one-step map. ``predict`` advances by
    :math:`\\exp(L\\,\\Delta t)` with fitted ``time_step``. ``spectrum``
    uses :func:`~koopman_graph.spectrum_types.compute_generator_spectrum`.
    Training timestamps are unused: irregular :math:`\\Delta t` does not
    create :math:`L`. RBF / kernel dictionaries raise at ``fit``. Distinct
    from :func:`~koopman_graph.analysis.identify_sparse_dynamics`.

    Notes
    -----
    Constructor arguments match
    :class:`~koopman_graph.baselines.EDMDBaseline`. Only
    ``dictionary="polynomial"`` is supported. Pass generator-action data
    as ``Data.dx_dt`` (same shape as ``x``) or
    ``fit(..., derivatives=)``.
    """

    def fit(
        self,
        sequence: GraphSnapshotSequence | Sequence[Data],
        *,
        derivatives: Tensor | None = None,
    ) -> GEDMDBaseline:
        """Fit the generator and linear reconstruction matrix.

        Parameters
        ----------
        sequence : GraphSnapshotSequence or sequence of Data
            Training snapshots with shared topology. Each snapshot must
            carry ``dx_dt`` unless ``derivatives`` is passed.
        derivatives : Tensor or None, optional
            Optional stacked ``dx/dt`` with shape ``(T, state_dim)`` or
            ``(T, N, d)``. When set, snapshot ``dx_dt`` is ignored.

        Returns
        -------
        GEDMDBaseline
            The fitted baseline (``self``) for sklearn-style chaining.

        Raises
        ------
        ValueError
            If the dictionary is not polynomial, derivatives are missing
            or misshapen, topology is dynamic, or rank is invalid.
        """
        if self.dictionary != "polynomial":
            msg = (
                "GEDMDBaseline supports dictionary='polynomial' only, "
                f"got {self.dictionary!r}. RBF/kernel Jacobians are not "
                "implemented; use EDMDBaseline for those lifts."
            )
            raise ValueError(msg)

        resolved = resolve_sequence(sequence)
        require_static_topology(resolved)
        if resolved.num_timesteps < 1:
            msg = f"{type(self).__name__}.fit requires at least one snapshot"
            raise ValueError(msg)

        states = flatten_snapshots(resolved)
        dx_dt = resolve_state_derivatives(
            resolved,
            derivatives,
            num_timesteps=int(states.shape[0]),
            state_dim=int(states.shape[1]),
            num_nodes=int(resolved.num_nodes),
            in_channels=int(resolved.in_channels),
        ).to(dtype=states.dtype, device=states.device)

        self._nystrom_whitener = None
        self._rff_weight = None
        self._rff_bias = None
        self.centers = None
        observables = self._observables(states)
        d_observables = polynomial_observable_derivatives(
            states,
            dx_dt,
            polynomial_degree=int(self.polynomial_degree),
        )
        self.selected_rank = resolve_fit_rank(observables, self.rank)
        self.K = fit_row_operator(observables, d_observables, self.selected_rank)
        self.reconstruction_matrix = torch.linalg.lstsq(observables, states).solution.T
        self.num_nodes = resolved.num_nodes
        self.in_channels = resolved.in_channels
        self.state_dim = states.shape[1]
        self.observable_dim = observables.shape[1]
        return self

    def predict(self, initial_graph: Data, steps: int) -> list[Data]:
        """Advance snapshots by ``exp(L Δt)`` on the dictionary (Data-only).

        Each step uses fitted ``time_step`` as :math:`\\Delta t`. Training
        timestamps are not consulted. Uncontrolled peer call site shared
        with :class:`~koopman_graph.baselines.EDMDBaseline`.

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
        check_initial_graph(
            initial_graph,
            num_nodes=num_nodes,
            in_channels=in_channels,
        )

        observable = self._observables(initial_graph.x.reshape(1, -1)).squeeze(0)
        step = torch.linalg.matrix_exp(operator * self.time_step)
        topology = copy_topology(initial_graph)
        predictions: list[Data] = []
        for _ in range(steps):
            observable = observable @ step.T
            state = observable @ reconstruction.T
            x = state.reshape(num_nodes, in_channels)
            predictions.append(Data(x=x, **topology))
        return predictions

    def spectrum(self) -> KoopmanSpectrum:
        """Return the fitted generator spectrum.

        Takes no kwargs. Uses
        :func:`~koopman_graph.spectrum_types.compute_generator_spectrum`
        (native continuous-time growth rates), not a discrete map of
        :math:`L`.

        Returns
        -------
        KoopmanSpectrum
            Eigendecomposition of the fitted generator.
        """
        return compute_generator_spectrum(self._require_operator())
