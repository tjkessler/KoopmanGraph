"""Hankel-DMD baseline on delay-embedded flattened snapshots.

Arbabi and Mezić (2017) apply DMD to consecutive columns of a Hankel
matrix of observables. This teaching baseline delay-embeds flattened
graph states (topology-blind) and fits the package row-convention map
``delay_{t+1} = delay_t @ K.T``. It is **not**
:class:`~koopman_graph.nn.delay.DelayEmbeddingEncoder` (Takens-style
channel stacking around a GNN).

References
----------
Arbabi, H. & Mezić, I. (2017). Ergodic theory, dynamic mode
decomposition, and computation of spectral properties of the Koopman
operator. *SIAM Journal on Applied Dynamical Systems*, 16(4),
2096–2126. https://doi.org/10.1137/17M1125236
(``Arbabi2017HankelDMD``)
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor
from torch_geometric.data import Data

from koopman_graph.baselines.base import (
    ClassicalBaseline,
    RankSpec,
    check_initial_graph,
    copy_topology,
    fit_row_operator,
    flatten_snapshots,
    require_static_topology,
    resolve_fit_rank,
)
from koopman_graph.data import GraphSnapshotSequence, resolve_sequence
from koopman_graph.spectrum_types import KoopmanSpectrum, compute_spectrum

__all__ = [
    "HankelDMDBaseline",
    "assemble_delay_state",
    "delay_embed_rows",
]


def delay_embed_rows(states: Tensor, n_delays: int) -> Tensor:
    """Stack consecutive flattened states into delay rows (oldest → newest).

    Parameters
    ----------
    states : Tensor
        Flattened snapshot matrix with shape ``(T, state_dim)``.
    n_delays : int
        Window length. Must be ``>= 1``.

    Returns
    -------
    Tensor
        Delay rows with shape ``(T - n_delays + 1, n_delays * state_dim)``.

    Raises
    ------
    ValueError
        If ``n_delays < 1``, ``states`` is not 2-D, or ``T < n_delays``.
    """
    if n_delays < 1:
        msg = f"n_delays must be >= 1, got {n_delays}"
        raise ValueError(msg)
    if states.ndim != 2:
        msg = f"states must be 2-D (T, state_dim), got shape {tuple(states.shape)}"
        raise ValueError(msg)
    num_timesteps, state_dim = int(states.shape[0]), int(states.shape[1])
    if num_timesteps < n_delays:
        msg = (
            f"need at least n_delays={n_delays} snapshots to form a Hankel "
            f"window, got T={num_timesteps}"
        )
        raise ValueError(msg)
    windows = states.unfold(0, n_delays, 1).permute(0, 2, 1)
    return windows.reshape(windows.shape[0], n_delays * state_dim)


def assemble_delay_state(
    initial_graph: Data,
    *,
    history: Sequence[Data] | None,
    n_delays: int,
    num_nodes: int,
    in_channels: int,
    state_dim: int,
) -> Tensor:
    """Build one delay vector from an initial snapshot and optional history.

    Older delay slots are **zero-padded** when ``history`` is missing or
    shorter than ``n_delays - 1``. Extra history snapshots are truncated
    to the newest ``n_delays - 1`` frames (oldest → newest).

    Parameters
    ----------
    initial_graph : Data
        Newest physical snapshot (copied topology is not used here).
    history : sequence of Data or None
        Optional prior snapshots, oldest → newest.
    n_delays : int
        Window length.
    num_nodes, in_channels, state_dim : int
        Fitted layout.

    Returns
    -------
    Tensor
        Delay vector with shape ``(n_delays * state_dim,)``.

    Raises
    ------
    ValueError
        If a snapshot layout does not match the fit metadata.
    """
    check_initial_graph(
        initial_graph,
        num_nodes=num_nodes,
        in_channels=in_channels,
    )
    blocks: list[Tensor] = []
    needed = n_delays - 1
    priors: list[Data] = []
    if history:
        priors = list(history)[-needed:] if needed > 0 else []
    pad = needed - len(priors)
    zero = torch.zeros(
        state_dim,
        dtype=initial_graph.x.dtype,
        device=initial_graph.x.device,
    )
    for _ in range(max(pad, 0)):
        blocks.append(zero)
    for snapshot in priors:
        check_initial_graph(
            snapshot,
            num_nodes=num_nodes,
            in_channels=in_channels,
        )
        blocks.append(snapshot.x.reshape(-1))
    blocks.append(initial_graph.x.reshape(-1))
    return torch.cat(blocks, dim=0)


class HankelDMDBaseline(ClassicalBaseline):
    """Hankel-DMD on delay-embedded flattened graph states (teaching).

    Distinct from :class:`~koopman_graph.nn.delay.DelayEmbeddingEncoder`.
    ``predict`` advances the delay vector with ``K`` and emits the newest
    physical block as ``Data.x``. Without ``history``, older delays are
    zero-padded (teaching-thin).

    Parameters
    ----------
    time_step : float, optional
        Physical duration of one snapshot transition. Default is ``1.0``.
    rank : int or None or {"auto"}, optional
        Truncated-SVD rank for the delay data matrix. Default is ``None``.
    n_delays : int, optional
        Hankel window length. Default is ``2``.
    """

    def __init__(
        self,
        *,
        time_step: float = 1.0,
        rank: RankSpec = None,
        n_delays: int = 2,
    ) -> None:
        """Initialize Hankel-DMD hyperparameters.

        Parameters
        ----------
        time_step : float, optional
            Physical duration of one snapshot transition. Default is ``1.0``.
        rank : int or None or {"auto"}, optional
            Truncated-SVD rank for the delay data matrix. Default is ``None``.
        n_delays : int, optional
            Hankel window length. Default is ``2``.

        Raises
        ------
        ValueError
            If ``time_step`` is not positive, ``rank`` is invalid, or
            ``n_delays < 1``.
        """
        super().__init__(time_step=time_step, rank=rank)
        if n_delays < 1:
            msg = f"n_delays must be >= 1, got {n_delays}"
            raise ValueError(msg)
        self.n_delays = int(n_delays)

    def _is_fitted(self) -> bool:
        """Return whether the delay-coordinate operator has been fit.

        Returns
        -------
        bool
            ``True`` when ``K`` is available.
        """
        return self.K is not None

    def fit(
        self,
        sequence: GraphSnapshotSequence | Sequence[Data],
    ) -> HankelDMDBaseline:
        """Fit DMD on consecutive Hankel delay rows.

        Parameters
        ----------
        sequence : GraphSnapshotSequence or sequence of Data
            Training snapshots with shared topology.

        Returns
        -------
        HankelDMDBaseline
            The fitted baseline (``self``) for sklearn-style chaining.

        Raises
        ------
        ValueError
            If topology is dynamic, ``T < n_delays + 1``, or rank is invalid.
        """
        resolved = resolve_sequence(sequence)
        require_static_topology(resolved)
        min_steps = self.n_delays + 1
        if resolved.num_timesteps < min_steps:
            msg = (
                f"{type(self).__name__}.fit requires at least "
                f"n_delays+1={min_steps} snapshots, got {resolved.num_timesteps}"
            )
            raise ValueError(msg)

        states = flatten_snapshots(resolved)
        rows = delay_embed_rows(states, self.n_delays)
        left = rows[:-1]
        self.selected_rank = resolve_fit_rank(left, self.rank)
        self.K = fit_row_operator(left, rows[1:], self.selected_rank)
        self.num_nodes = resolved.num_nodes
        self.in_channels = resolved.in_channels
        self.state_dim = states.shape[1]
        return self

    def predict(
        self,
        initial_graph: Data,
        steps: int,
        *,
        history: Sequence[Data] | None = None,
    ) -> list[Data]:
        """Advance delay coordinates and emit newest physical snapshots.

        Optional ``history`` supplies older delay slots (oldest → newest).
        Without it, older slots are zeros. Uncontrolled Data-only call site
        remains ``predict(data, steps)``.

        Parameters
        ----------
        initial_graph : Data
            Newest physical snapshot. Topology is copied to predictions.
        steps : int
            Number of future snapshots to predict.
        history : sequence of Data or None, optional
            Prior snapshots, oldest → newest. Default is ``None`` (zero-pad).

        Returns
        -------
        list of Data
            Predicted physical snapshots.

        Raises
        ------
        RuntimeError
            If the baseline has not been fit.
        ValueError
            If ``steps < 1`` or snapshot layout does not match the fit.
        """
        operator = self._require_operator()
        num_nodes, in_channels = self._require_graph_metadata()
        if self.state_dim is None:
            raise RuntimeError(self._unfitted_message())
        if steps < 1:
            msg = f"steps must be >= 1, got {steps}"
            raise ValueError(msg)
        delay = assemble_delay_state(
            initial_graph,
            history=history,
            n_delays=self.n_delays,
            num_nodes=num_nodes,
            in_channels=in_channels,
            state_dim=int(self.state_dim),
        )
        topology = copy_topology(initial_graph)
        predictions: list[Data] = []
        for _ in range(steps):
            delay = delay @ operator.T
            x = delay[-int(self.state_dim) :].reshape(num_nodes, in_channels)
            predictions.append(Data(x=x, **topology))
        return predictions

    def spectrum(self) -> KoopmanSpectrum:
        """Return the delay-coordinate operator spectrum.

        Returns
        -------
        KoopmanSpectrum
            Spectrum of the fitted Hankel-DMD matrix.
        """
        return compute_spectrum(self._require_operator(), self.time_step)
