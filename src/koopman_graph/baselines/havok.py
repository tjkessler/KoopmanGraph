"""HAVOK teaching baseline on delay-embedded flattened snapshots.

Brunton et al. (2017) form a Hankel matrix of a measurement, take an SVD,
and fit a linear model on the leading time coordinates forced by the last
retained mode. This teaching path uses the same vector delay rows as
:class:`~koopman_graph.baselines.HankelDMDBaseline` (topology-blind). It is
**not** :class:`~koopman_graph.nn.delay.DelayEmbeddingEncoder` and **not**
a Mori–Zwanzig memory kernel. Autonomous ``predict`` sets the forcing
coordinate to zero.

References
----------
Brunton, S. L., Brunton, B. W., Proctor, J. L., Kaiser, E. & Kutz, J. N.
(2017). Chaos as an intermittently forced linear system. *Nature
Communications*, 8, 19. https://doi.org/10.1038/s41467-017-00030-8
(``Brunton2017HAVOK``)
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor
from torch_geometric.data import Data

from koopman_graph.baselines.base import (
    ClassicalBaseline,
    copy_topology,
    flatten_snapshots,
    require_static_topology,
)
from koopman_graph.baselines.hankel_dmd import assemble_delay_state, delay_embed_rows
from koopman_graph.data import GraphSnapshotSequence, resolve_sequence
from koopman_graph.spectrum_types import KoopmanSpectrum, compute_spectrum

__all__ = [
    "HAVOKBaseline",
]


class HAVOKBaseline(ClassicalBaseline):
    """Hankel alternative view of Koopman (teaching-thin SVD + forced linear map).

    Distinct from :class:`~koopman_graph.nn.delay.DelayEmbeddingEncoder`.
    ``K`` stores the autonomous factor :math:`A` on the first
    ``havok_rank - 1`` time coordinates. ``predict`` without ``history``
    zero-pads older delays and uses forcing :math:`u=0`.

    Parameters
    ----------
    time_step : float, optional
        Physical duration of one snapshot transition. Default is ``1.0``.
    n_delays : int, optional
        Hankel window length. Default is ``2``.
    havok_rank : int, optional
        SVD truncation ``r >= 2`` (last mode is forcing). Default is ``3``.
        Must not exceed ``min(num_windows, n_delays * state_dim)`` at fit.
    """

    def __init__(
        self,
        *,
        time_step: float = 1.0,
        n_delays: int = 2,
        havok_rank: int = 3,
    ) -> None:
        """Initialize HAVOK hyperparameters.

        Parameters
        ----------
        time_step : float, optional
            Physical duration of one snapshot transition. Default is ``1.0``.
        n_delays : int, optional
            Hankel window length. Default is ``2``.
        havok_rank : int, optional
            SVD truncation ``r >= 2``. Default is ``3``.

        Raises
        ------
        ValueError
            If ``time_step`` is not positive, ``n_delays < 1``, or
            ``havok_rank < 2``.
        """
        super().__init__(time_step=time_step, rank=None)
        if n_delays < 1:
            msg = f"n_delays must be >= 1, got {n_delays}"
            raise ValueError(msg)
        if havok_rank < 2:
            msg = f"havok_rank must be >= 2, got {havok_rank}"
            raise ValueError(msg)
        self.n_delays = int(n_delays)
        self.havok_rank = int(havok_rank)
        self.B: Tensor | None = None
        self.singular_values: Tensor | None = None
        self._spatial_modes: Tensor | None = None

    def _is_fitted(self) -> bool:
        """Return whether the HAVOK linear factors have been fit.

        Returns
        -------
        bool
            ``True`` when ``K`` (the autonomous ``A``) is available.
        """
        return self.K is not None and self._spatial_modes is not None

    def fit(
        self,
        sequence: GraphSnapshotSequence | Sequence[Data],
    ) -> HAVOKBaseline:
        """Fit SVD time coordinates and a forced linear map.

        Parameters
        ----------
        sequence : GraphSnapshotSequence or sequence of Data
            Training snapshots with shared topology.

        Returns
        -------
        HAVOKBaseline
            The fitted baseline (``self``) for sklearn-style chaining.

        Raises
        ------
        ValueError
            If topology is dynamic, the Hankel is too small for
            ``havok_rank``, or ``T < n_delays + 1``.
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
        hankel = rows.T.contiguous()
        max_rank = int(min(hankel.shape))
        if self.havok_rank > max_rank:
            msg = (
                f"havok_rank={self.havok_rank} exceeds Hankel rank bound "
                f"{max_rank} (shape {tuple(hankel.shape)}); increase "
                "n_delays or T, or reduce havok_rank"
            )
            raise ValueError(msg)

        spatial, singular_values, time_factors = torch.linalg.svd(
            hankel,
            full_matrices=False,
        )
        rank = self.havok_rank
        modes = spatial[:, :rank] * singular_values[:rank].unsqueeze(0)
        coords = time_factors[:rank, :].T.contiguous()
        state_coords = coords[:-1, :-1]
        forcing = coords[:-1, -1:]
        targets = coords[1:, :-1]
        augmented = torch.cat([state_coords, forcing], dim=1)
        solution = torch.linalg.lstsq(augmented, targets).solution
        self.K = solution[:-1, :].T.contiguous()
        self.B = solution[-1:, :].T.contiguous()
        self._spatial_modes = modes
        self.singular_values = singular_values[:rank].clone()
        self.selected_rank = rank
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
        """Advance HAVOK coordinates with forcing ``u=0`` and decode delays.

        Optional ``history`` supplies older delay slots (oldest → newest).
        Without it, older slots are zeros.

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
        if self._spatial_modes is None or self.state_dim is None:
            raise RuntimeError(self._unfitted_message())
        num_nodes, in_channels = self._require_graph_metadata()
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
        coords = torch.linalg.lstsq(self._spatial_modes, delay.unsqueeze(1)).solution
        state = coords[:-1, 0]
        topology = copy_topology(initial_graph)
        predictions: list[Data] = []
        zero_force = torch.zeros(
            1,
            dtype=state.dtype,
            device=state.device,
        )
        for _ in range(steps):
            state = state @ operator.T
            full = torch.cat([state, zero_force], dim=0)
            delay = self._spatial_modes @ full
            x = delay[-int(self.state_dim) :].reshape(num_nodes, in_channels)
            predictions.append(Data(x=x, **topology))
        return predictions

    def spectrum(self) -> KoopmanSpectrum:
        """Return the spectrum of the autonomous HAVOK factor ``A``.

        Returns
        -------
        KoopmanSpectrum
            Spectrum of ``K`` (the ``r-1`` linear block, not the forcing).
        """
        return compute_spectrum(self._require_operator(), self.time_step)
