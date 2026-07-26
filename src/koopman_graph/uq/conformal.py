"""Split and adaptive conformal prediction for Koopman forecasts.

:class:`ConformalKoopmanUQ` composes a fitted
:class:`~koopman_graph.model.GraphKoopmanModel` and returns the shared
:class:`~koopman_graph.uq.PredictionInterval` type. Calibration quantiles
live on this wrapper only — they are **not** part of the model checkpoint
``FORMAT_VERSION`` schema.

Coverage semantics
------------------
Split conformal provides frequentist **marginal** coverage
``P(Y ∈ C(X)) ≥ 1 − α`` when calibration and test nonconformity scores are
exchangeable (Vovk / Lei et al.). Graph time series typically violate exact
exchangeability; treat split coverage as approximate under temporal
dependence. Prefer ``method="adaptive"`` (ACI) when residuals drift.
This peer belongs to the package ``uncertainty_quantification`` profile
(power-user ``koopman_graph.uq`` path).
"""

from __future__ import annotations

from collections.abc import Sequence
from math import ceil
from pathlib import Path
from typing import Literal, Self

import torch
from torch import Tensor
from torch_geometric.data import Data

from koopman_graph.data import GraphSnapshotSequence, resolve_sequence
from koopman_graph.model import GraphKoopmanModel
from koopman_graph.model.validation import validate_controls
from koopman_graph.uq.common import PredictionInterval, snapshot_with_features

ConformalMethod = Literal["split", "adaptive"]
ConformalScore = Literal["aggregate", "per_node"]


def _nonconformity_score(
    prediction: Tensor,
    target: Tensor,
    score: ConformalScore,
) -> float:
    """Scalar nonconformity between predicted and target node features.

    Parameters
    ----------

    prediction : Tensor
        See the function signature / summary for ``prediction``.
    target : Tensor
        See the function signature / summary for ``target``.
    score : ConformalScore
        See the function signature / summary for ``score``.

    Returns
    -------

    float
        See summary line.

    Raises
    ------

    ValueError
        Raised when inputs are invalid."""
    residual = prediction - target
    if score == "aggregate":
        return float(residual.abs().max().item())
    if score == "per_node":
        # Max over nodes of per-node feature L2.
        per_node = torch.linalg.vector_norm(residual, dim=-1)
        return float(per_node.max().item())
    msg = f"score must be 'aggregate' or 'per_node', got {score!r}"
    raise ValueError(msg)


def _split_quantile(scores: Tensor, alpha: float) -> float:
    """Finite-sample split-conformal quantile of one-dimensional scores.

    Parameters
    ----------

    scores : Tensor
        See the function signature / summary for ``scores``.
    alpha : float
        See the function signature / summary for ``alpha``.

    Returns
    -------

    float
        See summary line.

    Raises
    ------

    ValueError
        Raised when inputs are invalid."""
    n = int(scores.numel())
    if n < 1:
        msg = "split conformal requires at least one calibration score"
        raise ValueError(msg)
    # Finite-sample split-conformal index (Lei et al. / Vovk).
    sorted_scores = torch.sort(scores.reshape(-1)).values
    index = min(max(ceil((n + 1) * (1.0 - alpha)) - 1, 0), n - 1)
    return float(sorted_scores[index].item())


class ConformalKoopmanUQ:
    """Distribution-free forecast intervals via conformal prediction.

    Parameters
    ----------
    model : GraphKoopmanModel
        Fitted model used for point forecasts during calibration and
        prediction. Never subclassed — this wrapper only composes it.
    method : {"split", "adaptive"}, optional
        ``"split"`` (default) uses batch empirical quantiles.
        ``"adaptive"`` runs Gibbs–Candès ACI updates over the calibration
        stream (recommended under residual drift).
    score : {"aggregate", "per_node"}, optional
        Nonconformity definition. Default ``"aggregate"`` uses the
        entrywise ``L_∞`` residual. ``"per_node"`` uses the max over nodes
        of per-node feature ``L_2``, then a single pooled quantile.
    gamma : float, optional
        ACI step size when ``method="adaptive"``. Default ``0.005``.

    Notes
    -----
    Call :meth:`calibrate` before :meth:`predict_interval`. Intervals are
    symmetric half-widths ``ŷ ± q_h`` in feature space. Marginal coverage
    ``≥ 1 − α`` holds under exchangeability of scores; see module docs.
    """

    def __init__(
        self,
        model: GraphKoopmanModel,
        *,
        method: ConformalMethod = "split",
        score: ConformalScore = "aggregate",
        gamma: float = 0.005,
    ) -> None:
        """Initialize conformal UQ settings.

        Parameters
        ----------
        model, method, score, gamma
            See the class docstring.
        """
        if method not in {"split", "adaptive"}:
            msg = f"method must be 'split' or 'adaptive', got {method!r}"
            raise ValueError(msg)
        if score not in {"aggregate", "per_node"}:
            msg = f"score must be 'aggregate' or 'per_node', got {score!r}"
            raise ValueError(msg)
        if gamma <= 0:
            msg = f"gamma must be positive, got {gamma}"
            raise ValueError(msg)
        self.model = model
        self.method: ConformalMethod = method
        self.score: ConformalScore = score
        self.gamma = float(gamma)
        self._quantiles: Tensor | None = None
        self._alpha: float | None = None
        self._n_calibration: int = 0
        self._calibrated_steps: int = 0

    @property
    def is_calibrated(self) -> bool:
        """Return whether per-horizon quantiles are available.

        Returns
        -------
        bool
            See summary line."""
        return self._quantiles is not None

    @property
    def calibrated_steps(self) -> int:
        """Forecast horizon used at the last successful :meth:`calibrate`.

        Notes
        -----
        Returns ``0`` when not yet calibrated.

        Returns
        -------
        int
            See summary line."""
        return self._calibrated_steps

    @property
    def quantiles(self) -> Tensor:
        """Per-horizon half-widths with shape ``(calibrated_steps,)``.

        Returns
        -------

        Tensor
            See summary line.

        Raises
        ------

        RuntimeError
            If :meth:`calibrate` has not been called."""
        if self._quantiles is None:
            msg = "ConformalKoopmanUQ is not calibrated; call calibrate() first"
            raise RuntimeError(msg)
        return self._quantiles

    def calibrate(
        self,
        calibration_sequences: Sequence[GraphSnapshotSequence | Sequence[Data]],
        *,
        steps: int,
        alpha: float = 0.1,
        controls: Sequence[Sequence[Tensor] | None] | None = None,
        future_topologies: Sequence[Sequence[Data] | None] | None = None,
    ) -> Self:
        """Estimate per-horizon conformal half-widths from held-out sequences.

        Parameters
        ----------
        calibration_sequences : sequence of trajectories
            Each trajectory must have length ``≥ steps + 1``. The first
            snapshot is the rollout origin; the next ``steps`` snapshots are
            targets.
        steps : int
            Forecast horizon used for calibration (and the maximum horizon
            for later :meth:`predict_interval` calls).
        alpha : float, optional
            Target miscoverage rate in ``(0, 1)``. Default ``0.1`` (90%
            nominal marginal coverage under exchangeability).
        controls : sequence of control sequences or None, optional
            Optional per-trajectory future controls aligned with ``steps``.
        future_topologies : sequence of topology schedules or None, optional
            Optional per-trajectory future topologies for hold-last rollout.

        Returns
        -------
        ConformalKoopmanUQ
            ``self`` (calibrated).

        Raises
        ------
        ValueError
            If inputs are invalid or sequences are too short.
        """
        if steps < 1:
            msg = f"steps must be >= 1, got {steps}"
            raise ValueError(msg)
        if not 0.0 < alpha < 1.0:
            msg = f"alpha must lie in (0, 1), got {alpha}"
            raise ValueError(msg)
        if not calibration_sequences:
            msg = "calibration_sequences must be non-empty"
            raise ValueError(msg)

        resolved = [resolve_sequence(seq) for seq in calibration_sequences]
        for index, sequence in enumerate(resolved):
            if sequence.num_timesteps < steps + 1:
                msg = (
                    f"calibration sequence {index} has "
                    f"{sequence.num_timesteps} snapshots; need >= {steps + 1}"
                )
                raise ValueError(msg)

        if controls is not None and len(controls) != len(resolved):
            msg = (
                "controls must align with calibration_sequences; "
                f"got {len(controls)} vs {len(resolved)}"
            )
            raise ValueError(msg)
        if future_topologies is not None and len(future_topologies) != len(resolved):
            msg = (
                "future_topologies must align with calibration_sequences; "
                f"got {len(future_topologies)} vs {len(resolved)}"
            )
            raise ValueError(msg)

        score_rows: list[list[float]] = [[] for _ in range(steps)]
        quantiles = torch.zeros(steps, dtype=torch.float64)

        for seq_id, sequence in enumerate(resolved):
            traj_controls = None if controls is None else controls[seq_id]
            traj_future = (
                None if future_topologies is None else future_topologies[seq_id]
            )
            if traj_controls is not None:
                validate_controls(
                    control_dim=self.model.control_dim,
                    controls=traj_controls,
                    steps=steps,
                )
            origin = sequence[0]
            forecast = self.model.predict(
                origin,
                steps,
                controls=traj_controls,
                future_topologies=traj_future,
            )
            for horizon in range(steps):
                pred_x = forecast[horizon].x
                target_x = sequence[horizon + 1].x
                if pred_x is None or target_x is None:
                    msg = "calibration snapshots must define node features x"
                    raise ValueError(msg)
                score = _nonconformity_score(pred_x, target_x, self.score)
                if self.method == "split":
                    score_rows[horizon].append(score)
                else:
                    q_h = float(quantiles[horizon].item())
                    err = 1.0 if score > q_h else 0.0
                    quantiles[horizon] = max(0.0, q_h + self.gamma * (err - alpha))

        if self.method == "split":
            for horizon in range(steps):
                scores = torch.tensor(score_rows[horizon], dtype=torch.float64)
                quantiles[horizon] = _split_quantile(scores, alpha)

        self._quantiles = quantiles.to(dtype=torch.float32)
        self._alpha = float(alpha)
        self._n_calibration = len(resolved)
        self._calibrated_steps = steps
        return self

    def predict_interval(
        self,
        initial_graph: Tensor | Data,
        steps: int,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
        controls: Sequence[Tensor] | None = None,
        future_topologies: Sequence[Data] | None = None,
        history: Sequence[Data] | None = None,
        *,
        level: float = 0.9,
    ) -> PredictionInterval:
        """Return mean forecast with conformal lower/upper bands.

        Parameters
        ----------
        initial_graph : Tensor or Data
            Rollout origin (same semantics as
            :meth:`~koopman_graph.model.GraphKoopmanModel.predict`).
        steps : int
            Number of forecast steps (``1 ≤ steps ≤ calibrated_steps``).
        edge_index, edge_weight, controls, future_topologies, history
            Forwarded to ``model.predict``.
        level : float, optional
            Nominal central coverage in ``(0, 1)``. Default ``0.9``. Must
            match the calibrated ``1 − alpha`` (validated with a small
            tolerance); use the same nominal level as calibration.

        Returns
        -------
        PredictionInterval
            Mean point forecast plus symmetric conformal bands
            ``mean ± q_h``. ``n_members`` is the calibration set size.

        Raises
        ------
        RuntimeError
            If not calibrated.
        ValueError
            If ``steps`` / ``level`` disagree with calibration.
        """
        if self._quantiles is None or self._alpha is None:
            msg = "ConformalKoopmanUQ is not calibrated; call calibrate() first"
            raise RuntimeError(msg)
        if steps < 1:
            msg = f"steps must be >= 1, got {steps}"
            raise ValueError(msg)
        if steps > self._calibrated_steps:
            msg = f"steps={steps} exceeds calibrated horizon {self._calibrated_steps}"
            raise ValueError(msg)
        if not 0.0 < level < 1.0:
            msg = f"level must lie in (0, 1), got {level}"
            raise ValueError(msg)
        expected_level = 1.0 - self._alpha
        if abs(level - expected_level) > 1e-6:
            msg = (
                f"level={level} does not match calibrated 1 - alpha = {expected_level}"
            )
            raise ValueError(msg)

        mean_snaps = self.model.predict(
            initial_graph,
            steps,
            edge_index=edge_index,
            edge_weight=edge_weight,
            controls=controls,
            future_topologies=future_topologies,
            history=history,
        )
        lower_snaps: list[Data] = []
        upper_snaps: list[Data] = []
        for horizon, mean_snap in enumerate(mean_snaps):
            if mean_snap.x is None:
                msg = "predicted snapshots must define node features x"
                raise ValueError(msg)
            half = self._quantiles[horizon].to(
                device=mean_snap.x.device,
                dtype=mean_snap.x.dtype,
            )
            lower_snaps.append(snapshot_with_features(mean_snap, mean_snap.x - half))
            upper_snaps.append(snapshot_with_features(mean_snap, mean_snap.x + half))

        return PredictionInterval(
            mean=mean_snaps,
            lower=lower_snaps,
            upper=upper_snaps,
            level=level,
            n_members=self._n_calibration,
        )

    def save_calibration(self, path: str | Path) -> None:
        """Persist calibration quantiles (wrapper state only).

        Parameters
        ----------
        path : str or Path
            Destination file for ``torch.save``.
        """
        if self._quantiles is None or self._alpha is None:
            msg = "ConformalKoopmanUQ is not calibrated; call calibrate() first"
            raise RuntimeError(msg)
        payload = {
            "kind": "ConformalKoopmanUQ.calibration",
            "method": self.method,
            "score": self.score,
            "gamma": self.gamma,
            "alpha": self._alpha,
            "quantiles": self._quantiles.detach().cpu(),
            "n_calibration": self._n_calibration,
            "calibrated_steps": self._calibrated_steps,
        }
        torch.save(payload, Path(path))

    def load_calibration(self, path: str | Path) -> Self:
        """Load calibration quantiles saved by :meth:`save_calibration`.

        Parameters
        ----------
        path : str or Path
            File written by :meth:`save_calibration`.

        Returns
        -------
        ConformalKoopmanUQ
            ``self`` with restored quantiles.

        Raises
        ------
        ValueError
            If the payload kind or method/score disagree with this instance.
        """
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if payload.get("kind") != "ConformalKoopmanUQ.calibration":
            msg = f"unsupported calibration payload kind: {payload.get('kind')!r}"
            raise ValueError(msg)
        if payload.get("method") != self.method or payload.get("score") != self.score:
            msg = (
                "calibration method/score "
                f"({payload.get('method')!r}, {payload.get('score')!r}) "
                f"do not match this instance ({self.method!r}, {self.score!r})"
            )
            raise ValueError(msg)
        self.gamma = float(payload.get("gamma", self.gamma))
        self._alpha = float(payload["alpha"])
        self._quantiles = payload["quantiles"].to(dtype=torch.float32)
        self._n_calibration = int(payload["n_calibration"])
        self._calibrated_steps = int(payload["calibrated_steps"])
        return self
