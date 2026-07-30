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

With ``score="node_wise"``, quantiles are **marginal per node**: each node
has its own coverage guarantee under exchangeability of that node's scores.
This is **not** joint / simultaneous coverage across nodes. Small calibration
sets make per-node quantiles noisy; ``node_wise`` calibration requires at
least ``ceil(1 / alpha)`` sequences.

Optional ``neighbor_smoothing`` applies DAPS-style diffusion to node-wise
regression residual scores. Zargarbashi et al. (ICML 2023) prove
exchangeability preservation for *classification* conformity scores; applying
the same diffusion to regression residuals is an empirical adaptation, not a
transferred theorem.

This peer belongs to the package ``uncertainty_quantification`` profile
(power-user ``koopman_graph.uq`` path).

References
----------
Zargarbashi, S. H., Antonelli, S., & Bojchevski, A. (2023). Conformal
prediction sets for graph neural networks. In *Proceedings of the 40th
International Conference on Machine Learning* (PMLR 202:12292–12318).
https://proceedings.mlr.press/v202/h-zargarbashi23a.html
(``Zargarbashi2023ConformalGNN``)
"""

from __future__ import annotations

from collections.abc import Sequence
from math import ceil
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor
from torch_geometric.data import Data, HeteroData

from koopman_graph.data import (
    GraphSnapshotSequence,
    HeteroGraphSnapshotSequence,
    resolve_sequence,
)
from koopman_graph.graph_utils import random_walk_normalized_adjacency_matvec
from koopman_graph.model import GraphKoopmanModel
from koopman_graph.model.validation import validate_controls
from koopman_graph.uq.common import PredictionInterval, snapshot_with_features

ConformalMethod = Literal["split", "adaptive"]
ConformalScore = Literal["aggregate", "per_node", "node_wise"]

_CALIBRATION_KIND = "ConformalKoopmanUQ.calibration.v2"


def _nonconformity_score(
    prediction: Tensor,
    target: Tensor,
    score: Literal["aggregate", "per_node"],
) -> float:
    """Scalar nonconformity between predicted and target node features.

    Parameters
    ----------
    prediction : Tensor
        Predicted node features ``(num_nodes, num_features)``.
    target : Tensor
        Target node features with the same shape.
    score : {"aggregate", "per_node"}
        Scalar score definition.

    Returns
    -------
    float
        Nonconformity score.

    Raises
    ------
    ValueError
        If ``score`` is not a scalar mode.
    """
    residual = prediction - target
    if score == "aggregate":
        return float(residual.abs().max().item())
    if score == "per_node":
        # Max over nodes of per-node feature L2 (legacy pooled scalar).
        per_node = torch.linalg.vector_norm(residual, dim=-1)
        return float(per_node.max().item())
    msg = f"score must be 'aggregate' or 'per_node', got {score!r}"
    raise ValueError(msg)


def _node_wise_scores(prediction: Tensor, target: Tensor) -> Tensor:
    """Per-node feature ``L_2`` residual norms with shape ``(num_nodes,)``.

    Parameters
    ----------
    prediction : Tensor
        Predicted node features ``(num_nodes, num_features)``.
    target : Tensor
        Target node features with the same shape.

    Returns
    -------
    Tensor
        Per-node nonconformity scores.
    """
    residual = prediction - target
    return torch.linalg.vector_norm(residual, dim=-1)


def _diffuse_node_scores(
    scores: Tensor,
    edge_index: Tensor,
    *,
    edge_weight: Tensor | None,
    lam: float,
) -> Tensor:
    """Apply DAPS-style score diffusion ``(1-λ)S + λ D_out^{-1} A S``.

    Parameters
    ----------
    scores : Tensor
        Per-node scores with shape ``(num_nodes,)``.
    edge_index : Tensor
        Graph edge index ``(2, num_edges)``.
    edge_weight : Tensor or None
        Optional edge weights.
    lam : float
        Diffusion weight in ``[0, 1]``.

    Returns
    -------
    Tensor
        Diffused scores with shape ``(num_nodes,)``.
    """
    if lam == 0.0:
        return scores
    diffused = random_walk_normalized_adjacency_matvec(
        edge_index,
        scores.unsqueeze(-1),
        edge_weight=edge_weight,
        num_nodes=int(scores.shape[0]),
        direction="forward",
    ).squeeze(-1)
    return (1.0 - lam) * scores + lam * diffused


def _split_quantile(scores: Tensor, alpha: float) -> float:
    """Finite-sample split-conformal quantile of one-dimensional scores.

    Parameters
    ----------
    scores : Tensor
        Calibration scores (any shape; flattened).
    alpha : float
        Miscoverage rate in ``(0, 1)``.

    Returns
    -------
    float
        Empirical quantile half-width.

    Raises
    ------
    ValueError
        If ``scores`` is empty.
    """
    n = int(scores.numel())
    if n < 1:
        msg = "split conformal requires at least one calibration score"
        raise ValueError(msg)
    # Finite-sample split-conformal index (Lei et al. / Vovk).
    sorted_scores = torch.sort(scores.reshape(-1)).values
    index = min(max(ceil((n + 1) * (1.0 - alpha)) - 1, 0), n - 1)
    return float(sorted_scores[index].item())


def _split_quantiles_per_column(score_matrix: Tensor, alpha: float) -> Tensor:
    """Column-wise split conformal quantiles.

    Parameters
    ----------
    score_matrix : Tensor
        Scores with shape ``(n_calibration, num_nodes)``.
    alpha : float
        Miscoverage rate.

    Returns
    -------
    Tensor
        Quantiles with shape ``(num_nodes,)`` (float64).
    """
    num_nodes = int(score_matrix.shape[1])
    out = torch.empty(num_nodes, dtype=torch.float64)
    for node in range(num_nodes):
        out[node] = _split_quantile(score_matrix[:, node], alpha)
    return out


def _minimum_calibration_count(alpha: float) -> int:
    """Return the minimum calibration-set size for ``node_wise`` mode.

    Parameters
    ----------
    alpha : float
        Target miscoverage rate in ``(0, 1)``.

    Returns
    -------
    int
        ``ceil(1 / alpha)``.
    """
    return int(ceil(1.0 / alpha))


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
    score : {"aggregate", "per_node", "node_wise"}, optional
        Nonconformity definition. Default ``"aggregate"`` uses the
        entrywise ``L_∞`` residual and one scalar quantile per horizon.
        ``"per_node"`` (legacy) takes the **max over nodes** of per-node
        feature ``L_2``, then a single pooled quantile — intervals remain
        homoscedastic across nodes. ``"node_wise"`` keeps a per-node score
        vector and ``(steps, num_nodes)`` quantiles for heteroscedastic
        widths. Prefer ``"node_wise"`` when per-node intervals are desired;
        ``"per_node"`` is retained for compatibility.
    gamma : float, optional
        ACI step size when ``method="adaptive"``. Default ``0.005``.
    neighbor_smoothing : float or None, optional
        DAPS-style diffusion weight ``λ`` applied to ``node_wise`` scores
        before quantiles: ``Ŝ = (1-λ)S + λ D_out^{-1} A S``. Must lie in
        ``[0, 1]``. ``None`` (default) disables smoothing. ``0`` reproduces
        unsmoothed ``node_wise`` exactly. Only valid with
        ``score="node_wise"``.

    Notes
    -----
    Call :meth:`calibrate` before :meth:`predict_interval`. Intervals are
    symmetric half-widths ``ŷ ± q`` in feature space. For ``node_wise``,
    ``q`` is per-node and broadcasts over features. Coverage is **marginal
    per node**, not joint across nodes. ``node_wise`` calibration requires
    at least ``ceil(1 / alpha)`` sequences.

    Score diffusion follows Zargarbashi et al. (ICML 2023) for classification
    conformity scores; using it on regression residuals is an adaptation with
    empirical rather than theoretical warrant.

    References
    ----------
    Zargarbashi, Antonelli & Bojchevski, ICML 2023 (PMLR 202:12292–12318).
    """

    def __init__(
        self,
        model: GraphKoopmanModel,
        *,
        method: ConformalMethod = "split",
        score: ConformalScore = "aggregate",
        gamma: float = 0.005,
        neighbor_smoothing: float | None = None,
    ) -> None:
        """Initialize conformal UQ settings.

        Parameters
        ----------
        model, method, score, gamma, neighbor_smoothing
            See the class docstring.
        """
        if method not in {"split", "adaptive"}:
            msg = f"method must be 'split' or 'adaptive', got {method!r}"
            raise ValueError(msg)
        if score not in {"aggregate", "per_node", "node_wise"}:
            msg = (
                f"score must be 'aggregate', 'per_node', or 'node_wise', got {score!r}"
            )
            raise ValueError(msg)
        if gamma <= 0:
            msg = f"gamma must be positive, got {gamma}"
            raise ValueError(msg)
        if neighbor_smoothing is not None:
            if score != "node_wise":
                msg = (
                    "neighbor_smoothing requires score='node_wise'; "
                    f"got score={score!r}"
                )
                raise ValueError(msg)
            if not 0.0 <= float(neighbor_smoothing) <= 1.0:
                msg = f"neighbor_smoothing must lie in [0, 1], got {neighbor_smoothing}"
                raise ValueError(msg)
        if getattr(model, "uses_hetero_koopman", False):
            msg = (
                "ConformalKoopmanUQ is homogeneous-only; multiplex / "
                "koopman='hetero_graph' models are not supported yet"
            )
            raise TypeError(msg)
        self.model = model
        self.method: ConformalMethod = method
        self.score: ConformalScore = score
        self.gamma = float(gamma)
        self.neighbor_smoothing = (
            None if neighbor_smoothing is None else float(neighbor_smoothing)
        )
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
            ``True`` after a successful :meth:`calibrate` or
            :meth:`load_calibration`.
        """
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
            Calibrated forecast horizon.
        """
        return self._calibrated_steps

    @property
    def quantiles(self) -> Tensor:
        """Per-horizon half-widths.

        Shape is ``(calibrated_steps,)`` for ``aggregate`` / ``per_node``,
        or ``(calibrated_steps, num_nodes)`` for ``node_wise``.

        Returns
        -------
        Tensor
            Calibration quantiles.

        Raises
        ------
        RuntimeError
            If :meth:`calibrate` has not been called.
        """
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
    ) -> ConformalKoopmanUQ:
        """Estimate per-horizon conformal half-widths from held-out sequences.

        Parameters
        ----------
        calibration_sequences : sequence of trajectories
            Each trajectory must have length ``≥ steps + 1``. The first
            snapshot is the rollout origin; the next ``steps`` snapshots are
            targets. For ``score="node_wise"``, length must be at least
            ``ceil(1 / alpha)``.
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
            If inputs are invalid, sequences are too short, or (for
            ``node_wise``) the calibration set is smaller than
            ``ceil(1 / alpha)``.
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
        for seq in calibration_sequences:
            if isinstance(seq, HeteroGraphSnapshotSequence) or (
                isinstance(seq, Sequence) and seq and isinstance(seq[0], HeteroData)
            ):
                msg = (
                    "ConformalKoopmanUQ is homogeneous-only; "
                    "HeteroData / HeteroGraphSnapshotSequence calibration "
                    "is not supported yet"
                )
                raise TypeError(msg)

        resolved = [resolve_sequence(seq) for seq in calibration_sequences]
        if self.score == "node_wise":
            min_count = _minimum_calibration_count(alpha)
            if len(resolved) < min_count:
                msg = (
                    "node_wise calibration requires at least "
                    f"ceil(1/alpha)={min_count} sequences for alpha={alpha}; "
                    f"got {len(resolved)}. Provide more calibration sequences "
                    "or increase alpha."
                )
                raise ValueError(msg)
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

        if self.score == "node_wise":
            self._calibrate_node_wise(
                resolved,
                steps=steps,
                alpha=alpha,
                controls=controls,
                future_topologies=future_topologies,
            )
        else:
            self._calibrate_scalar(
                resolved,
                steps=steps,
                alpha=alpha,
                controls=controls,
                future_topologies=future_topologies,
            )
        return self

    def _calibrate_scalar(
        self,
        resolved: list[GraphSnapshotSequence],
        *,
        steps: int,
        alpha: float,
        controls: Sequence[Sequence[Tensor] | None] | None,
        future_topologies: Sequence[Sequence[Data] | None] | None,
    ) -> None:
        """Calibrate pooled scalar quantiles (``aggregate`` / ``per_node``).

        Parameters
        ----------
        resolved : list of GraphSnapshotSequence
            Resolved calibration trajectories.
        steps : int
            Forecast horizon.
        alpha : float
            Miscoverage rate.
        controls : sequence or None
            Optional per-trajectory controls.
        future_topologies : sequence or None
            Optional per-trajectory topology schedules.
        """
        score_mode: Literal["aggregate", "per_node"] = self.score  # type: ignore[assignment]
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
                score = _nonconformity_score(pred_x, target_x, score_mode)
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

    def _calibrate_node_wise(
        self,
        resolved: list[GraphSnapshotSequence],
        *,
        steps: int,
        alpha: float,
        controls: Sequence[Sequence[Tensor] | None] | None,
        future_topologies: Sequence[Sequence[Data] | None] | None,
    ) -> None:
        """Calibrate per-node quantiles with optional score diffusion.

        Parameters
        ----------
        resolved : list of GraphSnapshotSequence
            Resolved calibration trajectories (fixed node count).
        steps : int
            Forecast horizon.
        alpha : float
            Miscoverage rate.
        controls : sequence or None
            Optional per-trajectory controls.
        future_topologies : sequence or None
            Optional per-trajectory topology schedules.
        """
        origin0 = resolved[0][0]
        if origin0.x is None:
            msg = "calibration snapshots must define node features x"
            raise ValueError(msg)
        num_nodes = int(origin0.x.shape[0])
        lam = 0.0 if self.neighbor_smoothing is None else self.neighbor_smoothing
        score_rows: list[list[Tensor]] = [[] for _ in range(steps)]
        quantiles = torch.zeros(steps, num_nodes, dtype=torch.float64)

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
            if origin.x is None or int(origin.x.shape[0]) != num_nodes:
                msg = (
                    "node_wise calibration requires a fixed node count; "
                    f"expected {num_nodes} nodes"
                )
                raise ValueError(msg)
            if lam > 0.0 and origin.edge_index is None:
                msg = "neighbor_smoothing requires edge_index on calibration origins"
                raise ValueError(msg)
            edge_weight = getattr(origin, "edge_weight", None)
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
                scores = _node_wise_scores(pred_x, target_x).to(dtype=torch.float64)
                if lam != 0.0:
                    scores = _diffuse_node_scores(
                        scores,
                        origin.edge_index,
                        edge_weight=edge_weight,
                        lam=lam,
                    )
                if self.method == "split":
                    score_rows[horizon].append(scores.detach().cpu())
                else:
                    q_h = quantiles[horizon]
                    err = (scores > q_h).to(dtype=torch.float64)
                    quantiles[horizon] = torch.clamp(
                        q_h + self.gamma * (err - alpha),
                        min=0.0,
                    )

        if self.method == "split":
            for horizon in range(steps):
                score_matrix = torch.stack(score_rows[horizon], dim=0)
                quantiles[horizon] = _split_quantiles_per_column(score_matrix, alpha)

        self._quantiles = quantiles.to(dtype=torch.float32)
        self._alpha = float(alpha)
        self._n_calibration = len(resolved)
        self._calibrated_steps = steps

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
            ``mean ± q``. For ``node_wise``, ``q`` has shape ``(num_nodes,)``
            and broadcasts over features. ``n_members`` is the calibration
            set size.

        Raises
        ------
        TypeError
            If ``initial_graph`` is multiplex ``HeteroData``.
        RuntimeError
            If not calibrated.
        ValueError
            If ``steps`` / ``level`` disagree with calibration.
        """
        if self._quantiles is None or self._alpha is None:
            msg = "ConformalKoopmanUQ is not calibrated; call calibrate() first"
            raise RuntimeError(msg)
        if isinstance(initial_graph, HeteroData):
            msg = (
                "ConformalKoopmanUQ is homogeneous-only; HeteroData origins "
                "are not supported yet"
            )
            raise TypeError(msg)
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
        node_wise = self.score == "node_wise"
        for horizon, mean_snap in enumerate(mean_snaps):
            if mean_snap.x is None:
                msg = "predicted snapshots must define node features x"
                raise ValueError(msg)
            half = self._quantiles[horizon].to(
                device=mean_snap.x.device,
                dtype=mean_snap.x.dtype,
            )
            if node_wise:
                if half.ndim != 1 or half.shape[0] != mean_snap.x.shape[0]:
                    msg = (
                        "node_wise quantiles must match the forecast node "
                        f"count; got {tuple(half.shape)} vs "
                        f"{mean_snap.x.shape[0]} nodes"
                    )
                    raise ValueError(msg)
                half = half.unsqueeze(-1)
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

        Notes
        -----
        Payload ``kind`` is ``ConformalKoopmanUQ.calibration.v2`` and includes
        ``score``, ``neighbor_smoothing``, and quantile tensors (scalar or
        per-node). This is independent of model ``FORMAT_VERSION``.
        """
        if self._quantiles is None or self._alpha is None:
            msg = "ConformalKoopmanUQ is not calibrated; call calibrate() first"
            raise RuntimeError(msg)
        payload = {
            "kind": _CALIBRATION_KIND,
            "method": self.method,
            "score": self.score,
            "neighbor_smoothing": self.neighbor_smoothing,
            "gamma": self.gamma,
            "alpha": self._alpha,
            "quantiles": self._quantiles.detach().cpu(),
            "n_calibration": self._n_calibration,
            "calibrated_steps": self._calibrated_steps,
        }
        torch.save(payload, Path(path))

    def load_calibration(self, path: str | Path) -> ConformalKoopmanUQ:
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
            If the payload kind or method/score/smoothing disagree with this
            instance.
        """
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if payload.get("kind") != _CALIBRATION_KIND:
            msg = f"unsupported calibration payload kind: {payload.get('kind')!r}"
            raise ValueError(msg)
        if payload.get("method") != self.method or payload.get("score") != self.score:
            msg = (
                "calibration method/score "
                f"({payload.get('method')!r}, {payload.get('score')!r}) "
                f"do not match this instance ({self.method!r}, {self.score!r})"
            )
            raise ValueError(msg)
        stored_smooth = payload.get("neighbor_smoothing", None)
        if stored_smooth != self.neighbor_smoothing:
            msg = (
                "calibration neighbor_smoothing "
                f"{stored_smooth!r} does not match this instance "
                f"{self.neighbor_smoothing!r}"
            )
            raise ValueError(msg)
        self.gamma = float(payload.get("gamma", self.gamma))
        self._alpha = float(payload["alpha"])
        self._quantiles = payload["quantiles"].to(dtype=torch.float32)
        self._n_calibration = int(payload["n_calibration"])
        self._calibrated_steps = int(payload["calibrated_steps"])
        return self
