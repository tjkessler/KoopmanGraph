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

Heterogeneous models (``koopman="hetero_graph"``) score **stacked** decoded
features with shape ``(N, F)`` where ``N = Σ_τ N_τ`` in the operator's
``node_types`` order (shared trailing feature width ``F``; unequal per-type
feature widths are unsupported here). Opt-in unequal latent widths ``d_τ``
are fine when decoded features still share ``F`` — scoring does not use the
latent layout. Interval bands are packed as ``HeteroData``. The same
marginal / exchangeability honesty applies to stacked node rows; this is not
joint coverage across types or relations.

Optional ``neighbor_smoothing`` applies DAPS-style diffusion to node-wise
regression residual scores. Zargarbashi et al. (ICML 2023) prove
exchangeability preservation for *classification* conformity scores; applying
the same diffusion to regression residuals is an empirical adaptation, not a
transferred theorem. On hetero graphs, diffusion uses the **union** of
relation banks in stacked global numbering.

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
    resolve_hetero_sequence,
    resolve_sequence,
)
from koopman_graph.data.hetero_layout import (
    global_relation_edge_indices,
    stack_typed_features,
)
from koopman_graph.graph_utils import random_walk_normalized_adjacency_matvec
from koopman_graph.model import GraphKoopmanModel
from koopman_graph.model.validation import validate_controls
from koopman_graph.operators import HeteroGraphKoopmanOperator
from koopman_graph.uq.common import (
    PredictionInterval,
    SnapshotLike,
    hetero_snapshot_with_features,
    snapshot_with_features,
)

CalibrationSequence = (
    GraphSnapshotSequence
    | HeteroGraphSnapshotSequence
    | Sequence[Data]
    | Sequence[HeteroData]
)
ResolvedSequence = GraphSnapshotSequence | HeteroGraphSnapshotSequence

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


def _is_hetero_sequence(sequence: object) -> bool:
    """Return whether ``sequence`` is a hetero trajectory container or list.

    Parameters
    ----------
    sequence
        Value for ``sequence``.

    Returns
    -------
    object
        Function result.
    """
    if isinstance(sequence, HeteroGraphSnapshotSequence):
        return True
    if isinstance(sequence, Sequence) and not isinstance(sequence, (str, bytes)):
        return bool(sequence) and isinstance(sequence[0], HeteroData)
    return False


def _stack_hetero_features(
    snapshot: HeteroData,
    node_types: Sequence[str],
) -> Tensor:
    """Stack per-type ``x`` blocks into ``(N, F)`` with shared trailing width.

    Parameters
    ----------
    snapshot : HeteroData
        Multiplex or typed snapshot.
    node_types : sequence of str
        Operator node-type order.

    Returns
    -------
    Tensor
        Stacked features ``(Σ_τ N_τ, F)``.

    Raises
    ------
    ValueError
        If a type is missing ``x`` or trailing widths disagree.
    """
    feature_dict: dict[str, Tensor] = {}
    for name in node_types:
        if name not in snapshot.node_types:
            msg = (
                f"HeteroData snapshot is missing node type {name!r}; "
                f"present types are {sorted(snapshot.node_types)!r}"
            )
            raise ValueError(msg)
        features = snapshot[name].x
        if features is None:
            msg = f"HeteroData node type {name!r} is missing feature matrix x"
            raise ValueError(msg)
        feature_dict[name] = features
    return stack_typed_features(feature_dict, node_types)


def _union_relation_edge_index(
    snapshot: HeteroData,
    edge_types: Sequence[Sequence[str]],
    node_types: Sequence[str],
) -> Tensor:
    """Concatenate relation banks into one stacked-global ``edge_index``.

    Parameters
    ----------
    snapshot : HeteroData
        Origin snapshot carrying per-relation banks.
    edge_types : sequence of sequence of str
        Operator edge-type schema.
    node_types : sequence of str
        Operator node-type order.

    Returns
    -------
    Tensor
        Union ``edge_index`` with shape ``(2, E_union)``.

    Raises
    ------
    ValueError
        If every relation bank is empty.
    """
    banks = global_relation_edge_indices(snapshot, edge_types, node_types)
    nonempty = [bank for bank in banks if bank.numel() > 0]
    if not nonempty:
        msg = (
            "neighbor_smoothing requires at least one non-empty relation "
            "bank on the calibration / predict origin"
        )
        raise ValueError(msg)
    return torch.cat(nonempty, dim=1)


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
    ``q`` is per-node (stacked global rows when hetero) and broadcasts over
    features. Reported coverage is **marginal** (and **marginal per node**
    under ``score="node_wise"``), not joint across nodes, types, or
    horizons. Exact frequentist coverage assumes exchangeable calibration
    and test scores; graph time series typically violate that assumption, so
    treat split coverage as approximate under temporal dependence (see the
    module ``Coverage semantics`` section). ``node_wise`` calibration
    requires at least ``ceil(1 / alpha)`` sequences.

    Score diffusion follows Zargarbashi et al. (ICML 2023) for classification
    conformity scores; using it on regression residuals is an adaptation with
    empirical rather than theoretical warrant. Hetero diffusion uses the
    union of relation banks.

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
    def _uses_hetero(self) -> bool:
        """Return whether the wrapped model uses hetero Koopman advance.

        Returns
        -------
        object
            Function result.
        """
        return bool(getattr(self.model, "uses_hetero_koopman", False))

    def _hetero_operator(self) -> HeteroGraphKoopmanOperator:
        """Return the wrapped hetero operator.

        Returns
        -------
        HeteroGraphKoopmanOperator
            Relational Koopman module.

        Raises
        ------
        TypeError
            If the model is not hetero.
        """
        koopman = self.model.koopman
        if not isinstance(koopman, HeteroGraphKoopmanOperator):
            msg = "hetero conformal helpers require HeteroGraphKoopmanOperator"
            raise TypeError(msg)
        return koopman

    def _stacked_features(self, snapshot: SnapshotLike) -> Tensor:
        """Return homogeneous ``x`` or stacked hetero features ``(N, F)``.

        Parameters
        ----------
        snapshot : Data or HeteroData
            Forecast or target snapshot.

        Returns
        -------
        Tensor
            Feature matrix with shape ``(num_nodes, num_features)``.
        """
        if isinstance(snapshot, HeteroData):
            return _stack_hetero_features(
                snapshot,
                self._hetero_operator().node_types,
            )
        if snapshot.x is None:
            msg = "snapshots must define node features x"
            raise ValueError(msg)
        return snapshot.x

    def _pack_band(
        self,
        template: SnapshotLike,
        features: Tensor,
    ) -> SnapshotLike:
        """Pack a feature matrix onto a homogeneous or hetero template.

        Parameters
        ----------
        template : Data or HeteroData
            Topology template from the point forecast.
        features : Tensor
            Replacement features (stacked when hetero).

        Returns
        -------
        Data or HeteroData
            Band snapshot matching the template container type.
        """
        if isinstance(template, HeteroData):
            return hetero_snapshot_with_features(
                template,
                features,
                self._hetero_operator().node_types,
            )
        assert isinstance(template, Data)
        return snapshot_with_features(template, features)

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
        calibration_sequences: Sequence[CalibrationSequence],
        *,
        steps: int,
        alpha: float = 0.1,
        controls: Sequence[Sequence[Tensor] | None] | None = None,
        future_topologies: Sequence[Sequence[SnapshotLike] | None] | None = None,
    ) -> ConformalKoopmanUQ:
        """Estimate per-horizon conformal half-widths from held-out sequences.

        Parameters
        ----------
        calibration_sequences : sequence of trajectories
            Each trajectory must have length ``≥ steps + 1``. The first
            snapshot is the rollout origin; the next ``steps`` snapshots are
            targets. Homogeneous models require ``Data`` /
            :class:`~koopman_graph.data.GraphSnapshotSequence`; hetero models
            require ``HeteroData`` /
            :class:`~koopman_graph.data.HeteroGraphSnapshotSequence`. For
            ``score="node_wise"``, length must be at least ``ceil(1 / alpha)``.
        steps : int
            Forecast horizon used for calibration (and the maximum horizon
            for later :meth:`predict_interval` calls).
        alpha : float, optional
            Target miscoverage rate in ``(0, 1)``. Default ``0.1`` (nominal
            90% **marginal** coverage when calibration and test scores are
            exchangeable; not a joint coverage claim across nodes or types).
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
        TypeError
            If sequence container types disagree with the wrapped model
            (homo vs hetero).
        ValueError
            If inputs are invalid, sequences are too short, or (for
            ``node_wise``) the calibration set is smaller than
            ``ceil(1 / alpha)``.

        Notes
        -----
        See the module ``Coverage semantics`` section for exchangeability
        limits on graph time series and hetero stacked-score honesty.
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

        uses_hetero = self._uses_hetero
        for seq in calibration_sequences:
            seq_is_hetero = _is_hetero_sequence(seq)
            if uses_hetero and not seq_is_hetero:
                msg = (
                    "hetero ConformalKoopmanUQ calibration requires "
                    "HeteroData / HeteroGraphSnapshotSequence trajectories"
                )
                raise TypeError(msg)
            if not uses_hetero and seq_is_hetero:
                msg = (
                    "homogeneous ConformalKoopmanUQ cannot calibrate on "
                    "HeteroData / HeteroGraphSnapshotSequence trajectories"
                )
                raise TypeError(msg)

        resolved: list[ResolvedSequence]
        if uses_hetero:
            resolved = [
                resolve_hetero_sequence(seq)  # type: ignore[arg-type]
                for seq in calibration_sequences
            ]
        else:
            resolved = [
                resolve_sequence(seq)  # type: ignore[arg-type]
                for seq in calibration_sequences
            ]
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
        resolved: list[ResolvedSequence],
        *,
        steps: int,
        alpha: float,
        controls: Sequence[Sequence[Tensor] | None] | None,
        future_topologies: Sequence[Sequence[SnapshotLike] | None] | None,
    ) -> None:
        """Calibrate pooled scalar quantiles (``aggregate`` / ``per_node``).

        Parameters
        ----------
        resolved : list of snapshot sequences
            Resolved calibration trajectories (homo or hetero).
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
                pred_x = self._stacked_features(forecast[horizon])
                target_x = self._stacked_features(sequence[horizon + 1])
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
        resolved: list[ResolvedSequence],
        *,
        steps: int,
        alpha: float,
        controls: Sequence[Sequence[Tensor] | None] | None,
        future_topologies: Sequence[Sequence[SnapshotLike] | None] | None,
    ) -> None:
        """Calibrate per-node quantiles with optional score diffusion.

        Parameters
        ----------
        resolved : list of snapshot sequences
            Resolved calibration trajectories (fixed stacked node count).
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
        features0 = self._stacked_features(origin0)
        num_nodes = int(features0.shape[0])
        lam = 0.0 if self.neighbor_smoothing is None else self.neighbor_smoothing
        score_rows: list[list[Tensor]] = [[] for _ in range(steps)]
        quantiles = torch.zeros(steps, num_nodes, dtype=torch.float64)
        hetero_op = self._hetero_operator() if self._uses_hetero else None

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
            origin_features = self._stacked_features(origin)
            if int(origin_features.shape[0]) != num_nodes:
                msg = (
                    "node_wise calibration requires a fixed node count; "
                    f"expected {num_nodes} nodes"
                )
                raise ValueError(msg)
            edge_index: Tensor | None
            edge_weight: Tensor | None
            if lam > 0.0:
                if hetero_op is not None:
                    assert isinstance(origin, HeteroData)
                    edge_index = _union_relation_edge_index(
                        origin,
                        hetero_op.edge_types,
                        hetero_op.node_types,
                    )
                    edge_weight = None
                else:
                    assert isinstance(origin, Data)
                    if origin.edge_index is None:
                        msg = (
                            "neighbor_smoothing requires edge_index on "
                            "calibration origins"
                        )
                        raise ValueError(msg)
                    edge_index = origin.edge_index
                    edge_weight = getattr(origin, "edge_weight", None)
            else:
                edge_index = None
                edge_weight = None
            forecast = self.model.predict(
                origin,
                steps,
                controls=traj_controls,
                future_topologies=traj_future,
            )
            for horizon in range(steps):
                pred_x = self._stacked_features(forecast[horizon])
                target_x = self._stacked_features(sequence[horizon + 1])
                scores = _node_wise_scores(pred_x, target_x).to(dtype=torch.float64)
                if lam != 0.0:
                    assert edge_index is not None
                    scores = _diffuse_node_scores(
                        scores,
                        edge_index,
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
        initial_graph: Tensor | SnapshotLike,
        steps: int,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
        controls: Sequence[Tensor] | None = None,
        future_topologies: Sequence[SnapshotLike] | None = None,
        history: Sequence[SnapshotLike] | None = None,
        *,
        level: float = 0.9,
    ) -> PredictionInterval:
        """Return mean forecast with conformal lower/upper bands.

        Parameters
        ----------
        initial_graph : Tensor, Data, or HeteroData
            Rollout origin (same semantics as
            :meth:`~koopman_graph.model.GraphKoopmanModel.predict`). Hetero
            models require ``HeteroData`` origins.
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
            over stacked global rows when hetero and broadcasts over
            features. ``n_members`` is the calibration set size. Mean /
            lower / upper snapshots are ``HeteroData`` for hetero models.

        Raises
        ------
        TypeError
            If ``initial_graph`` disagrees with the wrapped model
            (homo vs hetero).
        RuntimeError
            If not calibrated.
        ValueError
            If ``steps`` / ``level`` disagree with calibration.

        Notes
        -----
        Bands inherit the marginal / exchangeability assumptions documented
        on the class and in the module ``Coverage semantics`` section; they
        are not joint intervals across nodes, types, or horizons.
        """
        if self._quantiles is None or self._alpha is None:
            msg = "ConformalKoopmanUQ is not calibrated; call calibrate() first"
            raise RuntimeError(msg)
        uses_hetero = self._uses_hetero
        if uses_hetero and not isinstance(initial_graph, HeteroData):
            msg = (
                "hetero ConformalKoopmanUQ.predict_interval requires a "
                "HeteroData origin"
            )
            raise TypeError(msg)
        if not uses_hetero and isinstance(initial_graph, HeteroData):
            msg = (
                "homogeneous ConformalKoopmanUQ.predict_interval cannot "
                "accept HeteroData origins"
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
        lower_snaps: list[SnapshotLike] = []
        upper_snaps: list[SnapshotLike] = []
        node_wise = self.score == "node_wise"
        for horizon, mean_snap in enumerate(mean_snaps):
            if isinstance(mean_snap, Data) and mean_snap.x is None:
                msg = "predicted snapshots must define node features x"
                raise ValueError(msg)
            mean_x = self._stacked_features(mean_snap)
            half = self._quantiles[horizon].to(
                device=mean_x.device,
                dtype=mean_x.dtype,
            )
            if node_wise:
                if half.ndim != 1 or half.shape[0] != mean_x.shape[0]:
                    msg = (
                        "node_wise quantiles must match the forecast node "
                        f"count; got {tuple(half.shape)} vs "
                        f"{mean_x.shape[0]} nodes"
                    )
                    raise ValueError(msg)
                half = half.unsqueeze(-1)
            lower_snaps.append(self._pack_band(mean_snap, mean_x - half))
            upper_snaps.append(self._pack_band(mean_snap, mean_x + half))

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
