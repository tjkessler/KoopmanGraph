"""Residual-tube tightening for additive discrete Koopman-MPC.

:class:`TubeKoopmanMPC` shrinks nominal output boxes by conformal
quantiles or ensemble residual radii, then solves the same condensed
QP as :class:`~koopman_graph.mpc.KoopmanMPC`. Closed-loop
:meth:`~TubeKoopmanMPC.evaluate` reports constraint-violation rate,
feasibility rate, and quadratic stage cost against the **nominal**
boxes.

This is a residual-tube helper after Zhang, Pan, Scattolini, Yu, and
Xu, *Automatica* 137:110114 (2022), doi:10.1016/j.automatica.2021.110114
(``Zhang2022TubeMPC``). Local decoder linearization remains. The report
is not a recursive-feasibility or Lyapunov closed-loop certificate, and
the helper is not a chance-constraint solver. Zhang et al. prove
closed-loop robustness for an r-KMPC scheme with an offline nonlinear
ancillary law; this MVP does not implement that ancillary controller
or those proofs.

Additive discrete per-node operators only. Bilinear, networked, and
continuous plants are refused. Named chance-constraint language
requires :class:`~koopman_graph.uq.JointCoverageSpec` with the shipped
``per_node_marginal`` / ``block="none"`` pair.

This module must not import :mod:`koopman_graph.identification`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch_geometric.data import Data

from koopman_graph.model import GraphKoopmanModel
from koopman_graph.mpc.controller import (
    KoopmanMPC,
    ReferenceLike,
    _as_numpy,
    _conformal_stage_margins,
    _decoder_jacobian,
    _mean_latent,
    _resolve_reference,
    _validate_constraint_tightening,
    _validate_mpc_model,
)
from koopman_graph.mpc.qp import assemble_condensed_mpc, solve_dense_qp
from koopman_graph.uq.common import PredictionInterval, snapshot_with_features
from koopman_graph.uq.coverage import JointCoverageSpec, require_shipped_coverage

PlantFn = Callable[[Data, Tensor], Data]

__all__ = [
    "TubeKoopmanMPC",
    "TubeMPCReport",
    "ensemble_residual_radii",
]


def ensemble_residual_radii(interval: PredictionInterval) -> Tensor:
    """Pool per-step ensemble half-widths from a prediction interval.

    Parameters
    ----------
    interval : PredictionInterval
        Homogeneous mean / lower / upper bands. Each step uses
        ``(upper - lower) / 2``; the scalar radius is the maximum over
        nodes and features (same Minkowski radius as an aggregate
        conformal quantile).

    Returns
    -------
    Tensor
        Non-negative radii with shape ``(H,)`` for ``H`` forecast steps.

    Raises
    ------
    TypeError
        If a band lacks homogeneous ``x`` node features.
    ValueError
        If the interval is empty or a half-width is negative.
    """
    if len(interval.lower) < 1 or len(interval.upper) < 1:
        msg = "PredictionInterval must contain at least one forecast step"
        raise ValueError(msg)
    if len(interval.lower) != len(interval.upper):
        msg = (
            "PredictionInterval lower/upper lengths differ: "
            f"{len(interval.lower)} vs {len(interval.upper)}"
        )
        raise ValueError(msg)
    radii: list[Tensor] = []
    for lower, upper in zip(interval.lower, interval.upper, strict=True):
        lower_x = getattr(lower, "x", None)
        upper_x = getattr(upper, "x", None)
        if lower_x is None or upper_x is None:
            msg = (
                "ensemble residual radii require homogeneous node features "
                "on interval.lower / interval.upper"
            )
            raise TypeError(msg)
        half = 0.5 * (upper_x - lower_x)
        if bool(torch.any(half < -1e-12)):
            msg = "ensemble interval half-widths must be non-negative"
            raise ValueError(msg)
        radii.append(half.reshape(-1).max().clamp(min=0.0))
    return torch.stack(radii)


def _horizon_radii(
    residual_source: object,
    *,
    horizon: int,
) -> NDArray[np.float64]:
    """Coerce ensemble radii to a length-``H`` non-negative vector.

    Parameters
    ----------
    residual_source : Tensor or ndarray
        Scalar, ``(H,)``, or ``(H, F)`` half-widths. A trailing feature
        axis is reduced by a max so the tube is a scalar Minkowski
        radius per stage.
    horizon : int
        MPC horizon ``H``.

    Returns
    -------
    ndarray
        Radii with shape ``(H,)``.

    Raises
    ------
    ValueError
        If the shape is too short, has the wrong rank, or is negative.
    """
    arr = np.asarray(residual_source, dtype=np.float64)
    if arr.ndim == 0:
        radii = np.full((horizon,), float(arr), dtype=np.float64)
    elif arr.ndim == 1:
        if arr.shape[0] < horizon:
            msg = (
                f"ensemble residual radii must cover horizon={horizon}, "
                f"got shape {arr.shape}"
            )
            raise ValueError(msg)
        radii = arr[:horizon].astype(np.float64, copy=True)
    elif arr.ndim == 2:
        if arr.shape[0] < horizon:
            msg = (
                f"ensemble residual radii must cover horizon={horizon}, "
                f"got shape {arr.shape}"
            )
            raise ValueError(msg)
        radii = arr[:horizon].max(axis=1).astype(np.float64, copy=True)
    else:
        msg = (
            "ensemble residual radii must be a scalar, shape (H,), or "
            f"(H, F); got shape {arr.shape}"
        )
        raise ValueError(msg)
    if np.any(radii < -1e-12):
        msg = "ensemble residual radii must be non-negative"
        raise ValueError(msg)
    return np.maximum(radii, 0.0)


def _stage_margins_from_radii(
    radii: NDArray[np.floating],
    *,
    horizon: int,
) -> NDArray[np.float64]:
    """Map per-horizon radii to MPC stages ``h = 0..H``.

    Parameters
    ----------
    radii : ndarray
        Non-negative half-widths with shape ``(H,)``.
    horizon : int
        MPC horizon ``H``.

    Returns
    -------
    ndarray
        Margins with shape ``(H + 1,)``. Stage ``0`` is ``0``; stages
        ``1..H`` use ``radii[h - 1]``.
    """
    vec = np.asarray(radii, dtype=np.float64).reshape(-1)
    margins = np.zeros(horizon + 1, dtype=np.float64)
    margins[1:] = vec[:horizon]
    return margins


def _resolve_stage_margins(
    residual_source: object,
    *,
    model: GraphKoopmanModel,
    horizon: int,
    y_min: NDArray[np.float64],
    y_max: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Build per-stage output margins from conformal or ensemble residuals.

    Parameters
    ----------
    residual_source : ConformalKoopmanUQ or PredictionInterval or array
        Calibrated conformal wrapper on ``model``, a homogeneous
        prediction interval, or explicit radii.
    model : GraphKoopmanModel
        Plant model passed to :class:`TubeKoopmanMPC`.
    horizon : int
        MPC horizon ``H``.
    y_min, y_max : ndarray
        Nominal output boxes (required so margins have something to
        shrink).

    Returns
    -------
    ndarray
        Non-negative margins with shape ``(H + 1,)``.

    Raises
    ------
    TypeError
        If ``residual_source`` is not a supported residual object.
    """
    # Import at call time so mpc ↔ conformal stays a soft peer link.
    from koopman_graph.uq import ConformalKoopmanUQ

    if isinstance(residual_source, ConformalKoopmanUQ):
        tightening = _validate_constraint_tightening(
            residual_source,
            model=model,
            horizon=horizon,
            y_min=y_min,
            y_max=y_max,
        )
        assert tightening is not None
        return _conformal_stage_margins(tightening, horizon=horizon)
    if isinstance(residual_source, PredictionInterval):
        radii = ensemble_residual_radii(residual_source)
        return _stage_margins_from_radii(
            _horizon_radii(radii, horizon=horizon),
            horizon=horizon,
        )
    if isinstance(residual_source, Tensor | np.ndarray):
        return _stage_margins_from_radii(
            _horizon_radii(residual_source, horizon=horizon),
            horizon=horizon,
        )
    msg = (
        "residual_source must be a calibrated ConformalKoopmanUQ, "
        "a PredictionInterval, or non-negative residual radii, "
        f"got {type(residual_source).__name__}"
    )
    raise TypeError(msg)


def _mean_decoded_numpy(model: GraphKoopmanModel, graph: Data) -> NDArray[np.float64]:
    """Decode a snapshot and reduce node features to a mean output.

    Parameters
    ----------
    model : GraphKoopmanModel
        Fitted model whose decoder maps latents to features.
    graph : Data
        Current snapshot.

    Returns
    -------
    ndarray
        Mean decoded features with shape ``(F,)``.
    """
    with torch.no_grad():
        z = model.encode(graph)
        decoded = model.decoder(
            z,
            graph.edge_index,
            getattr(graph, "edge_weight", None),
        )
        return decoded.mean(dim=0).detach().cpu().numpy().astype(np.float64)


def _outside_nominal_box(
    y: NDArray[np.floating],
    y_min: NDArray[np.floating],
    y_max: NDArray[np.floating],
    *,
    atol: float = 1e-6,
) -> bool:
    """Return whether any coordinate leaves the nominal output box.

    Parameters
    ----------
    y : ndarray
        Mean decoded output with shape ``(F,)``.
    y_min, y_max : ndarray
        Nominal box bounds with shape ``(F,)``.
    atol : float, optional
        Absolute slack for floating-point comparisons.

    Returns
    -------
    bool
        ``True`` when any coordinate is outside ``[y_min, y_max]``.
    """
    vec = np.asarray(y, dtype=np.float64).reshape(-1)
    lo = np.asarray(y_min, dtype=np.float64).reshape(-1)
    hi = np.asarray(y_max, dtype=np.float64).reshape(-1)
    return bool(np.any(vec < lo - atol) or np.any(vec > hi + atol))


def _quadratic_stage_cost(
    y: NDArray[np.floating],
    reference: NDArray[np.floating],
    u: NDArray[np.floating],
    q_cost: NDArray[np.floating],
    r_cost: NDArray[np.floating],
) -> float:
    """Scalar tracking-plus-input cost for one applied stage.

    Parameters
    ----------
    y : ndarray
        Mean decoded output with shape ``(F,)``.
    reference : ndarray
        One-step output target with shape ``(F,)``.
    u : ndarray
        Applied control with shape ``(C,)``.
    q_cost, r_cost : ndarray
        PSD output and input weights.

    Returns
    -------
    float
        ``(y - r)^T Q (y - r) + u^T R u``.
    """
    err = np.asarray(y, dtype=np.float64).reshape(-1) - np.asarray(
        reference, dtype=np.float64
    ).reshape(-1)
    ctrl = np.asarray(u, dtype=np.float64).reshape(-1)
    q = np.asarray(q_cost, dtype=np.float64)
    r = np.asarray(r_cost, dtype=np.float64)
    return float(err @ q @ err + ctrl @ r @ ctrl)


def _fallback_control(controller: KoopmanMPC) -> Tensor:
    """Zero input clipped to the declared control box.

    Parameters
    ----------
    controller : KoopmanMPC
        Inner additive plant used for bound shapes.

    Returns
    -------
    Tensor
        Control with shape ``(control_dim,)``.
    """
    control_dim = controller._operator.control_dim
    first = np.zeros(control_dim, dtype=np.float64)
    if controller.u_min is not None:
        first = np.maximum(first, controller.u_min)
    if controller.u_max is not None:
        first = np.minimum(first, controller.u_max)
    return torch.tensor(first, dtype=torch.float32)


def _default_model_plant(
    model: GraphKoopmanModel,
    graph: Data,
    control: Tensor,
) -> Data:
    """Advance the fitted discrete plant by one controlled step.

    Parameters
    ----------
    model : GraphKoopmanModel
        Discrete controlled model.
    graph : Data
        Current snapshot.
    control : Tensor
        Global control with shape ``(control_dim,)``.

    Returns
    -------
    Data
        Decoded snapshot after ``koopman.advance``.
    """
    with torch.no_grad():
        z = model.encode(graph)
        z_next = model.koopman.advance(
            z,
            control=control,
            edge_index=graph.edge_index,
            edge_weight=getattr(graph, "edge_weight", None),
        )
        decoded = model.decoder(
            z_next,
            graph.edge_index,
            getattr(graph, "edge_weight", None),
        )
    return snapshot_with_features(graph, decoded)


@dataclass(frozen=True)
class TubeMPCReport:
    """Closed-loop residual-tube evaluation against nominal boxes.

    Attributes
    ----------
    violation_rate : float
        Fraction of steps whose mean decoded output leaves the
        **nominal** ``y_min`` / ``y_max`` box (any coordinate). This is
        a constraint-violation rate, not tracking MSE.
    feasibility_rate : float
        Fraction of steps whose tightened QP returned a solution.
    cost : float
        Mean quadratic stage cost on feasible steps. ``nan`` when no
        step was feasible.
    n_steps : int
        Closed-loop length.
    n_feasible : int
        Number of successful QP solves.
    n_violations : int
        Number of nominal-box violations.
    coverage : JointCoverageSpec
        Named estimand attached to the residual source. Shipped MVP is
        ``per_node_marginal`` / ``block="none"``.
    """

    violation_rate: float
    feasibility_rate: float
    cost: float
    n_steps: int
    n_feasible: int
    n_violations: int
    coverage: JointCoverageSpec

    def __post_init__(self) -> None:
        """Validate rate / count consistency.

        Raises
        ------
        ValueError
            If counts or rates are inconsistent.
        """
        if self.n_steps < 1:
            msg = f"n_steps must be >= 1, got {self.n_steps}"
            raise ValueError(msg)
        if self.n_feasible < 0 or self.n_feasible > self.n_steps:
            msg = f"n_feasible must lie in [0, {self.n_steps}], got {self.n_feasible}"
            raise ValueError(msg)
        if self.n_violations < 0 or self.n_violations > self.n_steps:
            msg = (
                f"n_violations must lie in [0, {self.n_steps}], got {self.n_violations}"
            )
            raise ValueError(msg)
        expected_feas = self.n_feasible / self.n_steps
        expected_viol = self.n_violations / self.n_steps
        if abs(self.feasibility_rate - expected_feas) > 1e-12:
            msg = "feasibility_rate must equal n_feasible / n_steps"
            raise ValueError(msg)
        if abs(self.violation_rate - expected_viol) > 1e-12:
            msg = "violation_rate must equal n_violations / n_steps"
            raise ValueError(msg)


class TubeKoopmanMPC:
    """Tube-tightened additive discrete Koopman-MPC.

    Residual radii come from a calibrated
    :class:`~koopman_graph.uq.ConformalKoopmanUQ`, a homogeneous
    :class:`~koopman_graph.uq.PredictionInterval`, or an explicit
    ``(H,)`` / ``(H, F)`` tensor. The inner QP uses the eroded boxes;
    :meth:`evaluate` scores the **nominal** boxes.

    This is not Zhang et al. r-KMPC (offline nonlinear feedback plus a
    closed-loop robustness proof; ``Zhang2022TubeMPC``). Local
    linearization honesty is unchanged. Simultaneous / event coverage
    targets raise.

    Parameters
    ----------
    model : GraphKoopmanModel
        Fitted model with a discrete per-node additive
        ``KoopmanOperator``.
    horizon : int
        Prediction horizon ``H >= 1``.
    Q, R : Tensor or ndarray
        PSD stage output and input cost matrices.
    residual_source : ConformalKoopmanUQ or PredictionInterval or array
        Calibrated conformal wrapper, ensemble interval, or explicit
        non-negative residual radii.
    y_min, y_max : Tensor or ndarray
        Nominal output boxes. Both are required so violations have a
        declared set and the tube has a set to erode.
    Qf : Tensor or ndarray or None, optional
        Terminal output cost (defaults to ``Q``).
    u_min, u_max : Tensor or ndarray or None, optional
        Box bounds on each control ``u_h``.
    coverage : JointCoverageSpec or None, optional
        Named chance-constraint estimand. Default is
        ``per_node_marginal`` / ``block="none"``. Other targets raise.

    References
    ----------
    Zhang, X., Pan, W., Scattolini, R., Yu, S., and Xu, X. (2022).
    Robust tube-based model predictive control with Koopman operators.
    *Automatica*, 137, 110114.
    https://doi.org/10.1016/j.automatica.2021.110114
    (``Zhang2022TubeMPC``; lineage for residual tubes — not a
    certificate that this helper inherits their closed-loop proofs.)
    """

    def __init__(
        self,
        model: GraphKoopmanModel,
        horizon: int,
        Q: Tensor | NDArray[np.floating],
        R: Tensor | NDArray[np.floating],
        *,
        residual_source: object,
        y_min: Tensor | NDArray[np.floating],
        y_max: Tensor | NDArray[np.floating],
        Qf: Tensor | NDArray[np.floating] | None = None,
        u_min: Tensor | NDArray[np.floating] | None = None,
        u_max: Tensor | NDArray[np.floating] | None = None,
        coverage: JointCoverageSpec | None = None,
    ) -> None:
        """Initialize residual-tube MPC from an additive discrete plant.

        Parameters
        ----------
        model, horizon, Q, R, residual_source, y_min, y_max, Qf, u_min,
        u_max, coverage
            See the class docstring.
        """
        operator = _validate_mpc_model(model)
        if operator.control_mode != "additive":
            msg = (
                "TubeKoopmanMPC requires control_mode='additive' "
                f"(got {operator.control_mode!r}); bilinear sequential "
                "linearization is refused"
            )
            raise ValueError(msg)
        self.coverage = require_shipped_coverage(
            JointCoverageSpec() if coverage is None else coverage
        )
        self._controller = KoopmanMPC(
            model,
            horizon,
            Q,
            R,
            Qf=Qf,
            u_min=u_min,
            u_max=u_max,
            y_min=y_min,
            y_max=y_max,
        )
        if self._controller.y_min is None or self._controller.y_max is None:
            msg = "TubeKoopmanMPC requires y_min and y_max"
            raise ValueError(msg)
        self._stage_margins = _resolve_stage_margins(
            residual_source,
            model=model,
            horizon=self._controller.horizon,
            y_min=self._controller.y_min,
            y_max=self._controller.y_max,
        )

    @property
    def model(self) -> GraphKoopmanModel:
        """Fitted plant composed by the inner additive controller.

        Returns
        -------
        GraphKoopmanModel
            The model passed at construction.
        """
        return self._controller.model

    @property
    def horizon(self) -> int:
        """Prediction horizon ``H``.

        Returns
        -------
        int
            Horizon used by the inner QP.
        """
        return self._controller.horizon

    def solve(
        self,
        current_graph: Data,
        reference: ReferenceLike,
    ) -> Tensor:
        """Compute the first receding-horizon control on the tightened boxes.

        Parameters
        ----------
        current_graph : Data
            Current snapshot (features + topology).
        reference : Tensor or sequence
            Constant ``(F,)`` reference or trajectory ``(H+1, F)``.

        Returns
        -------
        Tensor
            Control action with shape ``(control_dim,)``.

        Raises
        ------
        ImportError
            If OSQP (``[mpc]``) is not installed.
        RuntimeError
            If the tightened QP is infeasible.
        """
        controller = self._controller
        with torch.no_grad():
            z = controller.model.encode(current_graph)
            mean_z = _mean_latent(z)
            edges = current_graph.edge_index
            weights = getattr(current_graph, "edge_weight", None)
            c_mat = _decoder_jacobian(
                controller.model,
                mean_z,
                num_nodes=z.shape[0],
                edge_index=edges,
                edge_weight=weights,
            )
        a_mat, b_mat = controller._plant_matrices()
        refs = _resolve_reference(
            reference, horizon=controller.horizon, out_dim=c_mat.shape[0]
        )
        p_mat, q_vec, a_ineq, l_vec, u_vec = assemble_condensed_mpc(
            a_mat=a_mat,
            b_mat=b_mat,
            c_mat=c_mat,
            x0=mean_z.detach().cpu().numpy().astype(np.float64),
            references=refs,
            q_cost=controller.Q,
            r_cost=controller.R,
            qf_cost=controller.Qf,
            u_min=controller.u_min,
            u_max=controller.u_max,
            y_min=controller.y_min,
            y_max=controller.y_max,
            stage_margins=self._stage_margins,
        )
        u_star = solve_dense_qp(p_mat, q_vec, a_ineq, l_vec, u_vec)
        first = u_star[: controller._operator.control_dim].copy()
        if controller.u_min is not None:
            first = np.maximum(first, controller.u_min)
        if controller.u_max is not None:
            first = np.minimum(first, controller.u_max)
        return torch.tensor(first, dtype=torch.float32)

    def rollout(
        self,
        initial_graph: Data,
        reference: ReferenceLike,
        steps: int,
        *,
        plant: PlantFn | None = None,
    ) -> list[Data]:
        """Closed-loop receding-horizon simulation on a discrete plant.

        Infeasible QP solves apply a clipped-zero fallback control so
        the episode continues. Prefer :meth:`evaluate` when the
        constraint-violation rate is the quantity of interest.

        Parameters
        ----------
        initial_graph : Data
            Episode origin.
        reference : Tensor or sequence
            Tracking reference (constant or trajectory).
        steps : int
            Number of closed-loop steps.
        plant : callable or None, optional
            ``plant(graph, u) -> next_graph``. Default advances the
            fitted model.

        Returns
        -------
        list of Data
            Decoded snapshots after each applied control.

        Raises
        ------
        ValueError
            If ``steps < 1``.
        """
        report_plant = plant
        snapshots, _report = self._closed_loop(
            initial_graph,
            reference,
            steps,
            plant=report_plant,
        )
        return snapshots

    def evaluate(
        self,
        initial_graph: Data,
        reference: ReferenceLike,
        steps: int,
        *,
        plant: PlantFn | None = None,
    ) -> TubeMPCReport:
        """Closed-loop violation, feasibility, and cost on nominal boxes.

        Parameters
        ----------
        initial_graph : Data
            Episode origin.
        reference : Tensor or sequence
            Tracking reference (constant or trajectory).
        steps : int
            Number of closed-loop steps.
        plant : callable or None, optional
            ``plant(graph, u) -> next_graph``. Default advances the
            fitted model. Use a mismatched plant when the residual
            tube is meant to absorb a declared disturbance.

        Returns
        -------
        TubeMPCReport
            Constraint-violation rate, feasibility rate, and mean
            feasible-stage cost. Not tracking MSE alone, and not a
            recursive-feasibility certificate.

        Raises
        ------
        ValueError
            If ``steps < 1``.
        """
        _snapshots, report = self._closed_loop(
            initial_graph,
            reference,
            steps,
            plant=plant,
        )
        return report

    def _closed_loop(
        self,
        initial_graph: Data,
        reference: ReferenceLike,
        steps: int,
        *,
        plant: PlantFn | None,
    ) -> tuple[list[Data], TubeMPCReport]:
        """Run one closed-loop episode and accumulate the tube report.

        Parameters
        ----------
        initial_graph : Data
            Episode origin.
        reference : Tensor or sequence
            Tracking reference (constant or trajectory).
        steps : int
            Number of closed-loop steps.
        plant : callable or None
            Optional plant override.

        Returns
        -------
        tuple
            Decoded snapshots and the :class:`TubeMPCReport`.

        Raises
        ------
        ValueError
            If ``steps < 1``.
        """
        if steps < 1:
            msg = f"steps must be >= 1, got {steps}"
            raise ValueError(msg)
        controller = self._controller
        assert controller.y_min is not None
        assert controller.y_max is not None
        out_dim = int(controller.y_min.shape[0])
        refs = _resolve_reference(
            reference, horizon=controller.horizon, out_dim=out_dim
        )
        current = initial_graph
        outputs: list[Data] = []
        n_feasible = 0
        n_violations = 0
        cost_sum = 0.0
        for _ in range(steps):
            try:
                action = self.solve(current, reference)
            except RuntimeError:
                action = _fallback_control(controller)
                feasible = False
            else:
                feasible = True
                n_feasible += 1
            if plant is None:
                current = _default_model_plant(controller.model, current, action)
            else:
                current = plant(current, action)
            outputs.append(current)
            y = _mean_decoded_numpy(controller.model, current)
            if _outside_nominal_box(y, controller.y_min, controller.y_max):
                n_violations += 1
            if feasible:
                cost_sum += _quadratic_stage_cost(
                    y,
                    refs[1],
                    _as_numpy(action),
                    controller.Q,
                    controller.R,
                )
        mean_cost = float("nan") if n_feasible == 0 else cost_sum / float(n_feasible)
        report = TubeMPCReport(
            violation_rate=n_violations / steps,
            feasibility_rate=n_feasible / steps,
            cost=mean_cost,
            n_steps=steps,
            n_feasible=n_feasible,
            n_violations=n_violations,
            coverage=self.coverage,
        )
        return outputs, report
