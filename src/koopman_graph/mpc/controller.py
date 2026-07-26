"""Receding-horizon Koopman model-predictive control façade.

Composes a fitted :class:`~koopman_graph.model.GraphKoopmanModel` with an
additive discrete :class:`~koopman_graph.operators.KoopmanOperator`. The QP
plant state is the **mean** latent over nodes; controls are global
``(control_dim,)``. Output constraints use a local decoder linearization and
are therefore local (not a global closed-loop guarantee).

Optional ``constraint_tightening=ConformalKoopmanUQ`` shrinks output boxes by
calibrated per-horizon half-widths. Those margins inherit conformal
exchangeability caveats and the local linearization — they are calibrated
robustness margins, **not** a formal closed-loop guarantee.

References
----------
Korda, M. and Mezić, I. (2018). Linear predictors for nonlinear dynamical
systems: Koopman operator meets model predictive control. *Automatica*, 93,
149–160. https://doi.org/10.1016/j.automatica.2018.03.046
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch_geometric.data import Data

from koopman_graph.model import GraphKoopmanModel
from koopman_graph.mpc.qp import assemble_condensed_mpc, solve_dense_qp
from koopman_graph.operators import GraphKoopmanOperator, KoopmanOperator
from koopman_graph.operators.continuous import ContinuousKoopmanOperator
from koopman_graph.operators.continuous_graph import ContinuousGraphKoopmanOperator
from koopman_graph.operators.hypergraph import HypergraphKoopmanOperator
from koopman_graph.uq.common import snapshot_with_features

if TYPE_CHECKING:
    from koopman_graph.uq import ConformalKoopmanUQ


def _validate_mpc_model(model: GraphKoopmanModel) -> KoopmanOperator:
    """Reject unsupported operator kinds for 0.6.0 MPC.

    Parameters
    ----------

    model : GraphKoopmanModel
        See the function signature / summary for ``model``.

    Returns
    -------

    KoopmanOperator
        See summary line.

    Raises
    ------

    TypeError
        Raised when inputs are invalid.
    ValueError
        Raised when inputs are invalid."""
    koopman = model.koopman
    if isinstance(koopman, ContinuousKoopmanOperator | ContinuousGraphKoopmanOperator):
        msg = (
            "KoopmanMPC requires dynamics_mode='discrete' "
            "(continuous operators are not supported in 0.6.0)"
        )
        raise ValueError(msg)
    if isinstance(koopman, GraphKoopmanOperator | HypergraphKoopmanOperator):
        msg = (
            "KoopmanMPC does not support networked graph/hypergraph operators "
            "in 0.6.0; use a per-node KoopmanOperator"
        )
        raise ValueError(msg)
    if not isinstance(koopman, KoopmanOperator):
        msg = (
            "KoopmanMPC requires a discrete KoopmanOperator, "
            f"got {type(koopman).__name__}"
        )
        raise TypeError(msg)
    if koopman.control_dim <= 0:
        msg = "KoopmanMPC requires control_dim > 0"
        raise ValueError(msg)
    if koopman.control_mode != "additive":
        msg = (
            "KoopmanMPC supports control_mode='additive' only in 0.6.0; "
            f"got {koopman.control_mode!r} (bilinear iterated-QP is future work)"
        )
        raise ValueError(msg)
    if koopman.B is None:
        msg = "KoopmanMPC requires a control matrix B"
        raise ValueError(msg)
    return koopman


def _mean_latent(z: Tensor) -> Tensor:
    """Reduce ``(N, d)`` latents to a mean state ``(d,)``.

    Parameters
    ----------

    z : Tensor
        See the function signature / summary for ``z``.

    Returns
    -------

    Tensor
        See summary line.

    Raises
    ------

    ValueError
        Raised when inputs are invalid."""
    if z.ndim != 2:
        msg = f"latent must have shape (num_nodes, latent_dim), got {tuple(z.shape)}"
        raise ValueError(msg)
    return z.mean(dim=0)


def _expand_mean_latent(mean_z: Tensor, num_nodes: int) -> Tensor:
    """Broadcast a mean latent to all nodes.

    Parameters
    ----------

    mean_z : Tensor
        See the function signature / summary for ``mean_z``.
    num_nodes : int
        See the function signature / summary for ``num_nodes``.

    Returns
    -------

    Tensor
        See summary line."""
    return mean_z.unsqueeze(0).expand(num_nodes, -1).contiguous()


def _decoder_jacobian(
    model: GraphKoopmanModel,
    mean_z: Tensor,
    *,
    num_nodes: int,
    edge_index: Tensor,
    edge_weight: Tensor | None,
    eps: float = 1e-4,
) -> NDArray[np.float64]:
    """Finite-difference Jacobian of mean decoded features w.r.t. mean latent.

    Parameters
    ----------

    model : GraphKoopmanModel
        See the function signature / summary for ``model``.
    mean_z : Tensor
        See the function signature / summary for ``mean_z``.
    num_nodes : int
        See the function signature / summary for ``num_nodes``.
    edge_index : Tensor
        See the function signature / summary for ``edge_index``.
    edge_weight : Tensor | None
        See the function signature / summary for ``edge_weight``.
    eps : float
        See the function signature / summary for ``eps``.

    Returns
    -------

    ndarray
        Jacobian with shape ``(F, d)``."""
    latent_dim = int(mean_z.numel())
    z0 = _expand_mean_latent(mean_z, num_nodes)
    with torch.no_grad():
        y0 = model.decoder(z0, edge_index, edge_weight).mean(dim=0)
        out_dim = int(y0.numel())
        c_mat = torch.zeros(
            out_dim,
            latent_dim,
            dtype=mean_z.dtype,
            device=mean_z.device,
        )
        for index in range(latent_dim):
            delta = torch.zeros_like(mean_z)
            delta[index] = eps
            y_plus = model.decoder(
                _expand_mean_latent(mean_z + delta, num_nodes),
                edge_index,
                edge_weight,
            ).mean(dim=0)
            y_minus = model.decoder(
                _expand_mean_latent(mean_z - delta, num_nodes),
                edge_index,
                edge_weight,
            ).mean(dim=0)
            c_mat[:, index] = (y_plus - y_minus) / (2.0 * eps)
    return c_mat.detach().cpu().numpy().astype(np.float64)


def _as_numpy(value: Tensor | NDArray[np.floating]) -> NDArray[np.float64]:
    """Cast a Torch tensor or array to ``float64`` ndarray.

    Parameters
    ----------

    value : Tensor | NDArray[np.floating]
        See the function signature / summary for ``value``.

    Returns
    -------

    NDArray[np.float64]
        See summary line."""
    if isinstance(value, Tensor):
        return value.detach().cpu().numpy().astype(np.float64)
    return np.asarray(value, dtype=np.float64)


ReferenceLike = Tensor | NDArray[np.floating] | Sequence[Tensor | NDArray[np.floating]]


def _resolve_reference(
    reference: ReferenceLike,
    *,
    horizon: int,
    out_dim: int,
) -> NDArray[np.float64]:
    """Normalize reference to shape ``(horizon + 1, out_dim)``.

    Parameters
    ----------

    reference : ReferenceLike
        See the function signature / summary for ``reference``.
    horizon : int
        See the function signature / summary for ``horizon``.
    out_dim : int
        See the function signature / summary for ``out_dim``.

    Returns
    -------

    NDArray[np.float64]
        See summary line.

    Raises
    ------

    ValueError
        Raised when inputs are invalid."""
    if isinstance(reference, Sequence) and not isinstance(
        reference, Tensor | np.ndarray
    ):
        rows = [np.asarray(item, dtype=np.float64).reshape(-1) for item in reference]
        refs = np.stack(rows, axis=0)
    else:
        refs = np.asarray(reference, dtype=np.float64)
        if refs.ndim == 1:
            shape = (horizon + 1, refs.shape[0])
            refs = np.broadcast_to(refs.reshape(1, -1), shape).copy()
        elif refs.ndim == 2 and refs.shape[0] == 1:
            refs = np.broadcast_to(refs, (horizon + 1, refs.shape[1])).copy()
    if refs.shape != (horizon + 1, out_dim):
        msg = (
            f"reference must broadcast to shape {(horizon + 1, out_dim)}, "
            f"got {refs.shape}"
        )
        raise ValueError(msg)
    return refs


def _validate_constraint_tightening(
    tightening: object | None,
    *,
    model: GraphKoopmanModel,
    horizon: int,
    y_min: NDArray[np.float64] | None,
    y_max: NDArray[np.float64] | None,
) -> ConformalKoopmanUQ | None:
    """Validate optional conformal output-constraint tightening.

    Parameters
    ----------

    tightening : object | None
        See the function signature / summary for ``tightening``.
    model : GraphKoopmanModel
        See the function signature / summary for ``model``.
    horizon : int
        See the function signature / summary for ``horizon``.
    y_min : NDArray[np.float64] | None
        See the function signature / summary for ``y_min``.
    y_max : NDArray[np.float64] | None
        See the function signature / summary for ``y_max``.

    Returns
    -------

    ConformalKoopmanUQ | None
        See summary line.

    Raises
    ------

    RuntimeError
        Raised when inputs are invalid.
    TypeError
        Raised when inputs are invalid.
    ValueError
        Raised when inputs are invalid."""
    if tightening is None:
        return None
    # Import at call time so mpc ↔ uq stays a soft peer link for typing.
    from koopman_graph.uq import ConformalKoopmanUQ

    if not isinstance(tightening, ConformalKoopmanUQ):
        msg = (
            "constraint_tightening must be a ConformalKoopmanUQ instance, "
            f"got {type(tightening).__name__}"
        )
        raise TypeError(msg)
    if not tightening.is_calibrated:
        msg = "ConformalKoopmanUQ is not calibrated; call calibrate() first"
        raise RuntimeError(msg)
    if tightening.model is not model:
        msg = (
            "constraint_tightening.model must be the same GraphKoopmanModel "
            "instance passed to KoopmanMPC"
        )
        raise ValueError(msg)
    if tightening.calibrated_steps < horizon:
        msg = (
            f"constraint_tightening calibrated_steps="
            f"{tightening.calibrated_steps} is shorter than MPC horizon="
            f"{horizon}"
        )
        raise ValueError(msg)
    if y_min is None and y_max is None:
        msg = (
            "constraint_tightening requires y_min and/or y_max "
            "(output constraints to shrink)"
        )
        raise ValueError(msg)
    return tightening


def _conformal_stage_margins(
    tightening: ConformalKoopmanUQ,
    *,
    horizon: int,
) -> NDArray[np.float64]:
    """Map conformal quantiles to MPC stages ``h = 0..H``.

    Parameters
    ----------

    tightening : ConformalKoopmanUQ
        See the function signature / summary for ``tightening``.
    horizon : int
        See the function signature / summary for ``horizon``.

    Returns
    -------

    NDArray[np.float64]
        See summary line.

    Notes
    -----

    Stage ``0`` (current) uses margin ``0``; stages ``1..H`` use
    ``quantiles[h - 1]`` (k-step-ahead half-widths)."""
    quantiles = tightening.quantiles.detach().cpu().numpy().astype(np.float64)
    margins = np.zeros(horizon + 1, dtype=np.float64)
    margins[1:] = quantiles[:horizon]
    return margins


class KoopmanMPC:
    """Additive-control receding-horizon MPC on Koopman latent dynamics.

    Parameters
    ----------
    model : GraphKoopmanModel
        Fitted model with a discrete additive ``KoopmanOperator``.
    horizon : int
        Prediction horizon ``H ≥ 1``.
    Q, R : Tensor or ndarray
        PSD stage output and input cost matrices.
    Qf : Tensor or ndarray or None, optional
        Terminal output cost (defaults to ``Q``).
    u_min, u_max : Tensor or ndarray or None, optional
        Box bounds on each control ``u_h``.
    y_min, y_max : Tensor or ndarray or None, optional
        Optional box bounds on linearized decoded outputs (local guarantee).
    constraint_tightening : ConformalKoopmanUQ or None, optional
        Calibrated conformal wrapper on the **same** model. When set,
        per-horizon half-widths shrink output boxes
        (``y_min + m_h``, ``y_max - m_h``); input bounds stay exact.
        Stage ``h = 0`` uses margin ``0``; stages ``1..H`` use
        ``quantiles[h - 1]``. Scalar margins broadcast over features
        (matches aggregate ``L_∞`` scores; conservative for ``per_node``).
        Combined assumptions: conformal exchangeability is approximate for
        temporal graph data, and output maps are local decoder linearizations
        — margins are calibrated, not a formal closed-loop guarantee.

    Raises
    ------
    ValueError
        If the model/operator is unsupported, costs/bounds are invalid,
        or tightening arguments are inconsistent.
    TypeError
        If ``constraint_tightening`` is not a ``ConformalKoopmanUQ``.
    RuntimeError
        If ``constraint_tightening`` is not calibrated.
    """

    def __init__(
        self,
        model: GraphKoopmanModel,
        horizon: int,
        Q: Tensor | NDArray[np.floating],
        R: Tensor | NDArray[np.floating],
        *,
        Qf: Tensor | NDArray[np.floating] | None = None,
        u_min: Tensor | NDArray[np.floating] | None = None,
        u_max: Tensor | NDArray[np.floating] | None = None,
        y_min: Tensor | NDArray[np.floating] | None = None,
        y_max: Tensor | NDArray[np.floating] | None = None,
        constraint_tightening: ConformalKoopmanUQ | None = None,
    ) -> None:
        """Initialize the MPC controller from model, horizon, and costs.

        Parameters
        ----------
        model, horizon, Q, R, Qf, u_min, u_max, y_min, y_max,
        constraint_tightening
            See the class docstring.
        """
        if horizon < 1:
            msg = f"horizon must be >= 1, got {horizon}"
            raise ValueError(msg)

        self.model = model
        self.horizon = int(horizon)
        self._operator = _validate_mpc_model(model)
        self.Q = _as_numpy(Q)
        self.R = _as_numpy(R)
        self.Qf = self.Q.copy() if Qf is None else _as_numpy(Qf)
        self.u_min = None if u_min is None else _as_numpy(u_min).reshape(-1)
        self.u_max = None if u_max is None else _as_numpy(u_max).reshape(-1)
        self.y_min = None if y_min is None else _as_numpy(y_min).reshape(-1)
        self.y_max = None if y_max is None else _as_numpy(y_max).reshape(-1)
        self._tightening = _validate_constraint_tightening(
            constraint_tightening,
            model=model,
            horizon=self.horizon,
            y_min=self.y_min,
            y_max=self.y_max,
        )

    def _plant_matrices(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return column-convention ``(A, B_u)`` from the Koopman factors.

        Returns
        -------
        tuple[NDArray[np.float64], NDArray[np.float64]]
            See summary line."""
        # Row advance: z+ = z @ K.T + u @ B  →  x+ = K x + B.T u
        k = self._operator.K.detach().cpu().numpy().astype(np.float64)
        b_row = self._operator.B.detach().cpu().numpy().astype(np.float64)
        return k, b_row.T

    def solve(
        self,
        current_graph: Data,
        reference: ReferenceLike,
    ) -> Tensor:
        """Compute the first receding-horizon control action.

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
            If the QP is infeasible.
        """
        with torch.no_grad():
            z = self.model.encode(current_graph)
            mean_z = _mean_latent(z)
            edges = current_graph.edge_index
            weights = getattr(current_graph, "edge_weight", None)
            c_mat = _decoder_jacobian(
                self.model,
                mean_z,
                num_nodes=z.shape[0],
                edge_index=edges,
                edge_weight=weights,
            )
        a_mat, b_mat = self._plant_matrices()
        refs = _resolve_reference(
            reference, horizon=self.horizon, out_dim=c_mat.shape[0]
        )
        stage_margins = None
        if self._tightening is not None:
            stage_margins = _conformal_stage_margins(
                self._tightening, horizon=self.horizon
            )
        p_mat, q_vec, a_ineq, l_vec, u_vec = assemble_condensed_mpc(
            a_mat=a_mat,
            b_mat=b_mat,
            c_mat=c_mat,
            x0=mean_z.detach().cpu().numpy().astype(np.float64),
            references=refs,
            q_cost=self.Q,
            r_cost=self.R,
            qf_cost=self.Qf,
            u_min=self.u_min,
            u_max=self.u_max,
            y_min=self.y_min,
            y_max=self.y_max,
            stage_margins=stage_margins,
        )
        u_star = solve_dense_qp(p_mat, q_vec, a_ineq, l_vec, u_vec)
        control_dim = self._operator.control_dim
        first = u_star[:control_dim].copy()
        if self.u_min is not None:
            first = np.maximum(first, self.u_min)
        if self.u_max is not None:
            first = np.minimum(first, self.u_max)
        return torch.tensor(first, dtype=torch.float32)

    def rollout(
        self,
        initial_graph: Data,
        reference: ReferenceLike,
        steps: int,
    ) -> list[Data]:
        """Closed-loop receding-horizon simulation on the model plant.

        Parameters
        ----------
        initial_graph : Data
            Episode origin.
        reference : Tensor or sequence
            Tracking reference (constant or trajectory).
        steps : int
            Number of closed-loop steps.

        Returns
        -------
        list of Data
            Decoded snapshots after each applied control.
        """
        if steps < 1:
            msg = f"steps must be >= 1, got {steps}"
            raise ValueError(msg)
        current = initial_graph
        outputs: list[Data] = []
        for _ in range(steps):
            action = self.solve(current, reference)
            with torch.no_grad():
                z = self.model.encode(current)
                z_next = self.model.koopman.advance(
                    z,
                    control=action,
                    edge_index=current.edge_index,
                    edge_weight=getattr(current, "edge_weight", None),
                )
                decoded = self.model.decoder(
                    z_next,
                    current.edge_index,
                    getattr(current, "edge_weight", None),
                )
            current = snapshot_with_features(current, decoded)
            outputs.append(current)
        return outputs
