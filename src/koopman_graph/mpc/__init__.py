"""Koopman model-predictive control (receding-horizon QP).

Capability layout
-----------------
``controller``
    :class:`~koopman_graph.mpc.KoopmanMPC` façade — composes a fitted
    :class:`~koopman_graph.model.GraphKoopmanModel` with additive discrete
    latent dynamics; ``solve`` / ``rollout`` for closed-loop control.
``qp``
    Condensed QP assembly and OSQP solve helpers (call-site import of the
    ``[mpc]`` extra).
``tube``
    :class:`~koopman_graph.mpc.TubeKoopmanMPC` residual-tube tightening
    from conformal or ensemble radii on **additive discrete** plants;
    :class:`~koopman_graph.mpc.TubeMPCReport` records violation rate,
    feasibility rate, and cost. Not a chance-constraint solver and not
    a recursive-feasibility / Lyapunov certificate
    (``Zhang2022TubeMPC``).

Power-user module: import as ``koopman_graph.mpc``. Types are intentionally
omitted from root ``koopman_graph.__all__`` (see architecture docs).

Scope (0.14.0): discrete per-node ``KoopmanOperator`` with
``control_mode="additive"`` or iterated-QP ``"bilinear"``. Networked /
continuous operators are rejected. Output-constraint guarantees are local
(decoder Jacobian linearization). Optional ``constraint_tightening=``
accepts a calibrated :class:`~koopman_graph.uq.ConformalKoopmanUQ` and
shrinks output boxes by per-horizon half-widths (inputs unchanged;
margins are calibrated, not a formal closed-loop guarantee).
:class:`~koopman_graph.mpc.TubeKoopmanMPC` is additive-only and reports
constraint-violation rate rather than tracking MSE alone.

References
----------
Korda, M. and Mezić, I. (2018). Linear predictors for nonlinear dynamical
systems: Koopman operator meets model predictive control. *Automatica*, 93,
149–160.
Zhang, X., Pan, W., Scattolini, R., Yu, S., and Xu, X. (2022). Robust
tube-based model predictive control with Koopman operators.
*Automatica*, 137, 110114.
"""

from koopman_graph.mpc.controller import KoopmanMPC
from koopman_graph.mpc.tube import (
    TubeKoopmanMPC,
    TubeMPCReport,
    ensemble_residual_radii,
)

__all__ = [
    "KoopmanMPC",
    "TubeKoopmanMPC",
    "TubeMPCReport",
    "ensemble_residual_radii",
]
