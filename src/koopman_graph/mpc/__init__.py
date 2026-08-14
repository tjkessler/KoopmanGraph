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

Power-user module: import as ``koopman_graph.mpc``. Types are intentionally
omitted from root ``koopman_graph.__all__`` (see architecture docs).

Scope (0.14.0): discrete per-node ``KoopmanOperator`` with
``control_mode="additive"`` or iterated-QP ``"bilinear"``. Networked /
continuous operators are rejected. Output-constraint guarantees are local
(decoder Jacobian linearization). Optional ``constraint_tightening=``
accepts a calibrated :class:`~koopman_graph.uq.ConformalKoopmanUQ` and
shrinks output boxes by per-horizon half-widths (inputs unchanged;
margins are calibrated, not a formal closed-loop guarantee).

References
----------
Korda, M. and Mezić, I. (2018). Linear predictors for nonlinear dynamical
systems: Koopman operator meets model predictive control. *Automatica*, 93,
149–160.
"""

from koopman_graph.mpc.controller import KoopmanMPC

__all__ = [
    "KoopmanMPC",
]
