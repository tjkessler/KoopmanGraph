"""Optional interop helpers for external MSM / MD toolchains.

Capability layout
-----------------
``deeptime``
    Lazy ``[msm]`` boundary for deeptime trajectory-feature conversion
    (:func:`~koopman_graph.interop.trajectory_features_to_deeptime`,
    :func:`~koopman_graph.interop.trajectory_features_from_deeptime`).
    Importing this package does **not** import deeptime; call
    :func:`~koopman_graph.interop.require_deeptime` at use sites.

These modules are power-user surfaces (not root ``__all__``). They are
diagnostic / teaching bridges — not a PyEMMA replacement.

**Layer rule:** other ``koopman_graph`` modules must not import ``interop``
(keep the dependency graph acyclic). Callers are notebooks, tests, and
application code.
"""

from koopman_graph.interop.deeptime import (
    require_deeptime,
    trajectory_features_from_deeptime,
    trajectory_features_to_deeptime,
)

__all__ = [
    "require_deeptime",
    "trajectory_features_from_deeptime",
    "trajectory_features_to_deeptime",
]
