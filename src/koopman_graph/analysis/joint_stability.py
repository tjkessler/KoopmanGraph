"""Joint stability certificates (re-export from operators).

Implementation lives in :mod:`koopman_graph.operators.joint_stability` so
operator modules never import the analysis package. The analysis façade keeps
these names for the public analysis surface.
"""

from __future__ import annotations

from koopman_graph.operators.joint_stability import (
    JOINT_BOUND_KINDS,
    MAX_JOINT_LYAPUNOV_SIZE,
    MAX_JOINT_SCHUR_SIZE,
    JointBoundKind,
    JointStabilityCertificate,
    build_joint_stability_certificate,
    gershgorin_radius_bound,
    joint_certificate_from_assembled,
    lyapunov_joint_bound,
    require_joint_assembled_size,
    schur_radius_bound,
)

__all__ = [
    "JOINT_BOUND_KINDS",
    "MAX_JOINT_LYAPUNOV_SIZE",
    "MAX_JOINT_SCHUR_SIZE",
    "JointBoundKind",
    "JointStabilityCertificate",
    "build_joint_stability_certificate",
    "gershgorin_radius_bound",
    "joint_certificate_from_assembled",
    "lyapunov_joint_bound",
    "require_joint_assembled_size",
    "schur_radius_bound",
]
