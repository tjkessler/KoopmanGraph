"""Koopman operator capability package (discrete, continuous, and networked).

Capability layout
-----------------
``contract``
    Shared ``KoopmanOperatorContract``, parameterization types,
    ``StabilityCertificate``, and structural helpers.
``control``
    Additive / bilinear control helpers (``ControlMode``, bilinear factors).
``discrete``
    :class:`~koopman_graph.operators.discrete.KoopmanOperator`.
``continuous``
    :class:`~koopman_graph.operators.continuous.ContinuousKoopmanOperator`
    and Van Loan helpers.
``graph``
    :class:`~koopman_graph.operators.graph.GraphKoopmanOperator` (spatially
    coupled discrete advance).

Prefer ``from koopman_graph import KoopmanOperator, ContinuousKoopmanOperator,
GraphKoopmanOperator`` or ``from koopman_graph.operators import …``.
"""

from koopman_graph.operators.continuous import (
    VAN_LOAN_WRITEBACK_ATOL,
    ContinuousKoopmanOperator,
    GeneratorParameterization,
    matrix_log,
    van_loan_factors,
    van_loan_generator_from_discrete,
)
from koopman_graph.operators.contract import (
    DISSIPATIVE_MIN_EIGENVALUE,
    STABILITY_EPS_MARGIN,
    DynamicsMode,
    InitMode,
    KoopmanKind,
    KoopmanOperatorContract,
    Parameterization,
    StabilityCertificate,
    cayley_orthogonal,
    resolve_factory_stability_bound,
    strict_spectral_bound,
)
from koopman_graph.operators.control import ControlMode
from koopman_graph.operators.discrete import KoopmanOperator
from koopman_graph.operators.graph import GraphKoopmanOperator, GraphSparsity

__all__ = [
    "ContinuousKoopmanOperator",
    "ControlMode",
    "DISSIPATIVE_MIN_EIGENVALUE",
    "DynamicsMode",
    "GeneratorParameterization",
    "GraphKoopmanOperator",
    "GraphSparsity",
    "InitMode",
    "KoopmanKind",
    "KoopmanOperator",
    "KoopmanOperatorContract",
    "Parameterization",
    "STABILITY_EPS_MARGIN",
    "StabilityCertificate",
    "VAN_LOAN_WRITEBACK_ATOL",
    "cayley_orthogonal",
    "matrix_log",
    "resolve_factory_stability_bound",
    "strict_spectral_bound",
    "van_loan_factors",
    "van_loan_generator_from_discrete",
]
