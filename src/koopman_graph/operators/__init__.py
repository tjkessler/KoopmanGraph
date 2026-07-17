"""Koopman operator capability package (discrete and continuous).

Capability layout
-----------------
``contract``
    Shared ``KoopmanOperatorContract``, parameterization types,
    ``StabilityCertificate``, and structural helpers.
``discrete``
    :class:`~koopman_graph.operators.discrete.KoopmanOperator`.
``continuous``
    :class:`~koopman_graph.operators.continuous.ContinuousKoopmanOperator`
    and Van Loan helpers.

Prefer ``from koopman_graph import KoopmanOperator, ContinuousKoopmanOperator``
or ``from koopman_graph.operators import …``.
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
    KoopmanOperatorContract,
    Parameterization,
    StabilityCertificate,
    cayley_orthogonal,
    resolve_factory_stability_bound,
    strict_spectral_bound,
)
from koopman_graph.operators.discrete import KoopmanOperator

__all__ = [
    "ContinuousKoopmanOperator",
    "DISSIPATIVE_MIN_EIGENVALUE",
    "DynamicsMode",
    "GeneratorParameterization",
    "InitMode",
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
