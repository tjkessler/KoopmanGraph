"""Koopman operator capability package (discrete, continuous, and networked).

Capability layout
-----------------
``contract``
    Shared ``KoopmanOperatorContract``, parameterization types,
    ``StabilityCertificate``, and non-private structural helpers
    (``bounded_diagonal``, ``strict_diagonal_values``,
    ``safe_diagonal_inverse``, ``build_stability_certificate``).
``control``
    Shared additive / bilinear control helpers (``ControlMode``, bilinear
    factors, ``broadcast_control_term``, ``map_control_term``,
    ``write_dense_operator_parameters``,
    ``effective_bilinear_matrix`` / ``per_node_effective_bilinear_matrices``).
``discrete``
    :class:`~koopman_graph.operators.discrete.KoopmanOperator` thin
    string-mode orchestrator; re-exports discrete identity-init helpers.
``discrete_parameterizations``
    Dense / ODO / Schur / dissipative / Lyapunov assembly and reset helpers
    used by the discrete orchestrator (no parameterization class hierarchy).
``discrete_propagation``
    Controlled / uncontrolled advance, additive-control offset removal,
    bilinear inverse, and inverse-matrix execution helpers used by the
    discrete orchestrator (assembly / reset stay in
    ``discrete_parameterizations``).
``continuous``
    :class:`~koopman_graph.operators.continuous.ContinuousKoopmanOperator`
    thin string-mode orchestrator; re-exports Van Loan helpers.
``continuous_van_loan``
    Matrix-log / Van Loan factor construction (continuous-surface owned;
    prefer ``continuous`` or package re-exports).
``continuous_parameterizations``
    Dense / ODO / Schur / dissipative / Lyapunov assembly and reset helpers,
    plus continuous Hurwitz bound / certificate helpers used by the
    continuous orchestrator (no parameterization class hierarchy).
``continuous_propagation``
    Controlled / uncontrolled advance, Van Loan integral application,
    interval advance / inverse orchestration, and bilinear inverse
    execution helpers used by the continuous orchestrator (factor
    construction stays in ``continuous_van_loan``).
``auxiliary_spectral``
    Lusch-style state-dependent block-diagonal generator MLP plus
    advance / inverse / reset helpers for
    ``parameterization="auxiliary_spectral"``. Continuous retains Van Loan
    / structural / propagation peers and thin orchestration.
``graph``
    :class:`~koopman_graph.operators.graph.GraphKoopmanOperator` (spatially
    coupled discrete advance).
``graph_inverse``
    Block-diagonal / Jacobi approximate ``inverse_advance`` helpers for
    graph / hypergraph ``sparsity="block_diagonal"`` (no ``graph/`` subtree).
``hypergraph``
    :class:`~koopman_graph.operators.hypergraph.HypergraphKoopmanOperator`
    (hyperedge-coupled discrete advance; supports ``block_diagonal``).
``heterogeneous``
    :class:`~koopman_graph.operators.heterogeneous.HeteroGraphKoopmanOperator`
    (multiplex per-relation discrete advance; package export).
``global_local``
    :class:`~koopman_graph.operators.global_local.GlobalLocalKoopmanOperator`
    (discrete global backbone + low-rank local window correction).
``continuous_graph``
    :class:`~koopman_graph.operators.continuous_graph.ContinuousGraphKoopmanOperator`
    (continuous networked generator; ``koopman="graph"`` + continuous or alias
    ``koopman="continuous_graph"``).

Prefer ``from koopman_graph import KoopmanOperator, ContinuousKoopmanOperator,
GraphKoopmanOperator, HypergraphKoopmanOperator, GlobalLocalKoopmanOperator,
ContinuousGraphKoopmanOperator`` or ``from koopman_graph.operators import …``
(including :class:`~koopman_graph.operators.HeteroGraphKoopmanOperator`).
"""

from koopman_graph.operators.auxiliary_spectral import (
    DEFAULT_AUXILIARY_HIDDEN_DIMS,
    AuxiliarySpectralNetwork,
    assemble_block_diagonal_generator,
    normalize_auxiliary_hidden_dims,
    spectral_output_dim,
)
from koopman_graph.operators.continuous import (
    VAN_LOAN_WRITEBACK_ATOL,
    ContinuousKoopmanOperator,
    GeneratorParameterization,
    matrix_log,
    van_loan_factors,
    van_loan_generator_from_discrete,
)
from koopman_graph.operators.continuous_graph import (
    ContinuousGraphKoopmanOperator,
    ContinuousGraphSparsity,
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
    bounded_diagonal,
    build_stability_certificate,
    cayley_orthogonal,
    resolve_factory_stability_bound,
    safe_diagonal_inverse,
    strict_diagonal_values,
    strict_spectral_bound,
)
from koopman_graph.operators.control import ControlMode
from koopman_graph.operators.discrete import KoopmanOperator
from koopman_graph.operators.global_local import (
    DEFAULT_LOCAL_HIDDEN_DIMS,
    DEFAULT_LOCAL_RANK,
    DEFAULT_LOCAL_WINDOW,
    GlobalLocalKoopmanOperator,
    normalize_local_hidden_dims,
    pad_latent_window,
    stack_latent_window,
)
from koopman_graph.operators.graph import GraphKoopmanOperator
from koopman_graph.operators.graph_types import GraphAdjacency, GraphSparsity
from koopman_graph.operators.heterogeneous import HeteroGraphKoopmanOperator
from koopman_graph.operators.hypergraph import (
    HypergraphKoopmanOperator,
    HypergraphSparsity,
)

__all__ = [
    "AuxiliarySpectralNetwork",
    "ContinuousGraphKoopmanOperator",
    "ContinuousGraphSparsity",
    "ContinuousKoopmanOperator",
    "ControlMode",
    "DEFAULT_AUXILIARY_HIDDEN_DIMS",
    "DEFAULT_LOCAL_HIDDEN_DIMS",
    "DEFAULT_LOCAL_RANK",
    "DEFAULT_LOCAL_WINDOW",
    "DISSIPATIVE_MIN_EIGENVALUE",
    "DynamicsMode",
    "GeneratorParameterization",
    "GlobalLocalKoopmanOperator",
    "GraphAdjacency",
    "GraphKoopmanOperator",
    "GraphSparsity",
    "HeteroGraphKoopmanOperator",
    "HypergraphKoopmanOperator",
    "HypergraphSparsity",
    "InitMode",
    "KoopmanKind",
    "KoopmanOperator",
    "KoopmanOperatorContract",
    "Parameterization",
    "STABILITY_EPS_MARGIN",
    "StabilityCertificate",
    "VAN_LOAN_WRITEBACK_ATOL",
    "assemble_block_diagonal_generator",
    "bounded_diagonal",
    "build_stability_certificate",
    "cayley_orthogonal",
    "matrix_log",
    "normalize_auxiliary_hidden_dims",
    "normalize_local_hidden_dims",
    "pad_latent_window",
    "resolve_factory_stability_bound",
    "safe_diagonal_inverse",
    "spectral_output_dim",
    "stack_latent_window",
    "strict_diagonal_values",
    "strict_spectral_bound",
    "van_loan_factors",
    "van_loan_generator_from_discrete",
]
