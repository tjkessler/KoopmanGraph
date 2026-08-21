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
    coupled discrete advance; optional monomial ``filter_degree``).
``polynomial_graph``
    Dense monomial Kronecker assembly and repeated-adjacency matvec
    helpers used by the graph operator when ``filter_degree>1`` (not a
    second public operator class; not in package ``__all__``).
``linear``
    :class:`~koopman_graph.operators.LinearOperatorProtocol` plus
    polynomial-graph and one-tap
    :class:`~koopman_graph.operators.MatrixFreeGraphLinearOperator`
    wrappers (off root ``__all__``). Dense assembly is refused above
    :data:`~koopman_graph.operators.MAX_DENSE_LINEAR_OPERATOR_SIZE`.
    Trainer DDP does **not** shrink that representation. Exact
    Kronecker spectrum remains a special case.
``graph_inverse``
    Block-diagonal / Jacobi approximate ``inverse_advance`` helpers for
    graph / hypergraph ``sparsity="block_diagonal"`` (no ``graph/`` subtree).
``hypergraph``
    :class:`~koopman_graph.operators.hypergraph.HypergraphKoopmanOperator`
    (hyperedge-coupled discrete advance; supports ``block_diagonal``).
``heterogeneous``
    :class:`~koopman_graph.operators.heterogeneous.HeteroGraphKoopmanOperator`
    (multiplex per-relation discrete advance; package export).
``matrix_free``
    Matrix-free ``apply_k_eff_*`` matvecs, Richardson / Neumann
    ``invert_k_eff_*`` solvers, and Arnoldi ``spectrum_k_eff_*`` helpers
    for networked effective operators (primitives for
    ``sparsity="distributed"``; **not** trainer DDP / ``[distributed]``
    extras; does **not** enable multi-GPU training).
``kronecker_spectrum``
    Exact Kronecker-sum spectrum reduction for eligible networked
    operators. Discrete graph uses the polynomial pencil
    :math:`B(\\lambda)=\\sum_k\\lambda^k K_k` (one-tap when ``P=1``);
    continuous graph stays the one-tap generator
    ``I⊗L_self + Â⊗L_nbr``. Auto-routed from
    :meth:`~koopman_graph.operators.graph.GraphKoopmanOperator.spectrum`
    and
    :meth:`~koopman_graph.operators.continuous_graph.ContinuousGraphKoopmanOperator.spectrum`
    when eligible; also available as a power-user import from
    ``koopman_graph.operators.kronecker_spectrum`` (not in package
    ``__all__``). Distinct from Arnoldi ``matrix_free`` surrogates.
    Dual adjacency raises on the helper; operator ``.spectrum``
    dense-routes instead.
``global_local``
    :class:`~koopman_graph.operators.global_local.GlobalLocalKoopmanOperator`
    (discrete global backbone + low-rank local window correction).
``continuous_graph``
    :class:`~koopman_graph.operators.continuous_graph.ContinuousGraphKoopmanOperator`
    (continuous networked generator; ``koopman="graph"`` + continuous or alias
    ``koopman="continuous_graph"``).
``continuous_hetero``
    :class:`~koopman_graph.operators.continuous_hetero.ContinuousHeteroGraphKoopmanOperator`
    (continuous multiplex / typed relational generator; ``koopman="hetero_graph"``
    + ``dynamics_mode="continuous"``).
``switched``
    :class:`~koopman_graph.operators.switched.SwitchedKoopmanOperator`
    (finite bank of LTI maps; ``koopman="switched"``). Optional
    per-step ``phase_index`` does not mutate ``mode_index``. Not
    :math:`K(\\mu)`.
``mixture``
    :class:`~koopman_graph.operators.mixture.MixtureKoopmanOperator`
    (softmax mixture of LTI maps; ``koopman="mixture"``). Not
    :math:`K(\\mu)`.
``parametric``
    :class:`~koopman_graph.operators.parametric.ParametricKoopmanOperator`
    (discrete interpolant :math:`K(\\mu)=\\sum_j \\alpha_j(\\mu) K_j`;
    ``koopman="parametric"``). Distinct from switched / mixture.
    Leave-one-regime-out helper
    :func:`~koopman_graph.operators.leave_one_regime_out` is a package
    export, not in root ``__all__``.
``hodge``
    :class:`~koopman_graph.operators.hodge.HodgeKoopmanOperator`
    (Laplacian-structured neighbor term; ``koopman="hodge"``).
``cochain``
    :class:`~koopman_graph.operators.CochainState` and
    :class:`~koopman_graph.operators.CochainKoopmanOperator`
    (degree-specific :math:`k\\le 1` maps on a static signed
    :math:`B_1`). Not a factory kind. Distinct from
    ``koopman="hodge"``. Face latents are stored, not evolved.
    :func:`~koopman_graph.operators.boundary_nilpotency` scores
    :math:`B_1 B_2\\approx 0` on caller-supplied incidences.
``equivariant``
    :class:`~koopman_graph.operators.equivariant.EquivariantKoopmanOperator`
    (scalar, ``scale * I_3`` vector, and ``scale * I_5`` :math:`l=2`
    tensor blocks). Not a factory kind. Rotation tests use
    ``[equivariance]`` / ``e3nn``; the operator leaf does not import
    ``e3nn``. Not a molecular MD stack.
``graphon``
    :func:`~koopman_graph.operators.graphon.sample_graphon_adjacency`
    and :func:`~koopman_graph.operators.graphon.estimate_graphon` for
    dense teaching kernels at multiple :math:`N`.
``stochastic``
    Diagonal process-noise helpers for ``dynamics_mode="stochastic"``.
``stochastic_sde``
    :class:`~koopman_graph.operators.DriftDiffusionKoopman` opt-in
    Euler–Maruyama / Yosida stepper. Not a factory kind. Default
    ``dynamics_mode="stochastic"`` stays diagonal process noise. Not
    certified SDE theory.

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
from koopman_graph.operators.cochain import (
    DEFAULT_NILPOTENCY_ATOL,
    BoundaryNilpotencyReport,
    CochainKoopmanOperator,
    CochainState,
    boundary_nilpotency,
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
from koopman_graph.operators.continuous_hetero import (
    ContinuousHeteroGraphKoopmanOperator,
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
from koopman_graph.operators.equivariant import EquivariantKoopmanOperator
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
from koopman_graph.operators.graphon import (
    MAX_GRAPHON_NODES,
    GraphonEstimate,
    estimate_graphon,
    sample_graphon_adjacency,
)
from koopman_graph.operators.heterogeneous import HeteroGraphKoopmanOperator
from koopman_graph.operators.hodge import HodgeKoopmanOperator
from koopman_graph.operators.hypergraph import (
    HypergraphKoopmanOperator,
    HypergraphSparsity,
)
from koopman_graph.operators.linear import (
    MAX_DENSE_LINEAR_OPERATOR_SIZE,
    EigResult,
    LinearOperatorProtocol,
    MatrixFreeGraphLinearOperator,
    MemoryEstimate,
    PolynomialGraphLinearOperator,
)
from koopman_graph.operators.matrix_free import (
    DEFAULT_DISTRIBUTED_EIGREG_NUM_MODES,
    DEFAULT_DISTRIBUTED_SPECTRUM_NUM_MODES,
    DEFAULT_MATRIX_FREE_INVERSE_MAX_ITERS,
    DEFAULT_MATRIX_FREE_INVERSE_TOL,
    DEFAULT_MATRIX_FREE_SPECTRUM_TOL,
    MatrixFreeInverseResult,
    MatrixFreeSpectrumResult,
    apply_k_eff_graph,
    apply_k_eff_hetero,
    apply_k_eff_hypergraph,
    flatten_node_latents,
    invert_k_eff_graph,
    invert_k_eff_hetero,
    invert_k_eff_hypergraph,
    spectrum_k_eff_graph,
    spectrum_k_eff_hetero,
    spectrum_k_eff_hypergraph,
    unflatten_node_latents,
)
from koopman_graph.operators.mixture import MixtureKoopmanOperator
from koopman_graph.operators.parametric import (
    LeaveOneRegimeOutReport,
    ParametricKoopmanOperator,
    leave_one_regime_out,
)
from koopman_graph.operators.sparse_backend import sparse_leading_eigenvalues
from koopman_graph.operators.stochastic import (
    apply_process_noise,
    attach_process_noise,
    diagonal_process_covariance,
    maybe_apply_process_noise,
)
from koopman_graph.operators.stochastic_sde import DriftDiffusionKoopman
from koopman_graph.operators.switched import (
    DEFAULT_NUM_MODES,
    SwitchedKoopmanOperator,
)

__all__ = [
    "AuxiliarySpectralNetwork",
    "BoundaryNilpotencyReport",
    "CochainKoopmanOperator",
    "CochainState",
    "DEFAULT_NILPOTENCY_ATOL",
    "ContinuousGraphKoopmanOperator",
    "ContinuousGraphSparsity",
    "ContinuousHeteroGraphKoopmanOperator",
    "ContinuousKoopmanOperator",
    "ControlMode",
    "DEFAULT_AUXILIARY_HIDDEN_DIMS",
    "DEFAULT_LOCAL_HIDDEN_DIMS",
    "DEFAULT_LOCAL_RANK",
    "DEFAULT_LOCAL_WINDOW",
    "DEFAULT_DISTRIBUTED_EIGREG_NUM_MODES",
    "DEFAULT_DISTRIBUTED_SPECTRUM_NUM_MODES",
    "DEFAULT_MATRIX_FREE_INVERSE_MAX_ITERS",
    "DEFAULT_MATRIX_FREE_INVERSE_TOL",
    "DEFAULT_MATRIX_FREE_SPECTRUM_TOL",
    "DEFAULT_NUM_MODES",
    "DISSIPATIVE_MIN_EIGENVALUE",
    "DriftDiffusionKoopman",
    "DynamicsMode",
    "EigResult",
    "EquivariantKoopmanOperator",
    "GeneratorParameterization",
    "GlobalLocalKoopmanOperator",
    "GraphAdjacency",
    "GraphKoopmanOperator",
    "GraphSparsity",
    "GraphonEstimate",
    "HeteroGraphKoopmanOperator",
    "HodgeKoopmanOperator",
    "HypergraphKoopmanOperator",
    "HypergraphSparsity",
    "InitMode",
    "KoopmanKind",
    "KoopmanOperator",
    "KoopmanOperatorContract",
    "LinearOperatorProtocol",
    "MAX_DENSE_LINEAR_OPERATOR_SIZE",
    "MatrixFreeGraphLinearOperator",
    "MatrixFreeInverseResult",
    "MatrixFreeSpectrumResult",
    "MAX_GRAPHON_NODES",
    "MemoryEstimate",
    "MixtureKoopmanOperator",
    "ParametricKoopmanOperator",
    "LeaveOneRegimeOutReport",
    "Parameterization",
    "PolynomialGraphLinearOperator",
    "STABILITY_EPS_MARGIN",
    "StabilityCertificate",
    "SwitchedKoopmanOperator",
    "VAN_LOAN_WRITEBACK_ATOL",
    "apply_k_eff_graph",
    "apply_k_eff_hetero",
    "apply_k_eff_hypergraph",
    "apply_process_noise",
    "assemble_block_diagonal_generator",
    "attach_process_noise",
    "boundary_nilpotency",
    "bounded_diagonal",
    "build_stability_certificate",
    "cayley_orthogonal",
    "diagonal_process_covariance",
    "estimate_graphon",
    "flatten_node_latents",
    "invert_k_eff_graph",
    "invert_k_eff_hetero",
    "invert_k_eff_hypergraph",
    "leave_one_regime_out",
    "matrix_log",
    "maybe_apply_process_noise",
    "normalize_auxiliary_hidden_dims",
    "normalize_local_hidden_dims",
    "pad_latent_window",
    "resolve_factory_stability_bound",
    "safe_diagonal_inverse",
    "sample_graphon_adjacency",
    "sparse_leading_eigenvalues",
    "spectral_output_dim",
    "spectrum_k_eff_graph",
    "spectrum_k_eff_hetero",
    "spectrum_k_eff_hypergraph",
    "stack_latent_window",
    "strict_diagonal_values",
    "strict_spectral_bound",
    "unflatten_node_latents",
    "van_loan_factors",
    "van_loan_generator_from_discrete",
]
