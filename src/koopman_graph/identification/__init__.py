"""Opt-in operator identification types, closed-form solvers, and reports.

Capability layout
-----------------
``config``
    Frozen :class:`~koopman_graph.identification.IdentificationConfig`
    (solver name, ridge weight, selection keys).
``report``
    Frozen :class:`~koopman_graph.identification.IdentificationReport`
    with reconstruction, one-step, rollout, closure, invariance,
    spectral, and stability groups.
``protocol``
    :class:`~koopman_graph.identification.IdentificationBackend` plus
    :class:`~koopman_graph.identification.LatentPairs` /
    :class:`~koopman_graph.identification.OperatorSnapshot`.
``solvers``
    Ridge, TLS, and constrained least-squares maps on frozen encodings
    (:func:`~koopman_graph.identification.identify_operator`).
``metrics``
    One-step / short-rollout MSE and spectral-radius report helpers.
``invariance``
    Finite-sample subspace leakage :math:`\\eta`
    (:func:`~koopman_graph.identification.subspace_invariance_report`).
``gate``
    Residual-aware dictionary selection
    (:func:`~koopman_graph.identification.select_resdmd_gated`).
``sparse_factors``
    :func:`~koopman_graph.identification.identify_sparse_graph_factors`
    (STLSQ / group-lasso on :math:`K_{\\mathrm{self}}` /
    :math:`K_{\\mathrm{nbr}}`; unpenalized refit). Distinct from
    :func:`~koopman_graph.analysis.identify_sparse_dynamics` and
    :class:`~koopman_graph.losses.KoopmanSparsityLoss`.
``rank``
    :func:`~koopman_graph.identification.select_latent_rank` over a
    candidate grid (VAMP-2, ResDMD elbow, stability-penalized
    held-out MSE). Distinct from Ray Tune HPO for encoder
    ``latent_dim``.

Honesty
-------
This package is importable and **off** root ``__all__``. Pass
``fit(..., identification=IdentificationConfig(...))`` to alternate
encoder Adam steps with a closed-form :math:`K` update. Default
``identification=None`` keeps the 0.14 Adam path. Reports record
finite-sample MSE and :math:`\\rho(K)`; they are not Haseli–Cortés,
certified ResDMD, or stability certificates. ``solver="varpro"`` is not
implemented. Graph / hetero / continuous / controlled operators raise.
:func:`~koopman_graph.identification.subspace_invariance_report` is a
finite-sample projection-leakage ratio, not a Haseli–Cortés certificate.
``fit`` does not populate
:attr:`~koopman_graph.identification.IdentificationReport.invariance`.
``gate_resdmd=True`` fills the finite-dictionary ``spectral`` block on
the final report; :func:`~koopman_graph.identification.select_resdmd_gated`
rejects polluted RMSE-only winners.
:func:`~koopman_graph.identification.identify_sparse_graph_factors`
selects sparse graph factors on frozen encodings; it does not replace
latent SINDy or :class:`~koopman_graph.losses.KoopmanSparsityLoss` and
is not Pan et al. (2021) multi-task EDMD dictionary pruning.
:func:`~koopman_graph.identification.select_latent_rank` scores truncated
SVD ranks of frozen encodings; it is not Ray Tune AutoML for encoder
``latent_dim``.

Import rules
------------
This package must not import :mod:`koopman_graph.training`,
:mod:`koopman_graph.model`, or :mod:`koopman_graph.adaptation`.
:mod:`koopman_graph.training` must not import this package at module
load (lazy bind inside :func:`~koopman_graph.training.run_fit_loop`).
:class:`~koopman_graph.adaptation.JointStateTopologyObserver` may import
this package; RLS and :class:`~koopman_graph.adaptation.KoopmanObserver`
must not load it at package import.
"""

from koopman_graph.identification.config import (
    IDENTIFICATION_SOLVERS,
    IdentificationConfig,
    IdentificationSolver,
)
from koopman_graph.identification.gate import (
    DEFAULT_RESDMD_GATE_TOLERANCE,
    ResDMDGateCandidate,
    ResDMDGateResult,
    select_resdmd_gated,
)
from koopman_graph.identification.invariance import (
    SubspaceInvarianceReport,
    subspace_invariance_report,
)
from koopman_graph.identification.metrics import (
    DEFAULT_IDENTIFICATION_ROLLOUT_HORIZON,
    build_identification_report,
)
from koopman_graph.identification.protocol import (
    IdentificationBackend,
    LatentPairs,
    OperatorSnapshot,
)
from koopman_graph.identification.rank import (
    LatentRankReport,
    select_latent_rank,
)
from koopman_graph.identification.report import (
    IdentificationReport,
    InvarianceBlock,
    MetricBlock,
    SpectralReliabilityBlock,
    StabilityBlock,
)
from koopman_graph.identification.solvers import (
    ClosedFormBackend,
    apply_operator_snapshot,
    identify_operator,
)
from koopman_graph.identification.sparse_factors import (
    SparseFactorReport,
    identify_sparse_graph_factors,
)

__all__ = [
    "DEFAULT_IDENTIFICATION_ROLLOUT_HORIZON",
    "DEFAULT_RESDMD_GATE_TOLERANCE",
    "IDENTIFICATION_SOLVERS",
    "ClosedFormBackend",
    "IdentificationBackend",
    "IdentificationConfig",
    "IdentificationReport",
    "IdentificationSolver",
    "InvarianceBlock",
    "LatentPairs",
    "LatentRankReport",
    "MetricBlock",
    "OperatorSnapshot",
    "ResDMDGateCandidate",
    "ResDMDGateResult",
    "SpectralReliabilityBlock",
    "SparseFactorReport",
    "StabilityBlock",
    "SubspaceInvarianceReport",
    "apply_operator_snapshot",
    "build_identification_report",
    "identify_operator",
    "identify_sparse_graph_factors",
    "select_latent_rank",
    "select_resdmd_gated",
    "subspace_invariance_report",
]
