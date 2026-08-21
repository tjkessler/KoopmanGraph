"""Uncertainty quantification peers for KoopmanGraph.

Capability layout
-----------------
``common``
    Shared non-private helpers and result types
    (:class:`~koopman_graph.uq.PredictionInterval`,
    :func:`~koopman_graph.uq.quantile_levels`,
    :func:`~koopman_graph.uq.snapshot_with_features`,
    :func:`~koopman_graph.uq.hetero_snapshot_with_features`) used by ensemble
    and latent-Gaussian peers — no cross-module leading-``_`` imports and no
    peer-to-peer import of the shared interval type.
``ensemble``
    :class:`~koopman_graph.uq.EnsembleGraphKoopmanModel` deep ensembles with
    empirical predictive intervals, plus
    :func:`~koopman_graph.uq.empirical_coverage` and the optional
    :class:`~koopman_graph.uq.IntervalForecastModel` Protocol.
``latent_gaussian``
    :class:`~koopman_graph.uq.LatentGaussianKoopmanUQ` linear-Gaussian latent
    forecast with closed-form covariance propagation and optional Kalman
    refinement, plus :class:`~koopman_graph.uq.LatentGaussianForecast` and
    :func:`~koopman_graph.uq.propagate_gaussian_covariance`.
``coverage``
    :class:`~koopman_graph.uq.JointCoverageSpec` names the interval
    estimand (default ``per_node_marginal``). Simultaneous / event
    targets and temporal / graph blocks are named but not implemented.
``scores``
    Proper scores :func:`~koopman_graph.uq.gaussian_crps`,
    :func:`~koopman_graph.uq.gaussian_nll`, and
    :func:`~koopman_graph.uq.energy_score`. Not coverage certificates.
``conformal``
    :class:`~koopman_graph.uq.ConformalKoopmanUQ` split / adaptive (ACI)
    conformal intervals returning
    :class:`~koopman_graph.uq.PredictionInterval` (``Data`` or
    ``HeteroData`` bands). Hetero models score stacked decoded features
    (``N = Σ_τ N_τ``). Calibration state is wrapper-local (not model
    ``FORMAT_VERSION``). Marginal coverage ``≥ 1 − α`` under
    exchangeability; prefer ACI under drift. :attr:`coverage` always
    names ``per_node_marginal``.
``bayesian``
    :class:`~koopman_graph.uq.BayesianKoopmanUQ` diagonal Laplace posterior
    over dense linear factors (``K`` / ``K_self``+``K_nbr``) with
    :meth:`~koopman_graph.uq.BayesianKoopmanUQ.sample_forecast` intervals,
    plus :class:`~koopman_graph.uq.LaplacePosterior` and
    :class:`~koopman_graph.uq.LaplaceFactorSpec`. Not a BNN over
    encoder/decoder weights; no coverage guarantee.

Power-user module: import as ``koopman_graph.uq``. Types are intentionally
omitted from root ``koopman_graph.__all__`` (see architecture docs).

Deep ensembles estimate epistemic uncertainty by aggregating independently
seeded :class:`~koopman_graph.model.GraphKoopmanModel` members (Lakshminarayanan
et al., NeurIPS 2017). :class:`~koopman_graph.uq.LatentGaussianKoopmanUQ` is a
linear-Gaussian / Kalman-refined latent path related to the Kalman half of
K²VAE-style pipelines — **not** Deep Probabilistic Koopman (DPK), which
predicts time-varying distribution parameters, and **not** a full K²VAE
(VAE + KalmanNet) reimplementation. :class:`~koopman_graph.uq.BayesianKoopmanUQ`
is a diagonal Laplace approximation over operator factors only.

Latent-Gaussian forecasts reuse
:meth:`~koopman_graph.model.GraphKoopmanModel.encode_rollout_origin` and
:mod:`koopman_graph.graph_utils` ``propagate_latent`` with the same
``topology_policy`` as model ``predict``; closed-form Gaussian moment
updates remain local to this package.
"""

from koopman_graph.uq.bayesian import (
    BayesianKoopmanUQ,
    LaplaceFactorSpec,
    LaplacePosterior,
)
from koopman_graph.uq.common import (
    PredictionInterval,
    hetero_snapshot_with_features,
    quantile_levels,
    snapshot_with_features,
)
from koopman_graph.uq.conformal import ConformalKoopmanUQ
from koopman_graph.uq.coverage import JointCoverageSpec, require_shipped_coverage
from koopman_graph.uq.ensemble import (
    EnsembleGraphKoopmanModel,
    IntervalForecastModel,
    empirical_coverage,
)
from koopman_graph.uq.latent_gaussian import (
    LatentGaussianForecast,
    LatentGaussianKoopmanUQ,
    dense_nodewise_transition,
    propagate_gaussian_covariance,
)
from koopman_graph.uq.scores import energy_score, gaussian_crps, gaussian_nll

__all__ = [
    "BayesianKoopmanUQ",
    "ConformalKoopmanUQ",
    "EnsembleGraphKoopmanModel",
    "IntervalForecastModel",
    "JointCoverageSpec",
    "LaplaceFactorSpec",
    "LaplacePosterior",
    "LatentGaussianForecast",
    "LatentGaussianKoopmanUQ",
    "PredictionInterval",
    "dense_nodewise_transition",
    "empirical_coverage",
    "energy_score",
    "gaussian_crps",
    "gaussian_nll",
    "hetero_snapshot_with_features",
    "propagate_gaussian_covariance",
    "quantile_levels",
    "require_shipped_coverage",
    "snapshot_with_features",
]
