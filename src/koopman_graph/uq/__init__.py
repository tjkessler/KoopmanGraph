"""Uncertainty quantification peers for KoopmanGraph.

Capability layout
-----------------
``common``
    Shared non-private helpers and result types
    (:class:`~koopman_graph.uq.PredictionInterval`,
    :func:`~koopman_graph.uq.quantile_levels`,
    :func:`~koopman_graph.uq.snapshot_with_features`) used by ensemble and
    latent-Gaussian peers — no cross-module leading-``_`` imports and no
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

Power-user module: import as ``koopman_graph.uq``. Types are intentionally
omitted from root ``koopman_graph.__all__`` (see architecture docs).

Deep ensembles estimate epistemic uncertainty by aggregating independently
seeded :class:`~koopman_graph.model.GraphKoopmanModel` members (Lakshminarayanan
et al., NeurIPS 2017). :class:`~koopman_graph.uq.LatentGaussianKoopmanUQ` is a
linear-Gaussian / Kalman-refined latent path related to the Kalman half of
K²VAE-style pipelines — **not** Deep Probabilistic Koopman (DPK), which
predicts time-varying distribution parameters, and **not** a full K²VAE
(VAE + KalmanNet) reimplementation.

Latent-Gaussian forecasts reuse
:meth:`~koopman_graph.model.GraphKoopmanModel.encode_rollout_origin` and
:mod:`koopman_graph.graph_utils` hold-last topology / ``propagate_latent``
scheduling; closed-form Gaussian moment updates remain local to this package.
"""

from koopman_graph.uq.common import (
    PredictionInterval,
    quantile_levels,
    snapshot_with_features,
)
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

__all__ = [
    "EnsembleGraphKoopmanModel",
    "IntervalForecastModel",
    "LatentGaussianForecast",
    "LatentGaussianKoopmanUQ",
    "PredictionInterval",
    "dense_nodewise_transition",
    "empirical_coverage",
    "propagate_gaussian_covariance",
    "quantile_levels",
    "snapshot_with_features",
]
