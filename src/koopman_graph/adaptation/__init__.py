"""Online adaptation and latent-state estimation for Koopman models.

Capability layout
-----------------
``rls``
    :class:`~koopman_graph.adaptation.RecursiveKoopmanAdapter` and
    :class:`~koopman_graph.adaptation.AdaptationStepResult` for recursive
    least-squares operator updates.
``kalman``
    :class:`~koopman_graph.adaptation.FilterResult` plus reference Kalman /
    RTS numerics (``reference_kalman_filter``, ``rts_smooth``).
``impute``
    Heuristic ``graph_diffuse_impute`` neighbor-average warm-start.
``observer``
    :class:`~koopman_graph.adaptation.KoopmanObserver` façade for Kalman
    filtering / smoothing / imputation under ``observation_masks``.

RLS and observer types (``RecursiveKoopmanAdapter``, ``AdaptationStepResult``,
``KoopmanObserver``, ``FilterResult``, and related helpers) are available from
this package and are intentionally omitted from root ``koopman_graph.__all__``
(thin façade; capability-module imports only).
"""

from koopman_graph.adaptation.kalman import FilterResult
from koopman_graph.adaptation.observer import (
    KoopmanObserver,
    ObservationModel,
)
from koopman_graph.adaptation.rls import (
    AdaptationMode,
    AdaptationStepResult,
    RecursiveKoopmanAdapter,
)

__all__ = [
    "AdaptationMode",
    "AdaptationStepResult",
    "FilterResult",
    "KoopmanObserver",
    "ObservationModel",
    "RecursiveKoopmanAdapter",
]
