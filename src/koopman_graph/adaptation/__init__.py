"""Online adaptation and latent-state estimation for Koopman models.

Capability layout
-----------------
``rls``
    :class:`~koopman_graph.adaptation.RecursiveKoopmanAdapter` and
    :class:`~koopman_graph.adaptation.AdaptationStepResult` for recursive
    least-squares operator updates.
``observer``
    :class:`~koopman_graph.adaptation.KoopmanObserver` and
    :class:`~koopman_graph.adaptation.FilterResult` for Kalman filtering /
    smoothing / imputation under ``observation_masks``.

``RecursiveKoopmanAdapter`` remains on the root façade. Observer types are
power-user imports from this package (not newly promoted into
``koopman_graph.__all__``).
"""

from koopman_graph.adaptation.observer import (
    FilterResult,
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
