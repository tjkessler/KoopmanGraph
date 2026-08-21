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
``joint_observer``
    :class:`~koopman_graph.adaptation.JointStateTopologyObserver`
    composing the Kalman observer with group-sparse graph factors or
    dense RLS. Homomorphism claims require a separable dictionary
    encoder; default GNN encoders are not separable. Importing this
    peer loads :mod:`koopman_graph.identification`; RLS and
    :class:`~koopman_graph.adaptation.KoopmanObserver` stay
    identification-free at package import.

RLS and observer types (``RecursiveKoopmanAdapter``, ``AdaptationStepResult``,
``KoopmanObserver``, ``JointStateTopologyObserver``, ``FilterResult``, and
related helpers) are available from
this package and are intentionally omitted from root ``koopman_graph.__all__``
(thin façade; capability-module imports only).
"""

from __future__ import annotations

from typing import Any

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
    "JointObserverResult",
    "JointStateTopologyObserver",
    "KoopmanObserver",
    "ObservationModel",
    "RecursiveKoopmanAdapter",
]


def __getattr__(name: str) -> Any:
    """Lazy-load the joint observer without importing identification.

    Parameters
    ----------
    name : str
        Attribute requested on :mod:`koopman_graph.adaptation`.

    Returns
    -------
    object
        Public joint-observer symbol.

    Raises
    ------
    AttributeError
        If ``name`` is not a known lazy export.
    """
    if name in {"JointObserverResult", "JointStateTopologyObserver"}:
        from koopman_graph.adaptation.joint_observer import (
            JointObserverResult,
            JointStateTopologyObserver,
        )

        if name == "JointObserverResult":
            return JointObserverResult
        return JointStateTopologyObserver
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    """Return sorted public names including lazy joint-observer exports.

    Returns
    -------
    list of str
        Names in :data:`__all__`.
    """
    return sorted(__all__)
