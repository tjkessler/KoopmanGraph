"""Primary GraphKoopmanModel estimator and supporting peer helpers.

Capability layout
-----------------
``estimator``
    :class:`~koopman_graph.model.GraphKoopmanModel` sklearn-like workflow.
``factory``
    Operator construction, injection validation, and resolved-component
    writeback (``resolve_model_components`` / ``apply_resolved_components``).
``validation``
    Control and sequence validation helpers, including the ``fit`` preamble
    (``prepare_fit_inputs``).
``timing``
    Timestamp / increment policy helpers.
``encoding``
    Physics / delay / encode-origin helpers.
``inference``
    Spectrum / predict / evaluate orchestration helpers.
``online_adaptation``
    Online RLS adaptation façade-bridge helpers (does not own
    :mod:`koopman_graph.adaptation` implementations).

Prefer ``from koopman_graph.model import GraphKoopmanModel`` (or the root
façade). Peer modules may be imported directly for power-user work; do not
reach into leading-underscore helpers across module boundaries.
"""

from koopman_graph.protocols import DynamicsMode

from . import (
    encoding,
    estimator,
    factory,
    inference,
    online_adaptation,
    timing,
    validation,
)
from .estimator import GraphKoopmanModel

__all__ = ["GraphKoopmanModel"]
