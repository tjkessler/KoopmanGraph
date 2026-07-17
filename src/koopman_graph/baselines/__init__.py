"""Classical topology-agnostic Koopman baselines.

Capability layout
-----------------
``base``
    :class:`~koopman_graph.baselines.ClassicalBaseline` scaffolding and shared
    flattening / least-squares helpers.
``dmd``
    :class:`~koopman_graph.baselines.DMDBaseline`.
``dmdc``
    :class:`~koopman_graph.baselines.DMDcBaseline`.
``edmd``
    :class:`~koopman_graph.baselines.EDMDBaseline`.

All three baselines share :class:`ClassicalBaseline` scaffolding and
structurally implement :class:`~koopman_graph.protocols.ForecastModel`
(``fit`` / ``predict`` / ``spectrum``). Import the Protocol from
:mod:`koopman_graph.protocols` for typing; it is not re-exported in package
``__all__``.

Dynamic-topology sequences
(:attr:`~koopman_graph.data.GraphSnapshotSequence.is_dynamic_topology`) are
rejected at ``fit``: these baselines flatten node states and freeze the
initial graph's edges on ``predict``, so varying topology would be silently
ignored.

Per-node (3-D) control layouts are rejected by :class:`DMDcBaseline`:
classical DMDc uses a single global control vector per transition, while
neural / adaptation paths preserve per-node control rows. See the architecture
control layout capability matrix.
"""

from koopman_graph.baselines.base import ClassicalBaseline
from koopman_graph.baselines.dmd import DMDBaseline
from koopman_graph.baselines.dmdc import DMDcBaseline
from koopman_graph.baselines.edmd import EDMDBaseline

__all__ = [
    "ClassicalBaseline",
    "DMDBaseline",
    "DMDcBaseline",
    "EDMDBaseline",
]
