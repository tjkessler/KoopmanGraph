"""Classical topology-agnostic Koopman baselines.

Capability layout
-----------------
``base``
    :class:`~koopman_graph.baselines.ClassicalBaseline` scaffolding and shared
    non-private helpers (``require_static_topology``, ``flatten_snapshots``,
    ``fit_row_operator``, ``fit_controlled_row_operator``,
    ``require_global_controls``, ``transition_controls``, ``copy_topology``,
    ``check_initial_graph``). Not re-exported in package ``__all__``.
``dmd``
    :class:`~koopman_graph.baselines.DMDBaseline`.
``dmdc``
    :class:`~koopman_graph.baselines.DMDcBaseline`.
``edmd``
    :class:`~koopman_graph.baselines.EDMDBaseline` (polynomial / RBF / kernel
    dictionaries; Nyström / random-feature kernel approximations; full
    kernel centers are O(T^2)).
``fbdmd``
    :class:`~koopman_graph.baselines.FBDMDBaseline` — forward–backward DMD
    on flattened node states (topology-blind).
``tlsdmd``
    :class:`~koopman_graph.baselines.TLSDMDBaseline` — total-least-squares
    DMD on flattened node states (topology-blind).
``optdmd``
    :class:`~koopman_graph.baselines.OptDMDBaseline` — optimized DMD MVP
    (variable-projection style) on flattened node states.
``streaming_dmd``
    :class:`~koopman_graph.baselines.StreamingDMDBaseline` — online Gram
    least-squares DMD with :meth:`~koopman_graph.baselines.StreamingDMDBaseline.update`.
``mrdmd``
    :class:`~koopman_graph.baselines.MRDMDBaseline` — depth-2 multi-resolution
    DMD tree (forecast uses the root operator).
``transfer_operator``
    :class:`~koopman_graph.baselines.UlamTransferOperatorBaseline` — Ulam
    Galerkin transfer / Perron–Frobenius matrix on a fixed box indicator
    dictionary (density propagation; topology-blind).
``vamp2``
    Topology-blind VAMP-2 score / loss helpers
    (:func:`~koopman_graph.baselines.vamp2.vamp2_score`,
    :func:`~koopman_graph.baselines.vamp2.vamp2_loss`). Not a ForecastModel;
    not GraphVAMPnets / MD.
``gnn``
    Spatiotemporal GNN forecaster baselines
    (:class:`~koopman_graph.baselines.gnn.STGCNBaseline`,
    :class:`~koopman_graph.baselines.gnn.DCRNNBaseline`,
    :class:`~koopman_graph.baselines.gnn.GraphWaveNetBaseline`).

Classical DMD-family baselines share :class:`ClassicalBaseline` scaffolding and
structurally implement :class:`~koopman_graph.protocols.ForecastModel`
(``fit`` / ``predict`` / ``spectrum``). Import the Protocol from
:mod:`koopman_graph.protocols` for typing; it is not re-exported in package
``__all__``.

GNN forecasters are neural ``nn.Module`` baselines with sklearn-style ``fit``
returning ``self``. Their ``spectrum`` method raises ``RuntimeError`` (no linear
operator). Prefer ``from koopman_graph.baselines.gnn import …``.

Dynamic-topology sequences
(:attr:`~koopman_graph.data.GraphSnapshotSequence.is_dynamic_topology`) are
rejected at ``fit`` for both classical and GNN forecaster baselines: predictions
freeze the initial graph's edges, so varying topology would be silently ignored.

Per-node (3-D) control layouts are rejected by :class:`DMDcBaseline`:
classical DMDc uses a single global control vector per transition, while
neural / adaptation paths preserve per-node control rows. See the architecture
control layout capability matrix.
"""

from koopman_graph.baselines.base import ClassicalBaseline
from koopman_graph.baselines.dmd import DMDBaseline
from koopman_graph.baselines.dmdc import DMDcBaseline
from koopman_graph.baselines.edmd import EDMDBaseline
from koopman_graph.baselines.fbdmd import FBDMDBaseline
from koopman_graph.baselines.gnn import (
    DCRNNBaseline,
    GraphWaveNetBaseline,
    STGCNBaseline,
)
from koopman_graph.baselines.mrdmd import MRDMDBaseline
from koopman_graph.baselines.optdmd import OptDMDBaseline
from koopman_graph.baselines.streaming_dmd import StreamingDMDBaseline
from koopman_graph.baselines.tlsdmd import TLSDMDBaseline
from koopman_graph.baselines.transfer_operator import UlamTransferOperatorBaseline
from koopman_graph.baselines.vamp2 import vamp2_loss, vamp2_score

__all__ = [
    "ClassicalBaseline",
    "DCRNNBaseline",
    "DMDBaseline",
    "DMDcBaseline",
    "EDMDBaseline",
    "FBDMDBaseline",
    "GraphWaveNetBaseline",
    "MRDMDBaseline",
    "OptDMDBaseline",
    "STGCNBaseline",
    "StreamingDMDBaseline",
    "TLSDMDBaseline",
    "UlamTransferOperatorBaseline",
    "vamp2_loss",
    "vamp2_score",
]
