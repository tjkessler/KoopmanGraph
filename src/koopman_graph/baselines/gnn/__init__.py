"""Spatiotemporal GNN forecaster baselines (STGCN … GraphCast).

Lightweight in-repo reference implementations for comparing
:class:`~koopman_graph.model.GraphKoopmanModel` against nonlinear
spatiotemporal GNN forecasters under a shared ``fit`` / ``predict`` surface.
They are **not** substitutes for dedicated traffic-forecasting libraries.

Import as::

    from koopman_graph.baselines.gnn import (
        AGCRNBaseline,
        DCRNNBaseline,
        ForecasterProtocol,
        GraphCastBaseline,
        GraphWaveNetBaseline,
        MTGNNBaseline,
        STGCNBaseline,
        STGODEBaseline,
    )

Each baseline exposes
:meth:`~koopman_graph.baselines.gnn.base.GNNForecasterBaseline.protocol`
with a non-empty ``deviations`` tuple versus paper / LibCity-style scripts.

:meth:`~koopman_graph.baselines.gnn.base.GNNForecasterBaseline.spectrum`
raises :class:`RuntimeError` (no linear Koopman operator).

``STGODEBaseline`` requires the optional ``[baselines-ode]`` extra
(``torchdiffeq``) at fit/predict time; importing the class does not.
``GraphCastBaseline`` is a pure-PyTorch mesh-weather teaching slice (not a
PEMS/METR traffic forecaster); ``[baselines-graphcast]`` is reserved.
"""

from koopman_graph.baselines.gnn.agcrn import AGCRNBaseline
from koopman_graph.baselines.gnn.dcrnn import DCRNNBaseline
from koopman_graph.baselines.gnn.graphcast import GraphCastBaseline
from koopman_graph.baselines.gnn.mtgnn import MTGNNBaseline
from koopman_graph.baselines.gnn.protocol import (
    EmptyProtocolDeviationsError,
    ForecasterProtocol,
)
from koopman_graph.baselines.gnn.stgcn import STGCNBaseline
from koopman_graph.baselines.gnn.stgode import STGODEBaseline
from koopman_graph.baselines.gnn.wavenet import GraphWaveNetBaseline

# Source of truth for teaching forecasters (TASK-1928 registry tests).
TEACHING_BASELINES: tuple[type, ...] = (
    STGCNBaseline,
    DCRNNBaseline,
    GraphWaveNetBaseline,
    AGCRNBaseline,
    MTGNNBaseline,
    STGODEBaseline,
    GraphCastBaseline,
)

__all__ = [
    "AGCRNBaseline",
    "DCRNNBaseline",
    "EmptyProtocolDeviationsError",
    "ForecasterProtocol",
    "GraphCastBaseline",
    "GraphWaveNetBaseline",
    "MTGNNBaseline",
    "STGCNBaseline",
    "STGODEBaseline",
    "TEACHING_BASELINES",
]
