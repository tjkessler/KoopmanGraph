"""Spatiotemporal GNN forecaster baselines (STGCN / DCRNN / Graph WaveNet).

Lightweight in-repo reference implementations for comparing
:class:`~koopman_graph.model.GraphKoopmanModel` against nonlinear
spatiotemporal GNN forecasters under a shared ``fit`` / ``predict`` surface.
They are **not** substitutes for dedicated traffic-forecasting libraries.

Import as::

    from koopman_graph.baselines.gnn import (
        DCRNNBaseline,
        GraphWaveNetBaseline,
        STGCNBaseline,
    )

:meth:`~koopman_graph.baselines.gnn.base.GNNForecasterBaseline.spectrum`
raises :class:`RuntimeError` (no linear Koopman operator).
"""

from koopman_graph.baselines.gnn.dcrnn import DCRNNBaseline
from koopman_graph.baselines.gnn.stgcn import STGCNBaseline
from koopman_graph.baselines.gnn.wavenet import GraphWaveNetBaseline

__all__ = [
    "DCRNNBaseline",
    "GraphWaveNetBaseline",
    "STGCNBaseline",
]
