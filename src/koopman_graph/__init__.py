"""KoopmanGraph: topology-aware Koopman dynamics on graphs.

Public API
----------
``GraphKoopmanModel``
    End-to-end encode → Koopman advance → decode model.
``GNNEncoder``, ``GATEncoder``
    GNN encoders for latent lifting.
``GNNDecoder``
    GNN decoder for physical reconstruction.
``KoopmanOperator``
    Learnable finite-dimensional Koopman matrix.
``GraphSnapshotSequence``
    Container for time-ordered PyG graph snapshots.
``ForwardConsistencyLoss``
    Latent-space linear evolution consistency loss.
``BackwardConsistencyLoss``
    Latent-space inverse linear evolution consistency loss.
``FitHistory``
    Training history returned by :meth:`~koopman_graph.model.GraphKoopmanModel.fit`.
``LossWeights``
    Weights for reconstruction and consistency loss terms.
``__version__``
    Package version string.
"""

from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.decoder import GNNDecoder
from koopman_graph.encoder import GATEncoder, GNNEncoder
from koopman_graph.losses import BackwardConsistencyLoss, ForwardConsistencyLoss
from koopman_graph.model import GraphKoopmanModel
from koopman_graph.operator import KoopmanOperator
from koopman_graph.training import FitHistory, LossWeights

__all__ = [
    "BackwardConsistencyLoss",
    "FitHistory",
    "ForwardConsistencyLoss",
    "GATEncoder",
    "GNNDecoder",
    "GNNEncoder",
    "GraphKoopmanModel",
    "GraphSnapshotSequence",
    "KoopmanOperator",
    "LossWeights",
    "__version__",
]
__version__ = "0.1.0"
