"""Loss functions for Koopman graph dynamics training.

Capability layout
-----------------
``consistency``
    Forward / backward / Lie consistency losses.
``regularization``
    Eigenvalue hinge and Koopman sparsity penalties.
``reconstruction``
    Masked MSE helper and worst-case reconstruction loss.
``physics``
    PDE residual loss.
``rollout``
    Sequence / multi-start autoregressive rollout losses.

Prefer ``from koopman_graph.losses import …``. Peer modules may be imported
directly for power-user work; do not reach into leading-underscore helpers
across module boundaries.
"""

from koopman_graph.losses.consistency import (
    BackwardConsistencyLoss,
    ForwardConsistencyLoss,
    LieConsistencyLoss,
)
from koopman_graph.losses.persistence import PersistenceRegularizer
from koopman_graph.losses.physics import PDEResidualLoss
from koopman_graph.losses.reconstruction import (
    WorstCaseReconstructionLoss,
    masked_mse_loss,
)
from koopman_graph.losses.regularization import (
    MAX_ASSEMBLED_EIGREG_SIZE,
    EigenvalueRegularizationLoss,
    KoopmanSparsityLoss,
)
from koopman_graph.losses.rollout import (
    rollout_multi_start_loss,
    rollout_sequence_loss,
)

__all__ = [
    "BackwardConsistencyLoss",
    "EigenvalueRegularizationLoss",
    "ForwardConsistencyLoss",
    "KoopmanSparsityLoss",
    "LieConsistencyLoss",
    "MAX_ASSEMBLED_EIGREG_SIZE",
    "PDEResidualLoss",
    "PersistenceRegularizer",
    "WorstCaseReconstructionLoss",
    "masked_mse_loss",
    "rollout_multi_start_loss",
    "rollout_sequence_loss",
]
