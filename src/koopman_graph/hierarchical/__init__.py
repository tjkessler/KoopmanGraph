"""Hierarchical / multi-resolution GraphKoopman forecasting (power-user).

Capability layout
-----------------
``model``
    :class:`~koopman_graph.hierarchical.HierarchicalGraphKoopmanModel` —
    pool (TopK / optional SAG) → composed
    :class:`~koopman_graph.model.GraphKoopmanModel` on the coarse graph →
    learned scatter-unpool.
``pooling``
    Pooling helpers (:func:`~koopman_graph.hierarchical.build_pool_layer`,
    :class:`~koopman_graph.hierarchical.ScatterUnpool`, control pooling).

Import as ``koopman_graph.hierarchical``. Types are intentionally omitted from
root ``koopman_graph.__all__`` (see architecture docs).

This path is **coarse-level forecasting with unpooling**, not a
physics-augmented spatiotemporal super-resolution pipeline (e.g. P-K-GCN).
Forecasting only at the coarsest level can miss fine-scale structure; use
``predict(..., resolution=...)`` to inspect coarse vs fine outputs.
``pool_ratios=(1.0,)`` keeps all nodes and is intended as a no-op-size sanity
check against a flat model.
"""

from koopman_graph.hierarchical.model import HierarchicalGraphKoopmanModel
from koopman_graph.hierarchical.pooling import (
    PoolStep,
    ScatterUnpool,
    apply_pool_layer,
    build_pool_layer,
    pool_control,
    pool_control_sequence,
)

__all__ = [
    "HierarchicalGraphKoopmanModel",
    "PoolStep",
    "ScatterUnpool",
    "apply_pool_layer",
    "build_pool_layer",
    "pool_control",
    "pool_control_sequence",
]
