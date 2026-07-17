"""Neural network capability package (encoders, decoders, GNN primitives).

Capability layout
-----------------
``gnn``
    Shared :class:`~koopman_graph.nn.gnn.BaseGNNModule`, activation typing,
    validators, and GCN/GAT convolution builders (power-user).
``encoder``
    :class:`~koopman_graph.nn.encoder.GNNEncoder` /
    :class:`~koopman_graph.nn.encoder.GATEncoder`.
``decoder``
    :class:`~koopman_graph.nn.decoder.GNNDecoder` /
    :class:`~koopman_graph.nn.decoder.GATDecoder`.
``delay``
    :class:`~koopman_graph.nn.delay.DelayEmbeddingEncoder` Hankel wrapper and
    delay-window helpers.

The package itself is power-user; encoder/decoder classes remain in the root
``koopman_graph.__all__`` façade. Prefer
``from koopman_graph import GNNEncoder, …`` for application code, or
``from koopman_graph.nn import …`` for power-user imports.
"""

from koopman_graph.nn.decoder import GATDecoder, GNNDecoder
from koopman_graph.nn.delay import DelayEmbeddingEncoder
from koopman_graph.nn.encoder import GATEncoder, GNNEncoder
from koopman_graph.nn.gnn import (
    ActivationName,
    BaseGNNModule,
    build_gat_convs,
    build_gcn_convs,
    validate_gat_attention,
    validate_positive_dims,
)

__all__ = [
    "ActivationName",
    "BaseGNNModule",
    "DelayEmbeddingEncoder",
    "GATDecoder",
    "GATEncoder",
    "GNNDecoder",
    "GNNEncoder",
    "build_gat_convs",
    "build_gcn_convs",
    "validate_gat_attention",
    "validate_positive_dims",
]
