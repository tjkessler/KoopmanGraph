"""Neural network capability package (encoders, decoders, GNN primitives).

Capability layout
-----------------
``gnn``
    Shared :class:`~koopman_graph.nn.gnn.BaseGNNModule`, activation typing,
    validators, and GCN/GAT/SAGE/DiffConv/Transformer convolution builders
    (power-user).
``encoder``
    :class:`~koopman_graph.nn.encoder.GNNEncoder` /
    :class:`~koopman_graph.nn.encoder.GATEncoder` /
    :class:`~koopman_graph.nn.encoder.SAGEEncoder` /
    :class:`~koopman_graph.nn.encoder.DiffConvEncoder` /
    :class:`~koopman_graph.nn.encoder.GraphTransformerEncoder`.
``decoder``
    :class:`~koopman_graph.nn.decoder.GNNDecoder` /
    :class:`~koopman_graph.nn.decoder.GATDecoder` /
    :class:`~koopman_graph.nn.decoder.SAGEDecoder` /
    :class:`~koopman_graph.nn.decoder.DiffConvDecoder` /
    :class:`~koopman_graph.nn.decoder.GraphTransformerDecoder`.
``delay``
    :class:`~koopman_graph.nn.delay.DelayEmbeddingEncoder` Hankel wrapper and
    delay-window helpers.

The package itself is power-user; encoder/decoder classes remain in the root
``koopman_graph.__all__`` façade. Prefer
``from koopman_graph import GNNEncoder, …`` for application code, or
``from koopman_graph.nn import …`` for power-user imports.
"""

from koopman_graph.nn.decoder import (
    DiffConvDecoder,
    GATDecoder,
    GNNDecoder,
    GraphTransformerDecoder,
    SAGEDecoder,
)
from koopman_graph.nn.delay import DelayEmbeddingEncoder
from koopman_graph.nn.encoder import (
    DiffConvEncoder,
    GATEncoder,
    GNNEncoder,
    GraphTransformerEncoder,
    SAGEEncoder,
)
from koopman_graph.nn.gnn import (
    ActivationName,
    BaseGNNModule,
    DiffusionConv,
    build_diff_convs,
    build_gat_convs,
    build_gcn_convs,
    build_sage_convs,
    build_transformer_convs,
    validate_diffusion_steps,
    validate_gat_attention,
    validate_optional_edge_dim,
    validate_positive_dims,
)

__all__ = [
    "ActivationName",
    "BaseGNNModule",
    "DelayEmbeddingEncoder",
    "DiffConvDecoder",
    "DiffConvEncoder",
    "DiffusionConv",
    "GATDecoder",
    "GATEncoder",
    "GNNDecoder",
    "GNNEncoder",
    "GraphTransformerDecoder",
    "GraphTransformerEncoder",
    "SAGEDecoder",
    "SAGEEncoder",
    "build_diff_convs",
    "build_gat_convs",
    "build_gcn_convs",
    "build_sage_convs",
    "build_transformer_convs",
    "validate_diffusion_steps",
    "validate_gat_attention",
    "validate_optional_edge_dim",
    "validate_positive_dims",
]
