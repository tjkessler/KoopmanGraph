"""Neural network capability package (encoders, decoders, GNN primitives).

Capability layout
-----------------
``gnn``
    Shared :class:`~koopman_graph.nn.gnn.BaseGNNModule`, activation typing,
    validators, and GCN/GAT/SAGE/DiffConv/Transformer/Hypergraph convolution
    builders (power-user).
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
``hypergraph``
    :class:`~koopman_graph.nn.hypergraph.HypergraphEncoder` /
    :class:`~koopman_graph.nn.hypergraph.HypergraphDecoder`.
``simplicial``
    :class:`~koopman_graph.nn.simplicial.SimplicialEncoder` /
    :class:`~koopman_graph.nn.simplicial.SimplicialDecoder`
    (combinatorial ``L_1`` mixing; not sheaf/cell).
``equivariant``
    :class:`~koopman_graph.nn.equivariant.InvariantGeometryEncoder`
    (Tier A invariant distance/angle features from ``Data.pos``) and optional
    :class:`~koopman_graph.nn.equivariant.E3EquivariantEncoder` (Tier B
    ``e3nn`` / ``[equivariance]``; steerable encode to invariant latents; not
    equivariant ``K``).
``heterogeneous``
    :class:`~koopman_graph.nn.heterogeneous.RelGraphEncoder` /
    :class:`~koopman_graph.nn.heterogeneous.RelGraphDecoder`
    (multiplex / typed R-GCN-lite peers; factory-supported for
    ``koopman="hetero_graph"``) and optional
    :class:`~koopman_graph.nn.heterogeneous.HGTEncoder` /
    :class:`~koopman_graph.nn.heterogeneous.HGTDecoder` (typed PyG
    ``HGTConv`` peers; not required for hetero support).
``delay``
    :class:`~koopman_graph.nn.delay.DelayEmbeddingEncoder` Hankel wrapper and
    delay-window helpers.
``adaptive_topology``
    :class:`~koopman_graph.nn.adaptive_topology.AdaptiveAdjacency` self-adaptive
    pairwise adjacency (Graph WaveNet construction; power-user).

The package itself is power-user; encoder/decoder classes remain in the root
``koopman_graph.__all__`` façade. Prefer
``from koopman_graph import GNNEncoder, …`` for application code, or
``from koopman_graph.nn import …`` for power-user imports.
"""

from koopman_graph.nn.adaptive_topology import (
    DEFAULT_TOPOLOGY_EMBEDDING_DIM,
    AdaptiveAdjacency,
)
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
from koopman_graph.nn.equivariant import (
    GEOM_CHANNELS,
    E3EquivariantEncoder,
    InvariantGeometryEncoder,
    invariant_geometry_features,
)
from koopman_graph.nn.gnn import (
    ActivationName,
    BaseGNNModule,
    DiffusionConv,
    build_diff_convs,
    build_gat_convs,
    build_gcn_convs,
    build_hypergraph_convs,
    build_sage_convs,
    build_transformer_convs,
    validate_diffusion_steps,
    validate_gat_attention,
    validate_optional_edge_dim,
    validate_positive_dims,
)
from koopman_graph.nn.heterogeneous import (
    HGTDecoder,
    HGTEncoder,
    RelGraphConv,
    RelGraphDecoder,
    RelGraphEncoder,
    build_hgt_convs,
    build_relgraph_convs,
)
from koopman_graph.nn.hypergraph import (
    HypergraphDecoder,
    HypergraphEncoder,
    bind_hypergraph_decoder,
)
from koopman_graph.nn.simplicial import (
    SimplicialDecoder,
    SimplicialEncoder,
    bind_simplicial_decoder,
)

__all__ = [
    "ActivationName",
    "AdaptiveAdjacency",
    "BaseGNNModule",
    "DEFAULT_TOPOLOGY_EMBEDDING_DIM",
    "DelayEmbeddingEncoder",
    "DiffConvDecoder",
    "DiffConvEncoder",
    "DiffusionConv",
    "E3EquivariantEncoder",
    "GATDecoder",
    "GATEncoder",
    "GEOM_CHANNELS",
    "GNNDecoder",
    "GNNEncoder",
    "GraphTransformerDecoder",
    "GraphTransformerEncoder",
    "HGTDecoder",
    "HGTEncoder",
    "HypergraphDecoder",
    "HypergraphEncoder",
    "InvariantGeometryEncoder",
    "RelGraphConv",
    "RelGraphDecoder",
    "RelGraphEncoder",
    "SAGEDecoder",
    "SAGEEncoder",
    "SimplicialDecoder",
    "SimplicialEncoder",
    "bind_hypergraph_decoder",
    "bind_simplicial_decoder",
    "invariant_geometry_features",
    "build_diff_convs",
    "build_gat_convs",
    "build_gcn_convs",
    "build_hgt_convs",
    "build_hypergraph_convs",
    "build_relgraph_convs",
    "build_sage_convs",
    "build_transformer_convs",
    "validate_diffusion_steps",
    "validate_gat_attention",
    "validate_optional_edge_dim",
    "validate_positive_dims",
]
