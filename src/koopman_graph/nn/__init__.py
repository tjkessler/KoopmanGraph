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
``constraint_decoders``
    :class:`~koopman_graph.nn.MassConservingDecoder` /
    :class:`~koopman_graph.nn.PositivityDecoder` /
    :class:`~koopman_graph.nn.LinearConservingDecoder`
    (opt-in decoded-space simplex / positivity / linear conservation;
    not a factory kind; must not import ``model``).
``hypergraph``
    :class:`~koopman_graph.nn.hypergraph.HypergraphEncoder` /
    :class:`~koopman_graph.nn.hypergraph.HypergraphDecoder`.
``simplicial``
    :class:`~koopman_graph.nn.simplicial.SimplicialEncoder` /
    :class:`~koopman_graph.nn.simplicial.SimplicialDecoder`
    (combinatorial ``L_1`` mixing; not sheaf/cell).
``sheaf``
    :class:`~koopman_graph.nn.sheaf.SheafGNNEncoder` /
    :class:`~koopman_graph.nn.sheaf.SheafGNNDecoder`
    (diagonal restriction maps by default; opt-in general maps).
``cell_complex``
    :class:`~koopman_graph.nn.cell_complex.CellComplex` with boundary /
    Hodge helpers ``B_k`` / ``L_k`` for ``k ∈ {0, 1, 2, 3}``, plus
    :class:`~koopman_graph.nn.cell_complex.CellComplexGNNEncoder` /
    :class:`~koopman_graph.nn.cell_complex.CellComplexGNNDecoder`
    (factory ``encoder="cell_complex"``). Order-2 teaching
    (:func:`~koopman_graph.nn.order2_cochain_teaching`) binds
    :class:`~koopman_graph.operators.CochainKoopmanOperator` to a
    filled triangle. Degree 3 is the teaching ceiling — **not**
    TopologicX parity.
``equivariant``
    :class:`~koopman_graph.nn.equivariant.InvariantGeometryEncoder`
    (Tier A invariant distance/angle features from ``Data.pos``) and optional
    :class:`~koopman_graph.nn.equivariant.E3EquivariantEncoder` (Tier B
    ``e3nn`` / ``[equivariance]``; default projects to invariant latents;
    ``project_invariants=False`` keeps steerable vector channels for an
    equivariant ``K``).
``neural_operator``
    :class:`~koopman_graph.nn.neural_operator.FourierNeuralOperatorEncoder`
    (mesh-index Fourier lift; shared ``K`` across discretizations is a
    teaching MVP).
``predicted_topology``
    :class:`~koopman_graph.nn.predicted_topology.SparseCandidateTopologyHead`
    (default graph-state path; at most ``candidate_k`` destinations per
    node), power-user
    :class:`~koopman_graph.nn.predicted_topology.PredictedTopologyHead`
    (dense :math:`N\\times N` logits with an :math:`N` ceiling), and
    :class:`~koopman_graph.nn.predicted_topology.PresenceHead`. Distinct
    from static AdaptiveAdjacency.
``heterogeneous``
    :class:`~koopman_graph.nn.heterogeneous.RelGraphEncoder` /
    :class:`~koopman_graph.nn.heterogeneous.RelGraphDecoder`
    (multiplex / typed R-GCN-lite peers; factory-supported for
    ``koopman="hetero_graph"``) and optional
    :class:`~koopman_graph.nn.heterogeneous.HGTEncoder` /
    :class:`~koopman_graph.nn.heterogeneous.HGTDecoder` (typed PyG
    ``HGTConv`` peers; not required for hetero support).
``delay``
    :class:`~koopman_graph.nn.delay.DelayEmbeddingEncoder` Takens-style
    channel stacking (not
    :class:`~koopman_graph.baselines.HankelDMDBaseline` /
    :class:`~koopman_graph.baselines.HAVOKBaseline`) and delay-window
    helpers.
``receptive_field``
    Encoder vs discrete-graph-operator hop check
    (:func:`~koopman_graph.nn.receptive_field.check_encoder_operator_receptive_field`;
    warn-only; not in root ``__all__``).
``separable``
    :class:`~koopman_graph.nn.SeparableDictionaryEncoder` /
    :class:`~koopman_graph.nn.SeparableDictionaryDecoder` node-wise
    lifts (homomorphism precondition; not a GNN).
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
from koopman_graph.nn.cell_complex import (
    MAX_CELL_COMPLEX_DEGREE,
    CellComplex,
    CellComplexGNNDecoder,
    CellComplexGNNEncoder,
    Order2CochainTeaching,
    bind_cell_complex_decoder,
    bind_cochain_operator,
    boundary_incidence_b2,
    boundary_operator,
    cell_complex_boundary_nilpotency,
    hodge_laplacian,
    hodge_laplacian_matvec,
    order2_cochain_teaching,
    teaching_order2_triangle,
    teaching_order3_tetrahedron,
)
from koopman_graph.nn.constraint_decoders import (
    LinearConservingDecoder,
    MassConservingDecoder,
    PositivityDecoder,
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
from koopman_graph.nn.neural_operator import FourierNeuralOperatorEncoder
from koopman_graph.nn.predicted_topology import (
    DEFAULT_CANDIDATE_K,
    DENSE_TOPOLOGY_MAX_NODES,
    PredictedTopologyHead,
    PresenceHead,
    SparseCandidateTopologyHead,
    TopologyPolicy,
    build_candidate_index,
    build_supervision_index,
)
from koopman_graph.nn.receptive_field import (
    ReceptiveFieldMismatchWarning,
    ReceptiveFieldReport,
    check_encoder_operator_receptive_field,
)
from koopman_graph.nn.separable import (
    SeparableDictionaryDecoder,
    SeparableDictionaryEncoder,
    is_separable_dictionary,
)
from koopman_graph.nn.sheaf import (
    SheafGNNDecoder,
    SheafGNNEncoder,
    bind_sheaf_decoder,
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
    "CellComplex",
    "CellComplexGNNDecoder",
    "CellComplexGNNEncoder",
    "DEFAULT_CANDIDATE_K",
    "DEFAULT_TOPOLOGY_EMBEDDING_DIM",
    "DENSE_TOPOLOGY_MAX_NODES",
    "DelayEmbeddingEncoder",
    "DiffConvDecoder",
    "DiffConvEncoder",
    "DiffusionConv",
    "E3EquivariantEncoder",
    "FourierNeuralOperatorEncoder",
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
    "LinearConservingDecoder",
    "MAX_CELL_COMPLEX_DEGREE",
    "MassConservingDecoder",
    "Order2CochainTeaching",
    "PositivityDecoder",
    "PresenceHead",
    "PredictedTopologyHead",
    "ReceptiveFieldMismatchWarning",
    "ReceptiveFieldReport",
    "RelGraphConv",
    "RelGraphDecoder",
    "RelGraphEncoder",
    "SAGEDecoder",
    "SAGEEncoder",
    "SeparableDictionaryDecoder",
    "SeparableDictionaryEncoder",
    "SheafGNNDecoder",
    "SheafGNNEncoder",
    "SimplicialDecoder",
    "SimplicialEncoder",
    "SparseCandidateTopologyHead",
    "TopologyPolicy",
    "bind_cell_complex_decoder",
    "bind_cochain_operator",
    "bind_hypergraph_decoder",
    "bind_sheaf_decoder",
    "bind_simplicial_decoder",
    "boundary_incidence_b2",
    "boundary_operator",
    "cell_complex_boundary_nilpotency",
    "order2_cochain_teaching",
    "teaching_order2_triangle",
    "teaching_order3_tetrahedron",
    "build_candidate_index",
    "build_supervision_index",
    "check_encoder_operator_receptive_field",
    "hodge_laplacian",
    "hodge_laplacian_matvec",
    "invariant_geometry_features",
    "is_separable_dictionary",
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
