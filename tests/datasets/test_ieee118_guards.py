"""Coverage and error-path tests for :mod:`koopman_graph.datasets`."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import HeteroData

from koopman_graph import (
    GNNDecoder,
    RelGraphDecoder,
    RelGraphEncoder,
)
from koopman_graph.data import HeteroGraphSnapshotSequence
from koopman_graph.datasets.ieee118 import (
    bus_type_name,
    homogeneous_features_to_typed_hetero,
    partition_buses_by_type,
)
from koopman_graph.losses.rollout import (
    _bind_hetero_decoder,
    _multiplex_target_features,
    _relation_topology_at_from_targets,
)
from koopman_graph.model.validation import (
    uses_relgraph_modules,
    validate_sequence_hyperedges,
)
from koopman_graph.nn.heterogeneous import (
    RelGraphDecoder as RelGraphDecoderCls,
)


def _multiplex_snapshot(*, num_nodes: int = 4, in_channels: int = 3) -> HeteroData:
    """Build a one-type, two-relation multiplex snapshot."""
    data = HeteroData()
    data["node"].x = torch.randn(num_nodes, in_channels)
    data["node", "r1", "node"].edge_index = torch.tensor(
        [[0, 1, 2], [1, 2, 3]],
        dtype=torch.long,
    )
    data["node", "r2", "node"].edge_index = torch.tensor(
        [[0, 2], [2, 0]],
        dtype=torch.long,
    )
    return data


def _typed_snapshot() -> HeteroData:
    """Build a two-type snapshot with one cross relation."""
    data = HeteroData()
    data["gen"].x = torch.randn(2, 3)
    data["load"].x = torch.randn(3, 2)
    data["gen", "feeds", "load"].edge_index = torch.tensor(
        [[0, 1], [0, 2]],
        dtype=torch.long,
    )
    return data


def test_ieee118_typed_helpers_and_rollout_guards() -> None:
    """IEEE typed helpers and rollout multiplex guards raise on bad inputs."""
    with pytest.raises(ValueError, match="Unsupported MATPOWER BUS_TYPE"):
        bus_type_name(9)
    with pytest.raises(ValueError, match="rank-1"):
        partition_buses_by_type(torch.ones(2, 2, dtype=torch.long))
    with pytest.raises(ValueError, match="Unsupported MATPOWER BUS_TYPE codes"):
        partition_buses_by_type(torch.tensor([1, 9], dtype=torch.long))

    with pytest.raises(ValueError, match="exactly one node type"):
        _multiplex_target_features(_typed_snapshot())
    topo = _relation_topology_at_from_targets(
        [_multiplex_snapshot()],
        num_relations=2,
    )
    edges, weights = topo(0)
    assert len(edges) == 2
    typed_topo = _relation_topology_at_from_targets(
        [_typed_snapshot()],
        num_relations=1,
        node_types=("gen", "load"),
        edge_types=(("gen", "feeds", "load"),),
    )
    t_edges, _t_weights = typed_topo(0)
    assert len(t_edges) == 1
    typed_dec = RelGraphDecoderCls(
        2,
        4,
        {"gen": 3, "load": 2},
        num_relations=1,
        node_types=("gen", "load"),
        edge_types=(("gen", "feeds", "load"),),
        num_layers=1,
    )
    bound = _bind_hetero_decoder(typed_dec, {"gen": 2, "load": 3})
    assert callable(bound)


def test_ieee_typed_conversion_and_model_validation_guards() -> None:
    """IEEE typed conversion and RelGraph peer checks cover remaining branches."""
    features = torch.randn(4, 4)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    bus_types = torch.tensor([1, 2, 3, 1], dtype=torch.long)
    with pytest.raises(ValueError, match=r"\(num_buses, F\)"):
        homogeneous_features_to_typed_hetero(features[0], edge_index, bus_types)
    with pytest.raises(ValueError, match="must match"):
        homogeneous_features_to_typed_hetero(
            features,
            edge_index,
            torch.tensor([1, 2, 3], dtype=torch.long),
        )
    with pytest.raises(ValueError, match="partition is empty"):
        homogeneous_features_to_typed_hetero(
            features,
            edge_index,
            bus_types,
            node_type_order=("ghost",),
        )
    hetero = homogeneous_features_to_typed_hetero(features, edge_index, bus_types)
    assert set(hetero.node_types) <= {"generator", "load", "slack"}

    with pytest.raises(ValueError, match="must be used together"):
        uses_relgraph_modules(
            RelGraphEncoder(3, 4, 2, num_relations=1, num_layers=1),
            GNNDecoder(2, 4, 3),
        )
    with pytest.raises(ValueError, match="num_relations"):
        uses_relgraph_modules(
            RelGraphEncoder(3, 4, 2, num_relations=1, num_layers=1),
            RelGraphDecoder(2, 4, 3, num_relations=2, num_layers=1),
        )
    with pytest.raises(ValueError, match="normalization"):
        uses_relgraph_modules(
            RelGraphEncoder(
                3, 4, 2, num_relations=1, num_layers=1, normalization="rgcn_in_degree"
            ),
            RelGraphDecoder(
                2, 4, 3, num_relations=1, num_layers=1, normalization="random_walk"
            ),
        )
    seq = HeteroGraphSnapshotSequence([_multiplex_snapshot() for _ in range(2)])
    with pytest.raises(ValueError, match="homogeneous hyperedge-carrying"):
        validate_sequence_hyperedges(seq, allow_hyperedges=True)
    validate_sequence_hyperedges(seq, allow_hyperedges=False)
