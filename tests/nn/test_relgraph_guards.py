"""Coverage and error-path tests for :mod:`koopman_graph.nn`."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import HeteroData

from koopman_graph.nn.heterogeneous import (
    RelGraphConv,
    RelGraphDecoder,
    RelGraphEncoder,
    _normalize_edge_type_order,
    _pack_relgraph_latents,
    _resolve_hgt_activation,
    _unpack_relgraph_latents,
    resolve_hgt_typed_inputs,
    resolve_multiplex_relation_inputs,
    resolve_typed_relation_inputs,
)

_TYPES = ("a", "b")

_EDGE_TYPES = (("a", "r", "b"),)

_LATENT_DIMS = {"a": 2, "b": 3}


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


def test_relgraph_latent_dimension_and_shape_guards() -> None:
    """RelGraph rectangular metadata and flat/block shapes are validated."""
    with pytest.raises(ValueError, match="single node type"):
        RelGraphEncoder(
            2,
            4,
            3,
            1,
            node_types=("a", "b"),
            latent_dims={"a": 3, "b": 3},
        )
    with pytest.raises(ValueError, match="must equal latent_dim"):
        RelGraphEncoder(2, 4, 3, 1, latent_dims={"node": 2})

    counts = {"a": 2, "b": 1}
    with pytest.raises(ValueError, match="latent block"):
        _pack_relgraph_latents(
            _TYPES,
            {"a": torch.zeros(2, 2), "b": torch.zeros(2, 3)},
            counts,
            _LATENT_DIMS,
        )
    with pytest.raises(ValueError, match="z_flat must have shape"):
        _unpack_relgraph_latents(torch.zeros(7, 1), _TYPES, counts, _LATENT_DIMS)

    decoder = RelGraphDecoder(
        4,
        4,
        {"a": 2, "b": 2},
        1,
        num_layers=1,
        node_types=_TYPES,
        edge_types=_EDGE_TYPES,
        latent_dims=_LATENT_DIMS,
    )
    edge_bank = [torch.tensor([[0], [2]], dtype=torch.long)]
    with pytest.raises(ValueError, match="num_nodes_dict is required"):
        decoder(torch.zeros(7), edge_bank)
    with pytest.raises(TypeError, match="flat latent Tensor"):
        decoder(HeteroData(), edge_bank, num_nodes_dict=counts)
    with pytest.raises(ValueError, match="edge_index relation banks"):
        decoder(torch.zeros(7), num_nodes_dict=counts)


def test_relgraph_conv_validation_and_bias_none() -> None:
    """RelGraphConv validates inputs and supports bias=False."""
    layer = RelGraphConv(3, 4, num_relations=2, bias=False)
    assert layer.bias is None
    layer.reset_parameters()
    x = torch.randn(4, 3)
    edges = [
        torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        torch.tensor([[0], [3]], dtype=torch.long),
    ]
    with pytest.raises(ValueError, match="Expected x with shape"):
        layer(torch.randn(4, 3, 1), edges)
    with pytest.raises(ValueError, match="Expected in_channels"):
        layer(torch.randn(4, 2), edges)
    with pytest.raises(ValueError, match="relation edge banks"):
        layer(x, edges[:1])
    with pytest.raises(ValueError, match="relation weight banks"):
        layer(x, edges, edge_weights=[None])
    weights = [torch.ones(2), torch.ones(1)]
    out = layer(x, edges, edge_weights=weights)
    assert out.shape == (4, 4)


def test_resolve_relation_input_helpers_errors() -> None:
    """Multiplex / typed / HGT resolvers reject malformed inputs."""
    with pytest.raises(ValueError, match="relation banks"):
        resolve_multiplex_relation_inputs(
            torch.randn(3, 2),
            edge_index={"r1": torch.tensor([[0], [1]], dtype=torch.long)},
            num_relations=2,
        )
    with pytest.raises(ValueError, match="exactly one node type"):
        resolve_multiplex_relation_inputs(_typed_snapshot(), num_relations=1)

    with pytest.raises(ValueError, match="node_types must be unique"):
        RelGraphEncoder(
            {"a": 2, "b": 3},
            hidden_channels=4,
            latent_dim=2,
            num_relations=1,
            node_types=("a", "a"),
            edge_types=(("a", "r", "b"),),
        )
    with pytest.raises(ValueError, match=r"\(src, rel, dst\)"):
        _normalize_edge_type_order([("a", "b")], num_relations=1)
    with pytest.raises(ValueError, match="non-empty strings"):
        _normalize_edge_type_order([("", "r", "b")], num_relations=1)
    with pytest.raises(ValueError, match="must match"):
        _normalize_edge_type_order([("a", "r0", "b")], num_relations=2)
    with pytest.raises(ValueError, match="must be unique"):
        _normalize_edge_type_order(
            [("a", "r0", "b"), ("a", "r0", "b")],
            num_relations=2,
        )
    with pytest.raises(ValueError, match="edge_types is required"):
        _normalize_edge_type_order(None, num_relations=1, required=True)

    typed = _typed_snapshot()
    feats, edges, _weights, counts = resolve_typed_relation_inputs(
        typed,
        node_types=("gen", "load"),
        edge_types=None,
        num_relations=1,
    )
    assert set(feats) == {"gen", "load"}
    assert len(edges) == 1
    assert counts["gen"] == 2

    with pytest.raises(TypeError, match="HeteroData or a mapping"):
        resolve_typed_relation_inputs(
            42,  # type: ignore[arg-type]
            node_types=("a",),
            num_relations=1,
        )
    with pytest.raises(ValueError, match="must have shape"):
        resolve_typed_relation_inputs(
            {"gen": torch.randn(4), "load": torch.randn(3, 2)},
            edge_index=[torch.tensor([[0], [2]], dtype=torch.long)],
            node_types=("gen", "load"),
            edge_types=(("gen", "feeds", "load"),),
            num_relations=1,
        )

    with pytest.raises(ValueError, match="Unknown activation"):
        _resolve_hgt_activation("gelu")  # type: ignore[arg-type]

    hgt_snap = HeteroData()
    hgt_snap["a"].x = torch.randn(2, 2)
    hgt_snap["b"].x = torch.randn(2, 2)
    hgt_snap["c"].x = torch.randn(1, 2)
    hgt_snap["a", "r0", "b"].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    with pytest.raises(ValueError, match="expects exactly"):
        resolve_hgt_typed_inputs(
            hgt_snap,
            node_types=("a", "b"),
            edge_types=(("a", "r0", "b"),),
        )
    missing_edge = HeteroData()
    missing_edge["a"].x = torch.randn(2, 2)
    missing_edge["b"].x = torch.randn(2, 2)
    with pytest.raises(ValueError, match="missing edge type"):
        resolve_hgt_typed_inputs(
            missing_edge,
            node_types=("a", "b"),
            edge_types=(("a", "r0", "b"),),
        )
    with pytest.raises(ValueError, match="edge_index relation mapping"):
        resolve_hgt_typed_inputs(
            {"a": torch.randn(2, 2), "b": torch.randn(2, 2)},
            edge_index=None,
            node_types=("a", "b"),
            edge_types=(("a", "r0", "b"),),
        )
