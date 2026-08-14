"""Multiplex / typed hetero orbit ties (TASK-1825 / TASK-1826)."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import HeteroData

pytest.importorskip("networkx")

from koopman_graph.data import HeteroGraphSnapshotSequence
from koopman_graph.model import GraphKoopmanModel
from koopman_graph.nn import RelGraphDecoder, RelGraphEncoder
from koopman_graph.operators import HeteroGraphKoopmanOperator
from koopman_graph.training.loop import bind_pending_orbit_ties

_TYPED_NODE_TYPES = ("a", "b")
_TYPED_EDGE_TYPES = (("a", "cycle", "a"), ("b", "cycle", "b"), ("a", "to", "b"))


def _cycle_edge_index(num_nodes: int) -> torch.Tensor:
    edges: list[list[int]] = []
    for node in range(num_nodes):
        nxt = (node + 1) % num_nodes
        edges.extend([[node, nxt], [nxt, node]])
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _star_edge_index(num_nodes: int) -> torch.Tensor:
    edges: list[list[int]] = []
    for leaf in range(1, num_nodes):
        edges.extend([[0, leaf], [leaf, 0]])
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _multiplex_cycle_snapshot(
    *,
    num_nodes: int = 6,
    feat_dim: int = 2,
    seed: int = 0,
) -> HeteroData:
    generator = torch.Generator().manual_seed(seed)
    data = HeteroData()
    data["node"].x = torch.randn(num_nodes, feat_dim, generator=generator)
    cycle = _cycle_edge_index(num_nodes)
    data["node", "r1", "node"].edge_index = cycle
    data["node", "r2", "node"].edge_index = cycle[:, :0]
    return data


def _multiplex_model(
    *,
    latent_dim: int = 2,
    auto_orbits: bool = False,
) -> GraphKoopmanModel:
    return GraphKoopmanModel(
        encoder=RelGraphEncoder(
            2,
            hidden_channels=4,
            latent_dim=latent_dim,
            num_relations=2,
            num_layers=1,
        ),
        decoder=RelGraphDecoder(
            latent_dim=latent_dim,
            hidden_channels=4,
            out_channels=2,
            num_relations=2,
            num_layers=1,
        ),
        latent_dim=latent_dim,
        time_step=1.0,
        koopman="hetero_graph",
        koopman_auto_orbits=auto_orbits,
    )


def _typed_dual_cycle_snapshot(
    *,
    num_a: int = 6,
    num_b: int = 4,
    seed: int = 0,
) -> HeteroData:
    generator = torch.Generator().manual_seed(seed)
    data = HeteroData()
    data["a"].x = torch.randn(num_a, 2, generator=generator)
    data["b"].x = torch.randn(num_b, 2, generator=generator)
    data["a", "cycle", "a"].edge_index = _cycle_edge_index(num_a)
    data["b", "cycle", "b"].edge_index = _cycle_edge_index(num_b)
    data["a", "to", "b"].edge_index = torch.tensor(
        [[0, 1], [0, 1]],
        dtype=torch.long,
    )
    return data


def _typed_model(*, auto_orbits: bool = False) -> GraphKoopmanModel:
    return GraphKoopmanModel(
        encoder=RelGraphEncoder(
            {"a": 2, "b": 2},
            hidden_channels=4,
            latent_dim=2,
            num_relations=len(_TYPED_EDGE_TYPES),
            num_layers=1,
            node_types=_TYPED_NODE_TYPES,
            edge_types=_TYPED_EDGE_TYPES,
        ),
        decoder=RelGraphDecoder(
            latent_dim=2,
            hidden_channels=4,
            out_channels={"a": 2, "b": 2},
            num_relations=len(_TYPED_EDGE_TYPES),
            num_layers=1,
            node_types=_TYPED_NODE_TYPES,
            edge_types=_TYPED_EDGE_TYPES,
        ),
        latent_dim=2,
        time_step=1.0,
        koopman="hetero_graph",
        koopman_node_types=_TYPED_NODE_TYPES,
        koopman_edge_types=_TYPED_EDGE_TYPES,
        koopman_auto_orbits=auto_orbits,
    )


def test_multiplex_cycle_auto_orbits_single_orbit() -> None:
    """Union of multiplex cycle banks yields one automorphism orbit."""
    num_nodes = 6
    latent_dim = 2
    op = HeteroGraphKoopmanOperator(
        latent_dim,
        num_relations=2,
        init_mode="identity",
        auto_orbits=True,
    )
    banks = [_cycle_edge_index(num_nodes), _cycle_edge_index(num_nodes)[:, :0]]
    z = torch.randn(num_nodes, latent_dim)
    _ = op.advance(z, edge_indices=banks)
    assert op.orbit_partition == ((0, 1, 2, 3, 4, 5),)
    assert op.uses_orbit_selves
    assert len(op._orbit_selves) == 1


def test_multiplex_star_orbit_tied_self_blocks_equal_within_orbit() -> None:
    """Nodes in the same multiplex orbit share identical self blocks."""
    num_nodes = 5
    latent_dim = 2
    op = HeteroGraphKoopmanOperator(
        latent_dim,
        num_relations=1,
        init_mode="identity",
        auto_orbits=True,
    )
    banks = [_star_edge_index(num_nodes)]
    op.ensure_orbit_binding(num_nodes, edge_index=banks[0])
    assert op.orbit_partition == ((0,), (1, 2, 3, 4))
    with torch.no_grad():
        op._orbit_selves[0].K.zero_()
        op._orbit_selves[0].K.fill_diagonal_(0.5)
        op._orbit_selves[1].K.zero_()
        op._orbit_selves[1].K.fill_diagonal_(0.8)
    blocks = op.tied_self_blocks(num_nodes)
    assert torch.allclose(blocks[1], blocks[2])
    assert torch.allclose(blocks[1], blocks[4])
    assert not torch.allclose(blocks[0], blocks[1])


def test_bind_pending_orbit_ties_binds_multiplex_hetero() -> None:
    """Fit-start bind resolves multiplex hetero auto_orbits from union banks."""
    model = _multiplex_model(auto_orbits=True)
    sequence = HeteroGraphSnapshotSequence(
        [_multiplex_cycle_snapshot(seed=t) for t in range(3)]
    )
    assert isinstance(model.koopman, HeteroGraphKoopmanOperator)
    assert model.koopman.orbit_partition is None
    bind_pending_orbit_ties(model, [sequence])
    assert model.koopman.orbit_partition == ((0, 1, 2, 3, 4, 5),)


def test_typed_dual_cycle_auto_orbits_within_type_blocks() -> None:
    """Typed auto-orbits stay inside each type's global index range."""
    num_a, num_b = 6, 4
    num_nodes = num_a + num_b
    op = HeteroGraphKoopmanOperator(
        2,
        num_relations=3,
        node_types=_TYPED_NODE_TYPES,
        edge_types=_TYPED_EDGE_TYPES,
        init_mode="identity",
        auto_orbits=True,
    )
    # Global banks: a-cycle, b-cycle (offset), a→b cross.
    banks = [
        _cycle_edge_index(num_a),
        _cycle_edge_index(num_b) + num_a,
        torch.tensor([[0, 1], [num_a, num_a + 1]], dtype=torch.long),
    ]
    counts = {"a": num_a, "b": num_b}
    z = torch.randn(num_nodes, 2)
    _ = op.advance(z, edge_indices=banks, num_nodes_dict=counts)
    assert op.orbit_partition == (
        (0, 1, 2, 3, 4, 5),
        (6, 7, 8, 9),
    )
    for orbit in op.orbit_partition or ():
        in_a = all(0 <= n < num_a for n in orbit)
        in_b = all(num_a <= n < num_nodes for n in orbit)
        assert in_a or in_b
        assert not (in_a and in_b)


def test_typed_star_orbit_tied_self_blocks_equal_within_orbit() -> None:
    """Within-type star orbits share self blocks; hub differs from leaves."""
    num_a, num_b = 5, 3
    op = HeteroGraphKoopmanOperator(
        2,
        num_relations=2,
        node_types=_TYPED_NODE_TYPES,
        edge_types=(("a", "star", "a"), ("b", "loop", "b")),
        init_mode="identity",
        auto_orbits=True,
    )
    banks = [
        _star_edge_index(num_a),
        _cycle_edge_index(num_b) + num_a,
    ]
    counts = {"a": num_a, "b": num_b}
    op.ensure_typed_orbit_binding(banks, counts)
    assert op.orbit_partition is not None
    assert (0,) in op.orbit_partition
    assert (1, 2, 3, 4) in op.orbit_partition
    hub_id = op.orbit_partition.index((0,))
    leaf_id = op.orbit_partition.index((1, 2, 3, 4))
    with torch.no_grad():
        for module in op._orbit_selves:
            module.K.zero_()
        op._orbit_selves[hub_id].K.fill_diagonal_(0.5)
        op._orbit_selves[leaf_id].K.fill_diagonal_(0.8)
    blocks = op.tied_self_blocks(num_a + num_b)
    assert torch.allclose(blocks[1], blocks[2])
    assert not torch.allclose(blocks[0], blocks[1])


def test_typed_explicit_partition_rejects_cross_type_orbit() -> None:
    """Explicit orbits that mix type blocks raise at typed ensure."""
    op = HeteroGraphKoopmanOperator(
        2,
        num_relations=1,
        node_types=_TYPED_NODE_TYPES,
        edge_types=(("a", "to", "b"),),
        init_mode="identity",
        orbit_partition=((0, 6), (1, 2, 3, 4, 5), (7, 8, 9)),
    )
    banks = [torch.tensor([[0], [6]], dtype=torch.long)]
    with pytest.raises(ValueError, match="must not mix node-type blocks"):
        op.ensure_typed_orbit_binding(banks, {"a": 6, "b": 4})


def test_rectangular_rejects_orbit_kwargs() -> None:
    """Rectangular typed operators still reject orbit kwargs."""
    with pytest.raises(ValueError, match="unsupported for rectangular"):
        HeteroGraphKoopmanOperator(
            4,
            num_relations=1,
            node_types=_TYPED_NODE_TYPES,
            edge_types=(("a", "r", "b"),),
            latent_dims={"a": 2, "b": 3},
            parameterization="dense",
            sparsity="dense",
            auto_orbits=True,
        )


def test_bind_pending_orbit_ties_binds_typed_hetero() -> None:
    """Fit-start bind resolves typed hetero auto_orbits per type block."""
    model = _typed_model(auto_orbits=True)
    sequence = HeteroGraphSnapshotSequence(
        [_typed_dual_cycle_snapshot(seed=t) for t in range(3)]
    )
    assert isinstance(model.koopman, HeteroGraphKoopmanOperator)
    assert model.koopman.orbit_partition is None
    bind_pending_orbit_ties(model, [sequence])
    assert model.koopman.orbit_partition == (
        (0, 1, 2, 3, 4, 5),
        (6, 7, 8, 9),
    )


def test_continuous_hetero_factory_rejects_orbit_kwargs() -> None:
    """Continuous hetero + orbit kwargs raise at the factory."""
    with pytest.raises(ValueError, match="unsupported for continuous hetero"):
        GraphKoopmanModel(
            encoder=RelGraphEncoder(
                2,
                hidden_channels=4,
                latent_dim=2,
                num_relations=1,
                num_layers=1,
            ),
            decoder=RelGraphDecoder(
                latent_dim=2,
                hidden_channels=4,
                out_channels=2,
                num_relations=1,
                num_layers=1,
            ),
            latent_dim=2,
            time_step=0.1,
            dynamics_mode="continuous",
            koopman="hetero_graph",
            koopman_auto_orbits=True,
        )


def test_homogeneous_orbit_path_unchanged_on_ring() -> None:
    """Homogeneous graph auto_orbits still bind a single ring orbit."""
    from koopman_graph.operators import GraphKoopmanOperator

    num_nodes = 6
    op = GraphKoopmanOperator(2, init_mode="identity", auto_orbits=True)
    edge_index = _cycle_edge_index(num_nodes)
    _ = op.advance(torch.randn(num_nodes, 2), edge_index=edge_index)
    assert op.orbit_partition == ((0, 1, 2, 3, 4, 5),)


def test_typed_orbit_binding_validation_boundaries() -> None:
    """Typed orbit binding validates operator mode and no-op configuration."""
    multiplex = HeteroGraphKoopmanOperator(2, num_relations=1)
    with pytest.raises(ValueError, match="requires a typed"):
        multiplex.ensure_typed_orbit_binding(
            [torch.empty((2, 0), dtype=torch.long)],
            {"node": 2},
        )

    rectangular = HeteroGraphKoopmanOperator(
        4,
        num_relations=1,
        node_types=_TYPED_NODE_TYPES,
        edge_types=(("a", "to", "b"),),
        latent_dims={"a": 2, "b": 3},
        parameterization="dense",
        sparsity="dense",
    )
    with pytest.raises(ValueError, match="unsupported for rectangular"):
        rectangular.ensure_typed_orbit_binding(
            [torch.tensor([[0], [2]], dtype=torch.long)],
            {"a": 2, "b": 2},
        )

    typed = HeteroGraphKoopmanOperator(
        2,
        num_relations=1,
        node_types=_TYPED_NODE_TYPES,
        edge_types=(("a", "to", "b"),),
    )
    typed.ensure_typed_orbit_binding(
        [torch.tensor([[0], [2]], dtype=torch.long)],
        {"a": 2, "b": 2},
    )
    with pytest.raises(RuntimeError, match="requires auto_orbits=True"):
        typed._bind_typed_auto_orbits(
            [torch.tensor([[0], [2]], dtype=torch.long)],
            {"a": 2, "b": 2},
        )


def test_typed_explicit_orbits_validate_ranges_and_bind() -> None:
    """Explicit typed orbits reject outside nodes and bind valid partitions."""
    outside = HeteroGraphKoopmanOperator(
        2,
        num_relations=1,
        node_types=_TYPED_NODE_TYPES,
        edge_types=(("a", "to", "b"),),
        orbit_partition=((0, 1), (2, 3, 4)),
    )
    banks = [torch.tensor([[0], [2]], dtype=torch.long)]
    with pytest.raises(ValueError, match="lies outside typed stacked ranges"):
        outside.ensure_typed_orbit_binding(banks, {"a": 2, "b": 2})

    valid = HeteroGraphKoopmanOperator(
        2,
        num_relations=1,
        node_types=_TYPED_NODE_TYPES,
        edge_types=(("a", "to", "b"),),
        orbit_partition=((0, 1), (2, 3)),
    )
    valid.ensure_typed_orbit_binding(banks, {"a": 2, "b": 2})
    valid.reset_parameters()
    assert valid.uses_orbit_selves


def test_typed_auto_orbits_handle_empty_intra_type_banks_and_reentry() -> None:
    """Auto-orbits handle cross-type-only banks and ignore repeated binding."""
    op = HeteroGraphKoopmanOperator(
        2,
        num_relations=1,
        node_types=_TYPED_NODE_TYPES,
        edge_types=(("a", "to", "b"),),
        auto_orbits=True,
    )
    banks = [torch.tensor([[0], [2]], dtype=torch.long)]
    counts = {"a": 2, "b": 2}
    op.ensure_typed_orbit_binding(banks, counts)
    partition = op.orbit_partition
    op._bind_typed_auto_orbits(banks, counts)
    assert op.orbit_partition == partition
