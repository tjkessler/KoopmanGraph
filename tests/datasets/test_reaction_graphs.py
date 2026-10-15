"""Tests for :mod:`koopman_graph.datasets.reaction.graphs`."""

from __future__ import annotations

import pytest
import torch

from koopman_graph.datasets.reaction import (
    PRODUCT_RELATION,
    REACTANT_RELATION,
    REACTION_NODE,
    SPECIES_NODE,
    ReactionSystem,
    mass_action_jacobian,
    reaction_directed_hyperedges,
    reaction_hyperedge_index,
    species_influence_graph,
    species_reaction_bipartite,
)


def _michaelis_menten() -> ReactionSystem:
    """``E + S <-> ES -> E + P`` plus a feed ``∅ -> S`` (species E, S, ES, P)."""
    return ReactionSystem(
        species=("E", "S", "ES", "P"),
        reactant_stoichiometry=torch.tensor(
            [
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ),
        product_stoichiometry=torch.tensor(
            [
                [0.0, 0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        ),
        rate_constants=torch.tensor([2.0, 1.0, 0.5, 0.1]),
    )


def test_influence_graph_matches_jacobian_sparsity() -> None:
    system = _michaelis_menten()
    edge_index, edge_weight = species_influence_graph(system, self_loops=True)
    c = torch.tensor([0.3, 0.7, 0.2, 0.1], dtype=torch.float64)
    jac = mass_action_jacobian(system, c)
    pattern = torch.zeros(4, 4, dtype=torch.bool)
    pattern[edge_index[1], edge_index[0]] = True  # J[target, source]
    assert torch.equal(pattern, jac != 0)
    assert edge_index.dtype == torch.long
    assert edge_weight.dtype == torch.float32
    assert edge_index.shape[1] == edge_weight.shape[0]
    # Edges are unique and sorted by (source, target).
    keys = (edge_index[0] * 4 + edge_index[1]).tolist()
    assert keys == sorted(set(keys))


def test_influence_graph_weights_and_self_loops() -> None:
    system = _michaelis_menten()
    edge_index, edge_weight = species_influence_graph(system)
    assert not bool((edge_index[0] == edge_index[1]).any())
    pairs = {
        (int(s), int(t)): float(w)
        for s, t, w in zip(edge_index[0], edge_index[1], edge_weight, strict=True)
    }
    # ES drives unbinding (-> E, S) and catalysis (-> E, P): E gets two.
    es, e, s, p = 2, 0, 1, 3
    assert pairs[(es, e)] == 2.0
    assert pairs[(es, s)] == 1.0
    assert pairs[(es, p)] == 1.0
    # P never drives anything; the zeroth-order feed adds no edges.
    assert p not in edge_index[0].tolist()


def test_reaction_hyperedge_index_layout() -> None:
    system = _michaelis_menten()
    hyperedge_index = reaction_hyperedge_index(system)
    assert hyperedge_index.dtype == torch.long
    assert hyperedge_index.shape[0] == 2
    members = {
        j: sorted(hyperedge_index[0, hyperedge_index[1] == j].tolist())
        for j in range(system.num_reactions)
    }
    assert members == {0: [0, 1, 2], 1: [0, 1, 2], 2: [0, 2, 3], 3: [1]}


def test_directed_hyperedges_split_reactants_and_products() -> None:
    system = _michaelis_menten()
    tail_index, head_index = reaction_directed_hyperedges(system)
    tails = {j: tail_index[0, tail_index[1] == j].tolist() for j in range(4)}
    heads = {j: head_index[0, head_index[1] == j].tolist() for j in range(4)}
    assert tails == {0: [0, 1], 1: [2], 2: [2], 3: []}
    assert heads == {0: [2], 1: [0, 1], 2: [0, 3], 3: [1]}


def test_species_reaction_bipartite_heterodata() -> None:
    system = ReactionSystem(
        species=("A", "A2"),
        reactant_stoichiometry=torch.tensor([[2.0, 0.0]]),
        product_stoichiometry=torch.tensor([[0.0, 1.0]]),
        rate_constants=torch.tensor([4.0]),
    )
    data = species_reaction_bipartite(system)
    assert data[SPECIES_NODE].num_nodes == 2
    assert data[REACTION_NODE].num_nodes == 1
    assert data[REACTION_NODE].rate_constant.tolist() == [4.0]
    assert data[REACTION_NODE].rate_constant.dtype == torch.float64
    assert data[REACTANT_RELATION].edge_index.tolist() == [[0], [0]]
    assert data[REACTANT_RELATION].edge_weight.tolist() == [2.0]
    assert data[PRODUCT_RELATION].edge_index.tolist() == [[0], [1]]
    assert data[PRODUCT_RELATION].edge_weight.tolist() == [1.0]
    data.validate()


def test_bipartite_keeps_tiny_rate_constants() -> None:
    system = ReactionSystem(
        species=("A", "B"),
        reactant_stoichiometry=torch.tensor([[1.0, 0.0]]),
        product_stoichiometry=torch.tensor([[0.0, 1.0]]),
        rate_constants=torch.tensor([1e-50], dtype=torch.float64),
    )
    rate = species_reaction_bipartite(system)[REACTION_NODE].rate_constant
    assert rate.tolist() == [1e-50]


@pytest.mark.parametrize(
    "fn",
    [
        species_influence_graph,
        reaction_hyperedge_index,
        reaction_directed_hyperedges,
        species_reaction_bipartite,
    ],
)
def test_graph_builders_require_reaction_system(fn: object) -> None:
    with pytest.raises(TypeError, match="must be a ReactionSystem"):
        fn("not a system")  # type: ignore[operator]
