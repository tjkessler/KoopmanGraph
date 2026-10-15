"""Graph encodings of a :class:`~koopman_graph.datasets.reaction.ReactionSystem`.

Everything here is derived from stoichiometry alone:

* :func:`species_influence_graph` builds a directed species graph with an
  edge ``s → t`` whenever species ``s`` is a reactant of a reaction that
  changes ``t``. This is the sparsity pattern of the mass-action Jacobian
  ``∂(dc_t/dt)/∂c_s``.
* :func:`reaction_hyperedge_index` builds an undirected hypergraph with one
  hyperedge per reaction, using the library's ``hyperedge_index`` layout
  ``(2, nnz)`` (row 0 = species, row 1 = reaction).
* :func:`reaction_directed_hyperedges` returns directed-hypergraph
  ``tail_index`` (reactants) and ``head_index`` (products) in the same
  layout.
* :func:`species_reaction_bipartite` builds a PyG
  :class:`~torch_geometric.data.HeteroData` graph with ``species`` and
  ``reaction`` nodes and stoichiometric coefficients as edge weights.

Pure PyTorch / PyG; importing this module does not import scipy or cantera.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from koopman_graph.datasets.reaction.system import ReactionSystem, _require_system

SPECIES_NODE = "species"
"""Node type name for species in :func:`species_reaction_bipartite`."""

REACTION_NODE = "reaction"
"""Node type name for reactions in :func:`species_reaction_bipartite`."""

REACTANT_RELATION = (SPECIES_NODE, "reactant_of", REACTION_NODE)
"""Edge type from a reactant species to its reaction (weight = ``α``)."""

PRODUCT_RELATION = (REACTION_NODE, "produces", SPECIES_NODE)
"""Edge type from a reaction to a product species (weight = ``β``)."""


def species_influence_graph(
    system: ReactionSystem,
    *,
    self_loops: bool = False,
) -> tuple[Tensor, Tensor]:
    """Return the directed species influence graph and its edge weights.

    Parameters
    ----------
    system : ReactionSystem
        Reaction network.
    self_loops : bool, optional
        When ``True``, keep ``s → s`` edges (a species that is consumed or
        produced by a reaction it drives). Default is ``False``.

    Returns
    -------
    edge_index : Tensor
        ``long`` tensor with shape ``(2, num_edges)``; row 0 = source species
        ``s``, row 1 = target species ``t``. Edges are unique and sorted by
        ``(source, target)``.
    edge_weight : Tensor
        ``float32`` tensor with shape ``(num_edges,)``: the number of
        reactions that link ``s`` to ``t``.

    Notes
    -----
    An edge exists when some reaction ``j`` has ``α_js > 0`` and
    ``N_tj ≠ 0``. A zeroth-order source (``∅ → A``) has no reactants and so
    adds no edges.
    """
    _require_system(system)
    drives = (system.reactant_stoichiometry > 0).to(torch.float64)  # (R, S)
    changes = (system.stoichiometric_matrix != 0).to(torch.float64)  # (S, R)
    counts = drives.T @ changes.T  # (S_source, S_target)
    if not self_loops:
        counts.fill_diagonal_(0.0)
    source, target = counts.nonzero(as_tuple=True)
    edge_index = torch.stack([source, target]).to(dtype=torch.long)
    edge_weight = counts[source, target].to(dtype=torch.float32)
    return edge_index, edge_weight


def reaction_hyperedge_index(system: ReactionSystem) -> Tensor:
    """Return an undirected hyperedge per reaction over its participating species.

    Parameters
    ----------
    system : ReactionSystem
        Reaction network.

    Returns
    -------
    Tensor
        ``long`` incidence with shape ``(2, nnz)``: row 0 = species index,
        row 1 = reaction (hyperedge) index, sorted by ``(reaction,
        species)``. A species is incident to reaction ``j`` when it is a
        reactant or a product of ``j`` (also when it is a catalyst that
        appears on both sides).
    """
    _require_system(system)
    participates = (system.reactant_stoichiometry > 0) | (
        system.product_stoichiometry > 0
    )
    return _incidence(participates)


def reaction_directed_hyperedges(system: ReactionSystem) -> tuple[Tensor, Tensor]:
    """Return directed-hypergraph tail (reactant) and head (product) incidences.

    Parameters
    ----------
    system : ReactionSystem
        Reaction network.

    Returns
    -------
    tail_index : Tensor
        ``long`` incidence ``(2, nnz_tail)`` from reactant species (row 0)
        to reaction (row 1).
    head_index : Tensor
        ``long`` incidence ``(2, nnz_head)`` from product species (row 0)
        to reaction (row 1).

    Notes
    -----
    Source reactions have no tail entries and sink reactions no head
    entries, so the hyperedge count is ``system.num_reactions``, not
    ``max(index[1]) + 1``. Pass it explicitly to consumers that infer it.
    """
    _require_system(system)
    tail_index = _incidence(system.reactant_stoichiometry > 0)
    head_index = _incidence(system.product_stoichiometry > 0)
    return tail_index, head_index


def species_reaction_bipartite(system: ReactionSystem) -> HeteroData:
    """Return the bipartite species-reaction graph as PyG ``HeteroData``.

    Parameters
    ----------
    system : ReactionSystem
        Reaction network.

    Returns
    -------
    HeteroData
        Node types :data:`SPECIES_NODE` (``num_species`` nodes) and
        :data:`REACTION_NODE` (``num_reactions`` nodes). Edge type
        :data:`REACTANT_RELATION` carries ``edge_index`` (species → reaction)
        and ``edge_weight`` = reactant coefficient ``α``;
        :data:`PRODUCT_RELATION` carries reaction → species edges with
        ``edge_weight`` = product coefficient ``β``. The ``reaction`` nodes
        also store ``float64`` ``rate_constant`` with shape
        ``(num_reactions,)``; it stays ``float64`` because rate constants
        span many decades and would under- or overflow in ``float32``.
    """
    _require_system(system)
    data = HeteroData()
    data[SPECIES_NODE].num_nodes = system.num_species
    data[REACTION_NODE].num_nodes = system.num_reactions
    data[REACTION_NODE].rate_constant = system.rate_constants.clone()

    alpha = system.reactant_stoichiometry
    reaction, species = (alpha > 0).nonzero(as_tuple=True)
    data[REACTANT_RELATION].edge_index = torch.stack([species, reaction])
    data[REACTANT_RELATION].edge_weight = alpha[reaction, species].to(torch.float32)

    beta = system.product_stoichiometry
    reaction, species = (beta > 0).nonzero(as_tuple=True)
    data[PRODUCT_RELATION].edge_index = torch.stack([reaction, species])
    data[PRODUCT_RELATION].edge_weight = beta[reaction, species].to(torch.float32)
    return data


def _incidence(mask: Tensor) -> Tensor:
    """Turn a ``(num_reactions, num_species)`` mask into ``(2, nnz)`` incidence.

    Parameters
    ----------
    mask
        See signature.

    Returns
    -------
        See signature."""
    reaction, species = mask.nonzero(as_tuple=True)
    return torch.stack([species, reaction]).to(dtype=torch.long)
