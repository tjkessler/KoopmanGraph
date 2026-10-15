"""Seeded sampler for random, element-balanced mass-action networks.

:func:`sample_random_network` builds the composition first, so every
sampled reaction conserves elements by construction:

1. The first ``num_elements`` species are the elements themselves (unit
   composition vectors).
2. Each further species is the union of two existing species. Its
   composition is the sum of theirs, capped at ``max_atoms`` atoms and
   unique, which guarantees at least one balanced association reaction.
3. Every reaction ``reactants → products`` with equal element totals
   becomes a candidate, where the reactants are a multiset of at most
   ``max_order`` species (the reaction order) and the products a multiset
   of at most two. Compositions are unique, so ``1 → 1`` isomerizations
   never balance, and ``max_order=1`` gives a linear network of
   dissociations. ``num_reactions`` candidates are drawn without
   replacement and given log-uniform rate constants.

All randomness comes from one ``torch.Generator`` seeded with ``seed``, and
candidates are enumerated in a fixed order, so a seed reproduces the
network exactly. The seed and every sampler argument are returned with the
network.

Pure PyTorch; importing this module does not import scipy or cantera.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from koopman_graph.datasets.reaction.system import ReactionSystem

_MAX_SPECIES = 32
_MAX_PRODUCTS = 2
_MAX_COMPOSITION_TRIES = 1000


@dataclass(frozen=True)
class SampledReactionNetwork:
    """A sampled network plus the seed and parameters that reproduce it.

    Attributes
    ----------
    system : ReactionSystem
        Element-balanced mass-action system (``elements`` are ``e0``,
        ``e1``, ...).
    seed : int
        Seed passed to :func:`sample_random_network`.
    parameters : dict
        Every sampler argument (including ``seed``), JSON-compatible.
    num_candidates : int
        Number of balanced candidate reactions the draw was made from.
    """

    system: ReactionSystem
    seed: int
    parameters: dict[str, Any]
    num_candidates: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize the network and its provenance.

        Returns
        -------
        dict
            ``{"system": ..., "seed": ..., "parameters": ...,
            "num_candidates": ...}`` with JSON-compatible values.
        """
        return {
            "system": self.system.to_dict(),
            "seed": self.seed,
            "parameters": dict(self.parameters),
            "num_candidates": self.num_candidates,
        }


def sample_random_network(
    *,
    num_species: int = 6,
    num_elements: int = 2,
    num_reactions: int = 8,
    max_atoms: int = 4,
    max_order: int = 2,
    rate_log10_range: tuple[float, float] = (-1.0, 1.0),
    seed: int = 0,
) -> SampledReactionNetwork:
    """Sample a random element-balanced mass-action network.

    Parameters
    ----------
    num_species : int, optional
        Total species count, greater than ``num_elements`` and at most 32.
        Default is ``6``.
    num_elements : int, optional
        Number of elements; the first ``num_elements`` species are the bare
        elements. Default is ``2``.
    num_reactions : int, optional
        Reactions to draw from the balanced candidates. Default is ``8``.
    max_atoms : int, optional
        Cap on the total atom count of any species (>= 2). Default is ``4``.
    max_order : int, optional
        Maximum reaction order (reactant species counted with
        multiplicity), ``1`` or ``2``. Products always have at most two
        species, so ``1`` yields a linear network. Default is ``2``.
    rate_log10_range : tuple of float, optional
        ``(low, high)`` bounds of ``log10 k``; rates are log-uniform.
        Default is ``(-1.0, 1.0)``.
    seed : int, optional
        Seed of the ``torch.Generator`` that drives every draw. Default is
        ``0``.

    Returns
    -------
    SampledReactionNetwork
        The system, its seed and parameters, and the candidate count.

    Raises
    ------
    ValueError
        If arguments are out of range, ``max_atoms`` leaves too few distinct
        compositions, or there are fewer balanced candidates than
        ``num_reactions``.
    """
    _check_int("num_elements", num_elements, low=1, high=_MAX_SPECIES - 1)
    _check_int("num_species", num_species, low=num_elements + 1, high=_MAX_SPECIES)
    _check_int("num_reactions", num_reactions, low=1, high=None)
    _check_int("max_atoms", max_atoms, low=2, high=None)
    _check_int("max_order", max_order, low=1, high=2)
    _check_int("seed", seed, low=0, high=None)
    low, high = (float(v) for v in rate_log10_range)
    if not (math.isfinite(low) and math.isfinite(high)) or low > high:
        msg = (
            "rate_log10_range must be finite with low <= high, got "
            f"{rate_log10_range!r}"
        )
        raise ValueError(msg)

    generator = torch.Generator().manual_seed(seed)
    compositions = _sample_compositions(
        num_species=num_species,
        num_elements=num_elements,
        max_atoms=max_atoms,
        generator=generator,
    )
    candidates = _balanced_candidates(compositions, max_order=max_order)
    if len(candidates) < num_reactions:
        msg = (
            f"only {len(candidates)} balanced candidate reactions for "
            f"num_reactions={num_reactions}; raise num_species / max_order or "
            "lower num_reactions"
        )
        raise ValueError(msg)
    order = torch.randperm(len(candidates), generator=generator)[:num_reactions]
    chosen = [candidates[int(i)] for i in torch.sort(order).values.tolist()]
    exponents = low + (high - low) * torch.rand(
        num_reactions, generator=generator, dtype=torch.float64
    )

    reactants = torch.zeros(num_reactions, num_species, dtype=torch.float64)
    products = torch.zeros(num_reactions, num_species, dtype=torch.float64)
    for row, (lhs, rhs) in enumerate(chosen):
        for s in lhs:
            reactants[row, s] += 1.0
        for s in rhs:
            products[row, s] += 1.0
    species = tuple(f"X{i}" for i in range(num_species))
    names = tuple(
        f"{_side(lhs, species)} -> {_side(rhs, species)}" for lhs, rhs in chosen
    )
    parameters: dict[str, Any] = {
        "num_species": num_species,
        "num_elements": num_elements,
        "num_reactions": num_reactions,
        "max_atoms": max_atoms,
        "max_order": max_order,
        "rate_log10_range": [low, high],
        "seed": seed,
    }
    system = ReactionSystem(
        species=species,
        reactant_stoichiometry=reactants,
        product_stoichiometry=products,
        rate_constants=torch.pow(10.0, exponents),
        element_matrix=compositions.T.contiguous(),
        elements=tuple(f"e{i}" for i in range(num_elements)),
        reaction_names=names,
        name=f"random_network_seed{seed}",
    )
    return SampledReactionNetwork(
        system=system,
        seed=seed,
        parameters=parameters,
        num_candidates=len(candidates),
    )


def _sample_compositions(
    *,
    num_species: int,
    num_elements: int,
    max_atoms: int,
    generator: torch.Generator,
) -> Tensor:
    """Return unique species compositions ``(num_species, num_elements)``.

    Parameters
    ----------
    num_species
        See signature.
    num_elements
        See signature.
    max_atoms
        See signature.
    generator
        See signature.

    Returns
    -------
        See signature."""
    rows: list[tuple[int, ...]] = [
        tuple(int(i == e) for i in range(num_elements)) for e in range(num_elements)
    ]
    seen = set(rows)
    tries = 0
    while len(rows) < num_species:
        tries += 1
        if tries > _MAX_COMPOSITION_TRIES:
            msg = (
                f"could not build {num_species} distinct species with "
                f"max_atoms={max_atoms}; raise max_atoms or lower num_species"
            )
            raise ValueError(msg)
        a, b = torch.randint(len(rows), (2,), generator=generator).tolist()
        combined = tuple(x + y for x, y in zip(rows[a], rows[b], strict=True))
        if sum(combined) > max_atoms or combined in seen:
            continue
        rows.append(combined)
        seen.add(combined)
    return torch.tensor(rows, dtype=torch.float64)


def _balanced_candidates(
    compositions: Tensor,
    *,
    max_order: int,
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Enumerate element-balanced ``(reactants, products)`` multiset pairs.

    Parameters
    ----------
    compositions
        See signature.
    max_order
        See signature.

    Returns
    -------
        See signature."""
    num_species = int(compositions.shape[0])
    comp = [tuple(int(v) for v in row) for row in compositions.tolist()]
    sides: list[tuple[int, ...]] = []
    for size in range(1, _MAX_PRODUCTS + 1):
        sides.extend(itertools.combinations_with_replacement(range(num_species), size))
    totals = {
        side: tuple(sum(comp[s][e] for s in side) for e in range(len(comp[0])))
        for side in sides
    }
    return [
        (lhs, rhs)
        for lhs in sides
        if len(lhs) <= max_order
        for rhs in sides
        if lhs != rhs and totals[lhs] == totals[rhs]
    ]


def _side(side: tuple[int, ...], species: tuple[str, ...]) -> str:
    """Format a multiset of species indices as ``"2 X0 + X3"``.

    Parameters
    ----------
    side
        See signature.
    species
        See signature.

    Returns
    -------
        See signature."""
    parts = []
    for s in sorted(set(side)):
        count = side.count(s)
        parts.append(species[s] if count == 1 else f"{count} {species[s]}")
    return " + ".join(parts)


def _check_int(name: str, value: Any, *, low: int, high: int | None) -> None:
    """Raise ``ValueError`` unless ``value`` is an int in ``[low, high]``.

    Parameters
    ----------
    name
        See signature.
    value
        See signature.
    low
        See signature.
    high
        See signature."""
    bad = isinstance(value, bool) or not isinstance(value, int) or value < low
    if not bad and high is not None and value > high:
        bad = True
    if bad:
        bound = f">= {low}" if high is None else f"in [{low}, {high}]"
        msg = f"{name} must be an int {bound}, got {value!r}"
        raise ValueError(msg)
