#!/usr/bin/env python3
"""Write the compact reaction-network payloads under ``benchmarks/v0.16/data/``.

Maintainer tool. The specs below define every payload that a
``benchmarks/v0.16/`` manifest binds:

* ``reaction_michaelis_menten_v1.pt``: the packaged Michaelis-Menten card,
  regenerated on a longer grid;
* ``reaction_van_de_vusse_dilution_v1.pt``: the packaged Van de Vusse CSTR
  card with the dilution rate as the only input (isothermal at the nominal
  temperature);
* ``reaction_random_family_v1/``: twelve random element-balanced networks,
  four at each ``S`` in {8, 12, 16} (the first four network seeds whose
  species graph is connected and uses every species), one payload per
  network, plus ``index.json`` listing each member's SHA-256.

Payloads are :meth:`ReactionDataset.save` files (``torch.save``). The
manifests bind the committed bytes: ``dataset.sha256`` is the SHA-256 of the
payload file (or of ``index.json`` for the family). ``torch.save`` writes the
file stem into the archive, so the bytes depend on the file name, and
floating-point results can differ in the last bits across platforms, so a
regenerated payload is compared to the committed one numerically, not by
bytes.

Usage::

    python scripts/build_reaction_payloads.py            # write all payloads
    python scripts/build_reaction_payloads.py --check    # exit 1 if any differ

``--check`` regenerates every payload in memory and compares it with the
committed file (cards equal; times, concentrations, and controls within
``rtol=1e-5``), and checks the family index against the member files.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from koopman_graph.datasets.reaction import (
    VAN_DE_VUSSE_NOMINAL_TEMPERATURE_K,
    ReactionDataset,
    ReactionNetworkBenchmark,
    species_influence_graph,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "v0.16" / "data"

#: Samples per trajectory: the longest manifest horizon (100) plus the
#: history length (8) fits in every trajectory.
NUM_TIMESTEPS = 128
RTOL = 1e-5
ATOL = 1e-12

FAMILY_ID = "reaction_random_family_v1"
FAMILY_FORMAT = "koopman_graph.reaction_family"
FAMILY_FORMAT_VERSION = 1
FAMILY_SPECIES = (8, 12, 16)
FAMILY_NETWORKS_PER_SIZE = 4
#: Network seeds scanned per size before giving up.
FAMILY_SEED_LIMIT = 64
FAMILY_ELEMENTS = 3
FAMILY_MAX_ATOMS = 5
FAMILY_RATE_LOG10_RANGE = (-1.0, 1.0)
FAMILY_TRAJECTORIES = 16
#: Protocol of the family's shifts; the test-side networks are not
#: committed and are drawn with the same sampler settings.
FAMILY_PROTOCOL = {
    "unseen_n": {"train_num_species": list(FAMILY_SPECIES), "test_num_species": 32},
    "rewire": {"held_fixed": ["num_species", "num_reactions"], "varies": "seed"},
    "parameter_shift": {
        "train_rate_log10_range": list(FAMILY_RATE_LOG10_RANGE),
        "test_rate_log10_range": [1.0, 2.0],
    },
}


def michaelis_menten() -> ReactionDataset:
    """Generate the packaged Michaelis-Menten card on the longer grid.

    Returns
    -------
    ReactionDataset
        ``reaction_michaelis_menten_v1`` with :data:`NUM_TIMESTEPS` samples.
    """
    card = ReactionNetworkBenchmark.load_card("reaction_michaelis_menten_v1")
    return ReactionNetworkBenchmark.generate_from_card(
        card, num_timesteps=NUM_TIMESTEPS
    )


def van_de_vusse_dilution_card() -> dict[str, Any]:
    """Derive the CSTR card with the dilution rate as the only input.

    Returns
    -------
    dict
        The ``reaction_van_de_vusse_v1`` card, isothermal at the nominal
        temperature (``controls["fixed"]``), without its oracle section.
    """
    card = ReactionNetworkBenchmark.load_card("reaction_van_de_vusse_v1")
    card.pop("oracle", None)
    controls = copy.deepcopy(card["controls"])
    card.update(
        {
            "dataset_id": "reaction_van_de_vusse_dilution_v1",
            "scope": "Van de Vusse CSTR (A -> B -> C, 2 A -> D) with a seeded "
            "piecewise-constant dilution rate as the only control input; the "
            f"reactor is isothermal at {VAN_DE_VUSSE_NOMINAL_TEMPERATURE_K} K.",
            "controls": {
                "names": ["dilution_rate"],
                "ranges": {"dilution_rate": controls["ranges"]["dilution_rate"]},
                "hold_steps": controls["hold_steps"],
                "units": {"dilution_rate": controls["units"]["dilution_rate"]},
                "fixed": {"temperature_k": VAN_DE_VUSSE_NOMINAL_TEMPERATURE_K},
            },
            "limitations": "Isothermal (no energy balance). The dilution rate "
            "multiplies the state in the true dynamics, D (c_feed - c), so a "
            "model that treats it as an additive input is approximate. The "
            "species graph covers the reactions only, not the feed.",
            "provenance": "Derived from reaction_van_de_vusse_v1 by "
            "scripts/build_reaction_payloads.py "
            "(ReactionNetworkBenchmark.generate_from_card).",
        }
    )
    return card


def van_de_vusse_dilution() -> ReactionDataset:
    """Generate the dilution-input CSTR dataset.

    Returns
    -------
    ReactionDataset
        ``reaction_van_de_vusse_dilution_v1`` with :data:`NUM_TIMESTEPS`
        samples.
    """
    return ReactionNetworkBenchmark.generate_from_card(
        van_de_vusse_dilution_card(), num_timesteps=NUM_TIMESTEPS
    )


def family_member_id(num_species: int, network_seed: int) -> str:
    """Return the dataset id of one family member.

    Parameters
    ----------
    num_species : int
        Species count.
    network_seed : int
        Sampler seed.

    Returns
    -------
    str
        For example ``reaction_random_family_v1_s08_n0``.
    """
    return f"{FAMILY_ID}_s{num_species:02d}_n{network_seed}"


def family_card(num_species: int, network_seed: int) -> dict[str, Any]:
    """Derive the card of one random-family member.

    Parameters
    ----------
    num_species : int
        Species count.
    network_seed : int
        Sampler seed.

    Returns
    -------
    dict
        The ``reaction_random_network_v1`` card with the family's sampler
        settings. It has no ``oracle`` section: the equilibrium Newton solve
        of ``compute_card_oracle`` does not converge on every sampled
        network.
    """
    card = ReactionNetworkBenchmark.load_card("reaction_random_network_v1")
    card.pop("oracle", None)
    params = {
        "num_species": num_species,
        "num_elements": FAMILY_ELEMENTS,
        "num_reactions": 3 * num_species // 2,
        "max_atoms": FAMILY_MAX_ATOMS,
        "max_order": 2,
        "rate_log10_range": list(FAMILY_RATE_LOG10_RANGE),
        "seed": network_seed,
    }
    card.update(
        {
            "dataset_id": family_member_id(num_species, network_seed),
            "scope": f"Random element-balanced mass-action network, {num_species} "
            f"species, {params['num_reactions']} reactions, network seed "
            f"{network_seed}; one member of {FAMILY_ID}.",
            "generator": {"kind": "random_network", "params": params},
            "default_num_trajectories": FAMILY_TRAJECTORIES,
            "initial_ranges": {f"X{i}": [0.1, 1.0] for i in range(num_species)},
            "provenance": "Derived from reaction_random_network_v1 by "
            "scripts/build_reaction_payloads.py "
            "(ReactionNetworkBenchmark.generate_from_card).",
        }
    )
    system = ReactionNetworkBenchmark.generate_from_card(
        card, num_trajectories=1, num_timesteps=2
    ).system
    card["species"] = list(system.species)
    card["elements"] = list(system.elements or ())
    card["num_reactions"] = system.num_reactions
    card.pop("oracle_relaxation_time", None)
    return card


def family_member(num_species: int, network_seed: int) -> ReactionDataset:
    """Generate one random-family member.

    Parameters
    ----------
    num_species : int
        Species count.
    network_seed : int
        Sampler seed.

    Returns
    -------
    ReactionDataset
        The member with :data:`NUM_TIMESTEPS` samples.

    Raises
    ------
    ValueError
        If a concentration is not finite.
    """
    dataset = ReactionNetworkBenchmark.generate_from_card(
        family_card(num_species, network_seed), num_timesteps=NUM_TIMESTEPS
    )
    if not bool(torch.isfinite(dataset.concentrations).all()):
        msg = f"{dataset.dataset_id}: non-finite concentrations"
        raise ValueError(msg)
    return dataset


def is_connected(system: Any) -> bool:
    """Check that every species is in a reaction and the graph is connected.

    Parameters
    ----------
    system : ReactionSystem
        Sampled network.

    Returns
    -------
    bool
        ``True`` when no species is idle and the species influence graph,
        read as undirected, has one component.
    """
    used = (system.reactant_stoichiometry + system.product_stoichiometry).sum(0) > 0
    if not bool(used.all()):
        return False
    edge_index, _ = species_influence_graph(system)
    parent = list(range(system.num_species))
    for s, t in edge_index.t().tolist():
        parent[_root(parent, s)] = _root(parent, t)
    return len({_root(parent, i) for i in range(system.num_species)}) == 1


def _root(parent: list[int], i: int) -> int:
    """Return the union-find root of ``i``, halving the path.

    Parameters
    ----------
    parent : list of int
        Parent pointers, updated in place.
    i : int
        Node index.

    Returns
    -------
    int
        Root index.
    """
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def family_members() -> list[tuple[int, int]]:
    """List the family members.

    Returns
    -------
    list of tuple of int
        ``(num_species, network_seed)`` pairs: for each size, the first
        :data:`FAMILY_NETWORKS_PER_SIZE` seeds whose network passes
        :func:`is_connected`.

    Raises
    ------
    RuntimeError
        If fewer seeds pass within :data:`FAMILY_SEED_LIMIT`.
    """
    members = []
    for num_species in FAMILY_SPECIES:
        found = []
        for seed in range(FAMILY_SEED_LIMIT):
            system = ReactionNetworkBenchmark.generate_from_card(
                family_card(num_species, seed), num_trajectories=1, num_timesteps=2
            ).system
            if is_connected(system):
                found.append(seed)
            if len(found) == FAMILY_NETWORKS_PER_SIZE:
                break
        else:
            msg = (
                f"fewer than {FAMILY_NETWORKS_PER_SIZE} connected networks "
                f"at S={num_species}"
            )
            raise RuntimeError(msg)
        members += [(num_species, seed) for seed in found]
    return members


#: Single-file payloads: published file name -> builder.
SINGLE: dict[str, Callable[[], ReactionDataset]] = {
    "reaction_michaelis_menten_v1.pt": michaelis_menten,
    "reaction_van_de_vusse_dilution_v1.pt": van_de_vusse_dilution,
}


def _sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file.

    Parameters
    ----------
    path : Path
        File to hash.

    Returns
    -------
    str
        Hex digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def family_index(members: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the family ``index.json`` mapping.

    Parameters
    ----------
    members : list of dict
        One entry per member (file, SHA-256, size, seed).

    Returns
    -------
    dict
        Format tag, sampler settings, shift protocol, members, license.
    """
    return {
        "format": FAMILY_FORMAT,
        "format_version": FAMILY_FORMAT_VERSION,
        "family_id": FAMILY_ID,
        "num_timesteps": NUM_TIMESTEPS,
        "sampler": {
            "num_elements": FAMILY_ELEMENTS,
            "max_atoms": FAMILY_MAX_ATOMS,
            "max_order": 2,
            "num_reactions": "3 * num_species // 2",
            "rate_log10_range": list(FAMILY_RATE_LOG10_RANGE),
            "networks_per_size": FAMILY_NETWORKS_PER_SIZE,
            "member_rule": "the first network seeds whose species graph is "
            "connected and uses every species",
        },
        "protocol": FAMILY_PROTOCOL,
        "members": members,
        "license": "Apache-2.0",
    }


def write_all(out: Path) -> list[Path]:
    """Write every payload under ``out``.

    Parameters
    ----------
    out : Path
        Output directory.

    Returns
    -------
    list of Path
        Written files, ``index.json`` last.
    """
    written = []
    for name, build in SINGLE.items():
        written.append(build().save(out / name))
    family_dir = out / FAMILY_ID
    members = []
    for num_species, seed in family_members():
        dataset = family_member(num_species, seed)
        path = dataset.save(family_dir / f"{dataset.dataset_id}.pt")
        written.append(path)
        members.append(
            {
                "dataset_id": dataset.dataset_id,
                "file": path.name,
                "sha256": _sha256(path),
                "num_species": num_species,
                "num_reactions": dataset.system.num_reactions,
                "network_seed": seed,
            }
        )
    index = family_dir / "index.json"
    index.write_text(
        json.dumps(family_index(members), indent=2) + "\n", encoding="utf-8"
    )
    written.append(index)
    return written


def payloads_match(fresh: ReactionDataset, committed: ReactionDataset) -> list[str]:
    """Compare a regenerated payload with a committed one.

    Parameters
    ----------
    fresh : ReactionDataset
        Regenerated dataset.
    committed : ReactionDataset
        Loaded committed payload.

    Returns
    -------
    list of str
        Names of the fields that differ (empty when they match).
    """
    problems = []
    if fresh.dataset_id != committed.dataset_id:
        problems.append("dataset_id")
    if not _close_json(fresh.card, committed.card):
        problems.append("card")
    for field in ("times", "concentrations", "controls"):
        a, b = getattr(fresh, field), getattr(committed, field)
        if a is None or b is None:
            if (a is None) != (b is None):
                problems.append(field)
        elif a.shape != b.shape or not torch.allclose(a, b, rtol=RTOL, atol=ATOL):
            problems.append(field)
    if fresh.seed != committed.seed or fresh.simulator != committed.simulator:
        problems.append("seed/simulator")
    return problems


def _close_json(a: Any, b: Any) -> bool:
    """Compare JSON values, numbers within :data:`RTOL`.

    Parameters
    ----------
    a, b : Any
        Parsed JSON values.

    Returns
    -------
    bool
        ``True`` when the structures match and every number is close.
    """
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_close_json(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(_close_json(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, int | float) and isinstance(b, int | float):
        return abs(a - b) <= ATOL + RTOL * abs(b) or a == b
    return a == b


def check_all(out: Path) -> list[str]:
    """Compare every regenerated payload with the committed files.

    Parameters
    ----------
    out : Path
        Directory holding the committed payloads.

    Returns
    -------
    list of str
        One line per problem (empty when everything matches).
    """
    problems = []
    for name, build in SINGLE.items():
        path = out / name
        if not path.is_file():
            problems.append(f"{name}: missing")
            continue
        committed = ReactionDataset.load(path)
        problems += [f"{name}: {p}" for p in payloads_match(build(), committed)]
    family_dir = out / FAMILY_ID
    index_path = family_dir / "index.json"
    if not index_path.is_file():
        return [*problems, f"{FAMILY_ID}/index.json: missing"]
    index = json.loads(index_path.read_text(encoding="utf-8"))
    listed = {m["file"]: m for m in index["members"]}
    expected = {f"{family_member_id(s, n)}.pt": (s, n) for s, n in family_members()}
    if set(listed) != set(expected):
        problems.append(f"{FAMILY_ID}/index.json: members differ from the specs")
    on_disk = {p.name for p in family_dir.glob("*.pt")}
    if on_disk != set(expected):
        problems.append(f"{FAMILY_ID}: payload files differ from the specs")
    for name, (num_species, seed) in expected.items():
        path = family_dir / name
        if not path.is_file():
            continue
        if name in listed and listed[name]["sha256"] != _sha256(path):
            problems.append(f"{FAMILY_ID}/{name}: sha256 differs from index.json")
        fresh = family_member(num_species, seed)
        problems += [
            f"{FAMILY_ID}/{name}: {p}"
            for p in payloads_match(fresh, ReactionDataset.load(path))
        ]
    if index.get("protocol") != FAMILY_PROTOCOL:
        problems.append(f"{FAMILY_ID}/index.json: protocol differs")
    return problems


def main(argv: list[str] | None = None) -> int:
    """Write or check the payloads.

    Parameters
    ----------
    argv : list of str or None, optional
        Command-line arguments (default ``sys.argv[1:]``).

    Returns
    -------
    int
        Exit status (``1`` when ``--check`` finds a difference).
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="compare, do not write")
    parser.add_argument(
        "--out", type=Path, default=DATA_DIR, help=f"output directory ({DATA_DIR})"
    )
    args = parser.parse_args(argv)
    if args.check:
        problems = check_all(args.out)
        for problem in problems:
            print(problem, file=sys.stderr)
        return 1 if problems else 0
    for path in write_all(args.out):
        print(f"{path}  {_sha256(path)}  {path.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
