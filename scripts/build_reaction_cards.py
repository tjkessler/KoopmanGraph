#!/usr/bin/env python3
"""Write the FAIR JSON cards for ``koopman_graph.datasets.reaction``.

Maintainer tool. The specs below define every frozen ``reaction_*_v1``
dataset. Running this script computes each card's ``oracle`` section with
:func:`koopman_graph.datasets.reaction.benchmark.compute_card_oracle` and
writes ``src/koopman_graph/datasets/reaction/data/<dataset_id>.json``.
The committed JSON is the source of truth at runtime;
``tests/datasets/test_reaction_benchmark.py`` recomputes the oracles to
catch drift. The Cantera card needs the ``[cantera]`` extra (Cantera 3.2).

Usage::

    python scripts/build_reaction_cards.py            # write all cards
    python scripts/build_reaction_cards.py --check    # exit 1 if any differ

``--check`` compares numbers with a tolerance (oracle values come from
eigensolvers, Newton, RK4, and Cantera, whose last digits vary across
platforms and BLAS builds) and also flags ``reaction_*.json`` files in the
data directory that no spec produces.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from koopman_graph.datasets.reaction.benchmark import (
    compute_card_oracle,
    system_from_card,
)

DATA_DIR = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "koopman_graph"
    / "datasets"
    / "reaction"
    / "data"
)

_LICENSE = "Apache-2.0"
# SHA-256 of h2o2.yaml in the Cantera 3.2.0 wheels (issue #65).
_H2O2_SHA256 = "0efc6c52862741a29e0c29b65d979c7d8cb409db5282bca83b9c5437b3d8c8d4"
_UNITS = {"time": "s", "concentration": "mol/L"}
_GENERATED = "In-repo mass-action generator (koopman_graph.datasets.reaction)."

SPECS: list[dict[str, Any]] = [
    {
        "dataset_id": "reaction_chain_v1",
        "scope": "Linear first-order chain S0 -> S1 -> S2 -> S3 (rates 1, "
        "0.5, 0.25); exact matrix-exponential trajectories with closed-form "
        "dynamic spectrum {-0.25, -0.5, -1} plus one conservation zero mode.",
        "generator": {
            "kind": "catalog",
            "name": "first_order_chain",
            "params": {"num_species": 4},
        },
        "simulator": {"kind": "expm"},
        "time_step": 0.25,
        "default_num_timesteps": 64,
        "default_num_trajectories": 32,
        "default_seed": 0,
        "initial_ranges": {
            "S0": [0.5, 1.5],
            "S1": [0.0, 0.2],
            "S2": [0.0, 0.2],
            "S3": [0.0, 0.2],
        },
        "transform": {"kind": "log_concentration", "floor": 1e-8},
        "graph_encoding": "species_influence",
        "limitations": "Linear and closed; every mode is a pure decay. Use as "
        "the exact-spectrum sanity check, not as a nonlinear benchmark.",
    },
    {
        "dataset_id": "reaction_michaelis_menten_v1",
        "scope": "Mass-action Michaelis-Menten E + S <-> ES -> E + P with two "
        "conservation laws (enzyme, substrate backbone) and a boundary "
        "equilibrium.",
        "generator": {"kind": "catalog", "name": "michaelis_menten", "params": {}},
        "simulator": {"kind": "rk4", "substeps": 50},
        "time_step": 0.5,
        "default_num_timesteps": 64,
        "default_num_trajectories": 32,
        "default_seed": 0,
        "initial_ranges": {
            "E": [0.05, 0.15],
            "S": [0.5, 1.5],
            "ES": [0.0, 0.05],
            "P": [0.0, 0.1],
        },
        "transform": {"kind": "log_concentration", "floor": 1e-8},
        "graph_encoding": "reaction_hyperedge",
        "limitations": "Irreversible catalysis: trajectories end on the "
        "boundary S = ES = 0, where the log transform saturates at the floor.",
    },
    {
        "dataset_id": "reaction_brusselator_v1",
        "scope": "Open Brusselator (a = 1, b = 3) with an unstable focus and a "
        "limit cycle; tests oscillatory, non-decaying dynamics.",
        "generator": {"kind": "catalog", "name": "brusselator", "params": {}},
        "simulator": {"kind": "rk4", "substeps": 10},
        "time_step": 0.1,
        "default_num_timesteps": 128,
        "default_num_trajectories": 32,
        "default_seed": 0,
        "initial_ranges": {"X": [0.5, 1.5], "Y": [2.0, 4.0]},
        "transform": None,
        "graph_encoding": "species_influence",
        "limitations": "Open system without conservation laws; the oracle "
        "fixed point is unstable, so its timescales are negative (growth).",
    },
    {
        "dataset_id": "reaction_random_network_v1",
        "scope": "Seeded random element-balanced mass-action network (6 "
        "species, 2 elements, 8 reactions, order <= 2).",
        "generator": {
            "kind": "random_network",
            "params": {
                "num_species": 6,
                "num_elements": 2,
                "num_reactions": 8,
                "max_atoms": 4,
                "max_order": 2,
                "rate_log10_range": [-1.0, 1.0],
                "seed": 0,
            },
        },
        "simulator": {"kind": "rk4", "substeps": 20},
        "time_step": 0.1,
        "default_num_timesteps": 64,
        "default_num_trajectories": 32,
        "default_seed": 0,
        "initial_ranges": {f"X{i}": [0.1, 1.0] for i in range(6)},
        "oracle_relaxation_time": 200.0,
        "transform": {"kind": "log_concentration", "floor": 1e-8},
        "graph_encoding": "species_influence",
        "limitations": "Synthetic chemistry with no physical species; rates "
        "are log-uniform draws, not measured kinetics.",
    },
    {
        "dataset_id": "reaction_van_de_vusse_v1",
        "scope": "Van de Vusse CSTR (A -> B -> C, 2 A -> D) with seeded "
        "piecewise-constant reactor temperature and dilution-rate inputs "
        "stored as global control_inputs.",
        "generator": {"kind": "cstr", "params": {}},
        "simulator": {"kind": "cstr", "substeps": 20},
        "time_step": 0.01,
        "default_num_timesteps": 100,
        "default_num_trajectories": 32,
        "default_seed": 0,
        "initial_ranges": {
            "A": [0.5, 3.0],
            "B": [0.0, 1.5],
            "C": [0.0, 1.5],
            "D": [0.0, 0.5],
        },
        "controls": {
            "names": ["temperature_k", "dilution_rate"],
            "ranges": {"temperature_k": [380.0, 400.0], "dilution_rate": [5.0, 30.0]},
            "hold_steps": 10,
            "units": {"temperature_k": "K", "dilution_rate": "1/h"},
        },
        "transform": None,
        "graph_encoding": "species_influence",
        "units": {"time": "h", "concentration": "mol/L"},
        "references": [
            "K.-U. Klatt and S. Engell, Comput. Chem. Eng. 22(4-5), 491-502, "
            "1998 (kinetics)",
            "H. Chen, A. Kremling and F. Allgower, Proc. 3rd ECC, 1995 "
            "(operating point)",
        ],
        "limitations": "Reactor temperature is a direct input (no energy "
        "balance); the graph uses the reaction network only, not the feed.",
    },
    {
        "dataset_id": "reaction_cantera_h2o2_v1",
        "scope": "Isothermal constant-volume H2/O2/Ar chemistry at 1200 K from "
        "Cantera's bundled h2o2.yaml (10 species, 29 reactions); "
        "trajectories come from Cantera's ReactorNet.",
        "generator": {"kind": "cantera", "params": {}},
        "simulator": {"kind": "cantera"},
        "time_step": 2e-5,
        "default_num_timesteps": 50,
        "default_num_trajectories": 16,
        "default_seed": 0,
        "cantera": {
            "mechanism": "h2o2.yaml",
            "sha256": _H2O2_SHA256,
            "cantera_version": "3.2.0",
            "temperature_k": 1200.0,
            "pressure_pa": 101325.0,
            "reactor": "const_volume",
            "isothermal": True,
            "state_dependent": "freeze",
            "reference_mole_fractions": "H2:2, O2:1, AR:4",
            "initial_mole_fraction_ranges": {
                "H2": [1.0, 3.0],
                "O2": [0.5, 1.5],
                "AR": [3.0, 5.0],
            },
            "note": "Cantera's bundled input files are provided for "
            "illustration only; they are not validated kinetic models for "
            "quantitative use.",
            "citation_doi": "10.5281/zenodo.17620923",
            "license": "Cantera is BSD-3-Clause",
        },
        "transform": {"kind": "log_concentration", "floor": 1e-12},
        "graph_encoding": "species_influence",
        "limitations": "Illustration-only mechanism; the graph and oracle use "
        "the mass-action conversion with the falloff reaction frozen at the "
        "reference state, while trajectories are exact Cantera integrations.",
    },
]


def build_card(spec: dict[str, Any]) -> dict[str, Any]:
    """Return the full card for ``spec`` with species, provenance, and oracle.

    Parameters
    ----------
    spec : dict
        One entry of :data:`SPECS`.

    Returns
    -------
    dict
        Card ready to serialize.
    """
    card = {
        "dataset_id": spec["dataset_id"],
        "version": spec["dataset_id"].rsplit("_", 1)[-1],
        **{k: v for k, v in spec.items() if k != "dataset_id"},
    }
    card.setdefault("units", dict(_UNITS))
    card.setdefault("controls", None)
    card.setdefault("cantera", None)
    card["format"] = (
        "Per-trajectory GraphSnapshotSequence (one node per species, float32 "
        "features of shape (num_species, 1), timestamps, optional global "
        "control_inputs); raw float64 concentrations in ReactionDataset."
    )
    card["provenance"] = (
        _GENERATED
        if card["cantera"] is None
        else "Cantera ReactorNet integration of the bundled mechanism; see the "
        "cantera section for version and SHA-256."
    )
    card["license"] = _LICENSE
    system = system_from_card(card)
    card["species"] = list(system.species)
    card["elements"] = None if system.elements is None else list(system.elements)
    card["num_reactions"] = system.num_reactions
    card["oracle"] = compute_card_oracle(card)
    return card


# Tolerances for ``--check``; they match tests/datasets/test_reaction_benchmark.py.
_RTOL, _ATOL = 1e-8, 1e-12
_CANTERA_RTOL, _CANTERA_ATOL = 1e-6, 1e-18


def cards_match(a: Any, b: Any, *, rtol: float, atol: float) -> bool:
    """Compare two parsed cards, with a tolerance on floating-point leaves.

    Parameters
    ----------
    a, b : Any
        Parsed JSON values.
    rtol, atol : float
        Relative and absolute tolerance for floats.

    Returns
    -------
    bool
        ``True`` when the structures are equal and every float pair is close.
    """
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(
            cards_match(a[k], b[k], rtol=rtol, atol=atol) for k in a
        )
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(
            cards_match(x, y, rtol=rtol, atol=atol) for x, y in zip(a, b, strict=True)
        )
    numeric = (int, float)
    if (
        isinstance(a, numeric)
        and isinstance(b, numeric)
        and not isinstance(a, bool)
        and not isinstance(b, bool)
    ):
        return math.isclose(a, b, rel_tol=rtol, abs_tol=atol)
    return type(a) is type(b) and a == b


def main(argv: list[str] | None = None) -> int:
    """Write (or check) every card.

    Parameters
    ----------
    argv : list of str or None, optional
        Command-line arguments.

    Returns
    -------
    int
        Exit status (``1`` when ``--check`` finds a difference).
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="compare, do not write")
    args = parser.parse_args(argv)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    stale = []
    for spec in SPECS:
        card = build_card(spec)
        path = DATA_DIR / f"{spec['dataset_id']}.json"
        if args.check:
            cantera = card["cantera"] is not None
            if not path.is_file() or not cards_match(
                json.loads(path.read_text(encoding="utf-8")),
                card,
                rtol=_CANTERA_RTOL if cantera else _RTOL,
                atol=_CANTERA_ATOL if cantera else _ATOL,
            ):
                stale.append(path.name)
        else:
            path.write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")
            print(f"wrote {path.relative_to(DATA_DIR.parents[3])}")
    expected = {f"{spec['dataset_id']}.json" for spec in SPECS}
    orphans = sorted(
        path.name
        for path in DATA_DIR.glob("reaction_*.json")
        if path.name not in expected
    )
    if stale:
        print("stale cards: " + ", ".join(stale), file=sys.stderr)
    if orphans:
        print("cards without a spec: " + ", ".join(orphans), file=sys.stderr)
    return 1 if stale or orphans else 0


if __name__ == "__main__":
    raise SystemExit(main())
