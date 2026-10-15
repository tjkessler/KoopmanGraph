"""Tests for :mod:`koopman_graph.datasets.reaction.benchmark`."""

from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from tests.helpers import REPO_ROOT

from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.datasets.reaction import (
    PAYLOAD_FORMAT_VERSION,
    ReactionDataset,
    ReactionNetworkBenchmark,
    check_element_balance,
    compute_card_oracle,
    integrate_reaction_system,
    oracle_spectrum,
)

_NON_CANTERA = (
    "reaction_brusselator_v1",
    "reaction_chain_v1",
    "reaction_michaelis_menten_v1",
    "reaction_random_network_v1",
    "reaction_van_de_vusse_v1",
)
_CLOSED = (
    "reaction_chain_v1",
    "reaction_michaelis_menten_v1",
    "reaction_random_network_v1",
)
_REQUIRED_CARD_KEYS = {
    "dataset_id",
    "version",
    "scope",
    "generator",
    "simulator",
    "time_step",
    "default_num_timesteps",
    "default_num_trajectories",
    "default_seed",
    "transform",
    "graph_encoding",
    "species",
    "num_reactions",
    "oracle",
    "units",
    "format",
    "provenance",
    "license",
    "limitations",
}


def _close(a: object, b: object, *, rtol: float = 1e-8, atol: float = 1e-12) -> bool:
    """Compare nested JSON values with float tolerance.

    Parameters
    ----------
    a
        See signature.
    b
        See signature.
    rtol
        See signature.
    atol
        See signature.

    Returns
    -------
        See signature."""
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(
            _close(x, y, rtol=rtol, atol=atol) for x, y in zip(a, b, strict=True)
        )
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=rtol, abs_tol=atol)
    return a == b


def test_packaged_cards_are_listed_and_complete() -> None:
    ids = ReactionNetworkBenchmark.dataset_ids()
    assert set(ids) == {*_NON_CANTERA, "reaction_cantera_h2o2_v1"}
    for dataset_id in ids:
        card = ReactionNetworkBenchmark.load_card(dataset_id)
        assert set(card) >= _REQUIRED_CARD_KEYS, dataset_id
        assert card["dataset_id"] == dataset_id
        assert card["version"] == "v1"
        assert card["license"] == "Apache-2.0"
        assert card["graph_encoding"] in {"species_influence", "reaction_hyperedge"}
    with pytest.raises(ValueError, match="unknown reaction dataset"):
        ReactionNetworkBenchmark.load_card("reaction_nope_v1")


@pytest.mark.parametrize("dataset_id", _NON_CANTERA)
def test_card_oracle_matches_recomputation(dataset_id: str) -> None:
    card = ReactionNetworkBenchmark.load_card(dataset_id)
    fresh = compute_card_oracle(card)
    for key, value in card["oracle"].items():
        assert _close(fresh[key], value), f"{dataset_id}: oracle[{key!r}] drifted"


@pytest.mark.cantera
def test_cantera_card_oracle_and_provenance() -> None:
    from koopman_graph.datasets.reaction import (
        load_cantera_mechanism,
        mechanism_sha256,
        resolve_mechanism_path,
    )

    card = ReactionNetworkBenchmark.load_card("reaction_cantera_h2o2_v1")
    cantera = card["cantera"]
    assert cantera["sha256"] == mechanism_sha256(resolve_mechanism_path("h2o2.yaml"))
    assert "illustration only" in cantera["note"]
    fresh = compute_card_oracle(card)
    for key, value in card["oracle"].items():
        assert _close(fresh[key], value, rtol=1e-6, atol=1e-18), key
    mechanism = load_cantera_mechanism(
        cantera["mechanism"],
        temperature_k=cantera["temperature_k"],
        mole_fractions=cantera["reference_mole_fractions"],
    )
    assert mechanism.system.species == tuple(card["species"])
    dataset = ReactionNetworkBenchmark.generate(
        "reaction_cantera_h2o2_v1", num_trajectories=2, num_timesteps=6
    )
    assert dataset.concentrations.shape == (2, 6, 10)
    assert dataset.simulator == "cantera"
    assert dataset.transform is not None
    assert dataset.transform.floor == 1e-12


def test_chain_oracle_is_closed_form() -> None:
    oracle = ReactionNetworkBenchmark.load_card("reaction_chain_v1")["oracle"]
    assert oracle["num_conservation_laws"] == 1
    assert _close(oracle["dynamic_eigenvalues_real"], [-0.25, -0.5, -1.0])
    assert _close(oracle["timescales"], [4.0, 2.0, 1.0])


def test_random_network_card_marks_drained_mode_infinite() -> None:
    oracle = ReactionNetworkBenchmark.load_card("reaction_random_network_v1")["oracle"]
    assert oracle["state"].endswith("after_relaxation")
    assert oracle["timescales"][0] is None  # numerically zero drained mode
    assert all(t is None or t > 0 for t in oracle["timescales"])


@pytest.mark.parametrize("dataset_id", _NON_CANTERA)
def test_generation_is_deterministic_by_seed(dataset_id: str) -> None:
    a = ReactionNetworkBenchmark.generate(
        dataset_id, num_trajectories=3, num_timesteps=8
    )
    b = ReactionNetworkBenchmark.generate(
        dataset_id, num_trajectories=3, num_timesteps=8
    )
    c = ReactionNetworkBenchmark.generate(
        dataset_id, num_trajectories=3, num_timesteps=8, seed=1
    )
    assert torch.equal(a.concentrations, b.concentrations)
    assert not torch.equal(a.concentrations, c.concentrations)
    if a.controls is not None:
        assert b.controls is not None
        assert torch.equal(a.controls, b.controls)
    assert a.concentrations.dtype == torch.float64
    assert bool(torch.isfinite(a.concentrations).all())
    assert a.times.tolist() == pytest.approx(
        [a.card["time_step"] * i for i in range(8)]
    )


def test_defaults_come_from_card() -> None:
    dataset = ReactionNetworkBenchmark.generate("reaction_chain_v1")
    card = dataset.card
    assert dataset.concentrations.shape == (
        card["default_num_trajectories"],
        card["default_num_timesteps"],
        4,
    )
    assert dataset.seed == card["default_seed"]
    assert dataset.simulator == "expm"


@pytest.mark.parametrize("dataset_id", _CLOSED)
def test_closed_datasets_conserve_elements(dataset_id: str) -> None:
    dataset = ReactionNetworkBenchmark.generate(dataset_id, num_trajectories=4)
    system = dataset.system
    assert system.element_matrix is not None
    assert check_element_balance(system).balanced
    totals = dataset.concentrations @ system.element_matrix.T
    drift = (totals - totals[:, :1]).abs().max()
    assert float(drift) < 1e-9


def test_initial_conditions_respect_card_ranges() -> None:
    dataset = ReactionNetworkBenchmark.generate(
        "reaction_michaelis_menten_v1", num_trajectories=50
    )
    ranges = dataset.card["initial_ranges"]
    initial = dataset.concentrations[:, 0]
    for s, name in enumerate(dataset.system.species):
        lo, hi = ranges[name]
        assert float(initial[:, s].min()) >= lo
        assert float(initial[:, s].max()) <= hi


def test_trajectories_follow_the_card_simulator() -> None:
    dataset = ReactionNetworkBenchmark.generate(
        "reaction_brusselator_v1", num_trajectories=2, num_timesteps=12
    )
    rerun = integrate_reaction_system(
        dataset.system,
        dataset.concentrations[:, 0],
        dataset.times,
        method="rk4",
        substeps=dataset.card["simulator"]["substeps"],
    ).permute(1, 0, 2)
    assert torch.allclose(rerun, dataset.concentrations)


def test_long_runs_relax_to_card_equilibrium() -> None:
    card = ReactionNetworkBenchmark.load_card("reaction_michaelis_menten_v1")
    dataset = ReactionNetworkBenchmark.generate(
        "reaction_michaelis_menten_v1", num_trajectories=1, num_timesteps=400
    )
    final = dataset.concentrations[0, -1]
    # Boundary equilibrium: all substrate converted, enzyme freed.
    assert float(final[1]) < 1e-6 and float(final[2]) < 1e-6
    spectrum = oracle_spectrum(dataset.system, final)
    assert spectrum.num_conservation_laws == card["oracle"]["num_conservation_laws"]


def test_sequence_layout_and_encodings() -> None:
    dataset = ReactionNetworkBenchmark.generate(
        "reaction_michaelis_menten_v1", num_trajectories=2, num_timesteps=5
    )
    sequence = dataset.sequence(1)
    assert isinstance(sequence, GraphSnapshotSequence)
    assert sequence.num_timesteps == 5
    snapshot = sequence[0]
    assert snapshot.x.shape == (4, 1)
    assert snapshot.x.dtype == torch.float32
    assert snapshot.hyperedge_index.shape[0] == 2  # reaction_hyperedge card
    assert dataset.transform is not None
    expected = torch.log(dataset.concentrations[1, 0].clamp(min=1e-8)).to(torch.float32)
    assert torch.allclose(snapshot.x.squeeze(-1), expected)
    assert torch.allclose(sequence.timestamps, dataset.times)
    raw = ReactionNetworkBenchmark.generate(
        "reaction_brusselator_v1", num_trajectories=1, num_timesteps=3
    )
    assert raw.transform is None
    assert torch.allclose(
        raw.sequence(0)[2].x.squeeze(-1),
        raw.concentrations[0, 2].to(torch.float32),
    )
    assert not hasattr(raw.sequence(0)[0], "hyperedge_index")
    with pytest.raises(IndexError, match="out of range"):
        raw.sequence(5)
    plain = dataset.sequence(1, graph_encoding="species_influence")
    assert not hasattr(plain[0], "hyperedge_index")
    assert torch.equal(plain[0].x, snapshot.x)
    with_hyper = raw.sequence(0, graph_encoding="reaction_hyperedge")
    assert with_hyper[0].hyperedge_index.shape[0] == 2
    with pytest.raises(ValueError, match="graph_encoding must be"):
        raw.sequence(0, graph_encoding="bipartite")  # type: ignore[arg-type]


def test_cstr_controls_are_piecewise_constant_and_in_range() -> None:
    dataset = ReactionNetworkBenchmark.generate(
        "reaction_van_de_vusse_v1", num_trajectories=2, num_timesteps=25
    )
    assert dataset.controls is not None
    assert dataset.controls.shape == (2, 25, 2)
    spec = dataset.card["controls"]
    for k, name in enumerate(spec["names"]):
        lo, hi = spec["ranges"][name]
        values = dataset.controls[..., k]
        assert float(values.min()) >= lo and float(values.max()) <= hi
    hold = spec["hold_steps"]
    first = dataset.controls[0, :hold]
    assert torch.allclose(first, first[:1].expand_as(first))
    assert not torch.allclose(dataset.controls[0, 0], dataset.controls[0, hold])
    sequence = dataset.sequence(0)
    assert sequence.has_controls
    assert sequence.control_inputs.shape == (25, 2)


def test_ssa_simulator_option() -> None:
    dataset = ReactionNetworkBenchmark.generate(
        "reaction_chain_v1",
        num_trajectories=2,
        num_timesteps=6,
        simulator="ssa",
        volume=200.0,
    )
    assert dataset.simulator == "ssa"
    assert dataset.volume == 200.0
    counts = dataset.concentrations * 200.0
    assert torch.allclose(counts, counts.round())
    repeat = ReactionNetworkBenchmark.generate(
        "reaction_chain_v1",
        num_trajectories=2,
        num_timesteps=6,
        simulator="ssa",
        volume=200.0,
    )
    assert torch.equal(dataset.concentrations, repeat.concentrations)
    with pytest.raises(ValueError, match="not available for 'cstr'"):
        ReactionNetworkBenchmark.generate("reaction_van_de_vusse_v1", simulator="ssa")
    with pytest.raises(ValueError, match="volume must be"):
        ReactionNetworkBenchmark.generate(
            "reaction_chain_v1", num_trajectories=1, simulator="ssa", volume=0.0
        )


def test_split_is_deterministic_and_covers_all() -> None:
    dataset = ReactionNetworkBenchmark.generate(
        "reaction_chain_v1", num_trajectories=10
    )
    split = ReactionNetworkBenchmark.split(dataset)
    sizes = (
        split.train.num_trajectories,
        split.val.num_trajectories,
        split.test.num_trajectories,
    )
    assert sizes == (7, 1, 2)
    rebuilt = torch.cat(
        [
            split.train.concentrations,
            split.val.concentrations,
            split.test.concentrations,
        ]
    )
    assert torch.equal(rebuilt, dataset.concentrations)
    tiny = ReactionNetworkBenchmark.split(dataset.subset([0, 1, 2]))
    assert tiny.train.num_trajectories == 1
    with pytest.raises(ValueError, match="at least 3 trajectories"):
        ReactionNetworkBenchmark.split(dataset.subset([0, 1]))
    with pytest.raises(ValueError, match="sum to 1"):
        ReactionNetworkBenchmark.split(dataset, train_ratio=0.5)
    with pytest.raises(ValueError, match="leave no training data"):
        ReactionNetworkBenchmark.split(
            dataset.subset([0, 1, 2]), train_ratio=0.0, val_ratio=0.5, test_ratio=0.5
        )


def test_payload_round_trip(tmp_path: Path) -> None:
    for dataset_id in ("reaction_van_de_vusse_v1", "reaction_chain_v1"):
        dataset = ReactionNetworkBenchmark.generate(
            dataset_id, num_trajectories=2, num_timesteps=6
        )
        path = dataset.save(tmp_path / "nested" / f"{dataset_id}.pt")
        loaded = ReactionDataset.load(path)
        assert loaded.dataset_id == dataset_id
        assert loaded.card == dataset.card
        assert loaded.system.to_dict() == dataset.system.to_dict()
        assert torch.equal(loaded.concentrations, dataset.concentrations)
        assert torch.equal(loaded.times, dataset.times)
        if dataset.controls is None:
            assert loaded.controls is None
        else:
            assert loaded.controls is not None
            assert torch.equal(loaded.controls, dataset.controls)
        assert (loaded.transform is None) == (dataset.transform is None)
        assert loaded.seed == dataset.seed
        assert torch.equal(loaded.sequence(0)[1].x, dataset.sequence(0)[1].x)


def test_payload_rejects_wrong_format_and_version(tmp_path: Path) -> None:
    bad = tmp_path / "bad.pt"
    torch.save({"format": "other"}, bad)
    with pytest.raises(ValueError, match="is not a"):
        ReactionDataset.load(bad)
    dataset = ReactionNetworkBenchmark.generate(
        "reaction_chain_v1", num_trajectories=1, num_timesteps=2
    )
    path = dataset.save(tmp_path / "ok.pt")
    payload = torch.load(path, weights_only=True)
    payload["format_version"] = PAYLOAD_FORMAT_VERSION + 1
    torch.save(payload, path)
    with pytest.raises(ValueError, match="unsupported reaction dataset format_version"):
        ReactionDataset.load(path)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"num_trajectories": 0}, "num_trajectories must be"),
        ({"num_timesteps": 1}, "num_timesteps must be"),
        ({"seed": -1}, "seed must be"),
        ({"simulator": "euler"}, "simulator must be"),
    ],
)
def test_generate_validation(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ReactionNetworkBenchmark.generate("reaction_chain_v1", **kwargs)  # type: ignore[arg-type]


def test_card_validation_helpers() -> None:
    from koopman_graph.datasets.reaction.benchmark import (
        _check_ranges,
        system_from_card,
    )

    with pytest.raises(ValueError, match="do not match species"):
        _check_ranges({"A": [0.0, 1.0]}, ("A", "B"))
    with pytest.raises(ValueError, match="0 <= low <= high"):
        _check_ranges({"A": [1.0, 0.0]}, ("A",))
    with pytest.raises(ValueError, match="unknown generator kind"):
        system_from_card({"generator": {"kind": "magic"}})
    with pytest.raises(ValueError, match="unknown catalog system"):
        system_from_card({"generator": {"kind": "catalog", "name": "nope"}})


def test_card_builder_script_check_is_clean() -> None:
    """Committed cards equal what ``scripts/build_reaction_cards.py`` writes."""
    if importlib.util.find_spec("cantera") is None:
        pytest.skip("card builder needs the [cantera] extra for the h2o2 card")
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "build_reaction_cards.py"),
            "--check",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
