"""Compact reaction-network payloads under ``benchmarks/v0.16/data/``."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
from tests.helpers import REPO_ROOT

from koopman_graph.datasets.reaction import ReactionDataset

_DATA = REPO_ROOT / "benchmarks" / "v0.16" / "data"
_SCRIPT = REPO_ROOT / "scripts" / "build_reaction_payloads.py"
_FAMILY = _DATA / "reaction_random_family_v1"


def _builder() -> ModuleType:
    """Import ``scripts/build_reaction_payloads.py`` as a module.

    Returns
    -------
    ModuleType
        The builder module.
    """
    spec = importlib.util.spec_from_file_location("build_reaction_payloads", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of ``path``.

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


@pytest.mark.parametrize(
    "name",
    ["reaction_michaelis_menten_v1", "reaction_van_de_vusse_dilution_v1"],
)
def test_single_payload_loads_with_its_card(name: str) -> None:
    """Each single-file payload loads and carries its own dataset id."""
    dataset = ReactionDataset.load(_DATA / f"{name}.pt")
    builder = _builder()
    assert dataset.dataset_id == name
    assert dataset.card["dataset_id"] == name
    assert dataset.times.shape == (builder.NUM_TIMESTEPS,)
    assert dataset.concentrations.shape[1] == builder.NUM_TIMESTEPS
    assert bool((dataset.concentrations >= 0).all())


def test_dilution_payload_has_one_control_and_a_fixed_temperature() -> None:
    """The CSTR variant drives the dilution rate only."""
    dataset = ReactionDataset.load(_DATA / "reaction_van_de_vusse_dilution_v1.pt")
    controls = dataset.card["controls"]
    assert controls["names"] == ["dilution_rate"]
    assert set(controls["fixed"]) == {"temperature_k"}
    assert dataset.controls is not None
    assert dataset.controls.shape == (*dataset.concentrations.shape[:2], 1)
    low, high = controls["ranges"]["dilution_rate"]
    assert bool(((dataset.controls >= low) & (dataset.controls <= high)).all())


def test_family_index_binds_every_member() -> None:
    """``index.json`` lists every member file with its SHA-256."""
    index = json.loads((_FAMILY / "index.json").read_text(encoding="utf-8"))
    builder = _builder()
    assert index["format"] == builder.FAMILY_FORMAT
    assert index["format_version"] == builder.FAMILY_FORMAT_VERSION
    listed = {member["file"] for member in index["members"]}
    assert listed == {path.name for path in _FAMILY.glob("*.pt")}
    assert len(listed) == len(builder.FAMILY_SPECIES) * builder.FAMILY_NETWORKS_PER_SIZE
    for member in index["members"]:
        path = _FAMILY / member["file"]
        assert _sha256(path) == member["sha256"]
        dataset = ReactionDataset.load(path)
        assert dataset.dataset_id == member["dataset_id"]
        assert dataset.system.num_species == member["num_species"]
        assert dataset.system.num_reactions == member["num_reactions"]
        assert builder.is_connected(dataset.system)


def test_family_members_follow_the_seed_rule() -> None:
    """The committed members are the builder's first connected network seeds."""
    index = json.loads((_FAMILY / "index.json").read_text(encoding="utf-8"))
    committed = [(m["num_species"], m["network_seed"]) for m in index["members"]]
    assert committed == _builder().family_members()


@pytest.mark.slow
def test_payload_builder_check_is_clean() -> None:
    """Regenerated payloads match the committed ones within ``rtol=1e-5``."""
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
