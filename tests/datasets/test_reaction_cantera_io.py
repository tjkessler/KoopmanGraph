"""Tests for :mod:`koopman_graph.datasets.reaction.cantera_io`.

Tests marked ``cantera`` need the ``[cantera]`` extra and are skipped by
``tests/conftest.py`` when it is absent. The missing-extra tests run
everywhere.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch

from koopman_graph.datasets.reaction import (
    CANTERA_CITATION_DOI,
    CANTERA_ILLUSTRATION_NOTE,
    check_element_balance,
    integrate_stiff,
    load_cantera_mechanism,
    mechanism_sha256,
    require_cantera,
    resolve_mechanism_path,
    simulate_cantera_reactor,
)
from koopman_graph.datasets.reaction.cantera_io import _legacy_rate_constants_enabled

# SHA-256 of h2o2.yaml in the Cantera 3.2.0 wheels (issue #65).
_H2O2_SHA256 = "0efc6c52862741a29e0c29b65d979c7d8cb409db5282bca83b9c5437b3d8c8d4"
_RADICALS = "H2:2, O2:1, AR:4, H:0.01, OH:0.01, O:0.01, HO2:0.001, H2O2:0.001, H2O:0.1"
_OTHER = "H2:1, O2:1, AR:1, H2O:2, OH:0.05, H:0.02"
_ALL = "H2:1, O2:1, AR:1, H2O:1, OH:0.1, H:0.1, O:0.1, HO2:0.01, H2O2:0.01, N2:1"
# Third bodies h2o2.yaml lacks: an explicit collider, a mass-action
# three-body Blowers-Masel rate (state dependent, frozen), and a
# non-mass-action ``(+M)`` collider with a plain Arrhenius rate.
_THIRD_BODY_REACTIONS = (
    "{equation: H + O2 + O2 <=> HO2 + O2,"
    " rate-constant: {A: 2.08e+19, b: -1.24, Ea: 0.0}}",
    "{equation: 2 O + M <=> O2 + M, type: Blowers-Masel,"
    " rate-constant: {A: 1.2e+17, b: -1.0, Ea0: 1.0e+7, w: 1.0e+9},"
    " efficiencies: {AR: 0.83, H2: 2.4, H2O: 15.4}}",
    "{equation: H + OH (+M) <=> H2O (+M),"
    " rate-constant: {A: 3.5e+13, b: 0.0, Ea: 0.0}}",
)


def _net_rates(path: str | Path, temperature: float, pressure: float, x: str):
    """Return Cantera concentrations and net production rates at a state.

    Parameters
    ----------
    path
        See signature.
    temperature
        See signature.
    pressure
        See signature.
    x
        See signature.

    Returns
    -------
        See signature."""
    import cantera as ct

    gas = ct.Solution(str(path))
    gas.TPX = temperature, pressure, x
    return (
        torch.tensor(gas.concentrations, dtype=torch.float64),
        torch.tensor(gas.net_production_rates, dtype=torch.float64),
    )


def _max_relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Return ``max |actual - expected| / max |expected|``.

    Parameters
    ----------
    actual
        See signature.
    expected
        See signature.

    Returns
    -------
        See signature."""
    return float((actual - expected).abs().max() / expected.abs().max())


@pytest.fixture
def no_falloff_mechanism(tmp_path: Path) -> Path:
    """Write ``h2o2.yaml`` minus its falloff reaction to a temp file.

    Parameters
    ----------
    tmp_path
        See signature.

    Returns
    -------
        See signature."""
    import cantera as ct

    base = ct.Solution("h2o2.yaml")
    reactions = [r for r in base.reactions() if r.reaction_type != "falloff-Troe"]
    gas = ct.Solution(
        thermo="ideal-gas", kinetics="gas", species=base.species(), reactions=reactions
    )
    path = tmp_path / "h2o2_no_falloff.yaml"
    gas.write_yaml(str(path))
    return path


@pytest.fixture
def third_body_mechanism(tmp_path: Path) -> Path:
    """Write a mechanism with the reactions in ``_THIRD_BODY_REACTIONS``.

    Parameters
    ----------
    tmp_path
        See signature.

    Returns
    -------
        See signature."""
    import cantera as ct

    base = ct.Solution("h2o2.yaml")
    reactions = [ct.Reaction.from_yaml(text, base) for text in _THIRD_BODY_REACTIONS]
    gas = ct.Solution(
        thermo="ideal-gas", kinetics="gas", species=base.species(), reactions=reactions
    )
    path = tmp_path / "third_bodies.yaml"
    gas.write_yaml(str(path))
    return path


@pytest.mark.cantera
def test_bundled_h2o2_matches_recorded_sha256() -> None:
    path = resolve_mechanism_path()
    assert path.name == "h2o2.yaml"
    assert mechanism_sha256(path) == _H2O2_SHA256
    assert resolve_mechanism_path(path) == path  # explicit paths pass through


@pytest.mark.cantera
def test_conversion_is_exact_at_reference_state() -> None:
    mechanism = load_cantera_mechanism(temperature_k=1200.0, mole_fractions=_RADICALS)
    system = mechanism.system
    assert system.num_species == 10
    assert check_element_balance(system).balanced
    assert system.elements == ("O", "H", "Ar", "N")
    assert system.arrhenius is None
    c_ref, expected = _net_rates(resolve_mechanism_path(), 1200.0, 101325.0, _RADICALS)
    assert torch.allclose(mechanism.reference_concentrations, c_ref)
    actual = system.time_derivative(mechanism.reference_concentrations)
    assert _max_relative_error(actual, expected) < 1e-12
    provenance = mechanism.provenance
    assert provenance.frozen_reactions == ("2 OH (+M) <=> H2O2 (+M)",)
    assert provenance.skipped_reactions == ()
    assert provenance.sha256 == _H2O2_SHA256
    assert provenance.cantera_version.startswith("3.2")
    assert provenance.temperature_k == 1200.0
    assert provenance.note == CANTERA_ILLUSTRATION_NOTE
    assert provenance.citation_doi == CANTERA_CITATION_DOI
    payload = json.loads(json.dumps(provenance.to_dict()))
    assert payload["mole_fractions"]["AR"] > 0.5
    # h2o2.yaml has duplicate reactions; their rows still get unique names.
    assert system.reaction_names is not None
    assert len(set(system.reaction_names)) == len(system.reaction_names)


@pytest.mark.cantera
def test_explicit_frozen_and_non_mass_action_third_bodies(
    third_body_mechanism: Path,
) -> None:
    mechanism = load_cantera_mechanism(
        third_body_mechanism, temperature_k=1200.0, mole_fractions=_ALL
    )
    assert mechanism.provenance.frozen_reactions == ("2 O + M <=> O2 + M",)
    names = mechanism.system.reaction_names
    assert names is not None
    assert sum("H + O2 + O2" in name for name in names) == 2  # one collider
    for pressure, x in ((101325.0, _ALL), (3.0e5, _OTHER)):
        # The frozen Blowers-Masel rate depends only on T, so with its [M]
        # expanded per collider the conversion stays exact at this T.
        c, expected = _net_rates(third_body_mechanism, 1200.0, pressure, x)
        actual = mechanism.system.time_derivative(c)
        assert _max_relative_error(actual, expected) < 1e-12


@pytest.mark.cantera
def test_legacy_rate_constant_mode_is_ignored_and_restored() -> None:
    import cantera as ct

    reference = load_cantera_mechanism(temperature_k=1200.0, mole_fractions=_RADICALS)
    assert not _legacy_rate_constants_enabled(ct)
    ct.use_legacy_rate_constants(True)
    try:
        assert _legacy_rate_constants_enabled(ct)
        legacy = load_cantera_mechanism(temperature_k=1200.0, mole_fractions=_RADICALS)
        assert _legacy_rate_constants_enabled(ct)
    finally:
        ct.use_legacy_rate_constants(False)
    assert torch.equal(legacy.system.rate_constants, reference.system.rate_constants)


@pytest.mark.cantera
def test_three_body_expansion_is_exact_away_from_reference(
    no_falloff_mechanism: Path,
) -> None:
    mechanism = load_cantera_mechanism(
        no_falloff_mechanism, temperature_k=1100.0, mole_fractions=_RADICALS
    )
    assert mechanism.provenance.frozen_reactions == ()
    # Different composition and pressure, same temperature: still exact.
    c, expected = _net_rates(no_falloff_mechanism, 1100.0, 2.0e5, _OTHER)
    assert _max_relative_error(mechanism.system.time_derivative(c), expected) < 1e-12


@pytest.mark.cantera
def test_frozen_falloff_is_approximate_away_from_reference() -> None:
    mechanism = load_cantera_mechanism(temperature_k=1200.0, mole_fractions=_RADICALS)
    c, expected = _net_rates(resolve_mechanism_path(), 1200.0, 101325.0, _OTHER)
    error = _max_relative_error(mechanism.system.time_derivative(c), expected)
    assert 0.0 < error < 0.05


@pytest.mark.cantera
def test_state_dependent_policies() -> None:
    frozen = load_cantera_mechanism(temperature_k=1200.0, mole_fractions=_RADICALS)
    skipped = load_cantera_mechanism(
        temperature_k=1200.0, mole_fractions=_RADICALS, state_dependent="skip"
    )
    assert skipped.provenance.skipped_reactions == ("2 OH (+M) <=> H2O2 (+M)",)
    assert skipped.provenance.frozen_reactions == ()
    # The reversible falloff reaction contributes a forward and a reverse row.
    assert skipped.system.num_reactions == frozen.system.num_reactions - 2
    with pytest.raises(ValueError, match="state-dependent rate type 'falloff-Troe'"):
        load_cantera_mechanism(state_dependent="error")
    with pytest.raises(ValueError, match="state_dependent must be"):
        load_cantera_mechanism(state_dependent="drop")  # type: ignore[arg-type]


@pytest.mark.cantera
def test_reaction_orders_are_skipped_or_rejected(tmp_path: Path) -> None:
    import cantera as ct

    base = ct.Solution("h2o2.yaml")
    ordered = ct.Reaction(
        equation="H2 + O => H + OH", rate=ct.ArrheniusRate(3.87e4, 2.7, 2.619e7)
    )
    ordered.orders = {"H2": 0.5}
    # Orders equal to the stoichiometry are mass action and are kept.
    matching = ct.Reaction(
        equation="H2 + O2 => 2 OH", rate=ct.ArrheniusRate(1.7e10, 0.0, 2.0e8)
    )
    matching.orders = {"H2": 1.0, "O2": 1.0}
    elementary = [r for r in base.reactions() if r.reaction_type == "Arrhenius"][:2]
    gas = ct.Solution(
        thermo="ideal-gas",
        kinetics="gas",
        species=base.species(),
        reactions=[ordered, matching, *elementary],
    )
    path = tmp_path / "orders.yaml"
    gas.write_yaml(str(path))
    mechanism = load_cantera_mechanism(path)
    assert mechanism.provenance.skipped_reactions == ("H2 + O => H + OH",)
    assert mechanism.system.num_reactions == 5  # two reversible + one kept
    with pytest.raises(ValueError, match="non-mass-action orders"):
        load_cantera_mechanism(path, state_dependent="error")


@pytest.mark.cantera
def test_isothermal_reactor_matches_mass_action_integration() -> None:
    times = torch.tensor([0.0, 1e-5, 1e-4, 1e-3], dtype=torch.float64)
    initial = "H2:2, O2:1, AR:4"
    trajectory = simulate_cantera_reactor(
        times=times,
        temperature_k=1200.0,
        mole_fractions=initial,
        reactor="const_volume",
        isothermal=True,
    )
    assert trajectory.concentrations.shape == (4, 10)
    assert torch.allclose(
        trajectory.temperatures_k, torch.full((4,), 1200.0, dtype=torch.float64)
    )
    assert trajectory.species[0] == "H2"
    assert trajectory.provenance.sha256 == _H2O2_SHA256
    assert trajectory.provenance.state_dependent is None
    mechanism = load_cantera_mechanism(temperature_k=1200.0, mole_fractions=initial)
    mass_action = integrate_stiff(
        mechanism.system,
        trajectory.concentrations[0],
        times,
        rtol=1e-10,
        atol=1e-16,
    )
    scale = float(trajectory.concentrations.abs().max())
    assert float((mass_action - trajectory.concentrations).abs().max()) < 1e-3 * scale


@pytest.mark.cantera
def test_adiabatic_reactor_heats_up() -> None:
    times = torch.tensor([0.0, 2e-3], dtype=torch.float64)
    trajectory = simulate_cantera_reactor(
        times=times, temperature_k=1200.0, mole_fractions="H2:2, O2:1, AR:4"
    )
    assert trajectory.reactor == "const_pressure"
    assert not trajectory.isothermal
    assert float(trajectory.temperatures_k[-1]) > 1500.0
    assert torch.allclose(
        trajectory.pressures_pa, torch.full((2,), 101325.0, dtype=torch.float64)
    )


@pytest.mark.cantera
def test_cantera_validation() -> None:
    with pytest.raises(FileNotFoundError, match="neither a file"):
        resolve_mechanism_path("no_such_mechanism.yaml")
    times = torch.tensor([0.0, 1e-4])
    kwargs = {"temperature_k": 1200.0, "mole_fractions": "H2:2, O2:1"}
    with pytest.raises(ValueError, match="reactor must be"):
        simulate_cantera_reactor(times=times, reactor="pfr", **kwargs)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="start at 0"):
        simulate_cantera_reactor(times=torch.tensor([1.0, 2.0]), **kwargs)
    with pytest.raises(ValueError, match="non-empty 1-D"):
        simulate_cantera_reactor(times=torch.zeros(0), **kwargs)
    with pytest.raises(ValueError, match="finite"):
        simulate_cantera_reactor(times=torch.tensor([0.0, float("inf")]), **kwargs)
    with pytest.raises(ValueError, match="temperature_k must be"):
        load_cantera_mechanism(temperature_k=-1.0)
    with pytest.raises(ValueError, match="pressure_pa must be"):
        simulate_cantera_reactor(times=times, pressure_pa=0.0, **kwargs)


@pytest.mark.cantera
def test_bare_name_prefers_bundled_over_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundled = resolve_mechanism_path()
    monkeypatch.chdir(tmp_path)
    (tmp_path / "h2o2.yaml").write_text("units: {}\n")
    (tmp_path / "only_here.yaml").write_text("units: {}\n")
    assert resolve_mechanism_path("h2o2.yaml") == bundled
    assert resolve_mechanism_path("./h2o2.yaml") == (tmp_path / "h2o2.yaml").resolve()
    assert (
        resolve_mechanism_path("only_here.yaml")
        == (tmp_path / "only_here.yaml").resolve()
    )


def test_missing_cantera_names_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "cantera", None)
    with pytest.raises(ImportError, match=r"koopman-graph\[cantera\]"):
        require_cantera()
    with pytest.raises(ImportError, match=r"koopman-graph\[cantera\]"):
        load_cantera_mechanism()
    with pytest.raises(ImportError, match=r"koopman-graph\[cantera\]"):
        resolve_mechanism_path("h2o2.yaml")
    with pytest.raises(ImportError, match=r"koopman-graph\[cantera\]"):
        simulate_cantera_reactor(
            times=torch.tensor([0.0]), temperature_k=1000.0, mole_fractions="H2:1"
        )


def test_sha256_and_explicit_path_need_no_cantera(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "cantera", None)
    path = tmp_path / "mech.yaml"
    content = b"units: {}\n"
    path.write_bytes(content)
    assert mechanism_sha256(path) == hashlib.sha256(content).hexdigest()
    assert resolve_mechanism_path(path) == path.resolve()
    with pytest.raises(FileNotFoundError, match="does not exist"):
        resolve_mechanism_path(tmp_path / "missing.yaml")
