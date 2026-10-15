"""Lazy ``[cantera]`` boundary: mechanism loading and reactor simulation.

:func:`load_cantera_mechanism` converts a Cantera gas-phase mechanism (by
default the bundled ``h2o2.yaml``, read from the installed package and not
vendored) into a mass-action
:class:`~koopman_graph.datasets.reaction.ReactionSystem` at a reference
state ``(T, P, X)``:

* Every reversible reaction becomes a forward and a reverse reaction. The
  rate constants are Cantera's ``forward_rate_constants`` /
  ``reverse_rate_constants`` at ``T``. Reverse constants come from
  thermochemistry, so they are not Arrhenius and the system carries no
  ``arrhenius`` data.
* **Elementary** reactions convert exactly.
* **Three-body** reactions (``A + B + M``, mass-action third bodies)
  convert exactly. ``[M] = Σ_s ε_s c_s``, so the reaction is expanded into
  one reaction per collider ``s`` with ``ε_s > 0``, written ``A + B + s →
  … + s`` with rate constant ``k ε_s``.
* **State-dependent** rates (falloff, chemically activated, PLOG,
  Chebyshev, Blowers-Masel, ...) are not mass action. ``state_dependent``
  picks what happens: ``"freeze"`` (default) uses the effective rate
  constant at the reference state, which is exact there and approximate
  away from it; ``"skip"`` drops the reaction; ``"error"`` raises. Every
  frozen or skipped reaction is listed in the provenance. A frozen
  reaction with a mass-action third body (e.g. three-body Blowers-Masel)
  keeps its ``[M]`` factor by the same per-collider expansion.
* Reactions with explicit non-stoichiometric ``orders`` are always
  rejected or skipped (``"freeze"`` does not apply to them). Orders equal
  to the reactant stoichiometry are mass action and convert normally.

Rate constants are read with Cantera's legacy rate-constant mode
(``cantera.use_legacy_rate_constants``) switched off, whatever the
process-wide setting is, and the setting is restored afterwards.

Units: Cantera's kmol/m³ equals mol/L, so concentrations (mol/L), rate
constants ((L/mol)^(order-1)/s) and time (s) carry over without
conversion.

Provenance records the Cantera version, the resolved mechanism path, and
its SHA-256. Cantera's bundled input files are for illustration only;
publications should cite the Cantera version used (3.2.0:
doi:10.5281/zenodo.17620923).

Importing this module does **not** import cantera; :func:`require_cantera`
does, with an install hint for the ``[cantera]`` extra.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor

from koopman_graph.datasets.reaction.system import ReactionSystem

StateDependentPolicy = Literal["freeze", "skip", "error"]
"""How :func:`load_cantera_mechanism` handles non-mass-action rate types."""

DEFAULT_MECHANISM = "h2o2.yaml"
"""Bundled Cantera mechanism used by default (10 species, 29 reactions)."""

CANTERA_ILLUSTRATION_NOTE = (
    "Cantera's bundled input files are provided for illustration only; they "
    "are not validated kinetic models for quantitative use."
)
"""Caveat attached to every provenance record (from Cantera's data README)."""

CANTERA_CITATION_DOI = "10.5281/zenodo.17620923"
"""DOI of the Cantera 3.2.0 software release, to cite alongside datasets."""

_CANTERA_IMPORT_ERROR = (
    "cantera is required for Cantera mechanism loading and reactor "
    "simulation. Install with: pip install koopman-graph[cantera]"
)

_EXACT_TYPES = {"Arrhenius", "three-body-Arrhenius"}

# Rate types whose effective rate constant already includes ``[M]``
# (through the reduced pressure), so no collider is expanded.
_FALLOFF_PREFIXES = ("falloff", "chemically-activated")

# One-reaction mechanism used to read the process-wide legacy rate-constant
# setting, which Cantera can set but not query: with A = 1 the forward rate
# constant is 1 normally and ``[M]`` (≈ 0.04 kmol/m³ here) in legacy mode.
_LEGACY_PROBE_YAML = """
phases:
- name: probe
  thermo: ideal-gas
  elements: [Ar]
  species: [A, A2]
  kinetics: gas
  state: {T: 300.0, P: 1 atm}
species:
- name: A
  composition: {Ar: 1}
  thermo: {model: constant-cp}
- name: A2
  composition: {Ar: 2}
  thermo: {model: constant-cp}
reactions:
- equation: 2 A + M <=> A2 + M
  rate-constant: {A: 1.0, b: 0.0, Ea: 0.0}
"""


@dataclass(frozen=True)
class CanteraProvenance:
    """Where a mechanism came from and how it was converted.

    Attributes
    ----------
    mechanism : str
        Name or path as requested (e.g. ``"h2o2.yaml"``).
    path : str
        Resolved absolute path of the mechanism file.
    sha256 : str
        SHA-256 hex digest of the mechanism file.
    cantera_version : str
        ``cantera.__version__`` used for loading.
    temperature_k : float
        Reference temperature (rate constants are evaluated here).
    pressure_pa : float
        Reference pressure.
    mole_fractions : dict of str to float
        Reference composition (non-zero mole fractions).
    state_dependent : str or None
        Policy used for non-mass-action rates, or ``None`` when no
        conversion happened (reactor simulation).
    frozen_reactions : tuple of str
        Equations whose effective rate was frozen at the reference state.
    skipped_reactions : tuple of str
        Equations left out of the system.
    note : str
        :data:`CANTERA_ILLUSTRATION_NOTE`.
    citation_doi : str
        :data:`CANTERA_CITATION_DOI`.
    """

    mechanism: str
    path: str
    sha256: str
    cantera_version: str
    temperature_k: float
    pressure_pa: float
    mole_fractions: dict[str, float]
    state_dependent: str | None
    frozen_reactions: tuple[str, ...]
    skipped_reactions: tuple[str, ...]
    note: str = CANTERA_ILLUSTRATION_NOTE
    citation_doi: str = CANTERA_CITATION_DOI

    def to_dict(self) -> dict[str, Any]:
        """Serialize provenance for a dataset card.

        Returns
        -------
        dict
            JSON-compatible copy of every field.
        """
        return {
            "mechanism": self.mechanism,
            "path": self.path,
            "sha256": self.sha256,
            "cantera_version": self.cantera_version,
            "temperature_k": self.temperature_k,
            "pressure_pa": self.pressure_pa,
            "mole_fractions": dict(self.mole_fractions),
            "state_dependent": self.state_dependent,
            "frozen_reactions": list(self.frozen_reactions),
            "skipped_reactions": list(self.skipped_reactions),
            "note": self.note,
            "citation_doi": self.citation_doi,
        }


@dataclass(frozen=True, eq=False)
class CanteraMechanism:
    """A Cantera mechanism converted to mass action at a reference state.

    Attributes
    ----------
    system : ReactionSystem
        Mass-action system over the mechanism's species, with Cantera's
        element matrix.
    reference_concentrations : Tensor
        ``float64`` concentrations ``(num_species,)`` in mol/L at the
        reference state. ``system.time_derivative`` here equals Cantera's
        ``net_production_rates``.
    provenance : CanteraProvenance
        Source file, versions, and conversion details.
    """

    system: ReactionSystem
    reference_concentrations: Tensor
    provenance: CanteraProvenance


@dataclass(frozen=True, eq=False)
class CanteraReactorTrajectory:
    """Output of :func:`simulate_cantera_reactor`.

    Attributes
    ----------
    times : Tensor
        ``float64`` output times ``(T,)`` in seconds.
    concentrations : Tensor
        ``float64`` concentrations ``(T, num_species)`` in mol/L.
    temperatures_k : Tensor
        ``float64`` reactor temperature ``(T,)``.
    pressures_pa : Tensor
        ``float64`` reactor pressure ``(T,)``.
    species : tuple of str
        Species names, aligned with the last axis of ``concentrations``.
    provenance : CanteraProvenance
        Mechanism source and the initial reactor state.
    reactor : str
        ``"const_pressure"`` or ``"const_volume"``.
    isothermal : bool
        Whether the energy equation was disabled.
    """

    times: Tensor
    concentrations: Tensor
    temperatures_k: Tensor
    pressures_pa: Tensor
    species: tuple[str, ...]
    provenance: CanteraProvenance
    reactor: str
    isothermal: bool


def require_cantera() -> Any:
    """Import and return :mod:`cantera`, or explain how to install it.

    Returns
    -------
    module
        The :mod:`cantera` module.

    Raises
    ------
    ImportError
        If Cantera is not installed; the message names the ``[cantera]``
        extra.
    """
    try:
        import cantera
    except ImportError as exc:
        raise ImportError(_CANTERA_IMPORT_ERROR) from exc
    return cantera


def resolve_mechanism_path(name_or_path: str | Path = DEFAULT_MECHANISM) -> Path:
    """Locate a mechanism file: an explicit path, or a bundled Cantera input.

    Parameters
    ----------
    name_or_path : str or Path, optional
        A :class:`~pathlib.Path` or a string with a directory part is an
        explicit path. A bare file name is searched in
        ``cantera.get_data_directories()``, bundled directories first and
        the current directory last, so a stray copy in the working
        directory does not shadow the bundled input. Default is
        :data:`DEFAULT_MECHANISM`.

    Returns
    -------
    Path
        Absolute path of the mechanism file.

    Raises
    ------
    ImportError
        If Cantera is not installed.
    FileNotFoundError
        If no such file exists.
    """
    candidate = Path(name_or_path).expanduser()
    if isinstance(name_or_path, Path) or candidate.name != str(name_or_path):
        if candidate.is_file():
            return candidate.resolve()
        msg = f"mechanism file {str(name_or_path)!r} does not exist"
        raise FileNotFoundError(msg)
    ct = require_cantera()
    directories = [Path(d) for d in ct.get_data_directories()]
    cwd = Path.cwd().resolve()
    directories.sort(key=lambda d: d.resolve() == cwd)  # stable: cwd last
    for directory in directories:
        path = directory / candidate.name
        if path.is_file():
            return path.resolve()
    msg = (
        f"mechanism {str(name_or_path)!r} is neither a file nor in Cantera's "
        "data directories"
    )
    raise FileNotFoundError(msg)


def load_cantera_mechanism(
    name_or_path: str | Path = DEFAULT_MECHANISM,
    *,
    temperature_k: float = 1000.0,
    pressure_pa: float = 101325.0,
    mole_fractions: str | dict[str, float] | None = None,
    state_dependent: StateDependentPolicy = "freeze",
    phase: str | None = None,
) -> CanteraMechanism:
    """Convert a Cantera gas mechanism to a mass-action system at ``(T, P, X)``.

    Parameters
    ----------
    name_or_path : str or Path, optional
        Mechanism file name (bundled with Cantera) or path. Default is
        :data:`DEFAULT_MECHANISM`.
    temperature_k : float, optional
        Reference temperature in K. Default is ``1000.0``.
    pressure_pa : float, optional
        Reference pressure in Pa. Default is one atmosphere.
    mole_fractions : str, dict, or None, optional
        Reference composition in any form Cantera's ``X`` accepts
        (``"H2:2, O2:1, AR:4"``). ``None`` keeps the phase's default
        composition. Only frozen state-dependent rates and
        ``reference_concentrations`` depend on it.
    state_dependent : {"freeze", "skip", "error"}, optional
        Handling of non-mass-action rate types; see the module docstring.
        Default is ``"freeze"``.
    phase : str or None, optional
        Phase name inside the input file. Default is the first phase.

    Returns
    -------
    CanteraMechanism
        System, reference concentrations, and provenance.

    Raises
    ------
    ImportError
        If Cantera is not installed.
    ValueError
        If the policy is unknown, the state is invalid, or a reaction cannot
        be represented under ``state_dependent="error"``.
    """
    if state_dependent not in ("freeze", "skip", "error"):
        msg = (
            "state_dependent must be 'freeze', 'skip', or 'error', got "
            f"{state_dependent!r}"
        )
        raise ValueError(msg)
    ct, path, gas, temperature, pressure = _open_solution(
        name_or_path, phase, temperature_k, pressure_pa
    )
    if mole_fractions is None:
        gas.TP = temperature, pressure
    else:
        gas.TPX = temperature, pressure, mole_fractions

    species = tuple(gas.species_names)
    index = {name: i for i, name in enumerate(species)}
    k_forward, k_reverse = _rate_constants(ct, gas)
    reactant_rows: list[list[float]] = []
    product_rows: list[list[float]] = []
    rates: list[float] = []
    names: list[str] = []
    frozen: list[str] = []
    skipped: list[str] = []

    def add(
        lhs: dict[str, float], rhs: dict[str, float], rate: float, label: str
    ) -> None:
        """Append one irreversible mass-action reaction.

        Parameters
        ----------
        lhs
            See signature.
        rhs
            See signature.
        rate
            See signature.
        label
            See signature."""
        alpha = [0.0] * len(species)
        beta = [0.0] * len(species)
        for name, coeff in lhs.items():
            alpha[index[name]] += float(coeff)
        for name, coeff in rhs.items():
            beta[index[name]] += float(coeff)
        reactant_rows.append(alpha)
        product_rows.append(beta)
        rates.append(float(rate))
        names.append(label)

    def collider_efficiencies(third_body: Any) -> list[tuple[str | None, float]]:
        """Return ``(species, ε_s)`` for every collider with ``ε_s > 0``.

        Parameters
        ----------
        third_body
            See signature.

        Returns
        -------
            See signature."""
        efficiencies = ((name, third_body.efficiency(name)) for name in species)
        return [(name, eff) for name, eff in efficiencies if eff > 0.0]

    for j, reaction in enumerate(gas.reactions()):
        equation = reaction.equation
        if not _has_mass_action_orders(reaction):
            if state_dependent == "error":
                msg = f"reaction {equation!r} has non-mass-action orders"
                raise ValueError(msg)
            skipped.append(equation)
            continue
        kind = reaction.reaction_type
        if kind not in _EXACT_TYPES:
            if state_dependent == "skip":
                skipped.append(equation)
                continue
            if state_dependent == "error":
                msg = (
                    f"reaction {equation!r} has state-dependent rate type "
                    f"{kind!r}; use state_dependent='freeze' or 'skip'"
                )
                raise ValueError(msg)
            frozen.append(equation)
        third_body = reaction.third_body
        colliders: list[tuple[str | None, float]] = [(None, 1.0)]
        # Falloff constants already contain [M] (through the reduced
        # pressure), and Cantera applies no [M] to non-mass-action third
        # bodies; every other third body is expanded per collider.
        if (
            third_body is not None
            and third_body.mass_action
            and not kind.startswith(_FALLOFF_PREFIXES)
        ):
            colliders = collider_efficiencies(third_body)
        for collider, efficiency in colliders:
            lhs = dict(reaction.reactants)
            rhs = dict(reaction.products)
            suffix = f" #{j}" if reaction.duplicate else ""
            if collider is not None:
                lhs[collider] = lhs.get(collider, 0.0) + 1.0
                rhs[collider] = rhs.get(collider, 0.0) + 1.0
                suffix = f"{suffix} [M={collider}]"
            add(lhs, rhs, k_forward[j] * efficiency, f"{equation}{suffix} (fwd)")
            if reaction.reversible:
                add(rhs, lhs, k_reverse[j] * efficiency, f"{equation}{suffix} (rev)")

    if not rates:
        msg = f"no representable reactions in {str(name_or_path)!r}"
        raise ValueError(msg)
    elements = tuple(gas.element_names)
    element_matrix = torch.tensor(
        [[gas.n_atoms(s, e) for s in species] for e in elements], dtype=torch.float64
    )
    system = ReactionSystem(
        species=species,
        reactant_stoichiometry=torch.tensor(reactant_rows, dtype=torch.float64),
        product_stoichiometry=torch.tensor(product_rows, dtype=torch.float64),
        rate_constants=torch.tensor(rates, dtype=torch.float64),
        element_matrix=element_matrix,
        elements=elements,
        reaction_names=tuple(names),
        name=f"cantera_{path.stem}",
    )
    provenance = _provenance(
        ct,
        gas,
        mechanism=str(name_or_path),
        path=path,
        state_dependent=state_dependent,
        frozen=tuple(frozen),
        skipped=tuple(skipped),
    )
    return CanteraMechanism(
        system=system,
        reference_concentrations=torch.tensor(gas.concentrations, dtype=torch.float64),
        provenance=provenance,
    )


def simulate_cantera_reactor(
    name_or_path: str | Path = DEFAULT_MECHANISM,
    *,
    times: Tensor,
    temperature_k: float,
    pressure_pa: float = 101325.0,
    mole_fractions: str | dict[str, float],
    reactor: Literal["const_pressure", "const_volume"] = "const_pressure",
    isothermal: bool = False,
    phase: str | None = None,
) -> CanteraReactorTrajectory:
    """Integrate an ideal-gas batch reactor with Cantera's ``ReactorNet``.

    Parameters
    ----------
    name_or_path : str or Path, optional
        Mechanism file name or path. Default is :data:`DEFAULT_MECHANISM`.
    times : Tensor
        Strictly increasing 1-D output times in seconds, starting at ``0``.
    temperature_k : float
        Initial temperature in K.
    pressure_pa : float, optional
        Initial pressure in Pa. Default is one atmosphere.
    mole_fractions : str or dict
        Initial composition (Cantera ``X`` syntax).
    reactor : {"const_pressure", "const_volume"}, optional
        ``IdealGasConstPressureReactor`` or ``IdealGasReactor``. Default is
        ``"const_pressure"``.
    isothermal : bool, optional
        Disable the energy equation (fixed temperature). Default is
        ``False``. An isothermal run is the one comparable to a mass-action
        system from :func:`load_cantera_mechanism`.
    phase : str or None, optional
        Phase name inside the input file.

    Returns
    -------
    CanteraReactorTrajectory
        Concentrations, temperature and pressure at each output time, plus
        provenance of the mechanism and initial state.

    Raises
    ------
    ImportError
        If Cantera is not installed.
    ValueError
        If ``times`` or ``reactor`` is invalid.
    """
    if reactor not in ("const_pressure", "const_volume"):
        msg = f"reactor must be 'const_pressure' or 'const_volume', got {reactor!r}"
        raise ValueError(msg)
    if not isinstance(times, Tensor) or times.ndim != 1 or times.numel() < 1:
        msg = "times must be a non-empty 1-D Tensor"
        raise ValueError(msg)
    t = times.to(torch.float64)
    if not bool(torch.isfinite(t).all()):
        msg = "times must be finite"
        raise ValueError(msg)
    if float(t[0]) != 0.0 or (t.numel() > 1 and not bool((t[1:] > t[:-1]).all())):
        msg = "times must start at 0 and be strictly increasing"
        raise ValueError(msg)
    ct, path, gas, temperature, pressure = _open_solution(
        name_or_path, phase, temperature_k, pressure_pa
    )
    gas.TPX = temperature, pressure, mole_fractions
    provenance = _provenance(
        ct,
        gas,
        mechanism=str(name_or_path),
        path=path,
        state_dependent=None,
        frozen=(),
        skipped=(),
    )
    reactor_cls = (
        ct.IdealGasConstPressureReactor
        if reactor == "const_pressure"
        else ct.IdealGasReactor
    )
    energy = "off" if isothermal else "on"
    # clone=True (the default after Cantera 3.2) gives the reactor its own
    # phase copy; state is read back through ``vessel.phase``.
    vessel = reactor_cls(gas, energy=energy, clone=True)
    network = ct.ReactorNet([vessel])
    concentrations, temperatures, pressures = [], [], []
    for time in t.tolist():
        if time > 0.0:
            network.advance(time)
        concentrations.append(list(vessel.phase.concentrations))
        temperatures.append(vessel.phase.T)
        pressures.append(vessel.phase.P)
    return CanteraReactorTrajectory(
        times=t,
        concentrations=torch.tensor(concentrations, dtype=torch.float64),
        temperatures_k=torch.tensor(temperatures, dtype=torch.float64),
        pressures_pa=torch.tensor(pressures, dtype=torch.float64),
        species=tuple(gas.species_names),
        provenance=provenance,
        reactor=reactor,
        isothermal=bool(isothermal),
    )


def mechanism_sha256(path: str | Path) -> str:
    """Return the SHA-256 hex digest of a mechanism file.

    Parameters
    ----------
    path : str or Path
        File to hash.

    Returns
    -------
    str
        Lower-case hex digest.
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _provenance(
    ct: Any,
    gas: Any,
    *,
    mechanism: str,
    path: Path,
    state_dependent: str | None,
    frozen: tuple[str, ...],
    skipped: tuple[str, ...],
) -> CanteraProvenance:
    """Build a provenance record for ``gas`` at its current state.

    Parameters
    ----------
    ct
        See signature.
    gas
        See signature.
    mechanism
        See signature.
    path
        See signature.
    state_dependent
        See signature.
    frozen
        See signature.
    skipped
        See signature.

    Returns
    -------
        See signature."""
    fractions = {
        name: float(x)
        for name, x in zip(gas.species_names, gas.X, strict=True)
        if float(x) > 0.0
    }
    return CanteraProvenance(
        mechanism=mechanism,
        path=str(path),
        sha256=mechanism_sha256(path),
        cantera_version=str(ct.__version__),
        temperature_k=float(gas.T),
        pressure_pa=float(gas.P),
        mole_fractions=fractions,
        state_dependent=state_dependent,
        frozen_reactions=frozen,
        skipped_reactions=skipped,
    )


def _open_solution(
    name_or_path: str | Path,
    phase: str | None,
    temperature_k: float,
    pressure_pa: float,
) -> tuple[Any, Path, Any, float, float]:
    """Validate the state, then load the mechanism as a Cantera ``Solution``.

    Parameters
    ----------
    name_or_path
        See signature.
    phase
        See signature.
    temperature_k
        See signature.
    pressure_pa
        See signature.

    Returns
    -------
        ``(cantera, path, solution, temperature, pressure)``."""
    temperature = _positive("temperature_k", temperature_k)
    pressure = _positive("pressure_pa", pressure_pa)
    ct = require_cantera()
    path = resolve_mechanism_path(name_or_path)
    gas = ct.Solution(str(path), phase) if phase else ct.Solution(str(path))
    return ct, path, gas, temperature, pressure


def _legacy_rate_constants_enabled(ct: Any) -> bool:
    """Return whether Cantera's legacy rate-constant mode is on.

    Cantera can set this process-wide flag but not report it, so a
    one-reaction probe mechanism with ``A = 1`` is evaluated instead.

    Parameters
    ----------
    ct
        See signature.

    Returns
    -------
        See signature."""
    probe = ct.Solution(yaml=_LEGACY_PROBE_YAML)
    return not math.isclose(float(probe.forward_rate_constants[0]), 1.0)


def _rate_constants(ct: Any, gas: Any) -> tuple[Any, Any]:
    """Return forward and reverse rate constants without ``[M]`` factors.

    Legacy mode (which multiplies three-body constants by ``[M]``) is
    switched off while reading and restored afterwards.

    Parameters
    ----------
    ct
        See signature.
    gas
        See signature.

    Returns
    -------
        See signature."""
    legacy = _legacy_rate_constants_enabled(ct)
    if legacy:
        ct.use_legacy_rate_constants(False)
    try:
        return gas.forward_rate_constants.copy(), gas.reverse_rate_constants.copy()
    finally:
        if legacy:
            ct.use_legacy_rate_constants(True)


def _has_mass_action_orders(reaction: Any) -> bool:
    """Return whether every explicit order equals the reactant coefficient.

    Parameters
    ----------
    reaction
        See signature.

    Returns
    -------
        See signature."""
    reactants = reaction.reactants
    return all(
        math.isclose(float(order), float(reactants.get(name, 0.0)))
        for name, order in reaction.orders.items()
    )


def _positive(name: str, value: float) -> float:
    """Return ``value`` as a finite positive float or raise ``ValueError``.

    Parameters
    ----------
    name
        See signature.
    value
        See signature.

    Returns
    -------
        See signature."""
    x = float(value)
    if not math.isfinite(x) or x <= 0.0:
        msg = f"{name} must be finite and > 0, got {value!r}"
        raise ValueError(msg)
    return x
