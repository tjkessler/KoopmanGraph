"""``ReactionNetworkBenchmark``: card-defined reaction-network datasets.

Each frozen dataset is defined by a FAIR JSON card under
``koopman_graph/datasets/reaction/data/`` (``<dataset_id>.json``), read with
:mod:`importlib.resources`. The card is the source of truth. It records:

* the generator (catalog constructor, random-network sampler, Van de Vusse
  CSTR, or Cantera mechanism) and its parameters;
* the simulator, time grid, default trajectory count, and seed;
* per-species initial-concentration ranges (and control ranges for the
  CSTR);
* the optional log-concentration floor and the graph encoding;
* oracle references (equilibrium, dynamic Jacobian eigenvalues,
  timescales, conservation laws) that a learned operator can be checked
  against;
* provenance, units, license, limitations, and Cantera provenance where
  relevant.

Oracle sections are computed by :func:`compute_card_oracle`;
``scripts/build_reaction_cards.py`` writes them, and the tests recompute
them so cards and code cannot drift apart.

:meth:`ReactionNetworkBenchmark.generate` draws seeded initial conditions
(and control schedules), integrates every trajectory, and returns a
:class:`ReactionDataset`. Raw concentrations stay ``float64``;
:meth:`ReactionDataset.sequence` exposes each trajectory as a
:class:`~koopman_graph.data.GraphSnapshotSequence` with one node per
species, with ``timestamps`` and (for the CSTR) global ``control_inputs``.

Importing this module does not import scipy or cantera; the Cantera card
needs the ``[cantera]`` extra at generation time.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor
from torch_geometric.data import Data

from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.datasets.reaction.catalog import (
    brusselator,
    first_order_chain,
    lotka_volterra,
    michaelis_menten,
    reversible_dimerization,
)
from koopman_graph.datasets.reaction.cstr import van_de_vusse_cstr
from koopman_graph.datasets.reaction.gillespie import gillespie_ssa
from koopman_graph.datasets.reaction.graphs import (
    reaction_hyperedge_index,
    species_influence_graph,
)
from koopman_graph.datasets.reaction.integrate import (
    integrate_reaction_system,
    mass_action_jacobian,
)
from koopman_graph.datasets.reaction.oracles import (
    conservation_laws,
    eigenvalue_timescales,
    equilibrium,
    oracle_spectrum,
)
from koopman_graph.datasets.reaction.random_network import sample_random_network
from koopman_graph.datasets.reaction.system import ReactionSystem
from koopman_graph.datasets.reaction.transform import LogConcentrationTransform

PAYLOAD_FORMAT = "koopman_graph.reaction_dataset"
"""``format`` tag written into saved dataset payloads."""

PAYLOAD_FORMAT_VERSION = 1
"""Current payload schema version accepted by :meth:`ReactionDataset.load`."""

GraphEncoding = Literal["species_influence", "reaction_hyperedge"]
"""Topology attached to each snapshot (see :mod:`.graphs`)."""

_PACKAGE = "koopman_graph.datasets.reaction"
_CATALOG_FACTORIES = {
    "first_order_chain": first_order_chain,
    "reversible_dimerization": reversible_dimerization,
    "michaelis_menten": michaelis_menten,
    "brusselator": brusselator,
    "lotka_volterra": lotka_volterra,
}


@dataclass(frozen=True, eq=False)
class ReactionDataset:
    """Generated reaction-network trajectories plus the card that defines them.

    Attributes
    ----------
    dataset_id : str
        Card identifier (e.g. ``"reaction_chain_v1"``).
    card : dict
        Parsed FAIR card.
    system : ReactionSystem
        Mass-action system the data were generated from (for the CSTR, rate
        constants are at the nominal temperature; for Cantera, the
        converted mechanism at the card's reference state).
    times : Tensor
        ``float64`` sample times ``(num_timesteps,)`` in the card's time unit.
    concentrations : Tensor
        ``float64`` concentrations ``(num_trajectories, num_timesteps,
        num_species)`` in the card's concentration unit.
    controls : Tensor or None
        ``float64`` inputs ``(num_trajectories, num_timesteps, num_controls)``;
        row ``t`` drives ``t → t+1`` (the last row repeats the last
        interval). ``None`` for uncontrolled datasets.
    transform : LogConcentrationTransform or None
        Feature transform applied by :meth:`sequence`.
    seed : int
        Seed of the generator that drew initial conditions and controls.
    simulator : str
        Simulator actually used (``"expm"``, ``"rk4"``, ``"ssa"``,
        ``"cstr"``, or ``"cantera"``).
    volume : float or None
        SSA volume when ``simulator == "ssa"``.
    """

    dataset_id: str
    card: dict[str, Any]
    system: ReactionSystem
    times: Tensor
    concentrations: Tensor
    controls: Tensor | None
    transform: LogConcentrationTransform | None
    seed: int
    simulator: str
    volume: float | None = None

    @property
    def num_trajectories(self) -> int:
        """Return the number of trajectories.

        Returns
        -------
        int
            Leading dimension of :attr:`concentrations`.
        """
        return int(self.concentrations.shape[0])

    @property
    def graph_encoding(self) -> str:
        """Return the card's graph encoding name.

        Returns
        -------
        str
            ``"species_influence"`` or ``"reaction_hyperedge"``.
        """
        return str(self.card["graph_encoding"])

    def features(self, index: int, *, dtype: torch.dtype = torch.float32) -> Tensor:
        """Return node features ``(num_timesteps, num_species, 1)`` of one trajectory.

        Parameters
        ----------
        index : int
            Trajectory index.
        dtype : torch.dtype, optional
            Output dtype. Default is ``torch.float32``.

        Returns
        -------
        Tensor
            Transformed concentrations when :attr:`transform` is set, raw
            concentrations otherwise.

        Raises
        ------
        IndexError
            If ``index`` is out of range.
        """
        self._check_index(index)
        c = self.concentrations[index]
        values = c if self.transform is None else self.transform.forward(c)
        return values.unsqueeze(-1).to(dtype)

    def sequence(
        self, index: int, *, dtype: torch.dtype = torch.float32
    ) -> GraphSnapshotSequence:
        """Return one trajectory as a species-node snapshot sequence.

        Parameters
        ----------
        index : int
            Trajectory index in ``[0, num_trajectories)``.
        dtype : torch.dtype, optional
            Feature dtype. Default is ``torch.float32``.

        Returns
        -------
        GraphSnapshotSequence
            ``num_timesteps`` snapshots with ``x`` of shape
            ``(num_species, 1)``, shared ``edge_index`` / ``edge_weight``
            from :func:`species_influence_graph` (plus ``hyperedge_index``
            for ``"reaction_hyperedge"``), ``timestamps``, and global
            ``control_inputs`` when present.

        Raises
        ------
        IndexError
            If ``index`` is out of range.
        """
        self._check_index(index)
        edge_index, edge_weight = species_influence_graph(self.system)
        hyperedge_index = (
            reaction_hyperedge_index(self.system)
            if self.graph_encoding == "reaction_hyperedge"
            else None
        )
        features = self.features(index, dtype=dtype)
        snapshots = []
        for x in features:
            data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight.to(dtype))
            if hyperedge_index is not None:
                data.hyperedge_index = hyperedge_index
            snapshots.append(data)
        controls = None if self.controls is None else self.controls[index].to(dtype)
        return GraphSnapshotSequence(
            snapshots,
            timestamps=self.times.to(dtype=torch.float64),
            control_inputs=controls,
        )

    def _check_index(self, index: int) -> None:
        """Raise ``IndexError`` unless ``0 <= index < num_trajectories``.

        Parameters
        ----------
        index
            See signature."""
        if not 0 <= index < self.num_trajectories:
            msg = f"index {index} out of range for {self.num_trajectories} trajectories"
            raise IndexError(msg)

    def subset(self, indices: list[int] | Tensor) -> ReactionDataset:
        """Return a dataset restricted to the given trajectory indices.

        Parameters
        ----------
        indices : list of int or Tensor
            Trajectory indices, in the order to keep.

        Returns
        -------
        ReactionDataset
            Same card, system, and times; selected trajectories only.
        """
        index = torch.as_tensor(indices, dtype=torch.long)
        return ReactionDataset(
            dataset_id=self.dataset_id,
            card=self.card,
            system=self.system,
            times=self.times,
            concentrations=self.concentrations[index],
            controls=None if self.controls is None else self.controls[index],
            transform=self.transform,
            seed=self.seed,
            simulator=self.simulator,
            volume=self.volume,
        )

    def save(self, path: str | Path) -> Path:
        """Write the dataset to a versioned ``torch.save`` payload.

        Parameters
        ----------
        path : str or Path
            Output file (parent directories are created).

        Returns
        -------
        Path
            The written path.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": PAYLOAD_FORMAT,
            "format_version": PAYLOAD_FORMAT_VERSION,
            "dataset_id": self.dataset_id,
            "card_json": json.dumps(self.card),
            "system_json": json.dumps(self.system.to_dict()),
            "times": self.times.clone(),
            "concentrations": self.concentrations.clone(),
            "controls": None if self.controls is None else self.controls.clone(),
            "transform_floor": None if self.transform is None else self.transform.floor,
            "seed": self.seed,
            "simulator": self.simulator,
            "volume": self.volume,
        }
        torch.save(payload, target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> ReactionDataset:
        """Read a payload written by :meth:`save` (``weights_only`` loading).

        Parameters
        ----------
        path : str or Path
            Payload file.

        Returns
        -------
        ReactionDataset
            The saved dataset.

        Raises
        ------
        ValueError
            If the payload format or version is not recognised.
        """
        payload = torch.load(Path(path), weights_only=True)
        if not isinstance(payload, dict) or payload.get("format") != PAYLOAD_FORMAT:
            msg = f"{path} is not a {PAYLOAD_FORMAT} payload"
            raise ValueError(msg)
        version = payload.get("format_version")
        if version != PAYLOAD_FORMAT_VERSION:
            msg = (
                f"unsupported reaction dataset format_version {version!r}; "
                f"expected {PAYLOAD_FORMAT_VERSION}"
            )
            raise ValueError(msg)
        floor = payload["transform_floor"]
        return cls(
            dataset_id=str(payload["dataset_id"]),
            card=json.loads(payload["card_json"]),
            system=ReactionSystem.from_dict(json.loads(payload["system_json"])),
            times=payload["times"],
            concentrations=payload["concentrations"],
            controls=payload["controls"],
            transform=None if floor is None else LogConcentrationTransform(floor),
            seed=int(payload["seed"]),
            simulator=str(payload["simulator"]),
            volume=payload["volume"],
        )


@dataclass(frozen=True, eq=False)
class ReactionDatasetSplit:
    """Trajectory-level train / validation / test split.

    Attributes
    ----------
    train : ReactionDataset
        First trajectories by index.
    val : ReactionDataset
        Following trajectories.
    test : ReactionDataset
        Last trajectories.
    """

    train: ReactionDataset
    val: ReactionDataset
    test: ReactionDataset


class ReactionNetworkBenchmark:
    """Card-defined, seeded mass-action reaction-network datasets.

    Notes
    -----
    Dataset card (per ``dataset_id``; see the JSON for exact values):

    * **Scope:** diagnostic / teaching data for checking whether a learned
      latent operator recovers slow kinetic timescales and respects
      conservation; not a production kinetics benchmark
    * **Format:** per-trajectory ``GraphSnapshotSequence`` (one node per
      species, ``float32`` features), plus raw ``float64`` concentrations
      and oracle references in the card
    * **Source:** in-repo generators; the Cantera card reads ``h2o2.yaml``
      from the installed Cantera package (illustration-only input)
    * **License:** Apache-2.0 for generated data; Cantera is BSD-3-Clause
    * **Version:** encoded in ``dataset_id`` (``*_v1``) and
      :data:`PAYLOAD_FORMAT_VERSION`
    """

    @staticmethod
    def dataset_ids() -> tuple[str, ...]:
        """List the packaged dataset cards.

        Returns
        -------
        tuple of str
            Sorted card identifiers.
        """
        root = resources.files(_PACKAGE).joinpath("data")
        return tuple(
            sorted(
                entry.name.removesuffix(".json")
                for entry in root.iterdir()
                if entry.name.startswith("reaction_") and entry.name.endswith(".json")
            )
        )

    @staticmethod
    def load_card(dataset_id: str) -> dict[str, Any]:
        """Load a packaged FAIR card.

        Parameters
        ----------
        dataset_id : str
            Card identifier from :meth:`dataset_ids`.

        Returns
        -------
        dict
            Parsed card.

        Raises
        ------
        ValueError
            If no card has this identifier.
        """
        if dataset_id not in ReactionNetworkBenchmark.dataset_ids():
            known = ", ".join(ReactionNetworkBenchmark.dataset_ids())
            msg = f"unknown reaction dataset {dataset_id!r}; known: {known}"
            raise ValueError(msg)
        resource = resources.files(_PACKAGE).joinpath(f"data/{dataset_id}.json")
        return json.loads(resource.read_text(encoding="utf-8"))

    @classmethod
    def generate(
        cls,
        dataset_id: str = "reaction_chain_v1",
        *,
        num_trajectories: int | None = None,
        num_timesteps: int | None = None,
        seed: int | None = None,
        simulator: Literal["card", "ssa"] = "card",
        volume: float = 1000.0,
    ) -> ReactionDataset:
        """Generate the trajectories a card defines.

        Parameters
        ----------
        dataset_id : str, optional
            Card identifier. Default is ``"reaction_chain_v1"``.
        num_trajectories : int or None, optional
            Trajectory count. Default is the card's
            ``default_num_trajectories``.
        num_timesteps : int or None, optional
            Samples per trajectory (>= 2). Default is the card's
            ``default_num_timesteps``.
        seed : int or None, optional
            Seed for initial conditions, controls, and SSA paths. Default is
            the card's ``default_seed``.
        simulator : {"card", "ssa"}, optional
            ``"card"`` (default) uses the card's deterministic simulator;
            ``"ssa"`` replaces it with Gillespie paths at ``volume``
            (uncontrolled, non-Cantera cards only).
        volume : float, optional
            SSA volume (counts = ``round(c0 * volume)``). Default is
            ``1000.0``.

        Returns
        -------
        ReactionDataset
            Concentrations, controls, and the card.

        Raises
        ------
        ValueError
            If arguments are invalid or ``simulator="ssa"`` does not apply.
        ImportError
            For the Cantera card when the ``[cantera]`` extra is missing.
        RuntimeError
            For the Cantera card when the installed mechanism's SHA-256 does
            not match the card.
        """
        card = cls.load_card(dataset_id)
        n_traj = _count(
            "num_trajectories",
            card["default_num_trajectories"]
            if num_trajectories is None
            else num_trajectories,
            low=1,
        )
        n_steps = _count(
            "num_timesteps",
            card["default_num_timesteps"] if num_timesteps is None else num_timesteps,
            low=2,
        )
        resolved_seed = _count(
            "seed", card["default_seed"] if seed is None else seed, low=0
        )
        kind = card["simulator"]["kind"]
        if simulator not in ("card", "ssa"):
            msg = f"simulator must be 'card' or 'ssa', got {simulator!r}"
            raise ValueError(msg)
        if simulator == "ssa" and kind not in ("expm", "rk4"):
            msg = f"simulator='ssa' is not available for {kind!r} cards"
            raise ValueError(msg)

        generator = torch.Generator().manual_seed(resolved_seed)
        times = float(card["time_step"]) * torch.arange(n_steps, dtype=torch.float64)
        system = system_from_card(card)
        controls: Tensor | None = None
        used_volume: float | None = None
        if kind == "cantera":
            concentrations = _simulate_cantera(card, times, n_traj, generator)
        else:
            initial = _sample_ranges(
                card["initial_ranges"], system.species, n_traj, generator
            )
            if simulator == "ssa":
                used_volume = float(volume)
                concentrations = _simulate_ssa(
                    system, initial, times, used_volume, generator
                )
            elif kind == "cstr":
                concentrations, controls = _simulate_cstr(
                    card, initial, times, generator
                )
            else:
                method = "expm" if kind == "expm" else "rk4"
                options = (
                    {}
                    if method == "expm"
                    else {"substeps": int(card["simulator"]["substeps"])}
                )
                states = integrate_reaction_system(
                    system, initial, times, method=method, **options
                )
                concentrations = states.permute(1, 0, 2).contiguous()
        floor = (card.get("transform") or {}).get("floor")
        return ReactionDataset(
            dataset_id=dataset_id,
            card=card,
            system=system,
            times=times,
            concentrations=concentrations,
            controls=controls,
            transform=None if floor is None else LogConcentrationTransform(floor),
            seed=resolved_seed,
            simulator="ssa" if simulator == "ssa" else kind,
            volume=used_volume,
        )

    @staticmethod
    def split(
        dataset: ReactionDataset,
        *,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        test_ratio: float = 0.2,
    ) -> ReactionDatasetSplit:
        """Split trajectories deterministically by index.

        Trajectories are i.i.d. draws, so contiguous index blocks are an
        unbiased split; the same seed always yields the same split.

        Parameters
        ----------
        dataset : ReactionDataset
            Dataset with at least three trajectories.
        train_ratio, val_ratio, test_ratio : float, optional
            Positive fractions summing to ``1``. Defaults are
            ``0.7 / 0.1 / 0.2``.

        Returns
        -------
        ReactionDatasetSplit
            Each part holds at least one trajectory.

        Raises
        ------
        ValueError
            If ratios are invalid or there are too few trajectories.
        """
        ratios = (train_ratio, val_ratio, test_ratio)
        if (
            any(r <= 0 or not math.isfinite(r) for r in ratios)
            or abs(sum(ratios) - 1.0) > 1e-6
        ):
            msg = f"ratios must be positive and sum to 1, got {ratios}"
            raise ValueError(msg)
        n = dataset.num_trajectories
        if n < 3:
            msg = f"need at least 3 trajectories to split, got {n}"
            raise ValueError(msg)
        n_val = max(1, round(val_ratio * n))
        n_test = max(1, round(test_ratio * n))
        n_train = n - n_val - n_test
        if n_train < 1:
            msg = f"{n} trajectories leave no training data for ratios {ratios}"
            raise ValueError(msg)
        return ReactionDatasetSplit(
            train=dataset.subset(list(range(n_train))),
            val=dataset.subset(list(range(n_train, n_train + n_val))),
            test=dataset.subset(list(range(n_train + n_val, n))),
        )


def system_from_card(card: dict[str, Any]) -> ReactionSystem:
    """Build the mass-action system a card's generator section describes.

    Parameters
    ----------
    card : dict
        Parsed card.

    Returns
    -------
    ReactionSystem
        The system (Cantera cards need the ``[cantera]`` extra).

    Raises
    ------
    ValueError
        If the generator kind or catalog name is unknown.
    RuntimeError
        For a Cantera card whose recorded mechanism SHA-256 differs from the
        installed mechanism file.
    """
    generator = card["generator"]
    kind = generator["kind"]
    params = dict(generator.get("params", {}))
    if kind == "catalog":
        name = generator["name"]
        if name not in _CATALOG_FACTORIES:
            msg = f"unknown catalog system {name!r}"
            raise ValueError(msg)
        return _CATALOG_FACTORIES[name](**params)
    if kind == "random_network":
        return sample_random_network(**params).system
    if kind == "cstr":
        return van_de_vusse_cstr(**params).system
    if kind == "cantera":
        from koopman_graph.datasets.reaction.cantera_io import (
            load_cantera_mechanism,
            mechanism_sha256,
            resolve_mechanism_path,
        )

        cantera = card["cantera"]
        installed = mechanism_sha256(resolve_mechanism_path(cantera["mechanism"]))
        if installed != cantera["sha256"]:
            msg = (
                f"installed {cantera['mechanism']} has SHA-256 {installed}, but "
                f"card {card.get('dataset_id', '?')!r} was built from "
                f"{cantera['sha256']} (Cantera {cantera['cantera_version']}); "
                "the generated data would not match the card"
            )
            raise RuntimeError(msg)
        return load_cantera_mechanism(
            cantera["mechanism"],
            temperature_k=cantera["temperature_k"],
            pressure_pa=cantera["pressure_pa"],
            mole_fractions=cantera["reference_mole_fractions"],
            state_dependent=cantera["state_dependent"],
        ).system
    msg = f"unknown generator kind {kind!r}"
    raise ValueError(msg)


def compute_card_oracle(card: dict[str, Any]) -> dict[str, Any]:
    """Compute the oracle section for a card from its generator spec.

    Parameters
    ----------
    card : dict
        Card with ``generator``, ``initial_ranges`` (or ``cantera``) and,
        for the CSTR, ``controls``.

    Returns
    -------
    dict
        ``state`` (how the reference point was found), ``reference_state``,
        ``num_conservation_laws``, ``conservation_laws``,
        ``dynamic_eigenvalues_real`` / ``_imag`` (slowest first), and
        ``timescales`` (``None`` for infinite). For the CSTR the spectrum
        is that of the controlled field ``J - D I`` at the nominal-input
        steady state.
    """
    kind = card["generator"]["kind"]
    if kind == "cstr":
        controls = card["controls"]["ranges"]
        temperature = sum(controls["temperature_k"]) / 2.0
        dilution = sum(controls["dilution_rate"]) / 2.0
        reactor = van_de_vusse_cstr(**card["generator"].get("params", {}))
        num_species = reactor.system.num_species
        state = reactor.steady_state(temperature_k=temperature, dilution_rate=dilution)
        jac = mass_action_jacobian(reactor.system.at_temperature(temperature), state)
        jac = jac - dilution * torch.eye(num_species, dtype=torch.float64)
        eigenvalues = torch.linalg.eigvals(jac)
        eigenvalues = eigenvalues[torch.argsort(eigenvalues.real, descending=True)]
        return _oracle_dict(
            "cstr_steady_state_at_control_midpoint",
            state,
            eigenvalues,
            torch.zeros(0, num_species, dtype=torch.float64),
        )
    system = system_from_card(card)
    if kind == "cantera":
        from koopman_graph.datasets.reaction.cantera_io import require_cantera

        ct = require_cantera()
        cantera = card["cantera"]
        gas = ct.Solution(cantera["mechanism"])
        gas.TPX = (
            cantera["temperature_k"],
            cantera["pressure_pa"],
            cantera["reference_mole_fractions"],
        )
        gas.equilibrate("TV")
        state = torch.tensor(gas.concentrations, dtype=torch.float64)
        label = "cantera_thermochemical_equilibrium_TV"
    else:
        reference = _range_midpoints(card["initial_ranges"], system.species)
        relax = card.get("oracle_relaxation_time")
        guess = None
        label = "equilibrium_from_range_midpoints"
        if relax is not None:
            # Warm-start Newton from an RK4 relaxation (networks that drain
            # species onto a boundary steady state converge slowly otherwise).
            steps_per_unit = int(card["simulator"]["substeps"]) / float(
                card["time_step"]
            )
            horizon = torch.tensor([0.0, float(relax)], dtype=torch.float64)
            guess = integrate_reaction_system(
                system,
                reference,
                horizon,
                method="rk4",
                substeps=max(1, math.ceil(steps_per_unit * float(relax))),
            )[-1]
            label = "equilibrium_from_range_midpoints_after_relaxation"
        state = equilibrium(system, reference, guess=guess)
    spectrum = oracle_spectrum(system, state)
    return _oracle_dict(
        label, state, spectrum.dynamic_eigenvalues, conservation_laws(system)
    )


def _oracle_dict(
    label: str, state: Tensor, eigenvalues: Tensor, laws: Tensor
) -> dict[str, Any]:
    """Pack oracle quantities into JSON-compatible primitives.

    Parameters
    ----------
    label
        See signature.
    state
        See signature.
    eigenvalues
        See signature.
    laws
        See signature.

    Returns
    -------
        See signature."""
    real = eigenvalues.real
    timescales = [
        None if math.isinf(t) else t
        for t in eigenvalue_timescales(eigenvalues).tolist()
    ]
    return {
        "state": label,
        "reference_state": state.tolist(),
        "num_conservation_laws": int(laws.shape[0]),
        "conservation_laws": laws.tolist(),
        "dynamic_eigenvalues_real": real.tolist(),
        "dynamic_eigenvalues_imag": eigenvalues.imag.tolist(),
        "timescales": timescales,
    }


def _range_midpoints(
    ranges: dict[str, list[float]], species: tuple[str, ...]
) -> Tensor:
    """Return per-species range midpoints in species order.

    Parameters
    ----------
    ranges
        See signature.
    species
        See signature.

    Returns
    -------
        See signature."""
    _check_ranges(ranges, species)
    return torch.tensor(
        [(ranges[s][0] + ranges[s][1]) / 2.0 for s in species], dtype=torch.float64
    )


def _sample_ranges(
    ranges: dict[str, list[float]],
    species: tuple[str, ...],
    count: int,
    generator: torch.Generator,
) -> Tensor:
    """Draw ``count`` uniform samples inside per-species ranges.

    Parameters
    ----------
    ranges
        See signature.
    species
        See signature.
    count
        See signature.
    generator
        See signature.

    Returns
    -------
        See signature."""
    _check_ranges(ranges, species)
    low = torch.tensor([ranges[s][0] for s in species], dtype=torch.float64)
    high = torch.tensor([ranges[s][1] for s in species], dtype=torch.float64)
    u = torch.rand(count, len(species), generator=generator, dtype=torch.float64)
    return low + (high - low) * u


def _check_ranges(ranges: dict[str, list[float]], species: tuple[str, ...]) -> None:
    """Validate that ``ranges`` covers exactly ``species`` with ``0 <= lo <= hi``.

    Parameters
    ----------
    ranges
        See signature.
    species
        See signature."""
    if set(ranges) != set(species):
        msg = (
            f"initial_ranges keys {sorted(ranges)} do not match species {list(species)}"
        )
        raise ValueError(msg)
    for name, (lo, hi) in ranges.items():
        if not (0.0 <= lo <= hi) or not math.isfinite(hi):
            msg = (
                f"initial range for {name!r} must satisfy 0 <= low <= high, "
                f"got {[lo, hi]}"
            )
            raise ValueError(msg)


def _simulate_ssa(
    system: ReactionSystem,
    initial: Tensor,
    times: Tensor,
    volume: float,
    generator: torch.Generator,
) -> Tensor:
    """Run one Gillespie path per initial state and sample it on ``times``.

    Each path gets its own seed drawn from ``generator``, so paths are
    independent within and across dataset seeds.

    Parameters
    ----------
    system
        See signature.
    initial
        See signature.
    times
        See signature.
    volume
        See signature.
    generator
        See signature.

    Returns
    -------
        See signature."""
    if not math.isfinite(volume) or volume <= 0.0:
        msg = f"volume must be finite and > 0, got {volume!r}"
        raise ValueError(msg)
    t_end = float(times[-1])
    path_seeds = torch.randint(
        0, 2**62, (initial.shape[0],), generator=generator, dtype=torch.long
    ).tolist()
    paths = []
    for c0, path_seed in zip(initial, path_seeds, strict=True):
        counts = torch.round(c0 * volume).to(torch.long)
        path = gillespie_ssa(system, counts, t_end, volume=volume, seed=path_seed)
        paths.append(path.concentrations_on_grid(times))
    return torch.stack(paths)


def _simulate_cstr(
    card: dict[str, Any],
    initial: Tensor,
    times: Tensor,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """Simulate the CSTR under seeded piecewise-constant control schedules.

    Parameters
    ----------
    card
        See signature.
    initial
        See signature.
    times
        See signature.
    generator
        See signature.

    Returns
    -------
        See signature."""
    reactor = van_de_vusse_cstr(**card["generator"].get("params", {}))
    spec = card["controls"]
    names = spec["names"]
    hold = int(spec["hold_steps"])
    substeps = int(card["simulator"]["substeps"])
    num_intervals = int(times.shape[0]) - 1
    num_blocks = math.ceil(num_intervals / hold)
    states, schedules = [], []
    for c0 in initial:
        columns = []
        for name in names:
            lo, hi = spec["ranges"][name]
            block = lo + (hi - lo) * torch.rand(
                num_blocks, generator=generator, dtype=torch.float64
            )
            columns.append(block.repeat_interleave(hold)[:num_intervals])
        schedule = torch.stack(columns, dim=-1)  # (T - 1, C)
        states.append(
            reactor.simulate(
                c0,
                times,
                temperature_k=schedule[:, names.index("temperature_k")],
                dilution_rate=schedule[:, names.index("dilution_rate")],
                substeps=substeps,
            )
        )
        schedules.append(torch.cat([schedule, schedule[-1:]], dim=0))
    return torch.stack(states), torch.stack(schedules)


def _simulate_cantera(
    card: dict[str, Any],
    times: Tensor,
    count: int,
    generator: torch.Generator,
) -> Tensor:
    """Run seeded Cantera reactors from mole-fraction ranges.

    Parameters
    ----------
    card
        See signature.
    times
        See signature.
    count
        See signature.
    generator
        See signature.

    Returns
    -------
        See signature."""
    from koopman_graph.datasets.reaction.cantera_io import simulate_cantera_reactor

    cantera = card["cantera"]
    ranges = cantera["initial_mole_fraction_ranges"]
    names = sorted(ranges)
    runs = []
    for _ in range(count):
        u = torch.rand(len(names), generator=generator, dtype=torch.float64)
        mix = {
            name: ranges[name][0] + (ranges[name][1] - ranges[name][0]) * float(u[k])
            for k, name in enumerate(names)
        }
        trajectory = simulate_cantera_reactor(
            cantera["mechanism"],
            times=times,
            temperature_k=cantera["temperature_k"],
            pressure_pa=cantera["pressure_pa"],
            mole_fractions=mix,
            reactor=cantera["reactor"],
            isothermal=cantera["isothermal"],
        )
        runs.append(trajectory.concentrations)
    return torch.stack(runs)


def _count(name: str, value: Any, *, low: int) -> int:
    """Validate an integer argument ``>= low``.

    Parameters
    ----------
    name
        See signature.
    value
        See signature.
    low
        See signature.

    Returns
    -------
        See signature."""
    if isinstance(value, bool) or not isinstance(value, int) or value < low:
        msg = f"{name} must be an int >= {low}, got {value!r}"
        raise ValueError(msg)
    return value
