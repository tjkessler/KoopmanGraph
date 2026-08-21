"""Frozen ``benchmark_manifest_v1`` types for protocol-locked experiments.

This schema records dataset identity, splits, methods, seeds, horizons,
metrics, controls, and an optional UQ block. It does **not** run a
benchmark, write artifacts, or claim LibCity / BasicTS leaderboard
parity. Teaching GNN ports listed as methods must keep a non-empty
``deviations`` tuple.

Ratios, coverage, and SHA-256 digests are dimensionless. Horizon and
``history_len`` are snapshot counts. ``max_epochs`` is an epoch count.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

__all__ = [
    "BENCHMARK_METRICS",
    "BENCHMARK_TRACKS",
    "CONTROL_TOKENS",
    "DATASET_REF_KEYS",
    "EmptyMethodDeviationsError",
    "MANIFEST_KEYS",
    "METHOD_ROLES",
    "MIN_MANIFEST_SEEDS",
    "ManifestError",
    "METHOD_SPEC_KEYS",
    "RATIO_SUM_ATOL",
    "SCHEMA_VERSION",
    "ComputeBudget",
    "DatasetRef",
    "ExperimentManifest",
    "MethodSpec",
    "OODShiftSpec",
    "PreprocessingSpec",
    "SplitSpec",
    "UQSpec",
    "manifest_from_mapping",
    "manifest_to_mapping",
]

SCHEMA_VERSION = "benchmark_manifest_v1"
MIN_MANIFEST_SEEDS = 3
RATIO_SUM_ATOL = 1e-6
_SHA256_HEX = re.compile(r"^[0-9a-fA-F]{64}$")

BENCHMARK_TRACKS: frozenset[str] = frozenset(
    {"telemetry", "multiphysics", "topology_transfer"}
)
BENCHMARK_METRICS: frozenset[str] = frozenset({"mae", "rmse", "mape", "mse"})
CONTROL_TOKENS: frozenset[str] = frozenset({"pernode", "joint_ls", "hold_last"})
METHOD_ROLES: frozenset[str] = frozenset(
    {
        "koopman",
        "teaching_gnn",
        "leaderboard",
        "control",
        "dedicated_library",
    }
)
_TOPOLOGY_REQUIRED_CONTROLS = frozenset({"hold_last", "pernode", "joint_ls"})
_FACTORIZATION_CONTROLS = frozenset({"pernode", "joint_ls"})

MANIFEST_KEYS: frozenset[str] = frozenset(
    {
        "manifest_id",
        "schema_version",
        "track",
        "dataset",
        "split",
        "preprocessing",
        "methods",
        "seeds",
        "horizons",
        "metrics",
        "ood_shifts",
        "uq",
        "compute_budget",
        "controls",
    }
)
DATASET_REF_KEYS: frozenset[str] = frozenset({"name", "version", "sha256", "card"})
_SPLIT_KEYS: frozenset[str] = frozenset(
    {"train_ratio", "val_ratio", "test_ratio", "history_len"}
)
_PREPROCESS_KEYS: frozenset[str] = frozenset({"zscore", "notes"})
METHOD_SPEC_KEYS: frozenset[str] = frozenset({"name", "role", "deviations", "version"})
_OOD_KEYS: frozenset[str] = frozenset({"name", "kind", "description"})
_UQ_KEYS: frozenset[str] = frozenset({"method", "coverage"})
_BUDGET_KEYS: frozenset[str] = frozenset({"max_epochs", "device", "notes"})

BenchmarkTrack = Literal["telemetry", "multiphysics", "topology_transfer"]
BenchmarkMetric = Literal["mae", "rmse", "mape", "mse"]
ControlToken = Literal["pernode", "joint_ls", "hold_last"]
MethodRole = Literal[
    "koopman",
    "teaching_gnn",
    "leaderboard",
    "control",
    "dedicated_library",
]


class ManifestError(ValueError):
    """Invalid benchmark-manifest mapping, field, or file.

    Notes
    -----
    Raised by schema construction and YAML/JSON loaders.
    """


class EmptyMethodDeviationsError(ValueError):
    """Raised when a teaching GNN method has empty ``deviations``.

    Notes
    -----
    Teaching ports cannot claim zero deviation from a paper / LibCity
    protocol when listed on a manifest.
    """


def _finite_float(value: float) -> bool:
    """Return whether ``value`` is a finite Python float.

    Parameters
    ----------
    value : float
        Scalar to test.

    Returns
    -------
    bool
        ``True`` when ``value`` is finite.
    """
    return value == value and value not in (float("inf"), float("-inf"))


def _require_mapping(value: object, *, name: str) -> dict[str, Any]:
    """Return ``value`` as a string-key mapping.

    Parameters
    ----------
    value : object
        Candidate mapping.
    name : str
        Field name for errors.

    Returns
    -------
    dict
        The mapping.

    Raises
    ------
    ManifestError
        If ``value`` is not a ``dict`` with string keys.
    """
    if not isinstance(value, dict):
        msg = f"{name} must be a mapping, got {type(value).__name__}"
        raise ManifestError(msg)
    for key in value:
        if not isinstance(key, str):
            msg = f"{name} keys must be strings, got {type(key).__name__}"
            raise ManifestError(msg)
    return value


def _reject_unknown(
    mapping: Mapping[str, Any],
    allowed: frozenset[str],
    *,
    name: str,
) -> None:
    """Raise if ``mapping`` contains keys outside ``allowed``.

    Parameters
    ----------
    mapping : mapping
        Parsed object.
    allowed : frozenset of str
        Permitted keys.
    name : str
        Field name for errors.

    Raises
    ------
    ManifestError
        If an unknown key is present.
    """
    extra = sorted(set(mapping) - allowed)
    if extra:
        msg = f"unknown {name} keys: {', '.join(extra)}"
        raise ManifestError(msg)


def _nonempty_str(value: object, *, name: str) -> str:
    """Return a stripped non-empty string.

    Parameters
    ----------
    value : object
        Candidate string.
    name : str
        Field name for errors.

    Returns
    -------
    str
        Non-empty string.

    Raises
    ------
    ManifestError
        If ``value`` is empty or not a string.
    """
    if not isinstance(value, str) or not value.strip():
        msg = f"{name} must be a non-empty string, got {value!r}"
        raise ManifestError(msg)
    return value


def _as_str_tuple(value: object, *, name: str) -> tuple[str, ...]:
    """Return a tuple of strings (empty strings rejected).

    Parameters
    ----------
    value : object
        Sequence of strings.
    name : str
        Field name for errors.

    Returns
    -------
    tuple of str
        Coerced strings.

    Raises
    ------
    ManifestError
        If ``value`` is not a string sequence.
    """
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        msg = f"{name} must be a sequence of strings"
        raise ManifestError(msg)
    parsed: list[str] = []
    for item in value:
        if not isinstance(item, str) or item == "":
            msg = f"{name} must contain non-empty strings, got {item!r}"
            raise ManifestError(msg)
        parsed.append(item)
    return tuple(parsed)


def _as_int_tuple(value: object, *, name: str, minimum: int) -> tuple[int, ...]:
    """Return a tuple of ints each at least ``minimum``.

    Parameters
    ----------
    value : object
        Sequence of ints.
    name : str
        Field name for errors.
    minimum : int
        Inclusive lower bound.

    Returns
    -------
    tuple of int
        Coerced ints.

    Raises
    ------
    ManifestError
        If ``value`` is not a sequence of ints in range.
    """
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        msg = f"{name} must be a sequence of ints"
        raise ManifestError(msg)
    parsed: list[int] = []
    for item in value:
        if type(item) is not int:
            msg = f"{name} must contain ints, got {item!r}"
            raise ManifestError(msg)
        if item < minimum:
            msg = f"{name} entries must be >= {minimum}, got {item}"
            raise ManifestError(msg)
        parsed.append(item)
    return tuple(parsed)


def _require_unique(values: tuple[Any, ...], *, name: str) -> None:
    """Reject duplicate entries.

    Parameters
    ----------
    values : tuple
        Sequence to check.
    name : str
        Field name for errors.

    Raises
    ------
    ManifestError
        If any value is repeated.
    """
    if len(set(values)) != len(values):
        msg = f"{name} must not contain duplicates, got {values!r}"
        raise ManifestError(msg)


def _ratio(value: object, *, name: str) -> float:
    """Return a split ratio in ``(0, 1]``.

    Parameters
    ----------
    value : object
        Candidate ratio (dimensionless).
    name : str
        Field name for errors.

    Returns
    -------
    float
        Finite ratio.

    Raises
    ------
    ManifestError
        If the ratio is out of range.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f"{name} must be a float, got {type(value).__name__}"
        raise ManifestError(msg)
    ratio = float(value)
    if not _finite_float(ratio) or not (0.0 < ratio <= 1.0):
        msg = f"{name} must satisfy 0 < ratio <= 1, got {ratio}"
        raise ManifestError(msg)
    return ratio


def _require_bool(value: object, *, name: str) -> bool:
    """Return a genuine ``bool``.

    Parameters
    ----------
    value : object
        Candidate flag.
    name : str
        Field name for errors.

    Returns
    -------
    bool
        The flag.

    Raises
    ------
    ManifestError
        If ``value`` is not a ``bool``.
    """
    if type(value) is not bool:
        msg = f"{name} must be a bool, got {type(value).__name__}"
        raise ManifestError(msg)
    return value


@dataclass(frozen=True)
class DatasetRef:
    """Named dataset with a SHA-256 digest and card path.

    ``sha256`` is a 64-character hexadecimal digest of the dataset
    payload bytes, not a hash of this manifest document.

    Attributes
    ----------
    name : str
        Dataset identifier.
    version : str
        Dataset version label.
    sha256 : str
        Hex SHA-256 of the payload.
    card : str
        Path or URI of the dataset card.
    """

    name: str
    version: str
    sha256: str
    card: str

    def __post_init__(self) -> None:
        """Validate identity strings and the digest.

        Raises
        ------
        ManifestError
            If a field is empty or ``sha256`` is not 64 hex characters.
        """
        object.__setattr__(self, "name", _nonempty_str(self.name, name="dataset.name"))
        object.__setattr__(
            self, "version", _nonempty_str(self.version, name="dataset.version")
        )
        object.__setattr__(self, "card", _nonempty_str(self.card, name="dataset.card"))
        digest = _nonempty_str(self.sha256, name="dataset.sha256")
        if _SHA256_HEX.fullmatch(digest) is None:
            msg = "dataset.sha256 must be a 64-character hex digest"
            raise ManifestError(msg)
        object.__setattr__(self, "sha256", digest.lower())


@dataclass(frozen=True)
class SplitSpec:
    """Temporal split fractions and optional lookback.

    Ratios are dimensionless and must sum to 1 within
    :data:`RATIO_SUM_ATOL`. ``history_len`` is a snapshot count.

    Attributes
    ----------
    train_ratio, val_ratio, test_ratio : float
        Split fractions in ``(0, 1]``.
    history_len : int or None
        Optional lookback length. Default ``None``.
    """

    train_ratio: float
    val_ratio: float
    test_ratio: float
    history_len: int | None = None

    def __post_init__(self) -> None:
        """Validate ratios and optional lookback.

        Raises
        ------
        ManifestError
            If ratios are invalid or ``history_len`` is not a positive int.
        """
        train = _ratio(self.train_ratio, name="split.train_ratio")
        val = _ratio(self.val_ratio, name="split.val_ratio")
        test = _ratio(self.test_ratio, name="split.test_ratio")
        object.__setattr__(self, "train_ratio", train)
        object.__setattr__(self, "val_ratio", val)
        object.__setattr__(self, "test_ratio", test)
        ratio_sum = train + val + test
        if abs(ratio_sum - 1.0) > RATIO_SUM_ATOL:
            msg = (
                "split ratios must sum to 1 within "
                f"atol={RATIO_SUM_ATOL}, got sum={ratio_sum}"
            )
            raise ManifestError(msg)
        if self.history_len is not None and (
            type(self.history_len) is not int or self.history_len < 1
        ):
            msg = (
                "split.history_len must be an int >= 1 or None, "
                f"got {self.history_len!r}"
            )
            raise ManifestError(msg)


@dataclass(frozen=True)
class PreprocessingSpec:
    """Declared preprocessing (not an executable pipeline).

    Attributes
    ----------
    zscore : bool
        Whether inputs are z-scored on the train split. Default ``False``.
    notes : tuple of str
        Optional free-text notes. Default ``()``.
    """

    zscore: bool = False
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate the flag and notes tuple.

        Raises
        ------
        ManifestError
            If ``zscore`` is not a bool or notes are invalid.
        """
        object.__setattr__(
            self, "zscore", _require_bool(self.zscore, name="preprocessing.zscore")
        )
        object.__setattr__(
            self, "notes", _as_str_tuple(self.notes, name="preprocessing.notes")
        )


@dataclass(frozen=True)
class MethodSpec:
    """One method listed on a manifest.

    ``role="teaching_gnn"`` requires a non-empty ``deviations`` tuple.
    ``role="dedicated_library"`` requires a version pin. Leaderboard
    adapters may have empty ``deviations``.

    Attributes
    ----------
    name : str
        Method identifier.
    role : {"koopman", "teaching_gnn", "leaderboard", "control", "dedicated_library"}
        Honesty class.
    deviations : tuple of str
        Protocol mismatches. Default ``()``.
    version : str or None
        Library version pin. Default ``None``.
    """

    name: str
    role: MethodRole
    deviations: tuple[str, ...] = ()
    version: str | None = None

    def __post_init__(self) -> None:
        """Validate role, deviations, and version pin.

        Raises
        ------
        EmptyMethodDeviationsError
            If a teaching GNN method has empty ``deviations``.
        ManifestError
            If ``role`` is unknown or a dedicated library lacks ``version``.
        """
        object.__setattr__(self, "name", _nonempty_str(self.name, name="method.name"))
        if self.role not in METHOD_ROLES:
            allowed = ", ".join(sorted(METHOD_ROLES))
            msg = f"method.role must be one of {allowed}, got {self.role!r}"
            raise ManifestError(msg)
        object.__setattr__(
            self, "deviations", _as_str_tuple(self.deviations, name="method.deviations")
        )
        if self.role == "teaching_gnn" and len(self.deviations) == 0:
            msg = (
                "teaching_gnn methods require non-empty deviations; "
                "teaching ports cannot claim zero deviation from a paper / "
                "LibCity protocol"
            )
            raise EmptyMethodDeviationsError(msg)
        if self.version is not None:
            object.__setattr__(
                self, "version", _nonempty_str(self.version, name="method.version")
            )
        elif self.role == "dedicated_library":
            msg = "dedicated_library methods require a non-empty version pin"
            raise ManifestError(msg)


@dataclass(frozen=True)
class OODShiftSpec:
    """Named out-of-distribution shift (descriptive, not executed).

    Attributes
    ----------
    name : str
        Shift identifier.
    kind : str
        Shift family (for example ``"rewire"`` or ``"unseen_n"``).
    description : str
        Optional prose. Default ``""``.
    """

    name: str
    kind: str
    description: str = ""

    def __post_init__(self) -> None:
        """Validate name and kind strings.

        Raises
        ------
        ManifestError
            If ``name`` or ``kind`` is empty or ``description`` is not a str.
        """
        object.__setattr__(self, "name", _nonempty_str(self.name, name="ood.name"))
        object.__setattr__(self, "kind", _nonempty_str(self.kind, name="ood.kind"))
        if not isinstance(self.description, str):
            msg = (
                "ood.description must be a string, "
                f"got {type(self.description).__name__}"
            )
            raise ManifestError(msg)


@dataclass(frozen=True)
class UQSpec:
    """Optional uncertainty-quantification declaration.

    ``coverage`` is a dimensionless target in ``(0, 1]`` when set.

    Attributes
    ----------
    method : str
        Named UQ procedure.
    coverage : float or None
        Optional coverage target. Default ``None``.
    """

    method: str
    coverage: float | None = None

    def __post_init__(self) -> None:
        """Validate method and optional coverage.

        Raises
        ------
        ManifestError
            If ``method`` is empty or ``coverage`` is out of range.
        """
        object.__setattr__(self, "method", _nonempty_str(self.method, name="uq.method"))
        if self.coverage is None:
            return
        if isinstance(self.coverage, bool) or not isinstance(
            self.coverage, (int, float)
        ):
            msg = (
                "uq.coverage must be a float or None, "
                f"got {type(self.coverage).__name__}"
            )
            raise ManifestError(msg)
        coverage = float(self.coverage)
        if not _finite_float(coverage) or not (0.0 < coverage <= 1.0):
            msg = f"uq.coverage must satisfy 0 < coverage <= 1, got {coverage}"
            raise ManifestError(msg)
        object.__setattr__(self, "coverage", coverage)


@dataclass(frozen=True)
class ComputeBudget:
    """Declared compute envelope (not a measured wall-clock).

    ``max_epochs`` is an epoch count, not a physical duration.

    Attributes
    ----------
    max_epochs : int
        Inclusive epoch budget.
    device : str
        Device label. Default ``"cpu"``.
    notes : tuple of str
        Optional notes. Default ``()``.
    """

    max_epochs: int
    device: str = "cpu"
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate epoch count and device label.

        Raises
        ------
        ManifestError
            If ``max_epochs`` is not a positive int.
        """
        if type(self.max_epochs) is not int or self.max_epochs < 1:
            msg = (
                "compute_budget.max_epochs must be an int >= 1, "
                f"got {self.max_epochs!r}"
            )
            raise ManifestError(msg)
        object.__setattr__(
            self, "device", _nonempty_str(self.device, name="compute_budget.device")
        )
        object.__setattr__(
            self, "notes", _as_str_tuple(self.notes, name="compute_budget.notes")
        )


@dataclass(frozen=True)
class ExperimentManifest:
    """Protocol-locked experiment declaration (schema ``benchmark_manifest_v1``).

    This record is not a runner and not a leaderboard host. Negative
    factorization or transfer outcomes remain allowed when a later
    runner executes the protocol.

    Attributes
    ----------
    manifest_id : str
        Stable identifier.
    schema_version : {"benchmark_manifest_v1"}
        Schema name.
    track : {"telemetry", "multiphysics", "topology_transfer"}
        Evidence track.
    dataset : DatasetRef
        Named payload with SHA-256.
    split : SplitSpec
        Temporal split.
    preprocessing : PreprocessingSpec
        Declared preprocessing.
    methods : tuple of MethodSpec
        Methods under comparison (non-empty).
    seeds : tuple of int
        At least :data:`MIN_MANIFEST_SEEDS` unique seeds.
    horizons : tuple of int
        Positive forecast horizons (snapshot counts).
    metrics : tuple of str
        Names from :data:`BENCHMARK_METRICS`.
    ood_shifts : tuple of OODShiftSpec
        Optional shifts. Default ``()``.
    uq : UQSpec or None
        Optional UQ block. Default ``None``.
    compute_budget : ComputeBudget
        Epoch / device envelope.
    controls : tuple of str
        Mandatory controls. ``telemetry`` / ``multiphysics`` require
        ``pernode`` and/or ``joint_ls``. ``topology_transfer`` requires
        ``hold_last``, ``pernode``, and ``joint_ls``.
    """

    manifest_id: str
    schema_version: Literal["benchmark_manifest_v1"]
    track: BenchmarkTrack
    dataset: DatasetRef
    split: SplitSpec
    preprocessing: PreprocessingSpec
    methods: tuple[MethodSpec, ...]
    seeds: tuple[int, ...]
    horizons: tuple[int, ...]
    metrics: tuple[str, ...]
    compute_budget: ComputeBudget
    controls: tuple[str, ...]
    ood_shifts: tuple[OODShiftSpec, ...] = ()
    uq: UQSpec | None = None

    def __post_init__(self) -> None:
        """Validate identity, grid, methods, and track controls.

        Raises
        ------
        ManifestError
            If a required field, control set, or grid is invalid.
        EmptyMethodDeviationsError
            If a teaching GNN method has empty ``deviations``.
        """
        object.__setattr__(
            self, "manifest_id", _nonempty_str(self.manifest_id, name="manifest_id")
        )
        if self.schema_version != SCHEMA_VERSION:
            msg = (
                f"schema_version must be {SCHEMA_VERSION!r}, "
                f"got {self.schema_version!r}"
            )
            raise ManifestError(msg)
        if self.track not in BENCHMARK_TRACKS:
            allowed = ", ".join(sorted(BENCHMARK_TRACKS))
            msg = f"track must be one of {allowed}, got {self.track!r}"
            raise ManifestError(msg)
        if not isinstance(self.dataset, DatasetRef):
            msg = "dataset must be a DatasetRef"
            raise ManifestError(msg)
        if not isinstance(self.split, SplitSpec):
            msg = "split must be a SplitSpec"
            raise ManifestError(msg)
        if not isinstance(self.preprocessing, PreprocessingSpec):
            msg = "preprocessing must be a PreprocessingSpec"
            raise ManifestError(msg)
        if not isinstance(self.compute_budget, ComputeBudget):
            msg = "compute_budget must be a ComputeBudget"
            raise ManifestError(msg)
        if self.uq is not None and not isinstance(self.uq, UQSpec):
            msg = "uq must be a UQSpec or None"
            raise ManifestError(msg)

        methods = self.methods
        if isinstance(methods, Sequence) and not isinstance(methods, (str, bytes)):
            coerced_methods = tuple(methods)
        else:
            msg = "methods must be a non-empty sequence of MethodSpec"
            raise ManifestError(msg)
        if not coerced_methods:
            msg = "methods must be a non-empty sequence of MethodSpec"
            raise ManifestError(msg)
        for method in coerced_methods:
            if not isinstance(method, MethodSpec):
                msg = f"methods entries must be MethodSpec, got {type(method).__name__}"
                raise ManifestError(msg)
        object.__setattr__(self, "methods", coerced_methods)

        seeds = _as_int_tuple(self.seeds, name="seeds", minimum=0)
        if len(seeds) < MIN_MANIFEST_SEEDS:
            msg = f"seeds must contain at least {MIN_MANIFEST_SEEDS} unique ints"
            raise ManifestError(msg)
        _require_unique(seeds, name="seeds")
        object.__setattr__(self, "seeds", seeds)

        horizons = _as_int_tuple(self.horizons, name="horizons", minimum=1)
        if not horizons:
            msg = "horizons must be a non-empty sequence of positive ints"
            raise ManifestError(msg)
        _require_unique(horizons, name="horizons")
        object.__setattr__(self, "horizons", horizons)

        metrics = _as_str_tuple(self.metrics, name="metrics")
        if not metrics:
            msg = "metrics must be a non-empty sequence"
            raise ManifestError(msg)
        _require_unique(metrics, name="metrics")
        unknown_metrics = sorted(set(metrics) - BENCHMARK_METRICS)
        if unknown_metrics:
            allowed = ", ".join(sorted(BENCHMARK_METRICS))
            msg = f"metrics must be a subset of {allowed}, got {unknown_metrics}"
            raise ManifestError(msg)
        object.__setattr__(self, "metrics", metrics)

        shifts = self.ood_shifts
        if isinstance(shifts, Sequence) and not isinstance(shifts, (str, bytes)):
            coerced_shifts = tuple(shifts)
        else:
            msg = "ood_shifts must be a sequence of OODShiftSpec"
            raise ManifestError(msg)
        for shift in coerced_shifts:
            if not isinstance(shift, OODShiftSpec):
                msg = (
                    "ood_shifts entries must be OODShiftSpec, "
                    f"got {type(shift).__name__}"
                )
                raise ManifestError(msg)
        object.__setattr__(self, "ood_shifts", coerced_shifts)

        controls = _as_str_tuple(self.controls, name="controls")
        _require_unique(controls, name="controls")
        unknown_controls = sorted(set(controls) - CONTROL_TOKENS)
        if unknown_controls:
            allowed = ", ".join(sorted(CONTROL_TOKENS))
            msg = f"controls must be a subset of {allowed}, got {unknown_controls}"
            raise ManifestError(msg)
        control_set = set(controls)
        if self.track == "topology_transfer":
            missing = sorted(_TOPOLOGY_REQUIRED_CONTROLS - control_set)
            if missing:
                msg = (
                    "topology_transfer controls must include hold_last, "
                    f"pernode, and joint_ls; missing {missing}"
                )
                raise ManifestError(msg)
        elif not (control_set & _FACTORIZATION_CONTROLS):
            msg = (
                f"{self.track} controls must include pernode and/or joint_ls, "
                f"got {controls}"
            )
            raise ManifestError(msg)
        object.__setattr__(self, "controls", controls)


def _dataset_from_mapping(payload: Mapping[str, Any]) -> DatasetRef:
    """Build :class:`DatasetRef` from a mapping.

    Parameters
    ----------
    payload : mapping
        Dataset fields.

    Returns
    -------
    DatasetRef
        Validated reference.
    """
    mapping = _require_mapping(payload, name="dataset")
    _reject_unknown(mapping, DATASET_REF_KEYS, name="dataset")
    missing = sorted(DATASET_REF_KEYS - set(mapping))
    if missing:
        msg = f"dataset missing required keys: {', '.join(missing)}"
        raise ManifestError(msg)
    return DatasetRef(
        name=mapping["name"],
        version=mapping["version"],
        sha256=mapping["sha256"],
        card=mapping["card"],
    )


def _split_from_mapping(payload: Mapping[str, Any]) -> SplitSpec:
    """Build :class:`SplitSpec` from a mapping.

    Parameters
    ----------
    payload : mapping
        Split fields.

    Returns
    -------
    SplitSpec
        Validated split.
    """
    mapping = _require_mapping(payload, name="split")
    _reject_unknown(mapping, _SPLIT_KEYS, name="split")
    missing = sorted({"train_ratio", "val_ratio", "test_ratio"} - set(mapping))
    if missing:
        msg = f"split missing required keys: {', '.join(missing)}"
        raise ManifestError(msg)
    return SplitSpec(
        train_ratio=mapping["train_ratio"],
        val_ratio=mapping["val_ratio"],
        test_ratio=mapping["test_ratio"],
        history_len=mapping.get("history_len"),
    )


def _preprocess_from_mapping(payload: Mapping[str, Any]) -> PreprocessingSpec:
    """Build :class:`PreprocessingSpec` from a mapping.

    Parameters
    ----------
    payload : mapping
        Preprocessing fields.

    Returns
    -------
    PreprocessingSpec
        Validated spec.
    """
    mapping = _require_mapping(payload, name="preprocessing")
    _reject_unknown(mapping, _PREPROCESS_KEYS, name="preprocessing")
    return PreprocessingSpec(
        zscore=mapping.get("zscore", False),
        notes=tuple(mapping["notes"]) if "notes" in mapping else (),
    )


def _method_from_mapping(payload: Mapping[str, Any]) -> MethodSpec:
    """Build :class:`MethodSpec` from a mapping.

    Parameters
    ----------
    payload : mapping
        Method fields.

    Returns
    -------
    MethodSpec
        Validated spec.
    """
    mapping = _require_mapping(payload, name="method")
    _reject_unknown(mapping, METHOD_SPEC_KEYS, name="method")
    missing = sorted({"name", "role"} - set(mapping))
    if missing:
        msg = f"method missing required keys: {', '.join(missing)}"
        raise ManifestError(msg)
    return MethodSpec(
        name=mapping["name"],
        role=mapping["role"],
        deviations=tuple(mapping["deviations"]) if "deviations" in mapping else (),
        version=mapping.get("version"),
    )


def _ood_from_mapping(payload: Mapping[str, Any]) -> OODShiftSpec:
    """Build :class:`OODShiftSpec` from a mapping.

    Parameters
    ----------
    payload : mapping
        Shift fields.

    Returns
    -------
    OODShiftSpec
        Validated spec.
    """
    mapping = _require_mapping(payload, name="ood_shift")
    _reject_unknown(mapping, _OOD_KEYS, name="ood_shift")
    missing = sorted({"name", "kind"} - set(mapping))
    if missing:
        msg = f"ood_shift missing required keys: {', '.join(missing)}"
        raise ManifestError(msg)
    return OODShiftSpec(
        name=mapping["name"],
        kind=mapping["kind"],
        description=mapping.get("description", ""),
    )


def _uq_from_mapping(payload: Mapping[str, Any]) -> UQSpec:
    """Build :class:`UQSpec` from a mapping.

    Parameters
    ----------
    payload : mapping
        UQ fields.

    Returns
    -------
    UQSpec
        Validated spec.
    """
    mapping = _require_mapping(payload, name="uq")
    _reject_unknown(mapping, _UQ_KEYS, name="uq")
    if "method" not in mapping:
        msg = "uq missing required key: method"
        raise ManifestError(msg)
    return UQSpec(method=mapping["method"], coverage=mapping.get("coverage"))


def _budget_from_mapping(payload: Mapping[str, Any]) -> ComputeBudget:
    """Build :class:`ComputeBudget` from a mapping.

    Parameters
    ----------
    payload : mapping
        Budget fields.

    Returns
    -------
    ComputeBudget
        Validated budget.
    """
    mapping = _require_mapping(payload, name="compute_budget")
    _reject_unknown(mapping, _BUDGET_KEYS, name="compute_budget")
    if "max_epochs" not in mapping:
        msg = "compute_budget missing required key: max_epochs"
        raise ManifestError(msg)
    return ComputeBudget(
        max_epochs=mapping["max_epochs"],
        device=mapping.get("device", "cpu"),
        notes=tuple(mapping["notes"]) if "notes" in mapping else (),
    )


def manifest_from_mapping(payload: Mapping[str, Any]) -> ExperimentManifest:
    """Build :class:`ExperimentManifest` from a JSON/YAML mapping.

    Unknown keys are rejected. ``ood_shifts`` and ``uq`` may be omitted.

    Parameters
    ----------
    payload : mapping
        Manifest document.

    Returns
    -------
    ExperimentManifest
        Validated frozen record.

    Raises
    ------
    ManifestError
        If keys or nested objects are invalid.
    EmptyMethodDeviationsError
        If a teaching GNN method has empty ``deviations``.
    """
    mapping = _require_mapping(payload, name="manifest")
    _reject_unknown(mapping, MANIFEST_KEYS, name="manifest")
    required = MANIFEST_KEYS - {"ood_shifts", "uq", "preprocessing"}
    missing = sorted(required - set(mapping))
    if missing:
        msg = f"manifest missing required keys: {', '.join(missing)}"
        raise ManifestError(msg)
    methods_raw = mapping["methods"]
    if not isinstance(methods_raw, Sequence) or isinstance(methods_raw, (str, bytes)):
        msg = "methods must be a sequence of mappings"
        raise ManifestError(msg)
    shifts_raw = mapping.get("ood_shifts", ())
    if not isinstance(shifts_raw, Sequence) or isinstance(shifts_raw, (str, bytes)):
        msg = "ood_shifts must be a sequence of mappings"
        raise ManifestError(msg)
    uq_raw = mapping.get("uq")
    preprocess_raw = mapping.get("preprocessing")
    return ExperimentManifest(
        manifest_id=mapping["manifest_id"],
        schema_version=mapping["schema_version"],
        track=mapping["track"],
        dataset=_dataset_from_mapping(mapping["dataset"]),
        split=_split_from_mapping(mapping["split"]),
        preprocessing=(
            _preprocess_from_mapping(preprocess_raw)
            if preprocess_raw is not None
            else PreprocessingSpec()
        ),
        methods=tuple(_method_from_mapping(item) for item in methods_raw),
        seeds=tuple(mapping["seeds"]),
        horizons=tuple(mapping["horizons"]),
        metrics=tuple(mapping["metrics"]),
        ood_shifts=tuple(_ood_from_mapping(item) for item in shifts_raw),
        uq=None if uq_raw is None else _uq_from_mapping(uq_raw),
        compute_budget=_budget_from_mapping(mapping["compute_budget"]),
        controls=tuple(mapping["controls"]),
    )


def manifest_to_mapping(manifest: ExperimentManifest) -> dict[str, Any]:
    """Return a JSON-friendly nested mapping.

    Tuples become lists. ``uq`` is ``None`` when unset.

    Parameters
    ----------
    manifest : ExperimentManifest
        Validated record.

    Returns
    -------
    dict
        Nested mapping suitable for JSON or YAML dump.
    """
    return asdict(manifest)
