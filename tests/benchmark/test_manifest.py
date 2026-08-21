"""Schema guards for frozen experiment manifests."""

from __future__ import annotations

import hashlib

import pytest

from koopman_graph.benchmark import (
    SCHEMA_VERSION,
    ComputeBudget,
    DatasetRef,
    EmptyMethodDeviationsError,
    ExperimentManifest,
    ManifestError,
    MethodSpec,
    OODShiftSpec,
    PreprocessingSpec,
    SplitSpec,
    UQSpec,
    manifest_from_mapping,
    manifest_to_mapping,
)

_DIGEST = hashlib.sha256(b"fixture-bytes").hexdigest()


def _dataset() -> DatasetRef:
    """Return a valid dataset reference.

    Returns
    -------
    DatasetRef
        Toy payload identity.
    """
    return DatasetRef(
        name="toy-path",
        version="1",
        sha256=_DIGEST,
        card="docs/data/toy.md",
    )


def _manifest(**overrides: object) -> ExperimentManifest:
    """Build a valid telemetry manifest.

    Parameters
    ----------
    **overrides : object
        Field replacements.

    Returns
    -------
    ExperimentManifest
        Validated record.
    """
    payload: dict[str, object] = {
        "manifest_id": "smoke-telemetry",
        "schema_version": SCHEMA_VERSION,
        "track": "telemetry",
        "dataset": _dataset(),
        "split": SplitSpec(0.7, 0.1, 0.2, history_len=12),
        "preprocessing": PreprocessingSpec(zscore=True),
        "methods": (MethodSpec(name="graph_koopman", role="koopman"),),
        "seeds": (0, 1, 2),
        "horizons": (1, 3, 12),
        "metrics": ("mae", "rmse"),
        "compute_budget": ComputeBudget(max_epochs=2),
        "controls": ("pernode",),
    }
    payload.update(overrides)
    return ExperimentManifest(**payload)  # type: ignore[arg-type]


def test_telemetry_manifest_accepts_pernode_control() -> None:
    """A minimal telemetry manifest with a per-node control is valid."""
    report = _manifest()
    assert report.schema_version == SCHEMA_VERSION
    assert report.dataset.sha256 == _DIGEST
    assert report.controls == ("pernode",)
    assert report.uq is None
    assert report.ood_shifts == ()


def test_topology_transfer_requires_three_controls() -> None:
    """Topology-transfer manifests must list hold-last, per-node, and joint LS."""
    with pytest.raises(ManifestError, match="topology_transfer controls must include"):
        _manifest(track="topology_transfer", controls=("pernode", "joint_ls"))
    report = _manifest(
        track="topology_transfer",
        controls=("hold_last", "pernode", "joint_ls"),
    )
    assert report.track == "topology_transfer"


def test_multiphysics_requires_factorization_control() -> None:
    """Multiphysics rejects a hold-last-only control list."""
    with pytest.raises(ManifestError, match="pernode and/or joint_ls"):
        _manifest(track="multiphysics", controls=("hold_last",))


def test_teaching_gnn_requires_deviations() -> None:
    """Teaching GNN methods cannot claim an empty deviation list."""
    with pytest.raises(EmptyMethodDeviationsError, match="non-empty deviations"):
        MethodSpec(name="stgcn", role="teaching_gnn")
    method = MethodSpec(
        name="stgcn",
        role="teaching_gnn",
        deviations=("teaching-scale channels",),
    )
    assert method.deviations == ("teaching-scale channels",)


def test_leaderboard_may_have_empty_deviations() -> None:
    """Leaderboard adapters may record an empty deviation tuple."""
    method = MethodSpec(name="metr-la-adapter", role="leaderboard")
    assert method.deviations == ()


def test_dedicated_library_requires_version() -> None:
    """Dedicated-library methods must pin a version."""
    with pytest.raises(ManifestError, match="version pin"):
        MethodSpec(name="libcity-dcrnn", role="dedicated_library")
    pinned = MethodSpec(name="libcity-dcrnn", role="dedicated_library", version="0.9.0")
    assert pinned.version == "0.9.0"


def test_malformed_sha256_and_seed_grid_rejected() -> None:
    """Digests must be 64 hex characters; seed grids must be unique and long enough."""
    with pytest.raises(ManifestError, match="64-character hex"):
        DatasetRef(name="toy", version="1", sha256="abcd", card="card.md")
    with pytest.raises(ManifestError, match="at least 3 unique"):
        _manifest(seeds=(0, 1))
    with pytest.raises(ManifestError, match="must not contain duplicates"):
        _manifest(seeds=(0, 1, 1))
    with pytest.raises(ManifestError, match="schema_version must be"):
        _manifest(schema_version="v0")


def test_unknown_keys_and_metrics_rejected() -> None:
    """Mappings reject extra keys; metrics are allowlisted."""
    mapping = manifest_to_mapping(_manifest())
    mapping["unexpected"] = True
    with pytest.raises(ManifestError, match="unknown manifest keys"):
        manifest_from_mapping(mapping)
    with pytest.raises(ManifestError, match="metrics must be a subset"):
        _manifest(metrics=("mae", "crps"))
    with pytest.raises(ManifestError, match="controls must be a subset"):
        _manifest(controls=("pernode", "sota"))


def test_mapping_round_trip_preserves_nested_records() -> None:
    """``asdict`` lists round-trip through ``manifest_from_mapping``."""
    original = _manifest(
        uq=UQSpec(method="conformal", coverage=0.9),
        ood_shifts=(OODShiftSpec(name="rewire", kind="rewire", description="p=0.1"),),
        methods=(
            MethodSpec(name="graph_koopman", role="koopman"),
            MethodSpec(
                name="stgcn",
                role="teaching_gnn",
                deviations=("teaching-scale channels",),
            ),
        ),
    )
    restored = manifest_from_mapping(manifest_to_mapping(original))
    assert restored == original
