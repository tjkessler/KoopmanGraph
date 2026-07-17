"""Smoke tests for package installation and imports."""

from __future__ import annotations

import importlib.metadata
import re
import subprocess
import sys
from pathlib import Path

import koopman_graph

_VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+")


def test_import_package() -> None:
    """Verify the package imports and exposes a semver-like ``__version__``."""
    assert koopman_graph.__version__
    assert _VERSION_PATTERN.match(koopman_graph.__version__)


def test_public_all_excludes_power_user_modules() -> None:
    """Power-user modules stay importable but out of the stable ``__all__``."""
    power_user = {
        "BaseGNNModule",
        "ForecastModel",
        "KoopmanOperatorContract",
        "Parameterization",
        "propagate_latent",
        "resolve_graph_inputs",
        "snapshot_to_device",
    }
    exported = set(koopman_graph.__all__)
    assert not (power_user & exported)
    assert "GATDecoder" in exported
    assert "GraphKoopmanModel" in exported

    import koopman_graph.graph_utils as graph_utils
    import koopman_graph.nn.gnn as gnn
    import koopman_graph.protocols as protocols
    from koopman_graph.operators import KoopmanOperatorContract, Parameterization

    assert protocols.ForecastModel is not None
    assert KoopmanOperatorContract is not None
    assert Parameterization is not None
    assert callable(graph_utils.propagate_latent)
    assert gnn.BaseGNNModule is not None
    assert callable(gnn.build_gcn_convs)


def test_metrics_secondaries_demoted_from_root() -> None:
    """Low-level metrics live in ``koopman_graph.metrics``, not root ``__all__``."""
    demoted = {"HorizonMetrics", "mae", "mape", "rmse"}
    exported = set(koopman_graph.__all__)
    assert not (demoted & exported)
    assert "evaluate_forecast" in exported
    assert "EvaluationResult" in exported

    from koopman_graph.metrics import HorizonMetrics, mae, mape, rmse

    assert HorizonMetrics is not None
    assert callable(mae)
    assert callable(mape)
    assert callable(rmse)


def test_analysis_secondaries_demoted_from_root() -> None:
    """Specialized analysis helpers live in ``koopman_graph.analysis``, not root."""
    demoted = {
        "AnomalyDetectionResult",
        "calibrate_anomaly_threshold",
        "compute_generator_spectrum",
        "decode_mode_shapes",
        "detect_anomaly",
        "discrete_spectrum_at_delta_t",
        "dynamical_similarity",
        "koopman_std",
        "plot_spectrum",
        "spectrum_distance",
    }
    exported = set(koopman_graph.__all__)
    assert not (demoted & exported)
    assert "KoopmanSpectrum" in exported
    assert "compute_spectrum" in exported

    from koopman_graph.analysis import (
        AnomalyDetectionResult,
        calibrate_anomaly_threshold,
        compute_generator_spectrum,
        decode_mode_shapes,
        detect_anomaly,
        discrete_spectrum_at_delta_t,
        dynamical_similarity,
        koopman_std,
        plot_spectrum,
        spectrum_distance,
    )

    assert AnomalyDetectionResult is not None
    assert callable(calibrate_anomaly_threshold)
    assert callable(compute_generator_spectrum)
    assert callable(decode_mode_shapes)
    assert callable(detect_anomaly)
    assert callable(discrete_spectrum_at_delta_t)
    assert callable(dynamical_similarity)
    assert callable(koopman_std)
    assert callable(plot_spectrum)
    assert callable(spectrum_distance)


def test_data_adaptation_observables_secondaries_demoted_from_root() -> None:
    """Data/adaptation/observables helpers live in capability modules, not root."""
    demoted = {
        "AdaptationStepResult",
        "as_multi_trajectory",
        "graph_laplacian_features",
    }
    exported = set(koopman_graph.__all__)
    assert not (demoted & exported)
    assert "RecursiveKoopmanAdapter" in exported
    assert "GraphSnapshotSequence" in exported
    assert "MultiTrajectory" in exported
    assert "GraphKoopmanEnv" in exported

    from koopman_graph.adaptation import AdaptationStepResult
    from koopman_graph.data import as_multi_trajectory
    from koopman_graph.observables import graph_laplacian_features

    assert AdaptationStepResult is not None
    assert callable(as_multi_trajectory)
    assert callable(graph_laplacian_features)


def test_root_all_matches_thin_facade_keep_inventory() -> None:
    """Root ``__all__`` matches the TASK-747 keep list exactly."""
    keep = {
        "BackwardConsistencyLoss",
        "ContinuousKoopmanOperator",
        "DMDBaseline",
        "DMDcBaseline",
        "EDMDBaseline",
        "EigenvalueRegularizationLoss",
        "EvaluationResult",
        "FitHistory",
        "ForwardConsistencyLoss",
        "GATDecoder",
        "GATEncoder",
        "GNNDecoder",
        "GNNEncoder",
        "GraphKoopmanEnv",
        "GraphKoopmanModel",
        "GraphSnapshotSequence",
        "KoopmanOperator",
        "KoopmanSpectrum",
        "LossWeights",
        "MultiTrajectory",
        "RecursiveKoopmanAdapter",
        "TemporalSplit",
        "WindowSampler",
        "__version__",
        "compute_spectrum",
        "evaluate_forecast",
        "temporal_split",
    }
    assert set(koopman_graph.__all__) == keep


def test_removed_deep_import_shims_are_unavailable() -> None:
    """v0.3.0 hard-cut: former root shim modules must not import."""
    import importlib

    for name in (
        "koopman_graph.encoder",
        "koopman_graph.decoder",
        "koopman_graph.gnn",
        "koopman_graph.operator",
        "koopman_graph.continuous",
    ):
        try:
            importlib.import_module(name)
        except ModuleNotFoundError:
            continue
        raise AssertionError(f"expected ModuleNotFoundError for {name}")


def test_installed_version_matches_metadata() -> None:
    """Editable install should expose the same version via importlib.metadata."""
    assert importlib.metadata.version("koopman-graph") == koopman_graph.__version__


def test_build_produces_wheel_and_sdist() -> None:
    """Verify ``python -m build`` produces installable artifacts."""
    root = Path(__file__).resolve().parents[1]
    dist_dir = root / "dist"
    dist_dir.mkdir(exist_ok=True)

    result = subprocess.run(
        [sys.executable, "-m", "build", "--outdir", str(dist_dir)],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout

    assert list(dist_dir.glob("koopman_graph-*.whl"))
    assert list(dist_dir.glob("koopman_graph-*.tar.gz"))
