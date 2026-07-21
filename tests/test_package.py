"""Smoke tests for package installation and imports."""

from __future__ import annotations

import importlib.metadata
import re
import subprocess
import sys
from pathlib import Path

import pytest

import koopman_graph

_VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+")

_RETAINED_ROOT = frozenset(
    {
        "ContinuousKoopmanOperator",
        "DelayEmbeddingEncoder",
        "DiffConvDecoder",
        "DiffConvEncoder",
        "GATDecoder",
        "GATEncoder",
        "GNNDecoder",
        "GNNEncoder",
        "GraphKoopmanModel",
        "GraphKoopmanOperator",
        "GraphSnapshotSequence",
        "GraphTransformerDecoder",
        "GraphTransformerEncoder",
        "KoopmanOperator",
        "KoopmanSpectrum",
        "MultiTrajectory",
        "SAGEDecoder",
        "SAGEEncoder",
        "__version__",
        "compute_spectrum",
    }
)

_DEMOTED_ROOT = frozenset(
    {
        "BackwardConsistencyLoss",
        "DMDBaseline",
        "DMDcBaseline",
        "EDMDBaseline",
        "EigenvalueRegularizationLoss",
        "EvaluationResult",
        "FitHistory",
        "ForwardConsistencyLoss",
        "GraphKoopmanEnv",
        "LossWeights",
        "RecursiveKoopmanAdapter",
        "TemporalSplit",
        "WindowSampler",
        "evaluate_forecast",
        "temporal_split",
    }
)


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
        "LieConsistencyLoss",
        "Parameterization",
        "PDEResidualLoss",
        "propagate_latent",
        "resolve_graph_inputs",
        "snapshot_to_device",
    }
    exported = set(koopman_graph.__all__)
    assert not (power_user & exported)
    assert "GATDecoder" in exported
    assert "SAGEEncoder" in exported
    assert "DiffConvDecoder" in exported
    assert "GraphTransformerEncoder" in exported
    assert "GraphTransformerDecoder" in exported
    assert "GraphKoopmanModel" in exported

    import koopman_graph.graph_utils as graph_utils
    import koopman_graph.nn.gnn as gnn
    import koopman_graph.protocols as protocols
    from koopman_graph.losses import LieConsistencyLoss, PDEResidualLoss
    from koopman_graph.operators import KoopmanOperatorContract, Parameterization

    assert protocols.ForecastModel is not None
    assert LieConsistencyLoss is not None
    assert PDEResidualLoss is not None
    assert KoopmanOperatorContract is not None
    assert Parameterization is not None
    assert callable(graph_utils.propagate_latent)
    assert callable(graph_utils.pack_rollout_snapshots)
    assert callable(graph_utils.dense_symmetric_normalized_laplacian)
    assert graph_utils.topology is not None
    assert graph_utils.propagation is not None
    assert gnn.BaseGNNModule is not None
    assert callable(gnn.build_gcn_convs)


def test_losses_capability_package_reexports() -> None:
    """Same-named ``losses`` package keeps prior import paths and peer modules."""
    import koopman_graph.losses as losses
    from koopman_graph.losses import (
        BackwardConsistencyLoss,
        EigenvalueRegularizationLoss,
        ForwardConsistencyLoss,
        KoopmanSparsityLoss,
        WorstCaseReconstructionLoss,
        masked_mse_loss,
        rollout_multi_start_loss,
        rollout_sequence_loss,
    )

    assert losses.consistency is not None
    assert losses.regularization is not None
    assert losses.reconstruction is not None
    assert losses.physics is not None
    assert losses.rollout is not None
    assert ForwardConsistencyLoss is losses.ForwardConsistencyLoss
    assert BackwardConsistencyLoss is losses.BackwardConsistencyLoss
    assert EigenvalueRegularizationLoss is losses.EigenvalueRegularizationLoss
    assert KoopmanSparsityLoss is losses.KoopmanSparsityLoss
    assert WorstCaseReconstructionLoss is losses.WorstCaseReconstructionLoss
    assert masked_mse_loss is losses.masked_mse_loss
    assert rollout_sequence_loss is losses.rollout_sequence_loss
    assert rollout_multi_start_loss is losses.rollout_multi_start_loss
    assert "ForwardConsistencyLoss" not in koopman_graph.__all__
    assert "LieConsistencyLoss" not in koopman_graph.__all__
    assert "KoopmanSparsityLoss" not in koopman_graph.__all__
    assert not hasattr(koopman_graph, "ForwardConsistencyLoss")


def test_model_capability_package_reexports() -> None:
    """Same-named ``model`` package keeps GraphKoopmanModel and peer modules."""
    import koopman_graph.model as model_pkg
    from koopman_graph.model import GraphKoopmanModel
    from koopman_graph.model.factory import build_koopman, parse_koopman_arg
    from koopman_graph.model.timing import resolve_time_increments
    from koopman_graph.model.validation import as_data, validate_controls

    assert model_pkg.estimator is not None
    assert model_pkg.factory is not None
    assert model_pkg.validation is not None
    assert model_pkg.timing is not None
    assert GraphKoopmanModel is model_pkg.GraphKoopmanModel
    assert koopman_graph.GraphKoopmanModel is GraphKoopmanModel
    assert callable(build_koopman)
    assert callable(parse_koopman_arg)
    assert callable(resolve_time_increments)
    assert callable(as_data)
    assert callable(validate_controls)


def test_specialized_root_exports_not_in_root_all() -> None:
    """Specialized symbols import from capability modules, not root ``__all__``."""
    exported = set(koopman_graph.__all__)
    assert exported == _RETAINED_ROOT
    assert not (_DEMOTED_ROOT & exported)

    from koopman_graph.adaptation import RecursiveKoopmanAdapter
    from koopman_graph.baselines import DMDBaseline, DMDcBaseline, EDMDBaseline
    from koopman_graph.data import TemporalSplit, WindowSampler, temporal_split
    from koopman_graph.env import GraphKoopmanEnv
    from koopman_graph.losses import (
        BackwardConsistencyLoss,
        EigenvalueRegularizationLoss,
        ForwardConsistencyLoss,
    )
    from koopman_graph.metrics import EvaluationResult, evaluate_forecast
    from koopman_graph.training import FitHistory, LossWeights

    assert RecursiveKoopmanAdapter is not None
    assert DMDBaseline is not None
    assert DMDcBaseline is not None
    assert EDMDBaseline is not None
    assert TemporalSplit is not None
    assert WindowSampler is not None
    assert callable(temporal_split)
    assert GraphKoopmanEnv is not None
    assert ForwardConsistencyLoss is not None
    assert BackwardConsistencyLoss is not None
    assert EigenvalueRegularizationLoss is not None
    assert EvaluationResult is not None
    assert callable(evaluate_forecast)
    assert FitHistory is not None
    assert LossWeights is not None

    for name in _DEMOTED_ROOT:
        assert not hasattr(koopman_graph, name), name

    with pytest.raises(ImportError):
        exec("from koopman_graph import LossWeights")
    with pytest.raises(ImportError):
        exec("from koopman_graph import DMDBaseline, evaluate_forecast")


def test_metrics_secondaries_demoted_from_root() -> None:
    """Forecast eval and low-level metrics live in ``koopman_graph.metrics``."""
    demoted = {
        "EvaluationResult",
        "HorizonMetrics",
        "evaluate_forecast",
        "mae",
        "mape",
        "rmse",
    }
    exported = set(koopman_graph.__all__)
    assert not (demoted & exported)

    from koopman_graph.metrics import (
        EvaluationResult,
        HorizonMetrics,
        evaluate_forecast,
        mae,
        mape,
        rmse,
    )

    assert EvaluationResult is not None
    assert HorizonMetrics is not None
    assert callable(evaluate_forecast)
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
        "FilterResult",
        "GraphKoopmanEnv",
        "KoopmanObserver",
        "RecursiveKoopmanAdapter",
        "TemporalSplit",
        "WindowSampler",
        "as_multi_trajectory",
        "graph_laplacian_features",
        "temporal_split",
    }
    exported = set(koopman_graph.__all__)
    assert not (demoted & exported)
    assert "GraphSnapshotSequence" in exported
    assert "MultiTrajectory" in exported

    from koopman_graph.adaptation import (
        AdaptationStepResult,
        FilterResult,
        KoopmanObserver,
        RecursiveKoopmanAdapter,
    )
    from koopman_graph.data import (
        TemporalSplit,
        WindowSampler,
        as_multi_trajectory,
        temporal_split,
    )
    from koopman_graph.env import GraphKoopmanEnv
    from koopman_graph.observables import graph_laplacian_features

    assert AdaptationStepResult is not None
    assert FilterResult is not None
    assert KoopmanObserver is not None
    assert RecursiveKoopmanAdapter is not None
    assert TemporalSplit is not None
    assert WindowSampler is not None
    assert callable(as_multi_trajectory)
    assert callable(temporal_split)
    assert GraphKoopmanEnv is not None
    assert callable(graph_laplacian_features)


def _documented_keep_in_all_names() -> set[str]:
    """Parse the documented root-export inventory from ``architecture.rst``."""
    architecture = (
        Path(__file__).resolve().parents[1] / "docs" / "source" / "architecture.rst"
    )
    text = architecture.read_text(encoding="utf-8")
    start = text.index("**Keep in** ``koopman_graph.__all__``")
    end = text.index("**Demote to module imports**", start)
    section = text[start:end]
    names = {
        match.group(1)
        for match in re.finditer(r":(?:class|func|data):`~[\w.]+\.(\w+)`", section)
    }
    names.update(re.findall(r"``(__version__)``", section))
    return names


def test_root_all_matches_architecture_keep_inventory() -> None:
    """Root ``__all__`` matches the documented architecture root-export inventory."""
    keep = _documented_keep_in_all_names()
    assert "DelayEmbeddingEncoder" in keep
    assert "GraphKoopmanOperator" in keep
    assert "DMDBaseline" not in keep
    assert "LossWeights" not in keep
    assert "evaluate_forecast" not in keep
    assert set(koopman_graph.__all__) == keep
    assert keep == _RETAINED_ROOT


def test_removed_deep_import_shims_are_unavailable() -> None:
    """Former root shim modules must not import."""
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
