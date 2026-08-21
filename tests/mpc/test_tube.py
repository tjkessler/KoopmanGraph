"""Tests for residual-tube Koopman-MPC (TASK-2347)."""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest
import torch
from tests.mpc.test_mpc import _calibrated_uq, _identity_plant_model, _origin
from torch_geometric.data import Data

import koopman_graph
from koopman_graph.mpc import (
    KoopmanMPC,
    TubeKoopmanMPC,
    TubeMPCReport,
    ensemble_residual_radii,
)
from koopman_graph.uq import ConformalKoopmanUQ, JointCoverageSpec
from koopman_graph.uq.common import PredictionInterval, snapshot_with_features

pytest.importorskip("osqp")


def _nominal_boxes() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.tensor([-10.0, -10.0]), torch.tensor([0.3, 10.0])


def _stuck_at_reference_plant(graph: Data, control: torch.Tensor) -> Data:
    """Plant whose mean output is the reference [1, 0] regardless of u."""
    del control
    stuck = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=graph.x.dtype)
    return snapshot_with_features(graph, stuck)


def test_tube_types_are_off_root_and_on_mpc_all() -> None:
    """Tube types stay off the root façade and on mpc.__all__."""
    exported = set(koopman_graph.__all__)
    assert "TubeKoopmanMPC" not in exported
    assert "TubeMPCReport" not in exported
    assert "ensemble_residual_radii" not in exported
    assert "TubeKoopmanMPC" in koopman_graph.mpc.__all__
    assert "TubeMPCReport" in koopman_graph.mpc.__all__
    assert "ensemble_residual_radii" in koopman_graph.mpc.__all__


def test_tube_module_cites_zhang_and_avoids_identification() -> None:
    """Module cites Zhang et al. and must not import identification."""
    import koopman_graph.mpc.tube as tube

    doc = tube.__doc__
    assert doc is not None
    assert "Zhang2022TubeMPC" in doc
    assert "10.1016/j.automatica.2021.110114" in doc
    assert "recursive" in doc.lower()
    tree = ast.parse(Path(tube.__file__).read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert all(not name.startswith("koopman_graph.identification") for name in imported)


def test_tube_rejects_bilinear_control() -> None:
    """Bilinear plants are refused (additive discrete only)."""
    model = _identity_plant_model(control_mode="bilinear")
    y_min, y_max = _nominal_boxes()
    with pytest.raises(ValueError, match="additive"):
        TubeKoopmanMPC(
            model,
            horizon=2,
            Q=torch.eye(2),
            R=torch.eye(1),
            residual_source=torch.tensor([0.1, 0.1]),
            y_min=y_min,
            y_max=y_max,
        )


def test_tube_rejects_uncalibrated_conformal() -> None:
    """Uncalibrated conformal residuals raise at construction."""
    model = _identity_plant_model()
    y_min, y_max = _nominal_boxes()
    with pytest.raises(RuntimeError, match="not calibrated"):
        TubeKoopmanMPC(
            model,
            horizon=2,
            Q=torch.eye(2),
            R=torch.eye(1),
            residual_source=ConformalKoopmanUQ(model),
            y_min=y_min,
            y_max=y_max,
        )


def test_tube_requires_nominal_output_boxes() -> None:
    """Tube construction requires both y_min and y_max."""
    model = _identity_plant_model()
    with pytest.raises(TypeError, match="y_min"):
        TubeKoopmanMPC(
            model,
            horizon=2,
            Q=torch.eye(2),
            R=torch.eye(1),
            residual_source=torch.tensor([0.1, 0.1]),
            y_max=torch.tensor([1.0, 1.0]),
        )


def test_tube_rejects_unshipped_coverage() -> None:
    """Simultaneous / event coverage targets raise (not a chance solver)."""
    model = _identity_plant_model()
    y_min, y_max = _nominal_boxes()
    with pytest.raises(ValueError, match="per_node_marginal"):
        TubeKoopmanMPC(
            model,
            horizon=2,
            Q=torch.eye(2),
            R=torch.eye(1),
            residual_source=torch.tensor([0.1, 0.1]),
            y_min=y_min,
            y_max=y_max,
            coverage=JointCoverageSpec(target="event"),
        )


def test_evaluate_reports_violation_rate_not_mse_alone() -> None:
    """Toy plant at the reference still reports a constraint-violation rate.

    The stuck plant sits at ``[1, 0]``, which matches the tracking
    reference (near-zero tracking error) while leaving ``y_max[0] = 0.3``.
    The report must expose ``violation_rate``, not an MSE-only summary.
    """
    model = _identity_plant_model()
    y_min, y_max = _nominal_boxes()
    controller = TubeKoopmanMPC(
        model,
        horizon=3,
        Q=torch.eye(2),
        R=0.05 * torch.eye(1),
        residual_source=torch.zeros(3),
        y_min=y_min,
        y_max=y_max,
    )
    reference = torch.tensor([1.0, 0.0])
    report = controller.evaluate(
        _origin(),
        reference,
        steps=4,
        plant=_stuck_at_reference_plant,
    )
    assert isinstance(report, TubeMPCReport)
    assert not hasattr(report, "mse")
    assert report.n_steps == 4
    assert report.violation_rate == pytest.approx(1.0)
    assert report.n_violations == 4
    assert 0.0 <= report.feasibility_rate <= 1.0
    assert report.coverage.target == "per_node_marginal"
    mean_x = torch.tensor([1.0, 0.0])
    tracking_mse = float(torch.mean((mean_x - reference) ** 2))
    assert tracking_mse == pytest.approx(0.0)
    assert report.violation_rate > tracking_mse


def test_evaluate_conformal_and_ensemble_radii() -> None:
    """Calibrated conformal quantiles and explicit ensemble radii both run."""
    model = _identity_plant_model()
    y_min = torch.tensor([-2.0, -2.0])
    y_max = torch.tensor([2.0, 2.0])
    conformal = TubeKoopmanMPC(
        model,
        horizon=3,
        Q=torch.eye(2),
        R=0.05 * torch.eye(1),
        residual_source=_calibrated_uq(model, steps=4, residual=0.05),
        y_min=y_min,
        y_max=y_max,
    )
    conformal_report = conformal.evaluate(
        _origin(),
        torch.tensor([0.0, 0.0]),
        steps=3,
    )
    assert conformal_report.violation_rate == pytest.approx(0.0)
    assert conformal_report.feasibility_rate == pytest.approx(1.0)
    assert math.isfinite(conformal_report.cost)

    radii = torch.tensor([0.05, 0.05, 0.05])
    ensemble = TubeKoopmanMPC(
        model,
        horizon=3,
        Q=torch.eye(2),
        R=0.05 * torch.eye(1),
        residual_source=radii,
        y_min=y_min,
        y_max=y_max,
    )
    ensemble_report = ensemble.evaluate(
        _origin(),
        torch.tensor([0.0, 0.0]),
        steps=3,
    )
    assert ensemble_report.violation_rate == pytest.approx(0.0)
    assert ensemble_report.feasibility_rate == pytest.approx(1.0)


def test_ensemble_residual_radii_from_interval() -> None:
    """PredictionInterval half-widths pool to a per-step scalar radius."""
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    template = Data(x=torch.zeros(2, 2), edge_index=edge_index)
    mean = snapshot_with_features(template, torch.zeros(2, 2))
    lower = snapshot_with_features(template, torch.full((2, 2), -0.4))
    upper = snapshot_with_features(template, torch.full((2, 2), 0.4))
    interval = PredictionInterval(
        mean=(mean, mean),
        lower=(lower, lower),
        upper=(upper, upper),
        level=0.9,
        n_members=3,
    )
    radii = ensemble_residual_radii(interval)
    assert radii.shape == (2,)
    assert torch.allclose(radii, torch.tensor([0.4, 0.4]))

    model = _identity_plant_model()
    controller = TubeKoopmanMPC(
        model,
        horizon=2,
        Q=torch.eye(2),
        R=torch.eye(1),
        residual_source=interval,
        y_min=torch.tensor([-2.0, -2.0]),
        y_max=torch.tensor([2.0, 2.0]),
    )
    action = controller.solve(_origin(), torch.tensor([0.0, 0.0]))
    assert action.shape == (1,)


def test_tube_solve_matches_koopman_mpc_without_residuals() -> None:
    """Zero residual radii recover the additive KoopmanMPC first action."""
    model = _identity_plant_model()
    y_min = torch.tensor([-2.0, -2.0])
    y_max = torch.tensor([2.0, 2.0])
    baseline = KoopmanMPC(
        model,
        horizon=4,
        Q=torch.eye(2),
        R=0.05 * torch.eye(1),
        u_min=torch.tensor([-2.0]),
        u_max=torch.tensor([2.0]),
        y_min=y_min,
        y_max=y_max,
    )
    tube = TubeKoopmanMPC(
        model,
        horizon=4,
        Q=torch.eye(2),
        R=0.05 * torch.eye(1),
        residual_source=torch.zeros(4),
        u_min=torch.tensor([-2.0]),
        u_max=torch.tensor([2.0]),
        y_min=y_min,
        y_max=y_max,
    )
    reference = torch.tensor([0.4, 0.0])
    origin = _origin()
    assert torch.allclose(
        tube.solve(origin, reference),
        baseline.solve(origin, reference),
    )
