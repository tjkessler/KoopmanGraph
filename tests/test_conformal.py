"""Tests for conformal prediction UQ (TASK-1313 / TASK-1407)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.model import GraphKoopmanModel
from koopman_graph.nn import GNNDecoder, GNNEncoder
from koopman_graph.uq import ConformalKoopmanUQ, empirical_coverage
from koopman_graph.uq.common import snapshot_with_features


def _path_edge_index(num_nodes: int = 3) -> torch.Tensor:
    edges: list[list[int]] = []
    for node in range(num_nodes - 1):
        edges.extend([[node, node + 1], [node + 1, node]])
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _tiny_model(num_features: int = 2) -> GraphKoopmanModel:
    encoder = GNNEncoder(
        in_channels=num_features,
        hidden_channels=4,
        latent_dim=num_features,
        num_layers=1,
    )
    decoder = GNNDecoder(
        latent_dim=num_features,
        hidden_channels=4,
        out_channels=num_features,
        num_layers=1,
    )
    return GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=num_features,
        time_step=0.1,
    )


def _patch_zero_predict(model: GraphKoopmanModel) -> None:
    """Force predict to return zero features with held topology."""

    def _predict(
        initial_graph,
        steps,
        edge_index=None,
        edge_weight=None,
        controls=None,
        future_topologies=None,
        history=None,
    ):
        if isinstance(initial_graph, Data):
            template = initial_graph
            assert template.x is not None
            zeros = torch.zeros_like(template.x)
        else:
            template = Data(x=initial_graph, edge_index=edge_index)
            zeros = torch.zeros_like(initial_graph)
        return [snapshot_with_features(template, zeros) for _ in range(steps)]

    model.predict = _predict  # type: ignore[method-assign]


def _sequence_with_constant_residual(
    *,
    residual: float,
    steps: int,
    edge_index: torch.Tensor,
    num_features: int = 2,
) -> GraphSnapshotSequence:
    """Origin zeros; each target is a constant residual (predict is zero)."""
    num_nodes = int(edge_index.max().item()) + 1
    origin = Data(
        x=torch.zeros(num_nodes, num_features),
        edge_index=edge_index,
    )
    snaps = [origin]
    for _ in range(steps):
        snaps.append(
            Data(
                x=torch.full((num_nodes, num_features), residual),
                edge_index=edge_index,
            )
        )
    return GraphSnapshotSequence(snaps)


def _iid_residual_sequences(
    *,
    n: int,
    steps: int,
    scale: float,
    seed: int,
    edge_index: torch.Tensor,
) -> list[GraphSnapshotSequence]:
    generator = torch.Generator().manual_seed(seed)
    sequences: list[GraphSnapshotSequence] = []
    num_nodes = int(edge_index.max().item()) + 1
    for _ in range(n):
        origin = Data(
            x=torch.zeros(num_nodes, 2),
            edge_index=edge_index,
        )
        snaps = [origin]
        for _horizon in range(steps):
            # Non-negative residuals so L_inf score equals the max entry.
            noise = scale * torch.rand(num_nodes, 2, generator=generator)
            snaps.append(Data(x=noise, edge_index=edge_index))
        sequences.append(GraphSnapshotSequence(snaps))
    return sequences


def test_uncalibrated_predict_raises() -> None:
    """predict_interval before calibrate raises RuntimeError."""
    model = _tiny_model()
    _patch_zero_predict(model)
    uq = ConformalKoopmanUQ(model)
    edge_index = _path_edge_index()
    origin = Data(x=torch.zeros(3, 2), edge_index=edge_index)
    with pytest.raises(RuntimeError, match="not calibrated"):
        uq.predict_interval(origin, steps=1)


def test_short_sequence_rejected() -> None:
    """Calibration sequences shorter than steps+1 raise."""
    model = _tiny_model()
    _patch_zero_predict(model)
    uq = ConformalKoopmanUQ(model)
    edge_index = _path_edge_index()
    short = GraphSnapshotSequence([Data(x=torch.zeros(3, 2), edge_index=edge_index)])
    with pytest.raises(ValueError, match="need >="):
        uq.calibrate([short], steps=2, alpha=0.1)


def test_interval_ordering_and_nonneg_widths() -> None:
    """lower ≤ mean ≤ upper and widths are non-negative."""
    model = _tiny_model()
    _patch_zero_predict(model)
    edge_index = _path_edge_index()
    sequences = [
        _sequence_with_constant_residual(
            residual=0.2 + 0.05 * i, steps=2, edge_index=edge_index
        )
        for i in range(20)
    ]
    uq = ConformalKoopmanUQ(model, method="split")
    uq.calibrate(sequences, steps=2, alpha=0.1)
    origin = Data(x=torch.zeros(3, 2), edge_index=edge_index)
    interval = uq.predict_interval(origin, steps=2, level=0.9)
    for mean, lower, upper in zip(
        interval.mean, interval.lower, interval.upper, strict=True
    ):
        assert mean.x is not None and lower.x is not None and upper.x is not None
        assert torch.all(lower.x <= mean.x)
        assert torch.all(mean.x <= upper.x)
        assert torch.all(upper.x - lower.x >= 0)


def test_widths_increase_with_noise() -> None:
    """Larger calibration residuals yield larger conformal half-widths."""
    model = _tiny_model()
    _patch_zero_predict(model)
    edge_index = _path_edge_index()
    low = [
        _sequence_with_constant_residual(residual=0.1, steps=1, edge_index=edge_index)
        for _ in range(30)
    ]
    high = [
        _sequence_with_constant_residual(residual=1.0, steps=1, edge_index=edge_index)
        for _ in range(30)
    ]
    uq_low = ConformalKoopmanUQ(model, method="split")
    uq_high = ConformalKoopmanUQ(model, method="split")
    uq_low.calibrate(low, steps=1, alpha=0.1)
    uq_high.calibrate(high, steps=1, alpha=0.1)
    assert float(uq_high.quantiles[0]) > float(uq_low.quantiles[0])


def test_split_coverage_near_nominal() -> None:
    """Empirical coverage is near 90% on i.i.d. residual data."""
    model = _tiny_model()
    _patch_zero_predict(model)
    edge_index = _path_edge_index()
    steps = 1
    alpha = 0.1
    cal = _iid_residual_sequences(
        n=80, steps=steps, scale=1.0, seed=0, edge_index=edge_index
    )
    test = _iid_residual_sequences(
        n=120, steps=steps, scale=1.0, seed=1, edge_index=edge_index
    )
    uq = ConformalKoopmanUQ(model, method="split", score="aggregate")
    uq.calibrate(cal, steps=steps, alpha=alpha)

    coverages: list[float] = []
    for sequence in test:
        interval = uq.predict_interval(sequence[0], steps=steps, level=0.9)
        targets = [sequence[h + 1] for h in range(steps)]
        coverages.append(empirical_coverage(targets, interval))
    mean_coverage = sum(coverages) / len(coverages)
    # Finite-sample / entrywise tolerance around nominal 0.9.
    assert 0.75 <= mean_coverage <= 1.0


def test_adaptive_beats_split_under_drift() -> None:
    """ACI tracks a late high-noise regime better than batch split.

    Calibration is mostly low-noise so the split 90% quantile stays small,
    then a short high-noise burst arrives. ACI with a large step size
    inflates ``q`` on that burst; holdout matches the high-noise regime.
    """
    model = _tiny_model()
    _patch_zero_predict(model)
    edge_index = _path_edge_index()
    steps = 1
    early = [
        _sequence_with_constant_residual(
            residual=0.05, steps=steps, edge_index=edge_index
        )
        for _ in range(95)
    ]
    late = [
        _sequence_with_constant_residual(
            residual=2.0, steps=steps, edge_index=edge_index
        )
        for _ in range(5)
    ]
    drifted_cal = early + late
    holdout = [
        _sequence_with_constant_residual(
            residual=2.0, steps=steps, edge_index=edge_index
        )
        for _ in range(20)
    ]

    split = ConformalKoopmanUQ(model, method="split")
    adaptive = ConformalKoopmanUQ(model, method="adaptive", gamma=0.5)
    split.calibrate(drifted_cal, steps=steps, alpha=0.1)
    adaptive.calibrate(drifted_cal, steps=steps, alpha=0.1)

    def _mean_coverage(uq: ConformalKoopmanUQ) -> float:
        vals = []
        for sequence in holdout:
            interval = uq.predict_interval(sequence[0], steps=steps, level=0.9)
            vals.append(empirical_coverage([sequence[1]], interval))
        return sum(vals) / len(vals)

    assert float(adaptive.quantiles[0]) > float(split.quantiles[0])
    assert _mean_coverage(adaptive) > _mean_coverage(split)


def test_calibration_round_trip(tmp_path: Path) -> None:
    """save_calibration / load_calibration restores quantiles."""
    model = _tiny_model()
    _patch_zero_predict(model)
    edge_index = _path_edge_index()
    sequences = [
        _sequence_with_constant_residual(residual=0.3, steps=2, edge_index=edge_index)
        for _ in range(15)
    ]
    uq = ConformalKoopmanUQ(model, method="split")
    uq.calibrate(sequences, steps=2, alpha=0.1)
    path = tmp_path / "cal.pt"
    uq.save_calibration(path)

    other = ConformalKoopmanUQ(model, method="split")
    other.load_calibration(path)
    assert torch.allclose(other.quantiles, uq.quantiles)


def test_per_node_score_calibrates() -> None:
    """per_node score mode produces a calibrated wrapper."""
    model = _tiny_model()
    _patch_zero_predict(model)
    edge_index = _path_edge_index()
    sequences = _iid_residual_sequences(
        n=25, steps=1, scale=0.5, seed=3, edge_index=edge_index
    )
    uq = ConformalKoopmanUQ(model, method="split", score="per_node")
    uq.calibrate(sequences, steps=1, alpha=0.1)
    assert uq.is_calibrated
    assert uq.quantiles.shape == (1,)


def _heteroscedastic_sequences(
    *,
    n: int,
    node_scales: torch.Tensor,
    edge_index: torch.Tensor,
    seed: int,
) -> list[GraphSnapshotSequence]:
    """Zero-origin sequences with per-node residual scales (predict=0)."""
    generator = torch.Generator().manual_seed(seed)
    num_nodes = int(node_scales.numel())
    sequences: list[GraphSnapshotSequence] = []
    for _ in range(n):
        origin = Data(x=torch.zeros(num_nodes, 2), edge_index=edge_index)
        noise = torch.rand(num_nodes, 2, generator=generator)
        noise = noise * node_scales.unsqueeze(-1)
        sequences.append(
            GraphSnapshotSequence([origin, Data(x=noise, edge_index=edge_index)])
        )
    return sequences


def test_legacy_scalar_quantiles_unchanged() -> None:
    """aggregate / per_node quantiles match pre-TASK-1407 golden values."""
    model = _tiny_model()
    _patch_zero_predict(model)
    edge_index = _path_edge_index()
    sequences = [
        _sequence_with_constant_residual(
            residual=0.2 + 0.05 * i, steps=2, edge_index=edge_index
        )
        for i in range(20)
    ]
    expected = {
        "aggregate": [1.100000023841858, 1.100000023841858],
        "per_node": [1.5556349754333496, 1.5556349754333496],
    }
    for score, values in expected.items():
        uq = ConformalKoopmanUQ(model, method="split", score=score)  # type: ignore[arg-type]
        uq.calibrate(sequences, steps=2, alpha=0.1)
        assert torch.allclose(uq.quantiles, torch.tensor(values))


def test_node_wise_quantiles_and_heteroscedastic_widths() -> None:
    """``node_wise`` yields ``(steps, N)`` quantiles ordered by per-node
    :math:`L_2` residual noise scales.
    """
    model = _tiny_model()
    _patch_zero_predict(model)
    edge_index = _path_edge_index(num_nodes=4)
    scales = torch.tensor([0.1, 0.4, 1.0, 2.0])
    cal = _heteroscedastic_sequences(
        n=40, node_scales=scales, edge_index=edge_index, seed=0
    )
    uq = ConformalKoopmanUQ(model, method="split", score="node_wise")
    uq.calibrate(cal, steps=1, alpha=0.1)
    assert uq.quantiles.shape == (1, 4)
    widths = uq.quantiles[0]
    assert torch.all(widths >= 0)
    # Pairwise: noisier nodes get wider intervals.
    assert float(widths[0]) < float(widths[1]) < float(widths[2]) < float(widths[3])

    origin = Data(x=torch.zeros(4, 2), edge_index=edge_index)
    interval = uq.predict_interval(origin, steps=1, level=0.9)
    mean_x = interval.mean[0].x
    lower_x = interval.lower[0].x
    upper_x = interval.upper[0].x
    assert mean_x is not None and lower_x is not None and upper_x is not None
    assert torch.all(lower_x <= mean_x)
    assert torch.all(mean_x <= upper_x)
    # Half-widths differ across nodes (broadcast over features).
    half = mean_x[:, 0] - lower_x[:, 0]
    assert float(half[0]) < float(half[3])


def test_node_wise_coverage_near_nominal() -> None:
    """Per-node marginal coverage stays near the nominal level."""
    model = _tiny_model()
    _patch_zero_predict(model)
    edge_index = _path_edge_index(num_nodes=3)
    scales = torch.tensor([0.5, 1.0, 1.5])
    cal = _heteroscedastic_sequences(
        n=80, node_scales=scales, edge_index=edge_index, seed=2
    )
    test = _heteroscedastic_sequences(
        n=120, node_scales=scales, edge_index=edge_index, seed=3
    )
    uq = ConformalKoopmanUQ(model, method="split", score="node_wise")
    uq.calibrate(cal, steps=1, alpha=0.1)
    coverages = [
        empirical_coverage(
            [sequence[1]],
            uq.predict_interval(sequence[0], steps=1, level=0.9),
        )
        for sequence in test
    ]
    mean_coverage = sum(coverages) / len(coverages)
    assert 0.75 <= mean_coverage <= 1.0


def test_neighbor_smoothing_bounds_and_zero_identity() -> None:
    """Smoothing in [0, 1]; λ=0 matches unsmoothed node_wise exactly."""
    model = _tiny_model()
    _patch_zero_predict(model)
    with pytest.raises(ValueError, match="neighbor_smoothing"):
        ConformalKoopmanUQ(model, score="node_wise", neighbor_smoothing=1.5)
    with pytest.raises(ValueError, match="neighbor_smoothing"):
        ConformalKoopmanUQ(model, score="aggregate", neighbor_smoothing=0.2)

    edge_index = _path_edge_index()
    cal = _iid_residual_sequences(
        n=20, steps=1, scale=1.0, seed=4, edge_index=edge_index
    )
    base = ConformalKoopmanUQ(model, method="split", score="node_wise")
    zero = ConformalKoopmanUQ(
        model, method="split", score="node_wise", neighbor_smoothing=0.0
    )
    base.calibrate(cal, steps=1, alpha=0.1)
    zero.calibrate(cal, steps=1, alpha=0.1)
    assert torch.equal(base.quantiles, zero.quantiles)


def test_neighbor_smoothing_reduces_neighbor_width_variance() -> None:
    """Diffusion reduces across-neighbor width variance on a path graph."""
    model = _tiny_model()
    _patch_zero_predict(model)
    edge_index = _path_edge_index(num_nodes=5)
    # Homophilous pattern: adjacent nodes share similar scales; ends differ.
    scales = torch.tensor([0.2, 0.25, 1.5, 0.3, 0.35])
    cal = _heteroscedastic_sequences(
        n=50, node_scales=scales, edge_index=edge_index, seed=5
    )
    plain = ConformalKoopmanUQ(model, method="split", score="node_wise")
    smooth = ConformalKoopmanUQ(
        model, method="split", score="node_wise", neighbor_smoothing=0.7
    )
    plain.calibrate(cal, steps=1, alpha=0.1)
    smooth.calibrate(cal, steps=1, alpha=0.1)
    # Focus on the high-noise node and its two neighbors.
    plain_local = plain.quantiles[0, 1:4]
    smooth_local = smooth.quantiles[0, 1:4]
    assert float(torch.var(smooth_local)) < float(torch.var(plain_local))


def test_minimum_calibration_count_enforced() -> None:
    """Too few calibration sequences raise an actionable error."""
    model = _tiny_model()
    _patch_zero_predict(model)
    edge_index = _path_edge_index()
    cal = _iid_residual_sequences(
        n=5, steps=1, scale=0.5, seed=6, edge_index=edge_index
    )
    uq = ConformalKoopmanUQ(model, method="split", score="node_wise")
    with pytest.raises(ValueError, match="node_wise calibration requires"):
        uq.calibrate(cal, steps=1, alpha=0.1)


def test_node_wise_calibration_round_trip(tmp_path: Path) -> None:
    """v2 payload carries score, smoothing, and per-node quantiles."""
    model = _tiny_model()
    _patch_zero_predict(model)
    edge_index = _path_edge_index()
    cal = _iid_residual_sequences(
        n=15, steps=1, scale=0.8, seed=7, edge_index=edge_index
    )
    uq = ConformalKoopmanUQ(
        model, method="split", score="node_wise", neighbor_smoothing=0.3
    )
    uq.calibrate(cal, steps=1, alpha=0.1)
    path = tmp_path / "cal_v2.pt"
    uq.save_calibration(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["kind"] == "ConformalKoopmanUQ.calibration.v2"
    assert payload["score"] == "node_wise"
    assert payload["neighbor_smoothing"] == 0.3

    other = ConformalKoopmanUQ(
        model, method="split", score="node_wise", neighbor_smoothing=0.3
    )
    other.load_calibration(path)
    assert torch.allclose(other.quantiles, uq.quantiles)

    mismatch = ConformalKoopmanUQ(
        model, method="split", score="node_wise", neighbor_smoothing=None
    )
    with pytest.raises(ValueError, match="neighbor_smoothing"):
        mismatch.load_calibration(path)


def test_conformal_error_paths_and_adaptive_node_wise(tmp_path: Path) -> None:
    """Cover validation, controls, adaptive node_wise, and load mismatches."""
    from koopman_graph.uq.conformal import _diffuse_node_scores

    model = _tiny_model()
    _patch_zero_predict(model)
    edge_index = _path_edge_index()
    scores = torch.tensor([1.0, 2.0, 3.0])
    assert torch.equal(
        _diffuse_node_scores(scores, edge_index, edge_weight=None, lam=0.0),
        scores,
    )

    bare = ConformalKoopmanUQ(model, method="split")
    with pytest.raises(RuntimeError, match="not calibrated"):
        _ = bare.quantiles
    with pytest.raises(RuntimeError, match="not calibrated"):
        bare.save_calibration(tmp_path / "missing.pt")

    cal = _iid_residual_sequences(
        n=20, steps=1, scale=0.5, seed=8, edge_index=edge_index
    )
    model_ctrl = GraphKoopmanModel(
        encoder=GNNEncoder(
            in_channels=2, hidden_channels=4, latent_dim=2, num_layers=1
        ),
        decoder=GNNDecoder(
            latent_dim=2, hidden_channels=4, out_channels=2, num_layers=1
        ),
        latent_dim=2,
        time_step=0.1,
        control_dim=1,
    )
    _patch_zero_predict(model_ctrl)
    controls = [[torch.zeros(1)] for _ in range(len(cal))]
    split = ConformalKoopmanUQ(model_ctrl, method="split")
    split.calibrate(cal, steps=1, alpha=0.1, controls=controls)

    # Missing features during calibration (dedicated model so patches stay local).
    model_bad = _tiny_model()
    bad_pred = ConformalKoopmanUQ(model_bad, method="split")

    def _predict_none(initial_graph, steps, **kwargs):
        return [Data(x=None, edge_index=edge_index) for _ in range(steps)]

    bad_pred.model.predict = _predict_none  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="must define node features"):
        bad_pred.calibrate(cal, steps=1, alpha=0.1)

    cal_nw = _iid_residual_sequences(
        n=15, steps=1, scale=0.4, seed=9, edge_index=edge_index
    )
    nw = ConformalKoopmanUQ(model, method="split", score="node_wise")
    smooth = ConformalKoopmanUQ(
        model, method="split", score="node_wise", neighbor_smoothing=0.5
    )
    # Empty edge_index is present but unused; clear it after construction so
    # neighbor_smoothing sees a missing topology on the origin.
    no_edge_seq = GraphSnapshotSequence(
        [
            Data(x=torch.zeros(3, 2), edge_index=edge_index),
            Data(x=torch.ones(3, 2), edge_index=edge_index),
        ]
    )
    no_edge_seq[0].edge_index = None
    with pytest.raises(ValueError, match="neighbor_smoothing requires edge_index"):
        smooth.calibrate([no_edge_seq] * 15, steps=1, alpha=0.1)

    other_edge = _path_edge_index(num_nodes=4)
    mixed = cal_nw[:-1] + [
        GraphSnapshotSequence(
            [
                Data(x=torch.zeros(4, 2), edge_index=other_edge),
                Data(x=torch.ones(4, 2), edge_index=other_edge),
            ]
        )
    ]
    with pytest.raises(ValueError, match="fixed node count"):
        nw.calibrate(mixed, steps=1, alpha=0.1)

    adaptive_nw = ConformalKoopmanUQ(
        model, method="adaptive", score="node_wise", gamma=0.2
    )
    adaptive_nw.calibrate(cal_nw, steps=1, alpha=0.1)
    assert adaptive_nw.quantiles.shape == (1, 3)

    # predict_interval: missing x / node count mismatch / payload mismatches.
    adaptive_nw._quantiles = torch.zeros(1, 2)  # wrong N vs forecast
    with pytest.raises(ValueError, match="match the forecast node"):
        adaptive_nw.predict_interval(
            Data(x=torch.zeros(3, 2), edge_index=edge_index), steps=1, level=0.9
        )

    ok = ConformalKoopmanUQ(model, method="split", score="node_wise")
    ok.calibrate(cal_nw, steps=1, alpha=0.1)

    def _predict_missing_x(initial_graph, steps, **kwargs):
        return [Data(edge_index=edge_index) for _ in range(steps)]

    ok.model.predict = _predict_missing_x  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="predicted snapshots must define"):
        ok.predict_interval(
            Data(x=torch.zeros(3, 2), edge_index=edge_index), steps=1, level=0.9
        )

    path = tmp_path / "cal_ok.pt"
    _patch_zero_predict(ok.model)
    ok.save_calibration(path)
    bad_kind = ConformalKoopmanUQ(model, method="split", score="node_wise")
    torch.save({"kind": "nope"}, tmp_path / "bad_kind.pt")
    with pytest.raises(ValueError, match="unsupported calibration payload"):
        bad_kind.load_calibration(tmp_path / "bad_kind.pt")
    score_mismatch = ConformalKoopmanUQ(model, method="split", score="aggregate")
    with pytest.raises(ValueError, match="method/score"):
        score_mismatch.load_calibration(path)

    nw_ctrl = ConformalKoopmanUQ(model_ctrl, method="split", score="node_wise")
    nw_controls = [[torch.zeros(1)] for _ in range(len(cal_nw))]
    nw_ctrl.calibrate(cal_nw, steps=1, alpha=0.1, controls=nw_controls)

    # node_wise calibrate with predict returning x=None.
    model_bad_nw = _tiny_model()
    bad_pred_nw = ConformalKoopmanUQ(model_bad_nw, method="split", score="node_wise")

    def _predict_none_nw(initial_graph, steps, **kwargs):
        return [Data(x=None, edge_index=edge_index) for _ in range(steps)]

    bad_pred_nw.model.predict = _predict_none_nw  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="must define node features"):
        bad_pred_nw.calibrate(cal_nw, steps=1, alpha=0.1)
