"""Tests for conformal prediction UQ (TASK-1313)."""

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
