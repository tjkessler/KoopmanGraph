"""Tests for diagonal Laplace Bayesian UQ (TASK-1839 / TASK-1843)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from unittest.mock import PropertyMock, patch

import pytest
import torch
from tests.helpers import REPO_ROOT
from torch import Tensor
from torch_geometric.data import Data, HeteroData

import koopman_graph
import koopman_graph.uq as uq_pkg
from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.datasets import SyntheticDynamicGraphBenchmark
from koopman_graph.operators import GraphKoopmanOperator, KoopmanOperator
from koopman_graph.uq import (
    BayesianKoopmanUQ,
    IntervalForecastModel,
    LaplaceFactorSpec,
    LaplacePosterior,
    PredictionInterval,
)
from koopman_graph.uq.bayesian import (
    _advance_from_theta,
    _apply_theta,
    _as_data_sequence,
    _control_offset,
    _factor_specs,
    _matrices_from_theta,
    _neighbor_term_from_factors,
    _read_factor_vector,
)

_MODULE_PATH = REPO_ROOT / "src" / "koopman_graph" / "uq" / "bayesian.py"


def _tiny_factory(*, koopman: str | None = None, **kwargs: object) -> GraphKoopmanModel:
    """Build a small GraphKoopmanModel suitable for fast Bayesian UQ tests."""
    encoder = GNNEncoder(in_channels=1, hidden_channels=4, latent_dim=2)
    decoder = GNNDecoder(latent_dim=2, hidden_channels=4, out_channels=1)
    opts: dict[str, object] = {
        "encoder": encoder,
        "decoder": decoder,
        "latent_dim": 2,
        "time_step": 0.1,
    }
    if koopman is not None:
        opts["koopman"] = koopman
    opts.update(kwargs)
    return GraphKoopmanModel(**opts)  # type: ignore[arg-type]


def _fit_tiny_uq(
    *,
    koopman: str | None = None,
    n_samples: int = 16,
) -> tuple[BayesianKoopmanUQ, list[Data]]:
    """Fit a tiny model and Laplace posterior for smoke tests."""
    torch.manual_seed(0)
    model = _tiny_factory(koopman=koopman)
    sequence = SyntheticDynamicGraphBenchmark.generate(
        num_nodes=4,
        num_timesteps=8,
        in_channels=1,
        noise_std=0.0,
        seed=0,
    )
    model.fit(sequence, epochs=2, lr=1e-2)
    uq = BayesianKoopmanUQ(model, n_samples=n_samples, prior_precision=1.0)
    uq.fit_posterior([sequence])
    return uq, list(sequence)


def test_sample_forecast_interval_ordering_smoke() -> None:
    """Seeded sample_forecast returns ordered PredictionInterval shapes."""
    uq, sequence = _fit_tiny_uq(n_samples=24)
    assert isinstance(uq, IntervalForecastModel)

    generator = torch.Generator().manual_seed(7)
    interval = uq.sample_forecast(
        sequence[0],
        steps=3,
        level=0.8,
        generator=generator,
    )
    assert isinstance(interval, PredictionInterval)
    assert interval.level == 0.8
    assert interval.n_members == 24
    assert len(interval.mean) == 3
    assert len(interval.lower) == 3
    assert len(interval.upper) == 3
    for mean, lower, upper in zip(
        interval.mean, interval.lower, interval.upper, strict=True
    ):
        assert isinstance(mean, Data)
        assert mean.x is not None and lower.x is not None and upper.x is not None
        assert mean.x.shape == (4, 1)
        assert torch.all(lower.x <= mean.x + 1e-4)
        assert torch.all(mean.x <= upper.x + 1e-4)


def test_sample_forecast_seeded_reproducibility() -> None:
    """Identical generators reproduce the same interval tensors."""
    uq, sequence = _fit_tiny_uq(n_samples=12)
    g1 = torch.Generator().manual_seed(11)
    g2 = torch.Generator().manual_seed(11)
    a = uq.sample_forecast(sequence[0], steps=2, level=0.9, generator=g1)
    b = uq.sample_forecast(sequence[0], steps=2, level=0.9, generator=g2)
    for left, right in zip(a.mean, b.mean, strict=True):
        assert torch.allclose(left.x, right.x, atol=1e-6)
    for left, right in zip(a.lower, b.lower, strict=True):
        assert torch.allclose(left.x, right.x, atol=1e-6)


def test_predict_interval_aliases_sample_forecast() -> None:
    """predict_interval matches sample_forecast under a shared generator seed."""
    uq, sequence = _fit_tiny_uq(n_samples=8)
    g1 = torch.Generator().manual_seed(3)
    g2 = torch.Generator().manual_seed(3)
    via_sample = uq.sample_forecast(sequence[0], steps=2, level=0.9, generator=g1)
    via_alias = uq.predict_interval(sequence[0], steps=2, level=0.9, generator=g2)
    for left, right in zip(via_sample.mean, via_alias.mean, strict=True):
        assert torch.allclose(left.x, right.x, atol=1e-6)


def test_graph_koopman_factor_layout() -> None:
    """Graph operators flatten K_self and K_nbr into the Laplace posterior."""
    uq, sequence = _fit_tiny_uq(koopman="graph", n_samples=8)
    posterior = uq.posterior
    assert posterior is not None
    names = [spec.name for spec in posterior.factors]
    assert names == ["K_self", "K_nbr"]
    assert posterior.mean.numel() == 2 * (2 * 2)
    interval = uq.sample_forecast(sequence[0], steps=2, n_samples=8, level=0.9)
    assert interval.n_members == 8
    for lower, mean, upper in zip(
        interval.lower, interval.mean, interval.upper, strict=True
    ):
        assert torch.all(lower.x <= mean.x + 1e-4)
        assert torch.all(mean.x <= upper.x + 1e-4)


def test_rejects_non_dense_parameterization() -> None:
    """Non-dense factorizations raise a clear ValueError."""
    model = _tiny_factory(koopman_parameterization="schur")
    with pytest.raises(ValueError, match="parameterization='dense'"):
        BayesianKoopmanUQ(model)


def test_rejects_bilinear_control() -> None:
    """Bilinear control mode is rejected at construction."""
    model = _tiny_factory(control_dim=1, control_mode="bilinear")
    with pytest.raises(ValueError, match="bilinear"):
        BayesianKoopmanUQ(model)


def test_rejects_hetero_model() -> None:
    """Hetero models are rejected with a conformal pointer."""
    model = _tiny_factory()
    with (
        patch.object(
            GraphKoopmanModel,
            "uses_hetero_koopman",
            new_callable=PropertyMock,
            return_value=True,
        ),
        pytest.raises(ValueError, match="ConformalKoopmanUQ"),
    ):
        BayesianKoopmanUQ(model)


def test_rejects_hetero_calibration_sequence() -> None:
    """HeteroData calibration trajectories raise ValueError."""
    model = _tiny_factory()
    uq = BayesianKoopmanUQ(model)
    hetero = HeteroData()
    hetero["node"].x = torch.zeros(2, 1)
    with pytest.raises(ValueError, match="HeteroData"):
        uq.fit_posterior([[hetero, hetero]])


def test_sample_forecast_requires_posterior() -> None:
    """sample_forecast before fit_posterior raises ValueError."""
    model = _tiny_factory()
    uq = BayesianKoopmanUQ(model)
    snap = Data(
        x=torch.zeros(3, 1),
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
    )
    with pytest.raises(ValueError, match="fit_posterior"):
        uq.sample_forecast(snap, steps=1)


def test_fit_posterior_restores_map_after_forecast() -> None:
    """sample_forecast restores MAP factors after Monte Carlo writes."""
    uq, sequence = _fit_tiny_uq(n_samples=4)
    assert uq.posterior is not None
    k_before = uq.model.koopman.K.detach().clone()
    uq.sample_forecast(sequence[0], steps=2, n_samples=4, level=0.9)
    assert torch.allclose(k_before, uq.model.koopman.K.detach(), atol=1e-6)


def test_bayesian_exports_and_not_on_root_all() -> None:
    """Bayesian symbols live on uq.__all__ only (power-user module)."""
    names = {"BayesianKoopmanUQ", "LaplacePosterior", "LaplaceFactorSpec"}
    assert names.issubset(set(uq_pkg.__all__))
    root = set(koopman_graph.__all__)
    assert names.isdisjoint(root)
    assert uq_pkg.BayesianKoopmanUQ is BayesianKoopmanUQ
    assert uq_pkg.LaplacePosterior is LaplacePosterior
    assert uq_pkg.LaplaceFactorSpec is LaplaceFactorSpec


def _interval_widths(interval: PredictionInterval) -> list[Tensor]:
    """Return per-step ``upper - lower`` feature tensors."""
    widths: list[Tensor] = []
    for lower, upper in zip(interval.lower, interval.upper, strict=True):
        assert lower.x is not None and upper.x is not None
        widths.append(upper.x - lower.x)
    return widths


def _assert_ordered_nonneg_widths(
    interval: PredictionInterval,
    *,
    atol: float = 1e-4,
) -> None:
    """Assert lower ≤ mean ≤ upper and nonnegative widths entrywise."""
    for mean, lower, upper in zip(
        interval.mean, interval.lower, interval.upper, strict=True
    ):
        assert mean.x is not None and lower.x is not None and upper.x is not None
        assert torch.all(lower.x <= mean.x + atol)
        assert torch.all(mean.x <= upper.x + atol)
        assert torch.all(upper.x - lower.x >= -atol)


def _mean_width(interval: PredictionInterval) -> float:
    """Scalar mean of all entrywise interval widths."""
    widths = _interval_widths(interval)
    stacked = torch.stack([w.reshape(-1) for w in widths], dim=0)
    return float(stacked.mean().item())


def test_bayesian_ordering_and_widths_dense_and_graph() -> None:
    """Seeded dense+graph intervals obey ordering and nonnegative widths.

    Uses ``n_samples=32`` Monte Carlo factor draws (TASK-1843). Does not
    assert exact numeric widths.
    """
    n_samples = 32
    for koopman in (None, "graph"):
        uq, sequence = _fit_tiny_uq(koopman=koopman, n_samples=n_samples)
        generator = torch.Generator().manual_seed(17)
        interval = uq.sample_forecast(
            sequence[0],
            steps=3,
            level=0.9,
            n_samples=n_samples,
            generator=generator,
        )
        assert interval.n_members == n_samples
        _assert_ordered_nonneg_widths(interval)
        for width in _interval_widths(interval):
            assert torch.all(width >= -1e-4)


def test_bayesian_weaker_prior_widens_mean_width() -> None:
    """Weaker prior_precision yields mean width ≥ stronger prior (seeded).

    Same fitted model and calibration; ``n_samples=32``, level ``0.9``.
    Compares mean of ``(upper - lower)`` only — no exact-width target.
    """
    torch.manual_seed(0)
    model = _tiny_factory()
    sequence = SyntheticDynamicGraphBenchmark.generate(
        num_nodes=4,
        num_timesteps=8,
        in_channels=1,
        noise_std=0.0,
        seed=0,
    )
    model.fit(sequence, epochs=2, lr=1e-2)

    n_samples = 32
    weak = BayesianKoopmanUQ(model, n_samples=n_samples, prior_precision=0.1)
    strong = BayesianKoopmanUQ(model, n_samples=n_samples, prior_precision=10.0)
    weak.fit_posterior([sequence])
    strong.fit_posterior([sequence])

    g_weak = torch.Generator().manual_seed(21)
    g_strong = torch.Generator().manual_seed(21)
    interval_weak = weak.sample_forecast(
        sequence[0],
        steps=2,
        level=0.9,
        n_samples=n_samples,
        generator=g_weak,
    )
    interval_strong = strong.sample_forecast(
        sequence[0],
        steps=2,
        level=0.9,
        n_samples=n_samples,
        generator=g_strong,
    )
    _assert_ordered_nonneg_widths(interval_weak)
    _assert_ordered_nonneg_widths(interval_strong)
    assert _mean_width(interval_weak) >= _mean_width(interval_strong)


def test_bayesian_honesty_docs() -> None:
    """Module docs bound scope: diagonal Laplace, not BNN/DPK/K2VAE."""
    source = _MODULE_PATH.read_text(encoding="utf-8")
    combined = source + (BayesianKoopmanUQ.__doc__ or "")
    lower = combined.lower()
    assert "diagonal" in lower
    assert "bnn" in lower or "bayesian neural" in lower
    assert "not" in lower
    assert "dpk" in lower or "deep probabilistic" in lower
    assert "encoder" in lower
    assert "coverage" in lower


def test_bayesian_uq_convention_self_review() -> None:
    """UQ convention: credible interpretation, assumptions, no false precision.

    Self-review against ``uncertainty-quantification``: docs distinguish
    approximate credible bands from coverage guarantees; encoders are
    point-estimated; width tests in this module use inequality checks only.
    """
    source = _MODULE_PATH.read_text(encoding="utf-8")
    combined = (source + (BayesianKoopmanUQ.__doc__ or "")).lower()
    assert "credible" in combined
    assert "coverage" in combined
    assert "diagonal" in combined
    assert "point-estimated" in combined or "point estimated" in combined
    assert "encoder" in combined

    suite = Path(__file__).read_text(encoding="utf-8")
    # Width checks are inequality-only (ordering / mean-width ≥), not exact targets.
    assert "_assert_ordered_nonneg_widths" in suite
    assert "_mean_width(interval_weak) >=" in suite
    forbidden = "assert " + "width =="
    assert forbidden not in suite.split("test_bayesian_uq_convention_self_review")[0]


def _transition_controls(
    sequence: Sequence[Data],
    *,
    control_dim: int = 1,
) -> list[Tensor]:
    """Build per-transition controls aligned with ``len(sequence) - 1``."""
    return [torch.zeros(control_dim) for _ in range(len(sequence) - 1)]


@pytest.mark.parametrize(
    ("kwargs", "pattern"),
    [
        ({"prior_precision": 0.0}, "prior_precision must be positive"),
        ({"prior_precision": -1.0}, "prior_precision must be positive"),
        ({"observation_noise": 0.0}, "observation_noise must be positive"),
        ({"n_samples": 0}, "n_samples must be >= 1"),
    ],
)
def test_init_rejects_invalid_hyperparameters(
    kwargs: dict[str, float | int],
    pattern: str,
) -> None:
    """Constructor validates prior, noise scale, and sample count."""
    model = _tiny_factory()
    with pytest.raises(ValueError, match=pattern):
        BayesianKoopmanUQ(model, **kwargs)  # type: ignore[arg-type]


def test_rejects_unsupported_koopman_type() -> None:
    """Non-operator koopman modules raise TypeError at construction."""
    model = _tiny_factory()
    model.koopman = torch.nn.Linear(2, 2)  # type: ignore[assignment]
    with pytest.raises(TypeError, match="GraphKoopmanOperator"):
        BayesianKoopmanUQ(model)


def test_as_data_sequence_rejects_empty_and_bad_entries() -> None:
    """Calibration normalization enforces non-empty homogeneous Data lists."""
    snap = Data(
        x=torch.zeros(2, 1),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
    )
    with pytest.raises(ValueError, match="non-empty"):
        _as_data_sequence([])
    with pytest.raises(ValueError, match="HeteroData"):
        _as_data_sequence([HeteroData()])
    with pytest.raises(TypeError, match="Data snapshots"):
        _as_data_sequence([snap, 42])  # type: ignore[list-item]
    assert _as_data_sequence([snap, snap]) == [snap, snap]


def test_fit_posterior_rejects_empty_and_misaligned_inputs() -> None:
    """fit_posterior validates trajectory and control alignment."""
    model = _tiny_factory()
    uq = BayesianKoopmanUQ(model)
    snap = Data(
        x=torch.zeros(3, 1),
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
    )
    with pytest.raises(ValueError, match="calibration_sequences must be non-empty"):
        uq.fit_posterior([])
    with pytest.raises(ValueError, match="at least two snapshots"):
        uq.fit_posterior([[snap]])
    with pytest.raises(ValueError, match="controls must align"):
        uq.fit_posterior([[snap, snap]], controls=[])
    with pytest.raises(ValueError, match="controls\\[i\\] must have length"):
        uq.fit_posterior([[snap, snap, snap]], controls=[[torch.zeros(1)]])


def test_fit_posterior_controlled_pernode_smoke() -> None:
    """Controlled per-node fit_posterior accepts aligned transition controls."""
    torch.manual_seed(0)
    snaps = SyntheticDynamicGraphBenchmark.generate(
        num_nodes=4,
        num_timesteps=6,
        in_channels=1,
        noise_std=0.0,
        seed=0,
    )
    snap_list = list(snaps)
    fit_sequence = GraphSnapshotSequence(
        snap_list,
        control_inputs=torch.zeros(len(snap_list), 1),
    )
    model = _tiny_factory(control_dim=1)
    model.fit(fit_sequence, epochs=2, lr=1e-2)
    uq = BayesianKoopmanUQ(model, n_samples=4)
    controls = [_transition_controls(snap_list, control_dim=1)]
    posterior = uq.fit_posterior([snap_list], controls=controls)
    assert posterior.n_data == len(snap_list) - 1
    assert posterior.diag_variance.numel() == posterior.mean.numel()


def test_dual_random_walk_factor_layout_and_forecast() -> None:
    """dual_random_walk graph operators include K_bwd in the Laplace layout."""
    torch.manual_seed(0)
    sequence = SyntheticDynamicGraphBenchmark.generate(
        num_nodes=4,
        num_timesteps=8,
        in_channels=1,
        noise_std=0.0,
        seed=0,
    )
    model = _tiny_factory(koopman="graph", koopman_adjacency="dual_random_walk")
    model.fit(sequence, epochs=2, lr=1e-2)
    uq = BayesianKoopmanUQ(model, n_samples=6)
    posterior = uq.fit_posterior([sequence])
    names = [spec.name for spec in posterior.factors]
    assert names == ["K_self", "K_nbr", "K_bwd"]
    interval = uq.sample_forecast(sequence[0], steps=2, n_samples=6, level=0.9)
    assert interval.n_members == 6
    _assert_ordered_nonneg_widths(interval)


def test_sample_forecast_n_samples_one_collapses_bounds() -> None:
    """A single Monte Carlo draw yields lower == mean == upper."""
    uq, sequence = _fit_tiny_uq(n_samples=8)
    interval = uq.sample_forecast(sequence[0], steps=2, n_samples=1, level=0.9)
    assert interval.n_members == 1
    for mean, lower, upper in zip(
        interval.mean, interval.lower, interval.upper, strict=True
    ):
        assert mean.x is not None and lower.x is not None and upper.x is not None
        assert torch.allclose(lower.x, mean.x)
        assert torch.allclose(mean.x, upper.x)


def test_sample_forecast_compose_latent_gaussian_smoke() -> None:
    """Aleatoric decode noise path runs without changing interval ordering."""
    uq, sequence = _fit_tiny_uq(n_samples=8)
    generator = torch.Generator().manual_seed(5)
    interval = uq.sample_forecast(
        sequence[0],
        steps=2,
        n_samples=8,
        level=0.9,
        compose_latent_gaussian=True,
        generator=generator,
    )
    _assert_ordered_nonneg_widths(interval)


def test_sample_forecast_rejects_bad_args_and_hetero_origin() -> None:
    """sample_forecast guards steps, samples, hetero origins, and extra args."""
    uq, sequence = _fit_tiny_uq(n_samples=4)
    with pytest.raises(ValueError, match="steps must be >= 1"):
        uq.sample_forecast(sequence[0], steps=0)
    with pytest.raises(ValueError, match="n_samples must be >= 1"):
        uq.sample_forecast(sequence[0], steps=1, n_samples=0)
    hetero = HeteroData()
    hetero["node"].x = torch.zeros(2, 1)
    with pytest.raises(ValueError, match="HeteroData origins"):
        uq.sample_forecast(hetero, steps=1)
    with pytest.raises(TypeError, match="no positional args"):
        uq.sample_forecast(sequence[0], 1, "extra")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="unexpected keyword arguments"):
        uq.sample_forecast(sequence[0], steps=1, unknown_flag=True)  # type: ignore[call-arg]


def test_factor_helpers_roundtrip_and_validation() -> None:
    """Internal factor flatten/unflatten helpers round-trip and validate inputs."""
    pernode = _tiny_factory()
    graph = _tiny_factory(koopman="graph")
    pernode_specs = _factor_specs(pernode.koopman)
    graph_specs = _factor_specs(graph.koopman)
    assert [s.name for s in pernode_specs] == ["K"]
    assert [s.name for s in graph_specs] == ["K_self", "K_nbr"]

    theta = _read_factor_vector(pernode.koopman, pernode_specs)
    mats = _matrices_from_theta(theta, pernode_specs)
    assert mats["K"].shape == (2, 2)
    _apply_theta(pernode.koopman, theta, pernode_specs)
    assert torch.allclose(theta, _read_factor_vector(pernode.koopman, pernode_specs))

    bad_spec = (LaplaceFactorSpec(name="K_unknown", shape=(2, 2), offset=0, numel=4),)
    with pytest.raises(ValueError, match="unknown factor"):
        _read_factor_vector(pernode.koopman, bad_spec)


def test_control_offset_and_neighbor_term_guards() -> None:
    """Control offsets and neighbor terms raise on inconsistent inputs."""
    uncontrolled = _tiny_factory()
    controlled = _tiny_factory(control_dim=1)
    z = torch.randn(3, 2)
    assert isinstance(uncontrolled.koopman, KoopmanOperator)
    assert isinstance(controlled.koopman, KoopmanOperator)

    with pytest.raises(ValueError, match="uncontrolled operator"):
        _control_offset(uncontrolled.koopman, z, torch.zeros(1))
    with pytest.raises(ValueError, match="control input is required"):
        _control_offset(controlled.koopman, z, None)

    dual = GraphKoopmanOperator(2, init_mode="identity", adjacency="dual_random_walk")
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    with pytest.raises(ValueError, match="K_bwd is required"):
        _neighbor_term_from_factors(
            dual,
            z,
            edge_index,
            None,
            torch.eye(2),
            None,
        )


def test_advance_from_theta_requires_graph_topology() -> None:
    """Graph one-step advance rejects missing edge_index."""
    model = _tiny_factory(koopman="graph")
    assert isinstance(model.koopman, GraphKoopmanOperator)
    factors = _factor_specs(model.koopman)
    theta = _read_factor_vector(model.koopman, factors)
    z = torch.randn(4, 2)
    with pytest.raises(ValueError, match="edge_index"):
        _advance_from_theta(
            model.koopman,
            z,
            theta,
            factors,
            edge_index=None,
            edge_weight=None,
            control=None,
        )
