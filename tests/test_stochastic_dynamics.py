"""Tests for ``dynamics_mode='stochastic'`` (TASK-1840 / TASK-1841)."""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor
from torch_geometric.data import Data

from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
from koopman_graph.nn import RelGraphDecoder, RelGraphEncoder
from koopman_graph.operators import (
    GraphKoopmanOperator,
    HeteroGraphKoopmanOperator,
    KoopmanOperator,
    apply_process_noise,
    attach_process_noise,
    diagonal_process_covariance,
    maybe_apply_process_noise,
)
from koopman_graph.operators.stochastic import process_noise_std, softplus_inverse

_MEAN_SAMPLES = 512
_MEAN_SEED = 42


def _tiny_model(**kwargs: object) -> GraphKoopmanModel:
    """Build a small homogeneous model for stochastic dynamics tests."""
    opts: dict[str, object] = {
        "encoder": GNNEncoder(in_channels=1, hidden_channels=4, latent_dim=2),
        "decoder": GNNDecoder(latent_dim=2, hidden_channels=4, out_channels=1),
        "latent_dim": 2,
        "time_step": 0.1,
    }
    opts.update(kwargs)
    return GraphKoopmanModel(**opts)  # type: ignore[arg-type]


def _hetero_model(**kwargs: object) -> GraphKoopmanModel:
    """Build a shared-d multiplex hetero model."""
    opts: dict[str, object] = {
        "encoder": RelGraphEncoder(
            3,
            hidden_channels=8,
            latent_dim=4,
            num_relations=2,
            num_layers=1,
        ),
        "decoder": RelGraphDecoder(
            latent_dim=4,
            hidden_channels=8,
            out_channels=3,
            num_relations=2,
            num_layers=1,
        ),
        "latent_dim": 4,
        "time_step": 1.0,
        "koopman": "hetero_graph",
    }
    opts.update(kwargs)
    return GraphKoopmanModel(**opts)  # type: ignore[arg-type]


def test_factory_accepts_stochastic_dense_graph_hetero() -> None:
    """Factory builds stochastic dense, graph, and shared-d hetero operators."""
    dense = _tiny_model(dynamics_mode="stochastic")
    assert dense.dynamics_mode == "stochastic"
    assert not dense.is_continuous
    assert isinstance(dense.koopman, KoopmanOperator)
    assert getattr(dense.koopman, "stochastic", False) is True
    assert dense.koopman.process_log_std.shape == (2,)

    graph = _tiny_model(dynamics_mode="stochastic", koopman="graph")
    assert isinstance(graph.koopman, GraphKoopmanOperator)
    assert graph.koopman.stochastic is True

    hetero = _hetero_model(dynamics_mode="stochastic")
    assert isinstance(hetero.koopman, HeteroGraphKoopmanOperator)
    assert hetero.koopman.stochastic is True
    assert not hetero.koopman.is_rectangular


def test_factory_rejects_unsupported_stochastic_kinds() -> None:
    """Hypergraph, continuous_graph, and rectangular hetero are rejected."""
    from koopman_graph.model.factory import resolve_model_components

    with pytest.raises(ValueError, match="stochastic"):
        _tiny_model(dynamics_mode="stochastic", koopman="hypergraph")
    with pytest.raises(ValueError, match="stochastic"):
        _tiny_model(dynamics_mode="stochastic", koopman="global_local")
    with pytest.raises(ValueError, match="continuous"):
        _tiny_model(dynamics_mode="stochastic", koopman="continuous_graph")

    encoder = RelGraphEncoder(
        {"a": 2, "b": 3},
        hidden_channels=8,
        latent_dim=4,
        num_relations=2,
        num_layers=1,
        node_types=("a", "b"),
        edge_types=(("a", "to_b", "b"), ("b", "to_a", "a")),
        latent_dims={"a": 2, "b": 3},
    )
    decoder = RelGraphDecoder(
        latent_dim=4,
        hidden_channels=8,
        out_channels={"a": 2, "b": 3},
        num_relations=2,
        num_layers=1,
        node_types=("a", "b"),
        edge_types=(("a", "to_b", "b"), ("b", "to_a", "a")),
        latent_dims={"a": 2, "b": 3},
    )
    with pytest.raises(ValueError, match="rectangular|latent_dims"):
        resolve_model_components(
            encoder,
            decoder,
            latent_dim=4,
            time_step=1.0,
            koopman="hetero_graph",
            koopman_node_types=("a", "b"),
            koopman_edge_types=(("a", "to_b", "b"), ("b", "to_a", "a")),
            koopman_latent_dims={"a": 2, "b": 3},
            dynamics_mode="stochastic",
            physics_position="prepend",
        )


def test_default_discrete_path_unchanged() -> None:
    """Default discrete models have no stochastic flag; advance equals forward."""
    model = _tiny_model()
    assert model.dynamics_mode == "discrete"
    assert getattr(model.koopman, "stochastic", False) is False
    assert not hasattr(model.koopman, "process_log_std")

    z = torch.randn(3, 2)
    assert torch.allclose(model.koopman.advance(z), model.koopman.forward(z))

    explicit = _tiny_model(dynamics_mode="discrete")
    assert explicit.dynamics_mode == "discrete"
    assert getattr(explicit.koopman, "stochastic", False) is False


def test_stochastic_advance_adds_noise_after_linear_map() -> None:
    """``advance`` differs from deterministic ``forward``; scale 0 matches."""
    model = _tiny_model(dynamics_mode="stochastic")
    koopman = model.koopman
    z = torch.randn(4, 2)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)

    det = koopman.forward(z)
    torch.manual_seed(1)
    noisy = koopman.advance(z)
    assert not torch.allclose(noisy, det)

    koopman.process_noise_scale = 0.0
    torch.manual_seed(1)
    assert torch.allclose(koopman.advance(z), det)

    graph = _tiny_model(dynamics_mode="stochastic", koopman="graph")
    g_det = graph.koopman.forward(z, edge_index)
    graph.koopman.process_noise_scale = 0.0
    assert torch.allclose(graph.koopman.advance(z, edge_index=edge_index), g_det)


def test_diagonal_process_covariance_structure() -> None:
    """Stacked Q is ``I_N ⊗ diag(σ²)`` with shape ``(N d, N d)``."""
    log_std = torch.tensor([-2.0, -1.0])
    q = diagonal_process_covariance(log_std, num_nodes=3)
    assert q.shape == (6, 6)
    sigma = torch.nn.functional.softplus(log_std)
    expected_diag = (sigma * sigma).repeat(3)
    assert torch.allclose(torch.diag(q), expected_diag)
    off = q.clone()
    off.fill_diagonal_(0.0)
    assert torch.allclose(off, torch.zeros_like(off))


def test_stochastic_predict_smoke() -> None:
    """Stochastic models still encode / predict finite snapshots."""
    model = _tiny_model(dynamics_mode="stochastic")
    snap = Data(
        x=torch.randn(3, 1),
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
    )
    preds = model.predict(snap, steps=2)
    assert len(preds) == 2
    assert preds[0].x is not None
    assert torch.isfinite(preds[0].x).all()


def _sample_advance_mean(
    koopman: KoopmanOperator | GraphKoopmanOperator,
    z: Tensor,
    *,
    edge_index: Tensor | None = None,
    n_samples: int,
    seed: int,
    scale: float,
) -> Tensor:
    """Monte Carlo mean of stochastic ``advance`` at a fixed noise scale.

    Parameters
    ----------
    koopman : KoopmanOperator or GraphKoopmanOperator
        Stochastic operator under test.
    z : Tensor
        Latent states ``(N, d)``.
    edge_index : Tensor or None, optional
        Topology for graph operators.
    n_samples : int
        Number of independent advances.
    seed : int
        Global RNG seed applied before sampling.
    scale : float
        Value written to ``process_noise_scale``.

    Returns
    -------
    Tensor
        Mean advanced latent with the same shape as ``z``.
    """
    koopman.process_noise_scale = scale
    torch.manual_seed(seed)
    samples: list[Tensor] = []
    for _ in range(n_samples):
        if edge_index is None:
            samples.append(koopman.advance(z).detach())
        else:
            samples.append(koopman.advance(z, edge_index=edge_index).detach())
    return torch.stack(samples, dim=0).mean(dim=0)


def _mean_atol(koopman: KoopmanOperator | GraphKoopmanOperator, scale: float) -> float:
    """Return a 6σ bound on the Monte Carlo mean error at ``scale``.

    Noise is i.i.d. ``N(0, (s σ)²)`` per latent entry. The sample mean of
    ``M`` draws has standard deviation ``s σ / sqrt(M)``. Tolerance uses
    ``6 * s * max(σ) / sqrt(M)``.
    """
    sigma_max = float(
        torch.nn.functional.softplus(koopman.process_log_std).detach().max()
    )
    return 6.0 * abs(scale) * sigma_max / math.sqrt(_MEAN_SAMPLES)


def _assert_mean_converges_to_forward(
    koopman: KoopmanOperator | GraphKoopmanOperator,
    z: Tensor,
    *,
    edge_index: Tensor | None = None,
) -> None:
    """Assert advance means approach ``forward`` as ``process_noise_scale`` → 0."""
    if edge_index is None:
        reference = koopman.forward(z).detach()
    else:
        reference = koopman.forward(z, edge_index).detach()

    # scale=0 skips sampling — bit-exact match to the linear map (no MC mean).
    koopman.process_noise_scale = 0.0
    if edge_index is None:
        zero_adv = koopman.advance(z).detach()
    else:
        zero_adv = koopman.advance(z, edge_index=edge_index).detach()
    assert torch.allclose(zero_adv, reference, atol=1e-7, rtol=0.0)

    scale_small = 1e-2
    mean_small = _sample_advance_mean(
        koopman,
        z,
        edge_index=edge_index,
        n_samples=_MEAN_SAMPLES,
        seed=_MEAN_SEED,
        scale=scale_small,
    )
    atol_small = _mean_atol(koopman, scale_small)
    assert torch.allclose(mean_small, reference, atol=atol_small, rtol=0.0)

    mean_unit = _sample_advance_mean(
        koopman,
        z,
        edge_index=edge_index,
        n_samples=_MEAN_SAMPLES,
        seed=_MEAN_SEED,
        scale=1.0,
    )
    assert torch.isfinite(mean_unit).all()
    err_unit = (mean_unit - reference).abs().max().item()
    err_small = (mean_small - reference).abs().max().item()
    assert err_unit >= err_small


def test_stochastic_mean_matches_deterministic_as_scale_to_zero() -> None:
    """Dense stochastic advance mean → ``forward`` as noise scale → 0.

    Tolerance uses a 6σ Monte Carlo mean bound:
    ``atol = 6 * scale * max(softplus(process_log_std)) / sqrt(M)`` with
    ``M = 512`` seeded draws (TASK-1841).
    """
    torch.manual_seed(0)
    model = _tiny_model(dynamics_mode="stochastic")
    z = torch.randn(4, 2)
    _assert_mean_converges_to_forward(model.koopman, z)


def test_softplus_inverse_roundtrip_and_rejects_nonpositive() -> None:
    """``softplus_inverse`` inverts positive values and rejects non-positive."""
    value = 0.01
    pre = softplus_inverse(value)
    assert torch.nn.functional.softplus(torch.tensor(pre)).item() == pytest.approx(
        value, rel=1e-6
    )
    with pytest.raises(ValueError, match="softplus_inverse requires a positive"):
        softplus_inverse(0.0)
    with pytest.raises(ValueError, match="softplus_inverse requires a positive"):
        softplus_inverse(-1.0)


def test_attach_process_noise_validation_and_registration() -> None:
    """``attach_process_noise`` validates args and registers parameters."""
    module = torch.nn.Module()
    with pytest.raises(ValueError, match="latent_dim must be positive"):
        attach_process_noise(module, latent_dim=0)
    with pytest.raises(ValueError, match="init_std must be positive"):
        attach_process_noise(module, latent_dim=2, init_std=0.0)

    attached = attach_process_noise(module, latent_dim=3, init_std=1e-3)
    assert attached is module
    assert module.stochastic is True
    assert module.process_noise_scale == 1.0
    assert module.process_log_std.shape == (3,)


def test_process_noise_std_and_apply_width_mismatch() -> None:
    """``process_noise_std`` / ``apply_process_noise`` guard missing or bad width."""
    bare = torch.nn.Module()
    with pytest.raises(AttributeError, match="process_log_std"):
        process_noise_std(bare)

    module = attach_process_noise(torch.nn.Module(), latent_dim=2, init_std=1e-3)
    std = process_noise_std(module)
    assert std.shape == (2,)
    assert (std > 0).all()

    z = torch.randn(4, 3)
    with pytest.raises(ValueError, match="process_log_std width must match"):
        apply_process_noise(z, module)


def test_diagonal_process_covariance_validation() -> None:
    """``diagonal_process_covariance`` rejects invalid ``num_nodes`` / ``log_std``."""
    log_std = torch.tensor([-1.0, -0.5])
    with pytest.raises(ValueError, match="num_nodes must be positive"):
        diagonal_process_covariance(log_std, num_nodes=0)
    with pytest.raises(ValueError, match="log_std must be 1-D"):
        diagonal_process_covariance(log_std.reshape(1, 2), num_nodes=2)


def test_maybe_apply_process_noise_skips_non_stochastic() -> None:
    """``maybe_apply_process_noise`` is a no-op when ``stochastic`` is false."""
    module = torch.nn.Module()
    z = torch.randn(3, 2)
    assert torch.equal(maybe_apply_process_noise(z, module), z)


def test_stochastic_graph_mean_matches_deterministic_as_scale_to_zero() -> None:
    """Graph stochastic advance mean → ``forward`` as noise scale → 0.

    Same Monte Carlo mean bound as the dense path (TASK-1841 networked case).
    """
    torch.manual_seed(1)
    model = _tiny_model(dynamics_mode="stochastic", koopman="graph")
    z = torch.randn(4, 2)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    _assert_mean_converges_to_forward(model.koopman, z, edge_index=edge_index)
