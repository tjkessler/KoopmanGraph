"""Tests for GlobalLocalKoopmanOperator (TASK-1306)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from koopman_graph import GlobalLocalKoopmanOperator, GraphKoopmanModel, KoopmanOperator
from koopman_graph.adaptation import RecursiveKoopmanAdapter
from koopman_graph.nn import GNNDecoder, GNNEncoder
from koopman_graph.operators.global_local import (
    DEFAULT_LOCAL_HIDDEN_DIMS,
    DEFAULT_LOCAL_RANK,
    DEFAULT_LOCAL_WINDOW,
    pad_latent_window,
)


def _tiny_model(*, koopman: str, seed: int = 0) -> GraphKoopmanModel:
    torch.manual_seed(seed)
    encoder = GNNEncoder(in_channels=2, hidden_channels=8, latent_dim=4, num_layers=2)
    decoder = GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=2, num_layers=2)
    return GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=4,
        time_step=1.0,
        koopman=koopman,  # type: ignore[arg-type]
    )


def test_global_local_defaults_and_exports() -> None:
    """Package export and constructor defaults match the approved plan."""
    operator = GlobalLocalKoopmanOperator(6)
    assert operator.local_window == DEFAULT_LOCAL_WINDOW
    assert operator.local_rank == DEFAULT_LOCAL_RANK
    assert operator.local_hidden_dims == DEFAULT_LOCAL_HIDDEN_DIMS
    assert GlobalLocalKoopmanOperator is not None


def test_advance_shape_and_cold_start() -> None:
    """Advance preserves shape; cold-start window matches padded z."""
    operator = GlobalLocalKoopmanOperator(5, local_window=3, local_rank=2)
    z = torch.randn(7, 5)
    z_next = operator.advance(z)
    assert z_next.shape == z.shape
    window = pad_latent_window(z, 3)
    z_next_explicit = operator.advance(z, latent_window=window)
    torch.testing.assert_close(z_next, z_next_explicit)


def test_matrix_and_spectrum_report_global_backbone() -> None:
    """matrix / spectral_radius / spectrum use K_g only."""
    operator = GlobalLocalKoopmanOperator(4, init_mode="identity")
    assert torch.allclose(operator.matrix, operator._global.K)
    radius = operator.spectral_radius()
    assert radius.ndim == 0
    spectrum = operator.spectrum()
    assert spectrum.eigenvalues.shape[0] == 4


def test_bound_metric_on_kg() -> None:
    """bound_metric delegates to the global backbone."""
    operator = GlobalLocalKoopmanOperator(3, parameterization="odo")
    torch.testing.assert_close(operator.bound_metric(), operator._global.bound_metric())


def test_approximate_inverse_rejects_inverse_matrix() -> None:
    """inverse_advance uses K_g and rejects inverse_matrix=."""
    operator = GlobalLocalKoopmanOperator(4, init_mode="identity")
    z = torch.randn(3, 4)
    recovered = operator.inverse_advance(z)
    assert recovered.shape == z.shape
    with pytest.raises(ValueError, match="inverse_matrix is not supported"):
        operator.inverse_advance(z, inverse_matrix=torch.eye(4))


def test_factory_global_local_and_continuous_reject() -> None:
    """Factory accepts global_local discrete; rejects continuous."""
    model = _tiny_model(koopman="global_local")
    assert model.koopman_kind == "global_local"
    assert isinstance(model.koopman, GlobalLocalKoopmanOperator)
    with pytest.raises(ValueError, match="requires dynamics_mode='discrete'"):
        GraphKoopmanModel(
            encoder=GNNEncoder(in_channels=2, hidden_channels=8, latent_dim=4),
            decoder=GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=2),
            latent_dim=4,
            time_step=1.0,
            dynamics_mode="continuous",
            koopman="global_local",
        )


def test_factory_validates_local_config() -> None:
    """Non-default local kwargs require global_local; window/rank validated."""
    with pytest.raises(ValueError, match="require koopman='global_local'"):
        GraphKoopmanModel(
            encoder=GNNEncoder(in_channels=2, hidden_channels=8, latent_dim=4),
            decoder=GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=2),
            latent_dim=4,
            time_step=1.0,
            koopman="pernode",
            koopman_local_window=8,
        )
    with pytest.raises(ValueError, match="local_window must be positive"):
        GlobalLocalKoopmanOperator(4, local_window=0)
    with pytest.raises(ValueError, match="local_rank must be positive"):
        GlobalLocalKoopmanOperator(4, local_rank=0)


def test_rls_rejects_global_local() -> None:
    """RLS adapter raises a clear TypeError for global/local operators."""
    operator = GlobalLocalKoopmanOperator(4, parameterization="dense")
    with pytest.raises(TypeError, match="does not support GlobalLocalKoopmanOperator"):
        RecursiveKoopmanAdapter.from_operator(operator, mode="discrete")


def test_format1_checkpoint_round_trip(tmp_path: Path) -> None:
    """Format-1 save/load preserves kind and local-network config."""
    model = GraphKoopmanModel(
        encoder=GNNEncoder(in_channels=2, hidden_channels=8, latent_dim=4),
        decoder=GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=2),
        latent_dim=4,
        time_step=1.0,
        koopman="global_local",
        koopman_local_window=5,
        koopman_local_rank=3,
        koopman_local_hidden_dims=(16,),
    )
    path = tmp_path / "gl.pt"
    model.save(path)
    loaded = GraphKoopmanModel.load(path)
    assert loaded.koopman_kind == "global_local"
    assert isinstance(loaded.koopman, GlobalLocalKoopmanOperator)
    assert loaded.koopman.local_window == 5
    assert loaded.koopman.local_rank == 3
    assert loaded.koopman.local_hidden_dims == (16,)


def test_regime_switching_beats_single_operator() -> None:
    """Seeded latent regime-switch: global/local beats a single dense operator.

    Trains operators directly on a two-regime latent series (matched latent
    dim / seed / steps) and compares held-out autoregressive rollout RMSE.
    """
    from koopman_graph.operators.global_local import stack_latent_window

    torch.manual_seed(0)
    latent_dim = 3
    k_a = 0.95 * torch.eye(latent_dim)
    k_b = torch.tensor(
        [
            [0.2, -0.7, 0.0],
            [0.7, 0.2, 0.0],
            [0.0, 0.0, -0.5],
        ],
        dtype=torch.float32,
    )
    z = torch.randn(latent_dim)
    series: list[torch.Tensor] = [z.clone()]
    for step in range(99):
        k = k_a if step < 50 else k_b
        z = z @ k.T
        series.append(z.clone())
    train_series = series[:70]
    test_series = series[70:]

    def _train(operator: torch.nn.Module, *, epochs: int = 400) -> None:
        optim = torch.optim.Adam(operator.parameters(), lr=1e-2)
        window = int(getattr(operator, "local_window", 1))
        for _ in range(epochs):
            optim.zero_grad()
            losses = []
            history: list[torch.Tensor] = []
            for index in range(len(train_series) - 1):
                current = train_series[index]
                target = train_series[index + 1]
                if isinstance(operator, GlobalLocalKoopmanOperator):
                    latent_window = stack_latent_window(
                        history, window=window, current=current
                    )
                    pred = operator.advance(current, latent_window=latent_window)
                    history.append(current)
                    if len(history) >= window:
                        history = history[-(window - 1) :]
                else:
                    pred = operator.advance(current)
                losses.append(torch.mean((pred - target) ** 2))
            torch.stack(losses).mean().backward()
            torch.nn.utils.clip_grad_norm_(operator.parameters(), 1.0)
            optim.step()

    def _rollout_rmse(operator: torch.nn.Module, horizon: int = 15) -> float:
        window = int(getattr(operator, "local_window", 1))
        warm = train_series[-window:]
        history = list(warm[:-1])
        latent = warm[-1].clone()
        errs = []
        for step in range(horizon):
            if isinstance(operator, GlobalLocalKoopmanOperator):
                latent_window = stack_latent_window(
                    history, window=window, current=latent
                )
                pred = operator.advance(latent, latent_window=latent_window)
                history.append(latent)
                if len(history) >= window:
                    history = history[-(window - 1) :]
            else:
                pred = operator.advance(latent)
            errs.append(torch.mean((pred - test_series[step]) ** 2))
            latent = pred.detach()
        return float(torch.sqrt(torch.stack(errs).mean()).item())

    torch.manual_seed(0)
    global_local = GlobalLocalKoopmanOperator(
        latent_dim, local_window=8, local_rank=3, local_hidden_dims=(64,)
    )
    torch.manual_seed(0)
    single = KoopmanOperator(latent_dim)
    _train(global_local)
    _train(single)
    rmse_gl = _rollout_rmse(global_local)
    rmse_single = _rollout_rmse(single)
    assert rmse_gl < rmse_single
    assert rmse_gl < 0.95 * rmse_single


def test_local_correction_small_at_init() -> None:
    """Fresh local MLP yields a small gated correction (not an exact zero)."""
    operator = GlobalLocalKoopmanOperator(4, init_mode="identity")
    z = torch.randn(2, 4)
    window = pad_latent_window(z, operator.local_window)
    k_ell = operator.local_correction(window)
    assert torch.max(torch.abs(k_ell)) < 0.05
    global_only = KoopmanOperator(4, init_mode="identity")
    with torch.no_grad():
        global_only.K.copy_(operator.matrix)
    # Advance stays close to the global backbone at initialization.
    torch.testing.assert_close(
        operator.advance(z),
        global_only.advance(z),
        atol=5e-2,
        rtol=5e-2,
    )


def test_gradient_flows_through_local_mlp() -> None:
    """Loss on advance backprops into the local network."""
    operator = GlobalLocalKoopmanOperator(3, local_window=2, local_rank=1)
    z = torch.randn(5, 3, requires_grad=False)
    window = pad_latent_window(z, 2)
    z_next = operator.advance(z, latent_window=window)
    loss = z_next.square().mean()
    loss.backward()
    grads = [parameter.grad for parameter in operator._local_net.parameters()]
    assert any(grad is not None and torch.any(grad != 0) for grad in grads)
    assert operator._local_logit.grad is not None
    assert torch.any(operator._local_logit.grad != 0)
