"""Tests for the discrete parametric interpolant :math:`K(\\mu)`."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from koopman_graph.model import GraphKoopmanModel
from koopman_graph.nn import GNNDecoder, GNNEncoder
from koopman_graph.operators import (
    LeaveOneRegimeOutReport,
    ParametricKoopmanOperator,
    leave_one_regime_out,
)
from koopman_graph.operators.parametric import INTERPOLANT_SAFE_PARAMETERIZATIONS


def _true_k(mu: float, k0: torch.Tensor, k1: torch.Tensor) -> torch.Tensor:
    """Affine interpolant :math:`(1-\\mu)K_0 + \\mu K_1`.

    Parameters
    ----------
    mu : float
        Scalar regime coordinate in ``[0, 1]``.
    k0, k1 : Tensor
        Endpoint maps with shape ``(d, d)``.

    Returns
    -------
    Tensor
        Mixed map.
    """
    return (1.0 - mu) * k0 + mu * k1


def test_parametric_rbf_midpoint_is_equal_mix() -> None:
    """RBF weights at the midpoint of two unit-interval anchors are 1/2, 1/2."""
    operator = ParametricKoopmanOperator(3, num_modes=2, parameter_dim=1)
    k0 = 0.9 * torch.eye(3)
    k1 = torch.diag(torch.tensor([0.2, 0.4, 0.6]))
    operator.modes[0].set_dense_matrix(k0)
    operator.modes[1].set_dense_matrix(k1)
    operator.set_anchors(torch.tensor([[0.0], [1.0]]))
    weights = operator.interpolation_weights(torch.tensor([0.5])).detach()
    expected_weights = torch.tensor([0.5, 0.5])
    torch.testing.assert_close(weights, expected_weights, rtol=0.0, atol=1e-6)
    mixed = operator.effective_matrix(torch.tensor([0.5]))
    expected = 0.5 * (k0 + k1)
    torch.testing.assert_close(mixed, expected, rtol=1e-6, atol=1e-8)


def test_parametric_simplex_weights_are_a_simplex() -> None:
    """Affine-softmax weights are nonnegative and sum to one."""
    operator = ParametricKoopmanOperator(
        2, num_modes=3, parameter_dim=2, weight_kind="simplex"
    )
    weights = operator.interpolation_weights(torch.tensor([0.3, -0.2]))
    assert torch.all(weights.detach() >= 0)
    assert float(weights.detach().sum()) == pytest.approx(1.0, rel=0.0, abs=1e-6)


def test_parametric_refuses_symplectic_mix() -> None:
    """Convex combination would silently drop the symplectic constraint."""
    with pytest.raises(ValueError, match="symplectic"):
        ParametricKoopmanOperator(4, parameterization="symplectic")
    for name in ("odo", "schur", "dissipative", "lyapunov", "auxiliary_spectral"):
        with pytest.raises(ValueError, match=name):
            ParametricKoopmanOperator(3, parameterization=name)  # type: ignore[arg-type]
    assert "dense" in INTERPOLANT_SAFE_PARAMETERIZATIONS
    _ = ParametricKoopmanOperator(3, parameterization="row_stochastic")
    _ = ParametricKoopmanOperator(3, parameterization="doubly_stochastic")


def test_parametric_requires_parameters() -> None:
    """Advance without μ raises rather than falling back to a latent gate."""
    operator = ParametricKoopmanOperator(2)
    z = torch.randn(4, 2)
    with pytest.raises(ValueError, match="requires parameters"):
        operator.advance(z)
    operator.set_parameters(torch.tensor([0.25]))
    assert operator.advance(z).shape == z.shape
    assert operator.inverse_advance(z).shape == z.shape
    assert operator(z, parameters=torch.tensor([0.1])).shape == z.shape
    with pytest.raises(ValueError, match="parameters must have shape"):
        operator.advance(z, parameters=torch.zeros(2))
    mixed = operator.matrix
    assert mixed.shape == (2, 2)
    assert operator.bound_metric().ndim == 0


def test_leave_one_regime_out_beats_pooled_lti() -> None:
    """Hold-out interior μ: interpolant one-step MSE is below pooled LTI.

    Truth is the affine mix :math:`K(\\mu)=(1-\\mu)K_0+\\mu K_1`. Train
    regimes are unbalanced (many pairs at μ=0, few at μ=1) so pooled
    least squares is biased toward :math:`K_0`. The held-out regime is
    μ=0.5. Synthetic oracle; rel/abs from noiseless construction.
    """
    torch.manual_seed(0)
    dim = 3
    k0 = 0.85 * torch.eye(dim)
    k1 = torch.tensor(
        [[0.15, 0.40, 0.00], [0.00, 0.25, 0.45], [0.10, 0.00, 0.55]],
        dtype=torch.float32,
    )

    def _pairs(mu: float, n_pairs: int) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.randn(n_pairs, dim)
        k_mu = _true_k(mu, k0, k1)
        return z, z @ k_mu.T

    z0, y0 = _pairs(0.0, 64)
    z1, y1 = _pairs(1.0, 8)
    z_h, y_h = _pairs(0.5, 32)
    report = leave_one_regime_out(
        [
            (torch.tensor([0.0]), z0, y0),
            (torch.tensor([1.0]), z1, y1),
            (torch.tensor([0.5]), z_h, y_h),
        ],
        hold_out=2,
        ridge=1e-6,
    )
    assert isinstance(report, LeaveOneRegimeOutReport)
    assert report.n_train_pairs == 72
    assert report.n_holdout_pairs == 32
    assert report.interpolant_mse < report.pooled_lti_mse
    # Noiseless affine interpolant at the midpoint of two anchors.
    assert report.interpolant_mse == pytest.approx(0.0, rel=0.0, abs=1e-5)


def test_factory_builds_parametric_kind() -> None:
    """``koopman='parametric'`` constructs the interpolant, not a refusal."""
    model = GraphKoopmanModel(
        GNNEncoder(2, 4, 2),
        GNNDecoder(2, 4, 2),
        latent_dim=2,
        time_step=0.1,
        koopman="parametric",
        koopman_parameter_dim=1,
        koopman_num_modes=2,
        koopman_weight_kind="rbf",
    )
    assert model.koopman_kind == "parametric"
    assert isinstance(model.koopman, ParametricKoopmanOperator)
    assert model.koopman.parameter_dim == 1
    with pytest.raises(ValueError, match="koopman_parameter_dim"):
        GraphKoopmanModel(
            GNNEncoder(2, 4, 2),
            GNNDecoder(2, 4, 2),
            latent_dim=2,
            time_step=0.1,
            koopman="pernode",
            koopman_parameter_dim=3,
        )
    with pytest.raises(ValueError, match="dynamics_mode='discrete'"):
        GraphKoopmanModel(
            GNNEncoder(2, 4, 2),
            GNNDecoder(2, 4, 2),
            latent_dim=2,
            time_step=0.1,
            koopman="parametric",
            dynamics_mode="continuous",
        )
    with pytest.raises(ValueError, match="stochastic"):
        GraphKoopmanModel(
            GNNEncoder(2, 4, 2),
            GNNDecoder(2, 4, 2),
            latent_dim=2,
            time_step=0.1,
            koopman="parametric",
            dynamics_mode="stochastic",
        )


def test_parametric_module_does_not_import_model_or_data() -> None:
    """L2 interpolant must not import L4 model or L1 data."""
    source = Path(__file__).resolve().parents[2] / (
        "src/koopman_graph/operators/parametric.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)
    forbidden = ("koopman_graph.model", "koopman_graph.data")
    offenders = [
        name
        for name in imported
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in forbidden)
    ]
    assert not offenders


def test_parametric_init_and_helper_guards() -> None:
    """Constructor and leave-one-out helpers reject invalid inputs."""
    with pytest.raises(ValueError, match="num_modes"):
        ParametricKoopmanOperator(2, num_modes=0)
    with pytest.raises(ValueError, match="parameter_dim"):
        ParametricKoopmanOperator(2, parameter_dim=0)
    with pytest.raises(ValueError, match="weight_kind"):
        ParametricKoopmanOperator(2, weight_kind="barycentric")  # type: ignore[arg-type]
    operator = ParametricKoopmanOperator(2)
    with pytest.raises(ValueError, match="anchors"):
        operator.set_anchors(torch.zeros(3, 1))
    with pytest.raises(ValueError, match="lengthscale"):
        operator.set_lengthscale(0.0)
    with pytest.raises(ValueError, match="at least 2 regimes"):
        leave_one_regime_out([], hold_out=0)
    z = torch.eye(2)
    with pytest.raises(ValueError, match="hold_out"):
        leave_one_regime_out(
            [
                (torch.tensor([0.0]), z, z),
                (torch.tensor([1.0]), z, z),
            ],
            hold_out=9,
        )
    with pytest.raises(ValueError, match="ridge"):
        leave_one_regime_out(
            [
                (torch.tensor([0.0]), z, z),
                (torch.tensor([1.0]), z, z),
            ],
            hold_out=0,
            ridge=-1.0,
        )
    with pytest.raises(ValueError, match="mu shape"):
        leave_one_regime_out(
            [
                (torch.tensor([0.0]), z, z),
                (torch.tensor([1.0, 0.0]), z, z),
            ],
            hold_out=0,
        )


def test_parametric_forward_uses_mu() -> None:
    """Homogeneous forward threads μ into the interpolant."""
    model = GraphKoopmanModel(
        GNNEncoder(2, 4, 2, num_layers=1),
        GNNDecoder(2, 4, 2, num_layers=1),
        latent_dim=2,
        time_step=0.1,
        koopman="parametric",
    )
    from torch_geometric.data import Data

    graph = Data(
        x=torch.randn(3, 2),
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
    )
    out = model(graph, parameters=torch.tensor([0.4]))
    assert out.shape == graph.x.shape
