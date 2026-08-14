"""Coverage and error-path tests for :mod:`koopman_graph.adaptation`."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from koopman_graph import (
    ContinuousKoopmanOperator,
)
from koopman_graph.adaptation import RecursiveKoopmanAdapter


def test_adapter_constructor_validation_and_defaults() -> None:
    with pytest.raises(ValueError, match="latent_dim must be positive"):
        RecursiveKoopmanAdapter(0)
    with pytest.raises(ValueError, match="control_dim must be non-negative"):
        RecursiveKoopmanAdapter(2, control_dim=-1)
    with pytest.raises(ValueError, match="forgetting_factor"):
        RecursiveKoopmanAdapter(2, forgetting_factor=0.0)
    with pytest.raises(ValueError, match="regularization must be positive"):
        RecursiveKoopmanAdapter(2, regularization=0.0)

    discrete = RecursiveKoopmanAdapter(3, mode="discrete")
    assert torch.allclose(discrete.discrete_matrix, torch.eye(3))

    continuous = RecursiveKoopmanAdapter(3, mode="continuous", control_dim=1)
    assert continuous.generator_matrix.shape == (3, 3)
    assert continuous.control_matrix is not None
    assert continuous.generator_control_matrix is not None

    uncontrolled = RecursiveKoopmanAdapter(2, mode="continuous")
    assert uncontrolled.generator_control_matrix is None


def test_adapter_from_operator_continuous_and_type_errors() -> None:
    continuous = ContinuousKoopmanOperator(3, control_dim=1, init_mode="identity")
    adapter = RecursiveKoopmanAdapter.from_operator(continuous, mode="continuous")
    assert adapter.mode == "continuous"
    assert adapter.control_dim == 1

    class FakeOp(nn.Module):
        parameterization = "dense"
        latent_dim = 2
        control_dim = 0

    fake = FakeOp()
    with pytest.raises(TypeError, match="built-in"):
        RecursiveKoopmanAdapter.from_operator(fake, mode="discrete")  # type: ignore[arg-type]

    controlled = RecursiveKoopmanAdapter(
        2,
        control_dim=1,
        mode="discrete",
        initial_b=torch.zeros(1, 2),
    )
    with pytest.raises(ValueError, match="initial_b is required"):
        controlled._set_from_discrete(torch.eye(2), None)


def test_adapter_update_and_apply_validation() -> None:
    adapter = RecursiveKoopmanAdapter(3, mode="continuous")
    z = torch.randn(3)
    with pytest.raises(ValueError, match="delta_t is required"):
        adapter.update(z, z)
    with pytest.raises(ValueError, match="matching shapes"):
        adapter.update(torch.randn(2, 3), torch.randn(3, 3), delta_t=0.1)
    with pytest.raises(ValueError, match="delta_t must be positive"):
        adapter.update(z, z, delta_t=0.0)

    adapter.update(z, z, delta_t=0.2)
    continuous = ContinuousKoopmanOperator(3, init_mode="identity")
    adapter.apply_to(continuous)

    class FakeOp(nn.Module):
        parameterization = "dense"
        latent_dim = 3
        control_dim = 0

    with pytest.raises(TypeError, match="built-in"):
        adapter.apply_to(FakeOp())  # type: ignore[arg-type]


def test_adapter_row_and_control_validation() -> None:
    adapter = RecursiveKoopmanAdapter(3, control_dim=0)
    with pytest.raises(ValueError, match="Expected latent vector"):
        adapter._as_rows(torch.randn(2))
    with pytest.raises(ValueError, match="Expected trailing latent dimension"):
        adapter._as_rows(torch.randn(2, 2))
    with pytest.raises(ValueError, match="latent tensors must have shape"):
        adapter._as_rows(torch.randn(2, 2, 3))
    with pytest.raises(ValueError, match="uncontrolled adapter"):
        adapter._resolve_control_row(torch.ones(1), batch_rows=1)

    controlled = RecursiveKoopmanAdapter(
        3,
        control_dim=2,
        initial_b=torch.zeros(2, 3),
    )
    with pytest.raises(ValueError, match="control input is required"):
        controlled._resolve_control_row(None, batch_rows=1)
    with pytest.raises(ValueError, match="Expected global control shape"):
        controlled._resolve_control_row(torch.ones(3), batch_rows=1)
    with pytest.raises(ValueError, match="Expected per-node control shape"):
        controlled._resolve_control_row(torch.ones(2, 3), batch_rows=2)
    with pytest.raises(ValueError, match="must match latent rows"):
        controlled._resolve_control_row(torch.ones(3, 2), batch_rows=2)
    with pytest.raises(ValueError, match="control input must have shape"):
        controlled._resolve_control_row(torch.ones(2, 2, 2), batch_rows=2)

    assert controlled._resolve_control_row(torch.ones(2), batch_rows=4).shape == (2,)
    assert controlled._resolve_control_row(torch.ones(4, 2), batch_rows=4).shape == (
        4,
        2,
    )
