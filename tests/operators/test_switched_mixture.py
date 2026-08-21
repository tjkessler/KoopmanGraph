"""Coverage and error-path tests for :mod:`koopman_graph.operators`."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data

from koopman_graph.graph_utils import propagate_latent
from koopman_graph.model import GraphKoopmanModel
from koopman_graph.nn import GNNDecoder, GNNEncoder
from koopman_graph.operators import (
    MixtureKoopmanOperator,
    SwitchedKoopmanOperator,
)


def test_switched_and_mixture_operator_surface() -> None:
    """Switched/mixture banks cover mode errors, matrix views, inverse, forward."""
    with pytest.raises(ValueError, match="num_modes"):
        SwitchedKoopmanOperator(3, num_modes=0)
    with pytest.raises(ValueError, match="num_modes"):
        MixtureKoopmanOperator(3, num_modes=0)
    switched = SwitchedKoopmanOperator(3, num_modes=2)
    with pytest.raises(ValueError, match="mode_index"):
        switched.set_mode(9)
    z = torch.randn(4, 3)
    assert 0 <= switched.infer_mode(z) < 2
    assert switched.matrix.shape == (3, 3)
    assert switched.bound_metric().ndim == 0
    _ = switched.stability_certificate()
    assert switched.inverse_advance(z).shape == z.shape
    assert switched(z).shape == z.shape
    mixture = MixtureKoopmanOperator(3, num_modes=2)
    assert mixture.matrix.shape == (3, 3)
    assert mixture.bound_metric().ndim == 0
    assert mixture.inverse_advance(z).shape == z.shape
    assert mixture(z).shape == z.shape


def _two_mode_switched() -> tuple[SwitchedKoopmanOperator, torch.Tensor]:
    """Return a two-mode bank with distinct maps and a latent batch."""
    switched = SwitchedKoopmanOperator(2, num_modes=2)
    switched.modes[0].set_dense_matrix(0.5 * torch.eye(2))
    switched.modes[1].set_dense_matrix(torch.diag(torch.tensor([0.2, 0.8])))
    switched.set_mode(0)
    return switched, torch.ones(3, 2)


def test_switched_phase_index_overrides_without_mutating_mode() -> None:
    """phase_index selects a mode for one step and leaves mode_index."""
    switched, z = _two_mode_switched()
    expected = switched.modes[1].advance(z)
    out = switched.advance(z, phase_index=1)
    torch.testing.assert_close(out, expected)
    assert switched.mode_index == 0
    inverse = switched.inverse_advance(expected, phase_index=1)
    torch.testing.assert_close(inverse, z, atol=1e-5, rtol=1e-5)
    assert switched.mode_index == 0
    with pytest.raises(ValueError, match="phase_index"):
        switched.advance(z, phase_index=2)
    forwarded = propagate_latent(switched, z, phase_index=1)
    torch.testing.assert_close(forwarded, expected)
    ignored = propagate_latent(MixtureKoopmanOperator(2, num_modes=2), z, phase_index=1)
    assert ignored.shape == z.shape


def test_model_forward_threads_phase_index() -> None:
    """Homogeneous forward uses phase_index without latching the mode."""
    model = GraphKoopmanModel(
        GNNEncoder(2, 4, 2, num_layers=1),
        GNNDecoder(2, 4, 2, num_layers=1),
        latent_dim=2,
        time_step=0.1,
        koopman="switched",
        koopman_num_modes=2,
    )
    operator = model.koopman
    assert isinstance(operator, SwitchedKoopmanOperator)
    operator.modes[0].set_dense_matrix(0.5 * torch.eye(2))
    operator.modes[1].set_dense_matrix(torch.diag(torch.tensor([0.25, 0.75])))
    operator.set_mode(0)
    graph = Data(
        x=torch.randn(3, 2),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
    )
    out_latched = model(graph)
    out_phase = model(graph, phase_index=1)
    assert operator.mode_index == 0
    assert out_latched.shape == graph.x.shape
    assert out_phase.shape == graph.x.shape
    assert not torch.allclose(out_latched, out_phase)
