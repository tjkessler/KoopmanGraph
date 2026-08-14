"""Coverage and error-path tests for :mod:`koopman_graph.model`."""

from __future__ import annotations

import pytest
import torch

from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
from koopman_graph.model.factory import parse_koopman_arg
from koopman_graph.operators import (
    HodgeKoopmanOperator,
    MixtureKoopmanOperator,
    SwitchedKoopmanOperator,
)


def _path_edges(num_nodes: int = 4) -> torch.Tensor:
    """Undirected path ``edge_index``."""
    src = list(range(num_nodes - 1)) + list(range(1, num_nodes))
    dst = list(range(1, num_nodes)) + list(range(num_nodes - 1))
    return torch.tensor([src, dst], dtype=torch.long)


def _tiny_model(*, koopman: str = "pernode", parameterization: str = "dense", **kwargs):
    """Build a tiny homogeneous GCN model."""
    return GraphKoopmanModel(
        encoder=GNNEncoder(2, 4, 4, num_layers=1),
        decoder=GNNDecoder(4, 4, 2, num_layers=1),
        latent_dim=4,
        time_step=0.1,
        koopman=koopman,
        koopman_parameterization=parameterization,
        **kwargs,
    )


def test_default_koopman_kind_remains_pernode() -> None:
    """Factory default is still per-node LTI."""
    kind, injected = parse_koopman_arg(None)
    assert kind == "pernode"
    assert injected is None
    model = _tiny_model()
    assert model.koopman_kind == "pernode"


def test_parse_koopman_arg_new_kinds() -> None:
    """String kinds include switched, mixture, and hodge."""
    for kind in ("switched", "mixture", "hodge"):
        resolved, injected = parse_koopman_arg(kind)
        assert resolved == kind
        assert injected is None
    with pytest.raises(ValueError, match="hodge"):
        parse_koopman_arg("not-a-kind")


def test_factory_switched_mixture_hodge() -> None:
    """Factory string kinds construct the new operator families."""
    switched = _tiny_model(koopman="switched")
    assert switched.koopman_kind == "switched"
    assert isinstance(switched.koopman, SwitchedKoopmanOperator)
    z = torch.randn(4, 4)
    assert switched.koopman.advance(z).shape == z.shape
    switched.koopman.set_mode(1)
    assert switched.koopman.mode_index == 1

    mixture = _tiny_model(koopman="mixture")
    assert mixture.koopman_kind == "mixture"
    assert isinstance(mixture.koopman, MixtureKoopmanOperator)
    assert mixture.koopman.advance(z).shape == z.shape

    hodge = _tiny_model(koopman="hodge")
    assert hodge.koopman_kind == "hodge"
    assert isinstance(hodge.koopman, HodgeKoopmanOperator)
    advanced = hodge.koopman.advance(z, edge_index=_path_edges(4))
    assert advanced.shape == z.shape
