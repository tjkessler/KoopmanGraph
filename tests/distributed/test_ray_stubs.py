"""Coverage and error-path tests for :mod:`koopman_graph.distributed`."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import Tensor
from torch_geometric.data import Data

from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
from koopman_graph.data import (
    GraphSnapshotSequence,
)
from koopman_graph.distributed import (
    fit_ensemble_with_ray,
)


def _path_edges(num_nodes: int = 4) -> Tensor:
    """Undirected path ``edge_index``."""
    src = list(range(num_nodes - 1)) + list(range(1, num_nodes))
    dst = list(range(1, num_nodes)) + list(range(num_nodes - 1))
    return torch.tensor([src, dst], dtype=torch.long)


def _tiny_model(**kwargs: Any) -> GraphKoopmanModel:
    """Build a tiny homogeneous GCN model."""
    return GraphKoopmanModel(
        encoder=GNNEncoder(2, 4, 4, num_layers=1),
        decoder=GNNDecoder(4, 4, 2, num_layers=1),
        latent_dim=4,
        time_step=0.1,
        **kwargs,
    )


def _tiny_sequence(*, num_nodes: int = 4, timesteps: int = 4) -> GraphSnapshotSequence:
    """Path-graph snapshot sequence."""
    edge_index = _path_edges(num_nodes)
    snapshots = [
        Data(x=torch.randn(num_nodes, 2), edge_index=edge_index)
        for _ in range(timesteps)
    ]
    return GraphSnapshotSequence(snapshots)


def test_ray_jobs_with_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensemble Ray jobs run member ``fit`` through a local Ray stand-in."""
    import koopman_graph.distributed.ray_jobs as ray_mod

    class _StubRay:
        def __init__(self) -> None:
            self.initialized = True

        def is_initialized(self) -> bool:
            return self.initialized

        def init(self, **_kwargs: Any) -> None:
            self.initialized = True

        def put(self, obj: Any) -> Any:
            return obj

        def remote(self, function: Any) -> Any:
            class _Remote:
                @staticmethod
                def remote(*args: Any, **kwargs: Any) -> Any:
                    return function(*args, **kwargs)

            return _Remote()

        def get(self, futures: list[Any]) -> list[Any]:
            return list(futures)

    stub = _StubRay()
    monkeypatch.setattr(ray_mod, "_import_ray", lambda: stub)
    sequence = _tiny_sequence(timesteps=4)
    states, histories = fit_ensemble_with_ray(
        _tiny_model,
        sequence,
        num_members=1,
        seeds=[0],
        epochs=1,
        lr=1e-2,
    )
    assert len(states) == 1
    assert histories[0].epochs == 1
    stub.initialized = False
    kwargs = ray_mod._ray_init_kwargs(ray_address="auto", num_cpus=1)
    assert kwargs["address"] == "auto"
    fit_ensemble_with_ray(_tiny_model, sequence, num_members=1, epochs=1)
    with pytest.raises(ValueError, match="num_members"):
        fit_ensemble_with_ray(_tiny_model, sequence, num_members=0)
    with pytest.raises(ValueError, match="length num_members"):
        fit_ensemble_with_ray(_tiny_model, sequence, num_members=2, seeds=[0])
    with pytest.raises(TypeError, match="does not accept"):
        fit_ensemble_with_ray(
            _tiny_model,
            sequence,
            num_members=1,
            parallel_backend="ray",
        )
