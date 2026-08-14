"""Coverage and error-path tests for :mod:`koopman_graph.distributed`."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import Tensor, nn
from torch_geometric.data import Data

from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
from koopman_graph.data import (
    GraphSnapshotSequence,
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


def test_fabric_fit_with_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fabric fit runs one CPU epoch against a stand-in Fabric class."""
    import koopman_graph.distributed.fabric as fabric_mod

    class _FakeFabric:
        def __init__(self, **kwargs: object) -> None:
            self._precision = kwargs.get("precision", 32)
            self.device = "cpu"
            self.global_rank = 0
            self.world_size = 1
            self.is_global_zero = True

        def setup(self, *args: object) -> object:
            if len(args) == 2:
                return args[0], args[1]
            return args[0]

        def all_reduce(self, tensor: Tensor, reduce_op: str = "mean") -> Tensor:
            del reduce_op
            return tensor

        def broadcast(self, tensor: Tensor, src: int = 0) -> Tensor:
            del src
            return tensor

    original_import = fabric_mod._import_fabric_class
    monkeypatch.setattr(fabric_mod, "_import_fabric_class", lambda: _FakeFabric)
    with pytest.raises(ValueError, match="val_sequences"):
        fabric_mod.fit_with_fabric(
            _tiny_model(),
            [_tiny_sequence()],
            epochs=1,
            early_stopping_monitor="val",
        )
    history = fabric_mod.fit_with_fabric(
        _tiny_model(),
        [_tiny_sequence()],
        epochs=1,
        lr=1e-2,
        accelerator="cpu",
        devices=1,
        precision=32,
    )
    assert history.epochs == 1
    fabric = _FakeFabric()
    fabric.world_size = 2
    value = fabric_mod._fabric_all_reduce_mean(fabric, 1.5)
    assert value == pytest.approx(1.5)
    fabric_mod._fabric_broadcast_module_state(fabric, _tiny_model())
    assert fabric_mod._import_fabric_class is not None

    fake_module = SimpleNamespace(Fabric=_FakeFabric)
    real_import_module = importlib.import_module

    def _import(name: str, package: str | None = None) -> Any:
        if name == "lightning.fabric":
            return fake_module
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", _import)
    assert original_import() is _FakeFabric


def test_fabric_val_windows_and_non_tuple_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fabric fit covers validation sequences, window sampling, and setup."""
    import koopman_graph.distributed.fabric as fabric_mod

    class _FakeFabric:
        def __init__(self, **kwargs: object) -> None:
            del kwargs
            self._precision = 32
            self.device = "cpu"
            self.global_rank = 0
            self.world_size = 1
            self.is_global_zero = True

        def setup(self, *args: object) -> object:
            return args[0]

        def all_reduce(self, tensor: Tensor, reduce_op: str = "mean") -> Tensor:
            del reduce_op
            return tensor

        def broadcast(self, tensor: Tensor, src: int = 0) -> Tensor:
            del src
            return tensor

    monkeypatch.setattr(fabric_mod, "_import_fabric_class", lambda: _FakeFabric)
    sequence = _tiny_sequence(timesteps=4)
    history = fabric_mod.fit_with_fabric(
        _tiny_model(),
        [sequence],
        val_sequences=[sequence],
        window_length=2,
        epochs=1,
        lr=1e-2,
        accelerator="cpu",
        devices=1,
        precision=32,
    )
    assert history.epochs == 1
    buffered = nn.Linear(2, 2)
    buffered.register_buffer("scale", torch.ones(2))
    fabric = _FakeFabric()
    fabric.world_size = 2
    fabric_mod._fabric_broadcast_module_state(fabric, buffered)
