"""Coverage and error-path tests for :mod:`koopman_graph.distributed`."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch
from torch import Tensor
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


def test_ray_train_with_stub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ray Train fit loop reports a checkpoint through local stand-ins."""
    import koopman_graph.distributed.ray_train as train_mod

    monkeypatch.setattr(train_mod, "_import_ray_train", lambda: None)
    model = _tiny_model()
    sequence = _tiny_sequence()
    with pytest.raises(ValueError, match="num_workers"):
        train_mod.run_ray_train_fit_loop(model, [sequence], num_workers=0)
    with pytest.raises(ValueError, match="val_sequences"):
        train_mod.run_ray_train_fit_loop(
            model,
            [sequence],
            early_stopping_monitor="val",
        )

    class _Checkpoint:
        def __init__(self, directory: str) -> None:
            self._directory = directory

        @classmethod
        def from_directory(cls, directory: str) -> _Checkpoint:
            dest = tmp_path / "ray_ckpt"
            dest.mkdir(exist_ok=True)
            src = Path(directory)
            for child in src.iterdir():
                (dest / child.name).write_bytes(child.read_bytes())
            return cls(str(dest))

        @contextmanager
        def as_directory(self) -> Any:
            yield self._directory

    captured: dict[str, Any] = {}

    def _report(metrics: dict[str, Any], checkpoint: Any = None) -> None:
        captured["metrics"] = metrics
        captured["checkpoint"] = checkpoint

    class _Trainer:
        def __init__(self, train_loop_per_worker: Any, scaling_config: Any) -> None:
            self.loop = train_loop_per_worker
            del scaling_config

        def fit(self) -> SimpleNamespace:
            self.loop()
            return SimpleNamespace(error=None, checkpoint=captured.get("checkpoint"))

    ray_train = ModuleType("ray.train")
    ray_train.ScalingConfig = lambda **_kwargs: SimpleNamespace()  # type: ignore[attr-defined]
    ray_train.Checkpoint = _Checkpoint  # type: ignore[attr-defined]
    ray_train.report = _report  # type: ignore[attr-defined]
    ray_train.get_context = lambda: SimpleNamespace(get_world_rank=lambda: 0)  # type: ignore[attr-defined]
    ray_torch = ModuleType("ray.train.torch")
    ray_torch.TorchTrainer = _Trainer  # type: ignore[attr-defined]
    ray_torch.get_device = lambda: torch.device("cpu")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ray.train", ray_train)
    monkeypatch.setitem(sys.modules, "ray.train.torch", ray_torch)
    monkeypatch.setattr(train_mod, "_worker_device", lambda: torch.device("cpu"))
    history = train_mod.run_ray_train_fit_loop(
        _tiny_model(),
        [_tiny_sequence()],
        epochs=1,
        lr=1e-2,
        seed=0,
        num_workers=1,
        use_gpu=False,
    )
    assert history.epochs == 1

    class _EmptyTrainer(_Trainer):
        def fit(self) -> SimpleNamespace:
            return SimpleNamespace(error=None, checkpoint=None)

    ray_torch.TorchTrainer = _EmptyTrainer  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError, match="without a result checkpoint"):
        train_mod.run_ray_train_fit_loop(_tiny_model(), [_tiny_sequence()], epochs=1)

    class _ErrTrainer(_Trainer):
        def fit(self) -> SimpleNamespace:
            return SimpleNamespace(error=RuntimeError("boom"), checkpoint=None)

    ray_torch.TorchTrainer = _ErrTrainer  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError, match="boom"):
        train_mod.run_ray_train_fit_loop(_tiny_model(), [_tiny_sequence()], epochs=1)
