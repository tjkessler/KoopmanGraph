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
from koopman_graph.distributed.ddp import run_ddp_fit_loop


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


def test_ddp_val_monitor_device_and_broadcast(monkeypatch: pytest.MonkeyPatch) -> None:
    """DDP helpers cover val-monitor, LOCAL_RANK CUDA, and broadcast loops."""
    import koopman_graph.distributed.ddp as ddp_mod

    model = _tiny_model()
    with pytest.raises(ValueError, match="val_sequences"):
        run_ddp_fit_loop(
            model,
            [_tiny_sequence()],
            early_stopping_monitor="val",
            epochs=1,
        )
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    device = ddp_mod._resolve_ddp_device(model, None)
    assert device.type == "cuda"
    monkeypatch.setattr(ddp_mod.dist, "is_available", lambda: True)
    monkeypatch.setattr(ddp_mod.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(ddp_mod, "get_world_size", lambda: 2)
    monkeypatch.setattr(ddp_mod.dist, "broadcast", lambda *_args, **_kwargs: None)
    ddp_mod._broadcast_module_state(model)
