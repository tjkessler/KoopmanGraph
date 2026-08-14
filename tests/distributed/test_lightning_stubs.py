"""Coverage and error-path tests for :mod:`koopman_graph.distributed`."""

from __future__ import annotations

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


def test_lightning_module_with_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lightning façade steps run against a local ``LightningModule`` stand-in."""
    import koopman_graph.distributed.lightning_module as lit_mod

    class _FakeLightningModule(nn.Module):
        def log(self, *_args: object, **_kwargs: object) -> None:
            return None

    fake_pkg = SimpleNamespace(LightningModule=_FakeLightningModule)
    monkeypatch.setattr(lit_mod, "_MODULE_CLS", None)
    monkeypatch.setattr(lit_mod, "_import_lightning", lambda: fake_pkg)
    with pytest.raises(ValueError, match="at least one"):
        lit_mod._as_sequence_batch([])
    with pytest.raises(TypeError, match="must be SnapshotSequence"):
        lit_mod._as_sequence_batch([1])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="or list thereof"):
        lit_mod._as_sequence_batch("nope")  # type: ignore[arg-type]
    sequence = _tiny_sequence()
    assert len(lit_mod._as_sequence_batch(sequence)) == 1
    cls = lit_mod.KoopmanLightningModule
    module = cls(_tiny_model(), lr=1e-2)
    loss = module.training_step(sequence, 0)
    assert loss.ndim == 0
    val = module.validation_step([sequence], 0)
    assert val.ndim == 0
    optim = module.configure_optimizers()
    assert isinstance(optim, torch.optim.Optimizer)
    scheduler_module = cls(
        _tiny_model(),
        lr=1e-2,
        lr_scheduler=lambda opt: torch.optim.lr_scheduler.StepLR(opt, step_size=1),
    )
    mapping = scheduler_module.configure_optimizers()
    assert "optimizer" in mapping
    with pytest.raises(TypeError, match="LRScheduler"):
        cls(
            _tiny_model(),
            lr_scheduler=lambda _opt: "nope",  # type: ignore[arg-type,return-value]
        ).configure_optimizers()
    assert "KoopmanLightningModule" in lit_mod.__dir__()
    with pytest.raises(AttributeError):
        _ = lit_mod.missing_symbol
