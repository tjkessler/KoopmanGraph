"""Tests for Lightning Fabric fit adapter.

Policy / import tests always run. The CPU one-epoch smoke is gated with
``pytest.importorskip("lightning")`` so default CI stays green without the
``[lightning]`` extra.
"""

from __future__ import annotations

import ast

import pytest
import torch
from tests.helpers import REPO_ROOT
from torch_geometric.data import Data

from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
from koopman_graph import distributed as distributed_pkg
from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.distributed import fit_with_fabric


def _tiny_decay_sequence() -> GraphSnapshotSequence:
    """Build a short deterministic decay trajectory for Fabric smoke tests."""
    edge = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    return GraphSnapshotSequence(
        [Data(x=torch.ones(2, 3) * (0.9**t), edge_index=edge) for t in range(4)]
    )


def _tiny_model(*, seed: int = 0) -> GraphKoopmanModel:
    """Build a small seeded trainable model."""
    torch.manual_seed(seed)
    return GraphKoopmanModel(
        encoder=GNNEncoder(3, 8, 4),
        decoder=GNNDecoder(4, 8, 3),
        latent_dim=4,
        time_step=0.1,
    )


def test_fit_with_fabric_exported() -> None:
    """``fit_with_fabric`` is part of the public distributed API."""
    assert "fit_with_fabric" in distributed_pkg.__all__
    assert hasattr(distributed_pkg, "fit_with_fabric")


def test_fit_with_fabric_missing_lightning_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing Lightning raises an actionable install hint."""
    import importlib

    import koopman_graph.distributed.fabric as fabric_mod

    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None) -> object:
        if name == "lightning.fabric" or name.startswith("lightning."):
            raise ImportError("simulated missing lightning")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    with pytest.raises(ImportError, match="koopman-graph\\[lightning\\]"):
        fabric_mod._import_fabric_class()


def test_fit_with_fabric_rejects_stacked_autocast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-FP32 Fabric precision cannot be combined with ``use_amp``."""
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

        def all_reduce(self, tensor: object, reduce_op: str = "mean") -> object:
            return tensor

        def broadcast(self, tensor: object, src: int = 0) -> object:
            return tensor

    monkeypatch.setattr(fabric_mod, "_import_fabric_class", lambda: _FakeFabric)

    with pytest.raises(ValueError, match="stacked autocast"):
        fabric_mod.fit_with_fabric(
            _tiny_model(),
            [_tiny_decay_sequence()],
            accelerator="cpu",
            devices=1,
            precision="16-mixed",
            use_amp=True,
            epochs=1,
        )


def test_training_package_has_no_lightning_imports() -> None:
    """``koopman_graph.training`` must not import Lightning."""
    training_root = REPO_ROOT / "src" / "koopman_graph" / "training"
    for path in training_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                names = [node.module]
            else:
                continue
            for name in names:
                top = name.split(".", maxsplit=1)[0]
                assert top not in {"lightning", "pytorch_lightning"}, path


def test_fit_with_fabric_cpu_one_epoch_finite_loss() -> None:
    """One-epoch CPU Fabric smoke; skips when Lightning is not installed."""
    pytest.importorskip("lightning")
    model = _tiny_model(seed=0)
    history = fit_with_fabric(
        model,
        [_tiny_decay_sequence()],
        accelerator="cpu",
        devices=1,
        precision=32,
        strategy="auto",
        epochs=1,
        lr=1e-2,
    )
    assert history.epochs == 1
    assert torch.isfinite(torch.tensor(history.loss[0]))
