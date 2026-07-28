"""Tests for optional Lightning Trainer façade (Tier 2)."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data

from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
from koopman_graph import distributed as distributed_pkg
from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.training import run_fit_loop


class _SingleSequenceDataset(Dataset):
    """Length-1 dataset yielding one trajectory per epoch step."""

    def __init__(self, sequence: GraphSnapshotSequence) -> None:
        self._sequence = sequence

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> GraphSnapshotSequence:
        del index
        return self._sequence


def _collate_sequences(
    batch: list[GraphSnapshotSequence],
) -> list[GraphSnapshotSequence]:
    """Identity collate keeping a list of snapshot sequences."""
    return batch


def _tiny_sequence() -> GraphSnapshotSequence:
    """Deterministic decay trajectory for parity smoke tests."""
    edge = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    return GraphSnapshotSequence(
        [Data(x=torch.ones(2, 3) * (0.9**t), edge_index=edge) for t in range(4)]
    )


def _tiny_model(*, seed: int = 0) -> GraphKoopmanModel:
    """Build a small identically seeded trainable model."""
    torch.manual_seed(seed)
    return GraphKoopmanModel(
        encoder=GNNEncoder(3, 8, 4),
        decoder=GNNDecoder(4, 8, 3),
        latent_dim=4,
        time_step=0.1,
    )


def test_koopman_lightning_module_exported() -> None:
    """``KoopmanLightningModule`` is listed in the public distributed API."""
    assert "KoopmanLightningModule" in distributed_pkg.__all__


def test_koopman_lightning_module_missing_lightning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing Lightning raises an actionable install hint."""
    import koopman_graph.distributed.lightning_module as mod

    monkeypatch.setattr(mod, "_MODULE_CLS", None)
    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None) -> object:
        if name == "lightning" or name.startswith("lightning."):
            raise ImportError("simulated missing lightning")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    with pytest.raises(ImportError, match="koopman-graph\\[lightning\\]"):
        mod._import_lightning()


def test_trainer_weight_parity_vs_run_fit_loop() -> None:
    """Seeded CPU Trainer steps match ``run_fit_loop`` weights (approx)."""
    pytest.importorskip("lightning")
    from lightning import Trainer

    from koopman_graph.distributed import KoopmanLightningModule

    sequence = _tiny_sequence()
    epochs = 3
    lr = 1e-2

    model_a = _tiny_model(seed=7)
    run_fit_loop(model_a, [sequence], epochs=epochs, lr=lr, device="cpu")

    model_b = _tiny_model(seed=7)
    module = KoopmanLightningModule(model_b, lr=lr)
    loader = DataLoader(
        _SingleSequenceDataset(sequence),
        batch_size=1,
        collate_fn=_collate_sequences,
    )
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=epochs,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(module, train_dataloaders=loader)

    for (name_a, param_a), (name_b, param_b) in zip(
        model_a.named_parameters(),
        model_b.named_parameters(),
        strict=True,
    ):
        assert name_a == name_b
        assert torch.allclose(param_a, param_b, rtol=0, atol=1e-5), name_a


def test_export_format1_checkpoint_no_module_prefix(tmp_path: Path) -> None:
    """Format-1 export writes unwrapped keys (no Lightning ``module.`` prefix)."""
    pytest.importorskip("lightning")
    from koopman_graph.distributed import KoopmanLightningModule

    model = _tiny_model(seed=3)
    module = KoopmanLightningModule(model, lr=1e-2)
    path = tmp_path / "lit_format1.pt"
    module.export_format1_checkpoint(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert isinstance(payload, dict)
    assert "state_dict" in payload
    assert not any(key.startswith("module.") for key in payload["state_dict"])
