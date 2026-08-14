"""Hetero Fabric / Lightning / Ray entry-point smoke and parity tests."""

from __future__ import annotations

import math

import pytest
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import HeteroData

from koopman_graph.data import HeteroGraphSnapshotSequence, SnapshotSequence
from koopman_graph.distributed import fit_with_fabric
from koopman_graph.distributed.lightning_module import _as_sequence_batch
from koopman_graph.model import GraphKoopmanModel
from koopman_graph.nn import RelGraphDecoder, RelGraphEncoder
from koopman_graph.training import run_fit_loop


def _multiplex_snapshot(*, seed: int = 0) -> HeteroData:
    generator = torch.Generator().manual_seed(seed)
    data = HeteroData()
    data["node"].x = torch.randn(4, 3, generator=generator)
    data["node", "r1", "node"].edge_index = torch.tensor(
        [[0, 1, 2], [1, 2, 0]],
        dtype=torch.long,
    )
    data["node", "r2", "node"].edge_index = torch.tensor(
        [[0, 2], [2, 3]],
        dtype=torch.long,
    )
    return data


def _hetero_sequence(
    *, num_timesteps: int = 5, seed: int = 0
) -> HeteroGraphSnapshotSequence:
    return HeteroGraphSnapshotSequence(
        [_multiplex_snapshot(seed=seed + t) for t in range(num_timesteps)]
    )


def _hetero_model(*, seed: int = 0) -> GraphKoopmanModel:
    torch.manual_seed(seed)
    return GraphKoopmanModel(
        encoder=RelGraphEncoder(
            3,
            hidden_channels=8,
            latent_dim=4,
            num_relations=2,
            num_layers=1,
        ),
        decoder=RelGraphDecoder(
            latent_dim=4,
            hidden_channels=8,
            out_channels=3,
            num_relations=2,
            num_layers=1,
        ),
        latent_dim=4,
        time_step=1.0,
        koopman="hetero_graph",
    )


class _SingleSequenceDataset(Dataset):
    """Length-1 dataset yielding one trajectory per epoch step."""

    def __init__(self, sequence: SnapshotSequence) -> None:
        self._sequence = sequence

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> SnapshotSequence:
        del index
        return self._sequence


def _collate_sequences(batch: list[SnapshotSequence]) -> list[SnapshotSequence]:
    """Identity collate keeping a list of snapshot sequences."""
    return batch


def _make_hetero_member_factory():
    """Return a nested factory Ray can cloudpickle without the test module."""

    def member_factory() -> GraphKoopmanModel:
        return GraphKoopmanModel(
            encoder=RelGraphEncoder(
                3,
                hidden_channels=8,
                latent_dim=4,
                num_relations=2,
                num_layers=1,
            ),
            decoder=RelGraphDecoder(
                latent_dim=4,
                hidden_channels=8,
                out_channels=3,
                num_relations=2,
                num_layers=1,
            ),
            latent_dim=4,
            time_step=1.0,
            koopman="hetero_graph",
        )

    return member_factory


def test_as_sequence_batch_accepts_hetero_sequence() -> None:
    """Lightning batch coercion accepts multiplex trajectories."""
    sequence = _hetero_sequence(seed=0)
    assert _as_sequence_batch(sequence) == [sequence]
    assert _as_sequence_batch([sequence]) == [sequence]


def test_fit_with_fabric_hetero_cpu_one_epoch_finite_loss() -> None:
    """One-epoch CPU Fabric smoke on multiplex sequences."""
    pytest.importorskip("lightning")
    history = fit_with_fabric(
        _hetero_model(seed=0),
        [_hetero_sequence(seed=0)],
        accelerator="cpu",
        devices=1,
        precision=32,
        strategy="auto",
        epochs=1,
        lr=1e-2,
    )
    assert history.epochs == 1
    assert torch.isfinite(torch.tensor(history.loss[0]))


def test_fit_with_fabric_hetero_matches_run_fit_loop() -> None:
    """World-size-1 Fabric multiplex loss matches single-process fit."""
    pytest.importorskip("lightning")
    sequence = _hetero_sequence(seed=3)
    model_a = _hetero_model(seed=7)
    model_b = _hetero_model(seed=7)
    kwargs = {"epochs": 3, "lr": 1e-2}
    history_a = run_fit_loop(model_a, [sequence], device="cpu", **kwargs)
    history_b = fit_with_fabric(
        model_b,
        [sequence],
        accelerator="cpu",
        devices=1,
        precision=32,
        strategy="auto",
        **kwargs,
    )
    assert history_a.epochs == history_b.epochs
    for left, right in zip(history_a.loss, history_b.loss, strict=True):
        assert left == pytest.approx(right, rel=0, abs=1e-5)


def test_koopman_lightning_module_hetero_parity_vs_run_fit_loop() -> None:
    """Seeded CPU Trainer steps match ``run_fit_loop`` weights on multiplex."""
    pytest.importorskip("lightning")
    from lightning import Trainer

    from koopman_graph.distributed import KoopmanLightningModule

    sequence = _hetero_sequence(seed=2)
    epochs = 3
    lr = 1e-2

    model_a = _hetero_model(seed=7)
    run_fit_loop(model_a, [sequence], epochs=epochs, lr=lr, device="cpu")

    model_b = _hetero_model(seed=7)
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


@pytest.mark.ray
def test_fit_ensemble_with_ray_hetero_two_members() -> None:
    """Ray ensemble members accept multiplex inputs once single-process fit does."""
    ray = pytest.importorskip("ray")
    from koopman_graph.distributed import fit_ensemble_with_ray
    from koopman_graph.distributed.ray_jobs import _ray_init_kwargs

    if not ray.is_initialized():
        ray.init(**_ray_init_kwargs(num_cpus=2))

    factory = _make_hetero_member_factory()
    state_dicts, histories = fit_ensemble_with_ray(
        factory,
        _hetero_sequence(seed=1),
        num_members=2,
        seeds=(0, 1),
        epochs=2,
        lr=1e-2,
    )
    assert len(state_dicts) == 2
    assert len(histories) == 2
    for history in histories:
        assert history.reconstruction_loss
        assert all(math.isfinite(value) for value in history.reconstruction_loss)
