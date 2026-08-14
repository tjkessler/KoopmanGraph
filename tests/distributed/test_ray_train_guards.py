"""Coverage and error-path tests for :mod:`koopman_graph.distributed`."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch_geometric.data import Data

from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
from koopman_graph.data import GraphSnapshotSequence


def _tiny_sequence() -> GraphSnapshotSequence:
    edge = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    return GraphSnapshotSequence(
        [Data(x=torch.ones(2, 3) * (0.9**t), edge_index=edge) for t in range(6)]
    )


def _tiny_model(*, seed: int = 0) -> GraphKoopmanModel:
    torch.manual_seed(seed)
    return GraphKoopmanModel(
        encoder=GNNEncoder(3, 8, 4, num_layers=1),
        decoder=GNNDecoder(4, 8, 3, num_layers=1),
        latent_dim=4,
        time_step=0.1,
    )


def _patch_ray_trainer(fake_trainer: MagicMock):
    """Patch ``ray.train.torch.TorchTrainer`` while keeping other Ray imports."""
    from koopman_graph.distributed import ray_train as ray_train_mod

    real_import_module = ray_train_mod.importlib.import_module

    def _import_module(name: str, package: str | None = None) -> object:
        if name == "ray.train.torch":
            mod = MagicMock()
            mod.TorchTrainer = MagicMock(return_value=fake_trainer)
            return mod
        return real_import_module(name, package)

    return patch.object(
        ray_train_mod.importlib,
        "import_module",
        side_effect=_import_module,
    )


def test_ray_train_rejects_val_monitor_without_val_sequences() -> None:
    """early_stopping_monitor='val' requires val_sequences."""
    pytest.importorskip("ray.train")
    from koopman_graph.distributed.ray_train import run_ray_train_fit_loop

    with pytest.raises(ValueError, match="requires val_sequences"):
        run_ray_train_fit_loop(
            _tiny_model(),
            [_tiny_sequence()],
            epochs=1,
            num_workers=1,
            early_stopping_monitor="val",
        )


def test_ray_train_raises_when_checkpoint_missing() -> None:
    """Missing result.checkpoint surfaces a RuntimeError."""
    pytest.importorskip("ray.train")
    from koopman_graph.distributed.ray_train import run_ray_train_fit_loop

    fake_result = SimpleNamespace(error=None, checkpoint=None)
    fake_trainer = MagicMock()
    fake_trainer.fit.return_value = fake_result

    with (
        _patch_ray_trainer(fake_trainer),
        pytest.raises(RuntimeError, match="without a result checkpoint"),
    ):
        run_ray_train_fit_loop(
            _tiny_model(),
            [_tiny_sequence()],
            epochs=1,
            num_workers=1,
        )


def test_ray_train_propagates_trainer_error() -> None:
    """result.error is re-raised from the driver."""
    pytest.importorskip("ray.train")
    from koopman_graph.distributed.ray_train import run_ray_train_fit_loop

    boom = RuntimeError("boom")
    fake_result = SimpleNamespace(error=boom, checkpoint=MagicMock())
    fake_trainer = MagicMock()
    fake_trainer.fit.return_value = fake_result

    with (
        _patch_ray_trainer(fake_trainer),
        pytest.raises(RuntimeError, match="boom"),
    ):
        run_ray_train_fit_loop(
            _tiny_model(),
            [_tiny_sequence()],
            epochs=1,
            num_workers=1,
        )


def test_ray_train_window_sampler_path() -> None:
    """windows_per_epoch exercises the window-sampler branch in the worker."""
    pytest.importorskip("ray.train")
    from koopman_graph.distributed import run_ray_train_fit_loop

    history = run_ray_train_fit_loop(
        _tiny_model(),
        [_tiny_sequence()],
        epochs=1,
        num_workers=1,
        window_length=3,
        windows_per_epoch=2,
        batch_size=1,
        window_seed=0,
    )
    assert history.epochs == 1
    assert len(history.loss) == 1


def test_ray_train_worker_seed_val_and_non_main_report() -> None:
    """Invoke train_loop in-process: seed+rank, val_sequences, non-main report."""
    pytest.importorskip("ray.train")
    import tempfile
    from contextlib import ExitStack, contextmanager
    from pathlib import Path

    from koopman_graph.distributed import ray_train as ray_train_mod
    from koopman_graph.training.history import FitHistory

    fake_history = FitHistory(loss=(0.25,), epochs=1)
    captured: dict[str, object] = {}
    seed_calls: list[int] = []
    model = _tiny_model()
    train_seq = _tiny_sequence()
    val_seq = _tiny_sequence()

    def _torch_trainer_factory(*, train_loop_per_worker, scaling_config):
        captured["loop"] = train_loop_per_worker
        captured["scaling"] = scaling_config
        return captured["trainer"]

    real_import_module = ray_train_mod.importlib.import_module

    def _import_module(name: str, package: str | None = None) -> object:
        if name == "ray":
            return MagicMock(name="ray")
        if name == "ray.train":
            mod = MagicMock(name="ray.train")
            mod.ScalingConfig = MagicMock(return_value=MagicMock())
            mod.Checkpoint = MagicMock()
            mod.Checkpoint.from_directory = MagicMock(return_value=MagicMock())
            mod.report = MagicMock()
            ctx = MagicMock()
            ctx.get_world_rank.return_value = 1
            mod.get_context.return_value = ctx
            return mod
        if name == "ray.train.torch":
            mod = MagicMock(name="ray.train.torch")
            mod.TorchTrainer = MagicMock(side_effect=_torch_trainer_factory)
            mod.get_device = MagicMock(return_value="cpu")
            return mod
        return real_import_module(name, package)

    result_dir = tempfile.mkdtemp()
    torch.save(
        {
            "fit_history": fake_history,
            "state_dict": model.state_dict(),
        },
        Path(result_dir) / ray_train_mod._RESULT_FILENAME,
    )

    @contextmanager
    def _as_directory():
        yield result_dir

    fake_checkpoint = MagicMock()
    fake_checkpoint.as_directory = _as_directory
    fake_trainer = MagicMock()
    captured["trainer"] = fake_trainer

    def _run_loop(*, is_main: bool) -> None:
        loop = captured["loop"]
        assert callable(loop)
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(ray_train_mod, "init_process_group_from_env")
            )
            stack.enter_context(
                patch.object(
                    ray_train_mod,
                    "fit_epochs_distributed",
                    return_value=fake_history,
                )
            )
            stack.enter_context(
                patch.object(
                    ray_train_mod,
                    "prepare_ddp_model",
                    side_effect=lambda module, **_kwargs: module,
                )
            )
            stack.enter_context(
                patch.object(
                    ray_train_mod,
                    "seed_everything",
                    side_effect=lambda value: seed_calls.append(int(value)),
                )
            )
            stack.enter_context(
                patch.object(
                    ray_train_mod,
                    "shard_sequences_for_rank",
                    side_effect=lambda sequences: sequences,
                )
            )
            stack.enter_context(
                patch.object(
                    ray_train_mod,
                    "is_main_process",
                    return_value=is_main,
                )
            )
            loop()

    def _fit() -> SimpleNamespace:
        # Non-main report branch, then main-process checkpoint report.
        _run_loop(is_main=False)
        _run_loop(is_main=True)
        return SimpleNamespace(error=None, checkpoint=fake_checkpoint)

    fake_trainer.fit.side_effect = _fit

    with patch.object(
        ray_train_mod.importlib,
        "import_module",
        side_effect=_import_module,
    ):
        history = ray_train_mod.run_ray_train_fit_loop(
            model,
            [train_seq],
            epochs=1,
            num_workers=1,
            seed=7,
            val_sequences=[val_seq],
            window_length=3,
            windows_per_epoch=1,
            batch_size=1,
            window_seed=0,
        )
    assert history.epochs == 1
    assert history.loss == (0.25,)
    # seed + rank (world_rank mocked to 1) → seed_everything(8) twice
    assert seed_calls.count(8) >= 2

    # Shard branch (no window sampler) in a second in-process fit.
    with patch.object(
        ray_train_mod.importlib,
        "import_module",
        side_effect=_import_module,
    ):
        history2 = ray_train_mod.run_ray_train_fit_loop(
            _tiny_model(seed=1),
            [_tiny_sequence()],
            epochs=1,
            num_workers=1,
            seed=3,
            val_sequences=[_tiny_sequence()],
        )
    assert history2.epochs == 1
    assert seed_calls.count(4) >= 2


def test_ray_train_worker_device_resolves_ray_torch_device() -> None:
    """``_worker_device`` reads ``ray.train.torch.get_device``."""
    pytest.importorskip("ray.train")
    from koopman_graph.distributed import ray_train as ray_train_mod

    fake_torch = MagicMock()
    fake_torch.get_device.return_value = "cpu"
    with patch.object(
        ray_train_mod.importlib,
        "import_module",
        return_value=fake_torch,
    ):
        device = ray_train_mod._worker_device()
    assert device == torch.device("cpu")
    fake_torch.get_device.assert_called_once_with()
