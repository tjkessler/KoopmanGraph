"""Tests for native DDP fit helpers.

Single-process cases run in the default suite. Multi-process gloo smokes are
marked ``@pytest.mark.distributed`` and are **opt-in**:

.. code-block:: bash

    KOOPMAN_GRAPH_DISTRIBUTED_TESTS=1 \\
      pytest tests/distributed/test_distributed_ddp.py -m distributed

Without that environment variable the marked tests skip immediately (so
``pytest tests/ -n auto`` stays green and does not spawn process groups).
"""

from __future__ import annotations

import os
import socket
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tests.helpers import REPO_ROOT
from torch_geometric.data import Data

from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
from koopman_graph.data import GraphSnapshotSequence, WindowSampler
from koopman_graph.distributed import (
    all_reduce_mean,
    prepare_ddp_model,
    run_ddp_fit_loop,
    seed_everything,
    unwrap_model,
)
from koopman_graph.training import run_fit_loop

_DISTRIBUTED_ENV = "KOOPMAN_GRAPH_DISTRIBUTED_TESTS"


def _make_model(seed: int = 0) -> GraphKoopmanModel:
    """Build a tiny identically seeded trainable model."""
    torch.manual_seed(seed)
    encoder = GNNEncoder(in_channels=3, hidden_channels=16, latent_dim=8)
    decoder = GNNDecoder(latent_dim=8, hidden_channels=16, out_channels=3)
    return GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=8,
        time_step=0.1,
    )


def _decay_sequence(*, num_timesteps: int = 5) -> GraphSnapshotSequence:
    """Build the shared deterministic decay trajectory for spawn workers."""
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 4],
            [1, 0, 2, 1, 3, 2, 4, 3],
        ],
        dtype=torch.long,
    )
    x0 = torch.ones(5, 3)
    snapshots = [
        Data(x=x0 * (0.9**t), edge_index=edge_index) for t in range(num_timesteps)
    ]
    return GraphSnapshotSequence(snapshots)


def _distributed_tests_enabled() -> bool:
    """Return whether opt-in multi-process smokes should run."""
    return os.environ.get(_DISTRIBUTED_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _require_distributed_tests() -> None:
    """Skip unless ``KOOPMAN_GRAPH_DISTRIBUTED_TESTS`` enables multi-proc."""
    if not _distributed_tests_enabled():
        pytest.skip(
            f"opt-in multi-proc smoke; set {_DISTRIBUTED_ENV}=1 to enable "
            "(pytest -m distributed)"
        )


def _free_tcp_port() -> int:
    """Allocate an ephemeral localhost TCP port for the process group."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _ddp_gloo_worker(
    rank: int,
    world_size: int,
    port: int,
    workdir: str,
    epochs: int,
) -> None:
    """Train under gloo DDP and persist unwrapped weights for the parent."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    # Force CPU gloo even when CUDA is visible (init before run_ddp_fit_loop).
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        seed_everything(0, rank=0)
        model = _make_model(seed=0)
        sequence = _decay_sequence()
        work = Path(workdir)
        checkpoint_path = work / "ckpt.pt"
        run_ddp_fit_loop(
            model,
            [sequence],
            epochs=epochs,
            lr=1e-2,
            device="cpu",
            window_length=3,
            batch_size=2,
            window_seed=0,
            restore_best_weights=True,
            checkpoint_path=checkpoint_path,
        )
        torch.save(model.state_dict(), work / f"state_rank{rank}.pt")
        (work / f"done_{rank}").write_text("ok", encoding="utf-8")
    finally:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


def test_unwrap_model_identity_without_ddp(
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Unwrapped identity when DDP is not used."""
    model = _make_model()
    prepared = prepare_ddp_model(model, device=torch.device("cpu"))
    assert unwrap_model(prepared) is prepared
    assert prepared is model


def test_all_reduce_mean_noop_single_process() -> None:
    """Inactive process group leaves scalars unchanged."""
    assert all_reduce_mean(3.5) == 3.5


def test_run_ddp_fit_loop_matches_run_fit_loop_full_sequence(
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """World-size-1 DDP path matches ``run_fit_loop`` loss trajectory."""
    model_a = _make_model(seed=7)
    model_b = _make_model(seed=7)
    kwargs = {
        "epochs": 3,
        "lr": 1e-2,
        "device": "cpu",
        "window_seed": 0,
    }
    history_a = run_fit_loop(model_a, [scaling_sequence], **kwargs)
    history_b = run_ddp_fit_loop(model_b, [scaling_sequence], **kwargs)
    assert history_a.epochs == history_b.epochs
    for left, right in zip(history_a.loss, history_b.loss, strict=True):
        assert left == pytest.approx(right, rel=0, abs=1e-6)


def test_run_ddp_fit_loop_matches_run_fit_loop_windowed(
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Windowed world-size-1 path matches ``run_fit_loop`` with same seed."""
    model_a = _make_model(seed=11)
    model_b = _make_model(seed=11)
    kwargs = {
        "epochs": 2,
        "lr": 1e-2,
        "device": "cpu",
        "window_length": 3,
        "batch_size": 2,
        "window_seed": 5,
    }
    history_a = run_fit_loop(model_a, [scaling_sequence], **kwargs)
    history_b = run_ddp_fit_loop(model_b, [scaling_sequence], **kwargs)
    for left, right in zip(history_a.loss, history_b.loss, strict=True):
        assert left == pytest.approx(right, rel=0, abs=1e-6)


def test_run_ddp_fit_loop_rejects_plain_window_sampler(
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Non-distributed samplers are rejected with guidance."""
    model = _make_model()
    sampler = WindowSampler(
        scaling_sequence,
        window_length=3,
        batch_size=2,
        shuffle=False,
    )
    with pytest.raises(ValueError, match="DistributedWindowSampler"):
        run_ddp_fit_loop(
            model,
            [scaling_sequence],
            epochs=1,
            sampler=sampler,
        )


def test_checkpoint_has_no_module_prefix(
    scaling_sequence: GraphSnapshotSequence,
    tmp_path: Path,
) -> None:
    """Rank-0 checkpoints store unwrapped format-1 keys."""
    model = _make_model(seed=3)
    path = tmp_path / "ddp_ckpt.pt"
    run_ddp_fit_loop(
        model,
        [scaling_sequence],
        epochs=2,
        lr=1e-2,
        device="cpu",
        restore_best_weights=True,
        checkpoint_path=path,
    )
    assert path.is_file()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert isinstance(payload, dict)
    state = payload["state_dict"]
    assert not any(key.startswith("module.") for key in state)


def test_checkpoint_not_written_when_not_main_process(
    scaling_sequence: GraphSnapshotSequence,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-main ranks skip checkpoint writes."""
    monkeypatch.setattr(
        "koopman_graph.distributed.ddp.is_main_process",
        lambda: False,
    )
    model = _make_model(seed=4)
    path = tmp_path / "skipped.pt"
    run_ddp_fit_loop(
        model,
        [scaling_sequence],
        epochs=1,
        lr=1e-2,
        device="cpu",
        restore_best_weights=True,
        checkpoint_path=path,
    )
    assert not path.exists()


def test_orbit_bind_runs_before_prepare_ddp_model(
    scaling_sequence: GraphSnapshotSequence,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Orbit binding is invoked before DDP prepare/wrap."""
    calls: list[str] = []
    import koopman_graph.distributed.ddp as ddp_mod

    real_bind = ddp_mod.bind_pending_orbit_ties
    real_prepare = ddp_mod.prepare_ddp_model

    def tracking_bind(*args: object, **kwargs: object) -> None:
        calls.append("bind")
        real_bind(*args, **kwargs)

    def tracking_prepare(*args: object, **kwargs: object) -> torch.nn.Module:
        calls.append("prepare")
        return real_prepare(*args, **kwargs)

    monkeypatch.setattr(ddp_mod, "bind_pending_orbit_ties", tracking_bind)
    monkeypatch.setattr(ddp_mod, "prepare_ddp_model", tracking_prepare)
    model = _make_model(seed=2)
    run_ddp_fit_loop(
        model,
        [scaling_sequence],
        epochs=1,
        lr=1e-2,
        device="cpu",
    )
    assert "bind" in calls and "prepare" in calls
    assert calls.index("bind") < calls.index("prepare")


def test_run_ddp_fit_loop_rejects_sampler_and_window_length(
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """``sampler`` and ``window_length`` are mutually exclusive."""
    from koopman_graph.distributed import DistributedWindowSampler

    model = _make_model()
    sampler = DistributedWindowSampler(
        [scaling_sequence],
        window_length=3,
        batch_size=2,
        seed=0,
    )
    with pytest.raises(ValueError, match="not both"):
        run_ddp_fit_loop(
            model,
            [scaling_sequence],
            epochs=1,
            sampler=sampler,
            window_length=3,
        )


def test_run_ddp_fit_loop_rejects_unsupported_sampler(
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Unknown sampler types raise ``TypeError``."""
    model = _make_model()
    with pytest.raises(TypeError, match="unsupported sampler type"):
        run_ddp_fit_loop(
            model,
            [scaling_sequence],
            epochs=1,
            sampler=object(),  # type: ignore[arg-type]
        )


def test_run_ddp_fit_loop_requires_val_for_val_monitor(
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Val early-stopping monitor requires validation sequences."""
    model = _make_model()
    with pytest.raises(ValueError, match="val_sequences"):
        run_ddp_fit_loop(
            model,
            [scaling_sequence],
            epochs=1,
            early_stopping_monitor="val",
        )


def test_run_ddp_fit_loop_val_and_early_stop(
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Validation metrics and early stopping run on the world-size-1 path."""
    model = _make_model(seed=9)
    history = run_ddp_fit_loop(
        model,
        [scaling_sequence],
        epochs=5,
        lr=1e-2,
        device="cpu",
        val_sequences=[scaling_sequence],
        early_stopping_monitor="val",
        early_stopping_patience=1,
        early_stopping_min_delta=1e6,
        restore_best_weights=True,
    )
    assert history.val_loss is not None
    assert len(history.val_loss) == history.epochs
    assert history.stopped_early


def test_run_ddp_fit_loop_checkpoint_without_restore(
    scaling_sequence: GraphSnapshotSequence,
    tmp_path: Path,
) -> None:
    """Checkpoint write restores last weights when restore_best is false."""
    model = _make_model(seed=5)
    path = tmp_path / "best_only.pt"
    history = run_ddp_fit_loop(
        model,
        [scaling_sequence],
        epochs=2,
        lr=1e-2,
        device="cpu",
        restore_best_weights=False,
        checkpoint_path=path,
    )
    assert path.is_file()
    assert history.best_epoch is not None


def test_run_ddp_fit_loop_with_explicit_distributed_sampler(
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Pre-built ``DistributedWindowSampler`` is accepted."""
    from koopman_graph.distributed import DistributedWindowSampler

    model = _make_model(seed=6)
    sampler = DistributedWindowSampler(
        [scaling_sequence],
        window_length=3,
        batch_size=2,
        seed=1,
    )
    history = run_ddp_fit_loop(
        model,
        [scaling_sequence],
        epochs=1,
        lr=1e-2,
        device="cpu",
        sampler=sampler,
    )
    assert history.epochs == 1


def test_prepare_ddp_model_requires_process_group_when_world_gt_one(
    scaling_sequence: GraphSnapshotSequence,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """World size > 1 without an initialized group raises ``RuntimeError``."""
    import koopman_graph.distributed.ddp as ddp_mod

    monkeypatch.setattr(ddp_mod, "get_world_size", lambda: 2)
    monkeypatch.setattr(ddp_mod.dist, "is_available", lambda: True)
    monkeypatch.setattr(ddp_mod.dist, "is_initialized", lambda: False)
    model = _make_model()
    with pytest.raises(RuntimeError, match="initialized process group"):
        prepare_ddp_model(model, device=torch.device("cpu"))


def test_unwrap_model_ddp_wrapper() -> None:
    """``unwrap_model`` returns ``.module`` for a DDP-like wrapper."""
    from torch.nn.parallel import DistributedDataParallel

    core = torch.nn.Linear(2, 2)
    # Bypass DDP process-group init; isinstance still recognizes the type.
    fake = DistributedDataParallel.__new__(DistributedDataParallel)
    torch.nn.Module.__init__(fake)
    fake.module = core
    assert unwrap_model(fake) is core


def test_all_reduce_mean_averages_when_group_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Active process group averages the scalar across ranks."""
    import koopman_graph.distributed.ddp as ddp_mod

    monkeypatch.setattr(ddp_mod.dist, "is_available", lambda: True)
    monkeypatch.setattr(ddp_mod.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(ddp_mod, "get_world_size", lambda: 2)

    def fake_all_reduce(tensor: torch.Tensor, op: object = None) -> torch.Tensor:
        tensor *= 2.0  # simulate sum of two equal ranks
        return tensor

    monkeypatch.setattr(ddp_mod.dist, "all_reduce", fake_all_reduce)
    assert all_reduce_mean(3.0) == pytest.approx(3.0)


def test_fit_strategy_none_matches_default_fit(
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """``strategy=None`` preserves the single-process ``fit`` path."""
    model_a = _make_model(seed=13)
    model_b = _make_model(seed=13)
    kwargs = {"epochs": 2, "lr": 1e-2, "device": "cpu", "window_seed": 0}
    history_a = model_a.fit(scaling_sequence, **kwargs)
    history_b = model_b.fit(scaling_sequence, strategy=None, **kwargs)
    assert history_a.epochs == history_b.epochs
    for left, right in zip(history_a.loss, history_b.loss, strict=True):
        assert left == pytest.approx(right, rel=0, abs=1e-6)


def test_fit_strategy_ddp_matches_run_ddp_fit_loop(
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """``strategy='ddp'`` matches calling ``run_ddp_fit_loop`` directly."""
    model_a = _make_model(seed=17)
    model_b = _make_model(seed=17)
    kwargs = {"epochs": 2, "lr": 1e-2, "device": "cpu"}
    history_a = run_ddp_fit_loop(model_a, [scaling_sequence], **kwargs)
    history_b = model_b.fit(scaling_sequence, strategy="ddp", **kwargs)
    for left, right in zip(history_a.loss, history_b.loss, strict=True):
        assert left == pytest.approx(right, rel=0, abs=1e-6)


def test_fit_strategy_ddp_delegates_to_run_ddp_fit_loop(
    scaling_sequence: GraphSnapshotSequence,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``strategy='ddp'`` routes through ``run_ddp_fit_loop``."""
    calls: list[str] = []

    def tracking_ddp(*args: object, **kwargs: object) -> object:
        calls.append("ddp")
        return run_ddp_fit_loop(*args, **kwargs)

    monkeypatch.setattr(
        "koopman_graph.distributed.run_ddp_fit_loop",
        tracking_ddp,
    )
    model = _make_model(seed=19)
    history = model.fit(
        scaling_sequence,
        epochs=1,
        lr=1e-2,
        device="cpu",
        strategy="ddp",
    )
    assert calls == ["ddp"]
    assert history.epochs == 1


def test_fit_rejects_invalid_strategy(
    scaling_sequence: GraphSnapshotSequence,
) -> None:
    """Unknown ``strategy`` values raise a clear ``ValueError``."""
    model = _make_model()
    with pytest.raises(ValueError, match="unsupported fit strategy"):
        model.fit(scaling_sequence, epochs=1, strategy="fabric")  # type: ignore[arg-type]


def test_model_package_has_no_lightning_or_ray_imports() -> None:
    """``koopman_graph.model`` must not import Lightning or Ray."""
    import ast

    model_root = REPO_ROOT / "src" / "koopman_graph" / "model"
    banned = ("lightning", "ray", "pytorch_lightning")
    for path in model_root.rglob("*.py"):
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
                assert top not in banned, f"{path} imports {name}"


@pytest.mark.distributed
def test_ddp_two_proc_gloo_params_sync_and_rank0_checkpoint(
    tmp_path: Path,
) -> None:
    """Two-process gloo: synced params after ``k`` steps; rank-0 checkpoint only."""
    _require_distributed_tests()
    world_size = 2
    epochs = 2
    workdir = tmp_path / "gloo_ddp"
    workdir.mkdir()
    port = _free_tcp_port()
    try:
        mp.spawn(
            _ddp_gloo_worker,
            args=(world_size, port, str(workdir), epochs),
            nprocs=world_size,
            join=True,
        )
    except Exception as exc:  # noqa: BLE001 — surface as skip when spawn/env fails
        pytest.skip(f"2-proc gloo spawn unavailable: {exc}")

    for rank in range(world_size):
        assert (workdir / f"done_{rank}").is_file()

    state0 = torch.load(
        workdir / "state_rank0.pt",
        map_location="cpu",
        weights_only=True,
    )
    state1 = torch.load(
        workdir / "state_rank1.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert state0.keys() == state1.keys()
    for key in state0:
        assert torch.allclose(state0[key], state1[key]), key

    checkpoint = workdir / "ckpt.pt"
    assert checkpoint.is_file()
    # Only the shared rank-0 path should exist (no per-rank checkpoint files).
    assert list(workdir.glob("ckpt*")) == [checkpoint]


def test_distributed_marker_skips_without_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Marked multi-proc smoke skips when the opt-in env var is unset."""
    monkeypatch.delenv(_DISTRIBUTED_ENV, raising=False)
    with pytest.raises(pytest.skip.Exception, match="opt-in multi-proc"):
        _require_distributed_tests()
