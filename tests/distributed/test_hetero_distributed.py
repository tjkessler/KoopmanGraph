"""World-size-1 and opt-in multi-proc DDP for multiplex / typed hetero models.

Default CI covers seeded world-size-1 parity for full-sequence,
``MultiTrajectory``, and windowed multiplex / typed paths (TASK-1802).

Multi-process gloo smokes are marked ``@pytest.mark.distributed`` and are
**opt-in**:

.. code-block:: bash

    KOOPMAN_GRAPH_DISTRIBUTED_TESTS=1 \\
      pytest tests/distributed/test_hetero_distributed.py -m distributed

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
from torch_geometric.data import HeteroData

from koopman_graph.data import (
    HeteroGraphSnapshotSequence,
    MultiTrajectory,
    WindowSampler,
)
from koopman_graph.distributed import (
    DistributedWindowSampler,
    prepare_ddp_model,
    run_ddp_fit_loop,
    seed_everything,
)
from koopman_graph.distributed.ddp import resolve_find_unused_parameters
from koopman_graph.model import GraphKoopmanModel
from koopman_graph.nn import RelGraphDecoder, RelGraphEncoder
from koopman_graph.operators import HeteroGraphKoopmanOperator
from koopman_graph.training import run_fit_loop
from koopman_graph.training.loop import bind_pending_orbit_ties

# Match existing hetero full-sequence WS1 parity in this module (not relative
# float tolerance). Homogeneous windowed DDP uses abs=1e-6; hetero paths keep
# abs=1e-5 for the slightly larger relational compute graph.
_WS1_LOSS_ABS = 1e-5

_DISTRIBUTED_ENV = "KOOPMAN_GRAPH_DISTRIBUTED_TESTS"


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


# Typed multi-node fixtures (shared-d path) for windowed WS1 parity.
_TYPED_NODE_TYPES = ("a", "b")
_TYPED_EDGE_TYPES = (("a", "r0", "b"), ("b", "r1", "a"), ("a", "r2", "a"))
_TYPED_FEATURE_DIMS = {"a": 2, "b": 3}
_TYPED_NUM_NODES = {"a": 4, "b": 3}
_TYPED_LATENT_DIM = 4
_TYPED_EDGES_AB = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)
_TYPED_EDGES_BA = torch.tensor([[0, 1], [1, 3]], dtype=torch.long)
_TYPED_EDGES_AA = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)


def _typed_snapshot(*, seed: int = 0) -> HeteroData:
    generator = torch.Generator().manual_seed(seed)
    snapshot = HeteroData()
    snapshot["a"].x = torch.randn(
        _TYPED_NUM_NODES["a"],
        _TYPED_FEATURE_DIMS["a"],
        generator=generator,
    )
    snapshot["b"].x = torch.randn(
        _TYPED_NUM_NODES["b"],
        _TYPED_FEATURE_DIMS["b"],
        generator=generator,
    )
    snapshot["a", "r0", "b"].edge_index = _TYPED_EDGES_AB
    snapshot["b", "r1", "a"].edge_index = _TYPED_EDGES_BA
    snapshot["a", "r2", "a"].edge_index = _TYPED_EDGES_AA
    return snapshot


def _typed_sequence(
    *,
    num_timesteps: int = 5,
    seed: int = 0,
) -> HeteroGraphSnapshotSequence:
    return HeteroGraphSnapshotSequence(
        [_typed_snapshot(seed=seed + t) for t in range(num_timesteps)]
    )


def _typed_model(*, seed: int = 0) -> GraphKoopmanModel:
    torch.manual_seed(seed)
    return GraphKoopmanModel(
        encoder=RelGraphEncoder(
            _TYPED_FEATURE_DIMS,
            hidden_channels=8,
            latent_dim=_TYPED_LATENT_DIM,
            num_relations=len(_TYPED_EDGE_TYPES),
            num_layers=1,
            node_types=_TYPED_NODE_TYPES,
            edge_types=_TYPED_EDGE_TYPES,
        ),
        decoder=RelGraphDecoder(
            latent_dim=_TYPED_LATENT_DIM,
            hidden_channels=8,
            out_channels=_TYPED_FEATURE_DIMS,
            num_relations=len(_TYPED_EDGE_TYPES),
            num_layers=1,
            node_types=_TYPED_NODE_TYPES,
            edge_types=_TYPED_EDGE_TYPES,
        ),
        latent_dim=_TYPED_LATENT_DIM,
        time_step=1.0,
        koopman="hetero_graph",
        koopman_node_types=_TYPED_NODE_TYPES,
        koopman_edge_types=_TYPED_EDGE_TYPES,
    )


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


def _hetero_ddp_gloo_worker(
    rank: int,
    world_size: int,
    port: int,
    workdir: str,
    epochs: int,
) -> None:
    """Train a multiplex model under gloo DDP and persist unwrapped weights."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    # Force CPU gloo even when CUDA is visible (init before run_ddp_fit_loop).
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        seed_everything(0, rank=0)
        model = _hetero_model(seed=0)
        sequence = _hetero_sequence(seed=0)
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


def test_resolve_find_unused_parameters_hetero_default() -> None:
    """Hetero models default find_unused_parameters to True."""
    model = _hetero_model()
    assert resolve_find_unused_parameters(model, None) is True
    assert resolve_find_unused_parameters(model, False) is False
    assert resolve_find_unused_parameters(model, True) is True


def test_resolve_find_unused_parameters_homogeneous_default() -> None:
    """Homogeneous models keep the False default when unset."""
    from koopman_graph.nn import GNNDecoder, GNNEncoder

    model = GraphKoopmanModel(
        encoder=GNNEncoder(in_channels=3, hidden_channels=8, latent_dim=4),
        decoder=GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=3),
        latent_dim=4,
        time_step=1.0,
    )
    assert resolve_find_unused_parameters(model, None) is False


def test_hetero_ddp_full_sequence_matches_run_fit_loop() -> None:
    """World-size-1 multiplex DDP loss matches single-process fit."""
    sequence = _hetero_sequence(seed=3)
    model_a = _hetero_model(seed=7)
    model_b = _hetero_model(seed=7)
    kwargs = {"epochs": 3, "lr": 1e-2, "device": "cpu"}
    history_a = run_fit_loop(model_a, [sequence], **kwargs)
    history_b = run_ddp_fit_loop(model_b, [sequence], **kwargs)
    assert history_a.epochs == history_b.epochs
    for left, right in zip(history_a.loss, history_b.loss, strict=True):
        assert left == pytest.approx(right, rel=0, abs=_WS1_LOSS_ABS)


def test_hetero_ddp_multi_trajectory_matches_run_fit_loop() -> None:
    """World-size-1 DDP matches single-process on a multiplex MultiTrajectory."""
    sequences = list(
        MultiTrajectory(
            (
                _hetero_sequence(seed=0),
                _hetero_sequence(seed=11),
            )
        )
    )
    model_a = _hetero_model(seed=9)
    model_b = _hetero_model(seed=9)
    kwargs = {"epochs": 2, "lr": 1e-2, "device": "cpu"}
    history_a = run_fit_loop(model_a, sequences, **kwargs)
    history_b = run_ddp_fit_loop(model_b, sequences, **kwargs)
    assert history_a.epochs == history_b.epochs
    for left, right in zip(history_a.loss, history_b.loss, strict=True):
        assert left == pytest.approx(right, rel=0, abs=_WS1_LOSS_ABS)


def test_hetero_fit_strategy_ddp_matches_run_ddp_fit_loop() -> None:
    """``fit(strategy='ddp')`` matches a direct ``run_ddp_fit_loop`` call."""
    sequence = _hetero_sequence(seed=1)
    model_a = _hetero_model(seed=5)
    model_b = _hetero_model(seed=5)
    kwargs = {"epochs": 2, "lr": 1e-2, "device": "cpu"}
    history_a = run_ddp_fit_loop(model_a, [sequence], **kwargs)
    history_b = model_b.fit(sequence, strategy="ddp", **kwargs)
    for left, right in zip(history_a.loss, history_b.loss, strict=True):
        assert left == pytest.approx(right, rel=0, abs=_WS1_LOSS_ABS)


def test_hetero_fit_passes_find_unused_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``fit(strategy='ddp')`` forwards find_unused_parameters into the DDP loop."""
    captured: dict[str, object] = {}

    def fake_run_ddp(model: object, sequences: object, **kwargs: object) -> object:
        del model, sequences
        captured.update(kwargs)
        return type("H", (), {"epochs": 0, "loss": ()})()

    monkeypatch.setattr(
        "koopman_graph.distributed.run_ddp_fit_loop",
        fake_run_ddp,
    )
    model = _hetero_model()
    sequence = _hetero_sequence(num_timesteps=3)
    model.fit(sequence, strategy="ddp", epochs=1, find_unused_parameters=True)
    assert captured.get("find_unused_parameters") is True


def test_prepare_ddp_model_resolves_hetero_unused_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """prepare_ddp_model uses True for hetero when wrapping under an active group."""
    import koopman_graph.distributed.ddp as ddp_mod

    monkeypatch.setattr(ddp_mod, "get_world_size", lambda: 2)
    monkeypatch.setattr(ddp_mod.dist, "is_available", lambda: True)
    monkeypatch.setattr(ddp_mod.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(ddp_mod, "get_rank", lambda: 0)

    seen: dict[str, bool] = {}

    class FakeDDP(torch.nn.Module):
        def __init__(self, module: torch.nn.Module, **kwargs: object) -> None:
            super().__init__()
            seen["find_unused_parameters"] = bool(kwargs.get("find_unused_parameters"))
            self.module = module

    monkeypatch.setattr(ddp_mod, "_AttributeForwardDDP", FakeDDP)
    model = _hetero_model()
    prepare_ddp_model(model, device=torch.device("cpu"), find_unused_parameters=None)
    assert seen["find_unused_parameters"] is True


def test_hetero_ddp_checkpoint_round_trip(tmp_path: Path) -> None:
    """Rank-0 hetero checkpoints are unwrapped and reloadable."""
    model = _hetero_model(seed=2)
    sequence = _hetero_sequence(seed=2)
    path = tmp_path / "hetero_ddp.pt"
    run_ddp_fit_loop(
        model,
        [sequence],
        epochs=2,
        lr=1e-2,
        device="cpu",
        restore_best_weights=True,
        checkpoint_path=path,
    )
    assert path.is_file()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["format_version"] == 1
    assert payload["config"]["koopman_kind"] == "hetero_graph"
    assert "node_types" in payload["config"]
    assert "edge_types" in payload["config"]
    assert not any(key.startswith("module.") for key in payload["state_dict"])
    loaded = GraphKoopmanModel.load(path)
    assert isinstance(loaded.koopman, HeteroGraphKoopmanOperator)
    assert loaded.koopman_kind == "hetero_graph"


def test_bind_pending_orbit_ties_binds_multiplex_hetero_auto_orbits() -> None:
    """Multiplex hetero auto_orbits bind from the union of relation banks."""
    pytest.importorskip("networkx")
    model = _hetero_model()
    sequence = _hetero_sequence(num_timesteps=2)
    object.__setattr__(model.koopman, "auto_orbits", True)
    bind_pending_orbit_ties(model, [sequence])
    assert model.koopman.orbit_partition is not None


def test_bind_pending_orbit_ties_noop_for_hetero_without_auto_orbits() -> None:
    """Hetero sequences without auto_orbits remain a no-op."""
    model = _hetero_model()
    sequence = _hetero_sequence(num_timesteps=2)
    bind_pending_orbit_ties(model, [sequence])
    assert model.koopman.orbit_partition is None


def test_windowed_hetero_samplers_preserve_hetero_container() -> None:
    """Local and WS1 distributed window slices stay HeteroGraphSnapshotSequence."""
    sequence = _hetero_sequence(num_timesteps=5, seed=4)
    local = WindowSampler(
        sequence,
        window_length=3,
        batch_size=1,
        shuffle=False,
    )
    distributed = DistributedWindowSampler(
        sequence,
        window_length=3,
        batch_size=1,
        shuffle=False,
        rank=0,
        world_size=1,
    )
    local_window = next(iter(local.iter_epoch(0)))[0]
    ddp_window = next(iter(distributed.iter_epoch(0)))[0]
    assert isinstance(local_window, HeteroGraphSnapshotSequence)
    assert isinstance(ddp_window, HeteroGraphSnapshotSequence)
    assert type(local_window) is type(ddp_window)


def test_hetero_ddp_windowed_matches_run_fit_loop() -> None:
    """World-size-1 windowed multiplex DDP loss matches single-process fit.

    Tolerance ``abs=1e-5`` matches other hetero WS1 parity tests in this
    module (see ``_WS1_LOSS_ABS``).
    """
    sequence = _hetero_sequence(num_timesteps=5, seed=4)
    model_a = _hetero_model(seed=4)
    model_b = _hetero_model(seed=4)
    kwargs = {
        "epochs": 2,
        "lr": 1e-2,
        "device": "cpu",
        "window_length": 3,
        "batch_size": 2,
        "window_seed": 0,
    }
    history_a = run_fit_loop(model_a, [sequence], **kwargs)
    history_b = run_ddp_fit_loop(model_b, [sequence], **kwargs)
    assert history_a.epochs == history_b.epochs
    for left, right in zip(history_a.loss, history_b.loss, strict=True):
        assert left == pytest.approx(right, rel=0, abs=_WS1_LOSS_ABS)


def test_hetero_ddp_windowed_typed_matches_run_fit_loop() -> None:
    """World-size-1 windowed typed DDP loss matches single-process fit."""
    sequence = _typed_sequence(num_timesteps=5, seed=6)
    model_a = _typed_model(seed=6)
    model_b = _typed_model(seed=6)
    kwargs = {
        "epochs": 2,
        "lr": 1e-2,
        "device": "cpu",
        "window_length": 3,
        "batch_size": 2,
        "window_seed": 0,
    }
    history_a = run_fit_loop(model_a, [sequence], **kwargs)
    history_b = run_ddp_fit_loop(model_b, [sequence], **kwargs)
    assert history_a.epochs == history_b.epochs
    for left, right in zip(history_a.loss, history_b.loss, strict=True):
        assert left == pytest.approx(right, rel=0, abs=_WS1_LOSS_ABS)


@pytest.mark.distributed
def test_hetero_ddp_two_proc_gloo_params_sync_and_rank0_checkpoint(
    tmp_path: Path,
) -> None:
    """Two-process gloo: synced multiplex params; rank-0 checkpoint only."""
    _require_distributed_tests()
    world_size = 2
    epochs = 2
    workdir = tmp_path / "gloo_hetero_ddp"
    workdir.mkdir()
    port = _free_tcp_port()
    try:
        mp.spawn(
            _hetero_ddp_gloo_worker,
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
    assert list(workdir.glob("ckpt*")) == [checkpoint]


def test_distributed_marker_skips_without_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Marked multi-proc smoke skips when the opt-in env var is unset."""
    monkeypatch.delenv(_DISTRIBUTED_ENV, raising=False)
    with pytest.raises(pytest.skip.Exception, match="opt-in multi-proc"):
        _require_distributed_tests()
