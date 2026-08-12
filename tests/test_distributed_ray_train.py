"""Tests for the Ray Train lazy-import boundary and fit loop (TASK-1910/1911)."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
from koopman_graph.data import GraphSnapshotSequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DISTRIBUTED_INIT = _REPO_ROOT / "src" / "koopman_graph" / "distributed" / "__init__.py"

# World-size-1 Ray Train vs native DDP (TASK-1913). Measured max |Δloss| on
# this fixture was 0.0 (2026-08-02); keep a small absolute slack for CI float
# noise / Ray orchestration, not a multi-GPU or multi-node gate (R6).
_WS1_LOSS_ABS = 1e-5
_WS1_EPOCHS = 3
_WS1_LR = 1e-2
_WS1_SEED = 0


def _tiny_sequence() -> GraphSnapshotSequence:
    """Deterministic two-node decay trajectory for fit smoke tests."""
    edge = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    return GraphSnapshotSequence(
        [Data(x=torch.ones(2, 3) * (0.9**t), edge_index=edge) for t in range(4)]
    )


def _tiny_model(*, seed: int = 0) -> GraphKoopmanModel:
    torch.manual_seed(seed)
    return GraphKoopmanModel(
        encoder=GNNEncoder(3, 8, 4, num_layers=1),
        decoder=GNNDecoder(4, 8, 3, num_layers=1),
        latent_dim=4,
        time_step=0.1,
    )


def test_distributed_init_does_not_eagerly_import_ray_train() -> None:
    """Top-level ``distributed/__init__.py`` must not import ``ray_train``."""
    tree = ast.parse(_DISTRIBUTED_INIT.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            assert "ray_train" not in node.module
            for alias in node.names:
                assert alias.name != "ray_train"
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "ray_train" not in alias.name
                assert not alias.name.startswith("ray.")


def test_package_imports_without_loading_ray_train() -> None:
    """Importing ``koopman_graph.distributed`` does not load ``ray_train``."""
    import sys

    sys.modules.pop("koopman_graph.distributed.ray_train", None)
    import koopman_graph.distributed as distributed_pkg

    assert "koopman_graph.distributed.ray_train" not in sys.modules
    assert "run_ray_train_fit_loop" in distributed_pkg.__all__


def test_run_ray_train_fit_loop_lazy_export() -> None:
    """``run_ray_train_fit_loop`` resolves via ``__getattr__`` without eager load."""
    import sys

    import koopman_graph.distributed as distributed_pkg

    sys.modules.pop("koopman_graph.distributed.ray_train", None)
    fn = distributed_pkg.run_ray_train_fit_loop
    assert callable(fn)
    assert "koopman_graph.distributed.ray_train" in sys.modules


def test_import_ray_train_missing_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing Ray / Ray Train raises an actionable install hint."""
    import koopman_graph.distributed.ray_train as mod

    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None) -> object:
        if name == "ray" or name.startswith("ray."):
            raise ImportError("simulated missing ray")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    with pytest.raises(ImportError, match=r"koopman-graph\[ray\]"):
        mod._import_ray_train()
    with pytest.raises(ImportError, match=r"koopman-graph\[ray\]"):
        mod.run_ray_train_fit_loop(_tiny_model(), [_tiny_sequence()], epochs=1)


def test_import_ray_train_when_installed() -> None:
    """Helper returns ``ray.train`` when the ``[ray]`` extra is present."""
    try:
        importlib.import_module("ray.train")
    except ImportError:
        pytest.skip("ray[train] not installed in this environment")
    from koopman_graph.distributed.ray_train import _import_ray_train

    train = _import_ray_train()
    assert train.__name__ == "ray.train"


def test_pyproject_ray_extra_pins_train() -> None:
    """``[ray]`` and ``[distributed]`` extras request ``ray[train]``."""
    text = (_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "ray[train]>=2.9.0" in text
    # Meta extra must not keep a bare ``ray>=`` pin beside the train pin.
    assert "ray>=2.9.0" not in text.replace("ray[train]>=2.9.0", "")


def test_run_ray_train_fit_loop_rejects_bad_num_workers() -> None:
    """``num_workers < 1`` raises before launching Ray."""
    try:
        importlib.import_module("ray.train")
    except ImportError:
        pytest.skip("ray[train] not installed in this environment")
    from koopman_graph.distributed.ray_train import run_ray_train_fit_loop

    with pytest.raises(ValueError, match="num_workers"):
        run_ray_train_fit_loop(
            _tiny_model(),
            [_tiny_sequence()],
            num_workers=0,
            epochs=1,
        )


@pytest.mark.ray
def test_run_ray_train_fit_loop_world_size_one_smoke() -> None:
    """World-size-1 TorchTrainer smoke reuses ``_fit_epochs`` successfully."""
    try:
        importlib.import_module("ray.train")
    except ImportError:
        pytest.skip("ray[train] not installed in this environment")
    from koopman_graph.distributed import run_ray_train_fit_loop

    model = _tiny_model(seed=0)
    history = run_ray_train_fit_loop(
        model,
        [_tiny_sequence()],
        num_workers=1,
        use_gpu=False,
        epochs=2,
        lr=1e-2,
        seed=0,
    )
    assert history.epochs == 2
    assert len(history.loss) == 2
    assert all(torch.isfinite(torch.tensor(v)) for v in history.loss)


def test_unwrapped_state_dict_strips_ddp_module_prefix() -> None:
    """``_unwrapped_state_dict`` reuses ``unwrap_model`` (no prefix shim)."""
    from torch.nn.parallel import DistributedDataParallel

    from koopman_graph.distributed.ddp import unwrap_model
    from koopman_graph.distributed.ray_train import _unwrapped_state_dict

    core = _tiny_model(seed=5)
    # Bypass process-group init; isinstance still recognizes DDP.
    wrapped = DistributedDataParallel.__new__(DistributedDataParallel)
    torch.nn.Module.__init__(wrapped)
    wrapped.module = core
    assert unwrap_model(wrapped) is core
    state = _unwrapped_state_dict(wrapped)
    assert state.keys() == core.state_dict().keys()
    assert not any(key.startswith("module.") for key in state)


@pytest.mark.ray
def test_ray_train_checkpoint_path_matches_single_process_keys(
    tmp_path: Path,
) -> None:
    """``checkpoint_path`` keys match single-process format-1 (no ``module.``)."""
    try:
        importlib.import_module("ray.train")
    except ImportError:
        pytest.skip("ray[train] not installed in this environment")
    from koopman_graph.distributed import run_ray_train_fit_loop
    from koopman_graph.serialization import save_checkpoint

    sequence = _tiny_sequence()
    single = _tiny_model(seed=0)
    single_path = tmp_path / "single.pt"
    save_checkpoint(single, single_path, format="legacy_pt")
    single_keys = set(
        torch.load(single_path, map_location="cpu", weights_only=False)["state_dict"]
    )

    ray_model = _tiny_model(seed=1)
    ray_path = tmp_path / "ray.pt"
    run_ray_train_fit_loop(
        ray_model,
        [sequence],
        num_workers=1,
        use_gpu=False,
        epochs=1,
        lr=1e-2,
        seed=1,
        restore_best_weights=True,
        checkpoint_path=ray_path,
    )
    assert ray_path.is_file()
    ray_payload = torch.load(ray_path, map_location="cpu", weights_only=False)
    ray_keys = set(ray_payload["state_dict"])
    assert not any(key.startswith("module.") for key in ray_keys)
    assert ray_keys == single_keys


@pytest.mark.ray
def test_world_size_one_ray_train_matches_native_ddp_loss() -> None:
    """Seeded world-size-1 Ray Train loss matches ``run_ddp_fit_loop``.

    **CI gate is world size 1 only** (design R6). Multi-GPU / multi-node
    smokes stay manual (TASK-1914). Tolerance: see ``_WS1_LOSS_ABS``.
    """
    try:
        importlib.import_module("ray.train")
    except ImportError:
        pytest.skip("ray[train] not installed in this environment")
    from koopman_graph.distributed import (
        run_ddp_fit_loop,
        run_ray_train_fit_loop,
        seed_everything,
    )

    seed_everything(_WS1_SEED)
    ddp_model = _tiny_model(seed=_WS1_SEED)
    ddp_history = run_ddp_fit_loop(
        ddp_model,
        [_tiny_sequence()],
        epochs=_WS1_EPOCHS,
        lr=_WS1_LR,
        device="cpu",
    )

    seed_everything(_WS1_SEED)
    ray_model = _tiny_model(seed=_WS1_SEED)
    ray_history = run_ray_train_fit_loop(
        ray_model,
        [_tiny_sequence()],
        num_workers=1,
        use_gpu=False,
        epochs=_WS1_EPOCHS,
        lr=_WS1_LR,
        seed=_WS1_SEED,
    )

    assert ray_history.epochs == ddp_history.epochs == _WS1_EPOCHS
    assert len(ray_history.loss) == len(ddp_history.loss) == _WS1_EPOCHS
    for epoch, (ray_loss, ddp_loss) in enumerate(
        zip(ray_history.loss, ddp_history.loss, strict=True)
    ):
        assert ray_loss == pytest.approx(ddp_loss, abs=_WS1_LOSS_ABS), (
            f"epoch {epoch}: ray={ray_loss} ddp={ddp_loss} (abs tol {_WS1_LOSS_ABS})"
        )


@pytest.mark.ray
def test_ray_train_result_payload_has_no_module_prefix(tmp_path: Path) -> None:
    """Train result ``state_dict`` restored onto the driver has bare keys."""
    try:
        importlib.import_module("ray.train")
    except ImportError:
        pytest.skip("ray[train] not installed in this environment")
    from koopman_graph.distributed import run_ray_train_fit_loop
    from koopman_graph.serialization import save_checkpoint

    model = _tiny_model(seed=2)
    run_ray_train_fit_loop(
        model,
        [_tiny_sequence()],
        num_workers=1,
        use_gpu=False,
        epochs=1,
        lr=1e-2,
        seed=2,
    )
    out = tmp_path / "after_ray.pt"
    save_checkpoint(model, out, format="legacy_pt")
    keys = set(torch.load(out, map_location="cpu", weights_only=False)["state_dict"])
    assert not any(key.startswith("module.") for key in keys)
    # Same key set as a never-wrapped peer architecture.
    peer = _tiny_model(seed=9)
    peer_path = tmp_path / "peer.pt"
    save_checkpoint(peer, peer_path, format="legacy_pt")
    peer_keys = set(
        torch.load(peer_path, map_location="cpu", weights_only=False)["state_dict"]
    )
    assert keys == peer_keys
