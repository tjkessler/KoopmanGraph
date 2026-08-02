"""Tests for optional Ray ensemble parallel fit (Tier 2)."""

from __future__ import annotations

import ast
import importlib
import math
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
from koopman_graph import distributed as distributed_pkg
from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.uq import EnsembleGraphKoopmanModel

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TUNE_EXAMPLE = _REPO_ROOT / "examples" / "scripts" / "ray_tune_koopman_example.py"


def _tiny_sequence() -> GraphSnapshotSequence:
    """Deterministic decay trajectory for ensemble smoke tests."""
    edge = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    return GraphSnapshotSequence(
        [Data(x=torch.ones(2, 3) * (0.9**t), edge_index=edge) for t in range(4)]
    )


def _make_member_factory():
    """Return a nested factory Ray can cloudpickle without the test module."""

    def member_factory() -> GraphKoopmanModel:
        return GraphKoopmanModel(
            encoder=GNNEncoder(3, 8, 4),
            decoder=GNNDecoder(4, 8, 3),
            latent_dim=4,
            time_step=0.1,
        )

    return member_factory


def test_fit_ensemble_with_ray_exported() -> None:
    """``fit_ensemble_with_ray`` is listed in the public distributed API."""
    assert "fit_ensemble_with_ray" in distributed_pkg.__all__


def test_fit_ensemble_with_ray_missing_ray(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing Ray raises an actionable install hint."""
    import koopman_graph.distributed.ray_jobs as mod

    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None) -> object:
        if name == "ray" or name.startswith("ray."):
            raise ImportError("simulated missing ray")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    with pytest.raises(ImportError, match="koopman-graph\\[ray\\]"):
        mod._import_ray()


def test_ensemble_fit_requires_factory_for_ray() -> None:
    """Ray backend without ``member_factory`` raises ``ValueError``."""
    factory = _make_member_factory()
    ensemble = EnsembleGraphKoopmanModel.from_factory(
        factory,
        n_members=2,
        seeds=(0, 1),
    )
    with pytest.raises(ValueError, match="member_factory"):
        ensemble.fit(_tiny_sequence(), epochs=1, parallel_backend="ray")


def test_ensemble_fit_rejects_unknown_parallel_backend() -> None:
    """Unknown ``parallel_backend`` raises ``ValueError``."""
    factory = _make_member_factory()
    ensemble = EnsembleGraphKoopmanModel.from_factory(
        factory,
        n_members=1,
        seeds=(0,),
    )
    with pytest.raises(ValueError, match="parallel_backend"):
        ensemble.fit(
            _tiny_sequence(),
            epochs=1,
            parallel_backend="bogus",
            member_factory=factory,
        )


def test_fit_ensemble_with_ray_validation_guards() -> None:
    """``num_members``, ``seeds``, and banned fit kwargs are validated early."""
    from koopman_graph.distributed.ray_jobs import fit_ensemble_with_ray

    factory = _make_member_factory()
    sequence = _tiny_sequence()
    with pytest.raises(ValueError, match="num_members must be >= 1"):
        fit_ensemble_with_ray(factory, sequence, num_members=0)
    with pytest.raises(ValueError, match="seeds must have length num_members"):
        fit_ensemble_with_ray(
            factory,
            sequence,
            num_members=2,
            seeds=(0,),
        )
    with pytest.raises(TypeError, match="parallel_backend"):
        fit_ensemble_with_ray(
            factory,
            sequence,
            num_members=1,
            parallel_backend="ray",
        )
    with pytest.raises(TypeError, match="member_factory"):
        fit_ensemble_with_ray(
            factory,
            sequence,
            num_members=1,
            member_factory=factory,
        )


def test_fit_member_task_in_process_returns_cpu_state() -> None:
    """Worker helper fits one member and returns CPU ``state_dict`` + history."""
    from koopman_graph.distributed.ray_jobs import _fit_member_task

    factory = _make_member_factory()
    state_dict, history = _fit_member_task(
        factory,
        _tiny_sequence(),
        seed=3,
        fit_kwargs={"epochs": 1, "lr": 1e-2},
    )
    assert history.reconstruction_loss
    assert all(math.isfinite(value) for value in history.reconstruction_loss)
    assert state_dict
    assert all(tensor.device.type == "cpu" for tensor in state_dict.values())


def test_fit_member_task_without_seed() -> None:
    """``seed=None`` skips ``torch.manual_seed`` and still returns a history."""
    from koopman_graph.distributed.ray_jobs import _fit_member_task

    state_dict, history = _fit_member_task(
        _make_member_factory(),
        _tiny_sequence(),
        seed=None,
        fit_kwargs={"epochs": 1, "lr": 1e-2},
    )
    assert state_dict
    assert history.reconstruction_loss


def test_fit_ensemble_with_ray_default_seeds_and_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default seeds are ``0..n-1``; uninitialized Ray calls ``ray.init``."""
    ray = pytest.importorskip("ray")
    import koopman_graph.distributed.ray_jobs as mod

    captured: dict[str, object] = {}

    class _Remote:
        def remote(self, *args: object, **kwargs: object) -> object:
            return ("future", args, kwargs)

    def fake_remote(fn: object) -> _Remote:
        captured["remote_fn"] = fn
        return _Remote()

    def fake_get(futures: object) -> list[tuple[dict[str, torch.Tensor], object]]:
        captured["futures"] = futures
        history = type("H", (), {"reconstruction_loss": [1.0]})()
        count = len(futures) if isinstance(futures, list) else 1
        return [({"w": torch.zeros(1)}, history) for _ in range(count)]

    def fake_init(**kwargs: object) -> None:
        captured["init"] = kwargs

    monkeypatch.setattr(ray, "is_initialized", lambda: False)
    monkeypatch.setattr(ray, "init", fake_init)
    monkeypatch.setattr(ray, "put", lambda data: ("ref", data))
    monkeypatch.setattr(ray, "remote", fake_remote)
    monkeypatch.setattr(ray, "get", fake_get)
    monkeypatch.setattr(mod, "_import_ray", lambda: ray)

    factory = _make_member_factory()
    state_dicts, histories = mod.fit_ensemble_with_ray(
        factory,
        _tiny_sequence(),
        num_members=2,
        ray_address="auto",
        epochs=1,
    )
    assert captured["init"] == mod._ray_init_kwargs(ray_address="auto")
    assert len(state_dicts) == 2
    assert len(histories) == 2
    # Default seeds are 0 and 1 when omitted.
    futures = captured["futures"]
    assert isinstance(futures, list)
    assert len(futures) == 2
    assert futures[0][1][2] == 0
    assert futures[1][1][2] == 1

    captured.clear()
    state_dicts, histories = mod.fit_ensemble_with_ray(
        factory,
        _tiny_sequence(),
        num_members=1,
        epochs=1,
    )
    assert captured["init"] == mod._ray_init_kwargs()
    assert len(state_dicts) == 1
    assert len(histories) == 1


def test_fit_ensemble_with_ray_two_members_finite_histories() -> None:
    """Ray path fits two members with finite losses under a local runtime."""
    ray = pytest.importorskip("ray")
    from koopman_graph.distributed import fit_ensemble_with_ray
    from koopman_graph.distributed.ray_jobs import _ray_init_kwargs

    if not ray.is_initialized():
        ray.init(**_ray_init_kwargs(num_cpus=2))

    factory = _make_member_factory()
    sequence = _tiny_sequence()
    state_dicts, histories = fit_ensemble_with_ray(
        factory,
        sequence,
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

    ensemble = EnsembleGraphKoopmanModel.from_factory(
        factory,
        n_members=2,
        seeds=(0, 1),
    )
    hook_histories = ensemble.fit(
        sequence,
        epochs=2,
        lr=1e-2,
        seeds=(10, 11),
        parallel_backend="ray",
        member_factory=factory,
    )
    assert len(hook_histories) == 2
    for history in hook_histories:
        assert history.reconstruction_loss
        assert all(math.isfinite(value) for value in history.reconstruction_loss)

    preds = ensemble.predict(sequence[0], steps=2)
    assert len(preds) == 2
    assert all(torch.isfinite(snap.x).all() for snap in preds)


def test_no_tune_api_in_distributed_all() -> None:
    """Library does not expose Ray Tune / AutoML symbols on ``__all__``."""
    banned = {
        "Tuner",
        "tune",
        "build_tuner",
        "fit_with_tune",
        "ray_tune",
        "TuneConfig",
    }
    exported = set(distributed_pkg.__all__)
    assert banned.isdisjoint(exported)
    for name in banned:
        assert not hasattr(distributed_pkg, name)


def test_ray_tune_example_script_exists_and_parses() -> None:
    """Examples-only Tune script is present and syntactically valid."""
    assert _TUNE_EXAMPLE.is_file()
    source = _TUNE_EXAMPLE.read_text(encoding="utf-8")
    ast.parse(source)
    assert "search space" in source.lower() or "Search space" in source
    assert "koopman-graph[ray]" in source
