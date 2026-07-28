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


def test_fit_ensemble_with_ray_two_members_finite_histories() -> None:
    """Ray path fits two members with finite losses under a local runtime."""
    ray = pytest.importorskip("ray")
    from koopman_graph.distributed import fit_ensemble_with_ray

    if not ray.is_initialized():
        ray.init(num_cpus=2, ignore_reinit_error=True)

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
