"""Tests for MD/MSM extras, contact graphs, and GraphVAMPBaseline."""

from __future__ import annotations

import builtins
import importlib.metadata
import re
import sys

import pytest
import torch
from torch_geometric.data import Data

from koopman_graph.baselines import GraphVAMPBaseline
from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.datasets.molecular import contact_edge_index


def _extra_requirement_names(extra: str) -> set[str]:
    """Return lowercased package names declared for a Provides-Extra."""
    requires = importlib.metadata.requires("koopman-graph") or []
    names: set[str] = set()
    marker = f'extra == "{extra}"'
    marker_alt = f"extra == '{extra}'"
    for req in requires:
        if marker not in req and marker_alt not in req:
            continue
        dep = req.split(";", maxsplit=1)[0].strip().lower()
        pkg = re.split(r"[<>=!~\[]", dep, maxsplit=1)[0].strip()
        names.add(pkg)
    return names


def test_md_and_msm_extras_declared() -> None:
    """``[md]`` and ``[msm]`` are declared and stay off the core require set."""
    dist = importlib.metadata.distribution("koopman-graph")
    extras = set(dist.metadata.get_all("Provides-Extra") or [])
    assert {"md", "msm"} <= extras

    requires = importlib.metadata.requires("koopman-graph") or []
    core_requires = [req for req in requires if "extra ==" not in req]
    core_joined = ";".join(core_requires).lower()
    assert "mdtraj" not in core_joined
    assert "deeptime" not in core_joined

    assert "mdtraj" in _extra_requirement_names("md")
    assert "deeptime" in _extra_requirement_names("msm")


def test_package_and_stub_modules_import_without_forcing_extras() -> None:
    """Core and stub modules import; mdtraj/deeptime stay optional at import."""
    import koopman_graph
    import koopman_graph.baselines.graph_vamp as graph_vamp
    import koopman_graph.datasets.molecular as molecular
    import koopman_graph.interop as interop

    assert koopman_graph.__version__
    assert "require_mdtraj" in molecular.__all__
    assert "require_deeptime" in interop.__all__
    assert hasattr(graph_vamp, "GraphVAMPBaseline")
    # Importing stubs must not place failed optional imports in sys.modules.
    assert sys.modules.get("mdtraj") is not False
    assert sys.modules.get("deeptime") is not False


def test_require_mdtraj_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing mdtraj raises an install-hinted ``ImportError``."""
    from koopman_graph.datasets.molecular import load_md_trajectory, require_mdtraj

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "mdtraj" or name.startswith("mdtraj."):
            raise ImportError("no mdtraj")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"koopman-graph\[md\]"):
        require_mdtraj()
    with pytest.raises(ImportError, match=r"koopman-graph\[md\]"):
        load_md_trajectory("unused.xtc")


def test_require_deeptime_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing deeptime raises an install-hinted ``ImportError``."""
    from koopman_graph.interop.deeptime import (
        require_deeptime,
        trajectory_features_to_deeptime,
    )

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "deeptime" or name.startswith("deeptime."):
            raise ImportError("no deeptime")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"koopman-graph\[msm\]"):
        require_deeptime()
    with pytest.raises(ImportError, match=r"koopman-graph\[msm\]"):
        trajectory_features_to_deeptime([[0.0, 1.0], [1.0, 0.0]], lag=1)
    # GraphVAMPBaseline does not require deeptime for construct / score.
    _ = GraphVAMPBaseline(in_channels=2, hidden_channels=4, latent_dim=2)


def test_md_loader_stub_raises_not_implemented_when_mdtraj_present() -> None:
    """When mdtraj imports, unfinished MD loader stubs still refuse."""
    pytest.importorskip("mdtraj")

    from koopman_graph.datasets.molecular import load_md_trajectory

    with pytest.raises(NotImplementedError, match="not implemented yet"):
        load_md_trajectory("unused.xtc")


def _synthetic_contact_sequence(
    *,
    seed: int = 0,
    num_timesteps: int = 16,
) -> tuple[GraphSnapshotSequence, torch.Tensor]:
    """Seeded three-atom trajectory with a static mid-cutoff contact graph."""
    generator = torch.Generator().manual_seed(seed)
    positions = _line_trimer_positions_nm().to(dtype=torch.float32)
    edge_index = contact_edge_index(positions, cutoff_nm=0.6)
    snapshots = []
    for t in range(num_timesteps):
        noise = 0.05 * torch.randn(3, 2, generator=generator)
        drift = 0.1 * torch.sin(torch.tensor(t / 3.0)) * torch.ones(3, 2)
        x = positions[:, :2] + drift + noise
        snapshots.append(Data(x=x, edge_index=edge_index.clone()))
    return GraphSnapshotSequence(snapshots), edge_index


def test_graph_vamp_exported_from_baselines_not_root() -> None:
    """GraphVAMPBaseline is on baselines ``__all__`` and off root ``__all__``."""
    import koopman_graph as kg
    import koopman_graph.baselines as baselines

    assert "GraphVAMPBaseline" in baselines.__all__
    assert "GraphVAMPBaseline" not in kg.__all__
    assert baselines.GraphVAMPBaseline is GraphVAMPBaseline


def test_graph_vamp_fit_score_smoke() -> None:
    """Fit/score smoke on a seeded synthetic contact-graph trajectory."""
    sequence, edge_index = _synthetic_contact_sequence(seed=7, num_timesteps=20)
    model = GraphVAMPBaseline(
        in_channels=2,
        hidden_channels=8,
        latent_dim=4,
        num_layers=1,
    )
    model.fit(sequence, lag=1, epochs=5, lr=1e-2, edge_index=edge_index)
    score = model.score(sequence, lag=1)
    assert isinstance(score, float)
    assert score == score  # not NaN
    assert model.edge_index is not None
    assert torch.equal(model.edge_index, edge_index)


def test_graph_vamp_fit_from_positions_nm() -> None:
    """Fit can build contacts from ``positions_nm`` / ``cutoff_nm``."""
    sequence, _ = _synthetic_contact_sequence(seed=3, num_timesteps=12)
    positions = _line_trimer_positions_nm().to(dtype=torch.float32)
    model = GraphVAMPBaseline(
        in_channels=2, hidden_channels=4, latent_dim=2, num_layers=1
    )
    model.fit(
        sequence,
        lag=1,
        epochs=3,
        positions_nm=positions,
        cutoff_nm=0.6,
    )
    assert model.score(sequence, lag=1) == model.score(sequence, lag=1)


def _line_trimer_positions_nm() -> torch.Tensor:
    """Three atoms on the x-axis at 0, 0.4, and 1.0 nm (hand-checked distances)."""
    return torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )


def test_atom_contacts_hand_checked() -> None:
    """Atom contacts match inclusive Euclidean distances in nanometres."""
    positions = _line_trimer_positions_nm()
    # d(0,1)=0.4, d(1,2)=0.6, d(0,2)=1.0
    edges_tight = contact_edge_index(positions, cutoff_nm=0.5)
    assert edges_tight.shape == (2, 1)
    assert torch.equal(edges_tight, torch.tensor([[0], [1]], dtype=torch.long))

    edges_mid = contact_edge_index(positions, cutoff_nm=0.6)
    assert edges_mid.shape == (2, 2)
    assert torch.equal(
        edges_mid,
        torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
    )

    edges_loose = contact_edge_index(positions, cutoff_nm=1.0)
    assert edges_loose.shape == (2, 3)
    assert torch.equal(
        edges_loose,
        torch.tensor([[0, 0, 1], [1, 2, 2]], dtype=torch.long),
    )


def test_residue_contacts_any_atom_pair() -> None:
    """Residue contacts fire when any inter-residue atom pair is in cutoff."""
    positions = _line_trimer_positions_nm()
    # Residues: {0,1} → node 0; {2} → node 1
    residue_ids = torch.tensor([10, 10, 20], dtype=torch.long)
    none = contact_edge_index(
        positions,
        cutoff_nm=0.5,
        granularity="residue",
        residue_ids=residue_ids,
    )
    assert none.shape == (2, 0)

    linked = contact_edge_index(
        positions,
        cutoff_nm=0.6,
        granularity="residue",
        residue_ids=residue_ids,
    )
    assert torch.equal(linked, torch.tensor([[0], [1]], dtype=torch.long))


def test_contact_edge_index_deterministic_ordering() -> None:
    """Edge columns are sorted by ``(i, j)`` with ``i < j``."""
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.1, 0.1, 0.0],
        ],
        dtype=torch.float64,
    )
    edges = contact_edge_index(positions, cutoff_nm=0.15)
    pairs = list(zip(edges[0].tolist(), edges[1].tolist(), strict=True))
    assert pairs == sorted(pairs)
    assert all(i < j for i, j in pairs)


def test_cutoff_and_shape_validation() -> None:
    """Non-positive / oversized cutoffs and bad shapes raise clearly."""
    positions = _line_trimer_positions_nm()
    with pytest.raises(ValueError, match="> 0 nm"):
        contact_edge_index(positions, cutoff_nm=0.0)
    with pytest.raises(ValueError, match="> 0 nm"):
        contact_edge_index(positions, cutoff_nm=-0.1)
    with pytest.raises(ValueError, match="check units"):
        contact_edge_index(positions, cutoff_nm=10.0)  # likely Å mistaken for nm
    with pytest.raises(ValueError, match="\\(num_atoms, 3\\)"):
        contact_edge_index(torch.zeros(3, 2), cutoff_nm=0.5)
    with pytest.raises(ValueError, match="residue_ids is required"):
        contact_edge_index(positions, cutoff_nm=0.5, granularity="residue")
    with pytest.raises(ValueError, match="must be None when granularity='atom'"):
        contact_edge_index(
            positions,
            cutoff_nm=0.5,
            granularity="atom",
            residue_ids=torch.zeros(3, dtype=torch.long),
        )
