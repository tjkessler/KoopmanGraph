"""Tests for optional Tier B E3EquivariantEncoder (TASK-1852)."""

from __future__ import annotations

import importlib

import pytest
import torch
from tests.helpers import REPO_ROOT
from torch_geometric.data import Data

import koopman_graph
from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.model import GraphKoopmanModel
from koopman_graph.nn import E3EquivariantEncoder, GNNDecoder

_MODULE_PATH = REPO_ROOT / "src" / "koopman_graph" / "nn" / "equivariant.py"

# Scalar latents of an E(3)-equivariant network are algebraically invariant
# under rotations of pos; float32 residuals are from matmul / SH evaluation.
_INVARIANT_ATOL = 1e-5
_INVARIANT_RTOL = 1e-4


def _square_graph(*, channels: int = 2) -> Data:
    """Four-node square with 3-D coordinates and bidirectional edges."""
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]],
        dtype=torch.long,
    )
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    x = torch.randn(4, channels)
    return Data(x=x, edge_index=edge_index, pos=pos)


def _random_rotation(*, seed: int = 0) -> torch.Tensor:
    """Haar-ish random SO(3) rotation via QR of a Gaussian matrix."""
    generator = torch.Generator().manual_seed(seed)
    raw = torch.randn(3, 3, generator=generator)
    q, r = torch.linalg.qr(raw)
    d = torch.diag(torch.sign(torch.diag(r)))
    q = q @ d
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def _pos_sequence(
    *,
    num_timesteps: int = 4,
    channels: int = 2,
) -> GraphSnapshotSequence:
    """Sequence sharing topology/pos with varying node features."""
    template = _square_graph(channels=channels)
    snapshots = [
        Data(
            x=torch.randn(4, channels),
            edge_index=template.edge_index.clone(),
            pos=template.pos.clone(),
        )
        for _ in range(num_timesteps)
    ]
    return GraphSnapshotSequence(snapshots)


def test_exports_from_nn_and_root() -> None:
    """Encoder is exported from ``nn`` and the root façade."""
    assert "E3EquivariantEncoder" in koopman_graph.nn.__all__
    assert "E3EquivariantEncoder" in koopman_graph.__all__
    assert koopman_graph.E3EquivariantEncoder is E3EquivariantEncoder


def test_import_e3nn_missing_raises_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing e3nn raises an actionable ``koopman-graph[equivariance]`` hint."""
    import koopman_graph.nn.equivariant as mod

    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None) -> object:
        if name == "e3nn" or name.startswith("e3nn."):
            raise ImportError("simulated missing e3nn")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    with pytest.raises(ImportError, match="koopman-graph\\[equivariance\\]"):
        mod._import_e3nn_modules()
    with pytest.raises(ImportError, match="koopman-graph\\[equivariance\\]"):
        E3EquivariantEncoder(2, 4, 3)


def test_module_importable_without_constructing_e3nn() -> None:
    """Loading equivariant.py does not require e3nn until construction."""
    assert _MODULE_PATH.is_file()
    assert E3EquivariantEncoder.__name__ == "E3EquivariantEncoder"


def test_forward_shape_and_rotation_invariance() -> None:
    """Forward shape is ``(N, latent_dim)``; SO(3) leaves scalars invariant."""
    pytest.importorskip("e3nn")
    data = _square_graph(channels=2)
    encoder = E3EquivariantEncoder(
        in_channels=2,
        hidden_channels=4,
        latent_dim=3,
        num_layers=2,
        lmax=1,
    )
    encoder.eval()
    with torch.no_grad():
        z = encoder(data)
        rotated = data.clone()
        rotated.pos = data.pos @ _random_rotation(seed=7).T
        z_rot = encoder(rotated)
    assert z.shape == (4, 3)
    assert torch.allclose(z, z_rot, atol=_INVARIANT_ATOL, rtol=_INVARIANT_RTOL)


def test_encoder_rejects_missing_x_and_edge_index() -> None:
    """``E3EquivariantEncoder`` requires ``data.x`` and ``edge_index``."""
    pytest.importorskip("e3nn")
    encoder = E3EquivariantEncoder(2, 4, 3)
    data = _square_graph()
    data.x = None
    with pytest.raises(ValueError, match="data.x is required"):
        encoder(data)
    data = _square_graph()
    data.edge_index = None
    with pytest.raises(ValueError, match="data.edge_index is required"):
        encoder(data)


def test_encoder_rejects_nonfinite_pos_and_x_pos_mismatch() -> None:
    """Non-finite ``pos`` and ``x``/``pos`` node-count mismatch raise."""
    pytest.importorskip("e3nn")
    encoder = E3EquivariantEncoder(2, 4, 3)
    data = _square_graph()
    data.pos = data.pos.clone()
    data.pos[0, 0] = float("inf")
    with pytest.raises(ValueError, match="data.pos must contain only finite"):
        encoder(data)
    data = _square_graph()
    data.x = torch.randn(3, 2)
    with pytest.raises(ValueError, match="node counts differ"):
        encoder(data)


def test_encoder_rejects_bad_x_shape_and_negative_lmax() -> None:
    """Wrong ``data.x`` width and ``lmax < 0`` raise clearly."""
    pytest.importorskip("e3nn")
    data = _square_graph(channels=3)
    encoder = E3EquivariantEncoder(2, 4, 3)
    with pytest.raises(ValueError, match="Expected data.x with shape"):
        encoder(data)
    with pytest.raises(ValueError, match="lmax must be >= 0"):
        E3EquivariantEncoder(2, 4, 3, lmax=-1)


def test_missing_pos_and_bad_pos_dim_raise() -> None:
    """Missing or non-3-D ``pos`` raises ``ValueError``."""
    pytest.importorskip("e3nn")
    encoder = E3EquivariantEncoder(2, 4, 3)
    data = _square_graph()
    del data.pos
    with pytest.raises(ValueError, match="data.pos"):
        encoder(data)
    data2 = _square_graph()
    data2.pos = data2.pos[:, :2]
    with pytest.raises(ValueError, match="\\(num_nodes, 3\\)"):
        encoder(data2)


def test_tensor_forward_raises() -> None:
    """Tensor-only forward is unsupported."""
    pytest.importorskip("e3nn")
    encoder = E3EquivariantEncoder(2, 4, 3)
    with pytest.raises(ValueError, match="Data"):
        encoder(torch.randn(4, 2), torch.zeros(2, 0, dtype=torch.long))


def test_model_fit_predict_smoke() -> None:
    """Short fit/predict smoke with linear graph Koopman operator."""
    pytest.importorskip("e3nn")
    sequence = _pos_sequence(num_timesteps=4, channels=2)
    model = GraphKoopmanModel(
        encoder=E3EquivariantEncoder(
            in_channels=2,
            hidden_channels=4,
            latent_dim=3,
            num_layers=2,
        ),
        decoder=GNNDecoder(latent_dim=3, hidden_channels=4, out_channels=2),
        latent_dim=3,
        time_step=1.0,
        koopman="graph",
    )
    history = model.fit(sequence, epochs=2)
    assert len(history.loss) == 2
    model.eval()
    with torch.no_grad():
        pred = model.predict(sequence[0], steps=2)
    assert len(pred) == 2
    assert pred[0].x.shape == (4, 2)
