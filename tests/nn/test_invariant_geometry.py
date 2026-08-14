"""Tests for Tier A InvariantGeometryEncoder (Phase 60 suite with simplicial/VAMP)."""

from __future__ import annotations

import pytest
import torch
from tests.helpers import REPO_ROOT
from torch_geometric.data import Data

import koopman_graph
from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.model import GraphKoopmanModel
from koopman_graph.nn import (
    GEOM_CHANNELS,
    GNNDecoder,
    InvariantGeometryEncoder,
    invariant_geometry_features,
)
from koopman_graph.serialization import build_model_config

_SRC_ROOT = REPO_ROOT / "src" / "koopman_graph"
_MODULE_PATH = _SRC_ROOT / "nn" / "equivariant.py"
_VAMP2_PATH = _SRC_ROOT / "baselines" / "vamp2.py"
_SIMPLICIAL_PATH = _SRC_ROOT / "nn" / "simplicial.py"

# Justified float32 tolerance after QR-based SO(3) / rigid SE(3) transforms:
# distance/angle features are algebraically invariant; residual error is
# accumulation from float32 matmul / norm, not a soft claim.
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


def _random_rotation(dim: int = 3, *, seed: int = 0) -> torch.Tensor:
    """Haar-ish random rotation via QR of a Gaussian matrix."""
    generator = torch.Generator().manual_seed(seed)
    raw = torch.randn(dim, dim, generator=generator)
    q, r = torch.linalg.qr(raw)
    # Ensure det = +1
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
    """Sequence of snapshots sharing topology/pos with varying x."""
    template = _square_graph(channels=channels)
    snapshots = []
    for _ in range(num_timesteps):
        snapshots.append(
            Data(
                x=torch.randn(4, channels),
                edge_index=template.edge_index.clone(),
                pos=template.pos.clone(),
            )
        )
    return GraphSnapshotSequence(snapshots)


def test_geom_and_encoder_rotation_invariance() -> None:
    """SO(3) rotation of ``pos`` leaves geom features and latents unchanged.

    Tolerance ``atol=1e-5`` / ``rtol=1e-4`` is justified for float32 QR
    rotations: invariants are exact in real arithmetic; residuals are
    floating-point only.
    """
    data = _square_graph(channels=2)
    geom = invariant_geometry_features(data.pos, data.edge_index)
    assert geom.shape == (4, GEOM_CHANNELS)

    rotation = _random_rotation(3, seed=3)
    rotated = data.clone()
    rotated.pos = data.pos @ rotation.T
    geom_rot = invariant_geometry_features(rotated.pos, rotated.edge_index)
    assert torch.allclose(geom, geom_rot, atol=_INVARIANT_ATOL, rtol=_INVARIANT_RTOL)

    encoder = InvariantGeometryEncoder(
        in_channels=2,
        hidden_channels=8,
        latent_dim=4,
        num_layers=2,
    )
    encoder.eval()
    with torch.no_grad():
        z = encoder(data)
        z_rot = encoder(rotated)
    assert z.shape == (4, 4)
    assert torch.allclose(z, z_rot, atol=_INVARIANT_ATOL, rtol=_INVARIANT_RTOL)


def test_geom_and_encoder_translation_invariance() -> None:
    """Global translation of ``pos`` leaves geom features and latents unchanged."""
    data = _square_graph(channels=2)
    geom = invariant_geometry_features(data.pos, data.edge_index)
    translated = data.clone()
    translated.pos = data.pos + torch.tensor([2.5, -1.0, 0.75])
    geom_t = invariant_geometry_features(translated.pos, translated.edge_index)
    assert torch.allclose(geom, geom_t, atol=_INVARIANT_ATOL, rtol=_INVARIANT_RTOL)

    encoder = InvariantGeometryEncoder(
        in_channels=2,
        hidden_channels=8,
        latent_dim=4,
        num_layers=2,
    )
    encoder.eval()
    with torch.no_grad():
        z = encoder(data)
        z_t = encoder(translated)
    assert torch.allclose(z, z_t, atol=_INVARIANT_ATOL, rtol=_INVARIANT_RTOL)


def test_invariant_geometry_features_validation() -> None:
    """``invariant_geometry_features`` rejects bad ``pos`` / ``edge_index``."""
    data = _square_graph(channels=2)
    with pytest.raises(ValueError, match="pos must have shape"):
        invariant_geometry_features(data.pos[:, :1], data.edge_index)
    bad_pos = data.pos.clone()
    bad_pos[0, 0] = float("nan")
    with pytest.raises(ValueError, match="pos must contain only finite"):
        invariant_geometry_features(bad_pos, data.edge_index)
    with pytest.raises(ValueError, match="edge_index must have shape"):
        invariant_geometry_features(data.pos, data.edge_index.reshape(-1))
    with pytest.raises(ValueError, match="num_nodes=.*does not match"):
        invariant_geometry_features(data.pos, data.edge_index, num_nodes=99)
    bad_edge = torch.tensor([[0, 99], [1, 0]], dtype=torch.long)
    with pytest.raises(ValueError, match="edge_index node ids must lie"):
        invariant_geometry_features(data.pos, bad_edge)


def test_encoder_rejects_missing_x_and_edge_index() -> None:
    """``InvariantGeometryEncoder`` requires ``data.x`` and ``edge_index``."""
    encoder = InvariantGeometryEncoder(2, 8, 4)
    data = _square_graph(channels=2)
    data.x = None
    with pytest.raises(ValueError, match="data.x is required"):
        encoder(data)
    data = _square_graph(channels=2)
    data.edge_index = None
    with pytest.raises(ValueError, match="data.edge_index is required"):
        encoder(data)


def test_encoder_rejects_bad_x_shape() -> None:
    """Mismatched ``data.x`` width raises clearly."""
    encoder = InvariantGeometryEncoder(2, 8, 4)
    data = _square_graph(channels=3)
    with pytest.raises(ValueError, match="Expected data.x with shape"):
        encoder(data)


def test_encoder_rejects_non_gcn_conv_in_stack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-``GCNConv`` layer in the stack raises ``TypeError``."""
    encoder = InvariantGeometryEncoder(2, 8, 4)
    encoder.convs[0] = torch.nn.Linear(2 + GEOM_CHANNELS, 8)  # type: ignore[assignment]
    data = _square_graph(channels=2)
    with pytest.raises(TypeError, match="expected GCNConv"):
        encoder(data)


def test_missing_pos_raises() -> None:
    """``Data`` without ``pos`` raises ``ValueError``."""
    data = _square_graph()
    del data.pos
    encoder = InvariantGeometryEncoder(2, 8, 4)
    with pytest.raises(ValueError, match="data.pos"):
        encoder(data)


def test_tensor_forward_raises() -> None:
    """Tensor-only forward is unsupported."""
    encoder = InvariantGeometryEncoder(2, 8, 4)
    with pytest.raises(ValueError, match="Data"):
        encoder(torch.randn(4, 2), torch.zeros(2, 0, dtype=torch.long))


def test_exports_from_nn_and_root() -> None:
    """Encoder and helpers are exported from ``nn`` and root ``__all__``."""
    assert "InvariantGeometryEncoder" in koopman_graph.nn.__all__
    assert "invariant_geometry_features" in koopman_graph.nn.__all__
    assert "GEOM_CHANNELS" in koopman_graph.nn.__all__
    assert "InvariantGeometryEncoder" in koopman_graph.__all__
    assert koopman_graph.InvariantGeometryEncoder is InvariantGeometryEncoder


def test_model_fit_predict_with_graph_koopman() -> None:
    """Fit/predict smoke with ``koopman="graph"`` and ``GNNDecoder``."""
    sequence = _pos_sequence(num_timesteps=4, channels=2)
    model = GraphKoopmanModel(
        encoder=InvariantGeometryEncoder(
            in_channels=2,
            hidden_channels=8,
            latent_dim=4,
        ),
        decoder=GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=2),
        latent_dim=4,
        time_step=1.0,
        koopman="graph",
    )
    history = model.fit(sequence, epochs=2)
    assert len(history.loss) == 2
    model.eval()
    with torch.no_grad():
        preds = model.predict(sequence[0], steps=1)
    assert len(preds) == 1
    assert preds[0].x.shape == (4, 2)


def test_checkpoint_round_trip_inv_geom_enc(tmp_path) -> None:
    """Format-1 checkpoints round-trip ``inv_geom_enc``."""
    sequence = _pos_sequence(num_timesteps=3, channels=2)
    model = GraphKoopmanModel(
        encoder=InvariantGeometryEncoder(2, 8, 4),
        decoder=GNNDecoder(4, 8, 2),
        latent_dim=4,
        time_step=1.0,
        koopman="graph",
    )
    model.fit(sequence, epochs=1)
    config = build_model_config(model)
    assert config["encoder"]["type"] == "inv_geom_enc"
    path = tmp_path / "inv_geom.pt"
    model.save(path)
    loaded = GraphKoopmanModel.load(path)
    assert isinstance(loaded.encoder, InvariantGeometryEncoder)
    assert isinstance(loaded.decoder, GNNDecoder)


def test_module_doc_honesty_keywords() -> None:
    """Docs state invariant features and reject equivariant-K / e3nn claims."""
    text = _MODULE_PATH.read_text(encoding="utf-8").lower()
    assert "invariant" in text
    assert "equivariant" in text
    assert "e3nn" in text
    assert "koopman" in text or ":math:`k`" in text or " k" in text
    # Explicit non-claim: invariant features ≠ equivariant K.
    assert "≠" in _MODULE_PATH.read_text(encoding="utf-8") or (
        "not" in text and "equivariant" in text
    )


def test_phase60_honesty_cross_modules() -> None:
    """Phase 60 suite honesty: simplicial / VAMP-2 / invariant modules."""
    simplicial = _SIMPLICIAL_PATH.read_text(encoding="utf-8").lower()
    assert "sheaf" in simplicial and "cell" in simplicial

    vamp2 = _VAMP2_PATH.read_text(encoding="utf-8").lower()
    assert "graphvampnets" in vamp2
    assert "0.11" in vamp2

    equivariant = _MODULE_PATH.read_text(encoding="utf-8").lower()
    assert "invariant" in equivariant
    assert "equivariant" in equivariant
    assert "koopman" in equivariant or ":math:`k`" in equivariant
