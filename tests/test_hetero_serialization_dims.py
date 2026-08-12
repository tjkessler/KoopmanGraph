"""Additive ``latent_dims`` checkpoint round-trip (TASK-1818, Q1=A)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch_geometric.data import HeteroData

from koopman_graph.model import GraphKoopmanModel
from koopman_graph.nn import RelGraphDecoder, RelGraphEncoder
from koopman_graph.operators import HeteroGraphKoopmanOperator
from koopman_graph.serialization import (
    FORMAT_VERSION,
    build_checkpoint,
    load_checkpoint,
)

NODE_TYPES = ("a", "b")
EDGE_TYPES = (
    ("a", "to_b", "b"),
    ("b", "to_a", "a"),
)
FEATURE_DIMS = {"a": 2, "b": 2}
NUM_NODES = {"a": 2, "b": 3}
LATENT_DIMS = {"a": 2, "b": 3}
SHARED_D = 4


def _typed_snapshot() -> HeteroData:
    """Tiny typed snapshot matching the rectangular fixture."""
    snapshot = HeteroData()
    snapshot["a"].x = torch.randn(NUM_NODES["a"], FEATURE_DIMS["a"])
    snapshot["b"].x = torch.randn(NUM_NODES["b"], FEATURE_DIMS["b"])
    snapshot["a", "to_b", "b"].edge_index = torch.tensor(
        [[0, 1], [0, 2]],
        dtype=torch.long,
    )
    snapshot["b", "to_a", "a"].edge_index = torch.tensor(
        [[0, 1], [0, 1]],
        dtype=torch.long,
    )
    return snapshot


def _shared_d_model() -> GraphKoopmanModel:
    """Typed shared-d hetero model (no latent_dims)."""
    return GraphKoopmanModel(
        encoder=RelGraphEncoder(
            FEATURE_DIMS,
            hidden_channels=4,
            latent_dim=SHARED_D,
            num_relations=len(EDGE_TYPES),
            num_layers=1,
            node_types=NODE_TYPES,
            edge_types=EDGE_TYPES,
        ),
        decoder=RelGraphDecoder(
            latent_dim=SHARED_D,
            hidden_channels=4,
            out_channels=FEATURE_DIMS,
            num_relations=len(EDGE_TYPES),
            num_layers=1,
            node_types=NODE_TYPES,
            edge_types=EDGE_TYPES,
        ),
        latent_dim=SHARED_D,
        time_step=1.0,
        koopman="hetero_graph",
        koopman_node_types=NODE_TYPES,
        koopman_edge_types=EDGE_TYPES,
    )


def _rectangular_model() -> GraphKoopmanModel:
    """Typed rectangular hetero model with unequal d_τ."""
    return GraphKoopmanModel(
        encoder=RelGraphEncoder(
            FEATURE_DIMS,
            hidden_channels=4,
            latent_dim=SHARED_D,
            num_relations=len(EDGE_TYPES),
            num_layers=1,
            node_types=NODE_TYPES,
            edge_types=EDGE_TYPES,
        ),
        decoder=RelGraphDecoder(
            latent_dim=SHARED_D,
            hidden_channels=4,
            out_channels=FEATURE_DIMS,
            num_relations=len(EDGE_TYPES),
            num_layers=1,
            node_types=NODE_TYPES,
            edge_types=EDGE_TYPES,
        ),
        latent_dim=SHARED_D,
        time_step=1.0,
        koopman="hetero_graph",
        koopman_node_types=NODE_TYPES,
        koopman_edge_types=EDGE_TYPES,
        koopman_latent_dims=LATENT_DIMS,
    )


def test_shared_d_round_trip_omits_latent_dims(tmp_path: Path) -> None:
    """Shared-d checkpoints omit latent_dims and still load (Q1=A)."""
    model = _shared_d_model()
    checkpoint = build_checkpoint(model)
    assert checkpoint["format_version"] == FORMAT_VERSION == 1
    assert "latent_dims" not in checkpoint["config"]
    assert not model.koopman.is_rectangular

    path = tmp_path / "shared.pt"
    model.save(path)
    loaded = GraphKoopmanModel.load(path)
    assert isinstance(loaded.koopman, HeteroGraphKoopmanOperator)
    assert loaded.koopman.latent_dims is None
    assert not loaded.koopman.is_rectangular
    assert loaded.encoder.latent_dims is None


def test_rectangular_round_trip_preserves_latent_dims(tmp_path: Path) -> None:
    """Rectangular checkpoints store latent_dims and restore factors / predict."""
    torch.manual_seed(0)
    model = _rectangular_model()
    assert model.koopman.is_rectangular
    assert model.encoder.is_rectangular
    k_self_a = model.koopman.k_self_for("a").detach().clone()
    k_rel0 = model.koopman.relation_matrix(0).detach().clone()
    origin = _typed_snapshot()
    model.eval()
    with torch.no_grad():
        before = model.predict(origin, steps=1)

    checkpoint = build_checkpoint(model)
    assert checkpoint["format_version"] == FORMAT_VERSION == 1
    assert checkpoint["config"]["latent_dims"] == LATENT_DIMS
    assert any(key.startswith("koopman._rel_rect.") for key in checkpoint["state_dict"])
    assert any("encoder.type_latent." in key for key in checkpoint["state_dict"])
    assert any("decoder.type_latent_in." in key for key in checkpoint["state_dict"])

    path = tmp_path / "rect.pt"
    model.save(path)
    loaded = GraphKoopmanModel.load(path)
    assert loaded.koopman.latent_dims == LATENT_DIMS
    assert loaded.koopman.is_rectangular
    assert loaded.encoder.is_rectangular
    assert loaded.decoder.is_rectangular
    torch.testing.assert_close(loaded.koopman.k_self_for("a"), k_self_a)
    torch.testing.assert_close(loaded.koopman.relation_matrix(0), k_rel0)

    loaded.eval()
    with torch.no_grad():
        after = loaded.predict(origin, steps=1)
    for name in NODE_TYPES:
        torch.testing.assert_close(after[0][name], before[0][name])


def test_incomplete_latent_dims_rejected(tmp_path: Path) -> None:
    """Missing type keys in latent_dims raise a clear re-save error."""
    model = _rectangular_model()
    path = tmp_path / "rect.pt"
    model.save(path, format="legacy_pt")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["config"]["latent_dims"] = {"a": 2}
    broken = tmp_path / "incomplete.pt"
    torch.save(payload, broken)
    with pytest.raises(ValueError, match="latent_dims is incomplete"):
        load_checkpoint(broken)


def test_stripped_latent_dims_with_rectangular_weights_rejected(
    tmp_path: Path,
) -> None:
    """Rectangular weights without config.latent_dims are rejected."""
    model = _rectangular_model()
    path = tmp_path / "rect.pt"
    model.save(path, format="legacy_pt")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    del payload["config"]["latent_dims"]
    broken = tmp_path / "broken.pt"
    torch.save(payload, broken)
    with pytest.raises(ValueError, match="latent_dims is missing"):
        load_checkpoint(broken)


def test_mismatched_self_factor_shape_rejected(tmp_path: Path) -> None:
    """latent_dims that disagree with self-factor shapes raise clearly."""
    model = _rectangular_model()
    path = tmp_path / "rect.pt"
    model.save(path, format="legacy_pt")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["config"]["latent_dims"] = {"a": 3, "b": 3}
    broken = tmp_path / "shape_mismatch.pt"
    torch.save(payload, broken)
    with pytest.raises(ValueError, match="expects \\(3, 3\\)"):
        load_checkpoint(broken)


def test_no_format_version_bump_for_latent_dims() -> None:
    """Additive latent_dims keep FORMAT_VERSION at 1."""
    assert FORMAT_VERSION == 1
    model = _rectangular_model()
    assert build_checkpoint(model)["format_version"] == 1
