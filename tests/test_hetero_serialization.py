"""Format-1 checkpoint round-trip for ``koopman='hetero_graph'``."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data, HeteroData

from koopman_graph.model import GraphKoopmanModel
from koopman_graph.nn import GNNDecoder, GNNEncoder, RelGraphDecoder, RelGraphEncoder
from koopman_graph.operators import HeteroGraphKoopmanOperator
from koopman_graph.serialization import (
    FORMAT_VERSION,
    build_checkpoint,
    build_model_config,
    load_checkpoint,
    reconstruct_model,
)


def _multiplex_snapshot(*, num_nodes: int = 4, in_channels: int = 3) -> HeteroData:
    data = HeteroData()
    data["node"].x = torch.randn(num_nodes, in_channels)
    data["node", "r1", "node"].edge_index = torch.tensor(
        [[0, 1, 2], [1, 2, 0]],
        dtype=torch.long,
    )
    data["node", "r2", "node"].edge_index = torch.tensor(
        [[0, 2], [2, 3]],
        dtype=torch.long,
    )
    return data


def _hetero_model(
    *,
    latent_dim: int = 4,
    in_channels: int = 3,
    num_relations: int = 2,
    edge_types: list[list[str]] | None = None,
) -> GraphKoopmanModel:
    return GraphKoopmanModel(
        encoder=RelGraphEncoder(
            in_channels,
            hidden_channels=8,
            latent_dim=latent_dim,
            num_relations=num_relations,
            num_layers=1,
        ),
        decoder=RelGraphDecoder(
            latent_dim=latent_dim,
            hidden_channels=8,
            out_channels=in_channels,
            num_relations=num_relations,
            num_layers=1,
        ),
        latent_dim=latent_dim,
        time_step=1.0,
        koopman="hetero_graph",
        koopman_edge_types=edge_types,
    )


def test_hetero_format1_round_trip_preserves_factors_and_predictions(
    tmp_path: Path,
) -> None:
    """Save/load restores type metadata, relation factors, and predict."""
    torch.manual_seed(0)
    edge_types = [["node", "r1", "node"], ["node", "r2", "node"]]
    model = _hetero_model(edge_types=edge_types)
    assert isinstance(model.koopman, HeteroGraphKoopmanOperator)
    k_self = model.koopman.K_self.detach().clone()
    k_relations = [matrix.detach().clone() for matrix in model.koopman.K_relations]
    origin = _multiplex_snapshot()
    model.eval()
    with torch.no_grad():
        before = model.predict(origin, steps=2)

    checkpoint = build_checkpoint(model)
    assert checkpoint["format_version"] == FORMAT_VERSION == 1
    assert set(checkpoint) == {
        "format_version",
        "package_version",
        "config",
        "state_dict",
    }
    config = checkpoint["config"]
    assert config["koopman_kind"] == "hetero_graph"
    assert config["node_types"] == ["node"]
    assert config["edge_types"] == edge_types
    assert config["relation_tying"] == "independent"
    assert config["basis_size"] is None
    assert config["relation_normalization"] == "rgcn_in_degree"
    assert config["adjacency"] is None
    assert any(
        "koopman._rel.node__r1__node." in key for key in checkpoint["state_dict"]
    )
    assert any(
        "koopman._rel.node__r2__node." in key for key in checkpoint["state_dict"]
    )
    assert not any("koopman._relations." in key for key in checkpoint["state_dict"])

    path = tmp_path / "hetero.pt"
    model.save(path)
    loaded = GraphKoopmanModel.load(path)
    assert loaded.koopman_kind == "hetero_graph"
    assert isinstance(loaded.koopman, HeteroGraphKoopmanOperator)
    assert list(loaded.koopman.node_types) == ["node"]
    assert [list(triple) for triple in loaded.koopman.edge_types] == edge_types
    torch.testing.assert_close(loaded.koopman.K_self, k_self)
    for original, restored in zip(k_relations, loaded.koopman.K_relations, strict=True):
        torch.testing.assert_close(restored, original)

    loaded.eval()
    with torch.no_grad():
        after = loaded.predict(origin, steps=2)
    assert len(after) == len(before)
    for pred_before, pred_after in zip(before, after, strict=True):
        torch.testing.assert_close(pred_after["node"].x, pred_before["node"].x)


def test_hetero_checkpoint_rejects_missing_edge_types(tmp_path: Path) -> None:
    """Hetero load rejects payloads missing edge_types."""
    model = _hetero_model()
    path = tmp_path / "missing_edge_types.pt"
    checkpoint = build_checkpoint(model)
    del checkpoint["config"]["edge_types"]
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="Incomplete hetero checkpoint schema"):
        load_checkpoint(path)


def test_hetero_checkpoint_rejects_missing_node_types(tmp_path: Path) -> None:
    """Hetero load rejects payloads missing node_types."""
    model = _hetero_model()
    path = tmp_path / "missing_node_types.pt"
    checkpoint = build_checkpoint(model)
    del checkpoint["config"]["node_types"]
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="Incomplete hetero checkpoint schema"):
        load_checkpoint(path)


def test_hetero_checkpoint_basis_tying_round_trip(tmp_path: Path) -> None:
    """relation_tying='basis' checkpoints restore V_b, coeffs, and predictions."""
    torch.manual_seed(3)
    model = GraphKoopmanModel(
        encoder=RelGraphEncoder(3, hidden_channels=8, latent_dim=4, num_relations=2),
        decoder=RelGraphDecoder(
            latent_dim=4,
            hidden_channels=8,
            out_channels=3,
            num_relations=2,
        ),
        latent_dim=4,
        time_step=1.0,
        koopman="hetero_graph",
        koopman_relation_tying="basis",
        koopman_basis_size=2,
    )
    assert isinstance(model.koopman, HeteroGraphKoopmanOperator)
    assert model.koopman.relation_tying == "basis"
    assert model.koopman.basis_size == 2
    coeffs = model.koopman._rel_coeff.detach().clone()
    basis = [module.K.detach().clone() for module in model.koopman._basis_modules()]
    origin = _multiplex_snapshot()
    model.eval()
    with torch.no_grad():
        before = model.predict(origin, steps=2)

    path = tmp_path / "basis.pt"
    checkpoint = build_checkpoint(model)
    assert checkpoint["config"]["relation_tying"] == "basis"
    assert checkpoint["config"]["basis_size"] == 2
    assert any("koopman._basis.b0." in key for key in checkpoint["state_dict"])
    assert any("koopman._rel_coeff" in key for key in checkpoint["state_dict"])
    assert not any("koopman._rel." in key for key in checkpoint["state_dict"])
    torch.save(checkpoint, path)

    loaded = load_checkpoint(path)
    assert isinstance(loaded.koopman, HeteroGraphKoopmanOperator)
    assert loaded.koopman.relation_tying == "basis"
    assert loaded.koopman.basis_size == 2
    assert torch.allclose(loaded.koopman._rel_coeff, coeffs)
    for left, right in zip(loaded.koopman._basis_modules(), basis, strict=True):
        assert torch.allclose(left.K, right)
    loaded.eval()
    with torch.no_grad():
        after = loaded.predict(origin, steps=2)
    for pred_before, pred_after in zip(before, after, strict=True):
        torch.testing.assert_close(pred_after["node"].x, pred_before["node"].x)


def test_hetero_checkpoint_rejects_basis_without_size(tmp_path: Path) -> None:
    """basis tying without a positive basis_size is rejected at load."""
    model = _hetero_model()
    path = tmp_path / "basis_bad.pt"
    checkpoint = build_checkpoint(model)
    checkpoint["config"]["relation_tying"] = "basis"
    checkpoint["config"]["basis_size"] = None
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="basis_size must be a positive int"):
        load_checkpoint(path)


def test_homogeneous_load_ignores_extraneous_hetero_keys(tmp_path: Path) -> None:
    """Homogeneous checkpoints with stray hetero keys still load."""
    model = GraphKoopmanModel(
        encoder=GNNEncoder(in_channels=3, hidden_channels=8, latent_dim=4),
        decoder=GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=3),
        latent_dim=4,
        time_step=1.0,
    )
    data = Data(
        x=torch.randn(3, 3),
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
    )
    model.eval()
    with torch.no_grad():
        before = model.predict(data, steps=2)

    path = tmp_path / "homo_extra.pt"
    checkpoint = build_checkpoint(model)
    checkpoint["config"]["node_types"] = ["node"]
    checkpoint["config"]["edge_types"] = [["node", "r0", "node"]]
    checkpoint["config"]["relation_tying"] = "independent"
    checkpoint["config"]["basis_size"] = None
    checkpoint["config"]["relation_normalization"] = "rgcn_in_degree"
    torch.save(checkpoint, path)

    loaded = load_checkpoint(path)
    assert loaded.koopman_kind == "pernode"
    assert "node_types" not in build_model_config(loaded)
    loaded.eval()
    with torch.no_grad():
        after = loaded.predict(data, steps=2)
    for pred_before, pred_after in zip(before, after, strict=True):
        torch.testing.assert_close(pred_after.x, pred_before.x)


def test_default_multiplex_edge_types_in_config() -> None:
    """Default multiplex metadata uses synthetic ``r{i}`` relation names."""
    model = _hetero_model()
    config = build_model_config(model)
    assert config["node_types"] == ["node"]
    assert config["edge_types"] == [["node", "r0", "node"], ["node", "r1", "node"]]
    reconstructed = reconstruct_model(config)
    assert isinstance(reconstructed.koopman, HeteroGraphKoopmanOperator)
    assert reconstructed.koopman.edge_types == (
        ("node", "r0", "node"),
        ("node", "r1", "node"),
    )
