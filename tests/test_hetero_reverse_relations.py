"""Factory reverse-relation synthesis + materialize + checkpoint metadata."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch_geometric.data import HeteroData

from koopman_graph.data import HeteroGraphSnapshotSequence
from koopman_graph.graph_utils import (
    materialize_reverse_relation_edges,
    synthesize_reverse_edge_types,
)
from koopman_graph.model import GraphKoopmanModel
from koopman_graph.nn import GNNDecoder, GNNEncoder, RelGraphDecoder, RelGraphEncoder
from koopman_graph.operators import HeteroGraphKoopmanOperator
from koopman_graph.serialization import build_model_config, load_checkpoint

_FORWARD_EDGE_TYPES = (
    ("node", "r1", "node"),
    ("node", "r2", "node"),
)
_EXPANDED_EDGE_TYPES = synthesize_reverse_edge_types(_FORWARD_EDGE_TYPES)


def _forward_snapshot(*, seed: int = 0) -> HeteroData:
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


def _peers(
    *,
    num_relations: int,
    edge_types=None,
) -> tuple[RelGraphEncoder, RelGraphDecoder]:
    return (
        RelGraphEncoder(
            3,
            hidden_channels=8,
            latent_dim=4,
            num_relations=num_relations,
            num_layers=1,
            edge_types=edge_types,
        ),
        RelGraphDecoder(
            latent_dim=4,
            hidden_channels=8,
            out_channels=3,
            num_relations=num_relations,
            num_layers=1,
            edge_types=edge_types,
        ),
    )


def test_default_off_preserves_relation_cardinality() -> None:
    """Default False keeps forward |R| and stores synthesize flag False."""
    encoder, decoder = _peers(num_relations=2)
    model = GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=4,
        time_step=1.0,
        koopman="hetero_graph",
        koopman_edge_types=_FORWARD_EDGE_TYPES,
    )
    assert model.synthesize_reverse_relations is False
    assert isinstance(model.koopman, HeteroGraphKoopmanOperator)
    assert model.koopman.num_relations == 2
    assert model.encoder.num_relations == 2
    assert tuple(model.koopman.edge_types) == _FORWARD_EDGE_TYPES
    config = build_model_config(model)
    assert config["synthesize_reverse_relations"] is False
    assert config["edge_types"] == [list(t) for t in _FORWARD_EDGE_TYPES]


def test_true_expands_banks_and_rebuilds_forward_sized_peers() -> None:
    """True expands schema and rebuilds peers sized for forward |R|."""
    encoder, decoder = _peers(num_relations=2)
    model = GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=4,
        time_step=1.0,
        koopman="hetero_graph",
        koopman_edge_types=_FORWARD_EDGE_TYPES,
        koopman_synthesize_reverse_relations=True,
    )
    assert model.synthesize_reverse_relations is True
    assert isinstance(model.koopman, HeteroGraphKoopmanOperator)
    assert model.koopman.num_relations == 4
    assert model.encoder.num_relations == 4
    assert model.decoder.num_relations == 4
    assert tuple(model.koopman.edge_types) == _EXPANDED_EDGE_TYPES
    assert tuple(model.encoder.edge_types) == _EXPANDED_EDGE_TYPES
    assert tuple(model.decoder.edge_types) == _EXPANDED_EDGE_TYPES


def test_true_accepts_peers_already_sized_for_expanded_schema() -> None:
    """Peers already matching expanded |R| are kept (not rebuilt)."""
    encoder, decoder = _peers(
        num_relations=len(_EXPANDED_EDGE_TYPES),
        edge_types=_EXPANDED_EDGE_TYPES,
    )
    model = GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=4,
        time_step=1.0,
        koopman="hetero_graph",
        koopman_edge_types=_FORWARD_EDGE_TYPES,
        koopman_synthesize_reverse_relations=True,
    )
    assert model.encoder is encoder
    assert model.decoder is decoder
    assert model.koopman.num_relations == 4


def test_true_rejects_mismatched_peer_cardinality() -> None:
    """Peers sized for neither forward nor expanded |R| raise."""
    encoder, decoder = _peers(num_relations=3)
    with pytest.raises(ValueError, match="num_relations equal to forward"):
        GraphKoopmanModel(
            encoder=encoder,
            decoder=decoder,
            latent_dim=4,
            time_step=1.0,
            koopman="hetero_graph",
            koopman_edge_types=_FORWARD_EDGE_TYPES,
            koopman_synthesize_reverse_relations=True,
        )


def test_true_requires_edge_types() -> None:
    """True without koopman_edge_types raises a clear error."""
    encoder, decoder = _peers(num_relations=2)
    with pytest.raises(ValueError, match="koopman_edge_types"):
        GraphKoopmanModel(
            encoder=encoder,
            decoder=decoder,
            latent_dim=4,
            time_step=1.0,
            koopman="hetero_graph",
            koopman_synthesize_reverse_relations=True,
        )


def test_true_rejects_non_hetero() -> None:
    """True on a homogeneous model raises."""
    with pytest.raises(ValueError, match="hetero_graph"):
        GraphKoopmanModel(
            encoder=GNNEncoder(in_channels=3, hidden_channels=8, latent_dim=4),
            decoder=GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=3),
            latent_dim=4,
            time_step=1.0,
            koopman="graph",
            koopman_edge_types=_FORWARD_EDGE_TYPES,
            koopman_synthesize_reverse_relations=True,
        )


def test_materialize_reverse_relation_edges_flips_index() -> None:
    """Materialize fills reverse banks from flipped forward edge_index."""
    origin = _forward_snapshot()
    filled = materialize_reverse_relation_edges(origin, _EXPANDED_EDGE_TYPES)
    assert ("node", "rev_r1", "node") in filled.edge_types
    assert ("node", "rev_r2", "node") in filled.edge_types
    torch.testing.assert_close(
        filled["node", "rev_r1", "node"].edge_index,
        origin["node", "r1", "node"].edge_index.flip(0),
    )
    # Original forward banks unchanged; origin not mutated.
    assert ("node", "rev_r1", "node") not in origin.edge_types


def test_fit_predict_smoke_with_materialized_reverses() -> None:
    """Short fit + predict succeed on materialized reverse banks."""
    torch.manual_seed(0)
    encoder, decoder = _peers(num_relations=2)
    model = GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=4,
        time_step=1.0,
        koopman="hetero_graph",
        koopman_edge_types=_FORWARD_EDGE_TYPES,
        koopman_synthesize_reverse_relations=True,
    )
    sequence = HeteroGraphSnapshotSequence(
        [
            materialize_reverse_relation_edges(
                _forward_snapshot(seed=t),
                _EXPANDED_EDGE_TYPES,
            )
            for t in range(5)
        ]
    )
    history = model.fit(sequence, epochs=1, lr=1e-2)
    assert len(history.loss) == 1
    assert torch.isfinite(torch.tensor(history.loss[0]))
    model.eval()
    with torch.no_grad():
        preds = model.predict(sequence[0], steps=2)
    assert len(preds) == 2
    assert isinstance(preds[0], HeteroData)
    assert preds[0]["node"].x.shape == (4, 3)


def test_checkpoint_round_trip_preserves_synthesize_flag(tmp_path: Path) -> None:
    """Format-1 config records synthesize_reverse_relations and reloads it."""
    encoder, decoder = _peers(num_relations=2)
    model = GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=4,
        time_step=1.0,
        koopman="hetero_graph",
        koopman_edge_types=_FORWARD_EDGE_TYPES,
        koopman_synthesize_reverse_relations=True,
    )
    path = tmp_path / "rev.pt"
    model.save(path, format="legacy_pt")
    loaded = GraphKoopmanModel.load(path)
    assert loaded.synthesize_reverse_relations is True
    assert loaded.koopman.num_relations == 4
    assert [tuple(t) for t in loaded.koopman.edge_types] == list(_EXPANDED_EDGE_TYPES)

    # Absent key on load defaults to False (0.9-style payload).
    checkpoint = torch.load(path, weights_only=False)
    del checkpoint["config"]["synthesize_reverse_relations"]
    legacy_path = tmp_path / "legacy.pt"
    torch.save(checkpoint, legacy_path)
    legacy = load_checkpoint(legacy_path)
    assert legacy.synthesize_reverse_relations is False
    # Expanded edge_types remain in the payload; banks stay sized to them.
    assert legacy.koopman.num_relations == 4
