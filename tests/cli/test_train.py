"""Coverage and error-path tests for :mod:`koopman_graph.cli`."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch_geometric.data import Data

import koopman_graph.cli.config as config_mod
import koopman_graph.cli.train as train_mod
from koopman_graph.data import GraphSnapshotSequence


def _minimal_model_config(*, encoder: str = "gcn") -> dict[str, object]:
    """Return the smallest valid CLI model config."""
    return {
        "encoder": encoder,
        "in_channels": 2,
        "hidden_channels": 3,
        "latent_dim": 2,
    }


def _snapshot_sequence() -> GraphSnapshotSequence:
    """Return a deterministic two-snapshot homogeneous sequence."""
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    snapshots = [
        Data(x=torch.full((2, 2), float(step)), edge_index=edge_index)
        for step in range(2)
    ]
    return GraphSnapshotSequence(snapshots)


@pytest.mark.parametrize("value", [True, 1.2, "1"])
def test_train_require_int_rejects_non_integer_values(value: object) -> None:
    """The CLI integer validator rejects bool and non-integer values."""
    with pytest.raises(config_mod.ConfigError, match="model.value must be an int"):
        train_mod._require_int(value, path="model.value")


@pytest.mark.parametrize("value", [False, "1.0", None])
def test_train_require_float_rejects_non_numeric_values(value: object) -> None:
    """The CLI float validator rejects bool and non-numeric values."""
    with pytest.raises(config_mod.ConfigError, match="must be a number"):
        train_mod._require_float(value, path="fit.lr")


def test_train_require_float_accepts_integer() -> None:
    """The CLI float validator converts an integer to float."""
    assert train_mod._require_float(2, path="fit.lr") == pytest.approx(
        2.0, rel=0.0, abs=0.0
    )


@pytest.mark.parametrize(
    ("model_config", "message"),
    [
        ({}, "model.encoder is required"),
        ({"encoder": "transformer"}, "Unsupported model.encoder"),
        ({"encoder": "gcn", "decoder": 1}, "model.decoder must be a string"),
        (
            {"encoder": "gcn", "decoder": "gat"},
            "must match model.encoder",
        ),
        ({"encoder": "gcn"}, "Missing required key"),
        (
            {
                "encoder": "gcn",
                "in_channels": 0,
                "hidden_channels": 2,
                "latent_dim": 2,
            },
            "must be positive",
        ),
    ],
)
def test_train_encoder_decoder_rejects_invalid_configs(
    model_config: dict[str, object],
    message: str,
) -> None:
    """Encoder/decoder config guards raise actionable errors."""
    with pytest.raises(config_mod.ConfigError, match=message):
        train_mod._build_encoder_decoder(model_config)


@pytest.mark.parametrize("kind", ["gat", "sage"])
def test_train_builds_non_gcn_encoder_decoder_branches(kind: str) -> None:
    """GAT and GraphSAGE CLI model branches construct matched peers."""
    encoder, decoder, latent_dim = train_mod._build_encoder_decoder(
        _minimal_model_config(encoder=kind)
    )
    assert encoder is not None
    assert decoder is not None
    assert latent_dim == 2


def test_train_model_defaults_time_step() -> None:
    """An omitted time step is forwarded to the model as 1.0."""
    sentinel = object()
    with patch.object(
        train_mod,
        "GraphKoopmanModel",
        return_value=sentinel,
    ) as model_cls:
        result = train_mod.build_model_from_config(_minimal_model_config())
    assert result is sentinel
    assert model_cls.call_args.kwargs["time_step"] == pytest.approx(
        1.0, rel=0.0, abs=0.0
    )


def test_train_synthetic_sequence_rejects_invalid_dimensions() -> None:
    """Synthetic path data enforces its node, time, and feature minima."""
    with pytest.raises(config_mod.ConfigError, match="num_nodes>=2"):
        train_mod._build_synthetic_path_sequence(
            {"num_nodes": 1, "num_timesteps": 2, "feature_dim": 1}
        )


def test_train_cached_sequence_loads_supported_payloads(tmp_path: Path) -> None:
    """Cached loaders accept sequence objects and non-empty Data lists."""
    path = tmp_path / "sequence.pt"
    sequence = _snapshot_sequence()
    torch.save(sequence, path)
    loaded_sequence = train_mod.load_cached_sequence({"path": path})
    assert len(loaded_sequence) == 2

    torch.save(list(sequence), path)
    loaded_list = train_mod.load_cached_sequence({"path": path})
    assert len(loaded_list) == 2


def test_train_cached_sequence_rejects_missing_and_invalid_payloads(
    tmp_path: Path,
) -> None:
    """Cached loaders reject missing paths and unsupported payload shapes."""
    missing = tmp_path / "missing.pt"
    with pytest.raises(config_mod.ConfigError, match="not found"):
        train_mod.load_cached_sequence({"path": missing})

    invalid = tmp_path / "invalid.pt"
    torch.save({"not": "a sequence"}, invalid)
    with pytest.raises(config_mod.ConfigError, match="must be a GraphSnapshotSequence"):
        train_mod.load_cached_sequence({"path": invalid})


def test_train_sequence_dispatches_cached_and_rejects_unknown_kind() -> None:
    """Data-kind dispatch covers the cached and unknown branches."""
    sentinel = _snapshot_sequence()
    with patch.object(train_mod, "load_cached_sequence", return_value=sentinel):
        result = train_mod.build_sequence_from_config(
            {"kind": "cached_sequence", "path": "ignored.pt"}
        )
    assert result is sentinel

    with pytest.raises(config_mod.ConfigError, match="Unsupported data.kind"):
        train_mod.build_sequence_from_config({"kind": "remote_code"})


def test_train_rejects_unknown_checkpoint_format() -> None:
    """Training refuses checkpoint formats outside the explicit allowlist."""
    fake_model = MagicMock()
    with (
        patch.object(train_mod, "build_model_from_config", return_value=fake_model),
        patch.object(
            train_mod,
            "build_sequence_from_config",
            return_value=_snapshot_sequence(),
        ),
        pytest.raises(config_mod.ConfigError, match="checkpoint.format"),
    ):
        train_mod.run_train(
            {
                "model": {},
                "data": {},
                "checkpoint": {"format": "unsafe_pickle"},
            }
        )
