"""Coverage and error-path tests for :mod:`koopman_graph.serialization`."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

import koopman_graph.serialization as serialization_mod
from koopman_graph import GraphKoopmanModel
from koopman_graph.serialization import (
    _require_hetero_schema,
    _state_dict_has_rectangular_hetero_markers,
    _validate_hetero_latent_dims_vs_state,
)

_LATENT_DIMS = {"a": 2, "b": 3}


def _hetero_schema(**overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "latent_dim": 4,
        "node_types": ["a", "b"],
        "edge_types": [["a", "r", "b"]],
        "relation_tying": "independent",
        "basis_size": None,
        "relation_normalization": "rgcn_in_degree",
    }
    config.update(overrides)
    return config


def _safe_meta(*, format_version: int = 1) -> dict[str, object]:
    """Return valid safetensors metadata."""
    return {
        "container": serialization_mod.SAFE_CONTAINER,
        "format_version": format_version,
        "package_version": "0.13.0",
    }


def _write_safe_directory(
    path: Path,
    *,
    meta_text: str | None = None,
    config_text: str | None = None,
    include_config: bool = True,
) -> None:
    """Write enough safe-checkpoint members to reach parser validation."""
    path.mkdir()
    (path / serialization_mod.SAFE_META_FILENAME).write_text(
        meta_text if meta_text is not None else json.dumps(_safe_meta()),
        encoding="utf-8",
    )
    if include_config:
        (path / serialization_mod.SAFE_CONFIG_FILENAME).write_text(
            config_text if config_text is not None else "{}",
            encoding="utf-8",
        )
    (path / serialization_mod.SAFE_WEIGHTS_FILENAME).write_bytes(b"placeholder")


def test_serialization_reverse_latent_and_rectangular_guards() -> None:
    """Checkpoint validation rejects malformed additive hetero metadata."""
    with pytest.raises(ValueError, match="synthesize_reverse_relations must be a bool"):
        _require_hetero_schema(_hetero_schema(synthesize_reverse_relations=1))
    with pytest.raises(ValueError, match="latent_dims must be a mapping"):
        _require_hetero_schema(_hetero_schema(latent_dims=[]))
    with pytest.raises(ValueError, match="latent_dims is incomplete or invalid"):
        _require_hetero_schema(_hetero_schema(latent_dims={"a": 2}))

    assert _state_dict_has_rectangular_hetero_markers(
        {"koopman._rel_rect.0.K": torch.eye(2)}
    )
    assert _state_dict_has_rectangular_hetero_markers(
        {"encoder.type_latent.a.weight": torch.zeros(2, 2)}
    )
    assert not _state_dict_has_rectangular_hetero_markers({"other": torch.zeros(1)})

    with pytest.raises(ValueError, match="config.latent_dims is missing"):
        _validate_hetero_latent_dims_vs_state(
            _hetero_schema(),
            {"decoder.type_latent_in.a.weight": torch.zeros(2, 2)},
        )
    with pytest.raises(ValueError, match="has no koopman._rel_rect"):
        _validate_hetero_latent_dims_vs_state(
            _hetero_schema(latent_dims=_LATENT_DIMS),
            {},
        )
    state = {
        "koopman._rel_rect.0.K": torch.zeros(3, 2),
        "koopman._selves.a.K": torch.zeros(3, 3),
    }
    with pytest.raises(ValueError, match="expects \\(2, 2\\)"):
        _validate_hetero_latent_dims_vs_state(
            _hetero_schema(latent_dims=_LATENT_DIMS),
            state,
        )


def test_serialization_symmetry_entity_and_cell_sheaf_guards() -> None:
    """Symmetry / entity_ids parsers and cell/sheaf rebuild branches."""
    from koopman_graph.model.factory import build_encoder_peers
    from koopman_graph.serialization import (
        _build_decoder,
        _build_encoder,
        _coerce_checkpoint_entity_ids,
        _parse_symmetry_config,
        build_model_config,
        reconstruct_model,
    )

    with pytest.raises(ValueError, match="symmetry config must be a dict"):
        _parse_symmetry_config("orbit")
    with pytest.raises(ValueError, match="symmetry.symmetry must be"):
        _parse_symmetry_config({"symmetry": "perm"})
    with pytest.raises(ValueError, match="symmetry.method must be"):
        _parse_symmetry_config({"method": "wl"})
    with pytest.raises(ValueError, match="orbit_partition must be a sequence"):
        _parse_symmetry_config({"orbit_partition": 3})
    with pytest.raises(ValueError, match="orbit must be a sequence of ints"):
        _parse_symmetry_config({"orbit_partition": [1]})
    parsed = _parse_symmetry_config(
        {"symmetry": "isotypic", "orbit_partition": [[0], [1]], "method": "auto"}
    )
    assert parsed[2] == "exact" and parsed[1] is False and parsed[3] == "isotypic"
    assert (
        _parse_symmetry_config({"auto_orbits": True, "orbit_partition": None})[0]
        is None
    )

    with pytest.raises(ValueError, match="entity_ids must be a list"):
        _coerce_checkpoint_entity_ids("a")
    with pytest.raises(ValueError, match="non-empty"):
        _coerce_checkpoint_entity_ids([])
    with pytest.raises(ValueError, match="entries must be str or int"):
        _coerce_checkpoint_entity_ids([1.5])
    assert _coerce_checkpoint_entity_ids(["n0", 1]) == ("n0", 1)

    sheaf_enc, sheaf_dec = build_encoder_peers(
        "sheaf",
        in_channels=2,
        hidden_channels=4,
        latent_dim=2,
        out_channels=2,
        num_layers=1,
    )
    cell_enc, cell_dec = build_encoder_peers(
        "cell_complex",
        in_channels=2,
        hidden_channels=4,
        latent_dim=2,
        out_channels=2,
        num_layers=1,
    )
    for encoder, decoder in ((sheaf_enc, sheaf_dec), (cell_enc, cell_dec)):
        model = GraphKoopmanModel(
            encoder=encoder,
            decoder=decoder,
            latent_dim=2,
            time_step=1.0,
            koopman="graph",
        )
        config = build_model_config(model)
        rebuilt_enc = _build_encoder(config["encoder"])
        rebuilt_dec = _build_decoder(config["decoder"])
        assert type(rebuilt_enc) is type(encoder)
        assert type(rebuilt_dec) is type(decoder)

    with (
        patch(
            "koopman_graph.serialization._RESERVED_KOOPMAN_KINDS",
            {"future_kind": "TASK-9999"},
        ),
        pytest.raises(ValueError, match="planned; lands in TASK-9999"),
    ):
        reconstruct_model(
            {
                "latent_dim": 2,
                "time_step": 1.0,
                "dynamics_mode": "discrete",
                "koopman_kind": "future_kind",
                "koopman_init_mode": "identity",
                "koopman_init_scale": 0.01,
                "koopman_parameterization": "dense",
                "koopman_max_spectral_radius": 1.0,
                "control_dim": 0,
                "control_mode": "additive",
                "n_delays": 1,
                "encoder": {
                    "type": "gcn",
                    "in_channels": 2,
                    "hidden_channels": 4,
                    "latent_dim": 2,
                    "num_layers": 1,
                    "activation": "relu",
                },
                "decoder": {
                    "type": "gcn",
                    "latent_dim": 2,
                    "hidden_channels": 4,
                    "out_channels": 2,
                    "num_layers": 1,
                    "activation": "relu",
                },
                "sparsity": "dense",
                "adjacency": None,
            }
        )


def test_serialization_json_and_device_helpers_cover_tuple_paths() -> None:
    """Tuple configs become lists and explicit devices become strings."""
    assert serialization_mod._json_ready({"shape": (1, (2, 3))}) == {
        "shape": [1, [2, 3]]
    }
    assert serialization_mod._json_ready([{"nested": (4,)}]) == [{"nested": [4]}]
    assert serialization_mod._resolve_safetensors_device(torch.device("cpu")) == "cpu"


def test_serialization_zip_writer_rejects_existing_directory(
    tmp_path: Path,
) -> None:
    """A zip checkpoint cannot overwrite an existing directory."""
    destination = tmp_path / "model.kgckpt"
    destination.mkdir()
    with pytest.raises(ValueError, match="must be a file"):
        serialization_mod._save_safetensors_v1_zip(object(), destination)


def test_serialization_zip_probe_covers_false_and_bad_zip(tmp_path: Path) -> None:
    """The safe-zip probe rejects absent files and malformed archives."""
    missing = tmp_path / "missing.kgckpt"
    assert not serialization_mod._is_safetensors_v1_zip(missing)

    candidate = tmp_path / "candidate.kgckpt"
    candidate.write_bytes(b"not a zip")
    with (
        patch.object(serialization_mod.zipfile, "is_zipfile", return_value=True),
        patch.object(
            serialization_mod.zipfile,
            "ZipFile",
            side_effect=zipfile.BadZipFile("bad"),
        ),
    ):
        assert not serialization_mod._is_safetensors_v1_zip(candidate)


def test_serialization_save_rejects_unknown_format(tmp_path: Path) -> None:
    """The checkpoint writer rejects unknown format tokens."""
    with pytest.raises(ValueError, match="Unsupported checkpoint format"):
        serialization_mod.save_checkpoint(
            object(),
            tmp_path / "model",
            format="unknown",  # type: ignore[arg-type]
        )


def test_serialization_directory_loader_rejects_missing_config(
    tmp_path: Path,
) -> None:
    """A safe directory must contain config.json."""
    checkpoint = tmp_path / "missing_config"
    _write_safe_directory(checkpoint, include_config=False)
    with pytest.raises(ValueError, match=serialization_mod.SAFE_CONFIG_FILENAME):
        serialization_mod._load_safetensors_v1_directory(
            checkpoint,
            map_location=None,
        )


@pytest.mark.parametrize(
    ("meta_text", "message"),
    [
        ("{", "Invalid meta.json"),
        ("[]", "meta.json must be a JSON object"),
        (json.dumps(_safe_meta(format_version=999)), "format_version"),
    ],
)
def test_serialization_directory_loader_rejects_invalid_metadata(
    tmp_path: Path,
    meta_text: str,
    message: str,
) -> None:
    """Malformed, non-object, and unsupported metadata are rejected."""
    checkpoint = tmp_path / "bad_meta"
    _write_safe_directory(checkpoint, meta_text=meta_text)
    with pytest.raises(ValueError, match=message):
        serialization_mod._load_safetensors_v1_directory(
            checkpoint,
            map_location=None,
        )


def test_serialization_directory_loader_rejects_non_object_config(
    tmp_path: Path,
) -> None:
    """Safe config JSON must decode to an object."""
    checkpoint = tmp_path / "bad_config"
    _write_safe_directory(checkpoint, config_text="[]")
    with pytest.raises(ValueError, match="config.json must be a JSON object"):
        serialization_mod._load_safetensors_v1_directory(
            checkpoint,
            map_location=None,
        )


def test_serialization_zip_loader_rejects_missing_members_and_bad_zip(
    tmp_path: Path,
) -> None:
    """The direct zip loader reports missing members and malformed archives."""
    incomplete = tmp_path / "incomplete.kgckpt"
    with zipfile.ZipFile(incomplete, mode="w") as archive:
        archive.writestr(serialization_mod.SAFE_META_FILENAME, "{}")
    with pytest.raises(ValueError, match="missing members"):
        serialization_mod._load_safetensors_v1_zip(
            incomplete,
            map_location=None,
        )

    malformed = tmp_path / "malformed.kgckpt"
    malformed.write_bytes(b"not a zip")
    with pytest.raises(ValueError, match="Invalid safetensors_v1 zip"):
        serialization_mod._load_safetensors_v1_zip(
            malformed,
            map_location=None,
        )


def test_serialization_load_rejects_unsupported_path_type() -> None:
    """An existing path that is neither file nor directory is rejected."""
    destination = MagicMock()
    destination.exists.return_value = True
    destination.is_dir.return_value = False
    destination.is_file.return_value = False
    with (
        patch.object(serialization_mod, "Path", return_value=destination),
        pytest.raises(ValueError, match="Unsupported checkpoint path type"),
    ):
        serialization_mod.load_checkpoint("special-device")
