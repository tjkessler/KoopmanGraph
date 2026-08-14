"""Coverage and error-path tests for :mod:`koopman_graph.distributed`."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from torch import nn

import koopman_graph.distributed.lightning_module as lightning_mod
import koopman_graph.serialization as serialization_mod


def test_lightning_export_forces_legacy_checkpoint_format(tmp_path: Path) -> None:
    """Lightning's Trainer hook exports an unwrapped legacy checkpoint."""
    fake_lightning = SimpleNamespace(LightningModule=nn.Module)
    with patch.object(lightning_mod, "_import_lightning", return_value=fake_lightning):
        module_cls = lightning_mod._build_koopman_lightning_module()
    model = nn.Linear(2, 2)
    module = module_cls(model)
    destination = tmp_path / "lightning.pt"

    with patch.object(serialization_mod, "save_checkpoint") as save_checkpoint:
        module.export_format1_checkpoint(destination)
    save_checkpoint.assert_called_once_with(
        model,
        destination,
        format="legacy_pt",
    )
