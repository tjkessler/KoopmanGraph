"""Coverage and error-path tests for :mod:`koopman_graph.cli`."""

from __future__ import annotations

import importlib
import runpy
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

cli_main_mod = importlib.import_module("koopman_graph.cli.main")


def test_cli_entry_points_cover_help_and_main_guards(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Direct and module entry points print help and exit successfully."""
    assert cli_main_mod.main([]) == 0
    assert "usage: koopman-graph" in capsys.readouterr().out

    main_path = Path(cli_main_mod.__file__)
    with (
        patch.object(sys, "argv", [str(main_path)]),
        pytest.raises(SystemExit) as main_exit,
    ):
        runpy.run_path(str(main_path), run_name="__main__")
    assert main_exit.value.code == 0

    module_path = main_path.with_name("__main__.py")
    with (
        patch.object(sys, "argv", [str(module_path)]),
        pytest.raises(SystemExit) as module_exit,
    ):
        runpy.run_module("koopman_graph.cli", run_name="__main__", alter_sys=True)
    assert module_exit.value.code == 0


def test_cli_dunder_main_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing the CLI ``__main__`` module does not exit; running it does."""
    sys.modules.pop("koopman_graph.cli.__main__", None)
    import koopman_graph.cli.__main__ as cli_main

    assert callable(cli_main.main)
    sys.modules.pop("koopman_graph.cli.__main__", None)
    monkeypatch.setattr(sys, "argv", ["koopman-graph"])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("koopman_graph.cli.__main__", run_name="__main__")
    assert exit_info.value.code == 0
