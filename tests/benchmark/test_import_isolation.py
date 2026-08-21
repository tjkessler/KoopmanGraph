"""Import-cycle guards for ``koopman_graph.benchmark``."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

from tests.helpers import REPO_ROOT

_PACKAGE_ROOT = REPO_ROOT / "src" / "koopman_graph"
_FORBIDDEN_FROM_BENCHMARK = (
    "koopman_graph.training",
    "koopman_graph.model",
)


def _is_type_checking(node: ast.AST) -> bool:
    """Return whether ``node`` is ``if TYPE_CHECKING:``.

    Parameters
    ----------
    node : ast.AST
        Statement to inspect.

    Returns
    -------
    bool
        ``True`` for a ``TYPE_CHECKING`` guard.
    """
    if not isinstance(node, ast.If):
        return False
    test = node.test
    return isinstance(test, ast.Name) and test.id == "TYPE_CHECKING"


def _import_names(node: ast.AST) -> list[str]:
    """Return imported module names on an import statement.

    Parameters
    ----------
    node : ast.AST
        Import node.

    Returns
    -------
    list of str
        Absolute module names.
    """
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom) and node.module:
        return [node.module]
    return []


def _module_import_offenders(
    root: Path,
    prefixes: tuple[str, ...],
    *,
    module_level_only: bool = False,
) -> list[str]:
    """Return ``path:module`` entries that import a forbidden prefix.

    Parameters
    ----------
    root : Path
        Package directory to walk.
    prefixes : tuple of str
        Forbidden module prefixes (exact or dotted children).
    module_level_only : bool, optional
        When ``True``, ignore function-body lazy imports and
        ``if TYPE_CHECKING:`` blocks. Default is ``False`` (walk all).

    Returns
    -------
    list of str
        Offending ``filename:module`` labels.
    """
    offenders: list[str] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if module_level_only:
            nodes: list[ast.AST] = [
                node for node in tree.body if not _is_type_checking(node)
            ]
        else:
            nodes = list(ast.walk(tree))
        for node in nodes:
            for name in _import_names(node):
                if any(
                    name == prefix or name.startswith(prefix + ".")
                    for prefix in prefixes
                ):
                    offenders.append(f"{path.relative_to(_PACKAGE_ROOT)}:{name}")
    return offenders


def test_no_inbound_imports_of_benchmark() -> None:
    """Other packages may not import ``benchmark`` except lazy CLI handlers."""
    offenders: list[str] = []
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        relative = path.relative_to(_PACKAGE_ROOT)
        if relative.parts[0] == "benchmark":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        module_level_only = relative.parts[0] == "cli"
        if module_level_only:
            nodes: list[ast.AST] = [
                node for node in tree.body if not _is_type_checking(node)
            ]
        else:
            nodes = list(ast.walk(tree))
        for node in nodes:
            for name in _import_names(node):
                if name == "koopman_graph.benchmark" or name.startswith(
                    "koopman_graph.benchmark."
                ):
                    offenders.append(f"{relative}:{name}")
    assert offenders == []


def test_benchmark_sources_do_not_import_training_or_model() -> None:
    """``benchmark/`` must not import ``training`` or ``model``."""
    assert (
        _module_import_offenders(
            _PACKAGE_ROOT / "benchmark",
            _FORBIDDEN_FROM_BENCHMARK,
        )
        == []
    )


def test_importing_cli_does_not_load_benchmark() -> None:
    """``import koopman_graph.cli.main`` must not load ``benchmark``."""
    script = (
        "import sys\n"
        "import koopman_graph.cli.main\n"
        "assert 'koopman_graph.benchmark' not in sys.modules, "
        "sorted(k for k in sys.modules if 'benchmark' in k)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
