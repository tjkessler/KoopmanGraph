"""Import-cycle guards for ``koopman_graph.identification``."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

from tests.helpers import REPO_ROOT

_PACKAGE_ROOT = REPO_ROOT / "src" / "koopman_graph"
_FORBIDDEN_FROM_IDENTIFICATION = (
    "koopman_graph.training",
    "koopman_graph.model",
    "koopman_graph.adaptation",
)
_FORBIDDEN_FROM_TRAINING = ("koopman_graph.identification",)


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


def test_identification_sources_do_not_import_training_model_or_adaptation() -> None:
    """``identification/`` must not import training, model, or adaptation."""
    assert (
        _module_import_offenders(
            _PACKAGE_ROOT / "identification",
            _FORBIDDEN_FROM_IDENTIFICATION,
        )
        == []
    )


def test_training_sources_do_not_import_identification_at_module_load() -> None:
    """``training`` may lazy-import identification inside functions only."""
    assert (
        _module_import_offenders(
            _PACKAGE_ROOT / "training",
            _FORBIDDEN_FROM_TRAINING,
            module_level_only=True,
        )
        == []
    )


def test_model_sources_do_not_import_identification_at_module_load() -> None:
    """``model`` may TYPE_CHECKING-import identification; not at runtime load."""
    assert (
        _module_import_offenders(
            _PACKAGE_ROOT / "model",
            _FORBIDDEN_FROM_TRAINING,
            module_level_only=True,
        )
        == []
    )


def test_importing_training_does_not_load_identification() -> None:
    """``import koopman_graph.training`` must not load identification."""
    script = (
        "import sys\n"
        "import koopman_graph.training\n"
        "assert 'koopman_graph.identification' not in sys.modules, "
        "sorted(k for k in sys.modules if 'identification' in k)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_importing_model_does_not_load_identification() -> None:
    """``import koopman_graph.model`` must not load identification."""
    script = (
        "import sys\n"
        "import koopman_graph.model\n"
        "assert 'koopman_graph.identification' not in sys.modules, "
        "sorted(k for k in sys.modules if 'identification' in k)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_default_fit_does_not_load_identification() -> None:
    """Adam ``fit`` (``identification=None``) does not import identification."""
    script = """\
import sys
import torch
from torch_geometric.data import Data
from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
from koopman_graph.data import GraphSnapshotSequence

edge = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
sequence = GraphSnapshotSequence(
    [
        Data(x=torch.ones(2, 3), edge_index=edge),
        Data(x=0.9 * torch.ones(2, 3), edge_index=edge),
    ]
)
model = GraphKoopmanModel(
    encoder=GNNEncoder(in_channels=3, hidden_channels=4, latent_dim=2, num_layers=1),
    decoder=GNNDecoder(latent_dim=2, hidden_channels=4, out_channels=3, num_layers=1),
    latent_dim=2,
    time_step=0.1,
)
model.fit(sequence, epochs=1, lr=1e-2, device="cpu")
assert "koopman_graph.identification" not in sys.modules, sorted(
    key for key in sys.modules if "identification" in key
)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_importing_metrics_does_not_load_identification() -> None:
    """``import koopman_graph.metrics`` must not load identification."""
    script = (
        "import sys\n"
        "import koopman_graph.metrics\n"
        "assert 'koopman_graph.identification' not in sys.modules, "
        "sorted(k for k in sys.modules if 'identification' in k)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_identification_package_is_importable() -> None:
    """``import koopman_graph.identification`` exposes solvers."""
    import koopman_graph.identification as ident

    assert ident.IdentificationReport is not None
    assert ident.IdentificationConfig().solver == "ridge"
    assert callable(ident.identify_operator)
    assert callable(ident.select_latent_rank)
