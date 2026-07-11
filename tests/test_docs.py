"""Tests for public API documentation completeness."""

from __future__ import annotations

import ast
import inspect
import pathlib
import re

import koopman_graph
import koopman_graph.datasets

_NUMPY_SECTION = re.compile(
    r"^\s*("
    r"Parameters|Returns|Raises|Yields|Attributes|Notes|Examples|"
    r"See Also|References|Warnings"
    r")\s*$",
    re.M,
)
_NUMPY_UNDERLINE = re.compile(r"^\s*-{3,}\s*$", re.M)
_DOC_ROOTS = (
    pathlib.Path(inspect.getfile(koopman_graph)).parent,
    pathlib.Path(__file__).resolve().parents[1] / "scripts",
)
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _has_numpy_style(doc: str) -> bool:
    return bool(
        doc
        and doc.strip()
        and _NUMPY_SECTION.search(doc)
        and _NUMPY_UNDERLINE.search(doc)
    )


def _assert_has_docstring(obj: object, qualname: str) -> None:
    doc = inspect.getdoc(obj)
    assert doc is not None and doc.strip(), f"{qualname} is missing a docstring"
    assert _has_numpy_style(doc), f"{qualname} is missing NumPy-style sections"


def _iter_definitions(path: pathlib.Path) -> list[tuple[int, str, str, ast.AST]]:
    tree = ast.parse(path.read_text())
    items: list[tuple[int, str, str, ast.AST]] = []

    def visit(node: ast.AST, parents: tuple[str, ...]) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            qualname = ".".join([*parents, node.name])
            items.append((node.lineno, "function", qualname, node))
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    child_qualname = f"{qualname}.{child.name}"
                    items.append((child.lineno, "function", child_qualname, child))
        elif isinstance(node, ast.ClassDef):
            qualname = ".".join([*parents, node.name])
            items.append((node.lineno, "class", qualname, node))
            for child in node.body:
                visit(child, (*parents, node.name))

    for top in tree.body:
        visit(top, (path.stem,))

    return items


def test_package_exports_have_docstrings() -> None:
    for name in koopman_graph.__all__:
        if name == "__version__":
            continue
        _assert_has_docstring(getattr(koopman_graph, name), f"koopman_graph.{name}")


def test_dataset_exports_have_docstrings() -> None:
    for name in koopman_graph.datasets.__all__:
        _assert_has_docstring(
            getattr(koopman_graph.datasets, name),
            f"koopman_graph.datasets.{name}",
        )


def test_source_definitions_have_numpy_docstrings() -> None:
    missing: list[str] = []
    for root in _DOC_ROOTS:
        for path in sorted(root.rglob("*.py")):
            for lineno, kind, qualname, node in _iter_definitions(path):
                doc = ast.get_docstring(node)
                rel = path.relative_to(_PROJECT_ROOT)
                label = f"{rel}:{lineno} {kind} {qualname}"
                if doc is None or not doc.strip():
                    missing.append(f"{label}: missing docstring")
                elif not _has_numpy_style(doc):
                    missing.append(f"{label}: not NumPy-style")
    assert not missing, "Docstring issues:\n" + "\n".join(missing)
