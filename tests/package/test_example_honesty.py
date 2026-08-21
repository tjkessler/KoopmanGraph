"""Honesty locks for examples 22 / 37 / 38 (saved rankings, not SOTA)."""

from __future__ import annotations

import json
import pathlib
import re

from tests.helpers import REPO_ROOT

_PROJECT_ROOT = REPO_ROOT

_EXAMPLE22_AGGREGATE_RMSE = ("0.6551", "0.7076", "1.0754", "0.9036")
_TRAILS_GNNS = re.compile(r"trails\s+gnns", re.I)
_TRANSFER_ADVANTAGE_FALSE = re.compile(
    r"transfer_advantage[`\s=]*is[`\s]*`?False",
    re.I,
)
_HOP_ORDER_NOT_CAUSE = re.compile(
    r"(?:does\s+not\s+attribute|not\s+claimed\s+to\s+explain|"
    r"neither\s+protocol\s+attributes).{0,80}hop\s+order"
    r"|hop\s+order.{0,80}(?:does\s+not|not\s+claimed|"
    r"neither\s+protocol\s+attributes)",
    re.I | re.S,
)
_NOT_SOTA = re.compile(r"not.{0,80}sota", re.I | re.S)
_HONESTY_SCAN_PATHS = (
    "examples/22_gnn_forecaster_comparison.ipynb",
    "examples/37_cross_topology_transfer.ipynb",
    "examples/38_operator_factorization_ablation.ipynb",
    "docs/source/limitations.rst",
    "docs/source/tutorials.rst",
    "docs/source/faq.rst",
    "README.md",
    "CHANGELOG.md",
)
_EXAMPLE22_PUBLIC_LOCKS = (
    "examples/22_gnn_forecaster_comparison.ipynb",
    "docs/source/limitations.rst",
    "docs/source/tutorials.rst",
    "docs/source/faq.rst",
    "README.md",
)


def _cell_source(cell: dict) -> str:
    source = cell.get("source", [])
    if isinstance(source, list):
        return "".join(source)
    if isinstance(source, str):
        return source
    return ""


def _notebook_markdown(path: pathlib.Path) -> str:
    """Flatten Markdown cell sources (discussion, not code outputs)."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    chunks = [
        _cell_source(cell)
        for cell in payload.get("cells", [])
        if cell.get("cell_type") == "markdown"
    ]
    return "\n".join(chunks)


def _artifact_text(relative: str) -> str:
    path = _PROJECT_ROOT / relative
    if path.suffix == ".ipynb":
        payload = json.loads(path.read_text(encoding="utf-8"))
        chunks = [_cell_source(cell) for cell in payload.get("cells", [])]
        return "\n".join(chunks)
    return path.read_text(encoding="utf-8")


def test_examples_22_37_38_forbid_trails_gnns_slogan() -> None:
    """Do not claim Koopman trails GNNs in the honesty-pass artifacts."""
    hits: list[str] = []
    for relative in _HONESTY_SCAN_PATHS:
        text = _artifact_text(relative)
        if _TRAILS_GNNS.search(text):
            hits.append(relative)
    assert not hits, f"'trails GNNs' remains in: {hits}"


def test_example_22_markdown_cites_saved_aggregate_rmse() -> None:
    """Example 22 discussion must cite the saved ranking and budget caveats."""
    markdown = _notebook_markdown(
        _PROJECT_ROOT / "examples/22_gnn_forecaster_comparison.ipynb"
    )
    for value in _EXAMPLE22_AGGREGATE_RMSE:
        assert value in markdown, f"example 22 markdown missing RMSE {value}"
    lower = markdown.lower()
    assert "unequal" in lower and "budget" in lower
    assert "teaching" in lower
    assert _NOT_SOTA.search(markdown), "example 22 markdown must hedge SOTA"


def test_example_22_public_docs_lock_aggregate_rmse() -> None:
    """Limitations, gallery, and README must carry the same four RMSE figures."""
    for relative in _EXAMPLE22_PUBLIC_LOCKS:
        if relative.endswith(".ipynb"):
            text = _notebook_markdown(_PROJECT_ROOT / relative)
        else:
            text = (_PROJECT_ROOT / relative).read_text(encoding="utf-8")
        missing = [value for value in _EXAMPLE22_AGGREGATE_RMSE if value not in text]
        assert not missing, f"{relative} missing RMSE {missing}"
        lower = text.lower()
        assert "unequal" in lower and "budget" in lower, (
            f"{relative} missing budget caveat"
        )
        assert "teaching" in lower, f"{relative} missing teaching-baseline language"


def test_example_37_markdown_keeps_negative_transfer_advantage() -> None:
    """Example 37 must keep transfer_advantage False versus the pernode control."""
    markdown = _notebook_markdown(
        _PROJECT_ROOT / "examples/37_cross_topology_transfer.ipynb"
    )
    assert "transfer_advantage" in markdown
    assert _TRANSFER_ADVANTAGE_FALSE.search(markdown), (
        "example 37 markdown must state transfer_advantage is False"
    )
    assert "pernode" in markdown.lower()


def test_example_38_markdown_keeps_joint_ls_gap_and_hop_disclaimer() -> None:
    """Example 38 must keep the historical MSE gap and hop-order disclaimer."""
    markdown = _notebook_markdown(
        _PROJECT_ROOT / "examples/38_operator_factorization_ablation.ipynb"
    )
    assert "0.71" in markdown
    assert "0.019" in markdown
    assert "hop order" in markdown.lower()
    assert _HOP_ORDER_NOT_CAUSE.search(markdown), (
        "example 38 markdown must not attribute the joint-LS gap to hop order"
    )
