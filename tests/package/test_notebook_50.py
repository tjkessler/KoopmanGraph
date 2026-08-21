"""Honesty locks for example 50 (wiring check vs hold-last, remap note)."""

from __future__ import annotations

import json
import pathlib

from tests.helpers import REPO_ROOT

_PROJECT_ROOT = REPO_ROOT
_NOTEBOOK = _PROJECT_ROOT / "examples" / "50_graph_state_closure.ipynb"
_REQUIRED_HEADINGS = (
    "# Graph-state closure versus hold-last",
    "## Setup",
    "## Motivation and background",
    "## Minimal example",
    "## Progressive deep dive",
    "## Results",
    "## Interpretation / discussion",
    "## Takeaways",
    "## Further reading",
)


def _notebook_markdown(path: pathlib.Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    chunks: list[str] = []
    for cell in payload.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        source = cell.get("source", [])
        if isinstance(source, list):
            chunks.append("".join(source))
        elif isinstance(source, str):
            chunks.append(source)
    return "\n".join(chunks)


def test_example_50_notebook_exists() -> None:
    """Example 50 must be a tracked Jupyter notebook."""
    assert _NOTEBOOK.is_file()


def test_example_50_markdown_has_scientific_arc() -> None:
    """Example 50 must follow the nine-section scientific-notebook arc."""
    markdown = _notebook_markdown(_NOTEBOOK)
    missing = [heading for heading in _REQUIRED_HEADINGS if heading not in markdown]
    assert not missing, f"example 50 missing headings: {missing}"


def test_example_50_markdown_wiring_check_and_remap_note() -> None:
    """Example 50 must beat hold-last as a wiring check and note remapping."""
    markdown = _notebook_markdown(_NOTEBOOK)
    plain = markdown.replace("**", "")
    lower = plain.lower()
    assert "wiring check" in lower
    assert "hold-last" in lower
    assert "not a learned-forecast" in lower
    assert "entityremap" in lower
    assert "unbounded growth" in lower
    assert "permute" in lower
    assert "self_adaptive" in lower or "adaptiveadjacency" in lower.replace(" ", "")
    assert "metr-la" in lower
    assert "sota" in lower
    assert "wu2019wavenet" in lower
    assert "g(n,p)" in lower.replace(" ", "")
    assert "graph_dynamics.html" in lower
    assert "sigmoid" in lower
