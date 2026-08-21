"""Honesty locks for example 47 (identity-bound CLI, no METR-LA download)."""

from __future__ import annotations

import json
import pathlib
import re

from tests.helpers import REPO_ROOT

_LOCAL_PATH = re.compile(
    r"(?:/Users/|/home/|/var/folders/|/private/var/|[A-Za-z]:\\Users\\)"
)

_PROJECT_ROOT = REPO_ROOT
_NOTEBOOK = _PROJECT_ROOT / "examples" / "47_benchmark_manifest.ipynb"
_REQUIRED_HEADINGS = (
    "# Identity-bound benchmark manifests",
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


def test_example_47_notebook_exists() -> None:
    """Example 47 must be a tracked Jupyter notebook."""
    assert _NOTEBOOK.is_file()


def test_example_47_markdown_has_scientific_arc() -> None:
    """Example 47 must follow the nine-section scientific-notebook arc."""
    markdown = _notebook_markdown(_NOTEBOOK)
    missing = [heading for heading in _REQUIRED_HEADINGS if heading not in markdown]
    assert not missing, f"example 47 missing headings: {missing}"


def test_example_47_markdown_identity_bound_no_telemetry_download() -> None:
    """Example 47 must stay on smoke fixtures and must not download METR-LA."""
    markdown = _notebook_markdown(_NOTEBOOK)
    plain = markdown.replace("**", "")
    lower = plain.lower()
    assert "executed=False" in plain
    assert "smoke_telemetry" in plain
    assert "verify" in lower
    assert "[cli]" in plain
    assert "does not download METR-LA" in plain
    assert "wget" not in lower
    assert "curl " not in lower
    assert "metr-la.h5" not in lower


def test_example_47_markdown_digest_semantics_and_citations() -> None:
    """Example 47 must state verify semantics and cite named incumbents."""
    markdown = _notebook_markdown(_NOTEBOOK)
    plain = markdown.replace("**", "")
    assert "ensure_ascii=False" in plain
    assert "package_version" in plain
    assert "manifest_sha256" in plain
    assert "dataset.sha256" in plain
    assert "does not enforce `executed=False`" in plain
    assert "wrong payload" in plain.lower()
    assert "benchmarks.html" in markdown
    assert "api.html" in markdown
    assert "doi:10.1145/3474717.3483923" in markdown
    assert "doi:10.1109/TKDE.2024.3484454" in markdown
    assert "openreview.net/forum?id=sjihxgwaz" in markdown.lower()


def _notebook_output_text(path: pathlib.Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    chunks: list[str] = []
    for cell in payload.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        for item in cell.get("outputs", []):
            text = item.get("text", "")
            if isinstance(text, list):
                chunks.append("".join(text))
            elif isinstance(text, str):
                chunks.append(text)
            traceback = item.get("traceback", [])
            if isinstance(traceback, list):
                chunks.extend(str(line) for line in traceback)
    return "\n".join(chunks)


def test_example_47_outputs_omit_local_paths() -> None:
    """Stored outputs must not leak checkout, temp, or home paths."""
    text = _notebook_output_text(_NOTEBOOK)
    hits = _LOCAL_PATH.findall(text)
    assert not hits, f"example 47 outputs contain local paths: {hits}"
