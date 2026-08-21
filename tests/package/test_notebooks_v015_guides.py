"""Honesty locks for examples 48–49 and 51–54 (guides; no telemetry download)."""

from __future__ import annotations

import json
import pathlib

from tests.helpers import REPO_ROOT

_PROJECT_ROOT = REPO_ROOT
_ARC = (
    "## Setup",
    "## Motivation and background",
    "## Minimal example",
    "## Progressive deep dive",
    "## Results",
    "## Interpretation / discussion",
    "## Takeaways",
    "## Further reading",
)
_NOTEBOOKS = {
    "48_identification_invariance.ipynb": (
        "# Identification reports, residual-aware gating, and finite-sample invariance",
        (
            "identification=None",
            "Haseli",
            "select_resdmd_gated",
            "polluted",
            "include_invariance",
            "does not download metr-la",
        ),
    ),
    "49_multi_hop_factorization.ipynb": (
        "# Multi-hop factorization: Kronecker spectrum versus dense",
        (
            "0.71",
            "0.019",
            "koopman_filter_degree",
            "Guo",
            "hop order",
            "does not download metr-la",
        ),
    ),
    "51_spectral_diagnostics.ipynb": (
        "# Spectral diagnostics: non-normal shear and Nyquist toys",
        (
            "CONDITION_WARN",
            "Nyquist",
            "finite-horizon",
            "Wilkinson",
            "Trefethen",
            "Zeng",
            "does not download metr-la",
        ),
    ),
    "52_cochain_hodge_modes.ipynb": (
        "# Hodge split of cycle modes",
        (
            "harmonic",
            'koopman="hodge"',
            "TopologicX",
            "physical circulation",
            "does not download metr-la",
            "degree=1",
            "relative L2",
            "compute_spectrum",
        ),
    ),
    "53_latent_rank_selection.ipynb": (
        "# Latent-rank selection on a linear Gaussian oracle",
        (
            "select_latent_rank",
            "Ray Tune",
            "latent_dim",
            "does not download metr-la",
        ),
    ),
    "54_criticality_monitor.ipynb": (
        "# Criticality monitor on a closing spectral gap",
        (
            "monitor_critical_transition",
            "not a certificate",
            "Ghosh",
            "does not download metr-la",
        ),
    ),
}


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


def _notebook_text(path: pathlib.Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    chunks: list[str] = []
    for cell in payload.get("cells", []):
        source = cell.get("source", [])
        if isinstance(source, list):
            chunks.append("".join(source))
        elif isinstance(source, str):
            chunks.append(source)
    return "\n".join(chunks)


def test_v015_guide_notebooks_exist_with_scientific_arc() -> None:
    """Examples 48–49 and 51–54 must exist and follow the scientific arc."""
    for name, (title, _needles) in _NOTEBOOKS.items():
        path = _PROJECT_ROOT / "examples" / name
        assert path.is_file(), f"missing {name}"
        markdown = _notebook_markdown(path)
        assert title in markdown, f"{name} missing title {title!r}"
        missing = [heading for heading in _ARC if heading not in markdown]
        assert not missing, f"{name} missing headings: {missing}"


def test_v015_guide_notebooks_honesty_and_no_telemetry_download() -> None:
    """Guide notebooks must keep honesty needles and must not fetch METR-LA."""
    for name, (_title, needles) in _NOTEBOOKS.items():
        path = _PROJECT_ROOT / "examples" / name
        markdown = _notebook_markdown(path)
        text = _notebook_text(path)
        plain = markdown.replace("**", "")
        folded = " ".join(plain.lower().split())
        lower = text.lower()
        for needle in needles:
            haystack = folded if needle.islower() else markdown
            assert needle in haystack, f"{name} markdown missing {needle!r}"
        assert "wget" not in lower
        assert "curl http" not in lower
        assert "metr-la.h5" not in lower
        assert "download_metr" not in lower
