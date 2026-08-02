"""Honesty gates for teaching traffic / GraphCast baselines (TASK-1930).

Design R2 / §7: teaching ports must not claim LibCity / BasicTS leaderboard
parity, must not use unguarded SOTA phrasing for those ports, and must not
present GraphCast teaching slices as METR-LA / PEMS traffic results.
"""

from __future__ import annotations

import json
import pathlib
import re
from collections.abc import Iterator

import pytest

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]

_TEACHING_NAME = re.compile(
    r"AGCRN|MTGNN|STGODE|GraphCast|ForecasterProtocol|"
    r"teaching\s+baseline|baselines\.gnn|STGCNBaseline|DCRNNBaseline|"
    r"GraphWaveNetBaseline",
    re.I,
)

# Positive-looking claims (case-insensitive). Matched spans need a local allow.
_FORBIDDEN: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"leaderboard[- ]parity", re.I),
        "Do not claim LibCity/BasicTS leaderboard parity for teaching baselines "
        "(design §7 / R2).",
    ),
    (
        re.compile(
            r"\b(?:achieves?|matches?|reproduces?|reproducing)\b"
            r".{0,48}\b(?:libcity|basicts)\b",
            re.I,
        ),
        "Do not claim the teaching baselines achieve / match / reproduce "
        "LibCity or BasicTS protocols (design §7).",
    ),
    (
        re.compile(
            r"\b(?:libcity|basicts)\b.{0,48}\b(?:parity|leaderboard)\b",
            re.I,
        ),
        "Do not claim LibCity/BasicTS leaderboard parity for teaching baselines "
        "(design §7).",
    ),
    (
        re.compile(r"\bSOTA\b"),
        "Unguarded SOTA phrasing is forbidden for teaching traffic / GraphCast "
        "baselines; say they are teaching ports with protocol deviations "
        "(design R2).",
    ),
    (
        re.compile(r"state[ -]of[ -]the[ -]art", re.I),
        "Unguarded 'state of the art' phrasing is forbidden for teaching "
        "traffic / GraphCast baselines (design R2).",
    ),
    (
        re.compile(
            r"graphcast.{0,100}(?:metr-?la|pems\b).{0,60}"
            r"(?:sota|leaderboard|state[ -]of[ -]the[ -]art)",
            re.I | re.DOTALL,
        ),
        "Do not present GraphCast teaching slices as METR-LA / PEMS traffic "
        "SOTA or leaderboard results (design §7).",
    ),
    (
        re.compile(
            r"(?:metr-?la|pems\b).{0,100}graphcast.{0,60}"
            r"(?:sota|leaderboard|state[ -]of[ -]the[ -]art)",
            re.I | re.DOTALL,
        ),
        "Do not present GraphCast teaching slices as METR-LA / PEMS traffic "
        "SOTA or leaderboard results (design §7).",
    ),
)

# Local context that marks an honest / negative use of a forbidden phrase.
_ALLOW = re.compile(
    r"(?:"
    r"not\b.{0,40}\bsota\b"
    r"|\bnot\s+sota\b"
    r"|\(not\s+sota\)"
    r"|without\s+claiming\s+sota"
    r"|claiming\s+sota"
    r"|do\s+not\s+(?:claim|read|compare|present|tabulate)"
    r"|cannot\s+claim"
    r"|must\s+not"
    r"|forbid(?:s|den)?"
    r"|out\s+of\s+scope"
    r"|non-?goals?"
    r"|deferred"
    r"|prefer\s+dedicated"
    r"|maintained\s+sota"
    r"|dedicated-?library\s+sota"
    r"|reproductions?\s+of\s+dedicated"
    r"|not\s+dedicated-?library"
    r"|teaching\s+sota"
    r"|adding\s+further\s+sota"
    r"|leaderboard[- ]matched\s+traffic\s+protocols"
    r"|leaderboard\s+claims"
    r"|leaderboard\s+(?:entry|numbers|project|reproductions?)"
    r"|not\s+a\s+leaderboard"
    r"|still\s+not\s+a\s+leaderboard"
    r"|accidental\s+leaderboard"
    r"|no\s+libcity"
    r"|not\s+libcity"
    r"|not\s+a\s+(?:drop-in\s+)?(?:substitute|replacement).*libcity"
    r"|libcity\s+parity\)"
    r"|zero\s+deviation"
    r"|deviations?"
    r"|teaching(?:-|\s)+(?:baseline|port|slice|adapter|comparison|vs)"
    r"|not\s+protocol-matched"
    r"|not\s+a\s+(?:pems|metr)"
    r"|problem\s+class"
    r")",
    re.I | re.DOTALL,
)


def _iter_text_files() -> Iterator[tuple[pathlib.Path, str, bool]]:
    """Yield ``(path, text, require_teaching_context)`` for scoped trees.

    ``require_teaching_context`` is True for broad Sphinx trees so unrelated
    SOTA / parity language (e.g. DDP window parity) is not scanned.
    """
    gnn_root = _PROJECT_ROOT / "src" / "koopman_graph" / "baselines" / "gnn"
    for path in sorted(gnn_root.rglob("*.py")):
        yield path, path.read_text(encoding="utf-8"), False

    examples = _PROJECT_ROOT / "examples"
    for path in sorted(examples.glob("42_*.ipynb")):
        yield path, _notebook_text(path), False

    docs = _PROJECT_ROOT / "docs" / "source"
    for path in sorted(docs.rglob("*.rst")):
        yield path, path.read_text(encoding="utf-8"), True


def _notebook_text(path: pathlib.Path) -> str:
    """Flatten notebook Markdown / code sources for scanning."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    chunks: list[str] = []
    for cell in payload.get("cells", []):
        source = cell.get("source", [])
        if isinstance(source, list):
            chunks.append("".join(source))
        elif isinstance(source, str):
            chunks.append(source)
    return "\n".join(chunks)


def _context_window(text: str, start: int, end: int, *, radius: int = 96) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    return text[left:right]


def _find_violations(text: str, *, require_teaching_context: bool) -> list[str]:
    """Return human-readable violation strings for ``text``."""
    violations: list[str] = []
    for pattern, rule in _FORBIDDEN:
        for match in pattern.finditer(text):
            window = _context_window(text, match.start(), match.end())
            if require_teaching_context and not _TEACHING_NAME.search(window):
                continue
            if _ALLOW.search(window):
                continue
            # Line number for the match start.
            line_no = text.count("\n", 0, match.start()) + 1
            snippet = match.group(0).replace("\n", " ")
            violations.append(
                f"line {line_no}: {snippet!r} — {rule} Nearby context: {window!r}"
            )
    return violations


def test_teaching_baseline_honesty_gates() -> None:
    """Scan teaching-baseline sources for leaderboard / SOTA honesty violations."""
    failures: list[str] = []
    scanned = 0
    for path, text, require_ctx in _iter_text_files():
        scanned += 1
        for message in _find_violations(text, require_teaching_context=require_ctx):
            rel = path.relative_to(_PROJECT_ROOT)
            failures.append(f"{rel}: {message}")

    assert scanned > 0, "honesty gate scanned no files"
    if failures:
        joined = "\n".join(failures)
        pytest.fail(
            "Teaching-baseline honesty gate failed (design §7 / R2).\n"
            "Teaching ports must not claim LibCity/BasicTS leaderboard parity, "
            "unguarded SOTA status, or GraphCast-as-METR/PEMS traffic results.\n"
            f"{joined}"
        )


def test_honesty_allowlist_accepts_negative_phrasing() -> None:
    """Sanity-check that honest caveats are not false positives."""
    honest = (
        "These are teaching baselines, not dedicated-library SOTA.\n"
        "Do not claim LibCity leaderboard parity.\n"
        "GraphCast is a mesh-weather teaching slice, not a PEMS/METR "
        "forecaster; do not compare as traffic SOTA.\n"
        "baselines cannot claim paper / LibCity parity).\n"
    )
    assert _find_violations(honest, require_teaching_context=False) == []


def test_honesty_gate_flags_positive_sota_claim() -> None:
    """A bare SOTA claim in teaching-baseline scope must fail."""
    bad = "AGCRNBaseline is SOTA on METR-LA.\n"
    hits = _find_violations(bad, require_teaching_context=False)
    assert hits, "expected unguarded SOTA claim to be flagged"
    assert "SOTA" in hits[0]


def test_honesty_gate_flags_graphcast_as_metr_sota() -> None:
    """GraphCast presented as METR-LA SOTA must fail."""
    bad = "GraphCastBaseline delivers METR-LA SOTA traffic forecasts.\n"
    hits = _find_violations(bad, require_teaching_context=False)
    assert hits, "expected GraphCast-as-METR SOTA claim to be flagged"
