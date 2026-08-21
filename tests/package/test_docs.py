"""Tests for public API documentation completeness."""

from __future__ import annotations

import ast
import inspect
import pathlib
import re

from tests.helpers import REPO_ROOT

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
_SECTION_HEADER = re.compile(
    r"^(Parameters|Returns|Raises|Yields|Attributes|Notes|Examples|"
    r"See Also|References|Warnings)\s*\n\s*-{3,}\s*$",
    re.M,
)
_DOC_ROOTS = (
    pathlib.Path(inspect.getfile(koopman_graph)).parent,
    REPO_ROOT / "scripts",
)
_PROJECT_ROOT = REPO_ROOT


def _has_numpy_style(doc: str) -> bool:
    return bool(
        doc
        and doc.strip()
        and _NUMPY_SECTION.search(doc)
        and _NUMPY_UNDERLINE.search(doc)
    )


def _split_sections(doc: str) -> dict[str, str]:
    matches = list(_SECTION_HEADER.finditer(doc))
    if not matches:
        return {}
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        name = match.group(1)
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(doc)
        sections[name] = doc[start:end]
    return sections


def _param_names_in_section(section: str) -> set[str]:
    """Collect parameter names from a NumPy Parameters section body.

    Supports single names, comma-groups (including trailing commas / wrapped
    continuation lines), and ``*args`` / ``**kwargs`` star prefixes.
    """
    names: set[str] = set()
    for line in section.splitlines():
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        # Description lines are indented ≥4 spaces in dedented docstrings.
        if indent >= 4:
            continue
        # Names appear before the optional ``: type`` annotation.
        head = line.split(":", 1)[0]
        if not re.search(r"[A-Za-z_]", head):
            continue
        for match in re.finditer(r"\*{0,2}([A-Za-z_][\w]*)", head):
            names.add(match.group(1))
    return names


def _arg_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    args = node.args
    names: list[str] = []
    for arg in [*args.posonlyargs, *args.args]:
        if arg.arg in {"self", "cls"}:
            continue
        names.append(arg.arg)
    if args.vararg is not None:
        names.append(args.vararg.arg)
    for arg in args.kwonlyargs:
        names.append(arg.arg)
    if args.kwarg is not None:
        names.append(args.kwarg.arg)
    return names


def _returns_non_none(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    annotation = node.returns
    if annotation is not None:
        if isinstance(annotation, ast.Constant) and annotation.value is None:
            return False
        return not (isinstance(annotation, ast.Name) and annotation.id == "None")
    for child in ast.walk(node):
        if isinstance(child, ast.Return) and child.value is not None:
            if isinstance(child.value, ast.Constant) and child.value.value is None:
                continue
            return True
    return False


def _assert_has_docstring(obj: object, qualname: str) -> None:
    doc = inspect.getdoc(obj)
    assert doc is not None and doc.strip(), f"{qualname} is missing a docstring"
    assert _has_numpy_style(doc), f"{qualname} is missing NumPy-style sections"


def _iter_definitions(path: pathlib.Path) -> list[tuple[int, str, str, ast.AST]]:
    """Yield every class / function / method, including nested definitions."""
    tree = ast.parse(path.read_text())
    items: list[tuple[int, str, str, ast.AST]] = []

    def visit(
        node: ast.AST,
        parents: tuple[str, ...],
        *,
        in_class: bool,
    ) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            qualname = ".".join([*parents, node.name])
            kind = "method" if in_class and len(parents) >= 2 else "function"
            items.append((node.lineno, kind, qualname, node))
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Nested defs are not methods even when defined inside a method.
                    visit(child, (*parents, node.name), in_class=False)
                elif isinstance(child, ast.ClassDef):
                    visit(child, (*parents, node.name), in_class=True)
        elif isinstance(node, ast.ClassDef):
            qualname = ".".join([*parents, node.name])
            items.append((node.lineno, "class", qualname, node))
            for child in node.body:
                visit(child, (*parents, node.name), in_class=True)

    for top in tree.body:
        if isinstance(top, ast.ClassDef):
            visit(top, (path.stem,), in_class=True)
        elif isinstance(top, (ast.FunctionDef, ast.AsyncFunctionDef)):
            visit(top, (path.stem,), in_class=False)

    return items


def _analyze_definition(kind: str, node: ast.AST, doc: str | None) -> list[str]:
    issues: list[str] = []
    if doc is None or not doc.strip():
        return ["missing docstring"]
    if not _has_numpy_style(doc):
        return ["not NumPy-style"]
    if kind == "class" or not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return issues
    sections = _split_sections(doc)
    names = _arg_names(node)
    if names:
        if "Parameters" not in sections:
            issues.append(f"missing Parameters section for {names}")
        else:
            documented = _param_names_in_section(sections["Parameters"])
            missing = [name for name in names if name not in documented]
            if missing:
                issues.append(f"Parameters missing: {missing}")
    if _returns_non_none(node):
        if "Returns" not in sections and "Yields" not in sections:
            issues.append("missing Returns/Yields section")
        else:
            body = sections.get("Returns") or sections.get("Yields") or ""
            if not body.strip():
                issues.append("empty Returns/Yields section")
    return issues


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
            # Maintainer-only underscore helpers under scripts/ are not API.
            if path.parent.name == "scripts" and path.name.startswith("_"):
                continue
            for lineno, kind, qualname, node in _iter_definitions(path):
                doc = ast.get_docstring(node)
                rel = path.relative_to(_PROJECT_ROOT)
                label = f"{rel}:{lineno} {kind} {qualname}"
                for issue in _analyze_definition(kind, node, doc):
                    missing.append(f"{label}: {issue}")
    assert not missing, "Docstring issues:\n" + "\n".join(missing)


# Headline JOSS features with direct literature precedents.
_REQUIRED_PAPER_BIB_KEYS = (
    "Azencot2020",
    "Bruder2021",
    "Li2017EDMD",
    "Li2018DCRNN",
    "Li2020CompositionalKoopman",
    "Korda2018",
    "Mukherjee2022",
    "Nandanoori2022",
    "Proctor2016DMDc",
    "Williams2015",
    "Wu2019WaveNet",
    "Yu2018STGCN",
)

# v0.7.0 method literature (must exist in paper.bib and be cited in src/docs).
_V070_BIB_KEYS = (
    "ColbrookTownsend2023ResDMD",
    "Colbrook2023ResidualDMD",
    "GavishDonoho2014",
    "Zargarbashi2023ConformalGNN",
    "Rosenstein1993Lyapunov",
    "Welch1967",
)
# Method / software keys that shipped with the post-0.14 catalog.
_V015_BIB_KEYS = (
    "Colbrook2023mpEDMD",
    "Klus2020gEDMD",
    "HaseliCortes2023",
    "Guo2025ModularEDMD",
    "Macesic2018Nonautonomous",
    "Zeng2022Sampling",
    "Pan2021SparseSubspace",
    "Ruiz2023Transferability",
    "Zhang2022TubeMPC",
    "Xu2025ResKoopNet",
    "LibCity2021",
    "BasicTS2024",
    "kooplearn2026",
    "PyKoopman2024",
    "deeptime2021",
    "PyDMD2018",
    "TopoX2024",
)
_SOFTWARE_BIB_KEY = "koopmangraph2026"
_STALE_SOFTWARE_VERSION = "0.11.0"
_STALE_SOFTWARE_DOI_RECORD = "21763908"
_ISOTYPIC_DOC_RELATIVE_PATHS = (
    "docs/source/limitations.rst",
    "CHANGELOG.md",
    "docs/source/capabilities.rst",
    "docs/source/architecture.rst",
)
_ISOTYPIC_UNSHIPPED_PRESENT_TENSE = (
    re.compile(r"isotypic tying is\s+\*{0,2}not\*{0,2}\s+shipped", re.I),
    re.compile(r"neighbor[- ]factor tying not shipped", re.I),
)
_STALE_CONTRIBUTING_VERSION = "0.6.0"
_FROZEN_PACKAGE_VERSION_ASSIGNMENT = re.compile(r'__version__\s*=\s*"\d+\.\d+\.\d+"')
_V015_ARCHITECTURE_HEADING = "v0.15.0 capability architecture"
_UNSHIPPED_V015_PACKAGES: tuple[str, ...] = ()
_UNSHIPPED_SPHINX_ROLE = re.compile(
    r":(?:mod|class|func|data):`~?koopman_graph\.(?:%s)(?:\.[\w.]*)?`"
    % ("|".join(_UNSHIPPED_V015_PACKAGES) or "NEVER_MATCH_UNSHIPPED")
)
_REQUIRED_PAPER_MD_CITES = (
    "Azencot2020",
    "Bruder2021",
    "Li2017EDMD",
    "Li2018DCRNN",
    "Li2020CompositionalKoopman",
    "Korda2018",
    "Mukherjee2022",
    "Nandanoori2022",
    "Proctor2016DMDc",
    "Williams2015",
    "Wu2019WaveNet",
    "Yu2018STGCN",
)


def test_literature_precedent_citations_in_paper_sources() -> None:
    """Require bib entries and paper.md cites for headline literature precedents."""
    bib_text = (_PROJECT_ROOT / "paper.bib").read_text()
    paper_text = (_PROJECT_ROOT / "paper.md").read_text()
    readme = (_PROJECT_ROOT / "README.md").read_text()
    missing_bib = [
        key
        for key in _REQUIRED_PAPER_BIB_KEYS
        if not re.search(rf"@\w+\{{{re.escape(key)}\s*,", bib_text)
    ]
    missing_cites = [
        key for key in _REQUIRED_PAPER_MD_CITES if f"@{key}" not in paper_text
    ]
    assert not missing_bib, f"paper.bib missing keys: {missing_bib}"
    assert not missing_cites, f"paper.md missing citations: {missing_cites}"
    assert "not claimed as a new theoretical contribution" in readme
    assert (
        "consistent Koopman autoencoder lineage" in readme
        or "consistent-autoencoder" in readme
    )


def test_v070_bibliography_entries_are_cited() -> None:
    """v0.7.0 method keys must exist in paper.bib and appear in src/ or docs."""
    bib_text = (_PROJECT_ROOT / "paper.bib").read_text(encoding="utf-8")
    corpus_parts: list[str] = []
    for root_name in ("src", "docs/source"):
        root = _PROJECT_ROOT / root_name
        for path in root.rglob("*"):
            if path.suffix in {".py", ".rst", ".md"} and path.is_file():
                corpus_parts.append(path.read_text(encoding="utf-8"))
    corpus = "\n".join(corpus_parts)

    missing_bib = [
        key
        for key in _V070_BIB_KEYS
        if not re.search(rf"@\w+\{{{re.escape(key)}\s*,", bib_text)
    ]
    missing_cites = [key for key in _V070_BIB_KEYS if key not in corpus]
    assert not missing_bib, f"paper.bib missing v0.7.0 keys: {missing_bib}"
    assert not missing_cites, f"uncited v0.7.0 bib keys: {missing_cites}"

    # DOI-bearing entries (PMLR Zargarbashi has a stable URL, not a DOI).
    for key, doi_fragment in (
        ("ColbrookTownsend2023ResDMD", "10.1002/cpa.22125"),
        ("Colbrook2023ResidualDMD", "10.1017/jfm.2022.1052"),
        ("GavishDonoho2014", "10.1109/TIT.2014.2323359"),
        ("Rosenstein1993Lyapunov", "10.1016/0167-2789(93)90009-P"),
        ("Welch1967", "10.1109/TAU.1967.1161901"),
    ):
        assert doi_fragment in bib_text, f"{key} missing DOI {doi_fragment}"
    assert "proceedings.mlr.press/v202/h-zargarbashi23a.html" in bib_text


def test_v015_bibliography_entries_are_cited() -> None:
    """Post-0.14 method keys must exist in paper.bib and appear in src/ or docs."""
    bib_text = (_PROJECT_ROOT / "paper.bib").read_text(encoding="utf-8")
    corpus_parts: list[str] = []
    for root_name in ("src", "docs/source"):
        root = _PROJECT_ROOT / root_name
        for path in root.rglob("*"):
            if path.suffix in {".py", ".rst", ".md"} and path.is_file():
                corpus_parts.append(path.read_text(encoding="utf-8"))
    corpus = "\n".join(corpus_parts)

    missing_bib = [
        key
        for key in _V015_BIB_KEYS
        if not re.search(rf"@\w+\{{{re.escape(key)}\s*,", bib_text)
    ]
    missing_cites = [key for key in _V015_BIB_KEYS if key not in corpus]
    assert not missing_bib, f"paper.bib missing v0.15 keys: {missing_bib}"
    assert not missing_cites, f"uncited v0.15 bib keys: {missing_cites}"

    for key, locator in (
        ("Colbrook2023mpEDMD", "10.1137/22M1521407"),
        ("Klus2020gEDMD", "10.1016/j.physd.2020.132416"),
        ("HaseliCortes2023", "2311.13033"),
        ("Guo2025ModularEDMD", "10.1016/j.physd.2025.134651"),
        ("Macesic2018Nonautonomous", "10.1137/17M1133610"),
        ("Zeng2022Sampling", "10.1109/CDC51059.2022.9992482"),
        ("Pan2021SparseSubspace", "10.1017/jfm.2021.271"),
        ("Ruiz2023Transferability", "10.1109/TSP.2023.3297848"),
        ("Zhang2022TubeMPC", "10.1016/j.automatica.2021.110114"),
        ("Xu2025ResKoopNet", "proceedings.mlr.press/v267/xu25y.html"),
        ("LibCity2021", "10.1145/3474717.3483923"),
        ("BasicTS2024", "10.1109/TKDE.2024.3484454"),
        ("kooplearn2026", "10.21105/joss.10342"),
        ("PyKoopman2024", "10.21105/joss.05881"),
        ("deeptime2021", "10.1088/2632-2153/ac3de0"),
        ("PyDMD2018", "10.21105/joss.00530"),
        ("TopoX2024", "2402.02441"),
    ):
        assert locator in bib_text, f"{key} missing locator {locator}"


def test_readme_highlights_identification_and_benchmarks() -> None:
    """README Highlights must name identification and identity-bound benchmarks."""
    readme = (_PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    version = koopman_graph.__version__
    assert "## Highlights" in readme
    assert "Identification (opt-in)" in readme
    assert "Identity-bound benchmarks" in readme
    assert "does not train" in readme.lower()
    assert "not a LibCity / BasicTS host" in readme
    assert "not Ray Tune" in readme
    assert f"new in {version}" in readme
    assert f"version      = {{{version}}}" in readme
    assert "0.2.0" not in readme


def test_release_metadata_versions_agree() -> None:
    """Package, CHANGELOG, CITATION.cff, README, and paper.bib versions agree."""
    version = koopman_graph.__version__
    changelog = (_PROJECT_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    readme = (_PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    bib_text = (_PROJECT_ROOT / "paper.bib").read_text(encoding="utf-8")
    cff_text = (_PROJECT_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    index = (_PROJECT_ROOT / "docs" / "source" / "index.rst").read_text(
        encoding="utf-8"
    )
    assert "## [Unreleased]" in changelog
    assert f"## [{version}]" in changelog
    assert _cff_top_level_field(cff_text, "version") == version
    assert _cff_top_level_field(cff_text, "doi") == "10.5281/zenodo.21926723"
    assert f"version      = {{{version}}}" in readme
    assert f"Version **{version}**" in index
    bib_version = _bib_field(_bib_entry(bib_text, _SOFTWARE_BIB_KEY), "version")
    assert bib_version == version


def _bib_entry(bib_text: str, key: str) -> str:
    """Return the BibTeX body for ``key`` (braces exclusive of the entry type)."""
    match = re.search(rf"@\w+\{{{re.escape(key)}\s*,", bib_text)
    if match is None:
        msg = f"paper.bib missing entry {key}"
        raise AssertionError(msg)
    start = bib_text.find("{", match.start())
    depth = 0
    for index in range(start, len(bib_text)):
        char = bib_text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return bib_text[start + 1 : index]
    msg = f"paper.bib entry {key} is unclosed"
    raise AssertionError(msg)


def _bib_field(entry: str, field: str) -> str:
    """Return a single braced BibTeX field value."""
    match = re.search(
        rf"(?im)^\s*{re.escape(field)}\s*=\s*\{{([^}}]*)\}}",
        entry,
    )
    if match is None:
        msg = f"paper.bib field {field} missing"
        raise AssertionError(msg)
    return match.group(1).strip()


def _cff_top_level_field(cff_text: str, field: str) -> str:
    """Return a top-level YAML scalar from CITATION.cff (no nested keys)."""
    match = re.search(rf"(?m)^{re.escape(field)}:\s*(\S+)\s*$", cff_text)
    if match is None:
        msg = f"CITATION.cff missing top-level {field}"
        raise AssertionError(msg)
    return match.group(1).strip().strip("\"'")


def test_software_citation_matches_citation_cff() -> None:
    """Software bib entry version/DOI must match CITATION.cff (not stale 0.11.0)."""
    bib_text = (_PROJECT_ROOT / "paper.bib").read_text(encoding="utf-8")
    cff_text = (_PROJECT_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    entry = _bib_entry(bib_text, _SOFTWARE_BIB_KEY)
    bib_version = _bib_field(entry, "version")
    bib_doi = _bib_field(entry, "doi")
    cff_version = _cff_top_level_field(cff_text, "version")
    cff_doi = _cff_top_level_field(cff_text, "doi")
    assert bib_version == cff_version, (
        f"{_SOFTWARE_BIB_KEY} version {bib_version!r} != CITATION.cff {cff_version!r}"
    )
    assert bib_doi == cff_doi, (
        f"{_SOFTWARE_BIB_KEY} doi {bib_doi!r} != CITATION.cff {cff_doi!r}"
    )
    assert _STALE_SOFTWARE_VERSION not in entry
    assert _STALE_SOFTWARE_DOI_RECORD not in entry


def test_isotypic_neighbor_tying_not_described_as_unshipped() -> None:
    """Present-tense docs must not claim neighbor-factor isotypic tying is unshipped."""
    hits: list[str] = []
    texts: dict[str, str] = {}
    for relative in _ISOTYPIC_DOC_RELATIVE_PATHS:
        path = _PROJECT_ROOT / relative
        text = path.read_text(encoding="utf-8")
        texts[relative] = text
        for pattern in _ISOTYPIC_UNSHIPPED_PRESENT_TENSE:
            if pattern.search(text):
                hits.append(f"{relative}: {pattern.pattern}")
    assert not hits, (
        "present-tense unshipped isotypic-tying wording remains in:\n" + "\n".join(hits)
    )
    assert "neighbor-factor ties" in texts["docs/source/capabilities.rst"]
    assert (
        "Neighbor-factor isotypic tying is shipped"
        in texts["docs/source/limitations.rst"]
    )
    assert "was not included in 0.11.0" in texts["CHANGELOG.md"]
    assert (
        "neighbor-factor tying added in 0.14" in texts["docs/source/architecture.rst"]
    )


_V015_GUIDE_PAGES = (
    "benchmarks",
    "identification",
    "spectral_diagnostics",
    "graph_dynamics",
    "matrix_free",
    "criticality",
    "time_conditioning",
)


def test_v015_guide_pages_exist_and_are_in_toctree() -> None:
    """v0.15 guide pages must be tracked RST files listed in the Sphinx toctree."""
    index = (_PROJECT_ROOT / "docs" / "source" / "index.rst").read_text(
        encoding="utf-8"
    )
    for name in _V015_GUIDE_PAGES:
        path = _PROJECT_ROOT / "docs" / "source" / f"{name}.rst"
        assert path.is_file(), f"missing docs/source/{name}.rst"
        assert f"\n   {name}\n" in index or f"\n   {name}\r\n" in index, (
            f"{name} missing from docs/source/index.rst toctree"
        )
        text = path.read_text(encoding="utf-8")
        assert "blueprint" not in text.lower()
        assert "DESIGN-v" not in text
        assert ":doc:`limitations`" in text or "limitations" in text


def test_limitations_rehomes_v015_ceilings() -> None:
    """Public limitations page must carry the post-0.14 honesty ceilings."""
    text = (_PROJECT_ROOT / "docs" / "source" / "limitations.rst").read_text(
        encoding="utf-8"
    )
    folded = _folded_rst(text)
    assert "identification=None" in folded
    assert "Adam" in folded
    assert "MpEDMDBaseline" in folded
    assert "SpectralDiagnostics" in folded
    assert "fixed-union" in folded
    assert "koopman_filter_degree=1" in folded
    assert "Nyquist" in folded
    assert "CochainKoopmanOperator" in folded
    assert "validate_uniform_discrete_increments" in folded
    assert "monitor_critical_transition" in folded
    assert "0.6551" in folded
    assert "hop order" in folded
    assert "Neighbor-factor isotypic tying is shipped" in text
    assert "Negative transfer" in folded
    assert "MassConservingDecoder" in folded
    assert "MAX_DENSE_LINEAR_OPERATOR_SIZE" in folded
    assert "blueprint" not in text.lower()
    assert "DESIGN-v" not in text


def test_contributing_does_not_freeze_package_version() -> None:
    """CONTRIBUTING must point at __init__.py, not freeze a package version string."""
    text = (_PROJECT_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    assert "src/koopman_graph/__init__.py" in text
    assert _STALE_CONTRIBUTING_VERSION not in text
    match = _FROZEN_PACKAGE_VERSION_ASSIGNMENT.search(text)
    assert match is None, (
        f"frozen __version__ assignment in CONTRIBUTING: {match.group(0)!r}"
    )


def _architecture_section(heading: str) -> str:
    """Return architecture.rst from ``heading`` through the next top-level section."""
    text = (_PROJECT_ROOT / "docs/source/architecture.rst").read_text(encoding="utf-8")
    start = text.index(heading)
    related = text.index("Related documentation", start)
    return text[start:related]


def _folded_rst(text: str) -> str:
    """Collapse RST wrapping so contract phrases can be matched."""
    return re.sub(r"\s+", " ", text)


def test_architecture_v015_scaffold_preserves_defaults_and_import_rules() -> None:
    """v0.15 architecture scaffold must keep LTI defaults and planned import rules."""
    section = _architecture_section(_V015_ARCHITECTURE_HEADING)
    folded = _folded_rst(section)
    assert "koopman=None" in folded
    assert '"pernode"' in folded
    assert "linear time-invariant per-node" in folded
    assert "koopman_filter_degree" in folded
    assert "default ``1``" in folded
    assert "FORMAT_VERSION" in folded
    assert "No other ``koopman_graph`` package imports ``benchmark``" in folded
    assert "``training`` must not import ``identification`` at module load" in folded
    assert "``identification`` must not import ``adaptation``" in folded
    assert "EntityRemap" in folded
    assert "estimate_graphon" in folded
    assert "ConditioningContext" in folded
    assert "diurnal_control_features" in folded
    assert "ParametricKoopmanOperator" in folded
    assert "DriftDiffusionKoopman" in folded
    assert "JointCoverageSpec" in folded
    assert "markov_closure_report" in folded
    assert "FiniteMemoryKoopman" in folded
    assert "CochainKoopmanOperator" in folded
    assert "CochainState" in folded
    assert "hodge_decompose_modes" in folded
    assert "HodgeModeComponents" in folded
    assert "order2_cochain_teaching" in folded
    assert "MAX_CELL_COMPLEX_DEGREE" in folded
    assert "EquivariantKoopmanOperator" in folded
    assert "n_tensors" in folded
    assert "MassConservingDecoder" in folded
    assert "LinearConservingDecoder" in folded
    assert "constraint_decoders" in folded
    assert "LinearOperatorProtocol" in folded
    assert "PolynomialGraphLinearOperator" in folded
    assert "MAX_DENSE_LINEAR_OPERATOR_SIZE" in folded
    assert "TubeKoopmanMPC" in folded
    assert "TubeMPCReport" in folded
    assert "SyntheticSCM" in folded
    assert "SyntheticInterventionReport" in folded
    assert "recover_synthetic_interventional_edges" in folded
    assert "koopman_graph.identification" in folded
    for name in _UNSHIPPED_V015_PACKAGES:
        assert f"koopman_graph.{name}" in folded
    assert "Planned" in folded
    assert "koopman_graph.benchmark" in folded
    role = _UNSHIPPED_SPHINX_ROLE.search(section)
    assert role is None, (
        f"unshipped package Sphinx role in v0.15 scaffold: {role.group(0)!r}"
    )


def test_joss_paper_narrative_word_count_at_most_1000() -> None:
    """JOSS narrative body must stay within the ≤1000-word gate."""
    paper_text = (_PROJECT_ROOT / "paper.md").read_text()
    body = re.sub(r"^---\n.*?\n---\n", "", paper_text, count=1, flags=re.S)
    body = re.split(r"^# References\s*$", body, maxsplit=1, flags=re.M)[0]
    n = len(re.findall(r"\b[\w'-]+\b", body))
    assert n <= 1000, f"paper.md narrative word count {n} exceeds 1000"
    assert not re.search(r"\\url\{\[", paper_text), (
        "malformed hybrid \\url{[...]} markup must not reappear in paper.md"
    )
