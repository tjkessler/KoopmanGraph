# Contributing to KoopmanGraph

Thank you for your interest in contributing to **KoopmanGraph**. This project
aims to provide a well-tested, documented, and reproducible open-source library
for spatiotemporal graph dynamics using Koopman operator theory. Community
contributions—bug reports, documentation improvements, tests, and new
features—are welcome and help keep the project well-tested, documented, and
reproducible for the research community.

This document describes how to set up a development environment, run quality
checks locally, and submit changes via pull request.

## Table of contents

- [Code of conduct](#code-of-conduct)
- [Ways to contribute](#ways-to-contribute)
- [Support](#support)
- [Development environment](#development-environment)
- [Running checks locally](#running-checks-locally)
- [Testing and coverage requirements](#testing-and-coverage-requirements)
- [Code style](#code-style)
- [Pull request process](#pull-request-process)
- [Documentation](#documentation)
- [Releasing](#releasing)
- [License](#license)

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md) (v2.1).
Please be respectful and constructive in all project interactions. To report
unacceptable behavior, email the maintainer at
[travis.j.kessler@gmail.com](mailto:travis.j.kessler@gmail.com) (see
[Enforcement](CODE_OF_CONDUCT.md#enforcement) in the full code of conduct).

## Ways to contribute

You can contribute without writing code:

- **Report bugs** using the [bug report template](.github/ISSUE_TEMPLATE/bug_report.yml).
- **Request features** using the [feature request template](.github/ISSUE_TEMPLATE/feature_request.yml).
- **Improve documentation** (README, Sphinx pages, tutorials, docstrings).
- **Add or extend tests** to improve coverage and guard against regressions.

For code changes, please follow the development setup and pull request process
below.

## Support

Need help using KoopmanGraph? Prefer the channels below so maintainers and other
users can find questions and answers in one place.

| Kind of request | Where to go |
|-----------------|-------------|
| Usage / how-to questions, modeling advice, “am I holding this wrong?” | [GitHub Discussions](https://github.com/tjkessler/KoopmanGraph/discussions) |
| Suspected bugs (crashes, incorrect results, install failures with a repro) | [Bug report](https://github.com/tjkessler/KoopmanGraph/issues/new?template=bug_report.yml) |
| New features or API changes | [Feature request](https://github.com/tjkessler/KoopmanGraph/issues/new?template=feature_request.yml) |

**Bug report vs usage question:** If you can show unexpected behavior with a
minimal reproducible example (versions, code, and expected vs actual output),
open a bug report. If you are unsure how to configure a model, interpret
outputs, or choose an API for your problem, start a Discussion instead.

Blank issues are disabled; the issue chooser also links to Discussions for
non-bug questions. Responses are **best-effort** from maintainers and the
community—there is **no SLA** or guaranteed response time.

For common install, import-path, and checkpoint failures, see the Sphinx
[FAQ / troubleshooting](https://koopmangraph.readthedocs.io/en/latest/faq.html)
page. To report a **security vulnerability**, use the private channel in
[SECURITY.md](SECURITY.md)—do not open a public issue.

## Development environment

### Prerequisites

KoopmanGraph requires:

| Requirement | Version |
|-------------|---------|
| Python | 3.10, 3.11, or 3.12 |
| PyTorch | ≥ 2.8 (CPU or CUDA) |
| PyTorch Geometric | ≥ 2.6 |

Install **PyTorch** and **PyTorch Geometric** before installing KoopmanGraph so
your installer can resolve compatible wheels for your platform. See the
[installation guide](https://koopmangraph.readthedocs.io/en/latest/installation.html)
for platform-specific instructions (pip and [uv](https://docs.astral.sh/uv/)).

### Clone and install

```bash
git clone https://github.com/tjkessler/KoopmanGraph.git
cd KoopmanGraph
pip install -e ".[dev]"
```

With [uv](https://docs.astral.sh/uv/) (creates `.venv`, uses the repo’s CPU
PyTorch index by default via `[tool.uv]` in `pyproject.toml`):

```bash
git clone https://github.com/tjkessler/KoopmanGraph.git
cd KoopmanGraph
uv sync --extra dev
uv run pytest
```

For CUDA / ROCm / XPU instead of the default CPU wheels, install PyTorch with
`uv pip` backend selection first (see Astral’s
[Using uv with PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/)
guide), then sync without replacing that torch build—or use
`uv pip install -e ".[dev]"` after `uv venv`.

The `[dev]` extra installs pytest, pytest-cov, Ruff, and pre-commit.

### Optional: documentation dependencies

To build Sphinx documentation locally:

```bash
pip install -e ".[docs]"
# or: uv sync --extra docs
cd docs && make html
```

Built HTML appears under `docs/build/html/`.

### Optional: pre-commit hooks

Install hooks so Ruff and basic file checks run before each commit:

```bash
pre-commit install
# or: uv run pre-commit install
```

Run hooks on all files without committing:

```bash
pre-commit run --all-files
```

Hook definitions are in `.pre-commit-config.yaml`.

## Running checks locally

Before opening a pull request, run the same checks enforced in
[CI](.github/workflows/ci.yml):

```bash
# Lint and format
ruff check src/ tests/
ruff format --check src/ tests/

# Tests with coverage gate (matches the CI 3.12 job; use -n auto for speed)
pytest tests/ -n auto --cov=koopman_graph --cov-report=term-missing --cov-fail-under=90
```

With uv after `uv sync --extra dev`:

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run pytest tests/ -n auto --cov=koopman_graph --cov-report=term-missing --cov-fail-under=90
```

To auto-fix lint issues and apply formatting:

```bash
ruff check --fix src/ tests/
ruff format src/ tests/
```

## Testing and coverage requirements

All contributions that change behavior must include appropriate tests in the
`tests/` directory.

- **New features** should include unit or integration tests demonstrating
  correct behavior and, where applicable, gradient flow or shape contracts.
- **Bug fixes** should include a regression test that fails without the fix.
- **Refactors** should preserve existing test coverage; do not reduce coverage
  below the project threshold without maintainer approval.

### Coverage threshold

The project enforces a minimum of **90% line coverage** on `koopman_graph`:

- Configuration: `[tool.coverage.report]` in `pyproject.toml` (`fail_under = 90`)
- CI: `--cov-fail-under=90` in `.github/workflows/ci.yml`
- Codecov: project and patch `target: 90%` in `codecov.yml` (uploaded from CI)

Generate a local coverage report:

```bash
pytest tests/ --cov=koopman_graph --cov-report=term-missing --cov-fail-under=90
```

Pull requests that drop coverage below 90% will fail CI (and Codecov status checks).

## Code style

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and
formatting. Configuration lives in `pyproject.toml` under `[tool.ruff]`.

Enabled rule sets include:

- **E**, **F** — pycodestyle and Pyflakes
- **I** — import sorting
- **UP** — pyupgrade (modern Python syntax)
- **B** — flake8-bugbear
- **SIM** — flake8-simplify

Match the style of surrounding code. Prefer clear, self-documenting
implementations over excessive comments. Public API changes should include
docstring updates (see [Documentation](#documentation)).

## Pull request process

1. **Fork** the repository and create a feature branch from `main`.
2. **Implement** your change with focused commits.
3. **Run local checks** (Ruff and pytest with coverage) as described above.
4. **Open a pull request** against `main`. The
   [PR template](.github/pull_request_template.md) will prompt you for:
   - A summary of changes and motivation
   - Related issue links (if applicable)
   - Confirmation that tests and lint pass locally
   - Documentation updates when the public API changes
5. **Address review feedback** and ensure CI passes (see required `ci` check;
   the matrix covers Python 3.10 and 3.12 when tests run).

Maintainers may request changes before merging. We aim to review pull requests
in a timely manner and appreciate clear descriptions and test coverage.

### Required status checks (maintainers)

Branch protection on `main` is configured through a GitHub ruleset
(**Settings → Rules → Rulesets**), not through committed files.
Mark the aggregator check from [`ci.yml`](.github/workflows/ci.yml) as
**required** before merge:

- `ci` — succeeds when path-selected jobs (`lint`, `test`, `notebooks`,
  `docs`) succeed or are intentionally skipped; fails if any selected job fails

Individual jobs still appear on the workflow run for diagnosis:

- `lint` — Ruff via `uvx` (no full torch install)
- `test (3.10)`, `test (3.12)` — pytest with `pytest-xdist`; coverage gate on 3.12 only
- `notebooks` — tutorial nbmake smoke tests (Python 3.12)
- `docs` — Sphinx documentation build (`sphinx-build -W`, warnings as errors)

The **Draft paper** workflow ([`draft-pdf.yml`](.github/workflows/draft-pdf.yml))
compiles `paper.md` on changes; it is informational and need not block merges.

## Documentation

User-facing documentation is hosted on
[Read the Docs](https://koopmangraph.readthedocs.io/).

Update documentation when your change affects:

- Public API (classes, functions, parameters exported from `koopman_graph`)
- Installation or usage workflows
- Tutorials or examples

Sphinx source lives in `docs/source/`. Rebuild locally with `make html` in
`docs/` after installing `[docs]` extras.

Before changing package layout, `__all__` exports, shared helpers, or device
handling, read the maintainer architecture guide:

- Sphinx: [`docs/source/architecture.rst`](docs/source/architecture.rst)
  (built page: *Architecture and API layers* on Read the Docs)
- It defines the public façade vs power-user modules vs private (`_`-prefixed)
  helpers (**thin root `__all__`**: keep core workflow symbols; demote
  specialized helpers to `metrics` / `analysis` / `data` / `adaptation` /
  `observables`), device/tensor conventions across `fit`, the RL env, online
  adaptation, and classical baselines/datasets, plus **optional-dependency**
  (fail-at-call / `[rl]` soft-import) and **frozen dataclass result-type**
  conventions.
- **Package layout** (same page): when to nest into one-level capability
  packages vs stay flat; keep `model.py` at the package root; no deep trees;
  three-layer API preserved under any folder move; compatibility contract for
  root `__all__` vs power-user deep imports (same-named packages or in-repo
  migration — no long-lived root shim modules).

## Releasing

KoopmanGraph follows [Semantic Versioning](https://semver.org/) (`MAJOR.MINOR.PATCH`).
While the project is pre-1.0, breaking API changes may appear in `0.x.y` releases;
document them in [CHANGELOG.md](CHANGELOG.md) and in GitHub Release notes.

### Version source of truth

Bump the version in a single place:

```python
# src/koopman_graph/__init__.py
__version__ = "0.5.0"
```

`pyproject.toml` reads this value dynamically at build time via
`[tool.setuptools.dynamic]`. Do not add a separate static `version` field to
`pyproject.toml`.

### Checkpoint format (current baseline)

Model checkpoints include a ``format_version`` field (see
``src/koopman_graph/serialization.py``):

| Version | Status | Notes |
| --- | --- | --- |
| 1 | Current baseline | Full config: discrete/continuous dynamics, hybrid physics, control (including bilinear metadata), delay embeddings, built-in operator kinds (``koopman_kind``), auxiliary-spectral settings, and GCN/GAT/SAGE/DiffConv/Transformer encoder-decoder types. Missing decoder ``type`` defaults to ``"gcn"``. |

``GraphKoopmanModel.load`` accepts only supported format versions (currently
``{1}``). Previously published format-2 checkpoints and sparse historical
format-1 payloads that omit required current-schema keys are rejected with a
clear error — there is no silent migration from those retired lineages.
Future incompatible schema changes must bump ``FORMAT_VERSION``, extend
``SUPPORTED_FORMAT_VERSIONS``, and add an explicit migration branch in
``_migrate_config``. Custom injected operators are rejected on ``save`` and
are therefore outside the checkpoint path.

Do not confuse checkpoint ``format_version`` with the package
``__version__`` (for example ``0.5.0``).

### Maintainer release checklist

1. Ensure `main` is green (CI tests, lint, notebook smoke tests).
2. Update `__version__` in `src/koopman_graph/__init__.py`.
3. Update [CHANGELOG.md](CHANGELOG.md) with user-facing notes for the new version
   (features, changes, fixes, and any breaking notes).
4. Merge any pending release-prep changes to `main`.
5. Create a GitHub Release tagged `vX.Y.Z` (for example `v0.3.0`). Publishing the
   release triggers `.github/workflows/release.yml`.
6. Approve the `pypi` environment deployment if required by branch protection.
7. Confirm the workflow uploaded artifacts to PyPI.
8. Verify installation:

   ```bash
   pip install koopman-graph==X.Y.Z
   python -c "import koopman_graph; print(koopman_graph.__version__)"
   ```

### PyPI trusted publishing

Releases use [PyPI trusted publishing](https://docs.pypi.org/trusted-publishers/)
(OpenID Connect) — no long-lived API tokens are stored in GitHub Secrets.

One-time setup (maintainers):

1. On [pypi.org](https://pypi.org/manage/account/publishing/), register a
   **pending publisher** (or add a publisher to an existing project) with:
   - PyPI project name: `koopman-graph`
   - Owner: `tjkessler`
   - Repository: `KoopmanGraph`
   - Workflow: `release.yml`
   - Environment: `pypi`
2. In GitHub **Settings → Environments**, create the `pypi` environment (optionally
   with required reviewers).

The release workflow requests short-lived upload credentials automatically via
[`pypa/gh-action-pypi-publish`](https://github.com/pypa/gh-action-pypi-publish).

### Software archiving (Zenodo)

For a citable, archived snapshot of the software, the repository is set up to
deposit tagged releases to [Zenodo](https://zenodo.org/):

- **[`.zenodo.json`](.zenodo.json)** supplies archive metadata (title, description,
  Apache-2.0 license, author + ORCID, keywords). Zenodo fills in the `version` and
  publication date automatically from the GitHub release tag, so those are not
  hard-coded.
- **[`CITATION.cff`](CITATION.cff)** provides citation metadata surfaced by GitHub's
  "Cite this repository" feature.

One-time setup (maintainers), performed **before** the archival release:

1. Sign in to [zenodo.org](https://zenodo.org/) with GitHub and enable archiving
   for the `tjkessler/KoopmanGraph` repository (Zenodo adds a release webhook).
2. Confirm `.zenodo.json` metadata is current.

At archival release time:

1. Create the GitHub Release (`vX.Y.Z`). Zenodo captures the tagged snapshot and
   mints a **version DOI** plus a **concept DOI** (all versions).
2. Add the version DOI to `CITATION.cff` (uncomment the `doi:`/`identifiers` block)
   and replace the `DOI: TBD` placeholder in `paper.md`.
3. Add the Zenodo DOI badge to the README.

Enable Zenodo archiving and mint the version DOI when you cut a release you intend
to cite; the metadata files above are prepared in advance so the archival step is
straightforward at release time.

### Local build (debugging)

To build wheels locally without publishing:

```bash
pip install build
python -m build
ls dist/
```

Install a built wheel in a clean virtual environment (install PyTorch and PyG first):

```bash
pip install torch torch-geometric
pip install dist/koopman_graph-*.whl
```

## License

By contributing to KoopmanGraph, you agree that your contributions will be
licensed under the [Apache License 2.0](LICENSE), consistent with the rest of
the project.
