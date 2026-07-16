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
- [Development environment](#development-environment)
- [Running checks locally](#running-checks-locally)
- [Testing and coverage requirements](#testing-and-coverage-requirements)
- [Code style](#code-style)
- [Pull request process](#pull-request-process)
- [Documentation](#documentation)
- [Releasing](#releasing)
- [Agent-assisted development](#agent-assisted-development)
- [License](#license)

## Code of conduct

Please be respectful and constructive in all project interactions. Report
behavior concerns through [GitHub Issues](https://github.com/tjkessler/KoopmanGraph/issues)
or by contacting the maintainers listed in `pyproject.toml`.

## Ways to contribute

You can contribute without writing code:

- **Report bugs** using the [bug report template](.github/ISSUE_TEMPLATE/bug_report.yml).
- **Request features** using the [feature request template](.github/ISSUE_TEMPLATE/feature_request.yml).
- **Improve documentation** (README, Sphinx pages, tutorials, docstrings).
- **Add or extend tests** to improve coverage and guard against regressions.

For code changes, please follow the development setup and pull request process
below.

## Development environment

### Prerequisites

KoopmanGraph requires:

| Requirement | Version |
|-------------|---------|
| Python | 3.10, 3.11, or 3.12 |
| PyTorch | ≥ 2.8 (CPU or CUDA) |
| PyTorch Geometric | ≥ 2.6 |

Install **PyTorch** and **PyTorch Geometric** before installing KoopmanGraph so
pip can resolve compatible wheels for your platform. See the
[installation guide](https://koopmangraph.readthedocs.io/en/latest/installation.html)
for platform-specific instructions.

### Clone and install

```bash
git clone https://github.com/tjkessler/KoopmanGraph.git
cd KoopmanGraph
pip install -e ".[dev]"
```

The `[dev]` extra installs pytest, pytest-cov, Ruff, and pre-commit.

### Optional: documentation dependencies

To build Sphinx documentation locally:

```bash
pip install -e ".[docs]"
cd docs && make html
```

Built HTML appears under `docs/build/html/`.

### Optional: pre-commit hooks

Install hooks so Ruff and basic file checks run before each commit:

```bash
pre-commit install
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

# Tests with coverage gate
pytest tests/ -v --cov=koopman_graph --cov-report=term-missing --cov-fail-under=80
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

The project enforces a minimum of **80% line coverage** on `koopman_graph`:

- Configuration: `[tool.coverage.report]` in `pyproject.toml` (`fail_under = 80`)
- CI: `--cov-fail-under=80` in `.github/workflows/ci.yml`

Generate a local coverage report:

```bash
pytest tests/ --cov=koopman_graph --cov-report=term-missing --cov-fail-under=80
```

Pull requests that drop coverage below 80% will fail CI.

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
5. **Address review feedback** and ensure CI passes on Python 3.10, 3.11, and 3.12.

Maintainers may request changes before merging. We aim to review pull requests
in a timely manner and appreciate clear descriptions and test coverage.

### Required status checks (maintainers)

Branch protection on `main` is configured through GitHub
(**Settings → Branches → Branch protection rules**), not through committed files.
The following checks from [`ci.yml`](.github/workflows/ci.yml) should be marked
as **required** before merge:

- `test (3.10)`, `test (3.11)`, `test (3.12)` — pytest, Ruff lint/format, and the
  80% coverage gate across supported Python versions
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

## Releasing

KoopmanGraph follows [Semantic Versioning](https://semver.org/) (`MAJOR.MINOR.PATCH`).
While the project is pre-1.0, breaking API changes may appear in `0.x.y` releases;
document them in release notes.

### Version source of truth

Bump the version in a single place:

```python
# src/koopman_graph/__init__.py
__version__ = "0.3.0"
```

`pyproject.toml` reads this value dynamically at build time via
`[tool.setuptools.dynamic]`. Do not add a separate static `version` field to
`pyproject.toml`.

### Checkpoint migration (v0.3.0)

Model checkpoints include a ``format_version`` field (see
``src/koopman_graph/serialization.py``):

| Version | Introduced | Notes |
| --- | --- | --- |
| 1 | v0.2.x | Discrete dynamics; config omits continuous-time, physics, and control fields |
| 2 | v0.3.0 | Full config for continuous mode, hybrid physics observables, and control |

``GraphKoopmanModel.load`` accepts both v1 and v2 checkpoints. v1 payloads are
migrated in memory by filling missing optional fields with defaults
(``dynamics_mode="discrete"``, no physics observables, ``control_dim=0``,
``koopman_parameterization="dense"``). New saves always use the current
``FORMAT_VERSION`` (2 as of v0.3.0).

### Maintainer release checklist

1. Ensure `main` is green (CI tests, lint, notebook smoke tests).
2. Update `__version__` in `src/koopman_graph/__init__.py`.
3. Merge any pending release-prep changes to `main`.
4. Create a GitHub Release tagged `vX.Y.Z` (for example `v0.3.0`). Publishing the
   release triggers `.github/workflows/release.yml`.
5. Approve the `pypi` environment deployment if required by branch protection.
6. Confirm the workflow uploaded artifacts to PyPI.
7. Verify installation:

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

## Agent-assisted development

This repository maintains a development blueprint for structured, task-oriented
work—particularly useful when using AI coding agents:

- **[Development Blueprint](docs/BLUEPRINT.md)** — phased task tracker with
  acceptance criteria, dependencies, and agent logs.

Agents working from the blueprint should read one task at a time, present a
plan for user approval, and update the blueprint status upon completion. Human
contributors may use the blueprint as a roadmap but are not required to follow
it.

## License

By contributing to KoopmanGraph, you agree that your contributions will be
licensed under the [Apache License 2.0](LICENSE), consistent with the rest of
the project.
