# Contributing to KoopmanGraph

Thank you for your interest in contributing to **KoopmanGraph**. This project
aims to provide a well-tested, documented, and reproducible open-source library
for spatiotemporal graph dynamics using Koopman operator theory. Community
contributions—bug reports, documentation improvements, tests, and new
features—are welcome and help us meet the standards expected of
[Journal of Open Source Software (JOSS)](https://joss.theoj.org/) submissions.

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

## Documentation

User-facing documentation is hosted on
[Read the Docs](https://koopmangraph.readthedocs.io/).

Update documentation when your change affects:

- Public API (classes, functions, parameters exported from `koopman_graph`)
- Installation or usage workflows
- Tutorials or examples

Sphinx source lives in `docs/source/`. Rebuild locally with `make html` in
`docs/` after installing `[docs]` extras.

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
