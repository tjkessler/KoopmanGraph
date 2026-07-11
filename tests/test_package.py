"""Smoke tests for package installation and imports."""

import koopman_graph


def test_import_package() -> None:
    """Verify the package imports and exposes ``__version__``."""
    assert koopman_graph.__version__ == "0.1.0"
