"""Sphinx configuration for KoopmanGraph."""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

project = "KoopmanGraph"
author = "Travis Kessler"
copyright = "2026, Travis Kessler"

try:
    from koopman_graph import __version__ as release
except ImportError:
    release = "0.5.0"

version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]

# Explicit HTML math renderer for documentation formulas.
# Prefer MathJax over imgmath so CI/ReadTheDocs do not require a LaTeX install.
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"

templates_path = ["_templates"]
exclude_patterns: list[str] = []

html_theme = "furo"
html_static_path = ["_static"]
html_logo = "_static/koopmangraph_logo.png"
html_title = "KoopmanGraph"

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "show-inheritance": True,
}

autodoc_property_in_members = False

autodoc_typehints = "description"
autodoc_typehints_description_target = "documented_params"

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
    "torch_geometric": ("https://pytorch-geometric.readthedocs.io/en/latest/", None),
}

suppress_warnings = ["ref.python"]
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
