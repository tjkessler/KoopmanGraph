"""Config-driven CLI for KoopmanGraph (``koopman-graph`` console script).

Power-user package: import as ``koopman_graph.cli``. Intentionally omitted
from root ``koopman_graph.__all__``. Other library modules must not import
this package (CLI is a leaf façade).
"""

from koopman_graph.cli.config import (
    ConfigError,
    load_config,
    load_train_config,
    validate_train_config,
)
from koopman_graph.cli.main import main

__all__ = [
    "ConfigError",
    "load_config",
    "load_train_config",
    "main",
    "validate_train_config",
]
