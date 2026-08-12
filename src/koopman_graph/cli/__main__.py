"""Module entry: ``python -m koopman_graph.cli``."""

from __future__ import annotations

import sys

from koopman_graph.cli.main import main

if __name__ == "__main__":
    sys.exit(main())
