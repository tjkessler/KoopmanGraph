"""``koopman-graph benchmark`` — identity-bound run / verify handlers.

Handlers lazy-import :mod:`koopman_graph.benchmark` so that
``import koopman_graph.cli`` does not load the benchmark package.
They do **not** fit :class:`~koopman_graph.model.GraphKoopmanModel`.
"""

from __future__ import annotations

import argparse
import sys


def handle_benchmark_run(args: argparse.Namespace) -> int:
    """Argparse handler for ``benchmark run``.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed namespace with ``manifest``, ``data``, and ``out``.

    Returns
    -------
    int
        ``0`` on success; ``1`` on schema / I/O / import errors.
    """
    try:
        from koopman_graph.benchmark.runner import run_manifest

        destination = run_manifest(args.manifest, args.data, args.out)
    except (ValueError, FileNotFoundError, OSError, ImportError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote summary: {destination}")
    return 0


def handle_benchmark_verify(args: argparse.Namespace) -> int:
    """Argparse handler for ``benchmark verify``.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed namespace with ``manifest`` and ``against``.

    Returns
    -------
    int
        ``0`` on success; ``1`` on tamper / schema / I/O errors.
    """
    try:
        from koopman_graph.benchmark.runner import resolve_summary_path, verify_summary

        verify_summary(args.manifest, args.against)
        destination = resolve_summary_path(args.against)
    except (ValueError, FileNotFoundError, OSError, ImportError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"verified summary: {destination}")
    return 0
