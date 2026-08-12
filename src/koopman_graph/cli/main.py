"""Argparse root for the ``koopman-graph`` console script."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from koopman_graph import __version__

_PROG = "koopman-graph"


def _build_parser() -> argparse.ArgumentParser:
    """Construct the root parser (subcommands added in later tasks).

    Returns
    -------
    argparse.ArgumentParser
        Root parser with ``version``, ``train``, and ``predict`` subcommands.
    """
    parser = argparse.ArgumentParser(
        prog=_PROG,
        description=("Config-driven train / predict entry points for KoopmanGraph."),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"{_PROG} {__version__}",
        help="Print package version and exit.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    version_parser = subparsers.add_parser(
        "version",
        help="Print package version and exit.",
    )
    version_parser.set_defaults(_handler=_handle_version)

    train_parser = subparsers.add_parser(
        "train",
        help="Train from a JSON/YAML config and save a checkpoint.",
    )
    train_parser.add_argument(
        "--config",
        required=True,
        help="Path to a train config (.json / .yaml / .yml).",
    )
    train_parser.add_argument(
        "--out",
        default=None,
        help=(
            "Optional output directory; relative checkpoint.path values "
            "are resolved under this directory."
        ),
    )
    train_parser.set_defaults(_handler=_handle_train)

    predict_parser = subparsers.add_parser(
        "predict",
        help=(
            "Load a checkpoint, forecast from a data source, and write a "
            ".pt forecast payload."
        ),
        description=(
            "Load a safetensors_v1 / legacy checkpoint, take the first "
            "snapshot from --data as the initial graph, run "
            "GraphKoopmanModel.predict for --steps, and write a torch "
            ".pt dict with keys 'steps', 'forecasts' (list of Data), and "
            "'summary' to --out."
        ),
    )
    predict_parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint path (safetensors_v1 directory/.kgckpt or legacy .pt).",
    )
    predict_parser.add_argument(
        "--data",
        required=True,
        help=(
            "Initial-sequence source: .pt GraphSnapshotSequence, or "
            "JSON/YAML with a data section (e.g. synthetic_path config)."
        ),
    )
    predict_parser.add_argument(
        "--steps",
        type=int,
        default=5,
        help="Autoregressive forecast horizon (default: 5).",
    )
    predict_parser.add_argument(
        "--out",
        required=True,
        help="Output .pt path for the forecast payload.",
    )
    predict_parser.set_defaults(_handler=_handle_predict)
    return parser


def _handle_version(_args: argparse.Namespace) -> int:
    """Print ``koopman-graph {version}`` to stdout.

    Parameters
    ----------
    _args : argparse.Namespace
        Unused parsed arguments (required by the handler protocol).

    Returns
    -------
    int
        Always ``0``.
    """
    print(f"{_PROG} {__version__}")
    return 0


def _handle_train(args: argparse.Namespace) -> int:
    """Delegate to :func:`koopman_graph.cli.train.handle_train`.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed ``train`` arguments (``config``, optional ``out``).

    Returns
    -------
    int
        Process exit code from the train handler.
    """
    from koopman_graph.cli.train import handle_train

    return handle_train(args)


def _handle_predict(args: argparse.Namespace) -> int:
    """Delegate to :func:`koopman_graph.cli.predict.handle_predict`.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed ``predict`` arguments (``checkpoint``, ``data``, ``steps``,
        ``out``).

    Returns
    -------
    int
        Process exit code from the predict handler.
    """
    from koopman_graph.cli.predict import handle_predict

    return handle_predict(args)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point used by ``[project.scripts]`` and ``python -m``.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Argument vector without the program name. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit code (``0`` success; argparse errors typically ``2``).
    """
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    handler = getattr(args, "_handler", None)
    if handler is None:
        parser.print_help()
        return 0
    return int(handler(args))


if __name__ == "__main__":
    sys.exit(main())
