#!/usr/bin/env python3
"""Tiny native-DDP fit demo for KoopmanGraph.

Launch with two CPU processes (requires PyTorch ``torchrun``)::

    torchrun --standalone --nproc_per_node=2 examples/scripts/ddp_fit_torchrun.py

If process-group init stalls on a local machine, pin the rendezvous host::

    MASTER_ADDR=127.0.0.1 torchrun --standalone --nproc_per_node=2 \\
        examples/scripts/ddp_fit_torchrun.py

Single-process smoke (no process group; DDP wrapping is skipped)::

    python examples/scripts/ddp_fit_torchrun.py --epochs 2

Uses public APIs only: ``seed_everything``, ``is_main_process``, and
``GraphKoopmanModel.fit(..., strategy="ddp")`` (which delegates to
``koopman_graph.distributed.run_ddp_fit_loop``). Windowed training
(``window_length=3``) shards windows across ranks so one trajectory works
when ``world_size > 1``. Does not require Lightning or Ray. This script is
intentionally outside ``nbmake`` CI.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch_geometric.data import Data

from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
from koopman_graph.data import GraphSnapshotSequence
from koopman_graph.distributed import is_main_process, seed_everything


def _path_edge_index() -> torch.Tensor:
    """Return a bidirectional path graph on five nodes."""
    return torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 4],
            [1, 0, 2, 1, 3, 2, 4, 3],
        ],
        dtype=torch.long,
    )


def _decay_sequence(*, num_timesteps: int = 5) -> GraphSnapshotSequence:
    """Build a deterministic decay trajectory ``x_{t+1} = 0.9 * x_t``."""
    edge_index = _path_edge_index()
    x0 = torch.ones(5, 3)
    snapshots = [
        Data(x=x0 * (0.9**t), edge_index=edge_index) for t in range(num_timesteps)
    ]
    return GraphSnapshotSequence(snapshots)


def _build_model() -> GraphKoopmanModel:
    """Construct a small identically structured trainable model."""
    encoder = GNNEncoder(in_channels=3, hidden_channels=16, latent_dim=8)
    decoder = GNNDecoder(latent_dim=8, hidden_channels=16, out_channels=3)
    return GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=8,
        time_step=0.1,
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI flags for the DDP demo script."""
    parser = argparse.ArgumentParser(
        description=(
            "Seeded KoopmanGraph DDP fit demo. Prefer launching with "
            "`torchrun --standalone --nproc_per_node=2 "
            "examples/scripts/ddp_fit_torchrun.py`."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base RNG seed passed to seed_everything (default: 0).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        help="Adam learning rate (default: 1e-2).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=("Optional rank-0 checkpoint path. When omitted, no file is written."),
    )
    return parser.parse_args()


def main() -> None:
    """Run a short ``strategy='ddp'`` fit and print the final loss on rank 0."""
    args = parse_args()
    if args.epochs < 1:
        msg = f"--epochs must be >= 1; got {args.epochs}"
        raise SystemExit(msg)

    # Identical effective seed on every rank so DDP starts from matching weights.
    seed_everything(args.seed, rank=0)
    sequence = _decay_sequence()
    model = _build_model()
    # Windowed mode shards temporal windows across ranks so a single trajectory
    # works under torchrun (full-sequence mode requires len(trajectories) >=
    # world_size).
    history = model.fit(
        sequence,
        epochs=args.epochs,
        lr=args.lr,
        device="cpu",
        strategy="ddp",
        window_length=3,
        batch_size=2,
        window_seed=args.seed,
        checkpoint_path=args.checkpoint,
    )
    if is_main_process():
        final_loss = history.loss[-1]
        finite = bool(torch.isfinite(torch.tensor(final_loss)).item())
        print(
            f"ddp_fit_torchrun: epochs={history.epochs} "
            f"final_loss={final_loss:.6g} finite={finite}"
        )
        if args.checkpoint is not None:
            print(f"checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()
