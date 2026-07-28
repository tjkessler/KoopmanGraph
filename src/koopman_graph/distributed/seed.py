"""Seeding helpers for reproducible distributed training."""

from __future__ import annotations

import random

import torch

from koopman_graph.distributed.process import get_rank

__all__ = [
    "seed_everything",
]


def seed_everything(seed: int, *, rank: int | None = None) -> int:
    """Seed Python, NumPy (when importable), and PyTorch RNGs.

    Parameters
    ----------
    seed : int
        Base seed shared across ranks before the optional rank offset.
    rank : int or None, optional
        Rank used as ``seed + rank``. When ``None``, uses
        :func:`~koopman_graph.distributed.get_rank` (``0`` when distributed
        is inactive).

    Returns
    -------
    int
        Effective seed applied (``seed + rank``).

    Notes
    -----
    NumPy is seeded only when ``numpy`` is already importable so this helper
    does not add a hard NumPy dependency. CUDA RNG is seeded when CUDA is
    available.
    """
    resolved_rank = get_rank() if rank is None else int(rank)
    effective = int(seed) + int(resolved_rank)
    random.seed(effective)
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(effective)
    torch.manual_seed(effective)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective)
    return effective
