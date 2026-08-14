"""False-data-injection / sensor-corruption helpers.

Data-integrity threat model on node features. Distinct from checkpoint
adversarial weights documented in ``SECURITY.md``.
"""

from __future__ import annotations

import torch
from torch_geometric.data import Data


def corrupt_node_features(
    data: Data,
    *,
    magnitude: float,
    generator: torch.Generator | None = None,
) -> Data:
    """Add bounded uniform noise to ``data.x``.

    Parameters
    ----------
    data : Data
        Homogeneous snapshot.
    magnitude : float
        Half-width of the uniform perturbation.
    generator : torch.Generator or None, optional
        Optional RNG.

    Returns
    -------
    Data
        Cloned snapshot with perturbed ``x``.

    Raises
    ------
    ValueError
        If ``x`` is missing or ``magnitude`` is negative.
    """
    if data.x is None:
        raise ValueError("corrupt_node_features requires Data.x")
    if magnitude < 0:
        raise ValueError(f"magnitude must be non-negative, got {magnitude}")
    noise = (
        torch.rand(data.x.shape, generator=generator, device=data.x.device) * 2.0 - 1.0
    )
    corrupted = data.clone()
    corrupted.x = data.x + float(magnitude) * noise
    return corrupted
