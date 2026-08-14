"""In-tree federated averaging (data stays on the client).

Not differential privacy. Optional ``[federated]`` extra is reserved for a
Flower adapter; this module needs only PyTorch.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor


def federated_average(state_dicts: Sequence[Mapping[str, Tensor]]) -> dict[str, Tensor]:
    """Mean-average aligned ``state_dict`` tensors.

    Parameters
    ----------
    state_dicts : sequence of mappings
        Client parameter dictionaries with identical keys.

    Returns
    -------
    dict of str to Tensor
        Averaged parameters.

    Raises
    ------
    ValueError
        If ``state_dicts`` is empty or keys disagree.
    """
    if not state_dicts:
        raise ValueError("state_dicts must be non-empty")
    keys = list(state_dicts[0].keys())
    for client in state_dicts[1:]:
        if list(client.keys()) != keys:
            raise ValueError("client state_dicts must share identical keys")
    averaged: dict[str, Tensor] = {}
    for key in keys:
        stacked = torch.stack([client[key].detach().float() for client in state_dicts])
        averaged[key] = stacked.mean(dim=0).to(dtype=state_dicts[0][key].dtype)
    return averaged
