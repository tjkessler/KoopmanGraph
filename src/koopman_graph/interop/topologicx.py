"""Optional TopologicX incidence-tensor bridge (``[tdl]`` extra).

Converts caller-supplied incidence arrays into in-repo tensors. This is not
feature parity with TopologicX.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

_TDL_INSTALL_HINT = (
    "TopologicX is optional for this bridge. "
    "Install with: pip install 'koopman-graph[tdl]'"
)


def require_topologicx() -> Any:
    """Import TopologicX or raise a guided ``ImportError``.

    Returns
    -------
    module
        The ``topologicx`` package when installed.
    """
    try:
        import topologicx
    except ImportError as exc:
        raise ImportError(_TDL_INSTALL_HINT) from exc
    return topologicx


def incidence_tensors_from_arrays(
    boundary_0_1: Tensor,
    *,
    boundary_1_2: Tensor | None = None,
) -> dict[str, Tensor]:
    """Copy incidence matrices into a named tensor dict.

    Parameters
    ----------
    boundary_0_1 : Tensor
        Node–edge incidence ``B_1``.
    boundary_1_2 : Tensor or None, optional
        Edge–face incidence ``B_2``.

    Returns
    -------
    dict of str to Tensor
        Named incidence tensors.
    """
    payload = {"B1": boundary_0_1.to(dtype=torch.float32)}
    if boundary_1_2 is not None:
        payload["B2"] = boundary_1_2.to(dtype=torch.float32)
    return payload
