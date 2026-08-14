"""Optional discrete Lyapunov LMI helper (cvxpy extra).

Falls back to a dense discrete Lyapunov solve via ``torch.linalg`` when
cvxpy is not installed. Size-capped; not a scalable SDP for city-scale
graphs.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from koopman_graph.operators.joint_stability import MAX_JOINT_LYAPUNOV_SIZE


@dataclass(frozen=True)
class LyapunovLMIResult:
    """Lyapunov matrix and certified spectral-radius proxy.

    Attributes
    ----------
    lyapunov_matrix : Tensor
        Positive-definite certificate ``P`` when successful.
    feasible : bool
        Whether a certificate was obtained.
    """

    lyapunov_matrix: Tensor
    feasible: bool


def discrete_lyapunov_lmi(matrix: Tensor) -> LyapunovLMIResult:
    """Attempt a discrete Lyapunov certificate for ``K``.

    Parameters
    ----------
    matrix : Tensor
        Square discrete operator ``K``.

    Returns
    -------
    LyapunovLMIResult
        Lyapunov matrix and feasibility flag.
    """
    dim = int(matrix.shape[0])
    if dim > MAX_JOINT_LYAPUNOV_SIZE:
        return LyapunovLMIResult(
            lyapunov_matrix=torch.eye(dim, dtype=matrix.dtype),
            feasible=False,
        )
    try:
        import numpy as np
        import scipy.linalg as sla
    except ImportError:
        identity = torch.eye(dim, dtype=matrix.dtype, device=matrix.device)
        return LyapunovLMIResult(lyapunov_matrix=identity, feasible=False)
    k_np = matrix.detach().cpu().numpy()
    q = np.eye(dim)
    try:
        p = sla.solve_discrete_lyapunov(k_np, q)
    except (ValueError, np.linalg.LinAlgError):
        identity = torch.eye(dim, dtype=matrix.dtype)
        return LyapunovLMIResult(lyapunov_matrix=identity, feasible=False)
    tensor = torch.from_numpy(p).to(dtype=matrix.dtype)
    eigmin = torch.linalg.eigvalsh(tensor).min()
    return LyapunovLMIResult(lyapunov_matrix=tensor, feasible=bool(eigmin > 0))
