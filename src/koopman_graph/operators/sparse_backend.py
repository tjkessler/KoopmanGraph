"""Optional scipy.sparse eigensolve helpers for assembled operators.

Core training stays dense PyTorch. This module is an opt-in analysis path
when SciPy is installed (``[dev]``). It is not a PETSc requirement.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def sparse_leading_eigenvalues(
    matrix: Tensor,
    num_modes: int = 6,
) -> Tensor:
    """Leading-modulus eigenvalues via SciPy ``eigs`` when available.

    Falls back to dense ``torch.linalg.eigvals`` when SciPy is missing or
    the matrix is small.

    Parameters
    ----------
    matrix : Tensor
        Square assembled operator.
    num_modes : int, optional
        Number of eigenvalues to return.

    Returns
    -------
    Tensor
        Complex eigenvalues sorted by descending magnitude.
    """
    dense = matrix.detach()
    dim = int(dense.shape[0])
    k = max(1, min(int(num_modes), dim))
    if dim <= 32:
        values = torch.linalg.eigvals(dense)
        order = values.abs().argsort(descending=True)
        return values[order][:k]
    try:
        import numpy as np
        from scipy.sparse.linalg import eigs
    except ImportError:
        values = torch.linalg.eigvals(dense)
        order = values.abs().argsort(descending=True)
        return values[order][:k]
    array = dense.cpu().numpy()
    vals: Any
    vals, _ = eigs(array, k=k, which="LM")
    tensor = torch.from_numpy(np.asarray(vals))
    order = tensor.abs().argsort(descending=True)
    return tensor[order]
