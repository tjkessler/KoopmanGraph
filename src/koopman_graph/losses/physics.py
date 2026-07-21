"""Physics residual (PDE) losses."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch_geometric.data import Data


class PDEResidualLoss(nn.Module):
    """Penalize a user-supplied PDE residual on decoded graph fields.

    ``pde_fn`` owns the equation-specific discretization and receives both the
    decoded field and its graph snapshot context. Graph-based implementations
    should use :mod:`koopman_graph.graph_utils` for the shared
    pseudoinverse-normalized ``L_sym = P - Â`` convention.

    Notes
    -----
    This generic residual interface does not prescribe a particular PDE
    discretization or claim a structure-preserving time integrator.
    """

    def forward(
        self,
        decoded: Tensor,
        snapshot: Data,
        *,
        pde_fn: Callable[[Tensor, Data], Tensor],
        mask: Tensor | None = None,
    ) -> Tensor:
        """Return mean squared PDE residual.

        Parameters
        ----------
        decoded : Tensor
            Decoded node field, normally shaped ``(num_nodes, features)``.
        snapshot : object
            Graph/time context passed unchanged to ``pde_fn``.
        pde_fn : callable
            Callable ``pde_fn(decoded, snapshot) -> residual``.
        mask : Tensor or None, optional
            Optional boolean node mask. The residual's first dimension must be
            the node dimension when a mask is supplied.

        Returns
        -------
        Tensor
            Scalar residual mean square.
        """
        residual = pde_fn(decoded, snapshot)
        if not isinstance(residual, Tensor):
            msg = "pde_fn must return a Tensor"
            raise TypeError(msg)
        if residual.numel() == 0:
            msg = "pde_fn must return a non-empty residual tensor"
            raise ValueError(msg)
        if mask is None:
            return residual.square().mean()
        if residual.ndim == 0 or residual.shape[0] != mask.numel():
            msg = (
                "masked PDE residual must have num_nodes as its first dimension, "
                f"got residual shape {tuple(residual.shape)} and "
                f"mask length {mask.numel()}"
            )
            raise ValueError(msg)
        selected = residual[mask.to(device=residual.device, dtype=torch.bool)]
        if selected.numel() == 0:
            return torch.zeros((), device=residual.device, dtype=residual.dtype)
        return selected.square().mean()
