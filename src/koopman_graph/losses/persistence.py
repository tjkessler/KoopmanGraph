"""Optional persistence regularizer on 0-dimensional diagrams.

Uses the core union-find diagram in :mod:`koopman_graph.analysis.tda`.
This is not a persistent-homology library replacement.
"""

from __future__ import annotations

from torch import Tensor, nn

from koopman_graph.analysis.tda import persistence_diagram_0d


class PersistenceRegularizer(nn.Module):
    """Penalize total 0-dimensional persistence of decoded coordinates.

    Parameters
    ----------
    weight : float, optional
        Scale of the mean death time. Default is ``1.0``.
    """

    def __init__(self, weight: float = 1.0) -> None:
        """Initialize the regularizer weight.

        Parameters
        ----------
        weight : float, optional
            Scale of the mean death time.
        """
        super().__init__()
        if weight < 0:
            raise ValueError(f"weight must be non-negative, got {weight}")
        self.weight = float(weight)

    def forward(self, points: Tensor) -> Tensor:
        """Return ``weight * mean(death)`` for a point cloud.

        Parameters
        ----------
        points : Tensor
            Coordinates ``(n, dim)``.

        Returns
        -------
        Tensor
            Scalar regularizer value.
        """
        diagram = persistence_diagram_0d(points)
        if diagram.pairs.numel() == 0:
            return points.new_zeros(())
        return self.weight * diagram.pairs[:, 1].mean()
