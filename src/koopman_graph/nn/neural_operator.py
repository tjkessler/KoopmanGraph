"""Mesh/resolution-conditioned Fourier lifting (neural-operator MVP).

Applies a learnable Fourier mixing along a 1-D mesh index, then a shared
linear map. Not a resolution-invariant PDE solver.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.data import Data

from koopman_graph.nn.gnn import validate_positive_dims


class FourierNeuralOperatorEncoder(nn.Module):
    """FFT mixing along the node axis followed by a linear lift.

    Parameters
    ----------
    in_channels, hidden_channels, out_channels : int
        Feature widths (hidden unused except for API parity with GNN peers).
    n_modes : int, optional
        Number of Fourier modes retained. Default is 8.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        *,
        n_modes: int = 8,
    ) -> None:
        """Initialize Fourier mixing weights.

        Parameters
        ----------
        in_channels, hidden_channels, out_channels : int
            Feature widths.
        n_modes : int, optional
            Number of Fourier modes retained.
        """
        super().__init__()
        validate_positive_dims(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
        )
        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.out_channels = int(out_channels)
        self.n_modes = int(n_modes)
        self.lift = nn.Linear(in_channels, out_channels)
        self.spectral = nn.Parameter(
            torch.randn(max(n_modes, 1), out_channels, out_channels) * 0.02
        )

    def forward(self, data: Data) -> Tensor:
        """Lift ``data.x`` with shared Fourier weights.

        Parameters
        ----------
        data : Data
            Snapshot with ``x`` of shape ``(N, F)``.

        Returns
        -------
        Tensor
            Lifted features ``(N, out_channels)``.
        """
        if data.x is None:
            raise ValueError("FourierNeuralOperatorEncoder requires Data.x")
        lifted = self.lift(data.x)
        spectrum = torch.fft.rfft(lifted, dim=0)
        modes = min(self.n_modes, spectrum.shape[0])
        mixed = spectrum.clone()
        for mode in range(modes):
            mixed[mode] = spectrum[mode] @ self.spectral[mode].to(dtype=spectrum.dtype)
        return torch.fft.irfft(mixed, n=lifted.shape[0], dim=0)
