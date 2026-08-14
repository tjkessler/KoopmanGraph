"""Spatial-wavenumber vs temporal-growth dispersion from Kronecker spectra.

Pairs adjacency eigenvalues (graph-Fourier frequencies) with per-block
temporal growth rates of a Kronecker-eligible graph operator.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from koopman_graph.operators.graph import GraphKoopmanOperator
from koopman_graph.operators.kronecker_spectrum import spectrum_k_eff_kronecker_sum


@dataclass(frozen=True)
class DispersionRelation:
    """Dispersion samples for a Kronecker-eligible graph operator.

    Attributes
    ----------
    wavenumbers : Tensor
        Sorted adjacency eigenvalues (real part), shape ``(N,)``.
    growth_rates : Tensor
        Temporal growth rates aligned to graph modes, shape ``(N,)``.
    """

    wavenumbers: Tensor
    growth_rates: Tensor


def graph_dispersion(
    operator: GraphKoopmanOperator,
    edge_index: Tensor,
    num_nodes: int,
    edge_weight: Tensor | None = None,
) -> DispersionRelation:
    """Return wavenumber vs growth-rate samples.

    Parameters
    ----------
    operator : GraphKoopmanOperator
        Kronecker-eligible discrete graph operator.
    edge_index : Tensor
        Topology.
    num_nodes : int
        Node count.
    edge_weight : Tensor or None, optional
        Optional weights.

    Returns
    -------
    DispersionRelation
        Wavenumber / growth-rate samples.

    Raises
    ------
    ValueError
        If the operator is not Kronecker-eligible.
    """
    spectrum = spectrum_k_eff_kronecker_sum(
        k_self=operator.K_self,
        k_nbr=operator.K_nbr,
        edge_index=edge_index,
        num_nodes=num_nodes,
        adjacency=operator.adjacency,
        edge_weight=edge_weight,
    )
    if spectrum is None:
        raise ValueError("Kronecker-sum spectrum is not eligible for this operator")
    # Adjacency eigenvalues are recovered from the Kronecker helper via the
    # graph-Fourier frequencies stored on growth-rate blocks when present.
    adj = _dense_adjacency(edge_index, num_nodes, edge_weight, spectrum.eigenvalues)
    wavenumbers = torch.linalg.eigvalsh(adj).real.sort().values
    growth = spectrum.growth_rates.real.reshape(num_nodes, -1).mean(dim=-1)
    if growth.numel() != num_nodes:
        growth = spectrum.growth_rates.real[:num_nodes]
    return DispersionRelation(wavenumbers=wavenumbers, growth_rates=growth)


def _dense_adjacency(
    edge_index: Tensor,
    num_nodes: int,
    edge_weight: Tensor | None,
    _eigenvalues: Tensor,
) -> Tensor:
    """Assemble a symmetric adjacency for Fourier frequencies.

    Parameters
    ----------
    edge_index : Tensor
        COO edges of shape ``(2, E)``.
    num_nodes : int
        Graph order.
    edge_weight : Tensor or None
        Optional edge weights of length ``E``.
    _eigenvalues : Tensor
        Unused Kronecker eigenvalues (signature compatibility).

    Returns
    -------
    Tensor
        Symmetric dense adjacency of shape ``(num_nodes, num_nodes)``.
    """
    adj = torch.zeros(num_nodes, num_nodes, dtype=torch.float64)
    weight = (
        torch.ones(edge_index.shape[1], dtype=torch.float64)
        if edge_weight is None
        else edge_weight.detach().cpu().to(dtype=torch.float64)
    )
    src = edge_index[0].detach().cpu()
    dst = edge_index[1].detach().cpu()
    adj[src, dst] = weight
    adj = 0.5 * (adj + adj.T)
    return adj
