"""Kronecker-sum exact spectrum helpers for networked graph operators.

For the one-tap form (discrete ``K`` or continuous generator ``L``)

.. math::

    M_{\\mathrm{eff}}
      = I_N \\otimes M_{\\mathrm{self}}
      + \\widehat{A} \\otimes M_{\\mathrm{nbr}},

if :math:`\\widehat{A} v_i = \\lambda_i v_i` and
:math:`(M_{\\mathrm{self}} + \\lambda_i M_{\\mathrm{nbr}}) w_{ij}
= \\mu_{ij} w_{ij}`, then

.. math::

    M_{\\mathrm{eff}} (v_i \\otimes w_{ij}) = \\mu_{ij}\\, (v_i \\otimes w_{ij})

The discrete monomial extension
:math:`K_{\\mathrm{eff}} = \\sum_{k=0}^{P} \\widehat{A}^{k} \\otimes K_{k}`
uses the same adjacency eigenpairs with the pencil
:math:`B(\\lambda_i) = \\sum_{k=0}^{P} \\lambda_i^{k} K_{k}`. That is
**not** a sum of independent factor eigenvalues
:math:`\\sum_k \\lambda_i^{k} \\nu_j^{(k)}`.

Assembly uses the package's node-block ``vec`` stacking (C-order
``Z.reshape(-1)`` with shape ``(N·d,)``, matching
:meth:`~koopman_graph.operators.graph.GraphKoopmanOperator.effective_matrix`
/
:meth:`~koopman_graph.operators.continuous_graph.ContinuousGraphKoopmanOperator.effective_generator`
and ``torch.kron``). Ambient eigenvector columns are unit-normalized
``torch.kron(v_i, w_{ij})``. Discrete spectra fill growth rates via
:func:`~koopman_graph.spectrum_types.compute_spectrum`; continuous generator
spectra use :func:`~koopman_graph.spectrum_types.compute_generator_spectrum`
conventions (real-part growth rates). Continuous graph remains the one-tap
generator form (no polynomial ``L`` reduction here).

Eligibility (callers must gate before use)
-----------------------------------------
* ``sparsity in {"dense", "block_diagonal"}`` (not ``"distributed"``)
* ``adjacency in {"symmetric", "random_walk"}`` (not dual)
* Shared self factor only (no orbit / isotypic per-orbit self banks)
* Discrete polynomial degree :math:`P \\ge 0` is eligible when the
  gates above hold (single shift)

Honesty
-------
* Exact when eligible; not a ``sparsity=`` mode and not Arnoldi
  (``matrix_free`` / ``sparsity="distributed"``). Continuous graph has no
  Arnoldi spectrum path — ineligible / failed reduction falls back to dense
  ``effective_generator`` eigendecomposition.
* Returns ``None`` when the reduction is numerically unreliable so callers
  can fall back to dense assembly — never a silent wrong spectrum.
  Structurally ineligible adjacency (dual / unknown) raises ``ValueError``.
* Random-walk :math:`\\widehat{A}` is checked with a residual on
  ``eig``; symmetric :math:`\\widehat{A}` uses ``eigh`` (Hermitian) with
  non-finite checks only.
* Eigenvector columns are defined up to complex phase; consumers should not
  assume alignment with dense ``torch.linalg.eig`` phases.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from koopman_graph.graph_utils.topology import (
    dense_random_walk_normalized_adjacency,
    dense_symmetric_normalized_adjacency,
)
from koopman_graph.operators.graph_types import (
    GRAPH_ADJACENCY_MODES,
    GraphAdjacency,
    GraphSparsity,
)
from koopman_graph.spectrum_types import KoopmanSpectrum

__all__ = [
    "DEFAULT_KRONECKER_ADJ_RESIDUAL_TOL",
    "eigenvalues_k_eff_kronecker_polynomial",
    "eigenvalues_k_eff_kronecker_sum",
    "kronecker_sum_spectrum_eligible",
    "spectrum_k_eff_kronecker_polynomial",
    "spectrum_k_eff_kronecker_sum",
    "spectrum_l_eff_kronecker_sum",
]

# Frobenius residual ||Â V - V Λ||_F / ||Â V||_F for random-walk eig.
# Justified as a loose float32 diagonalization sanity check (not a certificate).
DEFAULT_KRONECKER_ADJ_RESIDUAL_TOL = 1e-4

_ELIGIBLE_SPARSITY: frozenset[str] = frozenset({"dense", "block_diagonal"})
_ELIGIBLE_ADJACENCY: frozenset[str] = frozenset({"symmetric", "random_walk"})


def kronecker_sum_spectrum_eligible(
    *,
    adjacency: GraphAdjacency | str,
    sparsity: GraphSparsity | str,
    shared_self: bool,
) -> bool:
    """Return whether the Kronecker-sum exact spectrum path may be used.

    Parameters
    ----------
    adjacency : {"symmetric", "random_walk", "dual_random_walk"}
        Neighbor-coupling normalization. Dual mode has two non-commuting
        adjacency factors and is never eligible.
    sparsity : {"dense", "block_diagonal", "distributed"}
        Operator realization. ``"distributed"`` keeps the Arnoldi surrogate.
    shared_self : bool
        ``True`` when the self term is a single shared ``K_self``
        (``I ⊗ K_self``). ``False`` when orbit / isotypic per-orbit self
        banks replace that structure.

    Returns
    -------
    bool
        ``True`` only when shared-self single-adjacency eligibility holds.
    """
    if adjacency not in GRAPH_ADJACENCY_MODES:
        return False
    if sparsity not in {"dense", "block_diagonal", "distributed"}:
        return False
    return (
        shared_self
        and sparsity in _ELIGIBLE_SPARSITY
        and adjacency in _ELIGIBLE_ADJACENCY
    )


def _factors_finite(*tensors: Tensor) -> bool:
    """Return whether every tensor is finite.

    Parameters
    ----------
    *tensors : Tensor
        Tensors to check elementwise for finiteness.

    Returns
    -------
    bool
        ``True`` when every element of every tensor is finite.
    """
    return all(bool(torch.isfinite(tensor).all().item()) for tensor in tensors)


def _complex_dtype(dtype: torch.dtype) -> torch.dtype:
    """Map a real floating dtype to its complex counterpart.

    Parameters
    ----------
    dtype : torch.dtype
        Real floating dtype (``float32`` or ``float64``).

    Returns
    -------
    torch.dtype
        Matching complex dtype (``complex64`` or ``complex128``).
    """
    return torch.complex64 if dtype == torch.float32 else torch.complex128


def _random_walk_eigendecomposition(
    adj: Tensor,
    *,
    residual_tol: float,
) -> tuple[Tensor, Tensor] | None:
    """Eigendecompose a (possibly non-symmetric) adjacency with residual check.

    Parameters
    ----------
    adj : Tensor
        Dense :math:`N \\times N` random-walk normalized adjacency.
    residual_tol : float
        Maximum allowed relative Frobenius residual.

    Returns
    -------
    tuple[Tensor, Tensor] or None
        ``(eigenvalues, eigenvectors)`` with columns of ``eigenvectors`` the
        right eigenvectors, or ``None`` if the factorization is unusable.
    """
    eigenvalues, eigenvectors = torch.linalg.eig(adj)
    if not _factors_finite(eigenvalues, eigenvectors):
        return None
    reconstructed = eigenvectors @ torch.diag(eigenvalues)
    target = adj.to(dtype=eigenvectors.dtype) @ eigenvectors
    denom = torch.linalg.vector_norm(target).clamp_min(1e-30)
    residual = torch.linalg.vector_norm(target - reconstructed) / denom
    if not bool(torch.isfinite(residual).item()):
        return None
    if float(residual) > residual_tol:
        return None
    return eigenvalues, eigenvectors


def _adjacency_eigenpairs(
    edge_index: Tensor,
    num_nodes: int,
    *,
    adjacency: GraphAdjacency | str,
    edge_weight: Tensor | None,
    dtype: torch.dtype,
    residual_tol: float,
) -> tuple[Tensor, Tensor] | None:
    """Return eigenpairs of :math:`\\widehat{A}`, or ``None`` on failure.

    Parameters
    ----------
    edge_index : Tensor
        COO edge index of shape ``(2, E)``.
    num_nodes : int
        Number of graph nodes ``N``.
    adjacency : {"symmetric", "random_walk"} or str
        Normalized adjacency mode (eligible Kronecker modes only).
    edge_weight : Tensor or None
        Optional edge weights of length ``E``.
    dtype : torch.dtype
        Real floating dtype for dense adjacency assembly.
    residual_tol : float
        Relative Frobenius residual threshold for non-symmetric
        eigendecomposition.

    Returns
    -------
    tuple[Tensor, Tensor] or None
        Complex ``(eigenvalues, eigenvectors)`` with eigenvector columns, or
        ``None`` when the factorization is unusable.
    """
    complex_dtype = _complex_dtype(dtype)
    if adjacency == "symmetric":
        adj = dense_symmetric_normalized_adjacency(
            edge_index,
            num_nodes,
            edge_weight=edge_weight,
            dtype=dtype,
        )
        if not _factors_finite(adj):
            return None
        eigenvalues, eigenvectors = torch.linalg.eigh(adj)
        if not _factors_finite(eigenvalues, eigenvectors):
            return None
        return (
            eigenvalues.to(dtype=complex_dtype),
            eigenvectors.to(dtype=complex_dtype),
        )

    if adjacency == "random_walk":
        adj = dense_random_walk_normalized_adjacency(
            edge_index,
            num_nodes,
            edge_weight=edge_weight,
            dtype=dtype,
            direction="forward",
        )
        if not _factors_finite(adj):
            return None
        return _random_walk_eigendecomposition(adj, residual_tol=residual_tol)

    return None


def _require_eligible_adjacency(adjacency: GraphAdjacency | str) -> None:
    """Raise when adjacency is not a single commuting shift.

    Parameters
    ----------
    adjacency : {"symmetric", "random_walk", "dual_random_walk"} or str
        Requested neighbor-coupling mode.

    Raises
    ------
    ValueError
        If ``adjacency`` is not ``"symmetric"`` or ``"random_walk"``. Dual
        random-walk has two non-commuting shifts.
    """
    if adjacency in _ELIGIBLE_ADJACENCY:
        return
    msg = (
        "Kronecker-sum spectrum requires adjacency in "
        "{'symmetric', 'random_walk'}; received "
        f"{adjacency!r}. Dual random-walk has two non-commuting "
        "shifts; use dense effective_matrix eigendecomposition."
    )
    raise ValueError(msg)


def _factor_pencil(lam: Tensor, hop_matrices: Sequence[Tensor]) -> Tensor:
    """Return the latent pencil :math:`B(\\lambda) = \\sum_k \\lambda^k K_k`.

    Length-1 and length-2 cases use the closed forms of the historical
    one-tap helper so :math:`P=1` floats stay bit-identical.

    Parameters
    ----------
    lam : Tensor
        Scalar adjacency eigenvalue (0-d).
    hop_matrices : sequence of Tensor
        Complex hop factors ``(K_0, ..., K_P)`` of common shape ``(d, d)``.

    Returns
    -------
    Tensor
        Pencil of shape ``(d, d)``.
    """
    n_hops = len(hop_matrices)
    if n_hops == 1:
        return hop_matrices[0]
    block = hop_matrices[0] + lam * hop_matrices[1]
    if n_hops == 2:
        return block
    power = lam * lam
    for index in range(2, n_hops):
        block = block + power * hop_matrices[index]
        if index + 1 < n_hops:
            power = power * lam
    return block


def _k_eff_eigenpairs_kronecker_polynomial(
    *,
    hop_matrices: Sequence[Tensor],
    edge_index: Tensor,
    num_nodes: int,
    adjacency: GraphAdjacency | str,
    edge_weight: Tensor | None,
    residual_tol: float,
) -> tuple[Tensor, Tensor] | None:
    """Assemble unsorted ambient eigenpairs via the polynomial pencil.

    Parameters
    ----------
    hop_matrices : sequence of Tensor
        Square hop factors ``(K_0, ..., K_P)`` of common shape ``(d, d)``.
    edge_index : Tensor
        COO edge index of shape ``(2, E)``.
    num_nodes : int
        Number of graph nodes ``N``.
    adjacency : {"symmetric", "random_walk"} or str
        Normalized adjacency mode. Dual / unknown modes raise.
    edge_weight : Tensor or None
        Optional edge weights of length ``E``.
    residual_tol : float
        Relative residual threshold for adjacency eigendecomposition.

    Returns
    -------
    tuple[Tensor, Tensor] or None
        ``(eigenvalues, eigenvectors)`` with shapes ``(N·d,)`` and
        ``(N·d, N·d)`` (unit columns), or ``None`` on numerical failure.

    Raises
    ------
    ValueError
        If ``adjacency`` is structurally ineligible (dual / unknown).
    """
    _require_eligible_adjacency(adjacency)
    hops = tuple(hop_matrices)
    if not hops:
        return None
    first = hops[0]
    if first.ndim != 2 or first.shape[0] != first.shape[1]:
        return None
    for matrix in hops[1:]:
        if matrix.ndim != 2 or matrix.shape != first.shape:
            return None
    if num_nodes < 1:
        return None
    if not _factors_finite(*hops):
        return None

    adj_pairs = _adjacency_eigenpairs(
        edge_index,
        num_nodes,
        adjacency=adjacency,
        edge_weight=edge_weight,
        dtype=first.dtype,
        residual_tol=residual_tol,
    )
    if adj_pairs is None:
        return None
    adj_eigs, adj_vecs = adj_pairs

    latent_dim = first.shape[0]
    factor_dtype = _complex_dtype(first.dtype)
    hops_c = tuple(matrix.to(dtype=factor_dtype) for matrix in hops)
    ambient_dim = num_nodes * latent_dim

    eigenvalues: list[Tensor] = []
    eigenvectors: list[Tensor] = []
    for index, lam in enumerate(adj_eigs):
        block = _factor_pencil(lam, hops_c)
        block_eigs, block_vecs = torch.linalg.eig(block)
        if not _factors_finite(block_eigs, block_vecs):
            return None
        node_vec = adj_vecs[:, index]
        for mode in range(latent_dim):
            ambient = torch.kron(node_vec, block_vecs[:, mode])
            norm = torch.linalg.vector_norm(ambient).clamp_min(1e-30)
            ambient = ambient / norm
            if not _factors_finite(ambient):
                return None
            eigenvalues.append(block_eigs[mode].reshape(()))
            eigenvectors.append(ambient)

    evals = torch.stack(eigenvalues)
    evecs = torch.stack(eigenvectors, dim=1)
    if evals.numel() != ambient_dim or evecs.shape != (ambient_dim, ambient_dim):
        return None
    return evals, evecs


def _k_eff_eigenpairs_kronecker_sum(
    *,
    k_self: Tensor,
    k_nbr: Tensor,
    edge_index: Tensor,
    num_nodes: int,
    adjacency: GraphAdjacency | str,
    edge_weight: Tensor | None,
    residual_tol: float,
) -> tuple[Tensor, Tensor] | None:
    """Assemble unsorted ambient eigenpairs via the one-tap Kronecker sum.

    Thin wrapper over :func:`_k_eff_eigenpairs_kronecker_polynomial` with
    hop factors ``(k_self, k_nbr)``.

    Parameters
    ----------
    k_self : Tensor
        Shared self factor of shape ``(d, d)``.
    k_nbr : Tensor
        Neighbor factor of shape ``(d, d)``.
    edge_index : Tensor
        COO edge index of shape ``(2, E)``.
    num_nodes : int
        Number of graph nodes ``N``.
    adjacency : {"symmetric", "random_walk"} or str
        Normalized adjacency mode.
    edge_weight : Tensor or None
        Optional edge weights of length ``E``.
    residual_tol : float
        Relative residual threshold for adjacency eigendecomposition.

    Returns
    -------
    tuple[Tensor, Tensor] or None
        ``(eigenvalues, eigenvectors)`` with shapes ``(N·d,)`` and
        ``(N·d, N·d)`` (unit columns), or ``None`` on failure.

    Raises
    ------
    ValueError
        If ``adjacency`` is structurally ineligible (dual / unknown).
    """
    return _k_eff_eigenpairs_kronecker_polynomial(
        hop_matrices=(k_self, k_nbr),
        edge_index=edge_index,
        num_nodes=num_nodes,
        adjacency=adjacency,
        edge_weight=edge_weight,
        residual_tol=residual_tol,
    )


def _discrete_spectrum_from_eigenpairs(
    pairs: tuple[Tensor, Tensor],
    time_step: float,
) -> KoopmanSpectrum:
    """Sort eigenpairs by descending magnitude and fill discrete rates.

    Parameters
    ----------
    pairs : tuple of Tensor
        Unsorted ``(eigenvalues, eigenvectors)`` from the Kronecker core.
    time_step : float
        Positive discrete sampling interval.

    Returns
    -------
    KoopmanSpectrum
        Magnitude-sorted spectrum with growth rates and frequencies.
    """
    eigenvalues, eigenvectors = pairs
    magnitudes = eigenvalues.abs()
    order = torch.argsort(magnitudes, descending=True)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    magnitudes = magnitudes[order]
    growth_rates = torch.log(magnitudes.clamp_min(1e-30)) / time_step
    frequencies = torch.angle(eigenvalues) / (2 * torch.pi * time_step)
    return KoopmanSpectrum(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        magnitudes=magnitudes,
        growth_rates=growth_rates,
        frequencies=frequencies,
        time_step=float(time_step),
    )


def eigenvalues_k_eff_kronecker_sum(
    *,
    k_self: Tensor,
    k_nbr: Tensor,
    edge_index: Tensor,
    num_nodes: int,
    adjacency: GraphAdjacency | str,
    edge_weight: Tensor | None = None,
    residual_tol: float = DEFAULT_KRONECKER_ADJ_RESIDUAL_TOL,
) -> Tensor | None:
    """Compute :math:`K_{\\mathrm{eff}}` eigenvalues via Kronecker-sum reduction.

    Thin wrapper over the shared eigenpair core for the one-tap map
    :math:`I\\otimes K_{\\mathrm{self}} + \\widehat{A}\\otimes K_{\\mathrm{nbr}}`.
    Does **not** check ``sparsity`` / orbit eligibility — callers must gate
    with :func:`kronecker_sum_spectrum_eligible` (and ``shared_self``)
    first. Returns ``None`` when the reduction is numerically unreliable
    so the caller can fall back to dense assembly.

    Parameters
    ----------
    k_self : Tensor
        Shared self factor with shape ``(d, d)``.
    k_nbr : Tensor
        Neighbor factor with shape ``(d, d)``.
    edge_index : Tensor
        Edge index ``(2, E)``.
    num_nodes : int
        Node count ``N``.
    adjacency : {"symmetric", "random_walk"}
        Single-factor adjacency mode. Dual mode raises.
    edge_weight : Tensor or None, optional
        Optional edge weights ``(E,)``.
    residual_tol : float, optional
        Random-walk adjacency residual tolerance. Default
        :data:`DEFAULT_KRONECKER_ADJ_RESIDUAL_TOL`.

    Returns
    -------
    Tensor or None
        Complex eigenvalues with shape ``(N·d,)`` (unsorted), or ``None``
        when the reduction is unreliable.

    Raises
    ------
    ValueError
        If ``adjacency`` is dual or otherwise not a single commuting shift.
    """
    return eigenvalues_k_eff_kronecker_polynomial(
        hop_matrices=(k_self, k_nbr),
        edge_index=edge_index,
        num_nodes=num_nodes,
        adjacency=adjacency,
        edge_weight=edge_weight,
        residual_tol=residual_tol,
    )


def eigenvalues_k_eff_kronecker_polynomial(
    *,
    hop_matrices: Sequence[Tensor],
    edge_index: Tensor,
    num_nodes: int,
    adjacency: GraphAdjacency | str,
    edge_weight: Tensor | None = None,
    residual_tol: float = DEFAULT_KRONECKER_ADJ_RESIDUAL_TOL,
) -> Tensor | None:
    """Compute polynomial :math:`K_{\\mathrm{eff}}` eigenvalues via Kronecker reduction.

    Uses :math:`B(\\lambda_i)=\\sum_k \\lambda_i^k K_k` on each adjacency
    eigenpair. That is **not** a sum of independent factor eigenvalues.
    Does **not** check ``sparsity`` / orbit eligibility — callers must gate
    with :func:`kronecker_sum_spectrum_eligible`. Returns ``None`` when
    the reduction is numerically unreliable.

    Parameters
    ----------
    hop_matrices : sequence of Tensor
        Square hop factors ``(K_0, ..., K_P)`` of common shape ``(d, d)``.
        Length 1 is the self-only map; length 2 is the one-tap sum.
    edge_index : Tensor
        Edge index ``(2, E)``.
    num_nodes : int
        Node count ``N``.
    adjacency : {"symmetric", "random_walk"}
        Single-factor adjacency mode. Dual mode raises.
    edge_weight : Tensor or None, optional
        Optional edge weights ``(E,)``.
    residual_tol : float, optional
        Random-walk adjacency residual tolerance.

    Returns
    -------
    Tensor or None
        Complex eigenvalues with shape ``(N·d,)`` (unsorted), or ``None``
        when the reduction is unreliable.

    Raises
    ------
    ValueError
        If ``adjacency`` is dual or otherwise not a single commuting shift.
    """
    pairs = _k_eff_eigenpairs_kronecker_polynomial(
        hop_matrices=hop_matrices,
        edge_index=edge_index,
        num_nodes=num_nodes,
        adjacency=adjacency,
        edge_weight=edge_weight,
        residual_tol=residual_tol,
    )
    if pairs is None:
        return None
    return pairs[0]


def spectrum_k_eff_kronecker_polynomial(
    *,
    hop_matrices: Sequence[Tensor],
    edge_index: Tensor,
    num_nodes: int,
    adjacency: GraphAdjacency | str,
    edge_weight: Tensor | None = None,
    time_step: float = 1.0,
    residual_tol: float = DEFAULT_KRONECKER_ADJ_RESIDUAL_TOL,
) -> KoopmanSpectrum | None:
    """Exact polynomial :math:`K_{\\mathrm{eff}}` spectrum via Kronecker reduction.

    Builds ambient eigenvectors as unit-normalized ``kron(v_i, w_ij)``
    from :math:`B(\\lambda_i)=\\sum_k \\lambda_i^k K_k`, sorts by
    descending magnitude, and fills growth rates / frequencies with the
    same discrete conventions as
    :func:`~koopman_graph.spectrum_types.compute_spectrum`.

    Does **not** check ``sparsity`` / orbit eligibility — callers must gate
    with :func:`kronecker_sum_spectrum_eligible`. Returns ``None`` when the
    reduction is numerically unreliable.

    Parameters
    ----------
    hop_matrices : sequence of Tensor
        Square hop factors ``(K_0, ..., K_P)`` of common shape ``(d, d)``.
    edge_index : Tensor
        Edge index ``(2, E)``.
    num_nodes : int
        Node count ``N``.
    adjacency : {"symmetric", "random_walk"}
        Single-factor adjacency mode. Dual mode raises.
    edge_weight : Tensor or None, optional
        Optional edge weights ``(E,)``.
    time_step : float, optional
        Positive discrete sampling interval for growth rates / frequencies.
    residual_tol : float, optional
        Random-walk adjacency residual tolerance.

    Returns
    -------
    KoopmanSpectrum or None
        Magnitude-sorted spectrum with ambient eigenvectors, or ``None``.

    Raises
    ------
    ValueError
        If ``adjacency`` is dual or otherwise not a single commuting shift.
    """
    if time_step <= 0:
        return None
    pairs = _k_eff_eigenpairs_kronecker_polynomial(
        hop_matrices=hop_matrices,
        edge_index=edge_index,
        num_nodes=num_nodes,
        adjacency=adjacency,
        edge_weight=edge_weight,
        residual_tol=residual_tol,
    )
    if pairs is None:
        return None
    return _discrete_spectrum_from_eigenpairs(pairs, time_step)


def spectrum_k_eff_kronecker_sum(
    *,
    k_self: Tensor,
    k_nbr: Tensor,
    edge_index: Tensor,
    num_nodes: int,
    adjacency: GraphAdjacency | str,
    edge_weight: Tensor | None = None,
    time_step: float = 1.0,
    residual_tol: float = DEFAULT_KRONECKER_ADJ_RESIDUAL_TOL,
) -> KoopmanSpectrum | None:
    """Exact one-tap :math:`K_{\\mathrm{eff}}` spectrum via Kronecker-sum reduction.

    Thin wrapper over :func:`spectrum_k_eff_kronecker_polynomial` with hop
    factors ``(k_self, k_nbr)``. Builds ambient eigenvectors as
    unit-normalized ``kron(v_i, w_ij)``, sorts by descending magnitude,
    and fills growth rates / frequencies with the same discrete
    conventions as :func:`~koopman_graph.spectrum_types.compute_spectrum`.

    Does **not** check ``sparsity`` / orbit eligibility — callers must gate
    with :func:`kronecker_sum_spectrum_eligible`. Returns ``None`` when the
    reduction is numerically unreliable.

    Parameters
    ----------
    k_self : Tensor
        Shared self factor with shape ``(d, d)``.
    k_nbr : Tensor
        Neighbor factor with shape ``(d, d)``.
    edge_index : Tensor
        Edge index ``(2, E)``.
    num_nodes : int
        Node count ``N``.
    adjacency : {"symmetric", "random_walk"}
        Single-factor adjacency mode. Dual mode raises.
    edge_weight : Tensor or None, optional
        Optional edge weights ``(E,)``.
    time_step : float, optional
        Positive discrete sampling interval for growth rates / frequencies.
    residual_tol : float, optional
        Random-walk adjacency residual tolerance.

    Returns
    -------
    KoopmanSpectrum or None
        Magnitude-sorted spectrum with ambient eigenvectors, or ``None``.

    Raises
    ------
    ValueError
        If ``adjacency`` is dual or otherwise not a single commuting shift.
    """
    return spectrum_k_eff_kronecker_polynomial(
        hop_matrices=(k_self, k_nbr),
        edge_index=edge_index,
        num_nodes=num_nodes,
        adjacency=adjacency,
        edge_weight=edge_weight,
        time_step=time_step,
        residual_tol=residual_tol,
    )


def spectrum_l_eff_kronecker_sum(
    *,
    l_self: Tensor,
    l_nbr: Tensor,
    edge_index: Tensor,
    num_nodes: int,
    adjacency: GraphAdjacency | str,
    edge_weight: Tensor | None = None,
    residual_tol: float = DEFAULT_KRONECKER_ADJ_RESIDUAL_TOL,
) -> KoopmanSpectrum | None:
    """Exact :math:`L_{\\mathrm{eff}}` spectrum via one-tap Kronecker-sum reduction.

    Reuses the same eigenpair core as :func:`spectrum_k_eff_kronecker_sum`
    (identical Kronecker-sum algebra for generator factors). Continuous
    graph remains the one-tap generator form; there is no polynomial
    :math:`L` reduction here. Sorts by descending magnitude and fills
    growth rates / frequencies with the same continuous conventions as
    :func:`~koopman_graph.spectrum_types.compute_generator_spectrum`
    (``growth_rates = Re(λ)``, ``frequencies = Im(λ)/(2π)``,
    ``time_step=1.0``).

    Does **not** check ``sparsity`` eligibility — callers must gate with
    :func:`kronecker_sum_spectrum_eligible`. Returns ``None`` when the
    reduction is numerically unreliable.

    Parameters
    ----------
    l_self : Tensor
        Shared self generator with shape ``(d, d)``.
    l_nbr : Tensor
        Neighbor generator with shape ``(d, d)``.
    edge_index : Tensor
        Edge index ``(2, E)``.
    num_nodes : int
        Node count ``N``.
    adjacency : {"symmetric", "random_walk"}
        Single-factor adjacency mode. Dual mode raises.
    edge_weight : Tensor or None, optional
        Optional edge weights ``(E,)``.
    residual_tol : float, optional
        Random-walk adjacency residual tolerance.

    Returns
    -------
    KoopmanSpectrum or None
        Magnitude-sorted generator spectrum with ambient eigenvectors, or
        ``None``.

    Raises
    ------
    ValueError
        If ``adjacency`` is dual or otherwise not a single commuting shift.
    """
    pairs = _k_eff_eigenpairs_kronecker_polynomial(
        hop_matrices=(l_self, l_nbr),
        edge_index=edge_index,
        num_nodes=num_nodes,
        adjacency=adjacency,
        edge_weight=edge_weight,
        residual_tol=residual_tol,
    )
    if pairs is None:
        return None
    eigenvalues, eigenvectors = pairs
    magnitudes = eigenvalues.abs()
    order = torch.argsort(magnitudes, descending=True)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    magnitudes = magnitudes[order]
    growth_rates = eigenvalues.real
    frequencies = eigenvalues.imag / (2 * torch.pi)
    return KoopmanSpectrum(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        magnitudes=magnitudes,
        growth_rates=growth_rates,
        frequencies=frequencies,
        time_step=1.0,
    )
