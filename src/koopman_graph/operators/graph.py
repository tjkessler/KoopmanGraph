"""Networked (spatially-coupled) discrete Koopman operator."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

from koopman_graph.operators.contract import (
    InitMode,
    Parameterization,
    StabilityCertificate,
)
from koopman_graph.operators.control import (
    ControlMode,
    bilinear_state_control_term,
    effective_bilinear_matrix,
)
from koopman_graph.operators.discrete import KoopmanOperator
from koopman_graph.spectrum_types import KoopmanSpectrum

GraphSparsity = Literal["dense", "block_diagonal", "distributed"]


class GraphKoopmanOperator(nn.Module):
    """Discrete Koopman step with self and neighbor coupling on the graph.

    Advances stacked node latents ``Z ∈ R^{N×d}`` via the linear map::

        vec(Z_{t+1}) = (I_N ⊗ K_self + Â ⊗ K_nbr) vec(Z_t)

    implemented sparsely as::

        Z_next = Z @ K_self.T + (Â Z) @ K_nbr.T

    where ``Â`` is the symmetric normalized adjacency
    ``D^{-1/2} A D^{-1/2}``. Unlike :class:`KoopmanOperator`, topology enters
    the **linear** step, so mid-horizon rewiring changes latent advance (not
    only decode). Discrete-time only; continuous networked generators are out
    of scope for this module.

    When ``K_nbr = 0``, the step reduces exactly to the per-node map
    ``Z @ K_self.T``.

    Attributes
    ----------
    latent_dim : int
        Latent feature dimension ``d``.
    control_dim : int
        Exogenous control dimension (``0`` disables control).
    parameterization : Parameterization
        Shared soft/structural parameterization for ``K_self`` and ``K_nbr``.
    sparsity : {"dense", "block_diagonal", "distributed"}
        Realization mode. Only ``"dense"`` (sparse message-passing matvec) is
        implemented; other values are reserved and rejected.
    max_spectral_radius : float
        Stability bound forwarded to the factorized self/neighbor matrices.
    """

    def __init__(
        self,
        latent_dim: int,
        *,
        init_mode: InitMode = "identity_noise",
        init_scale: float = 1e-2,
        parameterization: Parameterization = "dense",
        max_spectral_radius: float = 1.0,
        control_dim: int = 0,
        control_mode: ControlMode = "additive",
        bilinear_rank: int | None = None,
        sparsity: GraphSparsity = "dense",
    ) -> None:
        """Initialize self and neighbor Koopman factors.

        Parameters
        ----------
        latent_dim : int
            Latent dimension ``d``.
        init_mode : {"identity", "identity_noise", "xavier"}, optional
            Initialization for ``K_self``. ``K_nbr`` starts at zero for
            ``identity`` / ``identity_noise`` (plus optional noise on the
            neighbor term for ``identity_noise`` / ``xavier``).
        init_scale : float, optional
            Noise scale for ``identity_noise`` / neighbor jitter.
        parameterization : Parameterization, optional
            Shared parameterization for both ``d×d`` factors.
        max_spectral_radius : float, optional
            Spectral bound for soft/structural modes.
        control_dim : int, optional
            Additive / bilinear control dimension. Default ``0``.
        control_mode : {"additive", "bilinear"}, optional
            Control coupling forwarded to the self-term operator.
        bilinear_rank : int or None, optional
            Low-rank bilinear size when ``control_mode="bilinear"``.
        sparsity : {"dense", "block_diagonal", "distributed"}, optional
            Only ``"dense"`` is supported in this release.

        Raises
        ------
        ValueError
            If ``sparsity`` is not ``"dense"`` or construction args are invalid.
        """
        super().__init__()
        if sparsity != "dense":
            msg = (
                "GraphKoopmanOperator sparsity modes "
                f"{sparsity!r} are not implemented yet; use sparsity='dense'"
            )
            raise ValueError(msg)

        self.latent_dim = latent_dim
        self.init_mode = init_mode
        self.init_scale = init_scale
        self.parameterization = parameterization
        self.max_spectral_radius = max_spectral_radius
        self.control_dim = control_dim
        self.control_mode = control_mode
        self.bilinear_rank = bilinear_rank
        self.sparsity = sparsity

        # Self-term owns the optional control matrix B (and bilinear factors).
        self._self = KoopmanOperator(
            latent_dim,
            init_mode=init_mode,
            init_scale=init_scale,
            parameterization=parameterization,
            max_spectral_radius=max_spectral_radius,
            control_dim=control_dim,
            control_mode=control_mode,
            bilinear_rank=bilinear_rank,
        )
        self._nbr = KoopmanOperator(
            latent_dim,
            init_mode="identity",
            init_scale=init_scale,
            parameterization=parameterization,
            max_spectral_radius=max_spectral_radius,
            control_dim=0,
        )
        self._reset_neighbor_parameters()

    def _reset_neighbor_parameters(self) -> None:
        """Initialize ``K_nbr`` near zero so the operator starts per-node-like.

        Notes
        -----
        Dense mode zeros the stored ``K`` factor; factorized modes zero raw
        parameters, optionally adding ``init_scale`` noise.
        """
        if self.parameterization == "dense":
            dense_k = self._nbr._parameters.get("K")
            if dense_k is None:
                raise AttributeError("K")
            with torch.no_grad():
                dense_k.zero_()
                if self.init_mode in {"identity_noise", "xavier"}:
                    dense_k.add_(torch.randn_like(dense_k) * self.init_scale)
            return

        # Factorized modes: drive assembled K_nbr toward zero via raw params.
        with torch.no_grad():
            for parameter in self._nbr.parameters():
                parameter.zero_()
            if self.init_mode in {"identity_noise", "xavier"}:
                for parameter in self._nbr.parameters():
                    parameter.add_(torch.randn_like(parameter) * self.init_scale)

    def reset_parameters(self) -> None:
        """Reinitialize ``K_self`` / ``K_nbr`` (and control ``B`` when present).

        Notes
        -----
        Delegates to the self/neighbor factor modules, then re-applies the
        near-zero neighbor initialization.
        """
        self._self.reset_parameters()
        if self.control_dim > 0:
            self._self.reset_control_parameters()
        self._nbr.reset_parameters()
        self._reset_neighbor_parameters()

    @property
    def K_self(self) -> Tensor:
        """Self-coupling matrix with shape ``(latent_dim, latent_dim)``.

        Returns
        -------
        Tensor
            Assembled ``K_self``.
        """
        return self._self.K

    @property
    def K_nbr(self) -> Tensor:
        """Neighbor-coupling matrix with shape ``(latent_dim, latent_dim)``.

        Returns
        -------
        Tensor
            Assembled ``K_nbr``.
        """
        return self._nbr.K

    @property
    def matrix(self) -> Tensor:
        """Self-term matrix (contract surface; topology-coupled spectrum differs).

        Returns
        -------
        Tensor
            ``K_self``. Use :meth:`effective_matrix` / :meth:`spectrum` for the
            full ``N·d`` networked operator on a given topology.
        """
        return self.K_self

    @property
    def K(self) -> Tensor:
        """Alias of :attr:`matrix` (``K_self``) for per-node API familiarity.

        Returns
        -------
        Tensor
            ``K_self``.
        """
        return self.K_self

    def set_dense_matrices(
        self,
        k_self: Tensor,
        k_nbr: Tensor,
        *,
        control_matrix: Tensor | None = None,
        bilinear_matrices: Tensor | None = None,
    ) -> None:
        """Write dense ``K_self`` / ``K_nbr`` (and optional control factors).

        Parameters
        ----------
        k_self : Tensor
            Dense self matrix ``(latent_dim, latent_dim)``.
        k_nbr : Tensor
            Dense neighbor matrix ``(latent_dim, latent_dim)``.
        control_matrix : Tensor or None, optional
            Control matrix ``B`` when ``control_dim > 0``.
        bilinear_matrices : Tensor or None, optional
            Full-rank bilinear stack when ``control_mode="bilinear"``.
        """
        self._self.set_dense_matrix(
            k_self,
            control_matrix=control_matrix,
            bilinear_matrices=bilinear_matrices,
        )
        self._nbr.set_dense_matrix(k_nbr, control_matrix=None)

    def bound_metric(self) -> Tensor:
        """Return ``max(bound(K_self), bound(K_nbr))`` for monitoring.

        Returns
        -------
        Tensor
            Scalar bound metric used by soft stability losses.
        """
        return torch.maximum(self._self.bound_metric(), self._nbr.bound_metric())

    def spectral_radius(self) -> Tensor:
        """Return ``max(|λ|)`` of ``K_self`` (not the full ``N·d`` operator).

        Returns
        -------
        Tensor
            Spectral radius of the self-coupling matrix only.
        """
        return self._self.spectral_radius()

    def stability_certificate(self) -> StabilityCertificate | None:
        """Return the self-term certificate when a structural mode is active.

        Returns
        -------
        StabilityCertificate or None
            Certificate from the self-coupling factor, if any.
        """
        return self._self.stability_certificate()

    def effective_matrix(
        self,
        edge_index: Tensor,
        num_nodes: int,
        edge_weight: Tensor | None = None,
        *,
        k_self: Tensor | None = None,
    ) -> Tensor:
        """Assemble the dense effective operator ``I⊗K_self + Â⊗K_nbr``.

        Parameters
        ----------
        edge_index : Tensor
            Edge index ``(2, E)``.
        num_nodes : int
            Number of nodes ``N``.
        edge_weight : Tensor or None, optional
            Optional edge weights ``(E,)``.
        k_self : Tensor or None, optional
            Optional override for the self-coupling matrix (used when folding
            a global bilinear term into ``K_self`` for inversion).

        Returns
        -------
        Tensor
            Dense matrix with shape ``(N·d, N·d)``.
        """
        from koopman_graph.graph_utils import dense_symmetric_normalized_adjacency

        self_matrix = self.K_self if k_self is None else k_self
        adj = dense_symmetric_normalized_adjacency(
            edge_index,
            num_nodes,
            edge_weight=edge_weight,
            dtype=self_matrix.dtype,
        )
        identity = torch.eye(num_nodes, dtype=adj.dtype, device=adj.device)
        return torch.kron(identity, self_matrix) + torch.kron(adj, self.K_nbr)

    def spectrum(
        self,
        edge_index: Tensor,
        num_nodes: int,
        *,
        edge_weight: Tensor | None = None,
        time_step: float = 1.0,
    ) -> KoopmanSpectrum:
        """Eigendecomposition of the effective ``N·d`` networked operator.

        Parameters
        ----------
        edge_index : Tensor
            Topology used to build ``Â``.
        num_nodes : int
            Node count ``N``.
        edge_weight : Tensor or None, optional
            Optional edge weights.
        time_step : float, optional
            Discrete sampling interval for growth rates / frequencies.

        Returns
        -------
        KoopmanSpectrum
            Spectrum of :meth:`effective_matrix`.
        """
        # Lazy import keeps operators → analysis dependency out of package init.
        from koopman_graph.analysis.spectrum import compute_spectrum

        return compute_spectrum(
            self.effective_matrix(edge_index, num_nodes, edge_weight=edge_weight),
            time_step,
        )

    def forward(
        self,
        z: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor | None = None,
        control: Tensor | None = None,
    ) -> Tensor:
        """Advance latents with topology-coupled linear message passing.

        Parameters
        ----------
        z : Tensor
            Latent node states with shape ``(num_nodes, latent_dim)``.
        edge_index : Tensor
            Edge index ``(2, num_edges)`` used to build ``Â``.
        edge_weight : Tensor or None, optional
            Optional edge weights.
        control : Tensor or None, optional
            Exogenous control when ``control_dim > 0``.

        Returns
        -------
        Tensor
            Advanced latents with the same shape as ``z``.
        """
        if z.ndim != 2:
            msg = (
                "GraphKoopmanOperator expects z with shape "
                f"(num_nodes, latent_dim), got {tuple(z.shape)}"
            )
            raise ValueError(msg)
        if z.shape[-1] != self.latent_dim:
            msg = (
                f"Expected trailing dimension {self.latent_dim}, "
                f"got shape {tuple(z.shape)}"
            )
            raise ValueError(msg)

        from koopman_graph.graph_utils import symmetric_normalized_adjacency_matvec

        neighbor = symmetric_normalized_adjacency_matvec(
            edge_index,
            z,
            edge_weight=edge_weight,
            num_nodes=z.shape[0],
        )
        z_next = z @ self.K_self.T + neighbor @ self.K_nbr.T

        if self.control_dim == 0:
            if control is not None:
                msg = "control input provided to an uncontrolled operator"
                raise ValueError(msg)
            return z_next
        if control is None:
            msg = "control input is required when control_dim > 0"
            raise ValueError(msg)
        offset = self._self.control_term(control, num_nodes=z.shape[0])
        if control.ndim == 1:
            offset = self._self._broadcast_control_term(z, offset)
        z_next = z_next + offset
        if self.control_mode == "bilinear":
            z_next = z_next + bilinear_state_control_term(
                z,
                control,
                self._self.bilinear_matrices(),
            )
        return z_next

    def advance(
        self,
        z: Tensor,
        delta_t: float | Tensor | None = None,
        *,
        control: Tensor | None = None,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        """Contract advance; requires ``edge_index`` for networked coupling.

        Parameters
        ----------
        z : Tensor
            Latent states ``(num_nodes, latent_dim)``.
        delta_t : float, Tensor, or None, optional
            Ignored (discrete operator).
        control : Tensor or None, optional
            Optional control input.
        edge_index : Tensor or None, optional
            Required graph topology for this step.
        edge_weight : Tensor or None, optional
            Optional edge weights.

        Returns
        -------
        Tensor
            Advanced latent states.
        """
        _ = delta_t
        if edge_index is None:
            msg = "edge_index is required for GraphKoopmanOperator.advance"
            raise ValueError(msg)
        return self.forward(z, edge_index, edge_weight, control=control)

    def inverse_advance(
        self,
        z: Tensor,
        delta_t: float | Tensor | None = None,
        *,
        control: Tensor | None = None,
        inverse_matrix: Tensor | None = None,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        """Recover previous latents by inverting the effective ``N·d`` map.

        Dense inversion is used (suitable for modest ``N``). ``inverse_matrix``,
        when provided, must be the effective ``(N·d, N·d)`` inverse for the
        same topology; otherwise it is assembled on demand.

        Parameters
        ----------
        z : Tensor
            Latents at ``t+1`` with shape ``(num_nodes, latent_dim)``.
        delta_t : float, Tensor, or None, optional
            Ignored.
        control : Tensor or None, optional
            Control that drove the forward step.
        inverse_matrix : Tensor or None, optional
            Optional precomputed effective inverse.
        edge_index : Tensor or None, optional
            Required topology.
        edge_weight : Tensor or None, optional
            Optional edge weights.

        Returns
        -------
        Tensor
            Recovered latents at ``t``.
        """
        _ = delta_t
        if edge_index is None:
            msg = "edge_index is required for GraphKoopmanOperator.inverse_advance"
            raise ValueError(msg)
        if z.ndim != 2 or z.shape[-1] != self.latent_dim:
            msg = (
                "GraphKoopmanOperator.inverse_advance expects z with shape "
                f"(num_nodes, {self.latent_dim}), got {tuple(z.shape)}"
            )
            raise ValueError(msg)

        adjusted = z
        if self.control_dim > 0:
            if control is None:
                msg = "control input is required when control_dim > 0"
                raise ValueError(msg)
            offset = self._self.control_term(control, num_nodes=z.shape[0])
            if control.ndim == 1:
                offset = self._self._broadcast_control_term(z, offset)
            adjusted = z - offset

        num_nodes = z.shape[0]
        if inverse_matrix is None:
            k_self_override: Tensor | None = None
            if self.control_mode == "bilinear":
                if control is None or control.ndim != 1:
                    msg = (
                        "GraphKoopmanOperator.inverse_advance supports bilinear "
                        "mode with global controls only"
                    )
                    raise ValueError(msg)
                k_self_override = effective_bilinear_matrix(
                    self.K_self,
                    control,
                    self._self.bilinear_matrices(),
                )
            effective = self.effective_matrix(
                edge_index,
                num_nodes,
                edge_weight=edge_weight,
                k_self=k_self_override,
            )
            try:
                inverse_matrix = torch.linalg.inv(effective)
            except RuntimeError:
                inverse_matrix = torch.linalg.pinv(effective)

        flat = adjusted.reshape(-1)
        recovered = (inverse_matrix @ flat).view_as(adjusted)
        return recovered
