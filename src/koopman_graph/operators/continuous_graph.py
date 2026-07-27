"""Continuous networked (spatially-coupled) Koopman generator.

Implements the topology-coupled generator (adjacency-dependent)::

    L_eff = I_N ⊗ L_self + Â_* ⊗ L_nbr  (+ Â_b ⊗ L_bwd in dual mode)

with one-step map ``K(Δt) = exp(L_eff Δt)``. Discrete networked peers live in
:mod:`koopman_graph.operators.graph`. Selected via
``koopman="graph"`` + ``dynamics_mode="continuous"`` or the alias
``koopman="continuous_graph"``.

The dense path forms an ``N·d`` matrix exponential (costly for large ``N``);
``sparsity="block_diagonal"`` advances with the self-term only (self-dominated
approximation; exact when neighbor factors are zero or the graph has no edges).
Neighbor adjacency mode does **not** change the block-diagonal shortcut.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

from koopman_graph.operators.continuous import ContinuousKoopmanOperator
from koopman_graph.operators.continuous_propagation import (
    advance_uncontrolled_fixed,
    advance_van_loan,
)
from koopman_graph.operators.continuous_van_loan import van_loan_factors
from koopman_graph.operators.contract import (
    InitMode,
    Parameterization,
    StabilityCertificate,
)
from koopman_graph.operators.control import (
    ControlMode,
    effective_bilinear_matrix,
    per_node_effective_bilinear_matrices,
)
from koopman_graph.operators.graph_types import GRAPH_ADJACENCY_MODES, GraphAdjacency
from koopman_graph.spectrum_types import KoopmanSpectrum, compute_generator_spectrum

ContinuousGraphSparsity = Literal["dense", "block_diagonal", "distributed"]


class ContinuousGraphKoopmanOperator(nn.Module):
    """Continuous-time networked Koopman generator with self/neighbor coupling.

    Advances stacked node latents ``Z ∈ R^{N×d}`` over ``Δt`` via::

        vec(Z(t+Δt)) = exp(L_eff Δt) vec(Z(t))

    where ``L_eff`` depends on ``adjacency``:

    * ``"symmetric"`` (default): ``I⊗L_self + Â_sym⊗L_nbr``
    * ``"random_walk"``: ``I⊗L_self + Â_f⊗L_nbr``
    * ``"dual_random_walk"``:
      ``I⊗L_self + Â_f⊗L_fwd + Â_b⊗L_bwd`` (``L_fwd`` aliases ``L_nbr``)

    Control (when enabled) is owned by the self factor and integrated with Van
    Loan factors on the dense ``N·d`` generator. Prefer modest ``N`` for the
    dense path; large graphs should use ``sparsity="block_diagonal"`` (self
    only; ignores neighbor coupling and adjacency). Dense uncontrolled
    advances may reuse assembled ``L_eff`` and ``Φ = exp(Δt L_eff)`` within
    one training-loss evaluation; call :meth:`clear_transition_cache` between
    evaluations (``compute_training_loss`` does this automatically).

    Attributes
    ----------
    latent_dim : int
        Latent feature dimension ``d``.
    control_dim : int
        Exogenous control dimension (``0`` disables control).
    parameterization : Parameterization
        Shared soft/structural parameterization for ``L_self`` / neighbor
        factors.
    adjacency : {"symmetric", "random_walk", "dual_random_walk"}
        Neighbor-coupling normalization (default ``"symmetric"``).
    sparsity : {"dense", "block_diagonal", "distributed"}
        Realization mode. ``"dense"`` uses the full ``N·d`` exponential;
        ``"block_diagonal"`` advances with ``L_self`` only; ``"distributed"``
        is rejected.
    max_real_eigenvalue : float
        Stability bound forwarded to the continuous factor modules.
    """

    def __init__(
        self,
        latent_dim: int,
        *,
        init_mode: InitMode = "identity_noise",
        init_scale: float = 1e-2,
        parameterization: Parameterization = "dense",
        max_real_eigenvalue: float = 1.0,
        control_dim: int = 0,
        control_mode: ControlMode = "additive",
        bilinear_rank: int | None = None,
        sparsity: ContinuousGraphSparsity = "dense",
        adjacency: GraphAdjacency = "symmetric",
    ) -> None:
        """Initialize self and neighbor continuous generators.

        Parameters
        ----------
        latent_dim : int
            Latent dimension ``d``.
        init_mode : {"identity", "identity_noise", "xavier"}, optional
            Initialization for ``L_self``. ``L_nbr`` / ``L_fwd`` starts near
            zero. ``L_bwd`` (dual mode only) initializes at exactly zero.
        init_scale : float, optional
            Noise scale for ``identity_noise`` / neighbor jitter.
        parameterization : Parameterization, optional
            Shared parameterization for the ``d×d`` generators. Continuous-only
            ``"auxiliary_spectral"`` is rejected.
        max_real_eigenvalue : float, optional
            Magnitude scale for structural Hurwitz modes.
        control_dim : int, optional
            Additive / bilinear control dimension. Default ``0``.
        control_mode : {"additive", "bilinear"}, optional
            Control coupling on the self-term.
        bilinear_rank : int or None, optional
            Low-rank bilinear size when ``control_mode="bilinear"``.
        sparsity : {"dense", "block_diagonal", "distributed"}, optional
            Realization mode. Default ``"dense"``. ``"block_diagonal"`` is a
            self-term-only shortcut and ignores neighbor / adjacency coupling.
        adjacency : {"symmetric", "random_walk", "dual_random_walk"}, optional
            Neighbor-coupling normalization. Default ``"symmetric"`` preserves
            historical undirected behavior bit-for-bit.

        Raises
        ------
        ValueError
            If ``sparsity`` / ``adjacency`` / parameterization are unsupported
            or args invalid.
        """
        super().__init__()
        if sparsity == "distributed":
            msg = (
                "ContinuousGraphKoopmanOperator sparsity='distributed' is "
                "planned; not in 0.6.0. Use sparsity='dense' or "
                "'block_diagonal'"
            )
            raise ValueError(msg)
        if sparsity not in {"dense", "block_diagonal"}:
            msg = (
                "ContinuousGraphKoopmanOperator sparsity must be 'dense' or "
                f"'block_diagonal', got {sparsity!r}"
            )
            raise ValueError(msg)
        if adjacency not in GRAPH_ADJACENCY_MODES:
            accepted = ", ".join(sorted(GRAPH_ADJACENCY_MODES))
            msg = (
                "ContinuousGraphKoopmanOperator adjacency must be one of "
                f"{{{accepted}}}, got {adjacency!r}"
            )
            raise ValueError(msg)
        if parameterization == "auxiliary_spectral":
            msg = (
                "parameterization='auxiliary_spectral' is not supported for "
                "ContinuousGraphKoopmanOperator (state-dependent + topology)"
            )
            raise ValueError(msg)

        self.latent_dim = latent_dim
        self.init_mode = init_mode
        self.init_scale = init_scale
        self.parameterization = parameterization
        self.max_real_eigenvalue = max_real_eigenvalue
        self.control_dim = control_dim
        self.control_mode = control_mode
        self.bilinear_rank = bilinear_rank
        self.sparsity = sparsity
        self.adjacency = adjacency

        self._self = ContinuousKoopmanOperator(
            latent_dim,
            init_mode=init_mode,
            init_scale=init_scale,
            parameterization=parameterization,
            max_real_eigenvalue=max_real_eigenvalue,
            control_dim=control_dim,
            control_mode=control_mode,
            bilinear_rank=bilinear_rank,
        )
        self._nbr = ContinuousKoopmanOperator(
            latent_dim,
            init_mode="identity",
            init_scale=init_scale,
            parameterization=parameterization,
            max_real_eigenvalue=max_real_eigenvalue,
            control_dim=0,
        )
        self._bwd: ContinuousKoopmanOperator | None
        if adjacency == "dual_random_walk":
            self._bwd = ContinuousKoopmanOperator(
                latent_dim,
                init_mode="identity",
                init_scale=init_scale,
                parameterization=parameterization,
                max_real_eigenvalue=max_real_eigenvalue,
                control_dim=0,
            )
        else:
            self._bwd = None
        self._reset_neighbor_parameters()
        # Ephemeral L_eff / Φ reuse within one training-loss evaluation.
        # Cleared by :meth:`clear_transition_cache` (wired from compute_training_loss).
        self._leff_cache: list[
            tuple[Tensor, Tensor | None, int, torch.dtype, torch.device, Tensor]
        ] = []
        self._phi_cache: list[
            tuple[Tensor, Tensor | None, float, torch.dtype, torch.device, Tensor]
        ] = []

    def clear_transition_cache(self) -> None:
        """Drop cached dense ``L_eff`` and transition matrices ``Φ = exp(Δt L_eff)``.

        Call at the start of each training-loss evaluation so cached
        generators and transitions never span an optimizer step. Ordinary
        topology / ``Δt`` changes miss the cache key and rebuild
        automatically.

        Notes
        -----
        Entries are ephemeral and never written to ``state_dict``. Bilinear
        pair-local generators (``l_self`` / ``l_self_blocks`` overrides) and
        Van Loan controlled advances do not use these caches.
        """
        self._leff_cache.clear()
        self._phi_cache.clear()

    def _topology_payload_equal(
        self,
        edge_index_a: Tensor,
        edge_weight_a: Tensor | None,
        edge_index_b: Tensor,
        edge_weight_b: Tensor | None,
    ) -> bool:
        """Return whether two pairwise topology payloads match by content.

        Parameters
        ----------
        edge_index_a, edge_index_b : Tensor
            COO edge indices.
        edge_weight_a, edge_weight_b : Tensor or None
            Optional edge weights.

        Returns
        -------
        bool
            ``True`` when indices and weights match (including both absent).
        """
        if not torch.equal(edge_index_a, edge_index_b):
            return False
        if (edge_weight_a is None) != (edge_weight_b is None):
            return False
        if edge_weight_a is None:
            return True
        assert edge_weight_b is not None
        return torch.allclose(edge_weight_a, edge_weight_b, equal_nan=True)

    def _lookup_cached_generator(
        self,
        edge_index: Tensor,
        edge_weight: Tensor | None,
        num_nodes: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor | None:
        """Return a cached ``L_eff`` for matching topology / size / dtype / device.

        Parameters
        ----------
        edge_index : Tensor
            Pairwise topology.
        edge_weight : Tensor or None
            Optional edge weights.
        num_nodes : int
            Node count ``N``.
        dtype : torch.dtype
            Floating dtype of ``L_eff``.
        device : torch.device
            Device of ``L_eff``.

        Returns
        -------
        Tensor or None
            Cached generator, or ``None`` on miss.
        """
        for (
            cached_index,
            cached_weight,
            cached_nodes,
            cached_dtype,
            cached_device,
            cached_generator,
        ) in self._leff_cache:
            if (
                cached_nodes == num_nodes
                and cached_dtype == dtype
                and cached_device == device
                and self._topology_payload_equal(
                    edge_index, edge_weight, cached_index, cached_weight
                )
            ):
                return cached_generator
        return None

    def _lookup_cached_transition(
        self,
        edge_index: Tensor,
        edge_weight: Tensor | None,
        delta_value: float,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor | None:
        """Return a cached ``Φ`` for matching topology / ``Δt`` / dtype / device.

        Parameters
        ----------
        edge_index : Tensor
            Pairwise topology.
        edge_weight : Tensor or None
            Optional edge weights.
        delta_value : float
            Scalar integration interval.
        dtype : torch.dtype
            Floating dtype of ``Φ``.
        device : torch.device
            Device of ``Φ``.

        Returns
        -------
        Tensor or None
            Cached transition, or ``None`` on miss.
        """
        for (
            cached_index,
            cached_weight,
            cached_delta,
            cached_dtype,
            cached_device,
            cached_phi,
        ) in self._phi_cache:
            if (
                cached_delta == delta_value
                and cached_dtype == dtype
                and cached_device == device
                and self._topology_payload_equal(
                    edge_index, edge_weight, cached_index, cached_weight
                )
            ):
                return cached_phi
        return None

    def _reset_factor_parameters(
        self,
        module: ContinuousKoopmanOperator,
        *,
        allow_noise: bool,
    ) -> None:
        """Zero a neighbor generator, optionally adding ``init_scale`` noise.

        Parameters
        ----------
        module : ContinuousKoopmanOperator
            Neighbor factor module to reset.
        allow_noise : bool
            When ``True`` and ``init_mode`` is noisy, add ``init_scale`` jitter.
        """
        if self.parameterization == "dense":
            dense_l = module.L
            with torch.no_grad():
                dense_l.zero_()
                if allow_noise and self.init_mode in {"identity_noise", "xavier"}:
                    dense_l.add_(torch.randn_like(dense_l) * self.init_scale)
            return
        with torch.no_grad():
            for parameter in module.parameters():
                parameter.zero_()
            if allow_noise and self.init_mode in {"identity_noise", "xavier"}:
                for parameter in module.parameters():
                    parameter.add_(torch.randn_like(parameter) * self.init_scale)

    def _reset_neighbor_parameters(self) -> None:
        """Initialize neighbor generators for a per-node-like starting point.

        Notes
        -----
        ``L_nbr`` / ``L_fwd`` may receive ``init_scale`` noise.
        ``L_bwd`` (dual mode) is always exactly zero so
        ``dual_random_walk`` begins equivalent to ``random_walk``.
        """
        self._reset_factor_parameters(self._nbr, allow_noise=True)
        if self._bwd is not None:
            self._reset_factor_parameters(self._bwd, allow_noise=False)

    def reset_parameters(self) -> None:
        """Reinitialize ``L_self`` / neighbor factors (and control when present).

        Returns
        -------
        None
            See summary line.
        """
        self._self.reset_parameters()
        if self.control_dim > 0:
            self._self.reset_control_parameters()
        self._nbr.reset_parameters()
        if self._bwd is not None:
            self._bwd.reset_parameters()
        self._reset_neighbor_parameters()

    @property
    def L_self(self) -> Tensor:
        """Self-coupling generator with shape ``(latent_dim, latent_dim)``.

        Returns
        -------
        Tensor
            Assembled ``L_self``.
        """
        return self._self.L

    @property
    def L_nbr(self) -> Tensor:
        """Forward / sole neighbor generator ``(latent_dim, latent_dim)``.

        Returns
        -------
        Tensor
            Assembled ``L_nbr`` (alias :attr:`L_fwd`).
        """
        return self._nbr.L

    @property
    def L_fwd(self) -> Tensor:
        """Alias of :attr:`L_nbr` (forward random-walk coupling).

        Returns
        -------
        Tensor
            Assembled forward neighbor generator.
        """
        return self.L_nbr

    @property
    def L_bwd(self) -> Tensor:
        """Backward random-walk generator (``dual_random_walk`` only).

        Returns
        -------
        Tensor
            Assembled ``L_bwd``.

        Raises
        ------
        AttributeError
            If ``adjacency`` is not ``"dual_random_walk"``.
        """
        if self._bwd is None:
            msg = "L_bwd is only available when adjacency='dual_random_walk'"
            raise AttributeError(msg)
        return self._bwd.L

    @property
    def matrix(self) -> Tensor:
        """Self-term generator (contract surface; topology-coupled spectrum differs).

        Returns
        -------
        Tensor
            ``L_self``.
        """
        return self.L_self

    @property
    def L(self) -> Tensor:
        """Alias of :attr:`matrix` (``L_self``).

        Returns
        -------
        Tensor
            ``L_self``.
        """
        return self.matrix

    def set_dense_matrices(
        self,
        l_self: Tensor,
        l_nbr: Tensor,
        *,
        l_bwd: Tensor | None = None,
        control_matrix: Tensor | None = None,
        bilinear_matrices: Tensor | None = None,
    ) -> None:
        """Write dense ``L_self`` / neighbor generators (and optional control).

        Parameters
        ----------
        l_self : Tensor
            Dense self generator ``(latent_dim, latent_dim)``.
        l_nbr : Tensor
            Dense forward / sole neighbor generator ``(latent_dim, latent_dim)``.
        l_bwd : Tensor or None, optional
            Dense backward neighbor generator when
            ``adjacency="dual_random_walk"``. Must be omitted otherwise.
            ``None`` leaves ``L_bwd`` unchanged in dual mode.
        control_matrix : Tensor or None, optional
            Control matrix ``B`` when ``control_dim > 0``.
        bilinear_matrices : Tensor or None, optional
            Full-rank bilinear stack when ``control_mode="bilinear"``.

        Raises
        ------
        ValueError
            If ``l_bwd`` is set when ``adjacency`` is not dual.
        """
        if l_bwd is not None and self._bwd is None:
            msg = "l_bwd is only valid when adjacency='dual_random_walk'"
            raise ValueError(msg)
        self._self.set_dense_matrix(
            l_self,
            control_matrix=control_matrix,
            bilinear_matrices=bilinear_matrices,
        )
        self._nbr.set_dense_matrix(l_nbr, control_matrix=None)
        if l_bwd is not None:
            assert self._bwd is not None
            self._bwd.set_dense_matrix(l_bwd, control_matrix=None)

    @property
    def B(self) -> Tensor | None:
        """Control matrix from the self factor, when controlled.

        Returns
        -------
        Tensor | None
            Control matrix ``B``, or ``None`` when uncontrolled.
        """
        if self.control_dim <= 0:
            return None
        return self._self.B

    def bound_metric(self) -> Tensor:
        """Return ``max`` of self / neighbor factor bounds (surrogate).

        Notes
        -----
        Not a whole-network Hurwitz certificate for the ``N·d`` generator.

        Returns
        -------
        Tensor
            Scalar factor bound metric.
        """
        metric = torch.maximum(self._self.bound_metric(), self._nbr.bound_metric())
        if self._bwd is not None:
            metric = torch.maximum(metric, self._bwd.bound_metric())
        return metric

    def max_real_part(self) -> Tensor:
        """Maximum real eigenvalue of ``L_self`` (not the full ``N·d`` generator).

        Returns
        -------
        Tensor
            See summary line."""
        return self._self.max_real_part()

    def spectral_radius(self) -> Tensor:
        """Contract alias: reports ``max_real_part`` of ``L_self``.

        Returns
        -------
        Tensor
            See summary line."""
        return self.max_real_part()

    def stability_certificate(self) -> StabilityCertificate | None:
        """Structural certificate for ``L_self`` when available.

        Returns
        -------
        StabilityCertificate | None
            See summary line."""
        return self._self.stability_certificate()

    def _dense_neighbor_coupling(
        self,
        edge_index: Tensor,
        num_nodes: int,
        *,
        edge_weight: Tensor | None,
        dtype: torch.dtype,
    ) -> Tensor:
        """Assemble neighbor Kronecker terms for the configured adjacency mode.

        Parameters
        ----------
        edge_index : Tensor
            Edge index ``(2, E)``.
        num_nodes : int
            Number of nodes ``N``.
        edge_weight : Tensor or None
            Optional edge weights ``(E,)``.
        dtype : torch.dtype
            Floating dtype for the dense adjacency factors.

        Returns
        -------
        Tensor
            Dense neighbor coupling with shape ``(N·d, N·d)``.
        """
        from koopman_graph.graph_utils.topology import (
            dense_random_walk_normalized_adjacency,
            dense_symmetric_normalized_adjacency,
        )

        if self.adjacency == "symmetric":
            adj = dense_symmetric_normalized_adjacency(
                edge_index,
                num_nodes,
                edge_weight=edge_weight,
                dtype=dtype,
            )
            return torch.kron(adj, self.L_nbr)

        adj_fwd = dense_random_walk_normalized_adjacency(
            edge_index,
            num_nodes,
            edge_weight=edge_weight,
            dtype=dtype,
            direction="forward",
        )
        coupling = torch.kron(adj_fwd, self.L_nbr)
        if self.adjacency == "random_walk":
            return coupling
        assert self._bwd is not None
        adj_bwd = dense_random_walk_normalized_adjacency(
            edge_index,
            num_nodes,
            edge_weight=edge_weight,
            dtype=dtype,
            direction="backward",
        )
        return coupling + torch.kron(adj_bwd, self.L_bwd)

    def effective_generator(
        self,
        edge_index: Tensor,
        num_nodes: int,
        edge_weight: Tensor | None = None,
        *,
        l_self: Tensor | None = None,
        l_self_blocks: Tensor | None = None,
    ) -> Tensor:
        """Assemble the dense effective networked generator ``(N·d, N·d)``.

        Parameters
        ----------
        edge_index : Tensor
            Edge index ``(2, E)``.
        num_nodes : int
            Number of nodes ``N``.
        edge_weight : Tensor or None, optional
            Optional edge weights ``(E,)``.
        l_self : Tensor or None, optional
            Optional override for a shared self generator.
        l_self_blocks : Tensor or None, optional
            Optional per-node self blocks ``(N, d, d)``.

        Returns
        -------
        Tensor
            Dense generator with shape ``(N·d, N·d)``.

        Raises
        ------
        ValueError
            If both ``l_self`` and ``l_self_blocks`` are set, or if
            ``l_self_blocks`` has the wrong shape.

        Notes
        -----
        When ``l_self`` and ``l_self_blocks`` are both omitted, repeated calls
        with the same topology reuse an evaluation-scoped ``L_eff`` (see
        :meth:`clear_transition_cache`). Overrides skip that cache.
        """
        if l_self is not None and l_self_blocks is not None:
            msg = "Pass at most one of l_self and l_self_blocks"
            raise ValueError(msg)

        use_cache = l_self is None and l_self_blocks is None
        if use_cache:
            cached = self._lookup_cached_generator(
                edge_index,
                edge_weight,
                num_nodes,
                self.L_self.dtype,
                self.L_self.device,
            )
            if cached is not None:
                return cached

        self_matrix = self.L_self if l_self is None else l_self
        neighbor = self._dense_neighbor_coupling(
            edge_index,
            num_nodes,
            edge_weight=edge_weight,
            dtype=self_matrix.dtype,
        )
        if l_self_blocks is None:
            identity = torch.eye(
                num_nodes,
                dtype=neighbor.dtype,
                device=neighbor.device,
            )
            generator = torch.kron(identity, self_matrix) + neighbor
        else:
            expected = (num_nodes, self.latent_dim, self.latent_dim)
            if l_self_blocks.shape != expected:
                msg = (
                    f"l_self_blocks must have shape {expected}, "
                    f"got {tuple(l_self_blocks.shape)}"
                )
                raise ValueError(msg)
            self_blocks = torch.block_diag(*l_self_blocks.unbind(0))
            generator = self_blocks + neighbor

        if use_cache:
            self._leff_cache.append(
                (
                    edge_index,
                    edge_weight,
                    num_nodes,
                    generator.dtype,
                    generator.device,
                    generator,
                )
            )
        return generator

    def spectrum(
        self,
        edge_index: Tensor,
        num_nodes: int,
        *,
        edge_weight: Tensor | None = None,
    ) -> KoopmanSpectrum:
        """Eigendecomposition of the effective ``N·d`` networked generator.

        Directed / dual modes may yield complex spectra; growth rates and
        frequencies come from the complex eigendecomposition.

        Parameters
        ----------
        edge_index : Tensor
            Pairwise edge index with shape ``(2, E)``.
        num_nodes : int
            Node count ``N``.
        edge_weight : Tensor or None, optional
            Optional scalar edge weights with shape ``(E,)``.

        Returns
        -------
        KoopmanSpectrum
            Magnitude-sorted spectrum of ``L_eff``.
        """
        return compute_generator_spectrum(
            self.effective_generator(edge_index, num_nodes, edge_weight=edge_weight)
        )

    def transition_matrix(
        self,
        delta_t: float | Tensor,
        edge_index: Tensor,
        num_nodes: int,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        """Return ``exp(L_eff Δt)`` for the dense networked generator.

        Within an evaluation, repeated calls with the same topology and
        scalar ``Δt`` reuse a cached ``Φ``; distinct ``Δt`` values reuse
        cached ``L_eff`` (see :meth:`clear_transition_cache`).

        Parameters
        ----------
        delta_t : float or Tensor
            Integration interval.
        edge_index : Tensor
            Edge index ``(2, E)``.
        num_nodes : int
            Number of nodes ``N``.
        edge_weight : Tensor or None, optional
            Optional edge weights ``(E,)``.

        Returns
        -------
        Tensor
            Dense transition matrix with shape ``(N·d, N·d)``.
        """
        dtype = self.L_self.dtype
        device = self.L_self.device
        delta = torch.as_tensor(delta_t, dtype=dtype, device=device)
        delta_value = float(delta.detach().reshape(-1)[0].item())
        cached_phi = self._lookup_cached_transition(
            edge_index,
            edge_weight,
            delta_value,
            dtype,
            device,
        )
        if cached_phi is not None:
            return cached_phi
        generator = self.effective_generator(
            edge_index, num_nodes, edge_weight=edge_weight
        )
        phi = torch.linalg.matrix_exp(generator * delta)
        self._phi_cache.append(
            (
                edge_index,
                edge_weight,
                delta_value,
                generator.dtype,
                generator.device,
                phi,
            )
        )
        return phi

    def _networked_control_matrix(self, num_nodes: int) -> Tensor:
        """Build ``B_eff`` with shape ``(C, N·d)`` for global additive control.

        Parameters
        ----------

        num_nodes : int
            See the function signature / summary for ``num_nodes``.

        Returns
        -------

        Tensor
            See summary line.

        Raises
        ------

        ValueError
            Raised when inputs are invalid."""
        if self.control_dim <= 0 or self.B is None:
            msg = "control matrix requested for an uncontrolled operator"
            raise ValueError(msg)
        # Stack the same B across nodes: B_eff = [B, B, ..., B].
        return self.B.repeat(1, num_nodes)

    def _advance_dense(
        self,
        z: Tensor,
        delta_t: Tensor,
        *,
        control: Tensor | None,
        edge_index: Tensor,
        edge_weight: Tensor | None,
    ) -> Tensor:
        """Dense ``N·d`` matrix-exponential advance (with optional Van Loan).

        Parameters
        ----------

        z : Tensor
            See the function signature / summary for ``z``.
        delta_t : Tensor
            See the function signature / summary for ``delta_t``.
        control : Tensor | None
            See the function signature / summary for ``control``.
        edge_index : Tensor
            See the function signature / summary for ``edge_index``.
        edge_weight : Tensor | None
            See the function signature / summary for ``edge_weight``.

        Returns
        -------

        Tensor
            See summary line.

        Raises
        ------

        ValueError
            Raised when inputs are invalid."""
        num_nodes = z.shape[0]
        flat = z.reshape(1, -1)

        l_self_override: Tensor | None = None
        l_self_blocks: Tensor | None = None
        if (
            self.control_dim > 0
            and self.control_mode == "bilinear"
            and control is not None
        ):
            coupling = self._self.bilinear_matrices()
            if control.ndim == 1:
                l_self_override = effective_bilinear_matrix(
                    self.L_self, control, coupling
                )
            elif control.ndim == 2:
                l_self_blocks = per_node_effective_bilinear_matrices(
                    self.L_self, control, coupling
                )

        # Uncontrolled dense path: reuse evaluation-scoped Φ when the generator
        # is the default (no bilinear self overrides).
        if self.control_dim == 0 and l_self_override is None and l_self_blocks is None:
            if control is not None:
                msg = "control input provided to an uncontrolled operator"
                raise ValueError(msg)
            transition = self.transition_matrix(
                delta_t, edge_index, num_nodes, edge_weight=edge_weight
            )
            return (flat @ transition.T).view_as(z)

        generator = self.effective_generator(
            edge_index,
            num_nodes,
            edge_weight=edge_weight,
            l_self=l_self_override,
            l_self_blocks=l_self_blocks,
        )

        if self.control_dim == 0:
            if control is not None:
                msg = "control input provided to an uncontrolled operator"
                raise ValueError(msg)
            advanced = advance_uncontrolled_fixed(flat, generator, delta_t)
            return advanced.view_as(z)

        if control is None:
            msg = "control input is required when control_dim > 0"
            raise ValueError(msg)

        b_eff = self._networked_control_matrix(num_nodes)
        if control.ndim == 1:
            advanced = advance_van_loan(
                flat,
                delta_t,
                control,
                generator=generator,
                control_matrix=b_eff,
                latent_dim=num_nodes * self.latent_dim,
            )
            return advanced.view_as(z)
        if control.ndim == 2:
            # Per-node controls: integrate each node against a shared Phi11 from
            # L_eff and a node-local Van Loan control block built from B.
            phi11, _ = van_loan_factors(generator, b_eff, delta_t)
            free = (flat @ phi11.T).view_as(z)
            # Rebuild per-node offsets via the self-term Van Loan factors.
            node_advanced = torch.empty_like(z)
            for node_idx in range(num_nodes):
                node_u = control[node_idx]
                node_gen = self.L_self
                if self.control_mode == "bilinear":
                    node_gen = effective_bilinear_matrix(
                        self.L_self,
                        node_u,
                        self._self.bilinear_matrices(),
                    )
                node_advanced[node_idx : node_idx + 1] = advance_van_loan(
                    z[node_idx : node_idx + 1],
                    delta_t,
                    node_u,
                    generator=node_gen,
                    control_matrix=self.B,
                    latent_dim=self.latent_dim,
                )
            # Topology-coupled free motion + per-node control offsets relative
            # to the uncontrolled self advance.
            self_free = advance_uncontrolled_fixed(z, self.L_self, delta_t)
            return free + (node_advanced - self_free)

        msg = (
            "control input must have shape (control_dim,) or "
            f"(num_nodes, control_dim), got {tuple(control.shape)}"
        )
        raise ValueError(msg)

    def _advance_block_diagonal(
        self,
        z: Tensor,
        delta_t: float | Tensor,
        *,
        control: Tensor | None,
    ) -> Tensor:
        """Self-dominated approximate advance via per-node ``exp(L_self Δt)``.

        Ignores neighbor factors and ``adjacency`` (documented self-only
        shortcut; exact when neighbor generators are zero).

        Parameters
        ----------
        z : Tensor
            Latent node states ``(num_nodes, latent_dim)``.
        delta_t : float or Tensor
            Integration interval.
        control : Tensor or None
            Optional control input.

        Returns
        -------
        Tensor
            Advanced latents with the same shape as ``z``.
        """
        return self._self.advance(z, delta_t, control=control)

    def forward(
        self,
        z: Tensor,
        delta_t: float | Tensor,
        edge_index: Tensor,
        edge_weight: Tensor | None = None,
        control: Tensor | None = None,
    ) -> Tensor:
        """Advance latents over ``Δt`` with topology-coupled continuous dynamics.

        Parameters
        ----------
        z : Tensor
            Latent node states ``(num_nodes, latent_dim)``.
        delta_t : float or Tensor
            Integration interval (required; ``0`` returns ``z``).
        edge_index : Tensor
            Edge index ``(2, E)``.
        edge_weight : Tensor or None, optional
            Optional edge weights.
        control : Tensor or None, optional
            Piecewise-constant control over ``[0, Δt]``.

        Returns
        -------
        Tensor
            Advanced latents with the same shape as ``z``.
        """
        if z.ndim != 2 or z.shape[-1] != self.latent_dim:
            msg = (
                "ContinuousGraphKoopmanOperator expects z with shape "
                f"(num_nodes, {self.latent_dim}), got {tuple(z.shape)}"
            )
            raise ValueError(msg)
        delta = torch.as_tensor(delta_t, dtype=z.dtype, device=z.device)
        if bool((delta == 0).all().item()):
            return z

        if self.sparsity == "block_diagonal":
            return self._advance_block_diagonal(z, delta, control=control)
        return self._advance_dense(
            z,
            delta,
            control=control,
            edge_index=edge_index,
            edge_weight=edge_weight,
        )

    def advance(
        self,
        z: Tensor,
        delta_t: float | Tensor | None = None,
        *,
        control: Tensor | None = None,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        """Contract advance; requires ``delta_t`` and ``edge_index``.

        Parameters
        ----------

        z : Tensor
            See the function signature / summary for ``z``.
        delta_t : float | Tensor | None
            See the function signature / summary for ``delta_t``.
        control : Tensor | None
            See the function signature / summary for ``control``.
        edge_index : Tensor | None
            See the function signature / summary for ``edge_index``.
        edge_weight : Tensor | None
            See the function signature / summary for ``edge_weight``.

        Returns
        -------

        Tensor
            See summary line.

        Raises
        ------

        ValueError
            Raised when inputs are invalid."""
        if delta_t is None:
            msg = "delta_t is required for ContinuousGraphKoopmanOperator.advance"
            raise ValueError(msg)
        if edge_index is None:
            msg = "edge_index is required for ContinuousGraphKoopmanOperator.advance"
            raise ValueError(msg)
        return self.forward(
            z, delta_t, edge_index, edge_weight=edge_weight, control=control
        )

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
        """Approximate inverse over ``-Δt`` (dense uses ``exp(-L_eff Δt)``).

        Parameters
        ----------

        z : Tensor
            See the function signature / summary for ``z``.
        delta_t : float | Tensor | None
            See the function signature / summary for ``delta_t``.
        control : Tensor | None
            See the function signature / summary for ``control``.
        inverse_matrix : Tensor | None
            See the function signature / summary for ``inverse_matrix``.
        edge_index : Tensor | None
            See the function signature / summary for ``edge_index``.
        edge_weight : Tensor | None
            See the function signature / summary for ``edge_weight``.

        Returns
        -------

        Tensor
            See summary line.

        Raises
        ------

        ValueError
            Raised when inputs are invalid.

        Notes
        -----

        ``sparsity="block_diagonal"`` inverts the self-term only. Controlled
        dense inverse uses Van Loan factors of ``L_eff``; ``inverse_matrix``
        is supported only for uncontrolled dense steps."""
        if delta_t is None:
            msg = (
                "delta_t is required for ContinuousGraphKoopmanOperator.inverse_advance"
            )
            raise ValueError(msg)
        if edge_index is None:
            msg = (
                "edge_index is required for "
                "ContinuousGraphKoopmanOperator.inverse_advance"
            )
            raise ValueError(msg)
        if z.ndim != 2 or z.shape[-1] != self.latent_dim:
            msg = (
                "ContinuousGraphKoopmanOperator.inverse_advance expects z with "
                f"shape (num_nodes, {self.latent_dim}), got {tuple(z.shape)}"
            )
            raise ValueError(msg)

        if self.sparsity == "block_diagonal":
            if inverse_matrix is not None:
                msg = (
                    "inverse_matrix is only supported for "
                    "ContinuousGraphKoopmanOperator sparsity='dense'"
                )
                raise ValueError(msg)
            return self._self.inverse_advance(z, delta_t, control=control)

        num_nodes = z.shape[0]
        delta = torch.as_tensor(delta_t, dtype=z.dtype, device=z.device)
        generator = self.effective_generator(
            edge_index, num_nodes, edge_weight=edge_weight
        )
        flat = z.reshape(1, -1)

        if self.control_dim == 0:
            if control is not None:
                msg = "control input provided to an uncontrolled operator"
                raise ValueError(msg)
            if inverse_matrix is None:
                inverse_matrix = torch.linalg.matrix_exp(generator * (-delta))
            recovered = flat @ inverse_matrix.T
            return recovered.view_as(z)

        if control is None:
            msg = "control input is required when control_dim > 0"
            raise ValueError(msg)
        if inverse_matrix is not None:
            msg = (
                "inverse_matrix is not supported for controlled "
                "ContinuousGraphKoopmanOperator.inverse_advance"
            )
            raise ValueError(msg)

        # Uncontrolled backward free step, then remove the forward Van Loan
        # control offset evaluated at +Δt (same pattern as continuous peers).
        b_eff = self._networked_control_matrix(num_nodes)
        phi11, phi12 = van_loan_factors(generator, b_eff, delta)
        if control.ndim == 1:
            offset = control @ phi12.T
            adjusted = flat - offset
        elif control.ndim == 2:
            # Approximate: subtract per-node self-term control offsets.
            adjusted = flat.clone()
            node_view = adjusted.view_as(z)
            for node_idx in range(num_nodes):
                _, node_phi12 = van_loan_factors(self.L_self, self.B, delta)
                offset = control[node_idx] @ node_phi12.T
                node_view[node_idx] = node_view[node_idx] - offset
            adjusted = node_view.reshape(1, -1)
        else:
            msg = (
                "control input must have shape (control_dim,) or "
                f"(num_nodes, control_dim), got {tuple(control.shape)}"
            )
            raise ValueError(msg)
        recovered = adjusted @ torch.linalg.inv(phi11).T
        return recovered.view_as(z)
