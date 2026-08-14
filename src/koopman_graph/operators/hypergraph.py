"""Hyperedge-coupled discrete Koopman operator."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from torch import Tensor, nn

from koopman_graph.graph_utils.symmetry import OrbitMethod
from koopman_graph.operators.contract import (
    InitMode,
    Parameterization,
    StabilityCertificate,
)
from koopman_graph.operators.control import (
    ControlMode,
    bilinear_state_control_term,
    broadcast_control_term,
    effective_bilinear_matrix,
    per_node_effective_bilinear_matrices,
)
from koopman_graph.operators.discrete import KoopmanOperator
from koopman_graph.operators.discrete_propagation import dense_inverse_or_pinv
from koopman_graph.operators.matrix_free import (
    flatten_node_latents,
    invert_k_eff_hypergraph,
    unflatten_node_latents,
)
from koopman_graph.operators.orbit_ties import OrbitTiedSelfMixin
from koopman_graph.spectrum_types import KoopmanSpectrum, compute_spectrum

HypergraphSparsity = Literal["dense", "block_diagonal", "distributed"]
HypergraphIncidenceMode = Literal[
    "zhou_symmetric", "forward_random_walk", "dual_random_walk"
]
HYPERGRAPH_INCIDENCE_MODES: frozenset[str] = frozenset(
    {"zhou_symmetric", "forward_random_walk", "dual_random_walk"}
)


class HypergraphKoopmanOperator(OrbitTiedSelfMixin, nn.Module):
    """Discrete Koopman step with self and hyperedge-mediated coupling.

    Advances stacked node latents ``Z ∈ R^{N×d}`` via a linear map whose
    incidence factor depends on ``incidence_mode``:

    * ``"zhou_symmetric"`` (default)::

          Z_next = Z @ K_self.T + (Ĥ Z) @ K_hedge.T

      with Zhou ``Ĥ = D_v^{-1/2} B W_e D_e^{-1} Bᵀ D_v^{-1/2}`` on undirected
      bipartite ``hyperedge_index``.

    * ``"forward_random_walk"``::

          Z_next = Z @ K_self.T + (P_fwd Z) @ K_hedge.T

      with Ducournau–Bretto ``P_fwd`` on directed ``tail_index`` /
      ``head_index``.

    * ``"dual_random_walk"``::

          Z_next = Z @ K_self.T + (P_fwd Z) @ K_hedge.T + (P_bwd Z) @ K_bwd.T

      (``K_hedge`` is the forward factor; ``K_bwd`` is the reverse factor).

    Encode / advance orientation
        The default encoder / decoder stacks
        (:class:`~koopman_graph.nn.HypergraphEncoder`,
        :class:`~torch_geometric.nn.HypergraphConv`) remain **undirected**:
        they consume bipartite ``hyperedge_index``. Random-walk
        ``incidence_mode`` values may still use directed ``tail_index`` /
        ``head_index`` on advance. Encode and advance therefore need not
        share an orientation; that asymmetry is intentional.

    Directed-mode scope
        ``forward_random_walk`` and ``dual_random_walk`` implement **one**
        documented directed-hypergraph random-walk normalization
        (Ducournau & Bretto, 2014), chosen and verified for this package.
        The literature admits other normalizations; these modes are not
        presented as the unique or canonical choice. They make **no** claim
        of equivalence to simplicial or Hodge Laplacians.

    Discrete-time only; continuous hypergraph generators are out of scope.
    Passing graph-style ``adjacency=...`` is rejected as an unexpected
    keyword — use ``incidence_mode`` instead.

    When ``K_hedge = 0`` (and ``K_bwd = 0`` in dual mode), the step reduces
    exactly to the per-node map ``Z @ K_self.T``. A 2-uniform undirected
    hypergraph under Zhou mode is related to
    :class:`~koopman_graph.operators.GraphKoopmanOperator` by
    ``Ĥ = ½(I + Â)`` (unweighted).

    Dense Zhou ``Ĥ`` for a static incidence may be reused across advances
    via :func:`~koopman_graph.graph_utils.clear_hyperedge_cache` /
    :meth:`clear_hyperedge_cache`. Caching does not remove the dense
    :math:`O(N^2)` representation of ``Ĥ``.

    Attributes
    ----------
    latent_dim : int
        Latent feature dimension ``d``.
    control_dim : int
        Exogenous control dimension (``0`` disables control).
    parameterization : Parameterization
        Shared soft/structural parameterization for ``K_self`` and
        hyperedge factors.
    incidence_mode : {"zhou_symmetric", "forward_random_walk", "dual_random_walk"}
        Incidence normalization (default ``"zhou_symmetric"``).
    sparsity : {"dense", "block_diagonal", "distributed"}
        Realization mode. ``"dense"`` and ``"block_diagonal"`` share the same
        forward hyperedge matvec; they differ in ``inverse_advance`` (exact
        ``N·d`` inverse vs approximate per-node Jacobi). ``"distributed"``
        is accepted for construction / checkpoints (not trainer DDP or
        multi-GPU training); matrix-free inverse and spectrum helpers are
        wired for discrete graph and hetero in 0.10 (hypergraph may still
        assemble for spectrum / inverse).
    max_spectral_radius : float
        Stability bound forwarded to the factorized self/hyperedge matrices.
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
        sparsity: HypergraphSparsity = "dense",
        incidence_mode: HypergraphIncidenceMode = "zhou_symmetric",
        orbit_partition: Sequence[Sequence[int]] | None = None,
        auto_orbits: bool = False,
        orbit_method: OrbitMethod = "auto",
    ) -> None:
        """Initialize self and hyperedge Koopman factors.

        Parameters
        ----------
        latent_dim : int
            Latent dimension ``d``.
        init_mode : {"identity", "identity_noise", "xavier"}, optional
            Initialization for ``K_self``. ``K_hedge`` starts at zero for
            ``identity`` / ``identity_noise`` (plus optional noise on the
            hyperedge term for ``identity_noise`` / ``xavier``).
            ``K_bwd`` (dual mode) initializes at exactly zero.
        init_scale : float, optional
            Noise scale for ``identity_noise`` / hyperedge jitter.
        parameterization : Parameterization, optional
            Shared parameterization for the ``d×d`` factors.
        max_spectral_radius : float, optional
            Spectral bound for soft/structural modes.
        control_dim : int, optional
            Additive / bilinear control dimension. Default ``0``.
        control_mode : {"additive", "bilinear"}, optional
            Control coupling forwarded to the self-term operator.
        bilinear_rank : int or None, optional
            Low-rank bilinear size when ``control_mode="bilinear"``.
        sparsity : {"dense", "block_diagonal", "distributed"}, optional
            ``"dense"`` (default) uses an exact dense ``inverse_advance``.
            ``"block_diagonal"`` keeps the same forward advance and uses an
            approximate per-node inverse. ``"distributed"`` is accepted for
            construction / checkpoints (matrix-free inverse / spectrum are
            wired for discrete graph and hetero in 0.10).
        incidence_mode : HypergraphIncidenceMode, optional
            Incidence normalization (``zhou_symmetric`` /
            ``forward_random_walk`` / ``dual_random_walk``). Default
            ``"zhou_symmetric"`` preserves historical undirected behavior.
        orbit_partition : sequence of sequence of int or None, optional
            Explicit node-orbit partition tying ``K_self`` and ``K_hedge``
            across orbit mates. Dual ``K_bwd`` stays globally shared.
        auto_orbits : bool, optional
            When ``True``, compute orbits from the hypergraph 2-section on
            first advance.
        orbit_method : {"auto", "exact"}, optional
            Orbit backend for ``auto_orbits``. Default ``"auto"``.

        Raises
        ------
        ValueError
            If ``sparsity`` / ``incidence_mode`` is unsupported or construction
            args are invalid.
        """
        super().__init__()
        if sparsity not in {"dense", "block_diagonal", "distributed"}:
            msg = (
                "HypergraphKoopmanOperator sparsity must be 'dense', "
                f"'block_diagonal', or 'distributed', got {sparsity!r}"
            )
            raise ValueError(msg)
        if incidence_mode not in HYPERGRAPH_INCIDENCE_MODES:
            accepted = ", ".join(sorted(HYPERGRAPH_INCIDENCE_MODES))
            msg = (
                "HypergraphKoopmanOperator incidence_mode must be one of "
                f"{{{accepted}}}, got {incidence_mode!r}"
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
        self.incidence_mode: HypergraphIncidenceMode = incidence_mode

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
        self._hedge = KoopmanOperator(
            latent_dim,
            init_mode="identity",
            init_scale=init_scale,
            parameterization=parameterization,
            max_spectral_radius=max_spectral_radius,
            control_dim=0,
        )
        self._bwd: KoopmanOperator | None
        if incidence_mode == "dual_random_walk":
            self._bwd = KoopmanOperator(
                latent_dim,
                init_mode="identity",
                init_scale=init_scale,
                parameterization=parameterization,
                max_spectral_radius=max_spectral_radius,
                control_dim=0,
            )
        else:
            self._bwd = None
        self._reset_hyperedge_parameters()
        self._init_orbit_config(
            orbit_partition=orbit_partition,
            auto_orbits=auto_orbits,
            orbit_method=orbit_method,
        )

    def _reset_factor_parameters(
        self,
        module: KoopmanOperator,
        *,
        allow_noise: bool,
    ) -> None:
        """Zero a hyperedge factor, optionally adding ``init_scale`` noise.

        Parameters
        ----------
        module
            See signature.
        allow_noise
            See signature."""
        if self.parameterization == "dense":
            dense_k = module.K
            with torch.no_grad():
                dense_k.zero_()
                if allow_noise and self.init_mode in {"identity_noise", "xavier"}:
                    dense_k.add_(torch.randn_like(dense_k) * self.init_scale)
            return
        with torch.no_grad():
            for parameter in module.parameters():
                parameter.zero_()
            if allow_noise and self.init_mode in {"identity_noise", "xavier"}:
                for parameter in module.parameters():
                    parameter.add_(torch.randn_like(parameter) * self.init_scale)

    def _reset_hyperedge_parameters(self) -> None:
        """Initialize hyperedge factors for a per-node-like starting point.

        Notes
        -----
        ``K_hedge`` may receive ``init_scale`` noise. ``K_bwd`` (dual mode)
        is always exactly zero so ``dual_random_walk`` begins equivalent to
        ``forward_random_walk``.
        """
        orbit_nbrs = getattr(self, "_orbit_nbrs", None)
        modules = list(orbit_nbrs) if orbit_nbrs is not None else [self._hedge]
        for module in modules:
            self._reset_factor_parameters(module, allow_noise=True)
        if self._bwd is not None:
            self._reset_factor_parameters(self._bwd, allow_noise=False)

    def reset_parameters(self) -> None:
        """Reinitialize ``K_self`` / hyperedge factors (and control ``B``).

        Notes
        -----
        Delegates to the self/hyperedge factor modules, then re-applies the
        near-zero hyperedge-factor initialization.
        """
        self.reset_orbit_selves()
        self._hedge.reset_parameters()
        if self._bwd is not None:
            self._bwd.reset_parameters()
        self._reset_hyperedge_parameters()

    def clear_hyperedge_cache(self) -> None:
        """Drop the shared dense Zhou ``Ĥ`` cache used by advance / eigen.

        Thin wrapper around
        :func:`~koopman_graph.graph_utils.clear_hyperedge_cache`. Call after
        in-place incidence or hyperedge-weight edits that keep the same
        storage pointers.

        Notes
        -----
        The cache is ephemeral and never written to ``state_dict``.
        """
        from koopman_graph.graph_utils import clear_hyperedge_cache

        clear_hyperedge_cache()

    @property
    def K_self(self) -> Tensor:
        """Self-coupling matrix with shape ``(latent_dim, latent_dim)``.

        When orbit-tied, returns the representative (orbit-0) self matrix.

        Returns
        -------
        Tensor
            Assembled ``K_self``.
        """
        return self._self.K

    @property
    def K_hedge(self) -> Tensor:
        """Hyperedge-coupling matrix with shape ``(latent_dim, latent_dim)``.

        When orbit-tied, returns the representative (orbit-0) hyperedge
        matrix; the per-orbit bank is applied after the incidence matvec.

        Returns
        -------
        Tensor
            Assembled ``K_hedge`` (forward factor under random-walk modes).
        """
        return self._hedge.K

    @property
    def K_bwd(self) -> Tensor:
        """Backward random-walk coupling (``dual_random_walk`` only).

        Returns
        -------
        Tensor
            Assembled ``K_bwd``.

        Raises
        ------
        RuntimeError
            If ``incidence_mode`` is not ``"dual_random_walk"``.
        """
        if self._bwd is None:
            msg = "K_bwd is only available when incidence_mode='dual_random_walk'"
            raise RuntimeError(msg)
        return self._bwd.K

    @property
    def matrix(self) -> Tensor:
        """Self-term matrix (contract surface; topology-coupled spectrum differs).

        Returns
        -------
        Tensor
            ``K_self``. This is the Protocol ``matrix`` for per-node API
            compatibility and is **not** the networked ``N·d`` operator.
            Use :meth:`effective_matrix` / :meth:`spectrum` (and
            :meth:`~koopman_graph.model.GraphKoopmanModel.spectrum` with
            topology) for the full topology-coupled spectrum.
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
        k_hedge: Tensor,
        *,
        k_bwd: Tensor | None = None,
        control_matrix: Tensor | None = None,
        bilinear_matrices: Tensor | None = None,
    ) -> None:
        """Write dense ``K_self`` / ``K_hedge`` (and optional dual / control).

        Parameters
        ----------
        k_self : Tensor
            Dense self matrix ``(latent_dim, latent_dim)``.
        k_hedge : Tensor
            Dense hyperedge / forward matrix ``(latent_dim, latent_dim)``.
        k_bwd : Tensor or None, optional
            Dense backward matrix when ``incidence_mode="dual_random_walk"``.
            Must be omitted otherwise.
        control_matrix : Tensor or None, optional
            Control matrix ``B`` when ``control_dim > 0``.
        bilinear_matrices : Tensor or None, optional
            Full-rank bilinear stack when ``control_mode="bilinear"``.
        """
        if k_bwd is not None and self._bwd is None:
            msg = "k_bwd is only valid when incidence_mode='dual_random_walk'"
            raise ValueError(msg)
        if self._orbit_selves is None:
            self._self.set_dense_matrix(
                k_self,
                control_matrix=control_matrix,
                bilinear_matrices=bilinear_matrices,
            )
        else:
            for orbit_id, module in enumerate(self._orbit_selves):
                module.set_dense_matrix(
                    k_self,
                    control_matrix=control_matrix if orbit_id == 0 else None,
                    bilinear_matrices=bilinear_matrices if orbit_id == 0 else None,
                )
        if self._orbit_nbrs is None:
            self._hedge.set_dense_matrix(k_hedge, control_matrix=None)
        else:
            for module in self._orbit_nbrs:
                module.set_dense_matrix(k_hedge, control_matrix=None)
        if k_bwd is not None:
            assert self._bwd is not None
            self._bwd.set_dense_matrix(k_bwd, control_matrix=None)

    def bound_metric(self) -> Tensor:
        """Return ``max`` of self / hyperedge factor bounds for monitoring.

        This is a **factor-level** soft/structural surrogate used by
        structural eigenvalue regularization. It is **not** the spectral
        radius of the topology-coupled effective operator and must not be
        treated as a whole-network stability certificate. For dense/ODO
        analysis and regularization of the networked map, use
        :meth:`effective_matrix` / :meth:`spectrum` (and the topology-aware
        eigenvalue loss path) instead.

        Returns
        -------
        Tensor
            Scalar factor bound metric.
        """
        metric = torch.maximum(self._self.bound_metric(), self._hedge.bound_metric())
        if self._bwd is not None:
            metric = torch.maximum(metric, self._bwd.bound_metric())
        return metric

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

    def _require_directed_incidence(
        self,
        *,
        tail_index: Tensor | None,
        head_index: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Validate directed incidence for random-walk modes.

        Parameters
        ----------
        tail_index
            See signature.
        head_index
            See signature.

        Returns
        -------
            See signature."""
        if tail_index is None or head_index is None:
            msg = (
                f"incidence_mode={self.incidence_mode!r} requires "
                "tail_index and head_index"
            )
            raise ValueError(msg)
        return tail_index, head_index

    def _orbit_incidence_for_binding(
        self,
        *,
        hyperedge_index: Tensor | None,
        tail_index: Tensor | None,
        head_index: Tensor | None,
    ) -> Tensor | None:
        """Choose bipartite incidence for orbit 2-section binding.

        Parameters
        ----------
        hyperedge_index
            See signature.
        tail_index
            See signature.
        head_index
            See signature.

        Returns
        -------
            See signature."""
        if self.incidence_mode == "zhou_symmetric":
            return hyperedge_index
        if tail_index is None or head_index is None:
            return hyperedge_index
        parts = [part for part in (tail_index, head_index) if part.numel() > 0]
        if not parts:
            device = tail_index.device
            return torch.zeros(2, 0, dtype=torch.long, device=device)
        return torch.cat(parts, dim=1)

    def _dense_coupling_adjacency(
        self,
        *,
        num_nodes: int,
        dtype: torch.dtype,
        hyperedge_index: Tensor | None,
        hyperedge_weight: Tensor | None,
        tail_index: Tensor | None,
        head_index: Tensor | None,
    ) -> tuple[Tensor, Tensor | None]:
        """Return ``(A_fwd, A_bwd|None)`` dense incidence operators.

        Parameters
        ----------
        num_nodes
            See signature.
        dtype
            See signature.
        hyperedge_index
            See signature.
        hyperedge_weight
            See signature.
        tail_index
            See signature.
        head_index
            See signature.

        Returns
        -------
            See signature."""
        from koopman_graph.graph_utils.topology import (
            dense_hyperedge_dual_random_walk_factors,
            dense_hyperedge_forward_random_walk_adjacency,
            dense_hyperedge_normalized_adjacency,
        )

        if self.incidence_mode == "zhou_symmetric":
            if hyperedge_index is None:
                msg = "hyperedge_index is required when incidence_mode='zhou_symmetric'"
                raise ValueError(msg)
            hat = dense_hyperedge_normalized_adjacency(
                hyperedge_index,
                num_nodes=num_nodes,
                hyperedge_weight=hyperedge_weight,
                dtype=dtype,
            )
            return hat, None
        tail, head = self._require_directed_incidence(
            tail_index=tail_index,
            head_index=head_index,
        )
        if self.incidence_mode == "forward_random_walk":
            forward = dense_hyperedge_forward_random_walk_adjacency(
                tail,
                head,
                num_nodes=num_nodes,
                hyperedge_weight=hyperedge_weight,
                dtype=dtype,
            )
            return forward, None
        forward, backward = dense_hyperedge_dual_random_walk_factors(
            tail,
            head,
            num_nodes=num_nodes,
            hyperedge_weight=hyperedge_weight,
            dtype=dtype,
        )
        return forward, backward

    def effective_matrix(
        self,
        hyperedge_index: Tensor | None = None,
        num_nodes: int | None = None,
        hyperedge_weight: Tensor | None = None,
        *,
        tail_index: Tensor | None = None,
        head_index: Tensor | None = None,
        k_self: Tensor | None = None,
        k_self_blocks: Tensor | None = None,
    ) -> Tensor:
        """Assemble the dense effective topology-coupled operator.

        Zhou mode builds ``I⊗K_self + Ĥ⊗K_hedge``. Random-walk modes use
        ``P_fwd`` / ``P_bwd`` with ``K_hedge`` / ``K_bwd``.

        Parameters
        ----------
        hyperedge_index : Tensor or None, optional
            Undirected bipartite incidence (Zhou mode).
        num_nodes : int or None, optional
            Number of nodes ``N``. Required.
        hyperedge_weight : Tensor or None, optional
            Optional hyperedge weights ``(M,)``.
        tail_index, head_index : Tensor or None, optional
            Directed incidence (random-walk modes).
        k_self : Tensor or None, optional
            Optional override for a **shared** self-coupling matrix (used when
            folding a global bilinear term into ``K_self`` for inversion).
        k_self_blocks : Tensor or None, optional
            Optional per-node self blocks with shape ``(N, d, d)`` (used when
            folding per-node bilinear terms). Mutually exclusive with
            ``k_self``.

        Returns
        -------
        Tensor
            Dense matrix with shape ``(N·d, N·d)``.

        Raises
        ------
        ValueError
            If both ``k_self`` and ``k_self_blocks`` are set, or if
            ``k_self_blocks`` has the wrong shape.
        """
        if num_nodes is None:
            msg = "num_nodes is required for effective_matrix"
            raise ValueError(msg)
        if k_self is not None and k_self_blocks is not None:
            msg = "Pass at most one of k_self and k_self_blocks"
            raise ValueError(msg)

        orbit_incidence = self._orbit_incidence_for_binding(
            hyperedge_index=hyperedge_index,
            tail_index=tail_index,
            head_index=head_index,
        )
        self.ensure_orbit_binding(num_nodes, hyperedge_index=orbit_incidence)
        if k_self_blocks is None and k_self is None:
            k_self_blocks = self.tied_self_blocks(num_nodes)
        self_matrix = self.K_self if k_self is None else k_self
        a_fwd, a_bwd = self._dense_coupling_adjacency(
            num_nodes=num_nodes,
            dtype=self_matrix.dtype,
            hyperedge_index=hyperedge_index,
            hyperedge_weight=hyperedge_weight,
            tail_index=tail_index,
            head_index=head_index,
        )
        coupling = torch.kron(a_fwd, self.K_hedge)
        if a_bwd is not None:
            coupling = coupling + torch.kron(a_bwd, self.K_bwd)
        if k_self_blocks is None:
            identity = torch.eye(num_nodes, dtype=a_fwd.dtype, device=a_fwd.device)
            return torch.kron(identity, self_matrix) + coupling

        expected = (num_nodes, self.latent_dim, self.latent_dim)
        if k_self_blocks.shape != expected:
            msg = (
                f"k_self_blocks must have shape {expected}, "
                f"got {tuple(k_self_blocks.shape)}"
            )
            raise ValueError(msg)
        self_blocks = torch.block_diag(*k_self_blocks.unbind(0))
        return self_blocks + coupling

    def dense_effective_inverse(
        self,
        hyperedge_index: Tensor | None = None,
        num_nodes: int | None = None,
        *,
        hyperedge_weight: Tensor | None = None,
        tail_index: Tensor | None = None,
        head_index: Tensor | None = None,
        k_self: Tensor | None = None,
        k_self_blocks: Tensor | None = None,
    ) -> Tensor:
        """Assemble and invert the dense effective hyperedge-coupled operator.

        Intended for evaluation-scoped reuse in backward consistency (static
        incidence, ``sparsity="dense"``). Pair-local bilinear overrides should
        be passed explicitly; otherwise default tied self blocks are used.

        Parameters
        ----------
        hyperedge_index : Tensor or None, optional
            Undirected bipartite incidence (Zhou mode).
        num_nodes : int or None, optional
            Number of nodes ``N``.
        hyperedge_weight : Tensor or None, optional
            Optional hyperedge weights ``(M,)``.
        tail_index, head_index : Tensor or None, optional
            Directed incidence (random-walk modes).
        k_self : Tensor or None, optional
            Optional shared self-coupling override (see
            :meth:`effective_matrix`).
        k_self_blocks : Tensor or None, optional
            Optional per-node self blocks (see :meth:`effective_matrix`).

        Returns
        -------
        Tensor
            Dense inverse (or pseudoinverse) with shape ``(N·d, N·d)``.

        Raises
        ------
        ValueError
            If ``sparsity`` is not ``"dense"``.
        """
        if self.sparsity != "dense":
            msg = "dense_effective_inverse requires sparsity='dense'"
            raise ValueError(msg)
        effective = self.effective_matrix(
            hyperedge_index,
            num_nodes,
            hyperedge_weight=hyperedge_weight,
            tail_index=tail_index,
            head_index=head_index,
            k_self=k_self,
            k_self_blocks=k_self_blocks,
        )
        return dense_inverse_or_pinv(effective)

    def spectrum(
        self,
        hyperedge_index: Tensor | None = None,
        num_nodes: int | None = None,
        *,
        hyperedge_weight: Tensor | None = None,
        tail_index: Tensor | None = None,
        head_index: Tensor | None = None,
        time_step: float = 1.0,
    ) -> KoopmanSpectrum:
        """Eigendecomposition of the effective ``N·d`` hyperedge-coupled operator.

        Parameters
        ----------
        hyperedge_index : Tensor or None, optional
            Undirected incidence (Zhou mode).
        num_nodes : int or None, optional
            Node count ``N``.
        hyperedge_weight : Tensor or None, optional
            Optional hyperedge weights.
        tail_index, head_index : Tensor or None, optional
            Directed incidence (random-walk modes).
        time_step : float, optional
            Discrete sampling interval for growth rates / frequencies.

        Returns
        -------
        KoopmanSpectrum
            Spectrum of :meth:`effective_matrix`.
        """
        return compute_spectrum(
            self.effective_matrix(
                hyperedge_index,
                num_nodes,
                hyperedge_weight=hyperedge_weight,
                tail_index=tail_index,
                head_index=head_index,
            ),
            time_step,
        )

    def forward(
        self,
        z: Tensor,
        hyperedge_index: Tensor | None = None,
        hyperedge_weight: Tensor | None = None,
        control: Tensor | None = None,
        *,
        tail_index: Tensor | None = None,
        head_index: Tensor | None = None,
    ) -> Tensor:
        """Advance latents with hyperedge-coupled linear message passing.

        Parameters
        ----------
        z : Tensor
            Latent node states with shape ``(num_nodes, latent_dim)``.
        hyperedge_index : Tensor or None, optional
            Undirected bipartite incidence (Zhou mode).
        hyperedge_weight : Tensor or None, optional
            Optional hyperedge weights.
        control : Tensor or None, optional
            Exogenous control when ``control_dim > 0``.
        tail_index, head_index : Tensor or None, optional
            Directed incidence (random-walk modes).

        Returns
        -------
        Tensor
            Advanced latents with the same shape as ``z``.
        """
        if z.ndim != 2:
            msg = (
                "HypergraphKoopmanOperator expects z with shape "
                f"(num_nodes, latent_dim), got {tuple(z.shape)}"
            )
            raise ValueError(msg)
        if z.shape[-1] != self.latent_dim:
            msg = (
                f"Expected trailing dimension {self.latent_dim}, "
                f"got shape {tuple(z.shape)}"
            )
            raise ValueError(msg)

        from koopman_graph.graph_utils.topology import (
            hyperedge_forward_random_walk_matvec,
            hyperedge_normalized_adjacency_matvec,
        )

        orbit_incidence = self._orbit_incidence_for_binding(
            hyperedge_index=hyperedge_index,
            tail_index=tail_index,
            head_index=head_index,
        )
        self.ensure_orbit_binding(z.shape[0], hyperedge_index=orbit_incidence)

        if self.incidence_mode == "zhou_symmetric":
            if hyperedge_index is None:
                msg = "hyperedge_index is required when incidence_mode='zhou_symmetric'"
                raise ValueError(msg)
            coupled = hyperedge_normalized_adjacency_matvec(
                hyperedge_index,
                z,
                hyperedge_weight=hyperedge_weight,
                num_nodes=z.shape[0],
            )
            z_next = self.apply_tied_self(z) + self.apply_tied_neighbor(coupled)
        else:
            tail, head = self._require_directed_incidence(
                tail_index=tail_index,
                head_index=head_index,
            )
            coupled_fwd = hyperedge_forward_random_walk_matvec(
                tail,
                head,
                z,
                hyperedge_weight=hyperedge_weight,
                num_nodes=z.shape[0],
            )
            z_next = self.apply_tied_self(z) + self.apply_tied_neighbor(coupled_fwd)
            if self.incidence_mode == "dual_random_walk":
                coupled_bwd = hyperedge_forward_random_walk_matvec(
                    head,
                    tail,
                    z,
                    hyperedge_weight=hyperedge_weight,
                    num_nodes=z.shape[0],
                )
                z_next = z_next + coupled_bwd @ self.K_bwd.T

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
            offset = broadcast_control_term(z, offset, latent_dim=self.latent_dim)
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
        hyperedge_index: Tensor | None = None,
        hyperedge_weight: Tensor | None = None,
        tail_index: Tensor | None = None,
        head_index: Tensor | None = None,
    ) -> Tensor:
        """Contract advance with mode-dependent incidence requirements.

        Pairwise ``edge_index`` / ``edge_weight`` are accepted for call-site
        symmetry with graph operators but are ignored.

        Parameters
        ----------
        z : Tensor
            Latent states ``(num_nodes, latent_dim)``.
        delta_t : float, Tensor, or None, optional
            Ignored (discrete operator).
        control : Tensor or None, optional
            Optional control input.
        edge_index, edge_weight : Tensor or None, optional
            Ignored pairwise topology (API symmetry).
        hyperedge_index : Tensor or None, optional
            Required for ``incidence_mode="zhou_symmetric"``.
        hyperedge_weight : Tensor or None, optional
            Optional hyperedge weights.
        tail_index, head_index : Tensor or None, optional
            Required for random-walk incidence modes.

        Returns
        -------
        Tensor
            Advanced latent states.
        """
        _ = delta_t, edge_index, edge_weight
        return self.forward(
            z,
            hyperedge_index,
            hyperedge_weight,
            control=control,
            tail_index=tail_index,
            head_index=head_index,
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
        hyperedge_index: Tensor | None = None,
        hyperedge_weight: Tensor | None = None,
        tail_index: Tensor | None = None,
        head_index: Tensor | None = None,
    ) -> Tensor:
        """Recover previous latents from a hyperedge-coupled forward step.

        ``sparsity="dense"`` inverts the effective ``N·d`` map (exact;
        suitable for modest ``N``). ``sparsity="block_diagonal"`` uses a
        one-step Jacobi / per-node ``d×d`` approximate inverse (Zhou mode
        only; exact when ``K_hedge = 0`` or there are no hyperedges).
        ``inverse_matrix`` is supported only for ``sparsity="dense"``.

        For ``control_mode="bilinear"``, global controls fold into a shared
        ``K_self`` override; per-node controls use node-specific bilinear self
        blocks plus the same coupling as forward advance.
        Singular dense maps fall back to a pseudoinverse.

        Parameters
        ----------
        z : Tensor
            Latents at ``t+1`` with shape ``(num_nodes, latent_dim)``.
        delta_t : float, Tensor, or None, optional
            Ignored.
        control : Tensor or None, optional
            Control that drove the forward step (global ``(C,)`` or per-node
            ``(N, C)``).
        inverse_matrix : Tensor or None, optional
            Optional precomputed effective inverse (``dense`` only).
        edge_index, edge_weight : Tensor or None, optional
            Ignored pairwise topology (API symmetry).
        hyperedge_index : Tensor or None, optional
            Undirected bipartite incidence (Zhou mode).
        hyperedge_weight : Tensor or None, optional
            Optional hyperedge weights.
        tail_index, head_index : Tensor or None, optional
            Directed incidence (random-walk modes).

        Returns
        -------
        Tensor
            Recovered latents at ``t``.

        Raises
        ------
        ValueError
            If topology / shapes are invalid, or ``inverse_matrix`` is passed
            with ``sparsity="block_diagonal"``.
        """
        from koopman_graph.operators.graph_inverse import (
            block_diagonal_hypergraph_inverse_advance,
        )

        _ = delta_t, edge_index, edge_weight
        if z.ndim != 2 or z.shape[-1] != self.latent_dim:
            msg = (
                "HypergraphKoopmanOperator.inverse_advance expects z with shape "
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
                offset = broadcast_control_term(z, offset, latent_dim=self.latent_dim)
            adjusted = z - offset

        num_nodes = z.shape[0]

        def _bilinear_self_factors() -> tuple[Tensor | None, Tensor | None]:
            """Resolve shared / per-node bilinear self overrides.

            Returns
            -------
            tuple[Tensor | None, Tensor | None]
                Shared override and optional per-node blocks.

            Raises
            ------
            ValueError
                Raised when control shape is invalid.
            """
            if self.control_mode != "bilinear":
                return None, None
            if control is None:
                msg = "control input is required when control_dim > 0"
                raise ValueError(msg)
            coupling = self._self.bilinear_matrices()
            if control.ndim == 1:
                return (
                    effective_bilinear_matrix(self.K_self, control, coupling),
                    None,
                )
            if control.ndim == 2:
                if control.shape[0] != num_nodes:
                    msg = (
                        f"Per-node control has {control.shape[0]} rows, "
                        f"expected {num_nodes}"
                    )
                    raise ValueError(msg)
                return (
                    None,
                    per_node_effective_bilinear_matrices(
                        self.K_self,
                        control,
                        coupling,
                    ),
                )
            msg = (
                "control input must have shape (control_dim,) for "
                "global control or (num_nodes, control_dim) for "
                f"per-node control, got {tuple(control.shape)}"
            )
            raise ValueError(msg)

        if self.sparsity == "block_diagonal":
            if self.incidence_mode != "zhou_symmetric":
                msg = (
                    "block_diagonal inverse_advance supports only "
                    "incidence_mode='zhou_symmetric'; use sparsity='dense' "
                    f"for {self.incidence_mode!r}"
                )
                raise ValueError(msg)
            if hyperedge_index is None:
                msg = (
                    "hyperedge_index is required for "
                    "HypergraphKoopmanOperator.inverse_advance"
                )
                raise ValueError(msg)
            if inverse_matrix is not None:
                msg = (
                    "inverse_matrix is only supported for "
                    "HypergraphKoopmanOperator sparsity='dense'"
                )
                raise ValueError(msg)
            k_self_override, k_self_blocks = _bilinear_self_factors()
            if k_self_blocks is None and k_self_override is None:
                k_self_blocks = self.tied_self_blocks(num_nodes)
            return block_diagonal_hypergraph_inverse_advance(
                adjusted,
                k_self=(
                    k_self_override if k_self_override is not None else self.K_self
                ),
                k_hedge=self.K_hedge,
                hyperedge_index=hyperedge_index,
                hyperedge_weight=hyperedge_weight,
                k_self_blocks=k_self_blocks,
            )

        if self.sparsity == "distributed" and inverse_matrix is not None:
            msg = (
                "inverse_matrix is only supported for "
                "HypergraphKoopmanOperator sparsity='dense'"
            )
            raise ValueError(msg)

        if inverse_matrix is None:
            k_self_override, k_self_blocks = _bilinear_self_factors()
            if k_self_blocks is None and k_self_override is None:
                k_self_blocks = self.tied_self_blocks(num_nodes)
            if self.sparsity == "distributed":
                if k_self_blocks is not None:
                    msg = (
                        "HypergraphKoopmanOperator sparsity='distributed' "
                        "inverse requires a shared K_self (orbit ties / "
                        "per-node bilinear self blocks are unsupported)"
                    )
                    raise ValueError(msg)
                if self.incidence_mode != "zhou_symmetric":
                    msg = (
                        "matrix-free hypergraph inverse supports "
                        "incidence_mode='zhou_symmetric' only; use "
                        "sparsity='dense' for directed incidence"
                    )
                    raise ValueError(msg)
                if hyperedge_index is None:
                    msg = (
                        "hyperedge_index is required for "
                        "HypergraphKoopmanOperator sparsity='distributed' inverse"
                    )
                    raise ValueError(msg)
                result = invert_k_eff_hypergraph(
                    flatten_node_latents(adjusted),
                    k_self=(
                        k_self_override if k_self_override is not None else self.K_self
                    ),
                    k_hedge=self.K_hedge,
                    hyperedge_index=hyperedge_index,
                    num_nodes=num_nodes,
                    hyperedge_weight=hyperedge_weight,
                )
                return unflatten_node_latents(
                    result.solution,
                    num_nodes=num_nodes,
                    latent_dim=self.latent_dim,
                )
            inverse_matrix = self.dense_effective_inverse(
                hyperedge_index,
                num_nodes,
                hyperedge_weight=hyperedge_weight,
                tail_index=tail_index,
                head_index=head_index,
                k_self=k_self_override,
                k_self_blocks=k_self_blocks,
            )

        flat = adjusted.reshape(-1)
        recovered = (inverse_matrix @ flat).view_as(adjusted)
        return recovered
