"""Networked (spatially-coupled) discrete Koopman operator."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from koopman_graph.graph_utils.symmetry import OrbitMethod
from koopman_graph.operators.contract import (
    InitMode,
    Parameterization,
    StabilityCertificate,
)

if TYPE_CHECKING:
    from koopman_graph.operators.joint_stability import JointStabilityCertificate
from koopman_graph.operators.control import (
    ControlMode,
    bilinear_state_control_term,
    broadcast_control_term,
    effective_bilinear_matrix,
    per_node_effective_bilinear_matrices,
)
from koopman_graph.operators.discrete import KoopmanOperator
from koopman_graph.operators.discrete_propagation import dense_inverse_or_pinv
from koopman_graph.operators.graph_types import (
    GRAPH_ADJACENCY_MODES,
    GraphAdjacency,
    GraphSparsity,
)
from koopman_graph.operators.matrix_free import (
    DEFAULT_DISTRIBUTED_SPECTRUM_NUM_MODES,
    flatten_node_latents,
    invert_k_eff_graph,
    spectrum_k_eff_graph,
    unflatten_node_latents,
)
from koopman_graph.operators.orbit_ties import OrbitTiedSelfMixin
from koopman_graph.spectrum_types import KoopmanSpectrum, compute_spectrum

__all__ = [
    "GraphAdjacency",
    "GraphKoopmanOperator",
    "GraphSparsity",
]


def _koopman_spectrum_from_eigenvalues(
    eigenvalues: Tensor,
    time_step: float,
) -> KoopmanSpectrum:
    """Build an eigenvalue-focused :class:`KoopmanSpectrum`.

    Parameters
    ----------
    eigenvalues
        Value for ``eigenvalues``.
    time_step
        Value for ``time_step``.

    Returns
    -------
    object
        Function result.
    """
    if time_step <= 0:
        msg = f"time_step must be positive, got {time_step}"
        raise ValueError(msg)
    magnitudes = eigenvalues.abs()
    growth_rates = torch.log(magnitudes.clamp_min(1e-30)) / time_step
    frequencies = torch.angle(eigenvalues) / (2 * torch.pi * time_step)
    num_modes = int(eigenvalues.numel())
    eigenvectors = torch.eye(
        num_modes,
        dtype=eigenvalues.dtype,
        device=eigenvalues.device,
    )
    return KoopmanSpectrum(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        magnitudes=magnitudes,
        growth_rates=growth_rates,
        frequencies=frequencies,
        time_step=float(time_step),
    )


class GraphKoopmanOperator(OrbitTiedSelfMixin, nn.Module):
    """Discrete Koopman step with self and neighbor coupling on the graph.

    Advances stacked node latents ``Z ∈ R^{N×d}`` via a linear map whose
    adjacency normalization depends on ``adjacency``:

    * ``"symmetric"`` (default)::

          vec(Z_{t+1}) = (I_N ⊗ K_self + Â_sym ⊗ K_nbr) vec(Z_t)
          Z_next = Z @ K_self.T + (Â_sym Z) @ K_nbr.T

      with ``Â_sym = D^{-1/2} A D^{-1/2}`` (undirected / symmetrically
      represented graphs).

    * ``"random_walk"``::

          Z_next = Z @ K_self.T + (Â_f Z) @ K_nbr.T

      with ``Â_f = D_out^{-1} A``.

    * ``"dual_random_walk"``::

          Z_next = Z @ K_self.T + (Â_f Z) @ K_fwd.T + (Â_b Z) @ K_bwd.T

      with ``Â_b = D_in^{-1} A^{\\top}``. ``K_fwd`` is an alias of ``K_nbr``.

    Unlike :class:`KoopmanOperator`, topology enters the **linear** step, so
    mid-horizon rewiring changes latent advance (not only decode).
    Discrete-time only; for continuous networked generators use
    :class:`~koopman_graph.operators.continuous_graph.ContinuousGraphKoopmanOperator`
    (``koopman="graph"`` + ``dynamics_mode="continuous"``).

    When neighbor factors are zero, the step reduces exactly to the per-node
    map ``Z @ K_self.T``. Orbit ties (when enabled) apply only to ``K_self``;
    ``K_nbr`` / ``K_fwd`` / ``K_bwd`` stay globally shared.

    Attributes
    ----------
    latent_dim : int
        Latent feature dimension ``d``.
    control_dim : int
        Exogenous control dimension (``0`` disables control).
    parameterization : Parameterization
        Shared soft/structural parameterization for ``K_self`` and neighbor
        factors.
    adjacency : {"symmetric", "random_walk", "dual_random_walk"}
        Neighbor-coupling normalization (default ``"symmetric"``).
    sparsity : {"dense", "block_diagonal", "distributed"}
        Realization mode. ``"dense"`` and ``"block_diagonal"`` share the same
        sparse forward matvec; they differ in ``inverse_advance`` (exact
        ``N·d`` inverse vs approximate per-node Jacobi). ``"distributed"``
        keeps the sparse forward path and uses matrix-free
        Richardson / Arnoldi inverse and spectrum helpers (not trainer DDP
        or multi-GPU training).
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
        adjacency: GraphAdjacency = "symmetric",
        orbit_partition: Sequence[Sequence[int]] | None = None,
        auto_orbits: bool = False,
        orbit_method: OrbitMethod = "auto",
        isotypic_symmetry: bool = False,
    ) -> None:
        """Initialize self and neighbor Koopman factors.

        Parameters
        ----------
        latent_dim : int
            Latent dimension ``d``.
        init_mode : {"identity", "identity_noise", "xavier"}, optional
            Initialization for ``K_self``. ``K_nbr`` / ``K_fwd`` starts near
            zero (plus optional noise for ``identity_noise`` / ``xavier``).
            ``K_bwd`` (dual mode only) initializes at exactly zero.
        init_scale : float, optional
            Noise scale for ``identity_noise`` / neighbor jitter.
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
            approximate per-node inverse. ``"distributed"`` uses matrix-free
            Richardson inverse and Arnoldi spectrum (not trainer DDP or
            multi-GPU training).
        adjacency : {"symmetric", "random_walk", "dual_random_walk"}, optional
            Neighbor-coupling normalization. Default ``"symmetric"`` preserves
            historical undirected behavior bit-for-bit.
        orbit_partition : sequence of sequence of int or None, optional
            Explicit node-orbit partition tying ``K_self`` across orbit mates.
            Overrides ``auto_orbits`` when provided. Neighbor factors are never
            orbit-tied.
        auto_orbits : bool, optional
            When ``True``, compute orbits from ``edge_index`` on first advance
            (requires the ``[symmetry]`` extra for non-identity partitions).
        orbit_method : {"auto", "exact"}, optional
            Orbit backend for ``auto_orbits``. Default ``"auto"``.
        isotypic_symmetry : bool, optional
            When ``True``, bind exact ``Aut(G)`` orbits for ``K_self`` ties
            and store the isotypic decomposition (factory
            ``koopman_symmetry="isotypic"``). Mutually exclusive with
            ``auto_orbits`` / ``orbit_partition``. Default ``False``.

        Raises
        ------
        ValueError
            If ``sparsity`` / ``adjacency`` are unsupported, or construction
            args are invalid.
        """
        super().__init__()
        if sparsity not in {"dense", "block_diagonal", "distributed"}:
            msg = (
                "GraphKoopmanOperator sparsity must be 'dense', "
                f"'block_diagonal', or 'distributed', got {sparsity!r}"
            )
            raise ValueError(msg)
        if adjacency not in GRAPH_ADJACENCY_MODES:
            accepted = ", ".join(sorted(GRAPH_ADJACENCY_MODES))
            msg = (
                "GraphKoopmanOperator adjacency must be one of "
                f"{{{accepted}}}, got {adjacency!r}"
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
        self.adjacency = adjacency

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
        self._bwd: KoopmanOperator | None
        if adjacency == "dual_random_walk":
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
        self._reset_neighbor_parameters()
        self._init_orbit_config(
            orbit_partition=orbit_partition,
            auto_orbits=auto_orbits,
            orbit_method=orbit_method,
            isotypic_symmetry=isotypic_symmetry,
        )

    def _reset_factor_parameters(
        self,
        module: KoopmanOperator,
        *,
        allow_noise: bool,
    ) -> None:
        """Zero a neighbor factor, optionally adding ``init_scale`` noise.

        Parameters
        ----------
        module : KoopmanOperator
            Neighbor factor module to reset.
        allow_noise : bool
            When ``True`` and ``init_mode`` is noisy, add ``init_scale`` jitter.
        """
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

    def _reset_neighbor_parameters(self) -> None:
        """Initialize neighbor factors for a per-node-like starting point.

        Notes
        -----
        ``K_nbr`` / ``K_fwd`` may receive ``init_scale`` noise.
        ``K_bwd`` (dual mode) is always exactly zero so
        ``dual_random_walk`` begins equivalent to ``random_walk``.
        """
        self._reset_factor_parameters(self._nbr, allow_noise=True)
        if self._bwd is not None:
            self._reset_factor_parameters(self._bwd, allow_noise=False)

    def reset_parameters(self) -> None:
        """Reinitialize ``K_self`` / neighbor factors (and control ``B``).

        Notes
        -----
        Delegates to the self/neighbor factor modules, then re-applies the
        neighbor initialization conventions (near-zero forward; exact-zero
        backward).
        """
        self.reset_orbit_selves()
        self._nbr.reset_parameters()
        if self._bwd is not None:
            self._bwd.reset_parameters()
        self._reset_neighbor_parameters()

    @property
    def K_self(self) -> Tensor:
        """Self-coupling matrix with shape ``(latent_dim, latent_dim)``.

        When orbit-tied, returns the representative (orbit-0) self matrix; use
        :meth:`tied_self_blocks` / :meth:`effective_matrix` for the full map.

        Returns
        -------
        Tensor
            Assembled ``K_self``.
        """
        return self._self.K

    @property
    def K_nbr(self) -> Tensor:
        """Forward / sole neighbor-coupling matrix ``(latent_dim, latent_dim)``.

        Returns
        -------
        Tensor
            Assembled ``K_nbr`` (alias :attr:`K_fwd`).
        """
        return self._nbr.K

    @property
    def K_fwd(self) -> Tensor:
        """Alias of :attr:`K_nbr` (forward random-walk coupling).

        Returns
        -------
        Tensor
            Assembled forward neighbor factor.
        """
        return self.K_nbr

    @property
    def K_bwd(self) -> Tensor:
        """Backward random-walk coupling (``dual_random_walk`` only).

        Returns
        -------
        Tensor
            Assembled ``K_bwd``.

        Raises
        ------
        AttributeError
            If ``adjacency`` is not ``"dual_random_walk"``.
        """
        if self._bwd is None:
            msg = "K_bwd is only available when adjacency='dual_random_walk'"
            raise AttributeError(msg)
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
        k_nbr: Tensor,
        *,
        k_bwd: Tensor | None = None,
        control_matrix: Tensor | None = None,
        bilinear_matrices: Tensor | None = None,
    ) -> None:
        """Write dense ``K_self`` / neighbor factors (and optional control).

        Parameters
        ----------
        k_self : Tensor
            Dense self matrix ``(latent_dim, latent_dim)``.
        k_nbr : Tensor
            Dense forward / sole neighbor matrix ``(latent_dim, latent_dim)``.
        k_bwd : Tensor or None, optional
            Dense backward neighbor matrix when
            ``adjacency="dual_random_walk"``. Must be omitted otherwise.
        control_matrix : Tensor or None, optional
            Control matrix ``B`` when ``control_dim > 0``.
        bilinear_matrices : Tensor or None, optional
            Full-rank bilinear stack when ``control_mode="bilinear"``.

        Raises
        ------
        ValueError
            If ``k_bwd`` is set when ``adjacency`` is not
            ``"dual_random_walk"``. When dual, ``k_bwd=None`` leaves
            ``K_bwd`` unchanged.
        """
        if k_bwd is not None and self._bwd is None:
            msg = "k_bwd is only valid when adjacency='dual_random_walk'"
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
        self._nbr.set_dense_matrix(k_nbr, control_matrix=None)
        if k_bwd is not None:
            assert self._bwd is not None
            self._bwd.set_dense_matrix(k_bwd, control_matrix=None)

    def bound_metric(self) -> Tensor:
        """Return ``max`` of self / neighbor factor bounds for monitoring.

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
        metric = torch.maximum(self._self.bound_metric(), self._nbr.bound_metric())
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

    def joint_bound_metric(
        self,
        edge_index: Tensor,
        num_nodes: int,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        """Return a Gershgorin upper bound on assembled ``ρ(K_eff)``.

        Assembles :meth:`effective_matrix` and applies
        :func:`~koopman_graph.operators.joint_stability.gershgorin_radius_bound`.
        The bound is **sufficient, not tight** (DESIGN R4): ``ρ(K_eff) ≤``
        the returned value. Distinct from :meth:`bound_metric` (factor-level)
        and from :meth:`spectral_radius` (self-factor only on this class).

        Parameters
        ----------
        edge_index : Tensor
            Edge index ``(2, E)``.
        num_nodes : int
            Number of nodes ``N``.
        edge_weight : Tensor or None, optional
            Optional edge weights ``(E,)``.

        Returns
        -------
        Tensor
            Scalar Gershgorin upper bound on ``ρ(K_eff)``.
        """
        from koopman_graph.operators.joint_stability import gershgorin_radius_bound

        effective = self.effective_matrix(
            edge_index,
            num_nodes,
            edge_weight=edge_weight,
        )
        return gershgorin_radius_bound(effective)

    def factor_stability_certificate(self) -> StabilityCertificate | None:
        """Return a **factor-level** self-term certificate, if any.

        Structural modes on ``K_self`` never certify joint ``ρ(K_eff)`` —
        use :meth:`stability_certificate` with topology for the Gershgorin
        joint bound object.

        Returns
        -------
        StabilityCertificate or None
            Certificate from the self-coupling factor, if any.
        """
        return self._self.stability_certificate()

    def stability_certificate(
        self,
        edge_index: Tensor,
        num_nodes: int,
        edge_weight: Tensor | None = None,
        *,
        kind: str = "gershgorin",
    ) -> JointStabilityCertificate:
        """Return a joint bound / certificate object for assembled ``K_eff``.

        Topology is required. Default ``kind="gershgorin"`` is a
        **sufficient** upper bound on ``ρ(K_eff)``, not a tight certificate
        and not soft assembled eigenvalue regularization. Opt-in
        ``kind="schur"`` / ``"lyapunov"`` (TASK-1824) use the assembled
        spectrum (and a discrete Lyapunov ``P`` when ``ρ < 1``) under size
        ceilings. For factor-level margins use
        :meth:`factor_stability_certificate`. Distinct from
        :meth:`spectral_radius`, which reports ``ρ(K_self)`` only on this
        class.

        Parameters
        ----------
        edge_index : Tensor
            Edge index ``(2, E)``.
        num_nodes : int
            Number of nodes ``N``.
        edge_weight : Tensor or None, optional
            Optional edge weights ``(E,)``.
        kind : {"gershgorin", "schur", "lyapunov"}, optional
            Joint certificate construction. Default ``"gershgorin"``.

        Returns
        -------
        JointStabilityCertificate
            Joint bound / unit-disk margin for ``K_eff``.

        Raises
        ------
        ValueError
            If ``kind`` is unsupported or an assembled Schur / Lyapunov
            size ceiling is exceeded.
        """
        from koopman_graph.operators.joint_stability import (
            JOINT_BOUND_KINDS,
            joint_certificate_from_assembled,
        )

        if kind not in JOINT_BOUND_KINDS:
            msg = f"kind must be one of {sorted(JOINT_BOUND_KINDS)}, got {kind!r}"
            raise ValueError(msg)
        if kind == "gershgorin":
            from koopman_graph.operators.joint_stability import (
                build_joint_stability_certificate,
            )

            bound = self.joint_bound_metric(
                edge_index,
                num_nodes,
                edge_weight=edge_weight,
            )
            return build_joint_stability_certificate(bound)
        effective = self.effective_matrix(
            edge_index,
            num_nodes,
            edge_weight=edge_weight,
        )
        return joint_certificate_from_assembled(effective, kind=kind)  # type: ignore[arg-type]

    def _dense_neighbor_coupling(
        self,
        edge_index: Tensor,
        num_nodes: int,
        *,
        edge_weight: Tensor | None,
        dtype: torch.dtype,
    ) -> Tensor:
        """Assemble ``Â ⊗ K_nbr`` (plus ``Â_b ⊗ K_bwd`` in dual mode).

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
            return torch.kron(adj, self.K_nbr)

        adj_fwd = dense_random_walk_normalized_adjacency(
            edge_index,
            num_nodes,
            edge_weight=edge_weight,
            dtype=dtype,
            direction="forward",
        )
        coupling = torch.kron(adj_fwd, self.K_nbr)
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
        return coupling + torch.kron(adj_bwd, self.K_bwd)

    def _sparse_neighbor_term(
        self,
        z: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor | None,
    ) -> Tensor:
        """Apply sparse neighbor message passing for the configured mode.

        Parameters
        ----------
        z : Tensor
            Latent node states ``(N, d)``.
        edge_index : Tensor
            Edge index ``(2, E)``.
        edge_weight : Tensor or None
            Optional edge weights.

        Returns
        -------
        Tensor
            Neighbor contribution with the same shape as ``z``.
        """
        from koopman_graph.graph_utils.topology import (
            random_walk_normalized_adjacency_matvec,
            symmetric_normalized_adjacency_matvec,
        )

        if self.adjacency == "symmetric":
            neighbor = symmetric_normalized_adjacency_matvec(
                edge_index,
                z,
                edge_weight=edge_weight,
                num_nodes=z.shape[0],
            )
            return neighbor @ self.K_nbr.T

        neighbor_fwd = random_walk_normalized_adjacency_matvec(
            edge_index,
            z,
            edge_weight=edge_weight,
            num_nodes=z.shape[0],
            direction="forward",
        )
        term = neighbor_fwd @ self.K_nbr.T
        if self.adjacency == "random_walk":
            return term
        neighbor_bwd = random_walk_normalized_adjacency_matvec(
            edge_index,
            z,
            edge_weight=edge_weight,
            num_nodes=z.shape[0],
            direction="backward",
        )
        return term + neighbor_bwd @ self.K_bwd.T

    def effective_matrix(
        self,
        edge_index: Tensor,
        num_nodes: int,
        edge_weight: Tensor | None = None,
        *,
        k_self: Tensor | None = None,
        k_self_blocks: Tensor | None = None,
    ) -> Tensor:
        """Assemble the dense effective networked operator ``(N·d, N·d)``.

        Parameters
        ----------
        edge_index : Tensor
            Edge index ``(2, E)``.
        num_nodes : int
            Number of nodes ``N``.
        edge_weight : Tensor or None, optional
            Optional edge weights ``(E,)``.
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
        if k_self is not None and k_self_blocks is not None:
            msg = "Pass at most one of k_self and k_self_blocks"
            raise ValueError(msg)

        self.ensure_orbit_binding(num_nodes, edge_index=edge_index)
        if k_self_blocks is None and k_self is None:
            k_self_blocks = self.tied_self_blocks(num_nodes)
        self_matrix = self.K_self if k_self is None else k_self
        neighbor = self._dense_neighbor_coupling(
            edge_index,
            num_nodes,
            edge_weight=edge_weight,
            dtype=self_matrix.dtype,
        )
        if k_self_blocks is None:
            identity = torch.eye(
                num_nodes,
                dtype=neighbor.dtype,
                device=neighbor.device,
            )
            return torch.kron(identity, self_matrix) + neighbor

        expected = (num_nodes, self.latent_dim, self.latent_dim)
        if k_self_blocks.shape != expected:
            msg = (
                f"k_self_blocks must have shape {expected}, "
                f"got {tuple(k_self_blocks.shape)}"
            )
            raise ValueError(msg)
        self_blocks = torch.block_diag(*k_self_blocks.unbind(0))
        return self_blocks + neighbor

    def dense_effective_inverse(
        self,
        edge_index: Tensor,
        num_nodes: int,
        *,
        edge_weight: Tensor | None = None,
        k_self: Tensor | None = None,
        k_self_blocks: Tensor | None = None,
    ) -> Tensor:
        """Assemble and invert the dense effective networked operator.

        Intended for evaluation-scoped reuse in backward consistency (static
        topology, ``sparsity="dense"``). Pair-local bilinear overrides should
        be passed explicitly; otherwise default tied self blocks are used.

        Parameters
        ----------
        edge_index : Tensor
            Edge index ``(2, E)``.
        num_nodes : int
            Number of nodes ``N``.
        edge_weight : Tensor or None, optional
            Optional edge weights ``(E,)``.
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
            edge_index,
            num_nodes,
            edge_weight=edge_weight,
            k_self=k_self,
            k_self_blocks=k_self_blocks,
        )
        return dense_inverse_or_pinv(effective)

    def spectrum(
        self,
        edge_index: Tensor,
        num_nodes: int,
        *,
        edge_weight: Tensor | None = None,
        time_step: float = 1.0,
        num_modes: int = DEFAULT_DISTRIBUTED_SPECTRUM_NUM_MODES,
    ) -> KoopmanSpectrum:
        """Eigendecomposition of the effective ``N·d`` networked operator.

        Directed / dual modes may yield complex spectra; magnitudes and
        frequencies are taken from the complex eigendecomposition (no
        real-dtype assumption on the eigenvalues).

        When ``sparsity="distributed"``, returns the ``num_modes``
        largest-modulus Ritz values from matrix-free Arnoldi (eigenvectors
        are a placeholder identity of size ``num_modes``, not ambient Ritz
        vectors). Dense / ``block_diagonal`` still assemble
        :meth:`effective_matrix` (``num_modes`` ignored).

        Parameters
        ----------
        edge_index : Tensor
            Topology used to build the adjacency factor(s).
        num_nodes : int
            Node count ``N``.
        edge_weight : Tensor or None, optional
            Optional edge weights.
        time_step : float, optional
            Discrete sampling interval for growth rates / frequencies.
        num_modes : int, optional
            Leading-modulus count for ``sparsity="distributed"``. Default
            :data:`~koopman_graph.operators.matrix_free.DEFAULT_DISTRIBUTED_SPECTRUM_NUM_MODES`.

        Returns
        -------
        KoopmanSpectrum
            Full dense spectrum, or distributed leading-modulus surrogate.
        """
        if self.sparsity == "distributed":
            result = spectrum_k_eff_graph(
                k_self=self.K_self,
                k_nbr=self.K_nbr,
                edge_index=edge_index,
                num_nodes=num_nodes,
                num_modes=num_modes,
                adjacency=self.adjacency,
                edge_weight=edge_weight,
                k_bwd=None if self._bwd is None else self.K_bwd,
            )
            return _koopman_spectrum_from_eigenvalues(result.eigenvalues, time_step)
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
            Edge index ``(2, num_edges)`` used to build adjacency factors.
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

        self.ensure_orbit_binding(z.shape[0], edge_index=edge_index)
        z_next = self.apply_tied_self(z) + self._sparse_neighbor_term(
            z,
            edge_index,
            edge_weight,
        )

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
            Advanced latent states. When ``dynamics_mode='stochastic'``,
            diagonal process noise is added after the linear map.
        """
        from koopman_graph.operators.stochastic import maybe_apply_process_noise

        _ = delta_t
        if edge_index is None:
            msg = "edge_index is required for GraphKoopmanOperator.advance"
            raise ValueError(msg)
        return maybe_apply_process_noise(
            self.forward(z, edge_index, edge_weight, control=control),
            self,
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
        """Recover previous latents from a networked forward step.

        ``sparsity="dense"`` inverts the effective ``N·d`` map (exact;
        suitable for modest ``N``). ``sparsity="block_diagonal"`` uses a
        one-step Jacobi / per-node ``d×d`` approximate inverse (exact when
        neighbor factors are zero or the graph has no edges; approximate for
        all three ``adjacency`` modes otherwise). ``sparsity="distributed"``
        uses matrix-free Richardson / Neumann iteration on a shared
        ``K_self`` (orbit / per-node bilinear self blocks are unsupported).
        ``inverse_matrix`` is supported only for ``sparsity="dense"``.

        For ``control_mode="bilinear"``, global controls fold into a shared
        ``K_self`` override; per-node controls use node-specific bilinear self
        blocks plus the same neighbor coupling as forward advance. Singular
        dense maps fall back to a pseudoinverse.

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
        edge_index : Tensor or None, optional
            Required topology.
        edge_weight : Tensor or None, optional
            Optional edge weights.

        Returns
        -------
        Tensor
            Recovered latents at ``t``.

        Raises
        ------
        ValueError
            If topology / shapes are invalid, or ``inverse_matrix`` is passed
            with ``sparsity`` other than ``"dense"``, or distributed inverse
            is requested with per-node self blocks.
        """
        from koopman_graph.operators.graph_inverse import (
            block_diagonal_graph_inverse_advance,
        )

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
                offset = broadcast_control_term(z, offset, latent_dim=self.latent_dim)
            adjusted = z - offset

        num_nodes = z.shape[0]

        def _bilinear_self_factors() -> tuple[Tensor | None, Tensor | None]:
            """Resolve shared / per-node bilinear self overrides.

            Returns
            -------
            tuple[Tensor | None, Tensor | None]
                See summary line.

            Raises
            ------
            ValueError
                Raised when inputs are invalid."""
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
            if inverse_matrix is not None:
                msg = (
                    "inverse_matrix is only supported for "
                    "GraphKoopmanOperator sparsity='dense'"
                )
                raise ValueError(msg)
            k_self_override, k_self_blocks = _bilinear_self_factors()
            if k_self_blocks is None and k_self_override is None:
                k_self_blocks = self.tied_self_blocks(num_nodes)
            return block_diagonal_graph_inverse_advance(
                adjusted,
                k_self=k_self_override if k_self_override is not None else self.K_self,
                k_nbr=self.K_nbr,
                edge_index=edge_index,
                edge_weight=edge_weight,
                k_self_blocks=k_self_blocks,
                adjacency=self.adjacency,
                k_bwd=None if self._bwd is None else self.K_bwd,
            )

        if self.sparsity == "distributed":
            if inverse_matrix is not None:
                msg = (
                    "inverse_matrix is only supported for "
                    "GraphKoopmanOperator sparsity='dense'"
                )
                raise ValueError(msg)
            k_self_override, k_self_blocks = _bilinear_self_factors()
            if k_self_blocks is None and k_self_override is None:
                k_self_blocks = self.tied_self_blocks(num_nodes)
            if k_self_blocks is not None:
                msg = (
                    "GraphKoopmanOperator sparsity='distributed' inverse "
                    "requires a shared K_self (orbit ties / per-node bilinear "
                    "self blocks are unsupported); use sparsity='dense' or "
                    "'block_diagonal'"
                )
                raise ValueError(msg)
            result = invert_k_eff_graph(
                flatten_node_latents(adjusted),
                k_self=k_self_override if k_self_override is not None else self.K_self,
                k_nbr=self.K_nbr,
                edge_index=edge_index,
                num_nodes=num_nodes,
                adjacency=self.adjacency,
                edge_weight=edge_weight,
                k_bwd=None if self._bwd is None else self.K_bwd,
            )
            return unflatten_node_latents(
                result.solution,
                num_nodes=num_nodes,
                latent_dim=self.latent_dim,
            )

        if inverse_matrix is None:
            k_self_override, k_self_blocks = _bilinear_self_factors()
            if k_self_blocks is None and k_self_override is None:
                k_self_blocks = self.tied_self_blocks(num_nodes)
            inverse_matrix = self.dense_effective_inverse(
                edge_index,
                num_nodes,
                edge_weight=edge_weight,
                k_self=k_self_override,
                k_self_blocks=k_self_blocks,
            )

        flat = adjusted.reshape(-1)
        recovered = (inverse_matrix @ flat).view_as(adjusted)
        return recovered
