"""Continuous-time multiplex / typed relational (hetero) Koopman generator.

Implements the topology-coupled generator::

    L_eff = I_N ⊗ L_self + Σ_r Â_r ⊗ L_r                       (multiplex)
    L_eff = diag_τ(I_{N_τ} ⊗ L_self^τ) + Σ_r Â_r ⊗ L_r          (typed)

with one-step transition ``Φ(Δt) = exp(Δt L_eff)``. Typed operators may opt
into unequal per-type latent widths ``d_τ`` via ``latent_dims`` (rectangular
mode); relation factors are dense ``L_r ∈ R^{d_src×d_dst}`` with the same
Appendix B / Q2=A Kronecker orientation as the discrete peer. Discrete
multiplex / typed peers live in
:mod:`koopman_graph.operators.heterogeneous`; the neighbor-only continuous
networked peer lives in
:mod:`koopman_graph.operators.continuous_graph`. This module intentionally
reuses private node/edge-type and relation-tying helpers from
:mod:`koopman_graph.operators.heterogeneous` (``_normalize_node_types``,
``_normalize_edge_types``, ``_validate_relation_tying``,
``_basis_factor_key``) so both the discrete and continuous hetero operators
share one validated notion of node types, edge types, and relation tying.

Dense-ceiling / Φ cost
----------------------
The dense path assembles ``L_eff ∈ R^{(N·d)×(N·d)}`` and forms
``Φ(Δt) = exp(Δt L_eff)``. Storage and factorization therefore scale as
``O((N·d)^2)`` (same dense-ceiling honesty as
:class:`~koopman_graph.operators.ContinuousGraphKoopmanOperator` and the
discrete
:class:`~koopman_graph.operators.HeteroGraphKoopmanOperator` ``K_eff``
path). Prefer modest ``N·d`` for the dense path; large graphs should use
``sparsity="block_diagonal"`` (self-term only — ignores relation coupling;
exact when relation generators are zero or relation graphs have no edges).
Ephemeral ``L_eff`` / ``Φ`` caches are evaluation-scoped and never written
to checkpoints (see
:meth:`ContinuousHeteroGraphKoopmanOperator.clear_transition_cache`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn

from koopman_graph.data import (
    latent_type_slices_from_dims,
    node_type_offsets,
    stacked_latent_numel,
    validate_latent_dims,
)
from koopman_graph.graph_utils import (
    RELATION_NORMALIZATION_MODES,
    RelationNormalization,
    dense_relation_normalized_adjacency,
)
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
from koopman_graph.operators.graph_types import GraphSparsity
from koopman_graph.operators.heterogeneous import (
    RELATION_TYING_MODES,
    RelationTying,
    _basis_factor_key,
    _normalize_edge_types,
    _normalize_node_types,
    _validate_relation_tying,
    relation_factor_key,
)
from koopman_graph.spectrum_types import KoopmanSpectrum, compute_generator_spectrum

__all__ = [
    "ContinuousHeteroGraphKoopmanOperator",
    "RELATION_TYING_MODES",
    "RelationTying",
    "relation_factor_key",
]

EdgeTypeTriple = tuple[str, str, str]
_LeffCacheEntry = tuple[
    list[Tensor], list[Tensor | None], int, torch.dtype, torch.device, Tensor
]
_PhiCacheEntry = tuple[
    list[Tensor], list[Tensor | None], float, torch.dtype, torch.device, Tensor
]


class ContinuousHeteroGraphKoopmanOperator(nn.Module):
    """Continuous-time multiplex / typed Koopman generator with relation coupling.

    Advances stacked node latents ``Z ∈ R^{N×d}`` over ``Δt`` via::

        vec(Z(t+Δt)) = exp(L_eff Δt) vec(Z(t))

    With one node type (**multiplex**) ``L_self`` is a single ``d×d``
    generator.     With two or more node types (**typed**) each type owns its own
    ``L_self^τ ∈ R^{d_τ×d_τ}``. By default all types share the same width
    ``d = latent_dim``; opt-in ``latent_dims`` enables unequal ``d_τ``
    (**rectangular** mode) with flat latents of length ``Σ_τ N_τ·d_τ``.
    Typed calls stack all types into one ``N = Σ_τ N_τ`` block ordered by
    :attr:`node_types`, take relation banks in **global** (offset) node
    numbering, and require ``num_nodes_dict`` so the self blocks can be
    sliced.

    ``Â_r`` is the per-relation degree-normalized adjacency from
    :func:`~koopman_graph.graph_utils.dense_relation_normalized_adjacency`
    (default ``normalization="rgcn_in_degree"``; Schlichtkrull et al. R-GCN
    in-degree convention — normalization only, not a full paper
    reproduction). Reverse relations are not synthesized.

    Control (additive or bilinear) lives on the self factor only; relation
    factors are uncontrolled, and control is rejected for typed operators
    (same restriction as
    :class:`~koopman_graph.operators.HeteroGraphKoopmanOperator`).
    ``parameterization="auxiliary_spectral"`` is rejected (state-dependent
    generator plus topology coupling is out of scope for the MVP).

    :meth:`effective_generator` / :meth:`spectrum` / :meth:`transition_matrix`
    assemble a dense ``(N·d, N·d)`` operator (or its exponential). Cost and
    memory scale as ``O((N·d)^2)`` — prefer modest ``N·d``; see the module
    ``Dense-ceiling / Φ cost`` section. For large graphs use
    ``sparsity="block_diagonal"`` (self-only approximation). Dense
    uncontrolled advances may reuse assembled ``L_eff`` and
    ``Φ = exp(Δt L_eff)`` within one training-loss evaluation; call
    :meth:`clear_transition_cache` between evaluations
    (``compute_training_loss`` does this automatically). Caching is skipped
    for typed operators: the assembled self blocks depend on
    ``num_nodes_dict``, which is not part of the cache key.

    Relation tying (R-GCN basis-decomposition motivation — not a full paper
    reproduction) mirrors
    :class:`~koopman_graph.operators.HeteroGraphKoopmanOperator`::

        L_r = sum_b a_{r,b} V_b    (relation_tying="basis")

    Attributes
    ----------
    latent_dim : int
        Shared latent feature dimension ``d`` (reference width for
        square factors; rectangular mode uses per-type ``d_τ``).
    latent_dims : dict of str to int or None
        Opt-in per-type widths when set; ``None`` keeps shared ``d``.
    is_rectangular : bool
        ``True`` when typed and any ``d_τ != latent_dim``.
    num_relations : int
        Number of relation banks ``|R|``.
    node_types : tuple of str
        Ordered node-type names; also the stacking order of ``Z``.
    is_typed : bool
        ``True`` when more than one node type is present.
    edge_types : tuple of tuple of str
        Ordered ``(src, rel, dst)`` triples aligned with relation banks.
    relation_tying : {"independent", "basis"}
        Relation-factor tying mode.
    basis_size : int or None
        Basis size ``B`` when ``relation_tying="basis"``; else ``None``.
    control_dim : int
        Exogenous control dimension (``0`` disables control).
    control_mode : {"additive", "bilinear"}
        Control coupling on the self term.
    parameterization : Parameterization
        Shared soft/structural parameterization for all ``d×d`` factors
        (``"auxiliary_spectral"`` is rejected).
    normalization : {"rgcn_in_degree", "random_walk"}
        Per-relation adjacency normalization mode.
    sparsity : {"dense", "block_diagonal", "distributed"}
        Realization mode. ``"block_diagonal"`` advances with the self
        term(s) only. ``"distributed"`` is accepted for construction /
        checkpoints (not trainer DDP or multi-GPU training; matrix-free
        inverse / spectrum are wired for discrete graph and hetero in 0.10).
    max_real_eigenvalue : float
        Stability bound forwarded to the continuous factor modules.
    """

    def __init__(
        self,
        latent_dim: int,
        num_relations: int,
        *,
        init_mode: InitMode = "identity_noise",
        init_scale: float = 1e-2,
        parameterization: Parameterization = "dense",
        max_real_eigenvalue: float = 1.0,
        sparsity: GraphSparsity = "dense",
        normalization: RelationNormalization = "rgcn_in_degree",
        control_dim: int = 0,
        control_mode: ControlMode = "additive",
        bilinear_rank: int | None = None,
        node_types: Sequence[str] | None = None,
        edge_types: Sequence[Sequence[str]] | None = None,
        relation_tying: RelationTying = "independent",
        basis_size: int | None = None,
        latent_dims: Mapping[str, int] | None = None,
    ) -> None:
        """Initialize self and per-relation continuous generators.

        Parameters
        ----------
        latent_dim : int
            Latent dimension ``d``.
        num_relations : int
            Number of relation banks (``|R| >= 1``).
        init_mode : {"identity", "identity_noise", "xavier"}, optional
            Initialization for ``L_self``. Relation / basis factors start
            near zero (plus optional noise for ``identity_noise`` /
            ``xavier``).
        init_scale : float, optional
            Noise scale for ``identity_noise`` / relation jitter.
        parameterization : Parameterization, optional
            Shared parameterization for every ``d×d`` factor.
            ``"auxiliary_spectral"`` is rejected (state-dependent generator
            plus topology coupling is out of scope).
        max_real_eigenvalue : float, optional
            Magnitude scale for structural Hurwitz modes.
        sparsity : {"dense", "block_diagonal", "distributed"}, optional
            ``"dense"`` (default) assembles the full ``N·d`` generator.
            ``"block_diagonal"`` advances with the self term(s) only
            (self-dominated approximation). ``"distributed"`` is accepted for
            construction / checkpoints (matrix-free inverse / spectrum are
            wired for discrete graph and hetero in 0.10; continuous paths may
            still assemble).
        normalization : {"rgcn_in_degree", "random_walk"}, optional
            Per-relation degree normalization. Default is
            ``"rgcn_in_degree"``.
        control_dim : int, optional
            Additive / bilinear control dimension on the self term.
            Default ``0``.
        control_mode : {"additive", "bilinear"}, optional
            Control coupling forwarded to the self-term operator.
        bilinear_rank : int or None, optional
            Low-rank bilinear size when ``control_mode="bilinear"``.
        node_types : sequence of str or None, optional
            Ordered node-type names. Defaults to ``("node",)`` (multiplex).
            Two or more names build one ``L_self^τ`` per type.
        edge_types : sequence of sequence of str or None, optional
            Ordered ``(src, rel, dst)`` triples (length ``num_relations``).
            Defaults to ``(node, r{i}, node)``; required for typed operators.
        relation_tying : {"independent", "basis"}, optional
            Relation-factor tying. ``"basis"`` shares ``B`` matrices
            ``V_b`` with coefficients ``a_{r,b}``. Default ``"independent"``.
        basis_size : int or None, optional
            Basis size ``B`` when ``relation_tying="basis"``
            (``1 <= B <= |R|``). Must be ``None`` for independent tying.
        latent_dims : mapping of str to int or None, optional
            Opt-in per-type widths ``τ -> d_τ``. ``None`` (default) is the
            shared-d square path. Rectangular mode activates when any
            ``d_τ != latent_dim``. Multiplex requires the sole width equal
            ``latent_dim``. Unsupported with basis tying or non-dense
            parameterization / sparsity.

        Raises
        ------
        ValueError
            If dimensions, ``sparsity``, ``normalization``,
            ``parameterization``, tying metadata, or type metadata are
            unsupported, or control is requested for a typed operator.
        """
        super().__init__()
        if latent_dim < 1:
            msg = f"latent_dim must be positive, got {latent_dim}"
            raise ValueError(msg)
        if num_relations < 1:
            msg = f"num_relations must be positive, got {num_relations}"
            raise ValueError(msg)
        if control_dim < 0:
            msg = f"control_dim must be non-negative, got {control_dim}"
            raise ValueError(msg)
        if sparsity not in {"dense", "block_diagonal", "distributed"}:
            msg = (
                "ContinuousHeteroGraphKoopmanOperator sparsity must be "
                f"'dense', 'block_diagonal', or 'distributed', got "
                f"{sparsity!r}"
            )
            raise ValueError(msg)
        if normalization not in RELATION_NORMALIZATION_MODES:
            msg = (
                "normalization must be one of "
                f"{sorted(RELATION_NORMALIZATION_MODES)}, got {normalization!r}"
            )
            raise ValueError(msg)
        if parameterization == "auxiliary_spectral":
            msg = (
                "parameterization='auxiliary_spectral' is not supported for "
                "ContinuousHeteroGraphKoopmanOperator (state-dependent + "
                "topology)"
            )
            raise ValueError(msg)

        resolved_tying, resolved_basis_size = _validate_relation_tying(
            relation_tying,
            basis_size,
            num_relations=num_relations,
        )
        resolved_node_types = _normalize_node_types(node_types)
        resolved_edge_types = _normalize_edge_types(
            edge_types,
            num_relations=num_relations,
            node_types=resolved_node_types,
        )
        is_typed = len(resolved_node_types) > 1
        if is_typed and control_dim > 0:
            msg = (
                "control is unsupported for typed "
                "ContinuousHeteroGraphKoopmanOperator "
                f"(node_types={resolved_node_types!r}); set control_dim=0"
            )
            raise ValueError(msg)

        resolved_latent_dims = validate_latent_dims(
            resolved_node_types,
            latent_dims,
            shared_latent_dim=latent_dim,
        )
        is_rectangular = False
        if resolved_latent_dims is not None:
            if not is_typed:
                sole = resolved_latent_dims[resolved_node_types[0]]
                if sole != latent_dim:
                    msg = (
                        "multiplex latent_dims must equal latent_dim "
                        f"({latent_dim}); got {resolved_latent_dims!r}"
                    )
                    raise ValueError(msg)
            else:
                is_rectangular = any(
                    width != latent_dim for width in resolved_latent_dims.values()
                )
        if is_rectangular:
            if resolved_tying != "independent":
                msg = (
                    "rectangular ContinuousHeteroGraphKoopmanOperator requires "
                    "relation_tying='independent'"
                )
                raise ValueError(msg)
            if parameterization != "dense":
                msg = (
                    "rectangular ContinuousHeteroGraphKoopmanOperator requires "
                    "parameterization='dense'"
                )
                raise ValueError(msg)
            if sparsity != "dense":
                msg = (
                    "rectangular ContinuousHeteroGraphKoopmanOperator requires "
                    "sparsity='dense'"
                )
                raise ValueError(msg)
            if control_dim > 0:
                msg = (
                    "control is unsupported for rectangular "
                    "ContinuousHeteroGraphKoopmanOperator; set control_dim=0"
                )
                raise ValueError(msg)

        self.latent_dim = latent_dim
        self.latent_dims = resolved_latent_dims
        self.is_rectangular = is_rectangular
        self.num_relations = num_relations
        self.node_types = resolved_node_types
        self.is_typed = is_typed
        self.edge_types = resolved_edge_types
        self.relation_tying: RelationTying = resolved_tying
        self.basis_size = resolved_basis_size
        self.init_mode = init_mode
        self.init_scale = init_scale
        self.parameterization = parameterization
        self.max_real_eigenvalue = max_real_eigenvalue
        self.sparsity = sparsity
        self.normalization: RelationNormalization = normalization
        self.control_dim = control_dim
        self.control_mode = control_mode
        self.bilinear_rank = bilinear_rank

        def _build_self_factor(width: int) -> ContinuousKoopmanOperator:
            """Return a fresh self-coupling generator at width ``width``.

            Parameters
            ----------
            width : int
                Latent width for this self factor.

            Returns
            -------
            ContinuousKoopmanOperator
                Per-node self factor (typed operators build one per type).
            """
            return ContinuousKoopmanOperator(
                width,
                init_mode=init_mode,
                init_scale=init_scale,
                parameterization=parameterization,
                max_real_eigenvalue=max_real_eigenvalue,
                control_dim=control_dim,
                control_mode=control_mode,
                bilinear_rank=bilinear_rank,
            )

        def _build_relation_factor() -> ContinuousKoopmanOperator:
            """Return a fresh uncontrolled ``d×d`` relation / basis generator.

            Returns
            -------
            ContinuousKoopmanOperator
                Relation bank or shared basis generator module.
            """
            return ContinuousKoopmanOperator(
                latent_dim,
                init_mode="identity",
                init_scale=init_scale,
                parameterization=parameterization,
                max_real_eigenvalue=max_real_eigenvalue,
                control_dim=0,
            )

        # Multiplex keeps the flat ``_self`` submodule (checkpoint key
        # parity with the discrete peer); typed operators key one factor per
        # node type under ``_selves``.
        if is_typed:
            self._selves = nn.ModuleDict(
                {
                    name: _build_self_factor(
                        resolved_latent_dims[name]
                        if is_rectangular and resolved_latent_dims is not None
                        else latent_dim
                    )
                    for name in resolved_node_types
                }
            )
        else:
            self._self = _build_self_factor(latent_dim)

        if is_rectangular:
            assert resolved_latent_dims is not None
            # Appendix B / Q2=A: L_r ∈ R^{d_src × d_dst}. L_eff relation
            # blocks use Â_{dst←src} ⊗ L_r.T (matching discrete K_eff).
            self._rel_rect = nn.ParameterDict()
            for edge_type in resolved_edge_types:
                src, _rel, dst = edge_type
                d_src = resolved_latent_dims[src]
                d_dst = resolved_latent_dims[dst]
                key = relation_factor_key(edge_type)
                self._rel_rect[key] = nn.Parameter(torch.zeros(d_src, d_dst))
        elif resolved_tying == "independent":
            self._rel = nn.ModuleDict(
                {
                    relation_factor_key(edge_type): _build_relation_factor()
                    for edge_type in resolved_edge_types
                }
            )
        else:
            assert resolved_basis_size is not None
            self._basis = nn.ModuleDict(
                {
                    _basis_factor_key(basis_idx): _build_relation_factor()
                    for basis_idx in range(resolved_basis_size)
                }
            )
            coeffs = torch.zeros(num_relations, resolved_basis_size)
            eye = torch.eye(num_relations, resolved_basis_size)
            coeffs.copy_(eye)
            self._rel_coeff = nn.Parameter(coeffs)
        self._reset_relation_parameters()

        # Ephemeral L_eff / Φ reuse within one training-loss evaluation.
        # Cleared by clear_transition_cache (wired from compute_training_loss).
        # Never populated for typed operators (see class docstring).
        self._leff_cache: list[_LeffCacheEntry] = []
        self._phi_cache: list[_PhiCacheEntry] = []

    # -- type / relation module lookups -------------------------------------

    def self_operator_for(self, node_type: str) -> ContinuousKoopmanOperator:
        """Return the self-coupling generator module for ``node_type``.

        Parameters
        ----------
        node_type : str
            Node-type name from :attr:`node_types`.

        Returns
        -------
        ContinuousKoopmanOperator
            Self factor owning ``L_self^τ`` (the sole factor for multiplex).

        Raises
        ------
        KeyError
            If ``node_type`` is not in :attr:`node_types`.
        """
        if node_type not in self.node_types:
            msg = (
                f"unknown node type {node_type!r}; "
                f"valid types are {list(self.node_types)!r}"
            )
            raise KeyError(msg)
        if self.is_typed:
            module = self._selves[node_type]
            assert isinstance(module, ContinuousKoopmanOperator)
            return module
        return self._self

    def _self_modules(self) -> tuple[ContinuousKoopmanOperator, ...]:
        """Return self-coupling factors in :attr:`node_types` order.

        Returns
        -------
        tuple of ContinuousKoopmanOperator
            One factor per node type (length 1 for multiplex).
        """
        return tuple(self.self_operator_for(name) for name in self.node_types)

    def l_self_for(self, node_type: str) -> Tensor:
        """Return the assembled ``L_self^τ`` for ``node_type``.

        Parameters
        ----------
        node_type : str
            Node-type name from :attr:`node_types`.

        Returns
        -------
        Tensor
            Self-coupling generator with shape ``(latent_dim, latent_dim)``.

        Raises
        ------
        KeyError
            If ``node_type`` is not in :attr:`node_types`.
        """
        return self.self_operator_for(node_type).L

    def typed_l_self_blocks(self, num_nodes_dict: Mapping[str, int]) -> Tensor:
        """Return per-node self blocks for the stacked typed layout.

        Parameters
        ----------
        num_nodes_dict : mapping of str to int
            Node count ``N_τ`` for every entry of :attr:`node_types`.

        Returns
        -------
        Tensor
            Stacked blocks with shape ``(N, latent_dim, latent_dim)`` where
            ``N = Σ_τ N_τ``; row ``i`` holds ``L_self^τ`` of the type owning
            stacked node ``i``. Suitable for the ``l_self_blocks`` argument of
            :meth:`effective_generator`.

        Raises
        ------
        ValueError
            If the operator is rectangular, or ``num_nodes_dict`` does not
            cover :attr:`node_types` exactly.
        """
        if self.is_rectangular:
            msg = (
                "typed_l_self_blocks is undefined for rectangular "
                "ContinuousHeteroGraphKoopmanOperator; use "
                "effective_generator / pack_typed_latents with latent_dims"
            )
            raise ValueError(msg)
        counts = self._validate_num_nodes_dict(num_nodes_dict)
        blocks = [
            self.l_self_for(name).expand(counts[name], self.latent_dim, self.latent_dim)
            for name in self.node_types
        ]
        return torch.cat(blocks, dim=0)

    def d_for(self, node_type: str) -> int:
        """Return the latent width for ``node_type``.

        Parameters
        ----------
        node_type : str
            Node-type name from :attr:`node_types`.

        Returns
        -------
        int
            ``d_τ`` when ``latent_dims`` is set, else :attr:`latent_dim`.

        Raises
        ------
        KeyError
            If ``node_type`` is not in :attr:`node_types`.
        """
        if node_type not in self.node_types:
            msg = (
                f"unknown node type {node_type!r}; "
                f"valid types are {list(self.node_types)!r}"
            )
            raise KeyError(msg)
        if self.latent_dims is not None:
            return int(self.latent_dims[node_type])
        return int(self.latent_dim)

    def pack_typed_latents(
        self,
        z_by_type: Mapping[str, Tensor],
        num_nodes_dict: Mapping[str, int],
    ) -> Tensor:
        """Flatten per-type ``(N_τ, d_τ)`` blocks into one C-order vector.

        Parameters
        ----------
        z_by_type : mapping of str to Tensor
            Latents keyed by node type.
        num_nodes_dict : mapping of str to int
            Per-type node counts.

        Returns
        -------
        Tensor
            Flat vector of length ``Σ_τ N_τ·d_τ`` (rectangular) or
            ``N·d`` (shared-d stacked rows flattened).

        Raises
        ------
        ValueError
            If keys or shapes disagree with the operator layout.
        """
        counts = self._validate_num_nodes_dict(num_nodes_dict)
        if set(z_by_type) != set(self.node_types):
            msg = (
                "z_by_type keys must match node_types "
                f"{list(self.node_types)!r}; got {sorted(z_by_type)!r}"
            )
            raise ValueError(msg)
        blocks: list[Tensor] = []
        for name in self.node_types:
            width = self.d_for(name)
            expected = (counts[name], width)
            block = z_by_type[name]
            if tuple(block.shape) != expected:
                msg = (
                    f"z_by_type[{name!r}] must have shape {expected}, "
                    f"got {tuple(block.shape)}"
                )
                raise ValueError(msg)
            blocks.append(block.reshape(-1))
        return torch.cat(blocks, dim=0)

    def unpack_typed_latents(
        self,
        z_flat: Tensor,
        num_nodes_dict: Mapping[str, int],
    ) -> dict[str, Tensor]:
        """Split a flat latent vector into per-type ``(N_τ, d_τ)`` blocks.

        Parameters
        ----------
        z_flat : Tensor
            Flat latent vector.
        num_nodes_dict : mapping of str to int
            Per-type node counts.

        Returns
        -------
        dict of str to Tensor
            Per-type blocks in :attr:`node_types` order.

        Raises
        ------
        ValueError
            If the flat length or counts are inconsistent.
        """
        counts = self._validate_num_nodes_dict(num_nodes_dict)
        if self.is_rectangular:
            assert self.latent_dims is not None
            expected = stacked_latent_numel(
                self.node_types,
                counts,
                self.latent_dims,
            )
            slices = latent_type_slices_from_dims(
                self.node_types,
                counts,
                self.latent_dims,
            )
        else:
            expected = sum(counts.values()) * self.latent_dim
            slices = {
                name: slice(
                    sum(counts[n] for n in self.node_types[:idx]) * self.latent_dim,
                    sum(counts[n] for n in self.node_types[: idx + 1])
                    * self.latent_dim,
                )
                for idx, name in enumerate(self.node_types)
            }
        if z_flat.ndim != 1 or int(z_flat.numel()) != expected:
            msg = f"z_flat must have shape ({expected},), got {tuple(z_flat.shape)}"
            raise ValueError(msg)
        return {
            name: z_flat[slices[name]].reshape(counts[name], self.d_for(name))
            for name in self.node_types
        }

    def _validate_num_nodes_dict(
        self,
        num_nodes_dict: Mapping[str, int],
        *,
        num_nodes: int | None = None,
    ) -> dict[str, int]:
        """Validate per-type node counts against :attr:`node_types`.

        Parameters
        ----------
        num_nodes_dict : mapping of str to int
            Node count ``N_τ`` per node type.
        num_nodes : int or None, optional
            Expected stacked total ``N``. Checked when provided.

        Returns
        -------
        dict of str to int
            Validated counts keyed by node type.

        Raises
        ------
        ValueError
            If keys mismatch :attr:`node_types`, a count is not positive, or
            the counts do not sum to ``num_nodes``.
        """
        if set(num_nodes_dict) != set(self.node_types):
            msg = (
                "num_nodes_dict keys must match operator node_types "
                f"{list(self.node_types)!r}; got {sorted(num_nodes_dict)!r}"
            )
            raise ValueError(msg)
        counts: dict[str, int] = {}
        for name in self.node_types:
            count = int(num_nodes_dict[name])
            if count < 1:
                msg = f"num_nodes_dict[{name!r}] must be positive, got {count}"
                raise ValueError(msg)
            counts[name] = count
        total = sum(counts.values())
        if num_nodes is not None and total != num_nodes:
            msg = (
                f"num_nodes_dict sums to {total} but the stacked latent block "
                f"has {num_nodes} rows"
            )
            raise ValueError(msg)
        return counts

    def _require_num_nodes_dict(
        self,
        num_nodes_dict: Mapping[str, int] | None,
        *,
        num_nodes: int | None = None,
        caller: str,
    ) -> dict[str, int] | None:
        """Resolve ``num_nodes_dict`` for typed calls (``None`` for multiplex).

        Parameters
        ----------
        num_nodes_dict : mapping of str to int or None
            Caller-supplied per-type counts.
        num_nodes : int or None, optional
            Expected stacked total ``N``.
        caller : str
            Method name used in error messages.

        Returns
        -------
        dict of str to int or None
            Validated counts for typed operators, ``None`` for multiplex.

        Raises
        ------
        ValueError
            If a typed operator is called without ``num_nodes_dict``, or the
            supplied counts are inconsistent.
        """
        if not self.is_typed:
            if num_nodes_dict is None:
                return None
            return self._validate_num_nodes_dict(num_nodes_dict, num_nodes=num_nodes)
        if num_nodes_dict is None:
            msg = (
                f"{caller} requires num_nodes_dict for typed "
                "ContinuousHeteroGraphKoopmanOperator (node_types="
                f"{list(self.node_types)!r}) so per-type L_self blocks can be "
                "sliced from the stacked latent block"
            )
            raise ValueError(msg)
        return self._validate_num_nodes_dict(num_nodes_dict, num_nodes=num_nodes)

    def _relation_modules(self) -> tuple[ContinuousKoopmanOperator, ...]:
        """Return independent relation factor modules in ``edge_types`` order.

        Returns
        -------
        tuple of ContinuousKoopmanOperator
            Ordered relation banks (independent tying only).

        Raises
        ------
        ValueError
            If ``relation_tying="basis"`` or the operator is rectangular.
        """
        if self.is_rectangular:
            msg = (
                "_relation_modules is undefined for rectangular operators; "
                "use relation_matrix() / _rel_rect"
            )
            raise ValueError(msg)
        if self.relation_tying != "independent":
            msg = (
                "_relation_modules is only defined for "
                "relation_tying='independent'; use _basis_modules() / "
                "relation_matrix() when relation_tying='basis'"
            )
            raise ValueError(msg)
        return tuple(
            self._rel[relation_factor_key(edge_type)] for edge_type in self.edge_types
        )

    def _basis_modules(self) -> tuple[ContinuousKoopmanOperator, ...]:
        """Return shared basis factor modules ``V_0 … V_{B-1}``.

        Returns
        -------
        tuple of ContinuousKoopmanOperator
            Ordered basis banks (basis tying only).

        Raises
        ------
        ValueError
            If ``relation_tying != "basis"``.
        """
        if self.relation_tying != "basis":
            msg = (
                "_basis_modules is only defined for relation_tying='basis'; "
                f"got {self.relation_tying!r}"
            )
            raise ValueError(msg)
        assert self.basis_size is not None
        return tuple(
            self._basis[_basis_factor_key(basis_idx)]
            for basis_idx in range(self.basis_size)
        )

    def _assembled_relation_matrix(self, relation_index: int) -> Tensor:
        """Assemble ``L_r`` for independent or basis tying.

        Parameters
        ----------
        relation_index : int
            Relation bank index in ``[0, num_relations)``.

        Returns
        -------
        Tensor
            Assembled ``L_r`` with shape ``(latent_dim, latent_dim)``.

        Raises
        ------
        IndexError
            If ``relation_index`` is out of range.
        """
        if not 0 <= relation_index < self.num_relations:
            msg = (
                f"relation_index must be in [0, {self.num_relations}), "
                f"got {relation_index}"
            )
            raise IndexError(msg)
        if self.is_rectangular:
            key = relation_factor_key(self.edge_types[relation_index])
            return self._rel_rect[key]
        if self.relation_tying == "independent":
            return self._relation_modules()[relation_index].L
        assert self.basis_size is not None
        assembled = self._rel_coeff.new_zeros(self.latent_dim, self.latent_dim)
        for basis_idx, module in enumerate(self._basis_modules()):
            weight = self._rel_coeff[relation_index, basis_idx]
            assembled = assembled + weight * module.L
        return assembled

    # -- parameter initialization / reset -----------------------------------

    def _reset_factor_parameters(
        self,
        module: ContinuousKoopmanOperator,
        *,
        allow_noise: bool,
    ) -> None:
        """Zero a relation generator, optionally adding ``init_scale`` noise.

        Parameters
        ----------
        module : ContinuousKoopmanOperator
            Relation factor module to reset.
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

    def _reset_relation_parameters(self) -> None:
        """Initialize relation / basis generators near zero (optional jitter).

        Returns
        -------
        None
        """
        if self.is_rectangular:
            with torch.no_grad():
                for parameter in self._rel_rect.values():
                    parameter.zero_()
                    if self.init_mode in {"identity_noise", "xavier"}:
                        parameter.add_(torch.randn_like(parameter) * self.init_scale)
            return
        if self.relation_tying == "independent":
            for module in self._relation_modules():
                self._reset_factor_parameters(module, allow_noise=True)
            return
        assert self.basis_size is not None
        for module in self._basis_modules():
            self._reset_factor_parameters(module, allow_noise=True)
        with torch.no_grad():
            self._rel_coeff.zero_()
            eye = torch.eye(
                self.num_relations,
                self.basis_size,
                dtype=self._rel_coeff.dtype,
                device=self._rel_coeff.device,
            )
            self._rel_coeff.copy_(eye)

    def reset_parameters(self) -> None:
        """Reinitialize self factors (and control) plus all relation factors.

        Returns
        -------
        None
        """
        for module in self._self_modules():
            module.reset_parameters()
        if self.control_dim > 0 and not self.is_typed:
            self._self.reset_control_parameters()
        if not self.is_rectangular:
            if self.relation_tying == "independent":
                for module in self._relation_modules():
                    module.reset_parameters()
            else:
                for module in self._basis_modules():
                    module.reset_parameters()
        self._reset_relation_parameters()

    # -- cache management -----------------------------------------------

    def clear_transition_cache(self) -> None:
        """Drop cached dense ``L_eff`` and transition matrices ``Φ = exp(Δt L_eff)``.

        Call at the start of each training-loss evaluation so cached
        generators and transitions never span an optimizer step. Ordinary
        topology / ``Δt`` changes miss the cache key and rebuild
        automatically.

        Notes
        -----
        Entries are ephemeral and never written to ``state_dict``. Bilinear
        pair-local generators (``l_self`` / ``l_self_blocks`` overrides), Van
        Loan controlled advances, and typed operators do not use these
        caches.
        """
        self._leff_cache.clear()
        self._phi_cache.clear()

    def _relation_banks_equal(
        self,
        indices_a: Sequence[Tensor],
        weights_a: Sequence[Tensor | None],
        indices_b: Sequence[Tensor],
        weights_b: Sequence[Tensor | None],
    ) -> bool:
        """Return whether two ordered relation-bank payloads match by content.

        Parameters
        ----------
        indices_a, indices_b : sequence of Tensor
            Ordered per-relation edge indices.
        weights_a, weights_b : sequence of Tensor or None
            Ordered optional per-relation edge weights.

        Returns
        -------
        bool
            ``True`` when every relation bank matches (indices via
            ``torch.equal``, weights via ``torch.allclose``).
        """
        if len(indices_a) != len(indices_b):
            return False
        for index_a, index_b in zip(indices_a, indices_b, strict=True):
            if not torch.equal(index_a, index_b):
                return False
        for weight_a, weight_b in zip(weights_a, weights_b, strict=True):
            if (weight_a is None) != (weight_b is None):
                return False
            if weight_a is not None:
                assert weight_b is not None
                if not torch.allclose(weight_a, weight_b, equal_nan=True):
                    return False
        return True

    def _lookup_cached_generator(
        self,
        indices: Sequence[Tensor],
        weights: Sequence[Tensor | None],
        num_nodes: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor | None:
        """Return a cached ``L_eff`` for matching relation banks / size / dtype.

        Parameters
        ----------
        indices : sequence of Tensor
            Ordered per-relation edge indices.
        weights : sequence of Tensor or None
            Ordered optional per-relation edge weights.
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
            cached_indices,
            cached_weights,
            cached_nodes,
            cached_dtype,
            cached_device,
            cached_generator,
        ) in self._leff_cache:
            if (
                cached_nodes == num_nodes
                and cached_dtype == dtype
                and cached_device == device
                and self._relation_banks_equal(
                    indices, weights, cached_indices, cached_weights
                )
            ):
                return cached_generator
        return None

    def _lookup_cached_transition(
        self,
        indices: Sequence[Tensor],
        weights: Sequence[Tensor | None],
        delta_value: float,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor | None:
        """Return a cached ``Φ`` for matching relation banks / ``Δt`` / dtype.

        Parameters
        ----------
        indices : sequence of Tensor
            Ordered per-relation edge indices.
        weights : sequence of Tensor or None
            Ordered optional per-relation edge weights.
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
            cached_indices,
            cached_weights,
            cached_delta,
            cached_dtype,
            cached_device,
            cached_phi,
        ) in self._phi_cache:
            if (
                cached_delta == delta_value
                and cached_dtype == dtype
                and cached_device == device
                and self._relation_banks_equal(
                    indices, weights, cached_indices, cached_weights
                )
            ):
                return cached_phi
        return None

    # -- assembled matrices / properties -------------------------------------

    @property
    def L_self(self) -> Tensor:
        """Self-coupling generator with shape ``(latent_dim, latent_dim)``.

        Returns
        -------
        Tensor
            Assembled ``L_self`` (multiplex only).

        Raises
        ------
        ValueError
            If the operator is typed; there is no single shared ``L_self``.
            Use :meth:`l_self_for` for one type or
            :meth:`typed_l_self_blocks` for the stacked block-diagonal form.
        """
        if self.is_typed:
            msg = (
                "L_self is undefined for typed "
                "ContinuousHeteroGraphKoopmanOperator "
                f"(node_types={list(self.node_types)!r}); use "
                "l_self_for(node_type) or typed_l_self_blocks(num_nodes_dict)"
            )
            raise ValueError(msg)
        return self._self.L

    @property
    def L_relations(self) -> tuple[Tensor, ...]:
        """Ordered relation-coupling generators, each ``(latent_dim, latent_dim)``.

        Returns
        -------
        tuple of Tensor
            Assembled ``L_r`` for ``r = 0 … |R|-1`` (basis-combined when
            ``relation_tying="basis"``).
        """
        return tuple(
            self._assembled_relation_matrix(relation_idx)
            for relation_idx in range(self.num_relations)
        )

    def relation_matrix(self, relation_index: int) -> Tensor:
        """Return the ``relation_index``-th relation generator ``L_r``.

        Parameters
        ----------
        relation_index : int
            Relation bank index in ``[0, num_relations)``.

        Returns
        -------
        Tensor
            Assembled ``L_r`` with shape ``(latent_dim, latent_dim)``.

        Raises
        ------
        IndexError
            If ``relation_index`` is out of range.
        """
        return self._assembled_relation_matrix(relation_index)

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

    @property
    def B(self) -> Tensor | None:
        """Control matrix from the self factor, when controlled.

        Returns
        -------
        Tensor or None
            Control matrix ``B``, or ``None`` when uncontrolled.
        """
        if self.control_dim <= 0:
            return None
        return self._self.B

    def set_dense_matrices(
        self,
        l_self: Tensor | Mapping[str, Tensor],
        l_relations: Sequence[Tensor],
        *,
        control_matrix: Tensor | None = None,
        bilinear_matrices: Tensor | None = None,
    ) -> None:
        """Write dense self / relation generators (and optional control).

        Parameters
        ----------
        l_self : Tensor or mapping of str to Tensor
            Dense self generator ``(latent_dim, latent_dim)`` for multiplex,
            or a mapping ``node_type -> (latent_dim, latent_dim)`` covering
            every entry of :attr:`node_types` for typed operators.
        l_relations : sequence of Tensor
            Dense relation generators, length ``num_relations``, each
            ``(latent_dim, latent_dim)``.
        control_matrix : Tensor or None, optional
            Control matrix ``B`` when ``control_dim > 0``.
        bilinear_matrices : Tensor or None, optional
            Full-rank bilinear stack when ``control_mode="bilinear"``.

        Raises
        ------
        ValueError
            If ``relation_tying="basis"``, the relation bank length mismatches
            ``num_relations``, or the ``l_self`` form does not match the
            multiplex / typed mode.
        """
        if self.relation_tying == "basis":
            msg = (
                "set_dense_matrices is unsupported when "
                "relation_tying='basis'; use set_basis_factors(...) for "
                "V_b / a_{r,b}, and self_operator_for(...).set_dense_matrix(...) "
                "for L_self"
            )
            raise ValueError(msg)
        if len(l_relations) != self.num_relations:
            msg = (
                f"Expected {self.num_relations} relation generators, "
                f"got {len(l_relations)}"
            )
            raise ValueError(msg)
        if self.is_typed:
            if not isinstance(l_self, Mapping):
                msg = (
                    "typed ContinuousHeteroGraphKoopmanOperator."
                    "set_dense_matrices requires a mapping node_type -> "
                    f"L_self^tau for node_types {list(self.node_types)!r}"
                )
                raise ValueError(msg)
            if set(l_self) != set(self.node_types):
                msg = (
                    "l_self keys must match node_types "
                    f"{list(self.node_types)!r}; got {sorted(l_self)!r}"
                )
                raise ValueError(msg)
            for name in self.node_types:
                self.self_operator_for(name).set_dense_matrix(
                    l_self[name],
                    control_matrix=None,
                )
        else:
            if isinstance(l_self, Mapping):
                msg = (
                    "multiplex ContinuousHeteroGraphKoopmanOperator."
                    "set_dense_matrices requires a single dense L_self "
                    "tensor, not a mapping"
                )
                raise ValueError(msg)
            self._self.set_dense_matrix(
                l_self,
                control_matrix=control_matrix,
                bilinear_matrices=bilinear_matrices,
            )
        if self.is_rectangular:
            assert self.latent_dims is not None
            for relation_idx, l_rel in enumerate(l_relations):
                src, _rel, dst = self.edge_types[relation_idx]
                expected = (
                    self.latent_dims[src],
                    self.latent_dims[dst],
                )
                if tuple(l_rel.shape) != expected:
                    msg = (
                        f"l_relations[{relation_idx}] must have shape "
                        f"{expected} for edge {self.edge_types[relation_idx]!r} "
                        f"(Appendix B d_src×d_dst), got {tuple(l_rel.shape)}"
                    )
                    raise ValueError(msg)
                key = relation_factor_key(self.edge_types[relation_idx])
                with torch.no_grad():
                    self._rel_rect[key].copy_(l_rel)
            return
        for module, l_rel in zip(self._relation_modules(), l_relations, strict=True):
            module.set_dense_matrix(l_rel, control_matrix=None)

    def set_basis_factors(
        self,
        basis_matrices: Sequence[Tensor],
        coefficients: Tensor,
    ) -> None:
        """Write dense basis generators ``V_b`` and coefficients ``a_{r,b}``.

        Parameters
        ----------
        basis_matrices : sequence of Tensor
            Dense basis generators, length ``basis_size``, each
            ``(latent_dim, latent_dim)``.
        coefficients : Tensor
            Coefficient matrix with shape ``(num_relations, basis_size)``.

        Raises
        ------
        ValueError
            If ``relation_tying != "basis"`` or shapes mismatch.
        """
        if self.relation_tying != "basis":
            msg = (
                "set_basis_factors requires relation_tying='basis'; "
                f"got {self.relation_tying!r}"
            )
            raise ValueError(msg)
        assert self.basis_size is not None
        if len(basis_matrices) != self.basis_size:
            msg = (
                f"Expected {self.basis_size} basis matrices, got {len(basis_matrices)}"
            )
            raise ValueError(msg)
        if coefficients.shape != (self.num_relations, self.basis_size):
            msg = (
                "coefficients must have shape "
                f"({self.num_relations}, {self.basis_size}), "
                f"got {tuple(coefficients.shape)}"
            )
            raise ValueError(msg)
        for module, basis_matrix in zip(
            self._basis_modules(), basis_matrices, strict=True
        ):
            module.set_dense_matrix(basis_matrix, control_matrix=None)
        with torch.no_grad():
            self._rel_coeff.copy_(coefficients.to(dtype=self._rel_coeff.dtype))

    def bound_metric(self) -> Tensor:
        """Return ``max`` of self / relation factor bounds for monitoring.

        This is a **factor-level** surrogate, not a whole-network Hurwitz
        certificate for ``L_eff``. Under basis tying the bound covers the
        shared ``V_b`` factors (not assembled ``L_r``).

        Returns
        -------
        Tensor
            Scalar factor bound metric.
        """
        if self.is_rectangular:
            metric = self._self_modules()[0].bound_metric()
            for module in self._self_modules()[1:]:
                metric = torch.maximum(metric, module.bound_metric())
            for parameter in self._rel_rect.values():
                rel_norm = torch.linalg.matrix_norm(parameter, ord=2)
                metric = torch.maximum(metric, rel_norm)
            return metric
        if self.relation_tying == "independent":
            modules = (*self._self_modules(), *self._relation_modules())
        else:
            modules = (*self._self_modules(), *self._basis_modules())
        metric = modules[0].bound_metric()
        for module in modules[1:]:
            metric = torch.maximum(metric, module.bound_metric())
        return metric

    def stability_certificate(self) -> StabilityCertificate | None:
        """Return a **factor-level** self-term certificate, if any.

        Typed operators report the certificate of the **first** node type in
        :attr:`node_types`. Factor certificates never certify a joint
        ``L_eff`` bound. Discrete hetero / graph operators expose topology-
        aware joint Gershgorin certificates; continuous joint Hurwitz
        Gershgorin is not wired here yet.

        Returns
        -------
        StabilityCertificate or None
            Certificate from a self-coupling factor, if any.
        """
        return self._self_modules()[0].stability_certificate()

    # -- relation bank validation / assembly ---------------------------------

    def _resolve_relation_banks(
        self,
        edge_indices: Sequence[Tensor],
        edge_weights: Sequence[Tensor | None] | None,
    ) -> tuple[list[Tensor], list[Tensor | None]]:
        """Validate and normalize ordered relation topology banks.

        Parameters
        ----------
        edge_indices : sequence of Tensor
            Per-relation edge indices, each ``(2, E_r)``.
        edge_weights : sequence of Tensor or None, optional
            Optional per-relation weights.

        Returns
        -------
        tuple of (list of Tensor, list of Tensor or None)
            Ordered edge indices and weights.

        Raises
        ------
        ValueError
            If bank lengths mismatch ``num_relations``.
        """
        if len(edge_indices) != self.num_relations:
            msg = (
                f"Expected {self.num_relations} relation edge banks, "
                f"got {len(edge_indices)}"
            )
            raise ValueError(msg)
        indices = list(edge_indices)
        if edge_weights is None:
            weights: list[Tensor | None] = [None] * self.num_relations
        else:
            if len(edge_weights) != self.num_relations:
                msg = (
                    f"Expected {self.num_relations} relation weight banks, "
                    f"got {len(edge_weights)}"
                )
                raise ValueError(msg)
            weights = list(edge_weights)
        return indices, weights

    def _relation_coupling_generator(
        self,
        edge_indices: Sequence[Tensor],
        num_nodes: int,
        edge_weights: Sequence[Tensor | None],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        """Assemble ``Σ_r Â_r ⊗ L_r``.

        Parameters
        ----------
        edge_indices : sequence of Tensor
            Ordered relation edge indices.
        num_nodes : int
            Node count ``N``.
        edge_weights : sequence of Tensor or None
            Ordered optional relation weights.
        dtype : torch.dtype
            Floating dtype for the dense factors.
        device : torch.device
            Device for assembled tensors.

        Returns
        -------
        Tensor
            Dense relation coupling with shape ``(N·d, N·d)``.

        Raises
        ------
        ValueError
            If the operator is rectangular (use
            :meth:`_rectangular_effective_generator`).
        """
        if self.is_rectangular:
            msg = (
                "_relation_coupling_generator is shared-d only; use "
                "_rectangular_effective_generator for unequal d_τ"
            )
            raise ValueError(msg)
        coupling = torch.zeros(
            (num_nodes * self.latent_dim, num_nodes * self.latent_dim),
            dtype=dtype,
            device=device,
        )
        for relation_idx, edge_index in enumerate(edge_indices):
            adj = dense_relation_normalized_adjacency(
                edge_index,
                num_nodes,
                edge_weight=edge_weights[relation_idx],
                dtype=dtype,
                normalization=self.normalization,
            )
            l_rel = self._assembled_relation_matrix(relation_idx)
            coupling = coupling + torch.kron(adj, l_rel)
        return coupling

    def _rectangular_effective_generator(
        self,
        edge_indices: Sequence[Tensor],
        num_nodes: int,
        edge_weights: Sequence[Tensor | None],
        num_nodes_dict: Mapping[str, int],
    ) -> Tensor:
        """Assemble dense ``L_eff`` for unequal ``d_τ`` (Appendix B).

        Self blocks are ``I_{N_τ} ⊗ L_self^τ``. For edge ``(src, r, dst)`` with
        ``L_r ∈ R^{d_src×d_dst}``, the contribution on the flat layout is
        ``Â_{dst←src} ⊗ L_r.T``, matching discrete ``K_eff`` orientation.

        Parameters
        ----------
        edge_indices : sequence of Tensor
            Ordered global relation edge indices.
        num_nodes : int
            Stacked node count ``N``.
        edge_weights : sequence of Tensor or None
            Ordered optional relation weights.
        num_nodes_dict : mapping of str to int
            Per-type node counts.

        Returns
        -------
        Tensor
            Dense generator with shape ``(Σ N_τ·d_τ, Σ N_τ·d_τ)``.
        """
        assert self.latent_dims is not None
        counts = self._validate_num_nodes_dict(num_nodes_dict, num_nodes=num_nodes)
        total = stacked_latent_numel(self.node_types, counts, self.latent_dims)
        slices = latent_type_slices_from_dims(
            self.node_types,
            counts,
            self.latent_dims,
        )
        offsets = node_type_offsets(self.node_types, counts)
        ref = self.l_self_for(self.node_types[0])
        effective = torch.zeros(
            (total, total),
            dtype=ref.dtype,
            device=ref.device,
        )
        for name in self.node_types:
            width = self.latent_dims[name]
            identity = torch.eye(
                counts[name],
                dtype=ref.dtype,
                device=ref.device,
            )
            block = torch.kron(identity, self.l_self_for(name))
            type_slice = slices[name]
            expected_block = (counts[name] * width, counts[name] * width)
            if tuple(block.shape) != expected_block:
                msg = (
                    f"self block for {name!r} has shape {tuple(block.shape)}, "
                    f"expected {expected_block}"
                )
                raise ValueError(msg)
            effective[type_slice, type_slice] = (
                effective[type_slice, type_slice] + block
            )

        for relation_idx, edge_index in enumerate(edge_indices):
            adj = dense_relation_normalized_adjacency(
                edge_index,
                num_nodes,
                edge_weight=edge_weights[relation_idx],
                dtype=ref.dtype,
                normalization=self.normalization,
            )
            src, _rel, dst = self.edge_types[relation_idx]
            src_nodes = slice(offsets[src], offsets[src] + counts[src])
            dst_nodes = slice(offsets[dst], offsets[dst] + counts[dst])
            adj_block = adj[dst_nodes, src_nodes]
            l_rel = self._assembled_relation_matrix(relation_idx)
            coupling = torch.kron(
                adj_block,
                l_rel.transpose(-2, -1).contiguous(),
            )
            effective[slices[dst], slices[src]] = (
                effective[slices[dst], slices[src]] + coupling
            )
        return effective

    def effective_generator(
        self,
        edge_indices: Sequence[Tensor],
        num_nodes: int,
        edge_weights: Sequence[Tensor | None] | None = None,
        *,
        l_self: Tensor | None = None,
        l_self_blocks: Tensor | None = None,
        num_nodes_dict: Mapping[str, int] | None = None,
    ) -> Tensor:
        """Assemble the dense effective generator ``(N·d, N·d)``.

        Builds ``I_N ⊗ L_self + Σ_r Â_r ⊗ L_r`` under the ``vec`` layout
        ``Z.reshape(-1)``. For typed operators the self term is the block
        diagonal ``diag_τ(I_{N_τ} ⊗ L_self^τ)`` and ``edge_indices`` must
        already use stacked global node numbering. This is a dense
        ``O((N·d)^2)`` representation — prefer modest ``N·d``.

        Parameters
        ----------
        edge_indices : sequence of Tensor
            Per-relation edge indices, each ``(2, E_r)``.
        num_nodes : int
            Number of stacked nodes ``N`` (``Σ_τ N_τ`` when typed).
        edge_weights : sequence of Tensor or None, optional
            Optional per-relation edge weights.
        l_self : Tensor or None, optional
            Optional override for a **shared** self generator (used when
            folding a global bilinear term into ``L_self``).
        l_self_blocks : Tensor or None, optional
            Optional per-node self blocks with shape ``(N, d, d)`` (used when
            folding per-node bilinear terms). Mutually exclusive with
            ``l_self``.
        num_nodes_dict : mapping of str to int or None, optional
            Per-type node counts. Required for typed operators unless
            ``l_self`` / ``l_self_blocks`` already supply the self term.

        Returns
        -------
        Tensor
            Dense generator with shape ``(N·d, N·d)``.

        Raises
        ------
        ValueError
            If relation bank lengths mismatch, ``num_nodes`` is invalid, both
            ``l_self`` and ``l_self_blocks`` are set, or a typed operator
            lacks ``num_nodes_dict``.

        Notes
        -----
        When ``l_self`` and ``l_self_blocks`` are both omitted and the
        operator is multiplex, repeated calls with the same relation banks
        reuse an evaluation-scoped ``L_eff`` (see
        :meth:`clear_transition_cache`). Overrides and typed operators skip
        that cache.
        """
        if num_nodes < 1:
            msg = f"num_nodes must be positive, got {num_nodes}"
            raise ValueError(msg)
        if l_self is not None and l_self_blocks is not None:
            msg = "Pass at most one of l_self and l_self_blocks"
            raise ValueError(msg)
        indices, weights = self._resolve_relation_banks(edge_indices, edge_weights)

        if self.is_rectangular:
            if l_self is not None or l_self_blocks is not None:
                msg = (
                    "l_self / l_self_blocks overrides are unsupported for "
                    "rectangular ContinuousHeteroGraphKoopmanOperator"
                )
                raise ValueError(msg)
            if num_nodes_dict is None:
                msg = (
                    "ContinuousHeteroGraphKoopmanOperator.effective_generator "
                    "requires num_nodes_dict when latent_dims is rectangular"
                )
                raise ValueError(msg)
            return self._rectangular_effective_generator(
                indices,
                num_nodes,
                weights,
                num_nodes_dict,
            )

        use_cache = l_self is None and l_self_blocks is None and not self.is_typed
        if self.is_typed and l_self is None and l_self_blocks is None:
            self._require_num_nodes_dict(
                num_nodes_dict,
                num_nodes=num_nodes,
                caller="ContinuousHeteroGraphKoopmanOperator.effective_generator",
            )
            assert num_nodes_dict is not None
            l_self_blocks = self.typed_l_self_blocks(num_nodes_dict)

        if l_self is not None:
            dtype_ref = l_self
        elif l_self_blocks is not None:
            dtype_ref = l_self_blocks[0]
        else:
            dtype_ref = self.L_self

        if use_cache:
            cached = self._lookup_cached_generator(
                indices,
                weights,
                num_nodes,
                dtype_ref.dtype,
                dtype_ref.device,
            )
            if cached is not None:
                return cached

        relation = self._relation_coupling_generator(
            indices,
            num_nodes,
            weights,
            dtype=dtype_ref.dtype,
            device=dtype_ref.device,
        )
        if l_self_blocks is None:
            self_matrix = self.L_self if l_self is None else l_self
            identity = torch.eye(
                num_nodes,
                dtype=relation.dtype,
                device=relation.device,
            )
            generator = torch.kron(identity, self_matrix) + relation
        else:
            expected = (num_nodes, self.latent_dim, self.latent_dim)
            if l_self_blocks.shape != expected:
                msg = (
                    f"l_self_blocks must have shape {expected}, "
                    f"got {tuple(l_self_blocks.shape)}"
                )
                raise ValueError(msg)
            self_blocks = torch.block_diag(*l_self_blocks.unbind(0))
            generator = self_blocks + relation

        if use_cache:
            self._leff_cache.append(
                (
                    indices,
                    weights,
                    num_nodes,
                    generator.dtype,
                    generator.device,
                    generator,
                )
            )
        return generator

    def spectrum(
        self,
        edge_indices: Sequence[Tensor],
        num_nodes: int,
        *,
        edge_weights: Sequence[Tensor | None] | None = None,
        num_nodes_dict: Mapping[str, int] | None = None,
    ) -> KoopmanSpectrum:
        """Eigendecomposition of the effective ``N·d`` generator.

        Parameters
        ----------
        edge_indices : sequence of Tensor
            Topology used to build the adjacency factors.
        num_nodes : int
            Stacked node count ``N``.
        edge_weights : sequence of Tensor or None, optional
            Optional per-relation edge weights.
        num_nodes_dict : mapping of str to int or None, optional
            Per-type node counts; required for typed operators.

        Returns
        -------
        KoopmanSpectrum
            Magnitude-sorted spectrum of :meth:`effective_generator`.
        """
        return compute_generator_spectrum(
            self.effective_generator(
                edge_indices,
                num_nodes,
                edge_weights=edge_weights,
                num_nodes_dict=num_nodes_dict,
            )
        )

    def transition_matrix(
        self,
        delta_t: float | Tensor,
        edge_indices: Sequence[Tensor],
        num_nodes: int,
        edge_weights: Sequence[Tensor | None] | None = None,
        *,
        num_nodes_dict: Mapping[str, int] | None = None,
    ) -> Tensor:
        """Return ``exp(L_eff Δt)`` for the dense effective generator.

        Within an evaluation, repeated multiplex calls with the same
        relation banks and scalar ``Δt`` reuse a cached ``Φ``; distinct
        ``Δt`` values reuse a cached ``L_eff`` (see
        :meth:`clear_transition_cache`). Typed operators never cache (see
        class docstring).

        Parameters
        ----------
        delta_t : float or Tensor
            Integration interval.
        edge_indices : sequence of Tensor
            Per-relation edge indices, each ``(2, E_r)``.
        num_nodes : int
            Number of stacked nodes ``N``.
        edge_weights : sequence of Tensor or None, optional
            Optional per-relation edge weights.
        num_nodes_dict : mapping of str to int or None, optional
            Per-type node counts; required for typed operators.

        Returns
        -------
        Tensor
            Dense transition matrix with shape ``(N·d, N·d)``.
        """
        indices, weights = self._resolve_relation_banks(edge_indices, edge_weights)
        reference = self._self_modules()[0].L
        dtype = reference.dtype
        device = reference.device
        delta = torch.as_tensor(delta_t, dtype=dtype, device=device)
        delta_value = float(delta.detach().reshape(-1)[0].item())

        if not self.is_typed:
            cached_phi = self._lookup_cached_transition(
                indices, weights, delta_value, dtype, device
            )
            if cached_phi is not None:
                return cached_phi

        generator = self.effective_generator(
            indices,
            num_nodes,
            edge_weights=weights,
            num_nodes_dict=num_nodes_dict,
        )
        phi = torch.linalg.matrix_exp(generator * delta)
        if not self.is_typed:
            self._phi_cache.append(
                (indices, weights, delta_value, generator.dtype, generator.device, phi)
            )
        return phi

    def _networked_control_matrix(self, num_nodes: int) -> Tensor:
        """Build ``B_eff`` with shape ``(C, N·d)`` for global additive control.

        Parameters
        ----------
        num_nodes : int
            Node count ``N``.

        Returns
        -------
        Tensor
            Stacked control matrix ``[B, B, ..., B]``.

        Raises
        ------
        ValueError
            If the operator is uncontrolled.
        """
        if self.control_dim <= 0 or self.B is None:
            msg = "control matrix requested for an uncontrolled operator"
            raise ValueError(msg)
        return self.B.repeat(1, num_nodes)

    # -- advance / inverse ----------------------------------------------

    def _advance_dense(
        self,
        z: Tensor,
        delta_t: Tensor,
        *,
        control: Tensor | None,
        edge_indices: Sequence[Tensor],
        edge_weights: Sequence[Tensor | None],
        num_nodes_dict: Mapping[str, int] | None,
    ) -> Tensor:
        """Dense ``N·d`` matrix-exponential advance (with optional Van Loan).

        Parameters
        ----------
        z : Tensor
            Latent node states ``(num_nodes, latent_dim)``.
        delta_t : Tensor
            Integration interval.
        control : Tensor or None
            Optional control input (multiplex only; typed rejects control).
        edge_indices : sequence of Tensor
            Ordered relation edge indices.
        edge_weights : sequence of Tensor or None
            Ordered optional relation weights.
        num_nodes_dict : mapping of str to int or None
            Per-type node counts; required for typed operators.

        Returns
        -------
        Tensor
            Advanced latents with the same shape as ``z``.

        Raises
        ------
        ValueError
            If control is invalid for the configured mode.
        """
        num_nodes = z.shape[0]
        flat = z.reshape(1, -1)

        l_self_override: Tensor | None = None
        l_self_blocks: Tensor | None = None
        if (
            not self.is_typed
            and self.control_dim > 0
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

        # Uncontrolled path (always taken for typed operators, which forbid
        # control): reuse the evaluation-scoped Φ when possible.
        if self.control_dim == 0 and l_self_override is None and l_self_blocks is None:
            if control is not None:
                msg = "control input provided to an uncontrolled operator"
                raise ValueError(msg)
            transition = self.transition_matrix(
                delta_t,
                edge_indices,
                num_nodes,
                edge_weights=edge_weights,
                num_nodes_dict=num_nodes_dict,
            )
            return (flat @ transition.T).view_as(z)

        generator = self.effective_generator(
            edge_indices,
            num_nodes,
            edge_weights=edge_weights,
            l_self=l_self_override,
            l_self_blocks=l_self_blocks,
            num_nodes_dict=num_nodes_dict,
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
            phi11, _ = van_loan_factors(generator, b_eff, delta_t)
            free = (flat @ phi11.T).view_as(z)
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
        num_nodes_dict: Mapping[str, int] | None,
    ) -> Tensor:
        """Self-dominated approximate advance via per-type ``exp(L_self^τ Δt)``.

        Ignores relation factors (documented self-only shortcut; exact when
        relation generators are zero).

        Parameters
        ----------
        z : Tensor
            Latent node states ``(num_nodes, latent_dim)``.
        delta_t : float or Tensor
            Integration interval.
        control : Tensor or None
            Optional control input (multiplex only).
        num_nodes_dict : mapping of str to int or None
            Per-type node counts; required for typed operators.

        Returns
        -------
        Tensor
            Advanced latents with the same shape as ``z``.
        """
        if not self.is_typed:
            return self._self.advance(z, delta_t, control=control)
        counts = self._require_num_nodes_dict(
            num_nodes_dict,
            num_nodes=int(z.shape[0]),
            caller="ContinuousHeteroGraphKoopmanOperator._advance_block_diagonal",
        )
        assert counts is not None
        blocks: list[Tensor] = []
        cursor = 0
        for name in self.node_types:
            stop = cursor + counts[name]
            blocks.append(
                self.self_operator_for(name).advance(
                    z[cursor:stop], delta_t, control=control
                )
            )
            cursor = stop
        return torch.cat(blocks, dim=0)

    def forward(
        self,
        z: Tensor,
        delta_t: float | Tensor,
        edge_indices: Sequence[Tensor],
        edge_weights: Sequence[Tensor | None] | None = None,
        control: Tensor | None = None,
        num_nodes_dict: Mapping[str, int] | None = None,
    ) -> Tensor:
        """Advance latents over ``Δt`` with relation-coupled continuous dynamics.

        Parameters
        ----------
        z : Tensor
            Shared-d: stacked latents ``(num_nodes, latent_dim)``. Rectangular:
            flat vector of length ``Σ_τ N_τ·d_τ`` (see
            :meth:`pack_typed_latents`).
        delta_t : float or Tensor
            Integration interval (required; ``0`` returns ``z``).
        edge_indices : sequence of Tensor
            Ordered per-relation edge indices, length ``num_relations``.
            Typed operators expect stacked global node numbering.
        edge_weights : sequence of Tensor or None, optional
            Optional per-relation edge weights.
        control : Tensor or None, optional
            Exogenous control when ``control_dim > 0`` (self-term only;
            multiplex operators only). Unsupported when rectangular.
        num_nodes_dict : mapping of str to int or None, optional
            Per-type node counts. Required for typed operators so the
            block-diagonal self term can be sliced from ``z``.

        Returns
        -------
        Tensor
            Advanced latents with the same shape as ``z``.

        Raises
        ------
        ValueError
            If ``z`` shape or relation banks are invalid.
        """
        if self.is_rectangular:
            if control is not None:
                msg = (
                    "control is unsupported for rectangular "
                    "ContinuousHeteroGraphKoopmanOperator"
                )
                raise ValueError(msg)
            if num_nodes_dict is None:
                msg = (
                    "ContinuousHeteroGraphKoopmanOperator.forward requires "
                    "num_nodes_dict when latent_dims is rectangular"
                )
                raise ValueError(msg)
            if z.ndim != 1:
                msg = (
                    "rectangular ContinuousHeteroGraphKoopmanOperator expects "
                    f"flat z with shape (Σ N_τ·d_τ,), got {tuple(z.shape)}; "
                    "use pack_typed_latents(...)"
                )
                raise ValueError(msg)
            indices, weights = self._resolve_relation_banks(
                edge_indices,
                edge_weights,
            )
            delta = torch.as_tensor(delta_t, dtype=z.dtype, device=z.device)
            if bool((delta == 0).all().item()):
                return z
            counts = self._validate_num_nodes_dict(num_nodes_dict)
            num_nodes = sum(counts.values())
            if self.sparsity == "block_diagonal":
                z_by_type = self.unpack_typed_latents(z, counts)
                next_by_type = {
                    name: self.self_operator_for(name).advance(
                        z_by_type[name], delta, control=None
                    )
                    for name in self.node_types
                }
                return self.pack_typed_latents(next_by_type, counts)
            transition = self.transition_matrix(
                delta,
                indices,
                num_nodes,
                edge_weights=weights,
                num_nodes_dict=counts,
            )
            return transition @ z

        if z.ndim != 2 or z.shape[-1] != self.latent_dim:
            msg = (
                "ContinuousHeteroGraphKoopmanOperator expects z with shape "
                f"(num_nodes, {self.latent_dim}), got {tuple(z.shape)}"
            )
            raise ValueError(msg)
        indices, weights = self._resolve_relation_banks(edge_indices, edge_weights)
        delta = torch.as_tensor(delta_t, dtype=z.dtype, device=z.device)
        if bool((delta == 0).all().item()):
            return z

        if self.sparsity == "block_diagonal":
            return self._advance_block_diagonal(
                z, delta, control=control, num_nodes_dict=num_nodes_dict
            )
        return self._advance_dense(
            z,
            delta,
            control=control,
            edge_indices=indices,
            edge_weights=weights,
            num_nodes_dict=num_nodes_dict,
        )

    def advance(
        self,
        z: Tensor,
        delta_t: float | Tensor | None = None,
        *,
        edge_indices: Sequence[Tensor] | None = None,
        edge_weights: Sequence[Tensor | None] | None = None,
        control: Tensor | None = None,
        num_nodes_dict: Mapping[str, int] | None = None,
    ) -> Tensor:
        """Contract advance; requires ``delta_t`` and ``edge_indices``.

        Parameters
        ----------
        z : Tensor
            Stacked latent states ``(num_nodes, latent_dim)``.
        delta_t : float, Tensor, or None, optional
            Integration interval (required).
        edge_indices : sequence of Tensor or None, optional
            Ordered per-relation edge indices (required).
        edge_weights : sequence of Tensor or None, optional
            Optional per-relation edge weights.
        control : Tensor or None, optional
            Optional control input when ``control_dim > 0``.
        num_nodes_dict : mapping of str to int or None, optional
            Per-type node counts; required for typed operators.

        Returns
        -------
        Tensor
            Advanced latents with the same shape as ``z``.

        Raises
        ------
        ValueError
            If ``delta_t`` or ``edge_indices`` is missing.
        """
        if delta_t is None:
            msg = "delta_t is required for ContinuousHeteroGraphKoopmanOperator.advance"
            raise ValueError(msg)
        if edge_indices is None:
            msg = (
                "edge_indices is required for "
                "ContinuousHeteroGraphKoopmanOperator.advance"
            )
            raise ValueError(msg)
        return self.forward(
            z,
            delta_t,
            edge_indices,
            edge_weights=edge_weights,
            control=control,
            num_nodes_dict=num_nodes_dict,
        )

    def _inverse_block_diagonal(
        self,
        z: Tensor,
        delta_t: float | Tensor,
        *,
        control: Tensor | None,
        num_nodes_dict: Mapping[str, int] | None,
    ) -> Tensor:
        """Self-dominated approximate inverse via per-type generators.

        Parameters
        ----------
        z : Tensor
            Latents at ``t+Δt`` with shape ``(num_nodes, latent_dim)``.
        delta_t : float or Tensor
            Integration interval.
        control : Tensor or None
            Control that drove the forward step (multiplex only).
        num_nodes_dict : mapping of str to int or None
            Per-type node counts; required for typed operators.

        Returns
        -------
        Tensor
            Recovered latents with the same shape as ``z``.
        """
        if not self.is_typed:
            return self._self.inverse_advance(z, delta_t, control=control)
        counts = self._require_num_nodes_dict(
            num_nodes_dict,
            num_nodes=int(z.shape[0]),
            caller="ContinuousHeteroGraphKoopmanOperator._inverse_block_diagonal",
        )
        assert counts is not None
        blocks: list[Tensor] = []
        cursor = 0
        for name in self.node_types:
            stop = cursor + counts[name]
            blocks.append(
                self.self_operator_for(name).inverse_advance(
                    z[cursor:stop], delta_t, control=control
                )
            )
            cursor = stop
        return torch.cat(blocks, dim=0)

    def inverse_advance(
        self,
        z: Tensor,
        delta_t: float | Tensor | None = None,
        *,
        control: Tensor | None = None,
        inverse_matrix: Tensor | None = None,
        edge_indices: Sequence[Tensor] | None = None,
        edge_weights: Sequence[Tensor | None] | None = None,
        num_nodes_dict: Mapping[str, int] | None = None,
    ) -> Tensor:
        """Approximate inverse over ``-Δt`` (dense uses ``exp(-L_eff Δt)``).

        ``sparsity="block_diagonal"`` inverts the self-term(s) only.
        Controlled dense inverse uses Van Loan factors of ``L_eff``;
        ``inverse_matrix`` is supported only for uncontrolled dense steps.

        Parameters
        ----------
        z : Tensor
            Latents at ``t+Δt`` with shape ``(num_nodes, latent_dim)``.
        delta_t : float, Tensor, or None, optional
            Integration interval (required).
        control : Tensor or None, optional
            Control that drove the forward step.
        inverse_matrix : Tensor or None, optional
            Optional precomputed effective inverse (dense uncontrolled only).
        edge_indices : sequence of Tensor or None, optional
            Ordered per-relation edge indices (required).
        edge_weights : sequence of Tensor or None, optional
            Optional per-relation edge weights.
        num_nodes_dict : mapping of str to int or None, optional
            Per-type node counts; required for typed operators.

        Returns
        -------
        Tensor
            Recovered latents at ``t``.

        Raises
        ------
        ValueError
            If ``delta_t`` / ``edge_indices`` are missing, shapes are
            invalid, or ``inverse_matrix`` is passed with
            ``sparsity="block_diagonal"`` or a controlled dense step.
        """
        if delta_t is None:
            msg = (
                "delta_t is required for "
                "ContinuousHeteroGraphKoopmanOperator.inverse_advance"
            )
            raise ValueError(msg)
        if edge_indices is None:
            msg = (
                "edge_indices is required for "
                "ContinuousHeteroGraphKoopmanOperator.inverse_advance"
            )
            raise ValueError(msg)
        if self.is_rectangular:
            if control is not None:
                msg = (
                    "control is unsupported for rectangular "
                    "ContinuousHeteroGraphKoopmanOperator.inverse_advance"
                )
                raise ValueError(msg)
            if num_nodes_dict is None:
                msg = (
                    "ContinuousHeteroGraphKoopmanOperator.inverse_advance "
                    "requires num_nodes_dict when latent_dims is rectangular"
                )
                raise ValueError(msg)
            if z.ndim != 1:
                msg = (
                    "rectangular inverse_advance expects flat z with shape "
                    f"(Σ N_τ·d_τ,), got {tuple(z.shape)}"
                )
                raise ValueError(msg)
            indices, weights = self._resolve_relation_banks(
                edge_indices,
                edge_weights,
            )
            counts = self._validate_num_nodes_dict(num_nodes_dict)
            num_nodes = sum(counts.values())
            delta = torch.as_tensor(delta_t, dtype=z.dtype, device=z.device)
            if inverse_matrix is None:
                generator = self.effective_generator(
                    indices,
                    num_nodes,
                    edge_weights=weights,
                    num_nodes_dict=counts,
                )
                inverse_matrix = torch.linalg.matrix_exp(generator * (-delta))
            return inverse_matrix @ z

        if z.ndim != 2 or z.shape[-1] != self.latent_dim:
            msg = (
                "ContinuousHeteroGraphKoopmanOperator.inverse_advance expects "
                f"z with shape (num_nodes, {self.latent_dim}), got "
                f"{tuple(z.shape)}"
            )
            raise ValueError(msg)
        indices, weights = self._resolve_relation_banks(edge_indices, edge_weights)

        if self.sparsity == "block_diagonal":
            if inverse_matrix is not None:
                msg = (
                    "inverse_matrix is only supported for "
                    "ContinuousHeteroGraphKoopmanOperator sparsity='dense'"
                )
                raise ValueError(msg)
            return self._inverse_block_diagonal(
                z, delta_t, control=control, num_nodes_dict=num_nodes_dict
            )

        num_nodes = z.shape[0]
        delta = torch.as_tensor(delta_t, dtype=z.dtype, device=z.device)
        generator = self.effective_generator(
            indices, num_nodes, edge_weights=weights, num_nodes_dict=num_nodes_dict
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
                "ContinuousHeteroGraphKoopmanOperator.inverse_advance"
            )
            raise ValueError(msg)

        b_eff = self._networked_control_matrix(num_nodes)
        phi11, phi12 = van_loan_factors(generator, b_eff, delta)
        if control.ndim == 1:
            offset = control @ phi12.T
            adjusted = flat - offset
        elif control.ndim == 2:
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
