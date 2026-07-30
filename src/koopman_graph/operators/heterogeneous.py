"""Multiplex / typed relational (hetero) discrete Koopman operator."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import torch
from torch import Tensor, nn

from koopman_graph.graph_utils import (
    RELATION_NORMALIZATION_MODES,
    RelationNormalization,
    dense_relation_normalized_adjacency,
    relation_normalized_adjacency_matvec,
)
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
from koopman_graph.operators.graph_types import GraphSparsity
from koopman_graph.spectrum_types import KoopmanSpectrum, compute_spectrum

__all__ = [
    "HeteroGraphKoopmanOperator",
    "RELATION_TYING_MODES",
    "RelationTying",
    "relation_factor_key",
]

EdgeTypeTriple = tuple[str, str, str]
RelationTying = Literal["independent", "basis"]
RELATION_TYING_MODES: frozenset[str] = frozenset({"independent", "basis"})


def _basis_factor_key(basis_index: int) -> str:
    """Return the ModuleDict key for basis matrix ``V_b``.

    Parameters
    ----------
    basis_index : int
        Zero-based basis index ``b``.

    Returns
    -------
    str
        Key ``"b{b}"`` used under ``koopman._basis.*``.
    """
    return f"b{basis_index}"


def _validate_relation_tying(
    relation_tying: str,
    basis_size: int | None,
    *,
    num_relations: int,
) -> tuple[RelationTying, int | None]:
    """Validate ``relation_tying`` / ``basis_size`` combinations.

    Parameters
    ----------
    relation_tying : str
        Requested tying mode.
    basis_size : int or None
        Basis size ``B`` when ``relation_tying="basis"``.
    num_relations : int
        Relation bank count ``|R|``.

    Returns
    -------
    relation_tying : {"independent", "basis"}
        Validated mode.
    basis_size : int or None
        Validated ``B`` (``None`` for independent).

    Raises
    ------
    ValueError
        If the mode is unknown or ``basis_size`` is inconsistent.
    """
    if relation_tying not in RELATION_TYING_MODES:
        msg = (
            "relation_tying must be one of "
            f"{sorted(RELATION_TYING_MODES)}, got {relation_tying!r}"
        )
        raise ValueError(msg)
    if relation_tying == "independent":
        if basis_size is not None:
            msg = (
                "basis_size must be None when relation_tying='independent'; "
                f"got {basis_size!r}"
            )
            raise ValueError(msg)
        return "independent", None
    if basis_size is None:
        msg = "basis_size is required when relation_tying='basis'"
        raise ValueError(msg)
    resolved = int(basis_size)
    if resolved < 1:
        msg = (
            "basis_size must be a positive int when "
            f"relation_tying='basis'; got {basis_size!r}"
        )
        raise ValueError(msg)
    if resolved > num_relations:
        msg = f"basis_size ({resolved}) must be <= num_relations ({num_relations})"
        raise ValueError(msg)
    return "basis", resolved


def relation_factor_key(edge_type: EdgeTypeTriple) -> str:
    """Return the stable ModuleDict key for a ``(src, rel, dst)`` edge type.

    Parameters
    ----------
    edge_type : tuple of str
        Triple ``(src, rel, dst)``.

    Returns
    -------
    str
        Key ``"{src}__{rel}__{dst}"`` used under ``koopman._rel.*``.
    """
    src, rel, dst = edge_type
    return f"{src}__{rel}__{dst}"


def _default_multiplex_node_types() -> tuple[str, ...]:
    """Return the multiplex MVP sole node-type name.

    Returns
    -------
    tuple of str
        ``("node",)``.
    """
    return ("node",)


def _default_multiplex_edge_types(num_relations: int) -> tuple[EdgeTypeTriple, ...]:
    """Return synthetic multiplex edge types ``(node, r{i}, node)``.

    Parameters
    ----------
    num_relations : int
        Number of relation banks ``|R|``.

    Returns
    -------
    tuple of tuple of str
        Ordered ``(src, rel, dst)`` triples.
    """
    return tuple(("node", f"r{i}", "node") for i in range(num_relations))


def _normalize_node_types(
    node_types: Sequence[str] | None,
) -> tuple[str, ...]:
    """Validate and freeze node-type metadata.

    Parameters
    ----------
    node_types : sequence of str or None
        Explicit names, or ``None`` for the multiplex default. One name selects
        the multiplex path; two or more select the typed path with one
        ``K_self`` block per type.

    Returns
    -------
    tuple of str
        Ordered node-type names (stacking order of ``Z``).

    Raises
    ------
    ValueError
        If the sequence is empty or names are empty / duplicated.
    """
    resolved = (
        _default_multiplex_node_types()
        if node_types is None
        else tuple(str(name) for name in node_types)
    )
    if not resolved:
        msg = "node_types must contain at least one node type"
        raise ValueError(msg)
    if any(not name for name in resolved):
        msg = f"node_types entries must be non-empty strings; got {resolved!r}"
        raise ValueError(msg)
    if len(set(resolved)) != len(resolved):
        msg = f"node_types must be unique; got {resolved!r}"
        raise ValueError(msg)
    return resolved


def _normalize_edge_types(
    edge_types: Sequence[Sequence[str]] | None,
    *,
    num_relations: int,
    node_types: tuple[str, ...],
) -> tuple[EdgeTypeTriple, ...]:
    """Validate and freeze ordered edge-type triples.

    Parameters
    ----------
    edge_types : sequence of sequence of str or None
        Explicit ``(src, rel, dst)`` triples, or ``None`` for defaults.
    num_relations : int
        Expected bank length ``|R|``.
    node_types : tuple of str
        Resolved node-type names.

    Returns
    -------
    tuple of tuple of str
        Ordered edge-type triples.

    Raises
    ------
    ValueError
        If length, triple shape, uniqueness, or node-type membership fails, or
        defaults are requested for a typed operator.
    """
    if edge_types is None:
        if node_types == _default_multiplex_node_types():
            return _default_multiplex_edge_types(num_relations)
        if len(node_types) > 1:
            msg = (
                "edge_types is required for typed HeteroGraphKoopmanOperator "
                f"(node_types={node_types!r}); default (src, rel, dst) triples "
                "are defined for a single node type only"
            )
            raise ValueError(msg)
        sole = node_types[0]
        return tuple((sole, f"r{i}", sole) for i in range(num_relations))

    resolved: list[EdgeTypeTriple] = []
    for entry in edge_types:
        if len(entry) != 3:
            msg = (
                "edge_types entries must be (src, rel, dst) triples; "
                f"got {tuple(entry)!r}"
            )
            raise ValueError(msg)
        src, rel, dst = (str(part) for part in entry)
        if not src or not rel or not dst:
            msg = "edge_types entries must use non-empty strings"
            raise ValueError(msg)
        resolved.append((src, rel, dst))

    if len(resolved) != num_relations:
        msg = (
            f"edge_types length ({len(resolved)}) must match "
            f"num_relations ({num_relations})"
        )
        raise ValueError(msg)

    keys = [relation_factor_key(triple) for triple in resolved]
    if len(set(keys)) != len(keys):
        msg = f"edge_types must be unique after key encoding; got {resolved!r}"
        raise ValueError(msg)

    known = set(node_types)
    for triple in resolved:
        src, _rel, dst = triple
        for name in (src, dst):
            if name not in known:
                msg = (
                    f"edge type {triple!r} references node type {name!r} "
                    f"outside node_types {list(node_types)!r}"
                )
                raise ValueError(msg)
    return tuple(resolved)


class HeteroGraphKoopmanOperator(nn.Module):
    """Discrete multiplex / typed Koopman step with per-relation coupling.

    Advances stacked node latents ``Z ∈ R^{N×d}`` via the linear map matching
    :class:`~koopman_graph.operators.GraphKoopmanOperator` row-state /
    left-multiply layout::

        Z_next = Z @ K_self.T + sum_r (Â_r Z) @ K_r.T

    Equivalently, with ``vec(Z)`` the C-order flatten ``Z.reshape(-1)``
    (node blocks of width ``d``, i.e. ``vec(Z^T)`` in column-stacking
    notation)::

        vec(Z_{t+1}) = (I_N ⊗ K_self + sum_r Â_r ⊗ K_r) vec(Z_t)

    With one node type (**multiplex**) ``K_self`` is a single ``d×d`` matrix.
    With two or more node types (**typed**) each type owns its own
    ``K_self^τ ∈ R^{d×d}`` at the same shared latent width ``d``, and the
    self term becomes block-diagonal
    ``diag_τ(I_{N_τ} ⊗ K_self^τ)``. Typed calls stack all types into one
    ``N = Σ_τ N_τ`` block ordered by :attr:`node_types`, take relation banks in
    **global** (offset) node numbering, and require ``num_nodes_dict`` so the
    self blocks can be sliced; see
    :mod:`koopman_graph.data.hetero_layout` for the layout helpers. Per-type
    latent widths ``d_τ`` are not supported.

    where ``Â_r`` is the per-relation degree-normalized adjacency from
    :func:`~koopman_graph.graph_utils.relation_degree_normalize`
    (default ``normalization="rgcn_in_degree"``; Schlichtkrull et al. R-GCN
    in-degree convention — normalization only, not a full paper
    reproduction). Reverse relations are not synthesized.

    There are **no activations** inside :meth:`advance` /
    :meth:`inverse_advance`. Soft/structural ``parameterization`` modes
    (``schur`` / ``lyapunov`` / ``dissipative``) apply to each ``d×d``
    factor independently and do **not** certify the joint spectral radius
    ``ρ(K_eff)`` (Gershgorin / joint spectral radius territory — not a
    whole-network stability guarantee). Prefer factor
    :meth:`bound_metric` for monitoring and the assembled eigenvalue hinge
    (:class:`~koopman_graph.losses.EigenvalueRegularizationLoss`) when
    ``N·d`` is modest.

    :meth:`effective_matrix` / :meth:`spectrum` / topology-aware
    :meth:`spectral_radius` assemble a dense ``(N·d, N·d)`` operator.
    Prefer modest ``N·d`` (same dense-ceiling honesty as networked graph /
    hypergraph operators in ``limitations.rst``). ``sparsity="dense"`` uses
    an exact dense inverse; ``"block_diagonal"`` shares the forward path and
    uses a **self-dominated** inverse that ignores relation terms
    (exact when all ``K_r = 0``). ``"distributed"`` raises at construction.

    Control (additive or bilinear) lives on the self factor only; relation
    factors are uncontrolled, and control is rejected for typed operators.
    Orbit ties and continuous generators are out of scope here.

    Relation tying (R-GCN basis-decomposition motivation — not a full paper
    reproduction) is selected by ``relation_tying``::

        K_r = sum_b a_{r,b} V_b    (relation_tying="basis")

    Independent mode keeps one ``d×d`` factor per edge type under ``_rel``.
    Basis mode stores shared ``V_b`` under ``_basis`` and coefficients
    ``_rel_coeff`` with shape ``(|R|, B)``.

    Attributes
    ----------
    latent_dim : int
        Latent feature dimension ``d`` (shared by all node types).
    num_relations : int
        Number of relation banks ``|R|``.
    node_types : tuple of str
        Ordered node-type names; also the stacking order of ``Z``.
    is_typed : bool
        ``True`` when more than one node type is present (per-type
        ``K_self^τ``).
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
        Shared soft/structural parameterization for all ``d×d`` factors.
    normalization : {"rgcn_in_degree", "random_walk"}
        Per-relation adjacency normalization mode.
    sparsity : {"dense", "block_diagonal", "distributed"}
        Realization mode (forward shared for dense / block_diagonal).
    max_spectral_radius : float
        Stability bound forwarded to factorized matrices.
    """

    def __init__(
        self,
        latent_dim: int,
        num_relations: int,
        *,
        init_mode: InitMode = "identity_noise",
        init_scale: float = 1e-2,
        parameterization: Parameterization = "dense",
        max_spectral_radius: float = 1.0,
        sparsity: GraphSparsity = "dense",
        normalization: RelationNormalization = "rgcn_in_degree",
        control_dim: int = 0,
        control_mode: ControlMode = "additive",
        bilinear_rank: int | None = None,
        node_types: Sequence[str] | None = None,
        edge_types: Sequence[Sequence[str]] | None = None,
        relation_tying: RelationTying = "independent",
        basis_size: int | None = None,
    ) -> None:
        """Initialize self and per-relation Koopman factors.

        Parameters
        ----------
        latent_dim : int
            Latent dimension ``d``.
        num_relations : int
            Number of relation banks (``|R| >= 1``).
        init_mode : {"identity", "identity_noise", "xavier"}, optional
            Initialization for ``K_self``. Relation / basis factors start
            near zero (plus optional noise for ``identity_noise`` /
            ``xavier``).
        init_scale : float, optional
            Noise scale for ``identity_noise`` / relation jitter.
        parameterization : Parameterization, optional
            Shared parameterization for every ``d×d`` factor.
        max_spectral_radius : float, optional
            Spectral bound for soft/structural modes.
        sparsity : {"dense", "block_diagonal", "distributed"}, optional
            ``"dense"`` (default) uses an exact dense ``inverse_advance``.
            ``"block_diagonal"`` keeps the same forward advance and uses a
            self-dominated approximate inverse. ``"distributed"`` is reserved
            and rejected.
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
            Two or more names build one ``K_self^τ`` per type.
        edge_types : sequence of sequence of str or None, optional
            Ordered ``(src, rel, dst)`` triples (length ``num_relations``).
            Defaults to ``(node, r{i}, node)``; required for typed operators.
        relation_tying : {"independent", "basis"}, optional
            Relation-factor tying. ``"basis"`` shares ``B`` matrices
            ``V_b`` with coefficients ``a_{r,b}`` (R-GCN basis-decomposition
            motivation — not a full paper reproduction). Default
            ``"independent"``.
        basis_size : int or None, optional
            Basis size ``B`` when ``relation_tying="basis"``
            (``1 <= B <= |R|``). Must be ``None`` for independent tying.

        Raises
        ------
        ValueError
            If dimensions, ``sparsity``, ``normalization``, tying metadata,
            or type metadata are unsupported, or control is requested for a
            typed operator.
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
        if sparsity == "distributed":
            msg = (
                "HeteroGraphKoopmanOperator sparsity='distributed' is "
                "planned and not implemented"
            )
            raise ValueError(msg)
        if sparsity not in {"dense", "block_diagonal"}:
            msg = (
                "HeteroGraphKoopmanOperator sparsity must be 'dense' or "
                f"'block_diagonal', got {sparsity!r}"
            )
            raise ValueError(msg)
        if normalization not in RELATION_NORMALIZATION_MODES:
            msg = (
                "normalization must be one of "
                f"{sorted(RELATION_NORMALIZATION_MODES)}, got {normalization!r}"
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
                "control is unsupported for typed HeteroGraphKoopmanOperator "
                f"(node_types={resolved_node_types!r}); set control_dim=0"
            )
            raise ValueError(msg)

        self.latent_dim = latent_dim
        self.num_relations = num_relations
        self.node_types = resolved_node_types
        self.is_typed = is_typed
        self.edge_types = resolved_edge_types
        self.relation_tying: RelationTying = resolved_tying
        self.basis_size = resolved_basis_size
        self.init_mode = init_mode
        self.init_scale = init_scale
        self.parameterization = parameterization
        self.max_spectral_radius = max_spectral_radius
        self.sparsity = sparsity
        self.normalization: RelationNormalization = normalization
        self.control_dim = control_dim
        self.control_mode = control_mode
        self.bilinear_rank = bilinear_rank

        def _build_self_factor() -> KoopmanOperator:
            """Return a fresh self-coupling factor at the shared width ``d``.

            Returns
            -------
            KoopmanOperator
                Per-node self factor (typed operators build one per type).
            """
            return KoopmanOperator(
                latent_dim,
                init_mode=init_mode,
                init_scale=init_scale,
                parameterization=parameterization,
                max_spectral_radius=max_spectral_radius,
                control_dim=control_dim,
                control_mode=control_mode,
                bilinear_rank=bilinear_rank,
            )

        def _build_relation_factor() -> KoopmanOperator:
            """Return a fresh uncontrolled ``d×d`` relation / basis factor.

            Returns
            -------
            KoopmanOperator
                Relation bank or shared basis matrix module.
            """
            return KoopmanOperator(
                latent_dim,
                init_mode="identity",
                init_scale=init_scale,
                parameterization=parameterization,
                max_spectral_radius=max_spectral_radius,
                control_dim=0,
            )

        # Multiplex keeps the flat ``_self`` submodule (checkpoint key parity);
        # typed operators key one factor per node type under ``_selves``.
        if is_typed:
            self._selves = nn.ModuleDict(
                {name: _build_self_factor() for name in resolved_node_types}
            )
        else:
            self._self = _build_self_factor()

        if resolved_tying == "independent":
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
            # Identity when B == |R|; otherwise truncated eye (first B columns).
            coeffs = torch.zeros(num_relations, resolved_basis_size)
            eye = torch.eye(num_relations, resolved_basis_size)
            coeffs.copy_(eye)
            self._rel_coeff = nn.Parameter(coeffs)
        self._reset_relation_parameters()

    def self_operator_for(self, node_type: str) -> KoopmanOperator:
        """Return the self-coupling factor module for ``node_type``.

        Parameters
        ----------
        node_type : str
            Node-type name from :attr:`node_types`.

        Returns
        -------
        KoopmanOperator
            Self factor owning ``K_self^τ`` (the sole factor for multiplex).

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
            assert isinstance(module, KoopmanOperator)
            return module
        return self._self

    def _self_modules(self) -> tuple[KoopmanOperator, ...]:
        """Return self-coupling factors in :attr:`node_types` order.

        Returns
        -------
        tuple of KoopmanOperator
            One factor per node type (length 1 for multiplex).
        """
        return tuple(self.self_operator_for(name) for name in self.node_types)

    def k_self_for(self, node_type: str) -> Tensor:
        """Return the assembled ``K_self^τ`` for ``node_type``.

        Parameters
        ----------
        node_type : str
            Node-type name from :attr:`node_types`.

        Returns
        -------
        Tensor
            Self-coupling matrix with shape ``(latent_dim, latent_dim)``.

        Raises
        ------
        KeyError
            If ``node_type`` is not in :attr:`node_types`.
        """
        return self.self_operator_for(node_type).K

    def typed_k_self_blocks(self, num_nodes_dict: Mapping[str, int]) -> Tensor:
        """Return per-node self blocks for the stacked typed layout.

        Parameters
        ----------
        num_nodes_dict : mapping of str to int
            Node count ``N_τ`` for every entry of :attr:`node_types`.

        Returns
        -------
        Tensor
            Stacked blocks with shape ``(N, latent_dim, latent_dim)`` where
            ``N = Σ_τ N_τ``; row ``i`` holds ``K_self^τ`` of the type owning
            stacked node ``i``. Suitable for the ``k_self_blocks`` argument of
            :meth:`effective_matrix`.

        Raises
        ------
        ValueError
            If ``num_nodes_dict`` does not cover :attr:`node_types` exactly.
        """
        counts = self._validate_num_nodes_dict(num_nodes_dict)
        blocks = [
            self.k_self_for(name).expand(counts[name], self.latent_dim, self.latent_dim)
            for name in self.node_types
        ]
        return torch.cat(blocks, dim=0)

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
            return self._validate_num_nodes_dict(
                num_nodes_dict,
                num_nodes=num_nodes,
            )
        if num_nodes_dict is None:
            msg = (
                f"{caller} requires num_nodes_dict for typed "
                "HeteroGraphKoopmanOperator (node_types="
                f"{list(self.node_types)!r}) so per-type K_self blocks can be "
                "sliced from the stacked latent block"
            )
            raise ValueError(msg)
        return self._validate_num_nodes_dict(num_nodes_dict, num_nodes=num_nodes)

    def _relation_modules(self) -> tuple[KoopmanOperator, ...]:
        """Return independent relation factor modules in ``edge_types`` order.

        Returns
        -------
        tuple of KoopmanOperator
            Ordered relation banks (independent tying only).

        Raises
        ------
        ValueError
            If ``relation_tying="basis"``.
        """
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

    def _basis_modules(self) -> tuple[KoopmanOperator, ...]:
        """Return shared basis factor modules ``V_0 … V_{B-1}``.

        Returns
        -------
        tuple of KoopmanOperator
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
        """Assemble ``K_r`` for independent or basis tying.

        Parameters
        ----------
        relation_index : int
            Relation bank index in ``[0, num_relations)``.

        Returns
        -------
        Tensor
            Assembled ``K_r`` with shape ``(latent_dim, latent_dim)``.

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
        if self.relation_tying == "independent":
            return self._relation_modules()[relation_index].K
        assert self.basis_size is not None
        assembled = self._rel_coeff.new_zeros(self.latent_dim, self.latent_dim)
        for basis_idx, module in enumerate(self._basis_modules()):
            weight = self._rel_coeff[relation_index, basis_idx]
            assembled = assembled + weight * module.K
        return assembled

    def _reset_factor_parameters(
        self,
        module: KoopmanOperator,
        *,
        allow_noise: bool,
    ) -> None:
        """Zero a relation factor, optionally adding ``init_scale`` noise.

        Parameters
        ----------
        module : KoopmanOperator
            Relation factor module to reset.
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

    def _reset_relation_parameters(self) -> None:
        """Initialize relation / basis factors near zero (optional jitter).

        Returns
        -------
        None
        """
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
        if self.relation_tying == "independent":
            for module in self._relation_modules():
                module.reset_parameters()
        else:
            for module in self._basis_modules():
                module.reset_parameters()
        self._reset_relation_parameters()

    @property
    def K_self(self) -> Tensor:
        """Self-coupling matrix with shape ``(latent_dim, latent_dim)``.

        Returns
        -------
        Tensor
            Assembled ``K_self`` (multiplex only).

        Raises
        ------
        ValueError
            If the operator is typed; there is no single shared ``K_self``.
            Use :meth:`k_self_for` for one type or
            :meth:`typed_k_self_blocks` for the stacked block-diagonal form.
        """
        if self.is_typed:
            msg = (
                "K_self is undefined for typed HeteroGraphKoopmanOperator "
                f"(node_types={list(self.node_types)!r}); use "
                "k_self_for(node_type) or "
                "typed_k_self_blocks(num_nodes_dict)"
            )
            raise ValueError(msg)
        return self._self.K

    @property
    def K_relations(self) -> tuple[Tensor, ...]:
        """Ordered relation-coupling matrices, each ``(latent_dim, latent_dim)``.

        Returns
        -------
        tuple of Tensor
            Assembled ``K_r`` for ``r = 0 … |R|-1`` (basis-combined when
            ``relation_tying="basis"``).
        """
        return tuple(
            self._assembled_relation_matrix(relation_idx)
            for relation_idx in range(self.num_relations)
        )

    def relation_matrix(self, relation_index: int) -> Tensor:
        """Return the ``relation_index``-th relation factor ``K_r``.

        Parameters
        ----------
        relation_index : int
            Relation bank index in ``[0, num_relations)``.

        Returns
        -------
        Tensor
            Assembled ``K_r`` with shape ``(latent_dim, latent_dim)``.

        Raises
        ------
        IndexError
            If ``relation_index`` is out of range.
        """
        return self._assembled_relation_matrix(relation_index)

    @property
    def matrix(self) -> Tensor:
        """Self-term matrix (not the networked ``N·d`` operator).

        Returns
        -------
        Tensor
            ``K_self``. Use :meth:`effective_matrix` for the full map.
        """
        return self.K_self

    @property
    def K(self) -> Tensor:
        """Alias of :attr:`matrix` (``K_self``).

        Returns
        -------
        Tensor
            ``K_self``.
        """
        return self.K_self

    def set_dense_matrices(
        self,
        k_self: Tensor | Mapping[str, Tensor],
        k_relations: Sequence[Tensor],
        *,
        control_matrix: Tensor | None = None,
        bilinear_matrices: Tensor | None = None,
    ) -> None:
        """Write dense self / relation factors (and optional control).

        Parameters
        ----------
        k_self : Tensor or mapping of str to Tensor
            Dense self matrix ``(latent_dim, latent_dim)`` for multiplex, or a
            mapping ``node_type -> (latent_dim, latent_dim)`` covering every
            entry of :attr:`node_types` for typed operators.
        k_relations : sequence of Tensor
            Dense relation matrices, length ``num_relations``, each
            ``(latent_dim, latent_dim)``.
        control_matrix : Tensor or None, optional
            Control matrix ``B`` when ``control_dim > 0``.
        bilinear_matrices : Tensor or None, optional
            Full-rank bilinear stack when ``control_mode="bilinear"``.

        Raises
        ------
        ValueError
            If ``relation_tying="basis"``, the relation bank length mismatches
            ``num_relations``, or the ``k_self`` form does not match the
            multiplex / typed mode.
        """
        if self.relation_tying == "basis":
            msg = (
                "set_dense_matrices is unsupported when "
                "relation_tying='basis'; use set_basis_factors(...) for "
                "V_b / a_{r,b}, and self_operator_for(...).set_dense_matrix(...) "
                "for K_self"
            )
            raise ValueError(msg)
        if len(k_relations) != self.num_relations:
            msg = (
                f"Expected {self.num_relations} relation matrices, "
                f"got {len(k_relations)}"
            )
            raise ValueError(msg)
        if self.is_typed:
            if not isinstance(k_self, Mapping):
                msg = (
                    "typed HeteroGraphKoopmanOperator.set_dense_matrices "
                    "requires a mapping node_type -> K_self^tau for node_types "
                    f"{list(self.node_types)!r}"
                )
                raise ValueError(msg)
            if set(k_self) != set(self.node_types):
                msg = (
                    "k_self keys must match node_types "
                    f"{list(self.node_types)!r}; got {sorted(k_self)!r}"
                )
                raise ValueError(msg)
            for name in self.node_types:
                self.self_operator_for(name).set_dense_matrix(
                    k_self[name],
                    control_matrix=None,
                )
        else:
            if isinstance(k_self, Mapping):
                msg = (
                    "multiplex HeteroGraphKoopmanOperator.set_dense_matrices "
                    "requires a single dense K_self tensor, not a mapping"
                )
                raise ValueError(msg)
            self._self.set_dense_matrix(
                k_self,
                control_matrix=control_matrix,
                bilinear_matrices=bilinear_matrices,
            )
        for module, k_rel in zip(self._relation_modules(), k_relations, strict=True):
            module.set_dense_matrix(k_rel, control_matrix=None)

    def set_basis_factors(
        self,
        basis_matrices: Sequence[Tensor],
        coefficients: Tensor,
    ) -> None:
        """Write dense basis matrices ``V_b`` and coefficients ``a_{r,b}``.

        Parameters
        ----------
        basis_matrices : sequence of Tensor
            Dense basis matrices, length ``basis_size``, each
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

        This is a **factor-level** surrogate, not
        ``ρ(K_eff)``. Prefer topology-aware :meth:`spectral_radius` /
        :meth:`effective_matrix` for joint analysis. Under basis tying the
        bound covers the shared ``V_b`` factors (not assembled ``K_r``).

        Returns
        -------
        Tensor
            Scalar factor bound metric.
        """
        if self.relation_tying == "independent":
            modules = (*self._self_modules(), *self._relation_modules())
        else:
            modules = (*self._self_modules(), *self._basis_modules())
        metric = modules[0].bound_metric()
        for module in modules[1:]:
            metric = torch.maximum(metric, module.bound_metric())
        return metric

    def spectral_radius(
        self,
        edge_indices: Sequence[Tensor] | None = None,
        num_nodes: int | None = None,
        *,
        edge_weights: Sequence[Tensor | None] | None = None,
        num_nodes_dict: Mapping[str, int] | None = None,
    ) -> Tensor:
        """Return assembled ``ρ(K_eff)`` (Q6); topology is required.

        Unlike the homogeneous graph operator's factor-only helper, this
        method does **not** silently return ``ρ(K_self)``. Pass
        ``edge_indices`` and ``num_nodes`` to assemble ``K_eff`` and take
        ``max(|λ|)``. For factor-level monitoring without topology, use
        :meth:`bound_metric`.

        Parameters
        ----------
        edge_indices : sequence of Tensor or None
            Ordered per-relation edge indices (required).
        num_nodes : int or None
            Node count ``N`` (required).
        edge_weights : sequence of Tensor or None, optional
            Optional per-relation edge weights.
        num_nodes_dict : mapping of str to int or None, optional
            Per-type node counts; required for typed operators.

        Returns
        -------
        Tensor
            Scalar spectral radius of the assembled effective operator.

        Raises
        ------
        ValueError
            If topology arguments are omitted (avoids a ``K_self``-only
            surprise).
        """
        if edge_indices is None or num_nodes is None:
            msg = (
                "HeteroGraphKoopmanOperator.spectral_radius requires "
                "edge_indices and num_nodes to return assembled ρ(K_eff); "
                "use bound_metric() for factor-level monitoring without "
                "topology"
            )
            raise ValueError(msg)
        effective = self.effective_matrix(
            edge_indices,
            num_nodes,
            edge_weights=edge_weights,
            num_nodes_dict=num_nodes_dict,
        )
        return torch.linalg.eigvals(effective).abs().max().real

    def stability_certificate(self) -> StabilityCertificate | None:
        """Return the self-term certificate when a structural mode is active.

        Typed operators report the certificate of the **first** node type in
        :attr:`node_types`; all self factors share ``parameterization``, and
        factor-level certificates never certify joint ``ρ(K_eff)``.

        Returns
        -------
        StabilityCertificate or None
            Certificate from a self-coupling factor, if any.
        """
        return self._self_modules()[0].stability_certificate()

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

    def _sparse_relation_term(
        self,
        z: Tensor,
        edge_indices: Sequence[Tensor],
        edge_weights: Sequence[Tensor | None],
    ) -> Tensor:
        """Accumulate ``sum_r (Â_r Z) @ K_r.T`` with sparse matvecs.

        Parameters
        ----------
        z : Tensor
            Latent states ``(N, d)``.
        edge_indices : sequence of Tensor
            Ordered relation edge indices.
        edge_weights : sequence of Tensor or None
            Ordered optional relation weights.

        Returns
        -------
        Tensor
            Relation coupling contribution ``(N, d)``.
        """
        contribution = z.new_zeros(z.shape)
        for relation_idx, edge_index in enumerate(edge_indices):
            aggregated = relation_normalized_adjacency_matvec(
                edge_index,
                z,
                edge_weight=edge_weights[relation_idx],
                num_nodes=z.size(0),
                normalization=self.normalization,
            )
            k_rel = self._assembled_relation_matrix(relation_idx)
            contribution = contribution + aggregated @ k_rel.transpose(-2, -1)
        return contribution

    def _typed_self_term(
        self,
        z: Tensor,
        counts: Mapping[str, int] | None,
    ) -> Tensor:
        """Apply per-type ``K_self^τ`` to the matching stacked row block.

        Parameters
        ----------
        z : Tensor
            Stacked latents ``(N, d)`` ordered by :attr:`node_types`.
        counts : mapping of str to int or None
            Validated per-type node counts (never ``None`` for typed calls).

        Returns
        -------
        Tensor
            Block-diagonal self contribution with the same shape as ``z``.
        """
        assert counts is not None
        blocks: list[Tensor] = []
        cursor = 0
        for name in self.node_types:
            stop = cursor + int(counts[name])
            k_self = self.k_self_for(name)
            blocks.append(z[cursor:stop] @ k_self.transpose(-2, -1))
            cursor = stop
        return torch.cat(blocks, dim=0)

    def _relation_coupling_matrix(
        self,
        edge_indices: Sequence[Tensor],
        num_nodes: int,
        edge_weights: Sequence[Tensor | None],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        """Assemble ``Σ_r Â_r ⊗ K_r``.

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
        """
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
            k_rel = self._assembled_relation_matrix(relation_idx)
            coupling = coupling + torch.kron(adj, k_rel)
        return coupling

    def effective_matrix(
        self,
        edge_indices: Sequence[Tensor],
        num_nodes: int,
        edge_weights: Sequence[Tensor | None] | None = None,
        *,
        k_self: Tensor | None = None,
        k_self_blocks: Tensor | None = None,
        num_nodes_dict: Mapping[str, int] | None = None,
    ) -> Tensor:
        """Assemble the dense effective operator ``(N·d, N·d)``.

        Builds ``I_N ⊗ K_self + Σ_r Â_r ⊗ K_r`` under the same ``vec`` layout
        as :meth:`forward` (``Z.reshape(-1)``). For typed operators the self
        term is the block diagonal ``diag_τ(I_{N_τ} ⊗ K_self^τ)`` and
        ``edge_indices`` must already use stacked global node numbering. This
        is a dense ``O((N·d)^2)`` representation — prefer modest ``N·d`` (see
        networked dense-ceiling notes in ``limitations.rst``).

        Parameters
        ----------
        edge_indices : sequence of Tensor
            Per-relation edge indices, each ``(2, E_r)``.
        num_nodes : int
            Number of stacked nodes ``N`` (``Σ_τ N_τ`` when typed).
        edge_weights : sequence of Tensor or None, optional
            Optional per-relation edge weights.
        k_self : Tensor or None, optional
            Optional override for a **shared** self-coupling matrix (used when
            folding a global bilinear term into ``K_self`` for inversion).
        k_self_blocks : Tensor or None, optional
            Optional per-node self blocks with shape ``(N, d, d)`` (used when
            folding per-node bilinear terms). Mutually exclusive with
            ``k_self``.
        num_nodes_dict : mapping of str to int or None, optional
            Per-type node counts. Required for typed operators unless
            ``k_self`` / ``k_self_blocks`` already supply the self term.

        Returns
        -------
        Tensor
            Dense matrix with shape ``(N·d, N·d)``.

        Raises
        ------
        ValueError
            If relation bank lengths mismatch, ``num_nodes`` is invalid, both
            ``k_self`` and ``k_self_blocks`` are set, or a typed operator lacks
            ``num_nodes_dict``.
        """
        if num_nodes < 1:
            msg = f"num_nodes must be positive, got {num_nodes}"
            raise ValueError(msg)
        if k_self is not None and k_self_blocks is not None:
            msg = "Pass at most one of k_self and k_self_blocks"
            raise ValueError(msg)
        indices, weights = self._resolve_relation_banks(edge_indices, edge_weights)
        if self.is_typed and k_self is None and k_self_blocks is None:
            self._require_num_nodes_dict(
                num_nodes_dict,
                num_nodes=num_nodes,
                caller="HeteroGraphKoopmanOperator.effective_matrix",
            )
            assert num_nodes_dict is not None
            k_self_blocks = self.typed_k_self_blocks(num_nodes_dict)
        if k_self is not None:
            self_matrix = k_self
        elif self.is_typed:
            assert k_self_blocks is not None
            self_matrix = k_self_blocks[0]
        else:
            self_matrix = self.K_self
        relation = self._relation_coupling_matrix(
            indices,
            num_nodes,
            weights,
            dtype=self_matrix.dtype,
            device=self_matrix.device,
        )
        if k_self_blocks is None:
            identity = torch.eye(
                num_nodes,
                dtype=self_matrix.dtype,
                device=self_matrix.device,
            )
            return torch.kron(identity, self_matrix) + relation

        expected = (num_nodes, self.latent_dim, self.latent_dim)
        if k_self_blocks.shape != expected:
            msg = (
                f"k_self_blocks must have shape {expected}, "
                f"got {tuple(k_self_blocks.shape)}"
            )
            raise ValueError(msg)
        self_blocks = torch.block_diag(*k_self_blocks.unbind(0))
        return self_blocks + relation

    def dense_effective_inverse(
        self,
        edge_indices: Sequence[Tensor],
        num_nodes: int,
        *,
        edge_weights: Sequence[Tensor | None] | None = None,
        k_self: Tensor | None = None,
        k_self_blocks: Tensor | None = None,
        num_nodes_dict: Mapping[str, int] | None = None,
    ) -> Tensor:
        """Assemble and invert the dense effective operator.

        Parameters
        ----------
        edge_indices : sequence of Tensor
            Per-relation edge indices.
        num_nodes : int
            Number of nodes ``N``.
        edge_weights : sequence of Tensor or None, optional
            Optional per-relation edge weights.
        k_self : Tensor or None, optional
            Optional shared self-coupling override (see
            :meth:`effective_matrix`).
        k_self_blocks : Tensor or None, optional
            Optional per-node self blocks (see :meth:`effective_matrix`).
        num_nodes_dict : mapping of str to int or None, optional
            Per-type node counts; required for typed operators.

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
            edge_indices,
            num_nodes,
            edge_weights=edge_weights,
            k_self=k_self,
            k_self_blocks=k_self_blocks,
            num_nodes_dict=num_nodes_dict,
        )
        return dense_inverse_or_pinv(effective)

    def spectrum(
        self,
        edge_indices: Sequence[Tensor],
        num_nodes: int,
        *,
        edge_weights: Sequence[Tensor | None] | None = None,
        time_step: float = 1.0,
        num_nodes_dict: Mapping[str, int] | None = None,
    ) -> KoopmanSpectrum:
        """Eigendecomposition of the effective ``N·d`` operator.

        Parameters
        ----------
        edge_indices : sequence of Tensor
            Topology used to build the adjacency factors.
        num_nodes : int
            Stacked node count ``N``.
        edge_weights : sequence of Tensor or None, optional
            Optional per-relation edge weights.
        time_step : float, optional
            Discrete sampling interval for growth rates / frequencies.
        num_nodes_dict : mapping of str to int or None, optional
            Per-type node counts; required for typed operators.

        Returns
        -------
        KoopmanSpectrum
            Spectrum of :meth:`effective_matrix`.
        """
        return compute_spectrum(
            self.effective_matrix(
                edge_indices,
                num_nodes,
                edge_weights=edge_weights,
                num_nodes_dict=num_nodes_dict,
            ),
            time_step,
        )

    def forward(
        self,
        z: Tensor,
        edge_indices: Sequence[Tensor],
        edge_weights: Sequence[Tensor | None] | None = None,
        control: Tensor | None = None,
        num_nodes_dict: Mapping[str, int] | None = None,
    ) -> Tensor:
        """Advance latents with per-relation linear message passing.

        Parameters
        ----------
        z : Tensor
            Stacked latent node states with shape ``(num_nodes, latent_dim)``.
        edge_indices : sequence of Tensor
            Ordered per-relation edge indices, length ``num_relations``. Typed
            operators expect stacked global node numbering.
        edge_weights : sequence of Tensor or None, optional
            Optional per-relation edge weights.
        control : Tensor or None, optional
            Exogenous control when ``control_dim > 0`` (self-term only;
            multiplex operators only).
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
            If ``z`` shape, relation banks, ``num_nodes_dict``, or control
            arguments are invalid.
        """
        if z.ndim != 2:
            msg = (
                "HeteroGraphKoopmanOperator expects z with shape "
                f"(num_nodes, latent_dim), got {tuple(z.shape)}"
            )
            raise ValueError(msg)
        if z.shape[-1] != self.latent_dim:
            msg = (
                f"Expected trailing dimension {self.latent_dim}, "
                f"got shape {tuple(z.shape)}"
            )
            raise ValueError(msg)
        indices, weights = self._resolve_relation_banks(edge_indices, edge_weights)
        counts = self._require_num_nodes_dict(
            num_nodes_dict,
            num_nodes=int(z.shape[0]),
            caller="HeteroGraphKoopmanOperator.forward",
        )
        self_term = (
            self._typed_self_term(z, counts)
            if self.is_typed
            else z @ self.K_self.transpose(-2, -1)
        )
        z_next = self_term + self._sparse_relation_term(
            z,
            indices,
            weights,
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
        edge_indices: Sequence[Tensor] | None = None,
        edge_weights: Sequence[Tensor | None] | None = None,
        control: Tensor | None = None,
        num_nodes_dict: Mapping[str, int] | None = None,
    ) -> Tensor:
        """Contract advance; requires ``edge_indices`` for relation coupling.

        Parameters
        ----------
        z : Tensor
            Stacked latent states ``(num_nodes, latent_dim)``.
        delta_t : float or Tensor or None, optional
            Unused for discrete advance (accepted for contract symmetry).
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
            If ``edge_indices`` is missing.
        """
        del delta_t  # discrete hetero advance ignores Δt
        if edge_indices is None:
            msg = "edge_indices is required for HeteroGraphKoopmanOperator.advance"
            raise ValueError(msg)
        return self.forward(
            z,
            edge_indices,
            edge_weights,
            control=control,
            num_nodes_dict=num_nodes_dict,
        )

    def _bilinear_self_factors(
        self,
        control: Tensor | None,
        num_nodes: int,
    ) -> tuple[Tensor | None, Tensor | None]:
        """Resolve shared / per-node bilinear self overrides.

        Parameters
        ----------
        control : Tensor or None
            Control that drove the forward step.
        num_nodes : int
            Node count ``N``.

        Returns
        -------
        tuple of (Tensor or None, Tensor or None)
            Shared ``k_self`` override and optional per-node blocks.

        Raises
        ------
        ValueError
            If control is missing or has an unsupported shape.
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
        """Recover previous latents from a hetero forward step.

        ``sparsity="dense"`` inverts the effective ``N·d`` map (exact for
        modest ``N``). ``sparsity="block_diagonal"`` uses a **self-dominated**
        approximate inverse that ignores relation coupling (exact when all
        ``K_r = 0``; approximate otherwise). ``inverse_matrix`` is supported
        only for ``sparsity="dense"``.

        For ``control_mode="bilinear"``, global controls fold into a shared
        ``K_self`` override; per-node controls use node-specific bilinear self
        blocks. Relation factors stay uncontrolled.

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
        edge_indices : sequence of Tensor or None, optional
            Required relation topology banks.
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
            If topology / shapes are invalid, or ``inverse_matrix`` is passed
            with ``sparsity="block_diagonal"``.
        """
        from koopman_graph.operators.graph_inverse import (
            apply_self_inverse,
        )

        del delta_t
        if edge_indices is None:
            msg = (
                "edge_indices is required for "
                "HeteroGraphKoopmanOperator.inverse_advance"
            )
            raise ValueError(msg)
        if z.ndim != 2 or z.shape[-1] != self.latent_dim:
            msg = (
                "HeteroGraphKoopmanOperator.inverse_advance expects z with "
                f"shape (num_nodes, {self.latent_dim}), got {tuple(z.shape)}"
            )
            raise ValueError(msg)

        num_nodes = z.shape[0]
        counts = self._require_num_nodes_dict(
            num_nodes_dict,
            num_nodes=int(num_nodes),
            caller="HeteroGraphKoopmanOperator.inverse_advance",
        )

        adjusted = z
        if self.control_dim > 0:
            if control is None:
                msg = "control input is required when control_dim > 0"
                raise ValueError(msg)
            offset = self._self.control_term(control, num_nodes=z.shape[0])
            if control.ndim == 1:
                offset = broadcast_control_term(z, offset, latent_dim=self.latent_dim)
            adjusted = z - offset

        k_self_override, k_self_blocks = self._bilinear_self_factors(
            control,
            num_nodes,
        )
        if self.is_typed and k_self_blocks is None:
            assert counts is not None
            k_self_blocks = self.typed_k_self_blocks(counts)

        if self.sparsity == "block_diagonal":
            if inverse_matrix is not None:
                msg = (
                    "inverse_matrix is only supported for "
                    "HeteroGraphKoopmanOperator sparsity='dense'"
                )
                raise ValueError(msg)
            # Self-dominated path: ignore relation terms (design §3.3).
            if k_self_blocks is not None:
                return apply_self_inverse(adjusted, k_self_blocks=k_self_blocks)
            return apply_self_inverse(
                adjusted,
                k_self=k_self_override if k_self_override is not None else self.K_self,
            )

        if inverse_matrix is None:
            inverse_matrix = self.dense_effective_inverse(
                edge_indices,
                num_nodes,
                edge_weights=edge_weights,
                k_self=k_self_override,
                k_self_blocks=k_self_blocks,
            )

        flat = adjusted.reshape(-1)
        return (inverse_matrix @ flat).view_as(adjusted)
