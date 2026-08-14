"""Shared orbit-tied ``K_self`` bank helpers for networked operators.

Also hosts the MVP ``isotypic`` symmetry path: exact-automorphism orbits for
``K_self`` ties (representation-theoretic ``Aut(G)``, not WL), with the
isotypic decomposition retained for diagnostics and later neighbor-factor
work.
"""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn

from koopman_graph.graph_utils.representation import (
    IsotypicDecomposition,
    compute_isotypic_decomposition,
)
from koopman_graph.graph_utils.symmetry import (
    OrbitMethod,
    OrbitPartition,
    apply_orbit_self,
    assemble_orbit_self_blocks,
    hyperedge_two_section,
    node_orbit_index,
    node_orbit_partition,
    validate_orbit_partition,
)
from koopman_graph.operators.contract import InitMode, Parameterization
from koopman_graph.operators.control import ControlMode
from koopman_graph.operators.discrete import KoopmanOperator


def build_orbit_self_bank(
    *,
    num_orbits: int,
    latent_dim: int,
    init_mode: InitMode,
    init_scale: float,
    parameterization: Parameterization,
    max_spectral_radius: float,
    control_dim: int,
    control_mode: ControlMode,
    bilinear_rank: int | None,
) -> nn.ModuleList:
    """Allocate one self-factor per orbit (control only on orbit 0).

    Parameters
    ----------

    num_orbits : int
        See the function signature / summary for ``num_orbits``.
    latent_dim : int
        See the function signature / summary for ``latent_dim``.
    init_mode : InitMode
        See the function signature / summary for ``init_mode``.
    init_scale : float
        See the function signature / summary for ``init_scale``.
    parameterization : Parameterization
        See the function signature / summary for ``parameterization``.
    max_spectral_radius : float
        See the function signature / summary for ``max_spectral_radius``.
    control_dim : int
        See the function signature / summary for ``control_dim``.
    control_mode : ControlMode
        See the function signature / summary for ``control_mode``.
    bilinear_rank : int | None
        See the function signature / summary for ``bilinear_rank``.

    Returns
    -------

    nn.ModuleList
        See summary line."""
    modules: list[KoopmanOperator] = []
    for orbit_id in range(num_orbits):
        modules.append(
            KoopmanOperator(
                latent_dim,
                init_mode=init_mode,
                init_scale=init_scale,
                parameterization=parameterization,
                max_spectral_radius=max_spectral_radius,
                control_dim=control_dim if orbit_id == 0 else 0,
                control_mode=control_mode,
                bilinear_rank=bilinear_rank if orbit_id == 0 else None,
            )
        )
    return nn.ModuleList(modules)


class OrbitTiedSelfMixin:
    """Mixin adding orbit-tied self factors to networked operators.

    Notes
    -----
    Host must define latent/control construction fields used by
    :meth:`_allocate_orbit_selves`."""

    latent_dim: int
    init_mode: InitMode
    init_scale: float
    parameterization: Parameterization
    max_spectral_radius: float
    control_dim: int
    control_mode: ControlMode
    bilinear_rank: int | None
    _self: KoopmanOperator
    auto_orbits: bool
    orbit_method: OrbitMethod
    orbit_partition: OrbitPartition | None
    isotypic_symmetry: bool
    isotypic_decomposition: IsotypicDecomposition | None
    _orbit_selves: nn.ModuleList | None
    _orbit_nbrs: nn.ModuleList | None
    _node_orbit: Tensor | None

    def _init_orbit_config(
        self,
        *,
        orbit_partition: Sequence[Sequence[int]] | None,
        auto_orbits: bool,
        orbit_method: OrbitMethod,
        isotypic_symmetry: bool = False,
    ) -> None:
        """Store symmetry config; allocate immediately when partition is given.

        Parameters
        ----------
        orbit_partition : Sequence[Sequence[int]] | None
            Explicit orbit partition (wins over auto / isotypic binding).
        auto_orbits : bool
            Bind orbits from topology on first use.
        orbit_method : OrbitMethod
            Orbit backend for ``auto_orbits``.
        isotypic_symmetry : bool, optional
            When ``True``, pending bind uses exact ``Aut(G)`` orbits for
            ``K_self`` ties and stores
            :func:`~koopman_graph.graph_utils.compute_isotypic_decomposition`
            (mutually exclusive with an explicit partition or plain
            ``auto_orbits``).

        Raises
        ------
        ValueError
            If symmetry flags conflict or ``orbit_method`` is invalid.
        """
        if orbit_method not in {"auto", "exact"}:
            msg = f"orbit_method must be 'auto' or 'exact', got {orbit_method!r}"
            raise ValueError(msg)
        if isotypic_symmetry and orbit_partition is not None:
            msg = (
                "isotypic_symmetry is mutually exclusive with an explicit "
                "orbit_partition"
            )
            raise ValueError(msg)
        if isotypic_symmetry and auto_orbits:
            msg = (
                "isotypic_symmetry is mutually exclusive with auto_orbits; "
                "use koopman_symmetry='isotypic' alone"
            )
            raise ValueError(msg)
        # Explicit partition always wins for binding; auto_orbits may remain
        # True in checkpoints that recorded a resolved partition from auto.
        self.isotypic_symmetry = bool(isotypic_symmetry)
        self.isotypic_decomposition = None
        self.auto_orbits = bool(auto_orbits) or self.isotypic_symmetry
        self.orbit_method = "exact" if self.isotypic_symmetry else orbit_method
        self.orbit_partition = None
        self._orbit_selves = None
        self._orbit_nbrs = None
        self._node_orbit = None
        if orbit_partition is not None:
            num_nodes = max(max(orbit) for orbit in orbit_partition) + 1
            self.set_orbit_partition(orbit_partition, num_nodes=num_nodes)

    @property
    def uses_orbit_selves(self) -> bool:
        """Return whether per-orbit ``K_self`` factors are active.

        Returns
        -------
        bool
            See summary line."""
        return self._orbit_selves is not None

    def set_orbit_partition(
        self,
        partition: Sequence[Sequence[int]],
        *,
        num_nodes: int,
    ) -> None:
        """Bind an explicit orbit partition and allocate orbit self factors.

        Parameters
        ----------

        partition : Sequence[Sequence[int]]
            See the function signature / summary for ``partition``.
        num_nodes : int
            See the function signature / summary for ``num_nodes``.

        Returns
        -------

        None
            See summary line."""
        validated = validate_orbit_partition(partition, num_nodes)
        self.orbit_partition = validated
        self._node_orbit = node_orbit_index(validated, num_nodes)
        self._orbit_selves = build_orbit_self_bank(
            num_orbits=len(validated),
            latent_dim=self.latent_dim,
            init_mode=self.init_mode,
            init_scale=self.init_scale,
            parameterization=self.parameterization,
            max_spectral_radius=self.max_spectral_radius,
            control_dim=self.control_dim,
            control_mode=self.control_mode,
            bilinear_rank=self.bilinear_rank,
        )
        self._orbit_nbrs = build_orbit_self_bank(
            num_orbits=len(validated),
            latent_dim=self.latent_dim,
            init_mode="identity",
            init_scale=self.init_scale,
            parameterization=self.parameterization,
            max_spectral_radius=self.max_spectral_radius,
            control_dim=0,
            control_mode=self.control_mode,
            bilinear_rank=None,
        )
        # Representative self factor for control / certificates.
        self._self = self._orbit_selves[0]

    def bind_auto_orbits(
        self,
        *,
        num_nodes: int,
        edge_index: Tensor | None = None,
        hyperedge_index: Tensor | None = None,
    ) -> None:
        """Compute orbits from topology when ``auto_orbits`` is enabled.

        Parameters
        ----------

        num_nodes : int
            See the function signature / summary for ``num_nodes``.
        edge_index : Tensor | None
            See the function signature / summary for ``edge_index``.
        hyperedge_index : Tensor | None
            See the function signature / summary for ``hyperedge_index``.

        Returns
        -------

        None
            See summary line.

        Raises
        ------

        RuntimeError
            Raised when inputs are invalid.
        ValueError
            Raised when inputs are invalid."""
        if not self.auto_orbits and not self.isotypic_symmetry:
            msg = "bind_auto_orbits requires auto_orbits=True or isotypic_symmetry"
            raise RuntimeError(msg)
        if self.orbit_partition is not None:
            return
        if edge_index is None and hyperedge_index is None:
            msg = "edge_index or hyperedge_index is required to bind auto_orbits"
            raise ValueError(msg)
        if edge_index is None:
            assert hyperedge_index is not None
            edge_index = hyperedge_two_section(hyperedge_index, num_nodes)
        if self.isotypic_symmetry:
            # Exact Aut(G) path: store isotypic projectors, then tie K_self on
            # the exact automorphism orbits (equivariant diagonal self maps).
            self.isotypic_decomposition = compute_isotypic_decomposition(
                edge_index,
                num_nodes,
                method="automorphism",
            )
            partition = node_orbit_partition(
                edge_index,
                num_nodes,
                method="exact",
            )
        else:
            partition = node_orbit_partition(
                edge_index,
                num_nodes,
                method=self.orbit_method,
            )
        self.set_orbit_partition(partition, num_nodes=num_nodes)

    def ensure_orbit_binding(
        self,
        num_nodes: int,
        *,
        edge_index: Tensor | None = None,
        hyperedge_index: Tensor | None = None,
    ) -> None:
        """Bind auto orbits on first use; validate node count when already bound.

        Parameters
        ----------

        num_nodes : int
            See the function signature / summary for ``num_nodes``.
        edge_index : Tensor | None
            See the function signature / summary for ``edge_index``.
        hyperedge_index : Tensor | None
            See the function signature / summary for ``hyperedge_index``.

        Returns
        -------

        None
            See summary line.

        Raises
        ------

        ValueError
            Raised when inputs are invalid."""
        pending = (self.auto_orbits or self.isotypic_symmetry) and (
            self.orbit_partition is None
        )
        if pending:
            self.bind_auto_orbits(
                num_nodes=num_nodes,
                edge_index=edge_index,
                hyperedge_index=hyperedge_index,
            )
            return
        if self._node_orbit is not None and self._node_orbit.numel() != num_nodes:
            msg = (
                "orbit partition was bound for "
                f"{self._node_orbit.numel()} nodes, got num_nodes={num_nodes}"
            )
            raise ValueError(msg)
        # Same-N topology change must not silently reuse Aut(G) / isotypic state.
        if (
            self.isotypic_symmetry
            and self.orbit_partition is not None
            and edge_index is not None
        ):
            fresh = node_orbit_partition(
                edge_index,
                num_nodes,
                method="exact",
            )
            if fresh != self.orbit_partition:
                msg = (
                    "isotypic orbit partition does not match the current "
                    "topology; rebind with a fresh model or clear the bound "
                    "partition before reuse"
                )
                raise ValueError(msg)
        # Same-N topology change must not silently reuse Aut(G) / isotypic state.
        if (
            self.isotypic_symmetry
            and self.orbit_partition is not None
            and edge_index is not None
        ):
            fresh = node_orbit_partition(
                edge_index,
                num_nodes,
                method="exact",
            )
            if fresh != self.orbit_partition:
                msg = (
                    "isotypic orbit partition does not match the current "
                    "topology; rebind with a fresh model or clear the bound "
                    "partition before reuse"
                )
                raise ValueError(msg)

    def orbit_self_matrices(self) -> list[Tensor]:
        """Return assembled ``K_self`` matrices for each orbit.

        Returns
        -------
        list[Tensor]
            See summary line.

        Raises
        ------
        RuntimeError
            Raised when inputs are invalid."""
        if self._orbit_selves is None:
            msg = "orbit self bank is not allocated"
            raise RuntimeError(msg)
        return [module.K for module in self._orbit_selves]

    def apply_tied_self(self, z: Tensor) -> Tensor:
        """Apply shared or orbit-tied self map to latents.

        Parameters
        ----------

        z : Tensor
            See the function signature / summary for ``z``.

        Returns
        -------

        Tensor
            See summary line."""
        if self._orbit_selves is None or self._node_orbit is None:
            return z @ self._self.K.T
        return apply_orbit_self(z, self.orbit_self_matrices(), self._node_orbit)

    def orbit_nbr_matrices(self) -> list[Tensor]:
        """Return assembled neighbor-factor matrices for each orbit.

        Returns
        -------
        list of Tensor
            Per-orbit ``K_nbr`` (or hyperedge) factors.
        """
        if self._orbit_nbrs is None:
            msg = "orbit neighbor bank is not allocated"
            raise RuntimeError(msg)
        return [module.K for module in self._orbit_nbrs]

    def apply_tied_neighbor(self, neighbor: Tensor) -> Tensor:
        """Apply shared or orbit-tied neighbor map after an adjacency matvec.

        Parameters
        ----------
        neighbor : Tensor
            Mixed neighbor states ``(N, d)``.

        Returns
        -------
        Tensor
            Neighbor contribution after the shared or orbit-tied ``K_nbr``.
        """
        if self._orbit_nbrs is None or self._node_orbit is None:
            hedge = getattr(self, "_hedge", None)
            factor = getattr(self, "_nbr", hedge)
            if factor is None:
                msg = "no neighbor factor is allocated"
                raise RuntimeError(msg)
            return neighbor @ factor.K.T
        return apply_orbit_self(neighbor, self.orbit_nbr_matrices(), self._node_orbit)

    def tied_self_blocks(self, num_nodes: int) -> Tensor | None:
        """Return ``(N, d, d)`` self blocks when orbit-tied, else ``None``.

        Parameters
        ----------

        num_nodes : int
            See the function signature / summary for ``num_nodes``.

        Returns
        -------

        Tensor | None
            See summary line."""
        if self._orbit_selves is None or self._node_orbit is None:
            return None
        return assemble_orbit_self_blocks(
            self.orbit_self_matrices(),
            self._node_orbit,
            num_nodes,
        )

    def reset_orbit_selves(self) -> None:
        """Reset orbit self factors (and control on orbit 0 when present).

        Returns
        -------
        None
            See summary line."""
        if self._orbit_selves is None:
            self._self.reset_parameters()
            if self.control_dim > 0:
                self._self.reset_control_parameters()
            return
        for orbit_id, module in enumerate(self._orbit_selves):
            module.reset_parameters()
            if orbit_id == 0 and self.control_dim > 0:
                module.reset_control_parameters()
        if self._orbit_nbrs is not None:
            for module in self._orbit_nbrs:
                module.reset_parameters()

    def symmetry_config(self) -> dict[str, object] | None:
        """Return a JSON-friendly symmetry config block for checkpoints.

        Returns
        -------
        dict[str, object] | None
            See summary line."""
        if (
            not self.auto_orbits
            and not self.isotypic_symmetry
            and self.orbit_partition is None
        ):
            return None
        partition = (
            [list(orbit) for orbit in self.orbit_partition]
            if self.orbit_partition is not None
            else None
        )
        config: dict[str, object] = {
            "auto_orbits": self.auto_orbits and not self.isotypic_symmetry,
            "orbit_partition": partition,
            "method": self.orbit_method,
        }
        if self.isotypic_symmetry:
            config["symmetry"] = "isotypic"
            if self.isotypic_decomposition is not None:
                config["isotypic_dimensions"] = list(
                    self.isotypic_decomposition.dimensions
                )
                config["group_order"] = self.isotypic_decomposition.group_order
        else:
            # Additive format-1 label for orbit ties (absent on older saves).
            config["symmetry"] = "orbit"
        return config
