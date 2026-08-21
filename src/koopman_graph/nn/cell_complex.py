"""Cell-complex boundary operators and Hodge Laplacians (teaching MVP).

Supports oriented 0-, 1-, and 2-cells plus optional tetrahedral
3-cells. Degrees above :data:`MAX_CELL_COMPLEX_DEGREE` (3) are
refused. That ceiling is the in-repo TDL teaching depth — **not**
TopologicX / TDA ecosystem parity (``TopoX2024``). Boundary ``B_1``
reuses :func:`~koopman_graph.observables.boundary_incidence_b1`; the
combinatorial graph Laplacian on 0-cells reuses
:func:`~koopman_graph.observables.simplicial_one_laplacian_matvec`
(the 0.10 simplicial helper named ``L_1 = B_1 B_1^T`` on node
features — Hodge degree ``L_0`` here).

:func:`order2_cochain_teaching` binds
:class:`~koopman_graph.operators.CochainKoopmanOperator` to a filled
triangle. The operator advances :math:`k\\le 1`; face latents may be
stored. Sheaf restriction maps stay learned-optional on the sheaf
peers (default diagonal).

Orientation convention
----------------------
* **1-cells.** Oriented ``edge_index`` columns ``(i → j)`` use the same
  signed incidence as :func:`~koopman_graph.observables.boundary_incidence_b1`:
  ``+1`` at the tail ``i`` and ``-1`` at the head ``j`` (one column per
  undirected edge).
* **2-cells.** Ordered triangular ``face_index`` columns ``(a, b, c)`` induce
  the boundary cycle ``(a → b) + (b → c) + (c → a)``. Each induced edge
  contributes ``+1`` / ``-1`` according to whether it agrees with the stored
  1-cell orientation.
* **3-cells.** Optional ``tetra_index`` columns ``(a, b, c, d)`` induce
  the four triangular faces of a tetrahedron. This is the teaching
  ceiling, not a general 3-complex library.

Factory ``encoder="cell_complex"`` builds matched
:class:`CellComplexGNNEncoder` / :class:`CellComplexGNNDecoder` peers that
mix with Hodge ``L_0`` (reusing the 0.10 simplicial matvec). ``Data`` inputs
**require** non-empty ``face_index`` (cell-complex identity); missing faces
raise rather than silently behaving like a plain GNN. Checkpoint types are
``cell_enc`` / ``cell_dec``. Latent Koopman operators stay linear.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
from torch_geometric.data import Data

from koopman_graph.nn.gnn import (
    ActivationName,
    BaseGNNModule,
    validate_positive_dims,
)
from koopman_graph.observables import (
    boundary_incidence_b1,
    coerce_face_index,
    simplicial_one_laplacian_matvec,
)

if TYPE_CHECKING:
    from koopman_graph.operators.cochain import (
        BoundaryNilpotencyReport,
        CochainKoopmanOperator,
    )

DecoderFn = Callable[[Tensor, Tensor, Tensor | None], Tensor]

# Teaching MVP: 0-/1-/2-cells, optional 3-cells via tetra_index.
MAX_CELL_COMPLEX_DEGREE = 3


def _validate_degree(k: int) -> None:
    """Refuse degrees outside the teaching MVP range.

    Parameters
    ----------
    k : int
        Requested cell degree.

    Raises
    ------
    ValueError
        If ``k`` is outside ``[0, MAX_CELL_COMPLEX_DEGREE]``.
    """
    if k < 0 or k > MAX_CELL_COMPLEX_DEGREE:
        msg = (
            f"cell-complex degree k must be in "
            f"[0, {MAX_CELL_COMPLEX_DEGREE}], got {k}; "
            f"teaching MVP supports 0-/1-/2-cells and optional 3-cells"
        )
        raise ValueError(msg)


@dataclass
class CellComplex:
    """Oriented cell complex with 0-/1-/2-cells and optional 3-cells.

    Parameters
    ----------
    num_nodes : int
        Number of 0-cells.
    edge_index : Tensor
        Oriented 1-cells with shape ``(2, num_edges)``.
    face_index : Tensor or None, optional
        Ordered triangular 2-cells with shape ``(3, num_faces)``, or
        ``None`` / empty for a pure 1-skeleton.
    tetra_index : Tensor or None, optional
        Ordered tetrahedral 3-cells with shape ``(4, num_tets)``. The
        teaching ceiling is :data:`MAX_CELL_COMPLEX_DEGREE`.
    """

    num_nodes: int
    edge_index: Tensor
    face_index: Tensor | None = None
    tetra_index: Tensor | None = None

    def __post_init__(self) -> None:
        """Validate topology tensors and freeze cloned copies.

        Notes
        -----
        Clones ``edge_index`` and coerces ``face_index`` in place.
        """
        if self.num_nodes < 1:
            msg = f"num_nodes must be >= 1, got {self.num_nodes}"
            raise ValueError(msg)
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            msg = (
                "edge_index must have shape (2, num_edges), "
                f"got {tuple(self.edge_index.shape)}"
            )
            raise ValueError(msg)
        edges = self.edge_index.to(dtype=torch.long).detach().clone()
        if edges.numel() > 0 and (
            bool((edges < 0).any()) or bool((edges >= self.num_nodes).any())
        ):
            msg = (
                f"edge_index node ids must lie in [0, {self.num_nodes}), "
                f"got min={int(edges.min())} max={int(edges.max())}"
            )
            raise ValueError(msg)
        object.__setattr__(self, "edge_index", edges)
        if self.face_index is None:
            faces = torch.empty((3, 0), dtype=torch.long, device=edges.device)
        else:
            faces = coerce_face_index(self.face_index, num_nodes=self.num_nodes)
        object.__setattr__(self, "face_index", faces)
        if self.tetra_index is None:
            tets = torch.empty((4, 0), dtype=torch.long, device=edges.device)
        else:
            tets = self.tetra_index.to(dtype=torch.long).detach().clone()
            if tets.ndim != 2 or tets.shape[0] != 4:
                msg = (
                    "tetra_index must have shape (4, num_tets), "
                    f"got {tuple(tets.shape)}"
                )
                raise ValueError(msg)
        object.__setattr__(self, "tetra_index", tets)
        _validate_undirected_edges_unique(self.edge_index)

    @property
    def num_edges(self) -> int:
        """Number of oriented 1-cells.

        Returns
        -------
        int
            Count of columns in ``edge_index``.
        """
        return int(self.edge_index.shape[1])

    @property
    def num_faces(self) -> int:
        """Number of ordered triangular 2-cells.

        Returns
        -------
        int
            Count of columns in ``face_index``.
        """
        assert self.face_index is not None
        return int(self.face_index.shape[1])

    @property
    def num_tets(self) -> int:
        """Number of optional tetrahedral 3-cells.

        Returns
        -------
        int
            Count of columns in ``tetra_index``, or ``0`` when absent.
        """
        if self.tetra_index is None:
            return 0
        return int(self.tetra_index.shape[1])

    def num_cells(self, k: int) -> int:
        """Return the number of ``k``-cells.

        Parameters
        ----------
        k : int
            Cell degree in ``{0, 1, 2, 3}``.

        Returns
        -------
        int
            Number of ``k``-cells.
        """
        _validate_degree(k)
        if k == 0:
            return self.num_nodes
        if k == 1:
            return self.num_edges
        if k == 2:
            return self.num_faces
        return self.num_tets


def _validate_undirected_edges_unique(edge_index: Tensor) -> None:
    """Reject duplicate undirected edges (ambiguous ``B_2`` signs).

    Parameters
    ----------
    edge_index
        See signature."""
    num_edges = int(edge_index.shape[1])
    seen: set[tuple[int, int]] = set()
    for e in range(num_edges):
        i = int(edge_index[0, e])
        j = int(edge_index[1, e])
        key = (i, j) if i <= j else (j, i)
        if key in seen:
            msg = (
                f"duplicate undirected edge {{{i}, {j}}} in edge_index; "
                f"pass one oriented column per undirected edge"
            )
            raise ValueError(msg)
        seen.add(key)


def _oriented_edge_lookup(
    edge_index: Tensor,
) -> dict[tuple[int, int], tuple[int, float]]:
    """Map oriented node pairs to ``(edge_column, sign)``.

    Parameters
    ----------
    edge_index
        See signature.

    Returns
    -------
        See signature."""
    lookup: dict[tuple[int, int], tuple[int, float]] = {}
    for e in range(int(edge_index.shape[1])):
        i = int(edge_index[0, e])
        j = int(edge_index[1, e])
        lookup[(i, j)] = (e, 1.0)
        lookup[(j, i)] = (e, -1.0)
    return lookup


def boundary_incidence_b2(
    edge_index: Tensor,
    face_index: Tensor,
    *,
    num_nodes: int,
) -> Tensor:
    """Build the signed boundary incidence ``B_2`` with shape ``(E, F)``.

    Column ``f`` for ordered triangle ``(a, b, c)`` encodes the induced cycle
    ``(a → b) + (b → c) + (c → a)`` relative to stored 1-cell orientations.

    Parameters
    ----------
    edge_index : Tensor
        Oriented 1-cells ``(2, E)``.
    face_index : Tensor
        Ordered triangles ``(3, F)``.
    num_nodes : int
        Number of 0-cells (used to validate ``face_index``).

    Returns
    -------
    Tensor
        Dense signed incidence of shape ``(num_edges, num_faces)``.

    Raises
    ------
    ValueError
        If topology is invalid or a face edge is missing from ``edge_index``.
    """
    complex_ = CellComplex(
        num_nodes=num_nodes,
        edge_index=edge_index,
        face_index=face_index,
    )
    return boundary_operator(complex_, 2)


def boundary_operator(complex_: CellComplex, k: int) -> Tensor:
    """Return the dense boundary matrix ``B_k`` of shape ``(n_{k-1}, n_k)``.

    Parameters
    ----------
    complex_ : CellComplex
        Oriented cell complex.
    k : int
        Boundary degree. ``k = 0`` returns an empty ``(0, n_0)`` matrix
        (no ``(-1)``-cells). ``k ∈ {1, 2, 3}`` return ``B_1`` /
        ``B_2`` / ``B_3``. ``k > 3`` is refused.

    Returns
    -------
    Tensor
        Dense ``float32`` boundary operator.

    Raises
    ------
    ValueError
        If ``k`` is outside the teaching MVP range.
    """
    _validate_degree(k)
    device = complex_.edge_index.device
    if k == 0:
        return torch.zeros(
            0,
            complex_.num_nodes,
            dtype=torch.float32,
            device=device,
        )
    if k == 1:
        # Reuse 0.10 simplicial incidence — do not reimplement signs.
        return boundary_incidence_b1(
            complex_.edge_index,
            num_nodes=complex_.num_nodes,
        )
    if k == 3:
        return _boundary_b3(complex_)
    assert complex_.face_index is not None
    num_edges = complex_.num_edges
    num_faces = complex_.num_faces
    incidence = torch.zeros(
        num_edges,
        num_faces,
        dtype=torch.float32,
        device=device,
    )
    if num_faces == 0:
        return incidence
    lookup = _oriented_edge_lookup(complex_.edge_index)
    faces = complex_.face_index
    for f in range(num_faces):
        a = int(faces[0, f])
        b = int(faces[1, f])
        c = int(faces[2, f])
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v)
            if key not in lookup:
                msg = (
                    f"face {f} edge ({u} → {v}) is missing from edge_index; "
                    f"2-cells require their 1-skeleton edges"
                )
                raise ValueError(msg)
            edge_col, sign = lookup[key]
            incidence[edge_col, f] += sign
    return incidence


def _boundary_b3(complex_: CellComplex) -> Tensor:
    """Build ``B_3`` with shape ``(n_faces, n_tets)``.

    Each tetrahedron ``(a, b, c, d)`` contributes the four triangular faces
    ``(a,b,c)``, ``(a,b,d)``, ``(a,c,d)``, ``(b,c,d)`` with alternating sign.
    Missing faces raise.

    Parameters
    ----------
    complex_ : CellComplex
        Oriented cell complex with optional ``tetra_index``.

    Returns
    -------
    Tensor
        Incidence of shape ``(n_faces, n_tets)``.
    """
    device = complex_.edge_index.device
    num_faces = complex_.num_faces
    num_tets = complex_.num_tets
    incidence = torch.zeros(num_faces, num_tets, dtype=torch.float32, device=device)
    if num_tets == 0 or complex_.tetra_index is None:
        return incidence
    faces = complex_.face_index
    assert faces is not None
    lookup: dict[tuple[int, int, int], int] = {}
    for f in range(num_faces):
        key = tuple(sorted((int(faces[0, f]), int(faces[1, f]), int(faces[2, f]))))
        lookup[key] = f  # type: ignore[assignment]
    tets = complex_.tetra_index
    face_sets = (
        (0, 1, 2, 1.0),
        (0, 1, 3, -1.0),
        (0, 2, 3, 1.0),
        (1, 2, 3, -1.0),
    )
    for t in range(num_tets):
        verts = [int(tets[i, t]) for i in range(4)]
        for i, j, k, sign in face_sets:
            key = tuple(sorted((verts[i], verts[j], verts[k])))
            if key not in lookup:
                msg = f"tetra {t} face {key} is missing from face_index"
                raise ValueError(msg)
            incidence[lookup[key], t] += sign  # type: ignore[index]
    return incidence


def hodge_laplacian(complex_: CellComplex, k: int) -> Tensor:
    """Return the dense Hodge Laplacian ``L_k`` of shape ``(n_k, n_k)``.

    Uses the standard combinatorial formula::

        L_k = B_k^T B_k + B_{k+1} B_{k+1}^T

    with ``B_4 = 0``. Empty ``tetra_index`` keeps ``B_3`` empty so ``L_2``
    matches the 2-cell MVP.

    Parameters
    ----------
    complex_ : CellComplex
        Oriented cell complex.
    k : int
        Form degree in ``{0, 1, 2, 3}``.

    Returns
    -------
    Tensor
        Dense ``float32`` Hodge Laplacian.

    Raises
    ------
    ValueError
        If ``k`` is outside ``[0, MAX_CELL_COMPLEX_DEGREE]``.
    """
    _validate_degree(k)
    b_k = boundary_operator(complex_, k)
    down = b_k.transpose(0, 1) @ b_k
    if k == MAX_CELL_COMPLEX_DEGREE:
        # No 3-cells: up-Laplacian vanishes.
        return down
    b_next = boundary_operator(complex_, k + 1)
    up = b_next @ b_next.transpose(0, 1)
    return down + up


def hodge_laplacian_matvec(
    complex_: CellComplex,
    k: int,
    x: Tensor,
) -> Tensor:
    """Apply ``L_k`` to cochain features without assembling when ``k = 0``.

    For ``k = 0``, delegates to
    :func:`~koopman_graph.observables.simplicial_one_laplacian_matvec` so the
    0.10 simplicial path is not duplicated. For ``k ∈ {1, 2}`` uses the dense
    :func:`hodge_laplacian`. For ``k = 3`` uses dense ``L_3``.

    Parameters
    ----------
    complex_ : CellComplex
        Oriented cell complex.
    k : int
        Form degree in ``{0, 1, 2, 3}``.
    x : Tensor
        Cochain features ``(n_k, channels)`` or ``(n_k,)``.

    Returns
    -------
    Tensor
        ``L_k @ x`` with the same shape as ``x``.
    """
    _validate_degree(k)
    if x.ndim == 1:
        x_mat = x.unsqueeze(-1)
        squeeze = True
    elif x.ndim == 2:
        x_mat = x
        squeeze = False
    else:
        msg = f"x must be 1D or 2D, got shape {tuple(x.shape)}"
        raise ValueError(msg)
    n_k = complex_.num_cells(k)
    if x_mat.shape[0] != n_k:
        msg = f"Expected x with leading dim n_{k}={n_k}, got {x_mat.shape[0]}"
        raise ValueError(msg)
    if k == 0:
        # Package 0.10 name ``L_1 = B_1 B_1^T`` on nodes == Hodge L_0 here.
        out = simplicial_one_laplacian_matvec(
            complex_.edge_index,
            x_mat,
            num_nodes=complex_.num_nodes,
        )
    else:
        lap = hodge_laplacian(complex_, k).to(dtype=x_mat.dtype, device=x_mat.device)
        out = lap @ x_mat
    return out.squeeze(-1) if squeeze else out


class CellComplexConv(nn.Module):
    """One Hodge-``L_0`` mixing layer: ``Linear(L_0 @ x)``.

    Parameters
    ----------
    in_channels : int
        Input feature width.
    out_channels : int
        Output feature width.
    residual : bool, optional
        Add a residual skip when widths match. Default is ``False``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        residual: bool = False,
    ) -> None:
        """Initialize the linear map after Hodge ``L_0`` mixing.

        Parameters
        ----------
        in_channels
            See signature.
        out_channels
            See signature.
        residual
            See signature."""
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.residual = bool(residual) and in_channels == out_channels
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Apply Hodge ``L_0`` then a learned linear map.

        Parameters
        ----------
        x
            See signature.
        edge_index
            See signature.

        Returns
        -------
            See signature."""
        mixed = simplicial_one_laplacian_matvec(
            edge_index,
            x,
            num_nodes=int(x.size(0)),
        )
        out = self.lin(mixed)
        if self.residual:
            out = out + x
        return out


def build_cell_complex_convs(
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    num_layers: int,
    *,
    residual: bool = False,
) -> nn.ModuleList:
    """Build a stack of :class:`CellComplexConv` layers.

    Parameters
    ----------
    in_channels
        See signature.
    hidden_channels
        See signature.
    out_channels
        See signature.
    num_layers
        See signature.
    residual
        See signature.

    Returns
    -------
        See signature."""
    if num_layers == 1:
        return nn.ModuleList(
            [CellComplexConv(in_channels, out_channels, residual=residual)]
        )
    layers: list[nn.Module] = [
        CellComplexConv(in_channels, hidden_channels, residual=residual)
    ]
    for _ in range(num_layers - 2):
        layers.append(
            CellComplexConv(hidden_channels, hidden_channels, residual=residual)
        )
    layers.append(CellComplexConv(hidden_channels, out_channels, residual=False))
    return nn.ModuleList(layers)


def _require_face_index(face_index: Tensor | None, *, num_nodes: int) -> Tensor:
    """Require a non-empty triangular ``face_index`` for cell-complex peers.

    Parameters
    ----------
    face_index
        See signature.
    num_nodes
        See signature.

    Returns
    -------
        See signature."""
    if face_index is None:
        msg = (
            "data.face_index is required for CellComplexGNNEncoder / "
            "CellComplexGNNDecoder (cell-complex identity); refusing silent "
            "fallback to a plain GNN encode/decode path"
        )
        raise ValueError(msg)
    faces = coerce_face_index(face_index, num_nodes=num_nodes)
    if int(faces.shape[1]) < 1:
        msg = (
            "data.face_index must contain at least one 2-cell for "
            "CellComplexGNNEncoder / CellComplexGNNDecoder"
        )
        raise ValueError(msg)
    return faces


def _resolve_cell_complex_forward_inputs(
    x_or_data: Tensor | Data,
    edge_index: Tensor | None,
    edge_weight: Tensor | None,
    *,
    require_faces: bool,
) -> tuple[Tensor, Tensor]:
    """Resolve node features and oriented edges; optionally require faces.

    Parameters
    ----------
    x_or_data
        See signature.
    edge_index
        See signature.
    edge_weight
        See signature.
    require_faces
        See signature.

    Returns
    -------
        See signature."""
    del edge_weight  # unused — Hodge L_0 on the 1-skeleton
    if isinstance(x_or_data, Data):
        if x_or_data.x is None:
            msg = "data.x is required for CellComplexGNNEncoder / CellComplexGNNDecoder"
            raise ValueError(msg)
        if x_or_data.edge_index is None:
            msg = (
                "data.edge_index is required for CellComplexGNNEncoder / "
                "CellComplexGNNDecoder"
            )
            raise ValueError(msg)
        num_nodes = int(x_or_data.x.size(0))
        if require_faces:
            faces = _require_face_index(
                getattr(x_or_data, "face_index", None),
                num_nodes=num_nodes,
            )
            # Validate oriented complex (skeleton + faces) before mixing.
            CellComplex(
                num_nodes=num_nodes,
                edge_index=x_or_data.edge_index,
                face_index=faces,
            )
        return x_or_data.x, x_or_data.edge_index
    if edge_index is None:
        msg = (
            "edge_index is required when x_or_data is a tensor for "
            "CellComplexGNNEncoder / CellComplexGNNDecoder"
        )
        raise ValueError(msg)
    return x_or_data, edge_index


def _cell_complex_message_passing(
    module: BaseGNNModule,
    x: Tensor,
    edge_index: Tensor,
) -> Tensor:
    """Run stacked :class:`CellComplexConv` layers with hidden activations.

    Parameters
    ----------
    module
        See signature.
    x
        See signature.
    edge_index
        See signature.

    Returns
    -------
        See signature."""
    if x.ndim != 2:
        msg = (
            f"Expected x with shape (num_nodes, {module.input_dim_name}), "
            f"got {tuple(x.shape)}"
        )
        raise ValueError(msg)
    if x.shape[1] != module.input_channels:
        msg = (
            f"Expected {module.input_dim_name}={module.input_channels}, "
            f"got feature dimension {x.shape[1]}"
        )
        raise ValueError(msg)
    for layer_idx, conv in enumerate(module.convs):
        if not isinstance(conv, CellComplexConv):
            msg = (
                f"expected CellComplexConv in cell-complex stack, "
                f"got {type(conv).__name__}"
            )
            raise TypeError(msg)
        x = conv(x, edge_index)
        if layer_idx < len(module.convs) - 1:
            x = module.activation(x)
    return x


class CellComplexGNNEncoder(BaseGNNModule):
    """Cell-complex encoder mixing with Hodge ``L_0`` on node features.

    Notes
    -----
    See class definition."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        latent_dim: int,
        *,
        num_layers: int = 2,
        activation: ActivationName = "relu",
        residual: bool = False,
    ) -> None:
        """Initialize the cell-complex encoder stack.

        Parameters
        ----------
        in_channels
            See signature.
        hidden_channels
            See signature.
        latent_dim
            See signature.
        num_layers
            See signature.
        activation
            See signature.
        residual
            See signature."""
        validate_positive_dims(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            latent_dim=latent_dim,
            num_layers=num_layers,
        )
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.latent_dim = latent_dim
        self.residual = bool(residual)
        super().__init__(
            input_channels=in_channels,
            input_dim_name="in_channels",
            num_layers=num_layers,
            activation=activation,
            convs=build_cell_complex_convs(
                in_channels,
                hidden_channels,
                latent_dim,
                num_layers,
                residual=residual,
            ),
        )

    def forward(
        self,
        x_or_data: Tensor | Data,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        """Encode node features with Hodge ``L_0`` mixing.

        Parameters
        ----------
        x_or_data
            See signature.
        edge_index
            See signature.
        edge_weight
            See signature.

        Returns
        -------
            See signature."""
        x, resolved_edges = _resolve_cell_complex_forward_inputs(
            x_or_data,
            edge_index,
            edge_weight,
            require_faces=isinstance(x_or_data, Data),
        )
        return _cell_complex_message_passing(self, x, resolved_edges)


class CellComplexGNNDecoder(BaseGNNModule):
    """Cell-complex decoder mixing with Hodge ``L_0`` on latent features.

    Notes
    -----
    See class definition."""

    def __init__(
        self,
        latent_dim: int,
        hidden_channels: int,
        out_channels: int,
        *,
        num_layers: int = 2,
        activation: ActivationName = "relu",
        residual: bool = False,
    ) -> None:
        """Initialize the cell-complex decoder stack.

        Parameters
        ----------
        latent_dim
            See signature.
        hidden_channels
            See signature.
        out_channels
            See signature.
        num_layers
            See signature.
        activation
            See signature.
        residual
            See signature."""
        validate_positive_dims(
            latent_dim=latent_dim,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
        )
        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.residual = bool(residual)
        super().__init__(
            input_channels=latent_dim,
            input_dim_name="latent_dim",
            num_layers=num_layers,
            activation=activation,
            convs=build_cell_complex_convs(
                latent_dim,
                hidden_channels,
                out_channels,
                num_layers,
                residual=residual,
            ),
        )

    def forward(
        self,
        x_or_data: Tensor | Data,
        edge_index: Tensor | None = None,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        """Decode latent features with Hodge ``L_0`` mixing.

        Parameters
        ----------
        x_or_data
            See signature.
        edge_index
            See signature.
        edge_weight
            See signature.

        Returns
        -------
            See signature."""
        x, resolved_edges = _resolve_cell_complex_forward_inputs(
            x_or_data,
            edge_index,
            edge_weight,
            require_faces=isinstance(x_or_data, Data),
        )
        return _cell_complex_message_passing(self, x, resolved_edges)


def bind_cell_complex_decoder(
    decoder: CellComplexGNNDecoder,
    edge_index: Tensor,
    face_index: Tensor,
) -> DecoderFn:
    """Bind static oriented edges (faces validated at bind time).

    Parameters
    ----------
    decoder : CellComplexGNNDecoder
        Cell-complex decoder instance.
    edge_index : Tensor
        Static oriented edges ``(2, num_edges)``.
    face_index : Tensor
        Non-empty triangular faces ``(3, num_faces)``.

    Returns
    -------
    callable
        ``decoder(z, edge_index, edge_weight) -> Tensor`` for rollouts.
    """
    edges = edge_index.to(dtype=torch.long)
    if edges.ndim != 2 or edges.shape[0] != 2:
        msg = f"edge_index must have shape (2, num_edges), got {tuple(edges.shape)}"
        raise ValueError(msg)
    maxima: list[int] = []
    if edges.numel() > 0:
        maxima.append(int(edges.max().item()))
    if face_index is not None and face_index.numel() > 0:
        maxima.append(int(face_index.max().item()))
    num_nodes = max(maxima) + 1 if maxima else 1
    faces = _require_face_index(face_index, num_nodes=num_nodes)
    CellComplex(num_nodes=num_nodes, edge_index=edges, face_index=faces)
    return lambda z, _edge_index, _edge_weight: decoder(z, edges, None)


@dataclass(frozen=True)
class Order2CochainTeaching:
    """Filled triangle bound to a :math:`k\\le 1` cochain operator.

    Attributes
    ----------
    complex : CellComplex
        Oriented 2-simplex (3 nodes, 3 edges, 1 face).
    operator : CochainKoopmanOperator
        Bound to the 1-skeleton. Not a factory kind; ``k=2`` is not
        evolved.
    nilpotency : BoundaryNilpotencyReport
        :math:`B_1 B_2\\approx 0` on this triangle
        (``Lim2020Hodge``).
    max_cell_degree : int
        Always :data:`MAX_CELL_COMPLEX_DEGREE` (3). Documents the
        teaching ceiling, not ecosystem parity (``TopoX2024``).
    """

    complex: CellComplex
    operator: CochainKoopmanOperator
    nilpotency: BoundaryNilpotencyReport
    max_cell_degree: int

    def __post_init__(self) -> None:
        """Validate the documented cell-complex ceiling.

        Raises
        ------
        ValueError
            If ``max_cell_degree`` is not the teaching ceiling.
        """
        if int(self.max_cell_degree) != int(MAX_CELL_COMPLEX_DEGREE):
            msg = (
                "max_cell_degree must equal MAX_CELL_COMPLEX_DEGREE "
                f"{MAX_CELL_COMPLEX_DEGREE}, got {self.max_cell_degree}"
            )
            raise ValueError(msg)


def teaching_order2_triangle() -> CellComplex:
    """Return the oriented filled triangle used as the order-2 path.

    Returns
    -------
    CellComplex
        Three 0-cells, three 1-cells, and one 2-cell. Combinatorial
        incidence only — not a sheaf or TopologicX complex
        (``Lim2020Hodge``, ``TopoX2024``).
    """
    edge_index = torch.tensor([[0, 1, 0], [1, 2, 2]], dtype=torch.long)
    face_index = torch.tensor([[0], [1], [2]], dtype=torch.long)
    return CellComplex(num_nodes=3, edge_index=edge_index, face_index=face_index)


def teaching_order3_tetrahedron() -> CellComplex:
    """Return a filled tetrahedron at the cell-complex degree ceiling.

    Returns
    -------
    CellComplex
        Four 0-cells, six 1-cells, four 2-cells, and one 3-cell.
        :data:`MAX_CELL_COMPLEX_DEGREE` is 3. This is not a general
        3-complex library (``TopoX2024``).
    """
    edge_index = torch.tensor(
        [[0, 1, 0, 0, 1, 2], [1, 2, 2, 3, 3, 3]],
        dtype=torch.long,
    )
    face_index = torch.tensor(
        [[0, 0, 0, 1], [1, 1, 2, 2], [2, 3, 3, 3]],
        dtype=torch.long,
    )
    tetra_index = torch.tensor([[0], [1], [2], [3]], dtype=torch.long)
    return CellComplex(
        num_nodes=4,
        edge_index=edge_index,
        face_index=face_index,
        tetra_index=tetra_index,
    )


def bind_cochain_operator(
    complex_: CellComplex,
    *,
    latent_dim: int,
    use_cross_degree: bool = False,
) -> CochainKoopmanOperator:
    """Bind a :math:`k\\le 1` cochain operator to a cell-complex 1-skeleton.

    Face and tetra incidences stay on the complex for nilpotency
    checks. The operator does not evolve :math:`k=2`. This hook lives
    in ``nn`` because :mod:`koopman_graph.operators.cochain` must not
    import ``nn``.

    Parameters
    ----------
    complex_ : CellComplex
        Oriented complex. ``edge_index`` and ``num_nodes`` bind
        :math:`B_1`.
    latent_dim : int
        Shared node / edge feature width.
    use_cross_degree : bool, optional
        Enable incidence cross terms. Default ``False``.

    Returns
    -------
    CochainKoopmanOperator
        Operator on the 1-skeleton. Not a factory kind.

    Raises
    ------
    TypeError
        If ``complex_`` is not a :class:`CellComplex`.
    ValueError
        If ``latent_dim`` or the 1-skeleton is invalid.
    """
    if not isinstance(complex_, CellComplex):
        msg = f"complex_ must be a CellComplex, got {type(complex_).__name__}"
        raise TypeError(msg)
    from koopman_graph.operators.cochain import CochainKoopmanOperator

    return CochainKoopmanOperator(
        int(latent_dim),
        complex_.edge_index,
        num_nodes=int(complex_.num_nodes),
        use_cross_degree=use_cross_degree,
    )


def cell_complex_boundary_nilpotency(
    complex_: CellComplex,
    *,
    atol: float | None = None,
) -> BoundaryNilpotencyReport:
    """Score :math:`B_1 B_2 \\approx 0` on a cell complex.

    This is the cochain-operator hook for order-2 incidence. It does
    not score :math:`B_2 B_3`; tetrahedra are ceiling fixtures only.

    Parameters
    ----------
    complex_ : CellComplex
        Oriented complex with at least one 2-cell.
    atol : float or None, optional
        Absolute entrywise tolerance. Default is the teaching
        triangle floor ``1e-6``.

    Returns
    -------
    BoundaryNilpotencyReport
        Product, max absolute entry, and flag.

    Raises
    ------
    TypeError
        If ``complex_`` is not a :class:`CellComplex`.
    ValueError
        If the complex has no 2-cells.
    """
    if not isinstance(complex_, CellComplex):
        msg = f"complex_ must be a CellComplex, got {type(complex_).__name__}"
        raise TypeError(msg)
    if int(complex_.num_faces) < 1:
        raise ValueError(
            "cell_complex_boundary_nilpotency requires at least one 2-cell"
        )
    from koopman_graph.operators.cochain import (
        DEFAULT_NILPOTENCY_ATOL,
        boundary_nilpotency,
    )

    tolerance = DEFAULT_NILPOTENCY_ATOL if atol is None else float(atol)
    return boundary_nilpotency(
        boundary_operator(complex_, 1),
        boundary_operator(complex_, 2),
        atol=tolerance,
    )


def order2_cochain_teaching(
    *,
    latent_dim: int = 2,
    use_cross_degree: bool = False,
) -> Order2CochainTeaching:
    """Build the order-2 teaching path and bind a cochain operator.

    The filled triangle satisfies :math:`B_1 B_2\\approx 0` at atol
    ``1e-6``. The bound operator advances node and edge latents only.
    Sheaf restriction maps are not required. Cell-complex degree 3
    remains the teaching ceiling (``Lim2020Hodge``, ``TopoX2024``).

    Parameters
    ----------
    latent_dim : int, optional
        Shared node / edge width. Default ``2``.
    use_cross_degree : bool, optional
        Enable incidence cross terms. Default ``False``.

    Returns
    -------
    Order2CochainTeaching
        Triangle, bound operator, nilpotency report, and ceiling.

    Raises
    ------
    ValueError
        If ``latent_dim`` is invalid.
    """
    complex_ = teaching_order2_triangle()
    operator = bind_cochain_operator(
        complex_,
        latent_dim=latent_dim,
        use_cross_degree=use_cross_degree,
    )
    return Order2CochainTeaching(
        complex=complex_,
        operator=operator,
        nilpotency=cell_complex_boundary_nilpotency(complex_),
        max_cell_degree=MAX_CELL_COMPLEX_DEGREE,
    )
