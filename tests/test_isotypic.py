"""Tests for exact-automorphism isotypic decomposition.

Textbook oracles
----------------
Complete graphs ``K_n`` and cycle graphs ``C_n`` have classical, hand-checkable
isotypic ranks for the permutation representation of ``Aut(G)``. Fixtures below
record the **derivation**, not only the expected tuples, so a reader can audit
without rebuilding the character table from scratch.

Tolerance policy
----------------
Projector algebra uses ``_PROJ_ATOL = 1e-6`` (float64 orbital / closed-form
path). Closed-form ``S_n`` comparisons use ``atol=1e-8``. Tests are seed-free /
deterministic (no RNG in the public API path).
"""

from __future__ import annotations

import builtins
import importlib.util
import math
from dataclasses import dataclass

import pytest
import torch

from koopman_graph.graph_utils import (
    MAX_ISOTYPIC_NODES,
    IsotypicDecomposition,
    compute_isotypic_decomposition,
)
from koopman_graph.graph_utils.representation import _PROJ_ATOL

_HAS_PYNAUTY = importlib.util.find_spec("pynauty") is not None

# Float64 closed-form projector comparison (exact rational entries).
_CLOSED_FORM_ATOL = 1e-8


def _cycle_edge_index(num_nodes: int) -> torch.Tensor:
    """Bidirectional cycle edges."""
    src = []
    dst = []
    for i in range(num_nodes):
        j = (i + 1) % num_nodes
        src.extend([i, j])
        dst.extend([j, i])
    return torch.tensor([src, dst], dtype=torch.long)


def _complete_edge_index(num_nodes: int) -> torch.Tensor:
    """Bidirectional complete-graph edges."""
    src = []
    dst = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            src.extend([i, j])
            dst.extend([j, i])
    return torch.tensor([src, dst], dtype=torch.long)


def _assert_projector_algebra(report: IsotypicDecomposition) -> None:
    """Idempotence, orthogonality, and partition of identity."""
    eye = torch.eye(report.num_nodes, dtype=torch.float64)
    total = torch.zeros_like(eye)
    for i, left in enumerate(report.projectors):
        assert left.shape == (report.num_nodes, report.num_nodes)
        assert float(torch.linalg.norm(left @ left - left)) <= _PROJ_ATOL
        total = total + left
        for j, right in enumerate(report.projectors):
            if i >= j:
                continue
            assert float(torch.linalg.norm(left @ right)) <= _PROJ_ATOL
    assert float(torch.linalg.norm(total - eye)) <= _PROJ_ATOL
    assert report.dimensions == tuple(
        int(round(float(torch.trace(p).item()))) for p in report.projectors
    )
    assert sum(report.dimensions) == report.num_nodes


def _symmetric_group_projectors(num_nodes: int) -> tuple[torch.Tensor, ...]:
    """Textbook projectors for the permutation representation of ``S_n``.

    Derivation
    ----------
    ``Aut(K_n) = Aut(\\bar K_n) = S_n`` acts on coordinates by permuting the
    standard basis of ``R^n``. The permutation representation decomposes as

        trivial (dim 1)  ⊕  standard (dim n − 1),

    each with multiplicity one (Fulton–Harris, *Representation Theory*, §1.3;
    Sagan, *The Symmetric Group*, §1.3). Explicit projectors:

        P_triv = (1/n) 11^T ,   P_std = I − P_triv .

    The MVP reports ``multiplicities`` equal to these ranks (dimension of each
    isotypic component), not a separate irrep-multiplicity table.
    """
    eye = torch.eye(num_nodes, dtype=torch.float64)
    trivial = torch.full(
        (num_nodes, num_nodes),
        1.0 / float(num_nodes),
        dtype=torch.float64,
    )
    if num_nodes == 1:
        return (trivial,)
    return (eye - trivial, trivial)  # descending dimension order


@dataclass(frozen=True)
class _CycleOracle:
    """Hand-derived isotypic ranks for ``Aut(C_n) = D_n`` on vertices.

    Derivation
    ----------
    The automorphism group of the cycle graph ``C_n`` is the dihedral group
    ``D_n`` of order ``2n`` (rotations and reflections of the regular n-gon).
    The permutation representation on vertices is the same as the natural
    action of ``D_n`` on the vertices of the n-gon. Its isotypic dimensions
    follow from the real irreducible representations of ``D_n`` (Serre,
    *Linear Representations of Finite Groups*, §5.3; James–Liebeck,
    *Representations and Characters of Groups*, §18):

    * Always one trivial summand of dimension 1 (constant vectors).
    * For **odd** ``n``: the remaining ``n − 1`` dimensions split into
      ``(n − 1) / 2`` distinct 2-dimensional isotypic components
      (Fourier cosine/sine pairs at frequencies ``1, …, (n−1)/2``).
      Example ``n = 5`` → ranks ``(2, 2, 1)`` after descending sort.
    * For **even** ``n``: there are two additional 1-dimensional summands
      (the ``±1`` characters of the index-2 rotation subgroup / reflection
      types) and ``n/2 − 1`` distinct 2-dimensional components.
      Example ``n = 6`` → ranks ``(2, 2, 1, 1)``.

    Group order is always ``|D_n| = 2n``.
    """

    num_nodes: int
    dimensions: tuple[int, ...]
    group_order: int


def _cycle_oracle(num_nodes: int) -> _CycleOracle:
    """Return the textbook dimension multiset for ``C_n`` (sorted descending)."""
    if num_nodes < 3:
        msg = f"cycle oracle requires n >= 3, got {num_nodes}"
        raise ValueError(msg)
    group_order = 2 * num_nodes
    if num_nodes % 2 == 1:
        twos = (num_nodes - 1) // 2
        dimensions = tuple([2] * twos + [1])
    else:
        twos = num_nodes // 2 - 1
        dimensions = tuple([2] * twos + [1, 1])
    return _CycleOracle(
        num_nodes=num_nodes,
        dimensions=dimensions,
        group_order=group_order,
    )


@pytest.mark.parametrize("num_nodes", [3, 4, 5])
def test_complete_graph_textbook_sn_decomposition(num_nodes: int) -> None:
    """``K_n`` matches trivial ⊕ standard projectors for ``S_n``."""
    expected = _symmetric_group_projectors(num_nodes)
    report = compute_isotypic_decomposition(
        _complete_edge_index(num_nodes),
        num_nodes,
    )
    assert report.group_order == math.factorial(num_nodes)
    assert report.dimensions == tuple(
        int(round(float(torch.trace(p).item()))) for p in expected
    )
    # MVP multiplicity metadata equals isotypic dimensions for these fixtures.
    assert report.multiplicities == report.dimensions
    assert len(report.projectors) == len(expected)
    for computed, reference in zip(report.projectors, expected, strict=True):
        assert torch.allclose(computed, reference, atol=_CLOSED_FORM_ATOL)
    _assert_projector_algebra(report)


def test_empty_graph_matches_same_sn_formula() -> None:
    """Empty ``\\bar K_n`` shares ``Aut = S_n`` with the complete-graph oracle."""
    num_nodes = 5
    expected = _symmetric_group_projectors(num_nodes)
    report = compute_isotypic_decomposition(
        torch.empty((2, 0), dtype=torch.long),
        num_nodes,
    )
    assert report.dimensions == tuple(
        int(round(float(torch.trace(p).item()))) for p in expected
    )
    for computed, reference in zip(report.projectors, expected, strict=True):
        assert torch.allclose(computed, reference, atol=_CLOSED_FORM_ATOL)
    _assert_projector_algebra(report)


@pytest.mark.skipif(not _HAS_PYNAUTY, reason="pynauty not installed")
@pytest.mark.parametrize("num_nodes", [5, 6])
def test_cycle_graph_textbook_dihedral_ranks(num_nodes: int) -> None:
    """``C_n`` isotypic ranks match the dihedral permutation-representation oracle."""
    oracle = _cycle_oracle(num_nodes)
    report = compute_isotypic_decomposition(
        _cycle_edge_index(num_nodes),
        num_nodes,
    )
    assert isinstance(report, IsotypicDecomposition)
    assert report.method == "automorphism"
    assert report.group_order == oracle.group_order
    assert report.dimensions == oracle.dimensions
    assert report.multiplicities == report.dimensions
    _assert_projector_algebra(report)


def test_cycle_oracle_fixture_matches_hand_derivation() -> None:
    """Fixture helper itself encodes the odd/even dihedral dimension rules."""
    assert _cycle_oracle(5).dimensions == (2, 2, 1)
    assert _cycle_oracle(5).group_order == 10
    assert _cycle_oracle(6).dimensions == (2, 2, 1, 1)
    assert _cycle_oracle(6).group_order == 12
    assert _cycle_oracle(7).dimensions == (2, 2, 2, 1)


def test_wl_method_refused() -> None:
    """WL / non-automorphism methods raise rather than fake isotypic labels."""
    edges = _cycle_edge_index(4)
    with pytest.raises(ValueError, match="refused|Weisfeiler"):
        compute_isotypic_decomposition(edges, 4, method="wl")
    with pytest.raises(ValueError, match="refused|Weisfeiler"):
        compute_isotypic_decomposition(edges, 4, method="auto")


def test_mvp_node_ceiling() -> None:
    """``N > 20`` raises naming the MVP ceiling."""
    n = MAX_ISOTYPIC_NODES + 1
    edges = torch.empty((2, 0), dtype=torch.long)
    with pytest.raises(ValueError, match=str(MAX_ISOTYPIC_NODES)):
        compute_isotypic_decomposition(edges, n)


def test_missing_pynauty_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing pynauty raises a guided ``ImportError`` (path graph needs it)."""
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "pynauty" or name.startswith("pynauty."):
            raise ImportError("no pynauty")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    # Path graph is not empty/complete, so the S_N shortcut does not apply.
    edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    with pytest.raises(ImportError, match="pynauty"):
        compute_isotypic_decomposition(edges, 3)


def test_exported_from_graph_utils() -> None:
    """Isotypic helpers are on the graph_utils surface, off root ``__all__``."""
    import koopman_graph as kg
    import koopman_graph.graph_utils as graph_utils

    assert "compute_isotypic_decomposition" in graph_utils.__all__
    assert "IsotypicDecomposition" in graph_utils.__all__
    assert "compute_isotypic_decomposition" not in kg.__all__


@pytest.mark.skipif(not _HAS_PYNAUTY, reason="pynauty not installed")
def test_factory_isotypic_ties_k_self_within_aut_orbits() -> None:
    """``koopman_symmetry='isotypic'`` shares ``K_self`` on exact Aut orbits."""
    from torch_geometric.data import Data

    from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
    from koopman_graph.data import GraphSnapshotSequence
    from koopman_graph.operators.graph import GraphKoopmanOperator
    from koopman_graph.training.loop import bind_pending_orbit_ties

    num_nodes = 5
    latent_dim = 2
    # Star: center orbit {0}, leaves {1,2,3,4}.
    edges = torch.tensor(
        [[0, 1, 0, 2, 0, 3, 0, 4], [1, 0, 2, 0, 3, 0, 4, 0]],
        dtype=torch.long,
    )
    model = GraphKoopmanModel(
        encoder=GNNEncoder(2, 4, latent_dim, num_layers=1),
        decoder=GNNDecoder(latent_dim, 4, 2, num_layers=1),
        latent_dim=latent_dim,
        time_step=1.0,
        koopman="graph",
        koopman_symmetry="isotypic",
    )
    assert isinstance(model.koopman, GraphKoopmanOperator)
    assert model.koopman.isotypic_symmetry
    sequence = GraphSnapshotSequence(
        [
            Data(
                x=torch.randn(num_nodes, 2),
                edge_index=edges.clone(),
            )
            for _ in range(4)
        ]
    )
    bind_pending_orbit_ties(model, [sequence])
    op = model.koopman
    assert op.orbit_partition == ((0,), (1, 2, 3, 4))
    assert op.isotypic_decomposition is not None
    with torch.no_grad():
        op._orbit_selves[0].K.zero_()
        op._orbit_selves[0].K.fill_diagonal_(0.5)
        op._orbit_selves[1].K.zero_()
        op._orbit_selves[1].K.fill_diagonal_(0.8)
    # One optimizer step on shared leaf bank keeps leaf blocks equal.
    params = list(op._orbit_selves.parameters())
    optimizer = torch.optim.SGD(params, lr=0.1)
    z = torch.randn(num_nodes, latent_dim, dtype=torch.float32)
    loss = op.apply_tied_self(z).pow(2).mean()
    loss.backward()
    optimizer.step()
    blocks = op.tied_self_blocks(num_nodes)
    assert torch.allclose(blocks[1], blocks[2])
    assert torch.allclose(blocks[1], blocks[4])
    assert not torch.allclose(blocks[0], blocks[1])


def test_factory_isotypic_rejects_unsupported_combinations() -> None:
    """Unsupported kinds / orbit conflicts raise clear errors."""
    from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
    from koopman_graph.model.factory import build_koopman

    encoder = GNNEncoder(2, 4, 2, num_layers=1)
    decoder = GNNDecoder(2, 4, 2, num_layers=1)
    with pytest.raises(ValueError, match="mutually exclusive"):
        GraphKoopmanModel(
            encoder=encoder,
            decoder=decoder,
            latent_dim=2,
            time_step=1.0,
            koopman="graph",
            koopman_symmetry="isotypic",
            koopman_auto_orbits=True,
        )
    with pytest.raises(ValueError, match="requires koopman='graph'"):
        build_koopman(
            koopman="hypergraph",
            latent_dim=2,
            control_dim=0,
            control_mode="additive",
            bilinear_rank=None,
            dynamics_mode="discrete",
            koopman_init_mode="identity_noise",
            koopman_init_scale=1e-2,
            koopman_parameterization="dense",
            koopman_max_spectral_radius=1.0,
            koopman_auxiliary_hidden_dims=None,
            koopman_symmetry="isotypic",
        )
    with pytest.raises(ValueError, match="None or 'isotypic'"):
        GraphKoopmanModel(
            encoder=encoder,
            decoder=decoder,
            latent_dim=2,
            time_step=1.0,
            koopman="graph",
            koopman_symmetry="orbit",
        )


def test_default_graph_path_unchanged_without_symmetry() -> None:
    """Default ``koopman='graph'`` does not allocate orbit / isotypic banks."""
    from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
    from koopman_graph.operators.graph import GraphKoopmanOperator

    model = GraphKoopmanModel(
        encoder=GNNEncoder(2, 4, 2, num_layers=1),
        decoder=GNNDecoder(2, 4, 2, num_layers=1),
        latent_dim=2,
        time_step=1.0,
        koopman="graph",
    )
    assert isinstance(model.koopman, GraphKoopmanOperator)
    assert not model.koopman.isotypic_symmetry
    assert not model.koopman.auto_orbits
    assert model.koopman.orbit_partition is None
    assert not model.koopman.uses_orbit_selves


@pytest.mark.skipif(not _HAS_PYNAUTY, reason="pynauty not installed")
def test_isotypic_checkpoint_round_trip(tmp_path) -> None:
    """Format-1 checkpoints restore ``symmetry='isotypic'`` and orbit banks."""
    from torch_geometric.data import Data

    from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
    from koopman_graph.data import GraphSnapshotSequence
    from koopman_graph.serialization import (
        build_model_config,
        load_checkpoint,
        save_checkpoint,
    )
    from koopman_graph.training.loop import bind_pending_orbit_ties

    num_nodes = 5
    latent_dim = 2
    edges = torch.tensor(
        [[0, 1, 0, 2, 0, 3, 0, 4], [1, 0, 2, 0, 3, 0, 4, 0]],
        dtype=torch.long,
    )
    model = GraphKoopmanModel(
        encoder=GNNEncoder(2, 4, latent_dim, num_layers=1),
        decoder=GNNDecoder(latent_dim, 4, 2, num_layers=1),
        latent_dim=latent_dim,
        time_step=1.0,
        koopman="graph",
        koopman_symmetry="isotypic",
    )
    sequence = GraphSnapshotSequence(
        [Data(x=torch.randn(num_nodes, 2), edge_index=edges.clone()) for _ in range(3)]
    )
    bind_pending_orbit_ties(model, [sequence])
    with torch.no_grad():
        model.koopman._orbit_selves[0].K.fill_(0.11)
        model.koopman._orbit_selves[1].K.fill_(0.22)

    config = build_model_config(model)
    assert config["symmetry"]["symmetry"] == "isotypic"
    assert config["symmetry"]["auto_orbits"] is False
    assert config["symmetry"]["method"] == "exact"

    path = tmp_path / "isotypic.pt"
    save_checkpoint(model, path)
    loaded = load_checkpoint(path)
    assert loaded.koopman.isotypic_symmetry
    assert loaded.koopman.orbit_partition == ((0,), (1, 2, 3, 4))
    assert torch.allclose(
        loaded.koopman._orbit_selves[0].K,
        model.koopman._orbit_selves[0].K,
    )
    assert torch.allclose(
        loaded.koopman._orbit_selves[1].K,
        model.koopman._orbit_selves[1].K,
    )


@pytest.mark.skipif(not _HAS_PYNAUTY, reason="pynauty not installed")
def test_isotypic_rejects_cross_n_and_topology_change() -> None:
    """Bound isotypic banks raise on cross-N or same-N Aut orbit mismatch."""
    from torch_geometric.data import Data

    from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
    from koopman_graph.data import GraphSnapshotSequence
    from koopman_graph.training.loop import bind_pending_orbit_ties

    star = torch.tensor(
        [[0, 1, 0, 2, 0, 3, 0, 4], [1, 0, 2, 0, 3, 0, 4, 0]],
        dtype=torch.long,
    )
    model = GraphKoopmanModel(
        encoder=GNNEncoder(2, 4, 2, num_layers=1),
        decoder=GNNDecoder(2, 4, 2, num_layers=1),
        latent_dim=2,
        time_step=1.0,
        koopman="graph",
        koopman_symmetry="isotypic",
    )
    bind_pending_orbit_ties(
        model,
        [
            GraphSnapshotSequence(
                [Data(x=torch.zeros(5, 2), edge_index=star.clone()) for _ in range(2)]
            )
        ],
    )
    with pytest.raises(ValueError, match="bound for 5 nodes"):
        model.koopman.ensure_orbit_binding(6, edge_index=star)
    # Path on the same N has a different Aut orbit partition than the star.
    path = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]],
        dtype=torch.long,
    )
    with pytest.raises(ValueError, match="does not match the current topology"):
        model.koopman.ensure_orbit_binding(5, edge_index=path)


def test_absent_symmetry_block_means_no_ties() -> None:
    """Absent / null symmetry config reconstructs without orbit banks."""
    from koopman_graph import GNNDecoder, GNNEncoder, GraphKoopmanModel
    from koopman_graph.serialization import build_model_config, reconstruct_model

    model = GraphKoopmanModel(
        encoder=GNNEncoder(2, 4, 2, num_layers=1),
        decoder=GNNDecoder(2, 4, 2, num_layers=1),
        latent_dim=2,
        time_step=1.0,
        koopman="graph",
    )
    config = build_model_config(model)
    assert config["symmetry"] is None
    rebuilt = reconstruct_model(config)
    assert not rebuilt.koopman.isotypic_symmetry
    assert rebuilt.koopman.orbit_partition is None
    assert not rebuilt.koopman.uses_orbit_selves
