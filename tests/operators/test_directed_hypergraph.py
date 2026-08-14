"""Directed-hypergraph incidence helpers (forward random walk).

Primary reference
-----------------
Ducournau & Bretto, *Computer Vision and Image Understanding* 120 (2014)
(`Ducournau2014DirectedHypergraphs`; Crossref DOI 10.1016/j.cviu.2013.10.012).
Transition operator

.. math::

    P = D_{v,\\mathrm{out}}^{-1} B_{\\mathrm{out}} W_e
        D_{e,\\mathrm{in}}^{-1} B_{\\mathrm{in}}^{\\top}.
"""

from __future__ import annotations

import pytest
import torch

from koopman_graph.graph_utils import (
    dense_hyperedge_backward_random_walk_adjacency,
    dense_hyperedge_dual_random_walk_adjacency,
    dense_hyperedge_dual_random_walk_factors,
    dense_hyperedge_forward_random_walk_adjacency,
    dense_random_walk_normalized_adjacency,
    hyperedge_dual_random_walk_matvec,
    hyperedge_forward_random_walk_matvec,
    snapshot_to_device,
)


def _bipartite(pairs: list[tuple[int, int]]) -> torch.Tensor:
    """Build ``(2, nnz)`` bipartite incidence from ``(node, hedge)`` pairs."""
    if not pairs:
        return torch.zeros((2, 0), dtype=torch.long)
    nodes, hedges = zip(*pairs, strict=True)
    return torch.tensor([nodes, hedges], dtype=torch.long)


def test_forward_rw_hand_computed_single_hyperarc() -> None:
    """Tail {0} → head {1,2}: walker at 0 splits mass 1/2 to each head."""
    # Nodes 0,1,2; one hyperarc e0 with tail={0}, head={1,2}, w=1.
    tail = _bipartite([(0, 0)])
    head = _bipartite([(1, 0), (2, 0)])
    p = dense_hyperedge_forward_random_walk_adjacency(
        tail,
        head,
        num_nodes=3,
        dtype=torch.float64,
    )
    expected = torch.tensor(
        [
            [0.0, 0.5, 0.5],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(p, expected)
    row_sums = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    assert torch.allclose(p.sum(dim=1), row_sums)


def test_forward_rw_matvec_matches_dense() -> None:
    """Matvec agrees with dense ``P @ x`` on a two-hyperarc fixture."""
    # e0: 0 → {1,2}; e1: {1,2} → 0.
    tail = _bipartite([(0, 0), (1, 1), (2, 1)])
    head = _bipartite([(1, 0), (2, 0), (0, 1)])
    x = torch.tensor([[1.0, 0.0], [2.0, 1.0], [3.0, -1.0]], dtype=torch.float64)
    dense = dense_hyperedge_forward_random_walk_adjacency(
        tail,
        head,
        num_nodes=3,
        dtype=torch.float64,
    )
    y_dense = dense @ x
    y_mv = hyperedge_forward_random_walk_matvec(tail, head, x)
    assert torch.allclose(y_mv, y_dense)


def test_forward_rw_isolated_vertex_zero_row() -> None:
    """Isolated node (never in a tail) gets a zero row."""
    # Node 2 never appears; e0: 0 → 1.
    tail = _bipartite([(0, 0)])
    head = _bipartite([(1, 0)])
    p = dense_hyperedge_forward_random_walk_adjacency(
        tail,
        head,
        num_nodes=3,
        dtype=torch.float64,
    )
    assert torch.equal(p[2], torch.zeros(3, dtype=torch.float64))
    assert torch.allclose(p[0], torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64))


def test_forward_rw_empty_head_contributes_no_mass() -> None:
    """Hyperarc with empty head (zero head size) injects no transition mass."""
    # e0 has a tail but no head entries; e1: 1 → 0 is well-formed.
    tail = _bipartite([(0, 0), (1, 1)])
    head = _bipartite([(0, 1)])  # only e1 has a head
    p = dense_hyperedge_forward_random_walk_adjacency(
        tail,
        head,
        num_nodes=2,
        dtype=torch.float64,
    )
    # Node 0's only outgoing hyperarc has empty head → zero row.
    assert torch.equal(p[0], torch.zeros(2, dtype=torch.float64))
    assert torch.allclose(p[1], torch.tensor([1.0, 0.0], dtype=torch.float64))


def test_forward_rw_empty_tail_ignored() -> None:
    """Hyperarc with empty tail never leaves any vertex."""
    # e0 head-only; e1: 0 → 1.
    tail = _bipartite([(0, 1)])
    head = _bipartite([(1, 0), (1, 1)])
    p = dense_hyperedge_forward_random_walk_adjacency(
        tail,
        head,
        num_nodes=2,
        dtype=torch.float64,
    )
    assert torch.allclose(p[0], torch.tensor([0.0, 1.0], dtype=torch.float64))
    assert torch.equal(p[1], torch.zeros(2, dtype=torch.float64))


def test_forward_rw_empty_incidence_is_zero() -> None:
    """No hyperedges ⇒ zero operator on ``N`` nodes."""
    empty = torch.zeros((2, 0), dtype=torch.long)
    p = dense_hyperedge_forward_random_walk_adjacency(
        empty,
        empty,
        num_nodes=4,
        dtype=torch.float32,
    )
    assert p.shape == (4, 4)
    assert torch.equal(p, torch.zeros(4, 4))


def test_forward_rw_rejects_bad_shapes() -> None:
    """Non-bipartite incidence tensors raise."""
    bad = torch.zeros((3, 2), dtype=torch.long)
    good = torch.zeros((2, 0), dtype=torch.long)
    with pytest.raises(ValueError, match="tail_index"):
        dense_hyperedge_forward_random_walk_adjacency(
            bad,
            good,
            num_nodes=1,
            dtype=torch.float32,
        )
    with pytest.raises(ValueError, match="head_index"):
        dense_hyperedge_forward_random_walk_adjacency(
            good,
            bad,
            num_nodes=1,
            dtype=torch.float32,
        )


def test_forward_rw_weighted_hyperarc() -> None:
    """Weights enter ``D_v_out`` and the middle ``W_e`` factor."""
    # Two parallel hyperarcs 0 → 1 with weights 1 and 3; head size 1 each.
    tail = _bipartite([(0, 0), (0, 1)])
    head = _bipartite([(1, 0), (1, 1)])
    weights = torch.tensor([1.0, 3.0], dtype=torch.float64)
    p = dense_hyperedge_forward_random_walk_adjacency(
        tail,
        head,
        num_nodes=2,
        hyperedge_weight=weights,
        dtype=torch.float64,
    )
    # d_out(0) = 1+3 = 4; mass to 1 is (1·1 + 3·1) / 4 = 1.
    assert torch.allclose(p[0], torch.tensor([0.0, 1.0], dtype=torch.float64))


def test_exported_from_graph_utils() -> None:
    """Helpers are on the graph_utils façade."""
    import koopman_graph.graph_utils as graph_utils

    assert "dense_hyperedge_forward_random_walk_adjacency" in graph_utils.__all__
    assert "hyperedge_forward_random_walk_matvec" in graph_utils.__all__
    assert "dense_hyperedge_dual_random_walk_adjacency" in graph_utils.__all__
    assert "dense_hyperedge_dual_random_walk_factors" in graph_utils.__all__
    assert "hyperedge_dual_random_walk_matvec" in graph_utils.__all__


def test_dual_is_forward_plus_backward_hand_oracle() -> None:
    """``P_dual = P_fwd + P_bwd`` on the single-hyperarc fixture."""
    # e0: 0 → {1,2}
    tail = _bipartite([(0, 0)])
    head = _bipartite([(1, 0), (2, 0)])
    p_fwd = dense_hyperedge_forward_random_walk_adjacency(
        tail, head, num_nodes=3, dtype=torch.float64
    )
    p_bwd = dense_hyperedge_backward_random_walk_adjacency(
        tail, head, num_nodes=3, dtype=torch.float64
    )
    # Backward: heads {1,2} → tail {0}; each of 1,2 has out-degree 1, head size 1.
    expected_bwd = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(p_bwd, expected_bwd)
    factors = dense_hyperedge_dual_random_walk_factors(
        tail, head, num_nodes=3, dtype=torch.float64
    )
    assert torch.allclose(factors[0], p_fwd)
    assert torch.allclose(factors[1], p_bwd)
    p_dual = dense_hyperedge_dual_random_walk_adjacency(
        tail, head, num_nodes=3, dtype=torch.float64
    )
    assert torch.allclose(p_dual, p_fwd + p_bwd)
    x = torch.arange(3, dtype=torch.float64).unsqueeze(1)
    assert torch.allclose(
        hyperedge_dual_random_walk_matvec(tail, head, x),
        p_dual @ x,
    )


def test_dual_does_not_require_reverse_hyperedge_list() -> None:
    """Backward uses swapped indices; callers never build reverse arcs."""
    tail = _bipartite([(0, 0), (1, 1)])
    head = _bipartite([(1, 0), (0, 1)])
    via_swap = dense_hyperedge_forward_random_walk_adjacency(
        head, tail, num_nodes=2, dtype=torch.float64
    )
    via_bwd = dense_hyperedge_backward_random_walk_adjacency(
        tail, head, num_nodes=2, dtype=torch.float64
    )
    assert torch.allclose(via_swap, via_bwd)


def _single_hyperarc_tail_head() -> tuple[torch.Tensor, torch.Tensor]:
    """Return directed incidence for the 3-node oracle fixture.

    Convention
    ----------
    One hyperarc ``e0`` with **tail** ``{0}`` and **head** ``{1, 2}``
    (bipartite ``(node, hedge)`` rows). Ducournau–Bretto forward walk:

    .. math::

        P_{\\mathrm{fwd}} =
        \\begin{bmatrix}
        0 & 1/2 & 1/2 \\\\
        0 & 0 & 0 \\\\
        0 & 0 & 0
        \\end{bmatrix},
        \\qquad
        P_{\\mathrm{bwd}} =
        \\begin{bmatrix}
        0 & 0 & 0 \\\\
        1 & 0 & 0 \\\\
        1 & 0 & 0
        \\end{bmatrix}.
    """
    return _bipartite([(0, 0)]), _bipartite([(1, 0), (2, 0)])


def test_operator_forward_rw_matvec_hand_oracle() -> None:
    """Operator forward RW matches hand Kronecker / dense arithmetic.

    Reference arithmetic (``d=2``, diagonal factors)::

        K_self = diag(0.5, 0.8),  K_hedge = diag(0.2, -0.3)
        z_next = z @ K_self.T + (P_fwd @ z) @ K_hedge.T

    with ``P_fwd`` from :func:`_single_hyperarc_tail_head`. Swapping
    ``tail_index`` / ``head_index`` must **not** match (convention guard).
    """
    from koopman_graph.operators.hypergraph import HypergraphKoopmanOperator

    tail, head = _single_hyperarc_tail_head()
    p_fwd = torch.tensor(
        [
            [0.0, 0.5, 0.5],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    k_self = torch.diag(torch.tensor([0.5, 0.8]))
    k_hedge = torch.diag(torch.tensor([0.2, -0.3]))
    z = torch.tensor([[1.0, 0.0], [2.0, 1.0], [3.0, -1.0]])
    expected = z @ k_self.T + (p_fwd @ z) @ k_hedge.T

    op = HypergraphKoopmanOperator(
        2, incidence_mode="forward_random_walk", init_mode="identity"
    )
    op.set_dense_matrices(k_self, k_hedge)
    got = op(z, None, tail_index=tail, head_index=head)
    assert torch.allclose(got, expected, atol=1e-6)

    flipped = op(z, None, tail_index=head, head_index=tail)
    assert not torch.allclose(flipped, expected, atol=1e-5), (
        "swapped tail/head must diverge from the documented oracle "
        "(silent convention flip)"
    )


def test_operator_dual_rw_matvec_hand_oracle() -> None:
    """Dual RW operator matches ``P_fwd``/``P_bwd`` hand arithmetic.

    Reference::

        z_next = z @ K_self.T
                 + (P_fwd @ z) @ K_hedge.T
                 + (P_bwd @ z) @ K_bwd.T

    with nonzero ``K_bwd`` (dual initializes ``K_bwd`` at zero). Swapping
    ``tail_index`` / ``head_index`` must fail the oracle.
    """
    from koopman_graph.operators.hypergraph import HypergraphKoopmanOperator

    tail, head = _single_hyperarc_tail_head()
    p_fwd = torch.tensor(
        [
            [0.0, 0.5, 0.5],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    p_bwd = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    k_self = torch.diag(torch.tensor([0.5, 0.8]))
    k_hedge = torch.diag(torch.tensor([0.2, -0.3]))
    k_bwd = torch.diag(torch.tensor([0.1, 0.4]))
    z = torch.tensor([[1.0, 0.0], [2.0, 1.0], [3.0, -1.0]])
    expected = z @ k_self.T + (p_fwd @ z) @ k_hedge.T + (p_bwd @ z) @ k_bwd.T

    op = HypergraphKoopmanOperator(
        2, incidence_mode="dual_random_walk", init_mode="identity"
    )
    op.set_dense_matrices(k_self, k_hedge, k_bwd=k_bwd)
    got = op(z, None, tail_index=tail, head_index=head)
    assert torch.allclose(got, expected, atol=1e-6)

    flipped = op(z, None, tail_index=head, head_index=tail)
    assert not torch.allclose(flipped, expected, atol=1e-5), (
        "swapped tail/head must diverge from the documented dual oracle "
        "(silent convention flip)"
    )


def test_incidence_mode_unknown_raises() -> None:
    """Unknown incidence_mode names raise clearly."""
    from koopman_graph.operators.hypergraph import HypergraphKoopmanOperator

    with pytest.raises(ValueError, match="incidence_mode"):
        HypergraphKoopmanOperator(2, incidence_mode="not_a_mode")  # type: ignore[arg-type]


def test_factory_and_checkpoint_round_trip_incidence_mode(tmp_path) -> None:
    """Factory keyword and format-1 checkpoint round-trip the mode."""
    from koopman_graph import GraphKoopmanModel
    from koopman_graph.nn.hypergraph import HypergraphDecoder, HypergraphEncoder
    from koopman_graph.serialization import (
        build_model_config,
        load_checkpoint,
        reconstruct_model,
        save_checkpoint,
    )

    model = GraphKoopmanModel(
        encoder=HypergraphEncoder(2, 4, 2, num_layers=1),
        decoder=HypergraphDecoder(2, 4, 2, num_layers=1),
        latent_dim=2,
        time_step=1.0,
        koopman="hypergraph",
        koopman_hypergraph_incidence_mode="forward_random_walk",
    )
    assert model.koopman.incidence_mode == "forward_random_walk"
    config = build_model_config(model)
    assert config["hypergraph_incidence_mode"] == "forward_random_walk"
    path = tmp_path / "hyp.pt"
    save_checkpoint(model, path)
    loaded = load_checkpoint(path)
    assert loaded.koopman.incidence_mode == "forward_random_walk"

    # Absent key on a Zhou model reconstructs as zhou_symmetric.
    zhou = GraphKoopmanModel(
        encoder=HypergraphEncoder(2, 4, 2, num_layers=1),
        decoder=HypergraphDecoder(2, 4, 2, num_layers=1),
        latent_dim=2,
        time_step=1.0,
        koopman="hypergraph",
    )
    zhou_config = build_model_config(zhou)
    assert zhou_config["hypergraph_incidence_mode"] == "zhou_symmetric"
    zhou_config.pop("hypergraph_incidence_mode", None)
    rebuilt = reconstruct_model(zhou_config)
    assert rebuilt.koopman.incidence_mode == "zhou_symmetric"


def test_snapshot_to_device_preserves_directed_incidence() -> None:
    """Device transfer keeps ``tail_index`` / ``head_index`` for fit paths."""
    from torch_geometric.data import Data

    tail = _bipartite([(0, 0)])
    head = _bipartite([(1, 0), (2, 0)])
    snapshot = Data(
        x=torch.randn(3, 2),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        hyperedge_index=_bipartite([(0, 0), (1, 0), (2, 0)]),
        tail_index=tail,
        head_index=head,
    )
    moved = snapshot_to_device(snapshot, torch.device("cpu"))
    assert torch.equal(moved.tail_index, tail)
    assert torch.equal(moved.head_index, head)


def test_fit_predict_smoke_each_incidence_mode() -> None:
    """Tiny directed hypergraph fit/predict for each incidence mode."""
    from torch_geometric.data import Data

    from koopman_graph import GraphKoopmanModel
    from koopman_graph.data import GraphSnapshotSequence
    from koopman_graph.nn.hypergraph import HypergraphDecoder, HypergraphEncoder

    # Undirected incidence for encoder/decoder Zhou path; directed for RW ops.
    # e0: 0 → {1,2}; also expose undirected members {0,1,2} for HypergraphConv.
    undirected = _bipartite([(0, 0), (1, 0), (2, 0)])
    tail = _bipartite([(0, 0)])
    head = _bipartite([(1, 0), (2, 0)])
    snapshots = []
    for _ in range(4):
        snapshots.append(
            Data(
                x=torch.randn(3, 2),
                edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
                hyperedge_index=undirected.clone(),
                tail_index=tail.clone(),
                head_index=head.clone(),
            )
        )
    sequence = GraphSnapshotSequence(snapshots)

    for mode in ("zhou_symmetric", "forward_random_walk", "dual_random_walk"):
        model = GraphKoopmanModel(
            encoder=HypergraphEncoder(2, 4, 2, num_layers=1),
            decoder=HypergraphDecoder(2, 4, 2, num_layers=1),
            latent_dim=2,
            time_step=1.0,
            koopman="hypergraph",
            koopman_hypergraph_incidence_mode=mode,
        )
        history = model.fit(sequence, epochs=1, lr=1e-2)
        assert history is not None
        pred = model.predict(sequence[0], steps=1)
        assert len(pred) == 1
        assert pred[0].x is not None
        assert pred[0].x.shape == (3, 2)


def test_two_uniform_reduces_to_graph_random_walk() -> None:
    """2-uniform directed cycle matches graph forward / backward RW."""
    # Digraph cycle 0→1→2→0 as one-vertex tail/head hyperarcs.
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    tail = _bipartite([(0, 0), (1, 1), (2, 2)])
    head = _bipartite([(1, 0), (2, 1), (0, 2)])
    p_fwd, p_bwd = dense_hyperedge_dual_random_walk_factors(
        tail, head, num_nodes=3, dtype=torch.float64
    )
    graph_fwd = dense_random_walk_normalized_adjacency(
        edge_index, 3, dtype=torch.float64, direction="forward"
    )
    graph_bwd = dense_random_walk_normalized_adjacency(
        edge_index, 3, dtype=torch.float64, direction="backward"
    )
    assert torch.allclose(p_fwd, graph_fwd)
    assert torch.allclose(p_bwd, graph_bwd)
    # Graph dual mode uses both factors; incidence dual sum matches Â_f+Â_b.
    assert torch.allclose(
        dense_hyperedge_dual_random_walk_adjacency(
            tail, head, num_nodes=3, dtype=torch.float64
        ),
        graph_fwd + graph_bwd,
    )
