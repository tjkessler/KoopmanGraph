"""CI-grade synthetic multiplex / typed relational Koopman oracles.

These tests identify known multi-relation linear latent dynamics with
:class:`~koopman_graph.operators.HeteroGraphKoopmanOperator` only (no RelGraph
encode/decode). A relation-ignoring / shared-self control is required to lose
on hold-out one-step MSE. Tolerances below are justified for seeded Adam on
exact linear pairs in float32 with the listed step counts.
"""

from __future__ import annotations

import torch

from koopman_graph.data.hetero_layout import offset_edge_index
from koopman_graph.operators import GraphKoopmanOperator, HeteroGraphKoopmanOperator

# Factor recovery after ~400 Adam steps on noiseless linear pairs (seeded).
_FACTOR_ATOL = 5e-2
# Relational hold-out MSE must beat the control by at least this ratio.
_BASELINE_MARGIN = 0.5
# Absolute hold-out MSE ceiling for a correctly specified relational fit.
_RELATIONAL_MSE_ATOL = 5e-4


def _multiplex_edges(num_nodes: int = 4) -> list[torch.Tensor]:
    """Two distinct directed relation banks on a 4-node multiplex graph."""
    assert num_nodes == 4
    r0 = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    r1 = torch.tensor([[3, 2, 0], [2, 0, 1]], dtype=torch.long)
    return [r0, r1]


def _union_edges(edge_indices: list[torch.Tensor]) -> torch.Tensor:
    """Concatenate relation banks into one adjacency (duplicates allowed)."""
    return torch.cat(edge_indices, dim=1)


def _make_latent_pairs(
    operator: HeteroGraphKoopmanOperator,
    edge_indices: list[torch.Tensor],
    *,
    num_nodes: int,
    num_samples: int,
    seed: int,
    num_nodes_dict: dict[str, int] | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Roll one-step pairs ``(z_t, Op(z_t))`` from independent initial conditions."""
    generator = torch.Generator().manual_seed(seed)
    sources: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    with torch.no_grad():
        for _ in range(num_samples):
            z = torch.randn(
                num_nodes,
                operator.latent_dim,
                generator=generator,
            )
            z_next = operator(
                z,
                edge_indices,
                num_nodes_dict=num_nodes_dict,
            )
            sources.append(z)
            targets.append(z_next)
    return sources, targets


def _fit_hetero(
    operator: HeteroGraphKoopmanOperator,
    sources: list[torch.Tensor],
    targets: list[torch.Tensor],
    edge_indices: list[torch.Tensor],
    *,
    steps: int,
    lr: float,
    seed: int,
    num_nodes_dict: dict[str, int] | None = None,
) -> float:
    """Adam-fit a hetero operator on latent pairs; return final train MSE."""
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(operator.parameters(), lr=lr)
    final_loss = float("inf")
    for _ in range(steps):
        optimizer.zero_grad()
        loss = torch.zeros(())
        for z, z_next in zip(sources, targets, strict=True):
            pred = operator(z, edge_indices, num_nodes_dict=num_nodes_dict)
            loss = loss + (pred - z_next).square().mean()
        loss = loss / len(sources)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
    return final_loss


def _fit_graph(
    operator: GraphKoopmanOperator,
    sources: list[torch.Tensor],
    targets: list[torch.Tensor],
    edge_index: torch.Tensor,
    *,
    steps: int,
    lr: float,
    seed: int,
) -> float:
    """Adam-fit a homogeneous graph operator on the same latent pairs."""
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(operator.parameters(), lr=lr)
    final_loss = float("inf")
    for _ in range(steps):
        optimizer.zero_grad()
        loss = torch.zeros(())
        for z, z_next in zip(sources, targets, strict=True):
            pred = operator(z, edge_index)
            loss = loss + (pred - z_next).square().mean()
        loss = loss / len(sources)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
    return final_loss


@torch.no_grad()
def _holdout_hetero_mse(
    operator: HeteroGraphKoopmanOperator,
    sources: list[torch.Tensor],
    targets: list[torch.Tensor],
    edge_indices: list[torch.Tensor],
    *,
    num_nodes_dict: dict[str, int] | None = None,
) -> float:
    """Mean one-step MSE on held-out latent pairs."""
    total = torch.zeros(())
    for z, z_next in zip(sources, targets, strict=True):
        pred = operator(z, edge_indices, num_nodes_dict=num_nodes_dict)
        total = total + (pred - z_next).square().mean()
    return float(total / len(sources))


@torch.no_grad()
def _holdout_graph_mse(
    operator: GraphKoopmanOperator,
    sources: list[torch.Tensor],
    targets: list[torch.Tensor],
    edge_index: torch.Tensor,
) -> float:
    """Mean one-step MSE for a homogeneous graph operator."""
    total = torch.zeros(())
    for z, z_next in zip(sources, targets, strict=True):
        pred = operator(z, edge_index)
        total = total + (pred - z_next).square().mean()
    return float(total / len(sources))


def test_multiplex_relational_recovers_known_factors() -> None:
    """Seeded Adam recovers known multiplex ``K_self`` / ``K_r`` within atol.

    Tolerance ``5e-2``: noiseless linear pairs, float32, 400 steps, lr=5e-2,
    seed 0 — well above float32 Kronecker assembly noise (~1e-5) and below
    the scale of the planted factors (~0.1–0.7).
    """
    torch.manual_seed(0)
    num_nodes = 4
    latent_dim = 2
    edge_indices = _multiplex_edges(num_nodes)
    k_self = torch.tensor([[0.70, 0.10], [0.00, 0.60]])
    k_relations = [
        torch.tensor([[0.20, 0.00], [0.05, 0.15]]),
        torch.tensor([[0.00, 0.12], [0.10, 0.00]]),
    ]
    truth = HeteroGraphKoopmanOperator(
        latent_dim,
        num_relations=2,
        init_mode="identity",
    )
    truth.set_dense_matrices(k_self, k_relations)

    train_src, train_tgt = _make_latent_pairs(
        truth,
        edge_indices,
        num_nodes=num_nodes,
        num_samples=24,
        seed=1,
    )
    learned = HeteroGraphKoopmanOperator(
        latent_dim,
        num_relations=2,
        init_mode="identity_noise",
        init_scale=0.05,
    )
    _fit_hetero(
        learned,
        train_src,
        train_tgt,
        edge_indices,
        steps=400,
        lr=5e-2,
        seed=2,
    )

    assert torch.allclose(learned.K_self, k_self, atol=_FACTOR_ATOL)
    for got, expected in zip(learned.K_relations, k_relations, strict=True):
        assert torch.allclose(got, expected, atol=_FACTOR_ATOL)


def test_multiplex_relational_beats_union_graph_baseline() -> None:
    """Relational hetero beats a union-adjacency GraphKoopman control.

    Control collapses both relation banks into one edge set and one neighbor
    factor (relation-ignoring / dense-joint baseline). Hold-out one-step MSE
    for the relational fit must be below ``5e-4`` and at most half the
    control MSE.
    """
    torch.manual_seed(3)
    num_nodes = 4
    latent_dim = 2
    edge_indices = _multiplex_edges(num_nodes)
    union = _union_edges(edge_indices)
    k_self = torch.tensor([[0.65, 0.08], [0.02, 0.55]])
    k_relations = [
        torch.tensor([[0.25, 0.00], [0.00, 0.18]]),
        torch.tensor([[0.00, -0.15], [0.12, 0.00]]),
    ]
    truth = HeteroGraphKoopmanOperator(
        latent_dim,
        num_relations=2,
        init_mode="identity",
    )
    truth.set_dense_matrices(k_self, k_relations)

    train_src, train_tgt = _make_latent_pairs(
        truth,
        edge_indices,
        num_nodes=num_nodes,
        num_samples=32,
        seed=4,
    )
    hold_src, hold_tgt = _make_latent_pairs(
        truth,
        edge_indices,
        num_nodes=num_nodes,
        num_samples=16,
        seed=5,
    )

    relational = HeteroGraphKoopmanOperator(
        latent_dim,
        num_relations=2,
        init_mode="identity_noise",
        init_scale=0.05,
    )
    baseline = GraphKoopmanOperator(
        latent_dim,
        init_mode="identity_noise",
        init_scale=0.05,
        adjacency="random_walk",
    )
    _fit_hetero(
        relational,
        train_src,
        train_tgt,
        edge_indices,
        steps=350,
        lr=5e-2,
        seed=6,
    )
    _fit_graph(
        baseline,
        train_src,
        train_tgt,
        union,
        steps=350,
        lr=5e-2,
        seed=6,
    )

    mse_rel = _holdout_hetero_mse(relational, hold_src, hold_tgt, edge_indices)
    mse_base = _holdout_graph_mse(baseline, hold_src, hold_tgt, union)
    assert mse_rel < _RELATIONAL_MSE_ATOL
    assert mse_rel <= _BASELINE_MARGIN * mse_base


def test_typed_relational_recovers_and_beats_shared_self_baseline() -> None:
    """Typed per-type ``K_self`` recovers; shared-self multiplex control loses.

    Planted dynamics use distinct ``K_self^a`` / ``K_self^b``. The typed
    operator must recover both self blocks (atol ``5e-2``). A multiplex
    hetero operator on the same stacked banks (one shared ``K_self``) is the
    type-ignoring control and must show higher hold-out MSE by the same
    margin rule.
    """
    torch.manual_seed(7)
    node_types = ("a", "b")
    edge_types = (("a", "r0", "b"), ("b", "r1", "a"))
    num_nodes_dict = {"a": 3, "b": 2}
    num_nodes = sum(num_nodes_dict.values())
    latent_dim = 2

    # Type-local edges lifted into stacked global numbering (a then b).
    local_ab = torch.tensor([[0, 1, 2], [0, 1, 0]], dtype=torch.long)
    local_ba = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_indices = [
        offset_edge_index(local_ab, src_offset=0, dst_offset=num_nodes_dict["a"]),
        offset_edge_index(
            local_ba,
            src_offset=num_nodes_dict["a"],
            dst_offset=0,
        ),
    ]

    k_self = {
        "a": torch.tensor([[0.80, 0.00], [0.05, 0.70]]),
        "b": torch.tensor([[0.20, 0.15], [-0.10, 0.55]]),
    }
    k_relations = [
        torch.tensor([[0.18, 0.00], [0.00, 0.14]]),
        torch.tensor([[0.00, 0.11], [0.09, 0.00]]),
    ]
    truth = HeteroGraphKoopmanOperator(
        latent_dim,
        num_relations=2,
        node_types=node_types,
        edge_types=edge_types,
        init_mode="identity",
    )
    truth.set_dense_matrices(k_self, k_relations)

    train_src, train_tgt = _make_latent_pairs(
        truth,
        edge_indices,
        num_nodes=num_nodes,
        num_samples=32,
        seed=8,
        num_nodes_dict=num_nodes_dict,
    )
    hold_src, hold_tgt = _make_latent_pairs(
        truth,
        edge_indices,
        num_nodes=num_nodes,
        num_samples=16,
        seed=9,
        num_nodes_dict=num_nodes_dict,
    )

    typed = HeteroGraphKoopmanOperator(
        latent_dim,
        num_relations=2,
        node_types=node_types,
        edge_types=edge_types,
        init_mode="identity_noise",
        init_scale=0.05,
    )
    shared = HeteroGraphKoopmanOperator(
        latent_dim,
        num_relations=2,
        init_mode="identity_noise",
        init_scale=0.05,
    )
    _fit_hetero(
        typed,
        train_src,
        train_tgt,
        edge_indices,
        steps=450,
        lr=5e-2,
        seed=10,
        num_nodes_dict=num_nodes_dict,
    )
    _fit_hetero(
        shared,
        train_src,
        train_tgt,
        edge_indices,
        steps=450,
        lr=5e-2,
        seed=10,
    )

    assert torch.allclose(typed.k_self_for("a"), k_self["a"], atol=_FACTOR_ATOL)
    assert torch.allclose(typed.k_self_for("b"), k_self["b"], atol=_FACTOR_ATOL)
    for got, expected in zip(typed.K_relations, k_relations, strict=True):
        assert torch.allclose(got, expected, atol=_FACTOR_ATOL)

    mse_typed = _holdout_hetero_mse(
        typed,
        hold_src,
        hold_tgt,
        edge_indices,
        num_nodes_dict=num_nodes_dict,
    )
    mse_shared = _holdout_hetero_mse(shared, hold_src, hold_tgt, edge_indices)
    assert mse_typed < _RELATIONAL_MSE_ATOL
    assert mse_typed <= _BASELINE_MARGIN * mse_shared
