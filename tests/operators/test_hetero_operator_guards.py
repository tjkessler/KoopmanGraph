"""Coverage and error-path tests for :mod:`koopman_graph.operators`."""

from __future__ import annotations

import pytest
import torch

from koopman_graph.operators.heterogeneous import (
    HeteroGraphKoopmanOperator,
    _normalize_edge_types,
    _normalize_node_types,
)


def test_hetero_operator_constructor_validation() -> None:
    """Constructor rejects bad dims, tying, normalization, and type metadata."""
    with pytest.raises(ValueError, match="latent_dim must be positive"):
        HeteroGraphKoopmanOperator(latent_dim=0, num_relations=1)
    with pytest.raises(ValueError, match="num_relations must be positive"):
        HeteroGraphKoopmanOperator(latent_dim=2, num_relations=0)
    with pytest.raises(ValueError, match="control_dim must be non-negative"):
        HeteroGraphKoopmanOperator(latent_dim=2, num_relations=1, control_dim=-1)
    with pytest.raises(ValueError, match="normalization must be one of"):
        HeteroGraphKoopmanOperator(
            latent_dim=2,
            num_relations=1,
            normalization="symmetric",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="must be a positive int"):
        HeteroGraphKoopmanOperator(
            latent_dim=2,
            num_relations=2,
            relation_tying="basis",
            basis_size=0,
        )
    with pytest.raises(ValueError, match="at least one node type"):
        _normalize_node_types(())
    with pytest.raises(ValueError, match="non-empty strings"):
        _normalize_node_types(("", "a"))
    with pytest.raises(ValueError, match="must be unique"):
        _normalize_node_types(("a", "a"))
    with pytest.raises(ValueError, match="edge_types is required for typed"):
        HeteroGraphKoopmanOperator(
            latent_dim=2,
            num_relations=1,
            node_types=("a", "b"),
        )
    with pytest.raises(ValueError, match=r"\(src, rel, dst\)"):
        _normalize_edge_types(
            [("a", "b")],
            num_relations=1,
            node_types=("a", "b"),
        )
    with pytest.raises(ValueError, match="non-empty strings"):
        _normalize_edge_types(
            [("", "r", "b")],
            num_relations=1,
            node_types=("a", "b"),
        )
    with pytest.raises(ValueError, match="must match"):
        _normalize_edge_types(
            [("a", "r0", "b")],
            num_relations=2,
            node_types=("a", "b"),
        )
    with pytest.raises(ValueError, match="unique after key encoding"):
        _normalize_edge_types(
            [("a", "r0", "b"), ("a", "r0", "b")],
            num_relations=2,
            node_types=("a", "b"),
        )
    with pytest.raises(ValueError, match="outside node_types"):
        _normalize_edge_types(
            [("a", "r0", "ghost")],
            num_relations=1,
            node_types=("a", "b"),
        )


def test_hetero_operator_runtime_validation_and_basis_paths() -> None:
    """Forward / effective / basis helpers reject invalid overrides."""
    edges = [
        torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        torch.tensor([[0], [1]], dtype=torch.long),
    ]
    op = HeteroGraphKoopmanOperator(latent_dim=2, num_relations=2)
    z = torch.randn(3, 2)
    with pytest.raises(ValueError, match="expects z with shape"):
        op(torch.randn(3), edges)
    with pytest.raises(ValueError, match="Expected trailing dimension"):
        op(torch.randn(3, 3), edges)
    with pytest.raises(ValueError, match="relation weight banks"):
        op(z, edges, edge_weights=[None])

    with pytest.raises(ValueError, match="at most one of k_self"):
        op.effective_matrix(
            edges,
            num_nodes=3,
            k_self=torch.eye(2),
            k_self_blocks=torch.eye(2).expand(3, 2, 2).clone(),
        )
    with pytest.raises(ValueError, match="k_self_blocks must have shape"):
        op.effective_matrix(
            edges,
            num_nodes=3,
            k_self_blocks=torch.zeros(3, 2, 3),
        )

    block = HeteroGraphKoopmanOperator(
        latent_dim=2,
        num_relations=1,
        sparsity="block_diagonal",
    )
    block_edges = [torch.tensor([[0], [1]], dtype=torch.long)]
    with pytest.raises(ValueError, match="requires sparsity='dense'"):
        block.dense_effective_inverse(block_edges, num_nodes=2)

    basis = HeteroGraphKoopmanOperator(
        latent_dim=2,
        num_relations=2,
        relation_tying="basis",
        basis_size=1,
    )
    with pytest.raises(ValueError, match="requires relation_tying='basis'"):
        op.set_basis_factors([torch.eye(2)], torch.ones(2, 1))
    with pytest.raises(ValueError, match="coefficients must have shape"):
        basis.set_basis_factors([torch.eye(2)], torch.ones(2, 2))
    with pytest.raises(ValueError, match="only defined for relation_tying"):
        basis._relation_modules()  # noqa: SLF001
    metric = basis.bound_metric()
    assert torch.isfinite(metric).all()

    control_op = HeteroGraphKoopmanOperator(
        latent_dim=2,
        num_relations=1,
        control_dim=1,
        control_mode="bilinear",
    )
    edge = [torch.tensor([[0, 1], [1, 0]], dtype=torch.long)]
    z_c = torch.randn(2, 2)
    nxt = control_op(z_c, edge, control=torch.ones(2, 1))
    back = control_op.inverse_advance(
        nxt,
        edge_indices=edge,
        control=torch.ones(2, 1),
    )
    assert back.shape == z_c.shape
    with pytest.raises(ValueError, match="only supported for"):
        block.inverse_advance(
            torch.randn(2, 2),
            edge_indices=block_edges,
            inverse_matrix=torch.eye(4),
        )
