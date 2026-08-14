"""Coverage and error-path tests for :mod:`koopman_graph.operators`."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data

from koopman_graph import (
    GNNDecoder,
    GNNEncoder,
    GraphKoopmanModel,
)
from koopman_graph.data import GraphSnapshotSequence


def _tiny_model(
    *,
    latent_dim: int = 4,
    control_dim: int = 0,
    dynamics_mode: str = "discrete",
    parameterization: str = "dense",
    physics_dim: int = 0,
    physics_preset: str | None = None,
) -> GraphKoopmanModel:
    gnn_dim = latent_dim - physics_dim
    encoder = GNNEncoder(2, 8, gnn_dim)
    decoder = GNNDecoder(latent_dim, 8, 2)
    return GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=latent_dim,
        time_step=0.1,
        control_dim=control_dim,
        dynamics_mode=dynamics_mode,
        koopman_parameterization=parameterization,
        physics_dim=physics_dim,
        physics_preset=physics_preset,
    )


def test_v06_operator_and_uq_coverage_gaps() -> None:
    """Hit remaining 0.6.0 error/control paths for the 90% coverage gate."""
    from koopman_graph.operators import (
        ContinuousGraphKoopmanOperator,
        HypergraphKoopmanOperator,
    )
    from koopman_graph.operators.graph_inverse import (
        _invert_square,
        apply_self_inverse,
        block_diagonal_graph_inverse_advance,
    )
    from koopman_graph.training import compute_eigenvalue_regularization_loss
    from koopman_graph.uq import ConformalKoopmanUQ
    from koopman_graph.uq.conformal import _nonconformity_score, _split_quantile

    # --- graph_inverse helpers ---
    assert torch.allclose(_invert_square(torch.eye(2)), torch.eye(2))
    singular = torch.zeros(2, 2)
    pinv = _invert_square(singular)
    assert pinv.shape == (2, 2)

    rhs = torch.randn(3, 2)
    recovered = apply_self_inverse(rhs, k_self=torch.eye(2))
    assert torch.allclose(recovered, rhs, atol=1e-5)
    blocks = torch.eye(2).expand(3, 2, 2).clone()
    recovered_b = apply_self_inverse(rhs, k_self_blocks=blocks)
    assert torch.allclose(recovered_b, rhs, atol=1e-5)
    singular_blocks = torch.zeros(3, 2, 2)
    fallback = apply_self_inverse(rhs, k_self_blocks=singular_blocks)
    assert fallback.shape == rhs.shape

    with pytest.raises(ValueError, match="exactly one"):
        apply_self_inverse(rhs)
    with pytest.raises(ValueError, match="exactly one"):
        apply_self_inverse(rhs, k_self=torch.eye(2), k_self_blocks=blocks)
    with pytest.raises(ValueError, match="rhs must have shape"):
        apply_self_inverse(torch.randn(2), k_self=torch.eye(2))
    with pytest.raises(ValueError, match="k_self must have shape"):
        apply_self_inverse(rhs, k_self=torch.eye(3))
    with pytest.raises(ValueError, match="k_self_blocks must have shape"):
        apply_self_inverse(rhs, k_self_blocks=torch.eye(2).expand(2, 2, 2))

    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    z_adj = torch.randn(3, 2)
    approx = block_diagonal_graph_inverse_advance(
        z_adj,
        k_self=torch.eye(2),
        k_nbr=0.1 * torch.eye(2),
        edge_index=edge_index,
    )
    assert approx.shape == z_adj.shape
    approx_b = block_diagonal_graph_inverse_advance(
        z_adj,
        k_self=torch.eye(2),
        k_nbr=0.1 * torch.eye(2),
        edge_index=edge_index,
        k_self_blocks=blocks,
    )
    assert approx_b.shape == z_adj.shape

    # --- continuous_graph controlled dense advance / inverse ---
    num_nodes = 3
    latent_dim = 2
    path = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    z = torch.randn(num_nodes, latent_dim)
    cg = ContinuousGraphKoopmanOperator(latent_dim, control_dim=1, init_mode="identity")
    with pytest.raises(ValueError, match="control input is required"):
        cg.advance(z, 0.1, edge_index=path)
    with pytest.raises(ValueError, match="control input provided"):
        ContinuousGraphKoopmanOperator(latent_dim, init_mode="identity").advance(
            z, 0.1, edge_index=path, control=torch.ones(1)
        )
    global_u = torch.tensor([0.2])
    per_node_u = torch.full((num_nodes, 1), 0.2)
    z_g = cg.advance(z, 0.1, edge_index=path, control=global_u)
    z_n = cg.advance(z, 0.1, edge_index=path, control=per_node_u)
    assert z_g.shape == z.shape and z_n.shape == z.shape
    with pytest.raises(ValueError, match="control input must have shape"):
        cg.advance(z, 0.1, edge_index=path, control=torch.ones(2, 2, 1))

    z_inv_g = cg.inverse_advance(z_g, 0.1, edge_index=path, control=global_u)
    z_inv_n = cg.inverse_advance(z_n, 0.1, edge_index=path, control=per_node_u)
    assert z_inv_g.shape == z.shape and z_inv_n.shape == z.shape
    with pytest.raises(ValueError, match="delta_t is required"):
        cg.advance(z, None, edge_index=path, control=global_u)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="edge_index is required"):
        cg.advance(z, 0.1, control=global_u)
    with pytest.raises(ValueError, match="delta_t is required"):
        cg.inverse_advance(z, None, edge_index=path, control=global_u)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="edge_index is required"):
        cg.inverse_advance(z, 0.1, control=global_u)
    with pytest.raises(ValueError, match="expects z with"):
        cg.inverse_advance(torch.randn(2), 0.1, edge_index=path, control=global_u)
    with pytest.raises(ValueError, match="control input is required"):
        cg.inverse_advance(z, 0.1, edge_index=path)
    with pytest.raises(ValueError, match="inverse_matrix is not supported"):
        cg.inverse_advance(
            z, 0.1, edge_index=path, control=global_u, inverse_matrix=torch.eye(6)
        )
    with pytest.raises(ValueError, match="control input must have shape"):
        cg.inverse_advance(z, 0.1, edge_index=path, control=torch.ones(2, 2, 1))

    bd = ContinuousGraphKoopmanOperator(
        latent_dim, sparsity="block_diagonal", init_mode="identity"
    )
    assert bd.advance(z, 0.1, edge_index=path).shape == z.shape
    with pytest.raises(ValueError, match="inverse_matrix is only supported"):
        bd.inverse_advance(z, 0.1, edge_index=path, inverse_matrix=torch.eye(2))
    assert bd.inverse_advance(z, 0.1, edge_index=path).shape == z.shape

    bilinear = ContinuousGraphKoopmanOperator(
        latent_dim,
        control_dim=1,
        control_mode="bilinear",
        init_mode="identity",
    )
    z_bi = bilinear.advance(z, 0.1, edge_index=path, control=per_node_u)
    assert z_bi.shape == z.shape

    # --- eigenvalue regularization for continuous_graph / hypergraph dense ---
    enc = GNNEncoder(2, 4, 2, num_layers=1)
    dec = GNNDecoder(2, 4, 2, num_layers=1)
    cg_model = GraphKoopmanModel(
        enc,
        dec,
        latent_dim=2,
        time_step=0.1,
        koopman="continuous_graph",
        dynamics_mode="continuous",
    )
    assert isinstance(cg_model.koopman, ContinuousGraphKoopmanOperator)
    with pytest.raises(ValueError, match="sequence is required"):
        compute_eigenvalue_regularization_loss(cg_model)
    snaps = [Data(x=torch.randn(3, 2), edge_index=path) for _ in range(3)]
    seq = GraphSnapshotSequence(snaps)
    assert torch.isfinite(compute_eigenvalue_regularization_loss(cg_model, seq))

    hyp_model = GraphKoopmanModel(
        GNNEncoder(2, 4, 2, num_layers=1),
        GNNDecoder(2, 4, 2, num_layers=1),
        latent_dim=2,
        time_step=0.1,
        koopman="hypergraph",
    )
    assert isinstance(hyp_model.koopman, HypergraphKoopmanOperator)
    with pytest.raises(ValueError, match="sequence is required"):
        compute_eigenvalue_regularization_loss(hyp_model)
    with pytest.raises(ValueError, match="hyperedge-carrying"):
        compute_eigenvalue_regularization_loss(hyp_model, seq)
    he_index = torch.tensor(
        [[0, 1, 2, 0, 1], [0, 0, 0, 1, 1]],
        dtype=torch.long,
    )
    hyp_snaps = [
        Data(
            x=torch.randn(3, 2),
            edge_index=path,
            hyperedge_index=he_index,
        )
        for _ in range(3)
    ]
    hyp_seq = GraphSnapshotSequence(hyp_snaps)
    assert torch.isfinite(compute_eigenvalue_regularization_loss(hyp_model, hyp_seq))

    # --- conformal validation branches ---
    tiny = _tiny_model(latent_dim=2)
    with pytest.raises(ValueError, match="method must be"):
        ConformalKoopmanUQ(tiny, method="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="score must be"):
        ConformalKoopmanUQ(tiny, score="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="gamma must be positive"):
        ConformalKoopmanUQ(tiny, gamma=0.0)
    with pytest.raises(ValueError, match="score must be"):
        _nonconformity_score(torch.zeros(2, 2), torch.ones(2, 2), "bogus")  # type: ignore[arg-type]
    assert _nonconformity_score(torch.zeros(2, 2), torch.ones(2, 2), "per_node") > 0
    with pytest.raises(ValueError, match="at least one"):
        _split_quantile(torch.tensor([]), 0.1)

    uq = ConformalKoopmanUQ(tiny, method="adaptive", score="per_node")
    with pytest.raises(RuntimeError, match="not calibrated"):
        uq.predict_interval(seq[0], steps=1)
    with pytest.raises(ValueError, match="steps must be"):
        uq.calibrate([], steps=0, alpha=0.1)
    with pytest.raises(ValueError, match="alpha must lie"):
        uq.calibrate([seq], steps=1, alpha=0.0)
    with pytest.raises(ValueError, match="non-empty"):
        uq.calibrate([], steps=1, alpha=0.1)
    short = GraphSnapshotSequence(snaps[:1])
    with pytest.raises(ValueError, match="need >="):
        uq.calibrate([short], steps=2, alpha=0.1)
    with pytest.raises(ValueError, match="controls must align"):
        uq.calibrate([seq], steps=1, alpha=0.1, controls=[])
    with pytest.raises(ValueError, match="future_topologies must align"):
        uq.calibrate([seq], steps=1, alpha=0.1, future_topologies=[])

    def _zero_predict(initial_graph, steps, **_kwargs):
        template = initial_graph if isinstance(initial_graph, Data) else seq[0]
        assert template.x is not None
        return [
            Data(x=torch.zeros_like(template.x), edge_index=template.edge_index)
            for _ in range(steps)
        ]

    tiny.predict = _zero_predict  # type: ignore[method-assign]
    uq.calibrate([seq], steps=1, alpha=0.1)
    with pytest.raises(ValueError, match="steps must be"):
        uq.predict_interval(seq[0], steps=0)
    with pytest.raises(ValueError, match="exceeds calibrated"):
        uq.predict_interval(seq[0], steps=2)
    with pytest.raises(ValueError, match="level must lie"):
        uq.predict_interval(seq[0], steps=1, level=0.0)
    with pytest.raises(ValueError, match="does not match"):
        uq.predict_interval(seq[0], steps=1, level=0.5)
