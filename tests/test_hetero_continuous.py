"""Tests for ContinuousHeteroGraphKoopmanOperator (continuous multiplex hetero)."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import HeteroData

from koopman_graph.data import HeteroGraphSnapshotSequence
from koopman_graph.model import GraphKoopmanModel
from koopman_graph.nn import GNNDecoder, GNNEncoder, RelGraphDecoder, RelGraphEncoder
from koopman_graph.operators import (
    ContinuousHeteroGraphKoopmanOperator,
    HeteroGraphKoopmanOperator,
    HypergraphKoopmanOperator,
)


def _relation_edge_indices() -> list[torch.Tensor]:
    """Return two small relation banks over 4 nodes."""
    r1 = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    r2 = torch.tensor([[0, 2], [2, 3]], dtype=torch.long)
    return [r1, r2]


def _multiplex_snapshot(*, num_nodes: int = 4, in_channels: int = 3) -> HeteroData:
    data = HeteroData()
    data["node"].x = torch.randn(num_nodes, in_channels)
    data["node", "r1", "node"].edge_index = torch.tensor(
        [[0, 1, 2], [1, 2, 0]],
        dtype=torch.long,
    )
    data["node", "r2", "node"].edge_index = torch.tensor(
        [[0, 2], [2, 3]],
        dtype=torch.long,
    )
    return data


def _continuous_hetero_model(
    *,
    latent_dim: int = 4,
    in_channels: int = 3,
    num_relations: int = 2,
    control_dim: int = 0,
    time_step: float = 0.2,
) -> GraphKoopmanModel:
    return GraphKoopmanModel(
        encoder=RelGraphEncoder(
            in_channels,
            hidden_channels=8,
            latent_dim=latent_dim,
            num_relations=num_relations,
            num_layers=1,
        ),
        decoder=RelGraphDecoder(
            latent_dim=latent_dim,
            hidden_channels=8,
            out_channels=in_channels,
            num_relations=num_relations,
            num_layers=1,
        ),
        latent_dim=latent_dim,
        time_step=time_step,
        dynamics_mode="continuous",
        koopman="hetero_graph",
        koopman_edge_types=(("node", "r1", "node"), ("node", "r2", "node")),
        control_dim=control_dim,
    )


def test_factory_builds_continuous_hetero_operator() -> None:
    """``koopman='hetero_graph'`` + continuous builds the generator operator."""
    model = _continuous_hetero_model()
    assert model.koopman_kind == "hetero_graph"
    assert model.dynamics_mode == "continuous"
    assert isinstance(model.koopman, ContinuousHeteroGraphKoopmanOperator)
    assert isinstance(model.encoder, RelGraphEncoder)
    assert isinstance(model.decoder, RelGraphDecoder)
    assert model.koopman.num_relations == 2
    assert model.koopman.normalization == "rgcn_in_degree"
    assert model.uses_hetero_koopman
    assert model.uses_continuous_hetero_koopman


def test_advance_matches_expm_effective_generator() -> None:
    """Dense advance matches vec⁻¹(exp(L_eff Δt) vec(Z)) for multiplex hetero."""
    torch.manual_seed(0)
    num_nodes = 4
    latent_dim = 3
    delta_t = 0.2
    edge_indices = _relation_edge_indices()
    operator = ContinuousHeteroGraphKoopmanOperator(
        latent_dim, num_relations=2, init_mode="xavier", init_scale=0.2
    )
    z = torch.randn(num_nodes, latent_dim)

    advanced = operator.advance(z, delta_t, edge_indices=edge_indices)
    transition = operator.transition_matrix(delta_t, edge_indices, num_nodes)
    expected = (transition @ z.reshape(-1)).view_as(z)
    assert torch.allclose(advanced, expected, atol=1e-5)

    generator = operator.effective_generator(edge_indices, num_nodes)
    expected_transition = torch.linalg.matrix_exp(generator * delta_t)
    assert torch.allclose(transition, expected_transition, atol=1e-5)


def test_matches_discrete_hetero_at_fixed_delta_t() -> None:
    """With L_r=0, continuous advance matches discrete K_self=exp(L_self Δt)."""
    torch.manual_seed(1)
    num_nodes = 4
    latent_dim = 3
    delta_t = 0.3
    edge_indices = _relation_edge_indices()

    l_self = torch.tensor([[-0.4, 0.1, 0.0], [-0.2, -0.5, 0.05], [0.0, 0.1, -0.3]])
    zeros = torch.zeros(latent_dim, latent_dim)

    continuous = ContinuousHeteroGraphKoopmanOperator(
        latent_dim, num_relations=2, init_mode="identity"
    )
    continuous.set_dense_matrices(l_self, [zeros, zeros])

    discrete = HeteroGraphKoopmanOperator(
        latent_dim, num_relations=2, init_mode="identity"
    )
    discrete.set_dense_matrices(
        torch.linalg.matrix_exp(l_self * delta_t), [zeros.clone(), zeros.clone()]
    )

    z = torch.randn(num_nodes, latent_dim)
    z_c = continuous.advance(z, delta_t, edge_indices=edge_indices)
    z_d = discrete.advance(z, edge_indices=edge_indices)
    assert torch.allclose(z_c, z_d, atol=1e-5)


def test_small_delta_t_first_order_matches_discrete_construction() -> None:
    """Small-Δt continuous advance matches discrete I+ΔtL construction.

    For tiny Δt the matrix exponential admits the expansion
    ``exp(Δt L_eff) = I + Δt L_eff + O((Δt ||L_eff||)^2)``. The discrete
    hetero one-step with ``K_self = I + Δt L_self`` and ``K_r = Δt L_r``
    realizes the same first-order map. With ``Δt = 1e-3`` the quadratic
    remainder is ~``1e-6`` times an ``O(1)`` Lipschitz factor of ``L_eff``
    on this tiny fixture, so ``atol=1e-5`` is a justified absolute tolerance
    (not a coverage or generalization claim).
    """
    torch.manual_seed(11)
    num_nodes = 4
    latent_dim = 3
    delta_t = 1e-3
    edge_indices = _relation_edge_indices()

    l_self = torch.tensor(
        [[-0.5, 0.2, 0.0], [-0.1, -0.4, 0.05], [0.0, 0.15, -0.35]],
    )
    l_r1 = 0.2 * torch.eye(latent_dim)
    l_r2 = torch.tensor(
        [[0.0, 0.1, 0.0], [-0.05, 0.0, 0.05], [0.0, -0.1, 0.0]],
    )

    continuous = ContinuousHeteroGraphKoopmanOperator(
        latent_dim, num_relations=2, init_mode="identity"
    )
    continuous.set_dense_matrices(l_self, [l_r1, l_r2])

    discrete = HeteroGraphKoopmanOperator(
        latent_dim, num_relations=2, init_mode="identity"
    )
    discrete.set_dense_matrices(
        torch.eye(latent_dim) + delta_t * l_self,
        [delta_t * l_r1, delta_t * l_r2],
    )

    z = torch.randn(num_nodes, latent_dim)
    z_c = continuous.advance(z, delta_t, edge_indices=edge_indices)
    z_d = discrete.advance(z, edge_indices=edge_indices)
    # atol justified above from O(Δt²) remainder of expm vs I+ΔtL.
    assert torch.allclose(z_c, z_d, atol=1e-5, rtol=0.0)

    # Direct first-order check against assembled L_eff (same remainder order).
    generator = continuous.effective_generator(edge_indices, num_nodes)
    first_order = (
        (torch.eye(num_nodes * latent_dim) + delta_t * generator) @ z.reshape(-1)
    ).view_as(z)
    assert torch.allclose(z_c, first_order, atol=1e-5, rtol=0.0)


def test_hetero_continuous_fit_predict_smoke() -> None:
    """Seeded uncontrolled multiplex fit yields finite loss and predict rolls out."""
    torch.manual_seed(2)
    sequence = HeteroGraphSnapshotSequence([_multiplex_snapshot() for _ in range(5)])
    model = _continuous_hetero_model()
    history = model.fit(sequence, epochs=2)
    assert len(history.loss) == 2
    assert all(torch.isfinite(torch.tensor(loss)) for loss in history.loss)
    model.eval()
    with torch.no_grad():
        preds = model.predict(sequence[0], steps=2)
    assert len(preds) == 2
    assert isinstance(preds[0], HeteroData)
    assert preds[0]["node"].x.shape == (4, 3)


def test_predict_at_irregular_deltas() -> None:
    """Irregular step_deltas round-trip through GraphKoopmanModel.predict_at."""
    torch.manual_seed(3)
    model = _continuous_hetero_model()
    origin = _multiplex_snapshot()
    model.eval()
    deltas = [0.1, 0.25, 0.15]
    with torch.no_grad():
        preds = model.predict_at(origin, step_deltas=deltas)
    assert len(preds) == len(deltas)
    for pred in preds:
        assert isinstance(pred, HeteroData)
        assert pred["node"].x.shape == (4, 3)


def test_typed_control_raises() -> None:
    """Typed + control_dim>0 is rejected (same restriction as discrete hetero)."""
    with pytest.raises(ValueError, match="control is unsupported for typed"):
        ContinuousHeteroGraphKoopmanOperator(
            3,
            num_relations=1,
            control_dim=1,
            node_types=("a", "b"),
            edge_types=(("a", "r0", "b"),),
        )


def test_compute_training_loss_clears_transition_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """compute_training_loss clears continuous hetero Φ cache at eval start."""
    from koopman_graph.training import LossWeights, compute_training_loss

    torch.manual_seed(4)
    model = _continuous_hetero_model()
    assert isinstance(model.koopman, ContinuousHeteroGraphKoopmanOperator)
    edge_indices = _relation_edge_indices()
    model.koopman.transition_matrix(0.1, edge_indices, 4)
    assert len(model.koopman._phi_cache) == 1
    hits = {"count": 0}
    original = model.koopman.clear_transition_cache

    def _spy() -> None:
        hits["count"] += 1
        original()

    monkeypatch.setattr(model.koopman, "clear_transition_cache", _spy)
    sequence = HeteroGraphSnapshotSequence([_multiplex_snapshot() for _ in range(3)])
    model.eval()
    with torch.no_grad():
        breakdown = compute_training_loss(
            model, sequence, LossWeights(reconstruction=1.0)
        )
    assert torch.isfinite(breakdown.total)
    assert hits["count"] == 1


def test_transition_cache_excluded_from_state_dict() -> None:
    """Φ / L_eff caches are ephemeral and absent from checkpoints."""
    operator = ContinuousHeteroGraphKoopmanOperator(3, num_relations=2)
    edge_indices = _relation_edge_indices()
    operator.transition_matrix(0.1, edge_indices, 4)
    assert len(operator._phi_cache) == 1
    assert len(operator._leff_cache) == 1
    state = operator.state_dict()
    assert all("phi" not in key and "leff" not in key for key in state)


def test_factory_rejects_continuous_hypergraph() -> None:
    """Continuous ``koopman='hypergraph'`` still raises (unimplemented)."""
    with pytest.raises(ValueError, match="dynamics_mode='discrete'"):
        GraphKoopmanModel(
            encoder=GNNEncoder(in_channels=3, hidden_channels=8, latent_dim=4),
            decoder=GNNDecoder(latent_dim=4, hidden_channels=8, out_channels=3),
            latent_dim=4,
            time_step=1.0,
            dynamics_mode="continuous",
            koopman="hypergraph",
        )


def test_factory_still_builds_discrete_hetero_by_default() -> None:
    """Discrete hetero (default dynamics_mode) is unaffected by the new operator."""
    model = GraphKoopmanModel(
        encoder=RelGraphEncoder(3, hidden_channels=8, latent_dim=4, num_relations=2),
        decoder=RelGraphDecoder(4, hidden_channels=8, out_channels=3, num_relations=2),
        latent_dim=4,
        time_step=1.0,
        koopman="hetero_graph",
    )
    assert isinstance(model.koopman, HeteroGraphKoopmanOperator)
    assert not isinstance(model.koopman, ContinuousHeteroGraphKoopmanOperator)
    assert not model.uses_continuous_hetero_koopman


def test_injected_continuous_hetero_requires_continuous_dynamics_mode() -> None:
    """Injected ContinuousHeteroGraphKoopmanOperator requires continuous mode."""
    injected = ContinuousHeteroGraphKoopmanOperator(4, num_relations=2)
    with pytest.raises(ValueError, match="dynamics_mode='continuous'"):
        GraphKoopmanModel(
            encoder=RelGraphEncoder(
                3, hidden_channels=8, latent_dim=4, num_relations=2
            ),
            decoder=RelGraphDecoder(
                4, hidden_channels=8, out_channels=3, num_relations=2
            ),
            latent_dim=4,
            time_step=1.0,
            dynamics_mode="discrete",
            koopman=injected,
        )


def test_model_spectrum_uses_generator() -> None:
    """Model-level spectrum for continuous hetero uses the effective generator."""
    model = _continuous_hetero_model()
    edge_indices = _relation_edge_indices()
    with pytest.raises(ValueError, match="edge_indices and num_nodes"):
        model.spectrum()
    spectrum = model.spectrum(edge_indices=edge_indices, num_nodes=4)
    assert spectrum.eigenvalues.shape == (16,)


def test_inverse_advance_uncontrolled_round_trip() -> None:
    """Dense uncontrolled inverse recovers the pre-image for multiplex hetero."""
    torch.manual_seed(5)
    num_nodes = 4
    latent_dim = 3
    delta_t = 0.2
    edge_indices = _relation_edge_indices()
    operator = ContinuousHeteroGraphKoopmanOperator(
        latent_dim, num_relations=2, init_mode="xavier", init_scale=0.1
    )
    z = torch.randn(num_nodes, latent_dim)
    advanced = operator.advance(z, delta_t, edge_indices=edge_indices)
    recovered = operator.inverse_advance(advanced, delta_t, edge_indices=edge_indices)
    assert torch.allclose(recovered, z, atol=1e-4)


def test_block_diagonal_ignores_relation_coupling() -> None:
    """BD sparsity advances with L_self only (self-dominated approximation)."""
    torch.manual_seed(6)
    num_nodes = 4
    latent_dim = 2
    delta_t = 0.2
    edge_indices = _relation_edge_indices()
    l_self = torch.tensor([[-0.5, 0.1], [0.0, -0.4]])
    l_rel = 0.5 * torch.eye(latent_dim)
    dense = ContinuousHeteroGraphKoopmanOperator(
        latent_dim, num_relations=2, init_mode="identity"
    )
    dense.set_dense_matrices(l_self, [l_rel, l_rel.clone()])
    block = ContinuousHeteroGraphKoopmanOperator(
        latent_dim, num_relations=2, init_mode="identity", sparsity="block_diagonal"
    )
    block.set_dense_matrices(l_self, [l_rel.clone(), l_rel.clone()])
    z = torch.randn(num_nodes, latent_dim)
    bd_next = block.advance(z, delta_t, edge_indices=edge_indices)
    self_only = ContinuousHeteroGraphKoopmanOperator(
        latent_dim, num_relations=2, init_mode="identity"
    )
    self_only.set_dense_matrices(
        l_self, [torch.zeros_like(l_rel), torch.zeros_like(l_rel)]
    )
    expected = self_only.advance(z, delta_t, edge_indices=edge_indices)
    assert torch.allclose(bd_next, expected, atol=1e-5)
    dense_next = dense.advance(z, delta_t, edge_indices=edge_indices)
    assert not torch.allclose(dense_next, bd_next, atol=1e-3)


def test_hurwitz_bound_metric_and_certificate() -> None:
    """Structural modes expose factor-level bound_metric / certificate."""
    operator = ContinuousHeteroGraphKoopmanOperator(
        3, num_relations=2, parameterization="lyapunov"
    )
    bound = operator.bound_metric()
    assert bound.ndim == 0
    assert torch.isfinite(bound)
    certificate = operator.stability_certificate()
    assert certificate is not None
    assert certificate.margin.item() > 0


def test_constructor_guards() -> None:
    """Constructor guards for sparsity / normalization / auxiliary_spectral."""
    dist = ContinuousHeteroGraphKoopmanOperator(
        3, num_relations=1, sparsity="distributed"
    )
    assert dist.sparsity == "distributed"
    with pytest.raises(ValueError, match="sparsity must be"):
        ContinuousHeteroGraphKoopmanOperator(3, num_relations=1, sparsity="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="auxiliary_spectral"):
        ContinuousHeteroGraphKoopmanOperator(
            3, num_relations=1, parameterization="auxiliary_spectral"
        )


def test_typed_operator_advance_and_effective_generator() -> None:
    """Typed operator advances via stacked block-diagonal self blocks."""
    torch.manual_seed(7)
    latent_dim = 2
    operator = ContinuousHeteroGraphKoopmanOperator(
        latent_dim,
        num_relations=1,
        node_types=("a", "b"),
        edge_types=(("a", "r0", "b"),),
        init_mode="xavier",
        init_scale=0.1,
    )
    num_nodes_dict = {"a": 2, "b": 3}
    num_nodes = 5
    edge_index = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    z = torch.randn(num_nodes, latent_dim)
    delta_t = 0.15
    advanced = operator.advance(
        z, delta_t, edge_indices=[edge_index], num_nodes_dict=num_nodes_dict
    )
    generator = operator.effective_generator(
        [edge_index], num_nodes, num_nodes_dict=num_nodes_dict
    )
    transition = torch.linalg.matrix_exp(generator * delta_t)
    expected = (transition @ z.reshape(-1)).view_as(z)
    assert torch.allclose(advanced, expected, atol=1e-5)


def test_typed_operator_requires_num_nodes_dict() -> None:
    """Typed operators reject calls missing ``num_nodes_dict``."""
    operator = ContinuousHeteroGraphKoopmanOperator(
        2,
        num_relations=1,
        node_types=("a", "b"),
        edge_types=(("a", "r0", "b"),),
    )
    edge_index = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    z = torch.randn(5, 2)
    with pytest.raises(ValueError, match="requires num_nodes_dict"):
        operator.advance(z, 0.1, edge_indices=[edge_index])


def test_eigenvalue_regularization_dense_hetero_continuous() -> None:
    """EigenvalueRegularizationLoss uses the topology-coupled generator."""
    from koopman_graph.training.objectives import compute_eigenvalue_regularization_loss

    torch.manual_seed(8)
    model = _continuous_hetero_model()
    sequence = HeteroGraphSnapshotSequence([_multiplex_snapshot() for _ in range(3)])
    penalty = compute_eigenvalue_regularization_loss(model, sequence)
    assert torch.isfinite(penalty)
    assert penalty.ndim == 0


def test_hypergraph_koopman_operator_unaffected_import() -> None:
    """Sanity: HypergraphKoopmanOperator remains importable alongside hetero peers."""
    assert HypergraphKoopmanOperator is not None


def _typed_continuous_operator(
    *,
    sparsity: str = "dense",
) -> ContinuousHeteroGraphKoopmanOperator:
    """Return a two-type shared-width operator for direct unit tests."""
    return ContinuousHeteroGraphKoopmanOperator(
        2,
        num_relations=1,
        node_types=("a", "b"),
        edge_types=(("a", "to", "b"),),
        sparsity=sparsity,  # type: ignore[arg-type]
        init_mode="identity",
    )


def _rectangular_continuous_operator() -> ContinuousHeteroGraphKoopmanOperator:
    """Return a two-type unequal-width operator for direct unit tests."""
    return ContinuousHeteroGraphKoopmanOperator(
        2,
        num_relations=2,
        node_types=("a", "b"),
        edge_types=(("a", "to_b", "b"), ("b", "to_a", "a")),
        latent_dims={"a": 2, "b": 3},
        init_mode="identity",
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"latent_dim": 0, "num_relations": 1}, "latent_dim must be positive"),
        ({"latent_dim": 2, "num_relations": 0}, "num_relations must be positive"),
        (
            {"latent_dim": 2, "num_relations": 1, "control_dim": -1},
            "control_dim must be non-negative",
        ),
        (
            {"latent_dim": 2, "num_relations": 1, "normalization": "symmetric"},
            "normalization must be one of",
        ),
    ],
)
def test_constructor_rejects_invalid_core_configuration(
    kwargs: dict[str, object],
    message: str,
) -> None:
    """Constructor rejects invalid dimensions and realization modes."""
    with pytest.raises(ValueError, match=message):
        ContinuousHeteroGraphKoopmanOperator(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "latent_dim": 2,
                "num_relations": 1,
                "latent_dims": {"node": 3},
            },
            "multiplex latent_dims must equal latent_dim",
        ),
        (
            {
                "latent_dim": 2,
                "num_relations": 1,
                "node_types": ("a", "b"),
                "edge_types": (("a", "to", "b"),),
                "latent_dims": {"a": 2, "b": 3},
                "relation_tying": "basis",
                "basis_size": 1,
            },
            "requires relation_tying='independent'",
        ),
        (
            {
                "latent_dim": 2,
                "num_relations": 1,
                "node_types": ("a", "b"),
                "edge_types": (("a", "to", "b"),),
                "latent_dims": {"a": 2, "b": 3},
                "parameterization": "lyapunov",
            },
            "requires parameterization='dense'",
        ),
        (
            {
                "latent_dim": 2,
                "num_relations": 1,
                "node_types": ("a", "b"),
                "edge_types": (("a", "to", "b"),),
                "latent_dims": {"a": 2, "b": 3},
                "sparsity": "block_diagonal",
            },
            "requires sparsity='dense'",
        ),
    ],
)
def test_constructor_rejects_invalid_latent_dims_configuration(
    kwargs: dict[str, object],
    message: str,
) -> None:
    """Unequal per-type widths enforce the rectangular-mode restrictions."""
    with pytest.raises(ValueError, match=message):
        ContinuousHeteroGraphKoopmanOperator(**kwargs)  # type: ignore[arg-type]


def test_typed_accessors_and_self_blocks() -> None:
    """Typed self accessors preserve type order and reject shared access."""
    operator = _typed_continuous_operator()
    l_a = torch.tensor([[-0.2, 0.1], [0.0, -0.3]])
    l_b = torch.tensor([[-0.4, 0.0], [0.2, -0.1]])
    operator.set_dense_matrices({"a": l_a, "b": l_b}, [torch.zeros(2, 2)])

    blocks = operator.typed_l_self_blocks({"a": 2, "b": 1})
    assert blocks.shape == (3, 2, 2)
    assert torch.equal(blocks[0], l_a)
    assert torch.equal(blocks[2], l_b)
    assert operator.d_for("a") == 2
    assert torch.equal(operator.l_self_for("b"), l_b)
    with pytest.raises(ValueError, match="L_self is undefined for typed"):
        _ = operator.L_self
    with pytest.raises(KeyError, match="unknown node type"):
        operator.self_operator_for("missing")
    with pytest.raises(KeyError, match="unknown node type"):
        operator.d_for("missing")


def test_pack_unpack_typed_latents_and_validation() -> None:
    """Typed latent packing round-trips and validates keys, counts, and shapes."""
    operator = _rectangular_continuous_operator()
    counts = {"a": 2, "b": 1}
    by_type = {
        "a": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "b": torch.tensor([[5.0, 6.0, 7.0]]),
    }
    packed = operator.pack_typed_latents(by_type, counts)
    unpacked = operator.unpack_typed_latents(packed, counts)
    assert torch.equal(unpacked["a"], by_type["a"])
    assert torch.equal(unpacked["b"], by_type["b"])
    assert operator.d_for("b") == 3
    with pytest.raises(ValueError, match="z_by_type keys"):
        operator.pack_typed_latents({"a": by_type["a"]}, counts)
    with pytest.raises(ValueError, match=r"must have shape \(2, 2\)"):
        operator.pack_typed_latents(
            {"a": torch.zeros(1, 2), "b": by_type["b"]},
            counts,
        )
    with pytest.raises(ValueError, match="z_flat must have shape"):
        operator.unpack_typed_latents(torch.zeros(6), counts)
    with pytest.raises(ValueError, match="keys must match"):
        operator.unpack_typed_latents(packed, {"a": 2, "c": 1})
    with pytest.raises(ValueError, match="must be positive"):
        operator.unpack_typed_latents(packed, {"a": 2, "b": 0})
    with pytest.raises(ValueError, match="undefined for rectangular"):
        operator.typed_l_self_blocks(counts)


def test_basis_tying_setters_relation_assembly_and_reset() -> None:
    """Basis factors assemble relation matrices and exercise basis-only guards."""
    operator = ContinuousHeteroGraphKoopmanOperator(
        2,
        num_relations=2,
        relation_tying="basis",
        basis_size=1,
        init_mode="identity",
    )
    basis = torch.tensor([[-0.2, 0.1], [0.0, -0.3]])
    coefficients = torch.tensor([[2.0], [-1.0]])
    operator.set_basis_factors([basis], coefficients)
    assert torch.allclose(operator.relation_matrix(0), 2.0 * basis)
    assert torch.allclose(operator.L_relations[1], -basis)
    with pytest.raises(ValueError, match="set_dense_matrices is unsupported"):
        operator.set_dense_matrices(torch.eye(2), [basis, basis])
    with pytest.raises(ValueError, match="only defined for relation_tying"):
        operator._relation_modules()
    with pytest.raises(ValueError, match="Expected 1 basis matrices"):
        operator.set_basis_factors([basis, basis], coefficients)
    with pytest.raises(ValueError, match="coefficients must have shape"):
        operator.set_basis_factors([basis], torch.ones(2, 2))
    with pytest.raises(IndexError, match="relation_index"):
        operator.relation_matrix(2)
    operator.reset_parameters()
    assert torch.isfinite(operator.bound_metric())

    independent = ContinuousHeteroGraphKoopmanOperator(2, num_relations=1)
    with pytest.raises(ValueError, match="requires relation_tying='basis'"):
        independent.set_basis_factors([basis], torch.ones(1, 1))
    with pytest.raises(ValueError, match="_basis_modules"):
        independent._basis_modules()


def test_typed_and_rectangular_dense_setters_and_guards() -> None:
    """Dense setters accept typed matrices and reject incompatible payloads."""
    typed = _typed_continuous_operator()
    l_self = {"a": -torch.eye(2), "b": -2.0 * torch.eye(2)}
    l_rel = torch.tensor([[0.0, 0.2], [0.1, 0.0]])
    typed.set_dense_matrices(l_self, [l_rel])
    assert torch.equal(typed.l_self_for("a"), l_self["a"])
    assert torch.equal(typed.relation_matrix(0), l_rel)
    with pytest.raises(ValueError, match="requires a mapping"):
        typed.set_dense_matrices(torch.eye(2), [l_rel])
    with pytest.raises(ValueError, match="l_self keys must match"):
        typed.set_dense_matrices({"a": torch.eye(2)}, [l_rel])
    with pytest.raises(ValueError, match="Expected 1 relation generators"):
        typed.set_dense_matrices(l_self, [])

    multiplex = ContinuousHeteroGraphKoopmanOperator(2, num_relations=1)
    with pytest.raises(ValueError, match="not a mapping"):
        multiplex.set_dense_matrices({"node": torch.eye(2)}, [l_rel])

    rectangular = _rectangular_continuous_operator()
    rect_self = {"a": -torch.eye(2), "b": -torch.eye(3)}
    relations = [torch.ones(2, 3), torch.ones(3, 2)]
    rectangular.set_dense_matrices(rect_self, relations)
    assert torch.equal(rectangular.relation_matrix(0), relations[0])
    assert torch.equal(rectangular.relation_matrix(1), relations[1])
    with pytest.raises(ValueError, match=r"must have shape \(2, 3\)"):
        rectangular.set_dense_matrices(
            rect_self,
            [torch.ones(2, 2), torch.ones(3, 2)],
        )
    with pytest.raises(ValueError, match="undefined for rectangular"):
        rectangular._relation_modules()


def test_effective_generator_guards_and_weighted_cache_keys() -> None:
    """Dense assembly validates overrides and distinguishes weighted cache keys."""
    operator = ContinuousHeteroGraphKoopmanOperator(
        2,
        num_relations=1,
        normalization="random_walk",
        init_mode="identity",
    )
    operator.set_dense_matrices(
        -torch.eye(2),
        [torch.tensor([[0.0, 0.2], [0.1, 0.0]])],
    )
    edges = [torch.tensor([[0, 1], [1, 0]], dtype=torch.long)]
    weights = [torch.tensor([1.0, 2.0])]
    generator = operator.effective_generator(
        edges,
        2,
        edge_weights=weights,
    )
    assert generator.shape == (4, 4)
    assert operator.effective_generator(edges, 2, edge_weights=weights) is generator
    changed = operator.effective_generator(
        edges,
        2,
        edge_weights=[torch.tensor([1.0, 3.0])],
    )
    assert changed is not generator
    with pytest.raises(ValueError, match="num_nodes must be positive"):
        operator.effective_generator(edges, 0)
    with pytest.raises(ValueError, match="at most one"):
        operator.effective_generator(
            edges,
            2,
            l_self=torch.eye(2),
            l_self_blocks=torch.eye(2).repeat(2, 1, 1),
        )
    with pytest.raises(ValueError, match="l_self_blocks must have shape"):
        operator.effective_generator(
            edges,
            2,
            l_self_blocks=torch.zeros(2, 2, 3),
        )
    with pytest.raises(ValueError, match="relation edge banks"):
        operator.effective_generator([], 2)
    with pytest.raises(ValueError, match="relation weight banks"):
        operator.effective_generator(edges, 2, edge_weights=[])
    operator.clear_transition_cache()
    assert not operator._leff_cache
    assert not operator._phi_cache


def test_rectangular_generator_forward_inverse_and_guards() -> None:
    """Rectangular dense generation handles weighted cross-type coupling."""
    operator = _rectangular_continuous_operator()
    counts = {"a": 1, "b": 1}
    l_self = {"a": -0.2 * torch.eye(2), "b": -0.3 * torch.eye(3)}
    relations = [
        torch.tensor([[0.1, 0.0, 0.2], [0.0, 0.1, 0.0]]),
        torch.tensor([[0.1, 0.0], [0.0, 0.2], [0.1, 0.1]]),
    ]
    operator.set_dense_matrices(l_self, relations)
    edges = [
        torch.tensor([[0], [1]], dtype=torch.long),
        torch.tensor([[1], [0]], dtype=torch.long),
    ]
    z = operator.pack_typed_latents(
        {"a": torch.tensor([[1.0, -1.0]]), "b": torch.tensor([[0.5, 0.0, 2.0]])},
        counts,
    )
    generator = operator.effective_generator(
        edges,
        2,
        edge_weights=[torch.tensor([2.0]), torch.tensor([0.5])],
        num_nodes_dict=counts,
    )
    assert generator.shape == (5, 5)
    advanced = operator.advance(
        z,
        0.1,
        edge_indices=edges,
        edge_weights=[torch.tensor([2.0]), torch.tensor([0.5])],
        num_nodes_dict=counts,
    )
    recovered = operator.inverse_advance(
        advanced,
        0.1,
        edge_indices=edges,
        edge_weights=[torch.tensor([2.0]), torch.tensor([0.5])],
        num_nodes_dict=counts,
    )
    assert torch.allclose(recovered, z, atol=1e-5, rtol=1e-5)
    precomputed = torch.linalg.matrix_exp(-0.1 * generator)
    assert torch.allclose(
        operator.inverse_advance(
            advanced,
            0.1,
            inverse_matrix=precomputed,
            edge_indices=edges,
            num_nodes_dict=counts,
        ),
        precomputed @ advanced,
    )
    with pytest.raises(ValueError, match="overrides are unsupported"):
        operator.effective_generator(
            edges,
            2,
            l_self=torch.eye(2),
            num_nodes_dict=counts,
        )
    with pytest.raises(ValueError, match="requires num_nodes_dict"):
        operator.effective_generator(edges, 2)
    with pytest.raises(ValueError, match="expects flat z"):
        operator(torch.zeros(2, 3), 0.1, edges, num_nodes_dict=counts)
    with pytest.raises(ValueError, match="control is unsupported"):
        operator(z, 0.1, edges, control=torch.ones(1), num_nodes_dict=counts)
    with pytest.raises(ValueError, match="requires num_nodes_dict"):
        operator.inverse_advance(z, 0.1, edge_indices=edges)
    with pytest.raises(ValueError, match="expects flat z"):
        operator.inverse_advance(
            torch.zeros(2, 3),
            0.1,
            edge_indices=edges,
            num_nodes_dict=counts,
        )


@pytest.mark.parametrize("per_node", [False, True])
def test_controlled_additive_dense_forward_inverse(per_node: bool) -> None:
    """Additive global and per-node controls exercise dense Van Loan paths."""
    operator = ContinuousHeteroGraphKoopmanOperator(
        2,
        num_relations=1,
        control_dim=1,
        init_mode="identity",
    )
    operator.set_dense_matrices(
        torch.tensor([[-0.3, 0.1], [0.0, -0.2]]),
        [torch.zeros(2, 2)],
        control_matrix=torch.tensor([[0.4, -0.2]]),
    )
    edges = [torch.tensor([[0, 1], [1, 0]], dtype=torch.long)]
    z = torch.tensor([[1.0, -0.5], [0.2, 0.8]])
    control = torch.tensor([[0.3], [-0.1]]) if per_node else torch.tensor([0.3])
    advanced = operator.advance(z, 0.2, edge_indices=edges, control=control)
    recovered = operator.inverse_advance(
        advanced,
        0.2,
        edge_indices=edges,
        control=control,
    )
    assert advanced.shape == z.shape
    assert torch.allclose(recovered, z, atol=1e-5, rtol=1e-5)
    with pytest.raises(ValueError, match="inverse_matrix is not supported"):
        operator.inverse_advance(
            advanced,
            0.2,
            edge_indices=edges,
            control=control,
            inverse_matrix=torch.eye(4),
        )


def test_controlled_dense_validation_and_bilinear_paths() -> None:
    """Controlled dense paths validate missing and malformed controls."""
    edges = [torch.tensor([[0, 1], [1, 0]], dtype=torch.long)]
    z = torch.randn(2, 2)
    additive = ContinuousHeteroGraphKoopmanOperator(
        2,
        num_relations=1,
        control_dim=1,
    )
    with pytest.raises(ValueError, match="control input is required"):
        additive.advance(z, 0.1, edge_indices=edges)
    with pytest.raises(ValueError, match="control input is required"):
        additive.inverse_advance(z, 0.1, edge_indices=edges)
    with pytest.raises(ValueError, match="control input must have shape"):
        additive.advance(z, 0.1, edge_indices=edges, control=torch.ones(1, 1, 1))
    with pytest.raises(ValueError, match="control input must have shape"):
        additive.inverse_advance(
            z,
            0.1,
            edge_indices=edges,
            control=torch.ones(1, 1, 1),
        )

    bilinear = ContinuousHeteroGraphKoopmanOperator(
        2,
        num_relations=1,
        control_dim=1,
        control_mode="bilinear",
    )
    global_out = bilinear.advance(
        z,
        0.1,
        edge_indices=edges,
        control=torch.tensor([0.2]),
    )
    node_out = bilinear.advance(
        z,
        0.1,
        edge_indices=edges,
        control=torch.tensor([[0.2], [-0.1]]),
    )
    assert global_out.shape == z.shape
    assert node_out.shape == z.shape
    uncontrolled = ContinuousHeteroGraphKoopmanOperator(2, num_relations=1)
    with pytest.raises(ValueError, match="uncontrolled operator"):
        uncontrolled._networked_control_matrix(2)
    with pytest.raises(ValueError, match="uncontrolled operator"):
        uncontrolled.advance(z, 0.1, edge_indices=edges, control=torch.ones(1))


def test_forward_advance_and_inverse_validation() -> None:
    """Public propagation entry points reject missing arguments and bad shapes."""
    operator = ContinuousHeteroGraphKoopmanOperator(2, num_relations=1)
    edges = [torch.tensor([[0], [1]], dtype=torch.long)]
    z = torch.randn(2, 2)
    with pytest.raises(ValueError, match="delta_t is required"):
        operator.advance(z, edge_indices=edges)
    with pytest.raises(ValueError, match="edge_indices is required"):
        operator.advance(z, 0.1)
    with pytest.raises(ValueError, match="expects z with shape"):
        operator(torch.randn(4), 0.1, edges)
    with pytest.raises(ValueError, match="delta_t is required"):
        operator.inverse_advance(z, edge_indices=edges)
    with pytest.raises(ValueError, match="edge_indices is required"):
        operator.inverse_advance(z, 0.1)
    with pytest.raises(ValueError, match="expects z with shape"):
        operator.inverse_advance(torch.randn(4), 0.1, edge_indices=edges)
    assert operator(z, 0.0, edges) is z


def test_typed_dense_and_block_diagonal_inverse_round_trips() -> None:
    """Typed dense and self-only block realizations both invert their advances."""
    edges = [torch.tensor([[0, 1], [2, 3]], dtype=torch.long)]
    counts = {"a": 2, "b": 2}
    z = torch.randn(4, 2)
    l_self = {
        "a": torch.tensor([[-0.2, 0.1], [0.0, -0.3]]),
        "b": torch.tensor([[-0.4, 0.0], [0.1, -0.2]]),
    }
    for sparsity in ("dense", "block_diagonal"):
        operator = _typed_continuous_operator(sparsity=sparsity)
        operator.set_dense_matrices(l_self, [torch.zeros(2, 2)])
        advanced = operator.advance(
            z,
            0.15,
            edge_indices=edges,
            num_nodes_dict=counts,
        )
        recovered = operator.inverse_advance(
            advanced,
            0.15,
            edge_indices=edges,
            num_nodes_dict=counts,
        )
        assert torch.allclose(recovered, z, atol=1e-5, rtol=1e-5)
    with pytest.raises(ValueError, match="inverse_matrix is only supported"):
        operator.inverse_advance(
            z,
            0.1,
            edge_indices=edges,
            num_nodes_dict=counts,
            inverse_matrix=torch.eye(8),
        )
    with pytest.raises(ValueError, match="sums to"):
        operator.advance(
            z,
            0.1,
            edge_indices=edges,
            num_nodes_dict={"a": 1, "b": 1},
        )


def test_rectangular_bound_metric_and_zero_delta() -> None:
    """Rectangular factor monitoring includes relation norms and zero-step identity."""
    operator = _rectangular_continuous_operator()
    counts = {"a": 1, "b": 1}
    relations = [torch.full((2, 3), 2.0), torch.zeros(3, 2)]
    operator.set_dense_matrices(
        {"a": -torch.eye(2), "b": -torch.eye(3)},
        relations,
    )
    metric = operator.bound_metric()
    assert metric >= torch.linalg.matrix_norm(relations[0], ord=2)
    edges = [
        torch.tensor([[0], [1]], dtype=torch.long),
        torch.tensor([[1], [0]], dtype=torch.long),
    ]
    z = operator.pack_typed_latents(
        {"a": torch.ones(1, 2), "b": torch.ones(1, 3)},
        counts,
    )
    assert operator(z, 0.0, edges, num_nodes_dict=counts) is z
    operator.reset_parameters()
    assert torch.isfinite(operator.bound_metric())


def test_miscellaneous_accessors_resets_and_private_guards() -> None:
    """Small contract surfaces and defensive relation helpers remain covered."""
    multiplex = ContinuousHeteroGraphKoopmanOperator(2, num_relations=1)
    assert multiplex.B is None
    assert torch.equal(multiplex.matrix, multiplex.L_self)
    assert torch.equal(multiplex.L, multiplex.L_self)
    assert multiplex._require_num_nodes_dict(
        {"node": 2},
        num_nodes=2,
        caller="test",
    ) == {"node": 2}

    typed = _typed_continuous_operator()
    counts = {"a": 1, "b": 1}
    packed = typed.pack_typed_latents(
        {"a": torch.ones(1, 2), "b": torch.zeros(1, 2)},
        counts,
    )
    unpacked = typed.unpack_typed_latents(packed, counts)
    assert torch.equal(unpacked["a"], torch.ones(1, 2))

    edge = torch.tensor([[0], [1]], dtype=torch.long)
    other_edge = torch.tensor([[1], [0]], dtype=torch.long)
    assert not multiplex._relation_banks_equal([edge], [None], [], [])
    assert not multiplex._relation_banks_equal(
        [edge],
        [None],
        [other_edge],
        [None],
    )
    assert not multiplex._relation_banks_equal(
        [edge],
        [None],
        [edge],
        [torch.ones(1)],
    )

    rectangular = ContinuousHeteroGraphKoopmanOperator(
        2,
        num_relations=1,
        node_types=("a", "b"),
        edge_types=(("a", "to", "b"),),
        latent_dims={"a": 2, "b": 3},
        init_mode="xavier",
    )
    rectangular.reset_parameters()
    with pytest.raises(ValueError, match="shared-d only"):
        rectangular._relation_coupling_generator(
            [edge],
            2,
            [None],
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
    z_rect = rectangular.pack_typed_latents(
        {"a": torch.ones(1, 2), "b": torch.ones(1, 3)},
        counts,
    )
    with pytest.raises(ValueError, match="forward requires num_nodes_dict"):
        rectangular(z_rect, 0.1, [edge])
    with pytest.raises(ValueError, match="control is unsupported"):
        rectangular.inverse_advance(
            z_rect,
            0.1,
            edge_indices=[edge],
            control=torch.ones(1),
            num_nodes_dict=counts,
        )

    controlled = ContinuousHeteroGraphKoopmanOperator(
        2,
        num_relations=1,
        control_dim=1,
    )
    controlled.reset_parameters()
    assert controlled.B is not None
    multiplex.reset_parameters()

    block = ContinuousHeteroGraphKoopmanOperator(
        2,
        num_relations=1,
        sparsity="block_diagonal",
    )
    z = torch.randn(2, 2)
    advanced = block.advance(z, 0.1, edge_indices=[edge])
    recovered = block.inverse_advance(advanced, 0.1, edge_indices=[edge])
    assert torch.allclose(recovered, z, atol=1e-5, rtol=1e-5)
    with pytest.raises(ValueError, match="uncontrolled operator"):
        multiplex.inverse_advance(
            z,
            0.1,
            edge_indices=[edge],
            control=torch.ones(1),
        )
