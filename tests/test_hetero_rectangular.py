"""Rectangular discrete hetero + RelGraph heads (TASK-1815 / TASK-1816 / TASK-1819).

Orientation lock (DESIGN Appendix B / Q2=A)
-------------------------------------------
Relation factors store ``K_r ∈ R^{d_src × d_dst}``. Sparse advance uses
``(Â Z_src) @ K_r``; dense ``K_eff`` relation blocks use
``Â_{dst←src} ⊗ K_r.T`` under the flat ``Σ N_τ·d_τ`` layout. Self blocks are
``I_{N_τ} ⊗ K_self^τ`` (row-state ``Z @ K_self.T``).
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from koopman_graph.analysis import attribute_mode_energy
from koopman_graph.data import (
    latent_type_slices_from_dims,
    node_type_offsets,
    stacked_latent_numel,
)
from koopman_graph.graph_utils import dense_relation_normalized_adjacency
from koopman_graph.losses import EigenvalueRegularizationLoss
from koopman_graph.model import GraphKoopmanModel
from koopman_graph.model.factory import resolve_model_components
from koopman_graph.nn import RelGraphDecoder, RelGraphEncoder
from koopman_graph.operators import HeteroGraphKoopmanOperator

NODE_TYPES = ("a", "b")
EDGE_TYPES = (
    ("a", "to_b", "b"),
    ("b", "to_a", "a"),
)
NUM_NODES = {"a": 2, "b": 3}
LATENT_DIMS = {"a": 2, "b": 3}
SHARED_D = 4

# Float32 Kronecker path (match shared-d hetero oracle tolerance).
_ATOL = 1e-5


def _rectangular_operator() -> HeteroGraphKoopmanOperator:
    """Return a typed rectangular hetero operator (unequal d_τ)."""
    return HeteroGraphKoopmanOperator(
        SHARED_D,
        num_relations=len(EDGE_TYPES),
        node_types=NODE_TYPES,
        edge_types=EDGE_TYPES,
        latent_dims=LATENT_DIMS,
        init_mode="identity",
        parameterization="dense",
        sparsity="dense",
    )


def _oracle_factors() -> tuple[dict[str, Tensor], list[Tensor]]:
    """Return deterministic self / relation factors for the oracle fixture."""
    k_self = {
        "a": torch.tensor([[0.7, 0.1], [0.0, 0.8]]),
        "b": torch.tensor(
            [
                [0.6, 0.0, 0.1],
                [0.0, 0.5, 0.0],
                [0.2, 0.0, 0.4],
            ]
        ),
    }
    # Appendix B: K_r ∈ R^{d_src × d_dst}.
    k_ab = torch.tensor(
        [
            [0.1, 0.0, 0.2],
            [0.0, 0.3, 0.0],
        ]
    )
    k_ba = torch.tensor(
        [
            [0.2, 0.0],
            [0.0, 0.1],
            [0.1, 0.2],
        ]
    )
    return k_self, [k_ab, k_ba]


def _global_edges() -> list[Tensor]:
    """Return global edge banks for a->b and b->a on the stacked layout."""
    # Stacked nodes: a0,a1,b0,b1,b2  (offsets a=0, b=2)
    edges_ab = torch.tensor([[0, 1], [2, 4]], dtype=torch.long)  # a0->b0, a1->b2
    edges_ba = torch.tensor([[2, 3], [0, 1]], dtype=torch.long)  # b0->a0, b1->a1
    return [edges_ab, edges_ba]


def _oracle_typed_latents() -> dict[str, Tensor]:
    """Deterministic per-type latents for the rectangular construction oracle."""
    return {
        "a": torch.tensor([[1.0, 0.5], [0.0, 2.0]]),
        "b": torch.tensor(
            [
                [0.3, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [0.0, 0.5, 0.5],
            ]
        ),
    }


def _hand_built_rectangular_effective(
    k_self: dict[str, Tensor],
    k_relations: list[Tensor],
    edge_indices: list[Tensor],
    *,
    transpose_relation: bool = True,
    normalization: str = "rgcn_in_degree",
) -> Tensor:
    """Assemble oracle rectangular ``K_eff`` without calling the operator.

    Self blocks: ``I_{N_τ} ⊗ K_self^τ`` on flat type slices.
    Relation blocks (Q2=A / Appendix B): for ``K_r ∈ R^{d_src×d_dst}``,
    ``Â_{dst←src} ⊗ K_r.T`` when ``transpose_relation`` is True (correct).
    Setting ``transpose_relation=False`` intentionally uses ``Â ⊗ K_r`` so
    flip-detection tests can prove the convention is not silent.
    """
    slices = latent_type_slices_from_dims(NODE_TYPES, NUM_NODES, LATENT_DIMS)
    offsets = node_type_offsets(NODE_TYPES, NUM_NODES)
    num_nodes = sum(NUM_NODES.values())
    total = stacked_latent_numel(NODE_TYPES, NUM_NODES, LATENT_DIMS)
    ref = k_self[NODE_TYPES[0]]
    effective = torch.zeros((total, total), dtype=ref.dtype)
    for name in NODE_TYPES:
        identity = torch.eye(NUM_NODES[name], dtype=ref.dtype)
        block = torch.kron(identity, k_self[name])
        type_slice = slices[name]
        effective[type_slice, type_slice] = effective[type_slice, type_slice] + block

    for relation_idx, edge_index in enumerate(edge_indices):
        adj = dense_relation_normalized_adjacency(
            edge_index,
            num_nodes,
            dtype=ref.dtype,
            normalization=normalization,  # type: ignore[arg-type]
        )
        src, _rel, dst = EDGE_TYPES[relation_idx]
        src_nodes = slice(offsets[src], offsets[src] + NUM_NODES[src])
        dst_nodes = slice(offsets[dst], offsets[dst] + NUM_NODES[dst])
        adj_block = adj[dst_nodes, src_nodes]
        k_rel = k_relations[relation_idx]
        expected_shape = (LATENT_DIMS[src], LATENT_DIMS[dst])
        if tuple(k_rel.shape) != expected_shape:
            msg = (
                f"oracle K_r[{relation_idx}] shape {tuple(k_rel.shape)} "
                f"!= d_src×d_dst {expected_shape}"
            )
            raise AssertionError(msg)
        factor = k_rel.transpose(-2, -1).contiguous() if transpose_relation else k_rel
        coupling = torch.kron(adj_block, factor)
        effective[slices[dst], slices[src]] = (
            effective[slices[dst], slices[src]] + coupling
        )
    return effective


def test_rectangular_construction_shapes() -> None:
    """Self factors are d_τ×d_τ; relation banks are d_src×d_dst."""
    op = _rectangular_operator()
    assert op.is_rectangular
    assert op.latent_dims == LATENT_DIMS
    assert op.k_self_for("a").shape == (2, 2)
    assert op.k_self_for("b").shape == (3, 3)
    assert op.relation_matrix(0).shape == (2, 3)
    assert op.relation_matrix(1).shape == (3, 2)


def test_shared_d_path_unchanged_without_latent_dims() -> None:
    """Absent latent_dims keeps the shared-d square path."""
    op = HeteroGraphKoopmanOperator(
        SHARED_D,
        num_relations=len(EDGE_TYPES),
        node_types=NODE_TYPES,
        edge_types=EDGE_TYPES,
    )
    assert not op.is_rectangular
    assert op.latent_dims is None
    assert op.k_self_for("a").shape == (SHARED_D, SHARED_D)
    assert op.relation_matrix(0).shape == (SHARED_D, SHARED_D)


def test_equal_latent_dims_uses_square_path() -> None:
    """All d_τ == latent_dim stores dims but stays on the square factor path."""
    dims = {"a": SHARED_D, "b": SHARED_D}
    op = HeteroGraphKoopmanOperator(
        SHARED_D,
        num_relations=len(EDGE_TYPES),
        node_types=NODE_TYPES,
        edge_types=EDGE_TYPES,
        latent_dims=dims,
    )
    assert not op.is_rectangular
    assert op.latent_dims == dims
    assert op.relation_matrix(0).shape == (SHARED_D, SHARED_D)


def test_multiplex_latent_dims_must_match_shared_width() -> None:
    """Multiplex rejects a sole width that disagrees with latent_dim."""
    with pytest.raises(ValueError, match="multiplex latent_dims must equal"):
        HeteroGraphKoopmanOperator(
            4,
            num_relations=1,
            latent_dims={"node": 3},
        )


def test_rectangular_rejects_basis_and_nondense() -> None:
    """Rectangular mode requires independent dense factors."""
    with pytest.raises(ValueError, match="relation_tying='independent'"):
        HeteroGraphKoopmanOperator(
            SHARED_D,
            num_relations=len(EDGE_TYPES),
            node_types=NODE_TYPES,
            edge_types=EDGE_TYPES,
            latent_dims=LATENT_DIMS,
            relation_tying="basis",
            basis_size=1,
        )
    with pytest.raises(ValueError, match="parameterization='dense'"):
        HeteroGraphKoopmanOperator(
            SHARED_D,
            num_relations=len(EDGE_TYPES),
            node_types=NODE_TYPES,
            edge_types=EDGE_TYPES,
            latent_dims=LATENT_DIMS,
            parameterization="schur",
        )


def test_forward_matches_effective_matrix_oracle() -> None:
    """Sparse forward agrees with operator-assembled K_eff @ flat."""
    op = _rectangular_operator()
    k_self, k_rels = _oracle_factors()
    op.set_dense_matrices(k_self, k_rels)
    banks = _global_edges()
    num_nodes = sum(NUM_NODES.values())

    z_flat = op.pack_typed_latents(_oracle_typed_latents(), NUM_NODES)
    assert z_flat.numel() == stacked_latent_numel(
        NODE_TYPES,
        NUM_NODES,
        LATENT_DIMS,
    )

    z_next = op.forward(z_flat, banks, num_nodes_dict=NUM_NODES)
    k_eff = op.effective_matrix(banks, num_nodes, num_nodes_dict=NUM_NODES)
    expected = k_eff @ z_flat
    assert torch.allclose(z_next, expected, atol=_ATOL)

    # Orientation lock: relation_matrix(0) is d_src×d_dst = 2×3.
    assert op.relation_matrix(0).shape == (LATENT_DIMS["a"], LATENT_DIMS["b"])


def test_hand_built_rectangular_keff_matches_advance() -> None:
    """Independent Kronecker assembly: advance matches K_eff @ flat (TASK-1819)."""
    op = _rectangular_operator()
    k_self, k_rels = _oracle_factors()
    op.set_dense_matrices(k_self, k_rels)
    banks = _global_edges()
    num_nodes = sum(NUM_NODES.values())

    # Fixture documents Q2=A: K_ab is 2×3 (d_a × d_b), K_ba is 3×2.
    assert k_rels[0].shape == (LATENT_DIMS["a"], LATENT_DIMS["b"])
    assert k_rels[1].shape == (LATENT_DIMS["b"], LATENT_DIMS["a"])

    oracle = _hand_built_rectangular_effective(k_self, k_rels, banks)
    z_flat = op.pack_typed_latents(_oracle_typed_latents(), NUM_NODES)
    sparse_next = op.forward(z_flat, banks, num_nodes_dict=NUM_NODES)
    dense_next = oracle @ z_flat
    assert torch.allclose(sparse_next, dense_next, atol=_ATOL)

    assembled = op.effective_matrix(banks, num_nodes, num_nodes_dict=NUM_NODES)
    assert torch.allclose(assembled, oracle, atol=_ATOL)
    total = stacked_latent_numel(NODE_TYPES, NUM_NODES, LATENT_DIMS)
    assert oracle.shape == (total, total)


def test_wrong_relation_kronecker_orientation_disagrees() -> None:
    """Forgetting ``K_r.T`` in ``Â ⊗ ·`` is shape-incompatible — loud flip guard.

    Under Q2=A storage ``K_r ∈ R^{d_src×d_dst}``, the Kronecker factor must be
    ``K_r.T ∈ R^{d_dst×d_src}``. Using ``Â ⊗ K_r`` (shared-d habit without the
    transpose) yields blocks that do not fit the flat ``Σ N_τ·d_τ`` slots, so
    a silent convention flip cannot pass the construction oracle.
    """
    _k_self, k_rels = _oracle_factors()
    banks = _global_edges()
    slices = latent_type_slices_from_dims(NODE_TYPES, NUM_NODES, LATENT_DIMS)
    offsets = node_type_offsets(NODE_TYPES, NUM_NODES)
    num_nodes = sum(NUM_NODES.values())
    dtype = k_rels[0].dtype

    for relation_idx, edge_index in enumerate(banks):
        src, _rel, dst = EDGE_TYPES[relation_idx]
        adj = dense_relation_normalized_adjacency(
            edge_index,
            num_nodes,
            dtype=dtype,
            normalization="rgcn_in_degree",
        )
        src_nodes = slice(offsets[src], offsets[src] + NUM_NODES[src])
        dst_nodes = slice(offsets[dst], offsets[dst] + NUM_NODES[dst])
        adj_block = adj[dst_nodes, src_nodes]
        k_rel = k_rels[relation_idx]
        assert k_rel.shape == (LATENT_DIMS[src], LATENT_DIMS[dst])

        dst_len = slices[dst].stop - slices[dst].start
        src_len = slices[src].stop - slices[src].start
        correct = torch.kron(adj_block, k_rel.transpose(-2, -1).contiguous())
        wrong = torch.kron(adj_block, k_rel)
        assert correct.shape == (dst_len, src_len)
        assert wrong.shape != (dst_len, src_len)

    # Hand-built assembly with the wrong Kronecker orientation must raise.
    with pytest.raises(RuntimeError, match="size of tensor"):
        _hand_built_rectangular_effective(
            _k_self,
            k_rels,
            banks,
            transpose_relation=False,
        )


def test_rectangular_inverse_round_trip() -> None:
    """Dense inverse recovers the pre-image of a rectangular advance."""
    op = _rectangular_operator()
    k_self, k_rels = _oracle_factors()
    op.set_dense_matrices(k_self, k_rels)
    banks = _global_edges()
    z_flat = op.pack_typed_latents(
        {
            "a": torch.randn(NUM_NODES["a"], LATENT_DIMS["a"]),
            "b": torch.randn(NUM_NODES["b"], LATENT_DIMS["b"]),
        },
        NUM_NODES,
    )
    z_next = op.forward(z_flat, banks, num_nodes_dict=NUM_NODES)
    recovered = op.inverse_advance(
        z_next,
        edge_indices=banks,
        num_nodes_dict=NUM_NODES,
    )
    assert torch.allclose(recovered, z_flat, atol=1e-5)


def test_rectangular_gradients_flow() -> None:
    """Gradients reach rectangular relation Parameters and self factors."""
    op = _rectangular_operator()
    banks = _global_edges()
    z_flat = op.pack_typed_latents(
        {
            "a": torch.randn(NUM_NODES["a"], LATENT_DIMS["a"]),
            "b": torch.randn(NUM_NODES["b"], LATENT_DIMS["b"]),
        },
        NUM_NODES,
    )
    z_next = op.forward(z_flat, banks, num_nodes_dict=NUM_NODES)
    z_next.sum().backward()
    assert op.k_self_for("a").grad is not None
    assert op.relation_matrix(0).grad is not None
    assert torch.any(op.relation_matrix(0).grad != 0)


FEATURE_DIMS = {"a": 2, "b": 2}


def _relgraph_peers(
    *,
    latent_dims: dict[str, int] | None = None,
) -> tuple[RelGraphEncoder, RelGraphDecoder]:
    """Return typed RelGraph peers (shared-d or rectangular)."""
    encoder = RelGraphEncoder(
        FEATURE_DIMS,
        hidden_channels=4,
        latent_dim=SHARED_D,
        num_relations=len(EDGE_TYPES),
        num_layers=1,
        node_types=NODE_TYPES,
        edge_types=EDGE_TYPES,
        latent_dims=latent_dims,
    )
    decoder = RelGraphDecoder(
        latent_dim=SHARED_D,
        hidden_channels=4,
        out_channels=FEATURE_DIMS,
        num_relations=len(EDGE_TYPES),
        num_layers=1,
        node_types=NODE_TYPES,
        edge_types=EDGE_TYPES,
        latent_dims=latent_dims,
    )
    return encoder, decoder


def _typed_snapshot() -> HeteroData:
    """Tiny typed snapshot for encode / decode smoke tests."""
    snapshot = HeteroData()
    snapshot["a"].x = torch.randn(NUM_NODES["a"], FEATURE_DIMS["a"])
    snapshot["b"].x = torch.randn(NUM_NODES["b"], FEATURE_DIMS["b"])
    # Local edges; encoder lifts them to stacked global numbering.
    snapshot["a", "to_b", "b"].edge_index = torch.tensor(
        [[0, 1], [0, 2]],
        dtype=torch.long,
    )
    snapshot["b", "to_a", "a"].edge_index = torch.tensor(
        [[0, 1], [0, 1]],
        dtype=torch.long,
    )
    return snapshot


def test_factory_wires_koopman_latent_dims_into_peers_and_operator() -> None:
    """Factory rebuilds RelGraph peers and builds a matching rectangular operator."""
    encoder, decoder = _relgraph_peers()
    components = resolve_model_components(
        encoder,
        decoder,
        latent_dim=SHARED_D,
        time_step=1.0,
        koopman="hetero_graph",
        koopman_node_types=NODE_TYPES,
        koopman_edge_types=EDGE_TYPES,
        koopman_latent_dims=LATENT_DIMS,
        physics_position="prepend",
    )
    assert isinstance(components.encoder, RelGraphEncoder)
    assert isinstance(components.decoder, RelGraphDecoder)
    assert isinstance(components.koopman, HeteroGraphKoopmanOperator)
    assert components.encoder.latent_dims == LATENT_DIMS
    assert components.decoder.latent_dims == LATENT_DIMS
    assert components.koopman.latent_dims == LATENT_DIMS
    assert components.encoder.is_rectangular
    assert components.decoder.is_rectangular
    assert components.koopman.is_rectangular


def test_injected_rectangular_operator_aligns_relgraph_peers() -> None:
    """Injected rectangular operator rebuilds shared-d RelGraph peers to match."""
    encoder, decoder = _relgraph_peers()
    op = _rectangular_operator()
    components = resolve_model_components(
        encoder,
        decoder,
        latent_dim=SHARED_D,
        time_step=1.0,
        koopman=op,
        koopman_node_types=NODE_TYPES,
        koopman_edge_types=EDGE_TYPES,
        physics_position="prepend",
    )
    assert components.koopman is op
    assert components.encoder.latent_dims == LATENT_DIMS
    assert components.decoder.latent_dims == LATENT_DIMS
    assert components.encoder.is_rectangular


def test_relgraph_encode_decode_shapes_match_per_type_widths() -> None:
    """Rectangular RelGraph encode/decode shapes match d_τ / F_τ."""
    encoder, decoder = _relgraph_peers(latent_dims=LATENT_DIMS)
    snapshot = _typed_snapshot()
    z = encoder(snapshot)
    assert z.ndim == 1
    assert z.numel() == stacked_latent_numel(NODE_TYPES, NUM_NODES, LATENT_DIMS)
    banks = _global_edges()
    reconstructed = decoder(z, banks, num_nodes_dict=NUM_NODES)
    assert set(reconstructed) == set(NODE_TYPES)
    for name in NODE_TYPES:
        assert reconstructed[name].shape == (NUM_NODES[name], FEATURE_DIMS[name])


def test_rectangular_model_forward_and_gradients() -> None:
    """Factory model encode→advance→decode runs with gradients on unequal d_τ."""
    encoder, decoder = _relgraph_peers()
    model = GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=SHARED_D,
        time_step=1.0,
        koopman="hetero_graph",
        koopman_node_types=NODE_TYPES,
        koopman_edge_types=EDGE_TYPES,
        koopman_latent_dims=LATENT_DIMS,
    )
    snapshot = _typed_snapshot()
    reconstructed = model(snapshot)
    assert set(reconstructed) == set(NODE_TYPES)
    loss = sum(tensor.sum() for tensor in reconstructed.values())
    loss.backward()
    assert model.encoder.type_latent["a"].weight.grad is not None
    assert model.decoder.type_latent_in["b"].weight.grad is not None
    assert torch.any(model.koopman.relation_matrix(0).grad != 0)


def test_shared_d_relgraph_path_unchanged() -> None:
    """Absent latent_dims keeps stacked (N, d) RelGraph encode output."""
    encoder, _decoder = _relgraph_peers()
    snapshot = _typed_snapshot()
    z = encoder(snapshot)
    assert z.shape == (sum(NUM_NODES.values()), SHARED_D)


def test_mismatched_injected_operator_and_koopman_latent_dims_rejected() -> None:
    """Factory rejects disagreeing injected operator dims vs koopman_latent_dims."""
    encoder, decoder = _relgraph_peers(latent_dims=LATENT_DIMS)
    op = _rectangular_operator()
    with pytest.raises(ValueError, match="must match injected"):
        resolve_model_components(
            encoder,
            decoder,
            latent_dim=SHARED_D,
            time_step=1.0,
            koopman=op,
            koopman_node_types=NODE_TYPES,
            koopman_edge_types=EDGE_TYPES,
            koopman_latent_dims={"a": 3, "b": 2},
            physics_position="prepend",
        )


def test_rectangular_spectrum_matches_effective_matrix() -> None:
    """Operator spectrum / ρ match eigvals of rectangular K_eff."""
    op = _rectangular_operator()
    k_self, k_rels = _oracle_factors()
    op.set_dense_matrices(k_self, k_rels)
    banks = _global_edges()
    num_nodes = sum(NUM_NODES.values())
    total = stacked_latent_numel(NODE_TYPES, NUM_NODES, LATENT_DIMS)

    k_eff = op.effective_matrix(banks, num_nodes, num_nodes_dict=NUM_NODES)
    spectrum = op.spectrum(banks, num_nodes, num_nodes_dict=NUM_NODES, time_step=0.1)
    radius = op.spectral_radius(banks, num_nodes, num_nodes_dict=NUM_NODES)

    assert k_eff.shape == (total, total)
    assert spectrum.eigenvalues.shape == (total,)
    expected = torch.linalg.eigvals(k_eff.detach()).abs().max().real
    assert float(radius.detach()) == pytest.approx(float(expected), abs=1e-5)
    assert torch.allclose(
        spectrum.magnitudes.sort().values,
        torch.linalg.eigvals(k_eff.detach()).abs().real.sort().values,
        atol=1e-5,
    )


def test_rectangular_eigenvalue_regularization_assembles() -> None:
    """Eig-reg hinge uses rectangular K_eff when num_nodes_dict is supplied."""
    op = _rectangular_operator()
    k_self, k_rels = _oracle_factors()
    # Inflate spectral radius so the hinge is positive.
    k_self = {name: 1.5 * matrix for name, matrix in k_self.items()}
    op.set_dense_matrices(k_self, k_rels)
    banks = _global_edges()
    loss_fn = EigenvalueRegularizationLoss()
    penalty = loss_fn(
        op,
        dynamics_mode="discrete",
        edge_indices=banks,
        num_nodes=sum(NUM_NODES.values()),
        num_nodes_dict=NUM_NODES,
    )
    assert penalty.ndim == 0
    assert float(penalty) > 0.0


def test_rectangular_mode_energy_type_fractions_sum_to_one() -> None:
    """Mode-energy type fractions tile Σ N_τ·d_τ without assuming shared d."""
    op = _rectangular_operator()
    k_self, k_rels = _oracle_factors()
    op.set_dense_matrices(k_self, k_rels)
    banks = _global_edges()
    num_nodes = sum(NUM_NODES.values())
    with torch.no_grad():
        k_eff = op.effective_matrix(banks, num_nodes, num_nodes_dict=NUM_NODES)
        spectrum = op.spectrum(banks, num_nodes, num_nodes_dict=NUM_NODES)
        offsets = node_type_offsets(NODE_TYPES, NUM_NODES)
        couplings: dict[str, Tensor] = {}
        for relation_idx, edge_type in enumerate(EDGE_TYPES):
            src, rel, dst = edge_type
            adj = dense_relation_normalized_adjacency(
                banks[relation_idx],
                num_nodes,
                dtype=k_eff.dtype,
                normalization="rgcn_in_degree",
            )
            src_nodes = slice(offsets[src], offsets[src] + NUM_NODES[src])
            dst_nodes = slice(offsets[dst], offsets[dst] + NUM_NODES[dst])
            adj_block = adj[dst_nodes, src_nodes]
            k_rel = op.relation_matrix(relation_idx)
            # Full-size coupling with Â⊗K_r.T placed on (dst, src) blocks.
            coupling = torch.zeros_like(k_eff)
            slices = latent_type_slices_from_dims(NODE_TYPES, NUM_NODES, LATENT_DIMS)
            coupling[slices[dst], slices[src]] = torch.kron(
                adj_block,
                k_rel.transpose(-2, -1).contiguous(),
            )
            couplings[rel] = coupling
        report = attribute_mode_energy(
            k_eff,
            spectrum.eigenvectors,
            latent_dim=SHARED_D,
            latent_dims=LATENT_DIMS,
            num_nodes_dict=NUM_NODES,
            relation_blocks=couplings,
            mode_indices=(0,),
        )
    type_sum = float(report.type_fractions["a"][0] + report.type_fractions["b"][0])
    assert type_sum == pytest.approx(1.0, abs=1e-5)
    assert float(report.relation_fractions["to_b"][0]) >= 0.0
    assert float(report.relation_fractions["to_a"][0]) >= 0.0


def test_rectangular_constructor_and_factor_validation_errors() -> None:
    """Rectangular construction rejects incompatible modes and factor shapes."""
    with pytest.raises(ValueError, match="sparsity='dense'"):
        HeteroGraphKoopmanOperator(
            SHARED_D,
            num_relations=len(EDGE_TYPES),
            node_types=NODE_TYPES,
            edge_types=EDGE_TYPES,
            latent_dims=LATENT_DIMS,
            parameterization="dense",
            sparsity="block_diagonal",
        )
    with pytest.raises(ValueError, match="multiplex latent_dims"):
        HeteroGraphKoopmanOperator(
            2,
            num_relations=1,
            latent_dims={"node": 3},
        )

    op = _rectangular_operator()
    with pytest.raises(ValueError, match="typed_k_self_blocks is undefined"):
        op.typed_k_self_blocks(NUM_NODES)
    with pytest.raises(KeyError, match="unknown node type"):
        op.d_for("missing")
    with pytest.raises(ValueError, match="_relation_modules is undefined"):
        op._relation_modules()
    k_self, k_relations = _oracle_factors()
    k_relations[0] = torch.zeros(3, 2)
    with pytest.raises(ValueError, match=r"d_src×d_dst"):
        op.set_dense_matrices(k_self, k_relations)


def test_rectangular_latent_pack_unpack_validation_errors() -> None:
    """Rectangular latent packing validates keys, block shapes, and flat length."""
    op = _rectangular_operator()
    latents = _oracle_typed_latents()
    with pytest.raises(ValueError, match="keys must match"):
        op.pack_typed_latents({"a": latents["a"]}, NUM_NODES)
    with pytest.raises(ValueError, match="must have shape"):
        op.pack_typed_latents(
            {"a": torch.zeros(1, 2), "b": latents["b"]},
            NUM_NODES,
        )
    with pytest.raises(ValueError, match="z_flat must have shape"):
        op.unpack_typed_latents(torch.zeros(2), NUM_NODES)

    shared = HeteroGraphKoopmanOperator(
        SHARED_D,
        num_relations=len(EDGE_TYPES),
        node_types=NODE_TYPES,
        edge_types=EDGE_TYPES,
    )
    flat = torch.arange(sum(NUM_NODES.values()) * SHARED_D, dtype=torch.float32)
    unpacked = shared.unpack_typed_latents(flat, NUM_NODES)
    assert unpacked["a"].shape == (NUM_NODES["a"], SHARED_D)
    assert shared.d_for("a") == SHARED_D


def test_rectangular_matrix_and_forward_validation_errors() -> None:
    """Rectangular assembly and forward paths reject incompatible arguments."""
    op = _rectangular_operator()
    banks = _global_edges()
    num_nodes = sum(NUM_NODES.values())
    flat = op.pack_typed_latents(_oracle_typed_latents(), NUM_NODES)

    with pytest.raises(ValueError, match="shared-d only"):
        op._relation_coupling_matrix(
            banks,
            num_nodes,
            [None, None],
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
    with pytest.raises(ValueError, match="overrides are unsupported"):
        op.effective_matrix(
            banks,
            num_nodes,
            k_self=torch.eye(SHARED_D),
            num_nodes_dict=NUM_NODES,
        )
    with pytest.raises(ValueError, match="requires num_nodes_dict"):
        op.effective_matrix(banks, num_nodes)
    with pytest.raises(ValueError, match="control is unsupported"):
        op.forward(flat, banks, control=torch.zeros(1), num_nodes_dict=NUM_NODES)
    with pytest.raises(ValueError, match="requires num_nodes_dict"):
        op.forward(flat, banks)
    with pytest.raises(ValueError, match="expects flat z"):
        op.forward(torch.zeros(num_nodes, SHARED_D), banks, num_nodes_dict=NUM_NODES)


def test_rectangular_inverse_validation_errors() -> None:
    """Rectangular inverse requires flat uncontrolled latents and type counts."""
    op = _rectangular_operator()
    banks = _global_edges()
    flat = op.pack_typed_latents(_oracle_typed_latents(), NUM_NODES)
    with pytest.raises(ValueError, match="control is unsupported"):
        op.inverse_advance(
            flat,
            edge_indices=banks,
            control=torch.zeros(1),
            num_nodes_dict=NUM_NODES,
        )
    with pytest.raises(ValueError, match="requires num_nodes_dict"):
        op.inverse_advance(flat, edge_indices=banks)
    with pytest.raises(ValueError, match="expects flat z"):
        op.inverse_advance(
            torch.zeros(sum(NUM_NODES.values()), SHARED_D),
            edge_indices=banks,
            num_nodes_dict=NUM_NODES,
        )


def test_rectangular_bound_metric_includes_relation_norms() -> None:
    """Rectangular factor monitoring includes self and rectangular relation norms."""
    op = _rectangular_operator()
    k_self, k_relations = _oracle_factors()
    k_relations[0] = 3.0 * torch.ones_like(k_relations[0])
    op.set_dense_matrices(k_self, k_relations)
    assert float(op.bound_metric().detach()) >= float(
        torch.linalg.matrix_norm(k_relations[0], ord=2)
    )
