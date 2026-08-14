"""Coverage and error-path tests for :mod:`koopman_graph.model`."""

from __future__ import annotations

import pytest
import torch

from koopman_graph import (
    GNNDecoder,
    GNNEncoder,
    RelGraphDecoder,
    RelGraphEncoder,
)
from koopman_graph.model.factory import (
    build_koopman,
    resolve_model_components,
    validate_typed_relgraph_peers,
)
from koopman_graph.nn.heterogeneous import (
    RelGraphDecoder as RelGraphDecoderCls,
)
from koopman_graph.operators.heterogeneous import (
    HeteroGraphKoopmanOperator,
    _normalize_edge_types,
)
from koopman_graph.serialization import _require_hetero_schema


def test_factory_typed_peer_validation() -> None:
    """Typed RelGraph peers must agree with the hetero operator schema."""
    operator = HeteroGraphKoopmanOperator(
        latent_dim=2,
        num_relations=1,
        node_types=("a", "b"),
        edge_types=(("a", "r0", "b"),),
    )
    enc_multi = RelGraphEncoder(3, 4, 2, num_relations=1, num_layers=1)
    dec_typed = RelGraphDecoderCls(
        2,
        4,
        {"a": 3, "b": 2},
        num_relations=1,
        node_types=("a", "b"),
        edge_types=(("a", "r0", "b"),),
        num_layers=1,
    )
    with pytest.raises(ValueError, match="must agree on typed channels"):
        validate_typed_relgraph_peers(enc_multi, dec_typed, operator)

    enc_typed = RelGraphEncoder(
        {"a": 3, "b": 2},
        4,
        2,
        num_relations=1,
        node_types=("a", "b"),
        edge_types=(("a", "r0", "b"),),
        num_layers=1,
    )
    with pytest.raises(ValueError, match="declares node types"):
        validate_typed_relgraph_peers(
            enc_multi,
            RelGraphDecoder(2, 4, 3, num_relations=1, num_layers=1),
            operator,
        )
    wrong_enc = RelGraphEncoder(
        {"b": 2, "a": 3},
        4,
        2,
        num_relations=1,
        node_types=("b", "a"),
        edge_types=(("a", "r0", "b"),),
        num_layers=1,
    )
    with pytest.raises(ValueError, match="RelGraphEncoder node types"):
        validate_typed_relgraph_peers(wrong_enc, dec_typed, operator)
    wrong_dec = RelGraphDecoderCls(
        2,
        4,
        {"a": 3, "b": 2},
        num_relations=1,
        node_types=("b", "a"),
        edge_types=(("a", "r0", "b"),),
        num_layers=1,
    )
    with pytest.raises(ValueError, match="RelGraphDecoder node types"):
        validate_typed_relgraph_peers(enc_typed, wrong_dec, operator)
    wrong_edge_dec = RelGraphDecoderCls(
        2,
        4,
        {"a": 3, "b": 2},
        num_relations=1,
        node_types=("a", "b"),
        edge_types=(("b", "r0", "a"),),
        num_layers=1,
    )
    with pytest.raises(ValueError, match="edge types"):
        validate_typed_relgraph_peers(enc_typed, wrong_edge_dec, operator)


def test_require_hetero_schema_validation() -> None:
    """Hetero checkpoint schema rejects incomplete / inconsistent fields."""
    base = {
        "node_types": ["node"],
        "edge_types": [("node", "r0", "node")],
        "relation_tying": "independent",
        "basis_size": None,
        "relation_normalization": "rgcn_in_degree",
    }
    with pytest.raises(ValueError, match="Incomplete hetero checkpoint"):
        _require_hetero_schema({"node_types": ["node"]})
    with pytest.raises(ValueError, match="non-empty sequence of strings"):
        _require_hetero_schema({**base, "node_types": []})
    with pytest.raises(ValueError, match="entries must be non-empty strings"):
        _require_hetero_schema({**base, "node_types": [""]})
    with pytest.raises(ValueError, match="non-empty sequence of"):
        _require_hetero_schema({**base, "edge_types": []})
    with pytest.raises(ValueError, match=r"\(src, rel, dst\)"):
        _require_hetero_schema({**base, "edge_types": [("node", "r0")]})
    with pytest.raises(ValueError, match="triples must use non-empty"):
        _require_hetero_schema({**base, "edge_types": [("node", "", "node")]})
    with pytest.raises(ValueError, match="relation_tying must be one of"):
        _require_hetero_schema({**base, "relation_tying": "tied"})
    with pytest.raises(ValueError, match="basis_size must be null"):
        _require_hetero_schema({**base, "basis_size": 1})
    with pytest.raises(ValueError, match="positive int"):
        _require_hetero_schema({**base, "relation_tying": "basis", "basis_size": True})
    with pytest.raises(ValueError, match="positive int"):
        _require_hetero_schema({**base, "relation_tying": "basis", "basis_size": 0})
    with pytest.raises(ValueError, match="relation_normalization must be"):
        _require_hetero_schema({**base, "relation_normalization": "symmetric"})
    with pytest.raises(ValueError, match="must match"):
        _require_hetero_schema(
            {
                **base,
                "encoder": {"normalization": "random_walk"},
            }
        )


def test_factory_and_operator_codecov_gap_guards() -> None:
    """Cover factory RelGraph policy and hetero operator inverse/basis gaps."""
    from unittest.mock import patch

    enc = RelGraphEncoder(3, 4, 2, num_relations=1, num_layers=1)
    dec = RelGraphDecoder(2, 4, 3, num_relations=1, num_layers=1)
    with (
        patch(
            "koopman_graph.model.factory.resolve_delay_encoder",
            side_effect=lambda encoder, n_delays: (encoder, n_delays),
        ),
        pytest.raises(ValueError, match="n_delays > 1"),
    ):
        resolve_model_components(
            enc,
            dec,
            latent_dim=2,
            time_step=1.0,
            koopman="hetero_graph",
            n_delays=2,
            physics_position="concat",
        )
    physics_dec = RelGraphDecoder(3, 4, 3, num_relations=1, num_layers=1)
    with pytest.raises(ValueError, match="physics-informed"):
        resolve_model_components(
            enc,
            physics_dec,
            latent_dim=3,
            time_step=1.0,
            koopman="hetero_graph",
            physics_dim=1,
            physics_lifting_fn=lambda features: features[:, :1],
            physics_position="concat",
        )
    with pytest.raises(ValueError, match="requires RelGraphEncoder"):
        resolve_model_components(
            GNNEncoder(3, 4, 2),
            GNNDecoder(2, 4, 3),
            latent_dim=2,
            time_step=1.0,
            koopman=HeteroGraphKoopmanOperator(2, 1),
            physics_position="concat",
        )
    bad_op = HeteroGraphKoopmanOperator(2, 2)
    with pytest.raises(ValueError, match="num_relations"):
        resolve_model_components(
            enc,
            dec,
            latent_dim=2,
            time_step=1.0,
            koopman=bad_op,
            physics_position="concat",
        )
    mismatched_norm = HeteroGraphKoopmanOperator(2, 1, normalization="random_walk")
    with pytest.raises(ValueError, match="normalization"):
        resolve_model_components(
            enc,
            dec,
            latent_dim=2,
            time_step=1.0,
            koopman=mismatched_norm,
            physics_position="concat",
        )
    from koopman_graph.model.factory import resolve_injected_koopman

    with pytest.raises(ValueError, match="dynamics_mode='discrete'"):
        resolve_injected_koopman(
            HeteroGraphKoopmanOperator(2, 1),
            latent_dim=2,
            control_dim=0,
            control_mode="additive",
            bilinear_rank=None,
            dynamics_mode="continuous",
            koopman_init_mode="identity_noise",
            koopman_init_scale=0.01,
            koopman_parameterization="dense",
            koopman_max_spectral_radius=1.0,
            koopman_auxiliary_hidden_dims=None,
        )
    with pytest.raises(ValueError, match="koopman_auxiliary_hidden_dims"):
        build_koopman(
            koopman="hetero_graph",
            latent_dim=2,
            control_dim=0,
            control_mode="additive",
            bilinear_rank=None,
            dynamics_mode="discrete",
            koopman_init_mode="identity",
            koopman_init_scale=0.01,
            koopman_parameterization="dense",
            koopman_max_spectral_radius=1.0,
            koopman_auxiliary_hidden_dims=(4,),
            num_relations=1,
        )
    with pytest.raises(ValueError, match="num_relations can be resolved"):
        build_koopman(
            koopman="hetero_graph",
            latent_dim=2,
            control_dim=0,
            control_mode="additive",
            bilinear_rank=None,
            dynamics_mode="discrete",
            koopman_init_mode="identity",
            koopman_init_scale=0.01,
            koopman_parameterization="dense",
            koopman_max_spectral_radius=1.0,
            koopman_auxiliary_hidden_dims=None,
            num_relations=None,
        )

    # Sole non-default multiplex node type defaults edge types.
    edges = _normalize_edge_types(None, node_types=("bus",), num_relations=2)
    assert edges == (("bus", "r0", "bus"), ("bus", "r1", "bus"))

    op = HeteroGraphKoopmanOperator(2, 1)
    with pytest.raises(ValueError, match="_basis_modules"):
        op._basis_modules()
    with pytest.raises(IndexError, match="relation_index"):
        op._assembled_relation_matrix(3)
    with pytest.raises(ValueError, match="set_basis_factors"):
        op.set_basis_factors([torch.eye(2)], torch.ones(1, 1))
    basis_op = HeteroGraphKoopmanOperator(2, 2, relation_tying="basis", basis_size=1)
    basis_op.reset_parameters()
    with pytest.raises(ValueError, match="Expected 1 basis matrices"):
        basis_op.set_basis_factors([torch.eye(2), torch.eye(2)], torch.ones(2, 1))
    # Typed operator without num_nodes_dict.
    typed_op = HeteroGraphKoopmanOperator(
        2,
        1,
        node_types=("a", "b"),
        edge_types=(("a", "r0", "b"),),
    )
    with pytest.raises(ValueError, match="requires num_nodes_dict"):
        typed_op._require_num_nodes_dict(None, num_nodes=4, caller="test")
    # Multiplex path validates supplied counts.
    assert op._require_num_nodes_dict({"node": 3}, num_nodes=3, caller="test") == {
        "node": 3
    }

    z = torch.randn(3, 2)
    edges_bank = [torch.tensor([[0, 1], [1, 2]], dtype=torch.long)]
    with pytest.raises(ValueError, match="edge_indices is required"):
        op.inverse_advance(z)
    with pytest.raises(ValueError, match="expects z with shape"):
        op.inverse_advance(torch.randn(3), edge_indices=edges_bank)
    recovered = op.inverse_advance(z, edge_indices=edges_bank)
    assert recovered.shape == z.shape

    ctrl_op = HeteroGraphKoopmanOperator(2, 1, control_dim=1, control_mode="bilinear")
    with pytest.raises(ValueError, match="control input is required"):
        ctrl_op._bilinear_self_factors(None, num_nodes=3)
    with pytest.raises(ValueError, match="Per-node control"):
        ctrl_op._bilinear_self_factors(torch.randn(2, 1), num_nodes=3)
    with pytest.raises(ValueError, match="control input must have shape"):
        ctrl_op._bilinear_self_factors(torch.randn(3, 2, 1), num_nodes=3)

    # Typed block-diagonal inverse path.
    bd = HeteroGraphKoopmanOperator(
        2,
        1,
        node_types=("a", "b"),
        edge_types=(("a", "r0", "b"),),
        sparsity="block_diagonal",
    )
    typed_z = torch.randn(5, 2)
    typed_edges = [torch.tensor([[0, 1], [2, 3]], dtype=torch.long)]
    out = bd.inverse_advance(
        typed_z,
        edge_indices=typed_edges,
        num_nodes_dict={"a": 2, "b": 3},
    )
    assert out.shape == typed_z.shape
