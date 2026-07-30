"""Tests for interpretive mode-energy attribution on assembled ``K_eff``."""

from __future__ import annotations

import pytest
import torch

from koopman_graph.analysis import ModeEnergyAttribution, attribute_mode_energy
from koopman_graph.analysis.spectrum import attribute_mode_energy as attribute_direct
from koopman_graph.data import latent_type_slices, node_type_slices
from koopman_graph.graph_utils import dense_relation_normalized_adjacency
from koopman_graph.operators import HeteroGraphKoopmanOperator


def test_attribute_mode_energy_exported_from_analysis() -> None:
    """Capability-module export without root ``__all__`` promotion."""
    import koopman_graph
    from koopman_graph import analysis as kg_analysis

    assert "attribute_mode_energy" in kg_analysis.__all__
    assert "ModeEnergyAttribution" in kg_analysis.__all__
    assert "attribute_mode_energy" not in koopman_graph.__all__
    assert attribute_mode_energy is attribute_direct


def test_latent_type_slices_expand_node_rows() -> None:
    """Node-row slices expand by ``d`` into flat ``vec(Z)`` indices."""
    slices = node_type_slices(("gen", "load"), {"gen": 2, "load": 3})
    flat = latent_type_slices(slices, latent_dim=2)
    assert flat["gen"] == slice(0, 4)
    assert flat["load"] == slice(4, 10)


def test_block_diagonal_type_mass_concentrates() -> None:
    """Synthetic block-diagonal K_eff: dominant mode mass on generating type."""
    # Two types, d=1: K_eff = diag(0.9, 0.9, 0.1, 0.1, 0.1).
    # Leading eigenvectors live in the gen block.
    k_eff = torch.diag(torch.tensor([0.9, 0.9, 0.1, 0.1, 0.1], dtype=torch.float64))
    eigenvalues, eigenvectors = torch.linalg.eig(k_eff)
    order = torch.argsort(eigenvalues.abs(), descending=True)
    eigenvectors = eigenvectors[:, order].real
    slices = node_type_slices(("gen", "load"), {"gen": 2, "load": 3})
    report = attribute_mode_energy(
        k_eff,
        eigenvectors,
        latent_dim=1,
        node_type_slices=slices,
        mode_indices=(0, 1),
    )
    assert isinstance(report, ModeEnergyAttribution)
    assert report.mode_indices == (0, 1)
    assert report.type_fractions["gen"][0] > 0.99
    assert report.type_fractions["load"][0] < 0.01
    assert torch.allclose(
        report.type_fractions["gen"] + report.type_fractions["load"],
        torch.ones(2, dtype=torch.float64),
        atol=1e-8,
    )


def test_relation_action_mass_concentrates_on_coupling() -> None:
    """Relation action mass peaks on the relation that generates the mode."""
    num_nodes = 3
    latent_dim = 1
    # Single bidirectional edge → Â has ±1 spectrum under R-GCN in-degree.
    edge = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    adj = dense_relation_normalized_adjacency(
        edge,
        num_nodes,
        dtype=torch.float64,
        normalization="rgcn_in_degree",
    )
    k_self = torch.tensor([[0.2]], dtype=torch.float64)
    k_rel = torch.tensor([[0.8]], dtype=torch.float64)
    identity = torch.eye(num_nodes, dtype=torch.float64)
    self_term = torch.kron(identity, k_self)
    coupling = torch.kron(adj, k_rel)
    # Zero coupling for a second unused relation.
    empty_adj = torch.zeros_like(adj)
    unused = torch.kron(empty_adj, k_rel)
    k_eff = self_term + coupling

    eigenvalues, eigenvectors = torch.linalg.eig(k_eff)
    order = torch.argsort(eigenvalues.abs(), descending=True)
    eigenvectors = eigenvectors[:, order]
    if torch.allclose(eigenvectors.imag, torch.zeros_like(eigenvectors.imag)):
        modes = eigenvectors.real
    else:
        modes = eigenvectors
    report = attribute_mode_energy(
        k_eff,
        modes,
        latent_dim=latent_dim,
        node_type_slices=node_type_slices(("node",), {"node": num_nodes}),
        relation_blocks={"active": coupling, "unused": unused},
        mode_indices=(0,),
    )
    active = report.relation_fractions["active"][0]
    unused_frac = report.relation_fractions["unused"][0]
    assert active > unused_frac
    assert unused_frac == pytest.approx(0.0, abs=1e-8)


def test_hetero_operator_oracle_attribution() -> None:
    """Hand-built multiplex K_eff: type mass full; relation action nonzero."""
    num_nodes = 4
    latent_dim = 2
    edge_indices = [
        torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        torch.tensor([[3, 2, 0], [2, 0, 1]], dtype=torch.long),
    ]
    k_self = torch.tensor([[0.7, 0.0], [0.0, 0.6]])
    k_relations = [
        torch.tensor([[0.2, 0.0], [0.0, 0.15]]),
        torch.tensor([[0.0, 0.1], [0.1, 0.0]]),
    ]
    op = HeteroGraphKoopmanOperator(latent_dim, num_relations=2, init_mode="identity")
    op.set_dense_matrices(k_self, k_relations)
    with torch.no_grad():
        k_eff = op.effective_matrix(edge_indices, num_nodes)
        spectrum = op.spectrum(edge_indices, num_nodes)
        couplings = []
        for edge_index, k_rel in zip(edge_indices, k_relations, strict=True):
            adj = dense_relation_normalized_adjacency(
                edge_index,
                num_nodes,
                dtype=k_eff.dtype,
                normalization="rgcn_in_degree",
            )
            couplings.append(torch.kron(adj, k_rel))
        report = attribute_mode_energy(
            k_eff,
            spectrum.eigenvectors,
            latent_dim=latent_dim,
            node_type_slices=node_type_slices(("node",), {"node": num_nodes}),
            relation_blocks={"r0": couplings[0], "r1": couplings[1]},
            mode_indices=(0,),
        )
    assert float(report.type_fractions["node"][0]) == pytest.approx(1.0, abs=1e-5)
    assert float(report.relation_fractions["r0"][0]) >= 0.0
    assert float(report.relation_fractions["r1"][0]) >= 0.0


def test_attribute_mode_energy_rejects_bad_shapes() -> None:
    """Shape / index validation raises clearly."""
    k_eff = torch.eye(4)
    eigenvectors = torch.eye(4)
    with pytest.raises(ValueError, match="square matrix"):
        attribute_mode_energy(
            torch.ones(3, 4),
            eigenvectors,
            latent_dim=2,
        )
    with pytest.raises(ValueError, match="mode_indices"):
        attribute_mode_energy(
            k_eff,
            eigenvectors,
            latent_dim=2,
            mode_indices=(99,),
        )
    with pytest.raises(ValueError, match="relation_blocks"):
        attribute_mode_energy(
            k_eff,
            eigenvectors,
            latent_dim=2,
            relation_blocks={"r0": torch.eye(3)},
        )


def test_honesty_docstring_mentions_interpretive_diagnostic() -> None:
    """Public docstring must not overclaim ResDMD / causal attribution."""
    doc = attribute_mode_energy.__doc__ or ""
    assert "interpretive" in doc.lower()
    assert "ResDMD" in doc or "resdmd" in doc.lower()
    assert "not" in doc.lower()
    result_doc = ModeEnergyAttribution.__doc__ or ""
    assert "interpretive" in result_doc.lower()


def test_model_spectrum_requires_hetero_topology() -> None:
    """Hetero model.spectrum refuses the silent K_self fallback."""
    from koopman_graph.model import GraphKoopmanModel
    from koopman_graph.nn import RelGraphDecoder, RelGraphEncoder

    model = GraphKoopmanModel(
        encoder=RelGraphEncoder(3, 8, 4, num_relations=2, num_layers=1),
        decoder=RelGraphDecoder(4, 8, 3, num_relations=2, num_layers=1),
        latent_dim=4,
        time_step=1.0,
        koopman="hetero_graph",
    )
    with pytest.raises(ValueError, match="edge_indices and num_nodes"):
        model.spectrum()
