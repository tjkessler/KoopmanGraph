"""Tests for symmetry-adapted orbit ties (TASK-1310)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import torch

pytest.importorskip("networkx")

from koopman_graph.graph_utils import (
    identity_orbit_partition,
    node_orbit_partition,
    validate_orbit_partition,
)
from koopman_graph.model import GraphKoopmanModel
from koopman_graph.nn import GNNDecoder, GNNEncoder
from koopman_graph.operators import GraphKoopmanOperator
from koopman_graph.serialization import (
    build_model_config,
    load_checkpoint,
    reconstruct_model,
    save_checkpoint,
)


def _cycle_edge_index(num_nodes: int) -> torch.Tensor:
    edges: list[list[int]] = []
    for node in range(num_nodes):
        nxt = (node + 1) % num_nodes
        edges.extend([[node, nxt], [nxt, node]])
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _star_edge_index(num_nodes: int) -> torch.Tensor:
    edges: list[list[int]] = []
    for leaf in range(1, num_nodes):
        edges.extend([[0, leaf], [leaf, 0]])
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _trainable_params(module: torch.nn.Module) -> int:
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


def test_ring_orbits_single_orbit() -> None:
    """Cycle graphs put all vertices in one automorphism orbit."""
    edge_index = _cycle_edge_index(6)
    partition = node_orbit_partition(edge_index, 6, method="auto")
    assert partition == ((0, 1, 2, 3, 4, 5),)


def test_star_orbits_center_and_leaves() -> None:
    """Stars separate the hub from the leaf orbit."""
    edge_index = _star_edge_index(5)
    partition = node_orbit_partition(edge_index, 5, method="auto")
    assert partition == ((0,), (1, 2, 3, 4))


def test_validate_orbit_partition_rejects_malformed() -> None:
    """Malformed partitions raise ValueError."""
    with pytest.raises(ValueError, match="missing nodes"):
        validate_orbit_partition(((0, 1),), num_nodes=3)
    with pytest.raises(ValueError, match="repeats"):
        validate_orbit_partition(((0, 1), (1, 2)), num_nodes=3)


def test_identity_fallback_warns_without_backends() -> None:
    """Missing networkx/pynauty warns and returns the identity partition."""
    edge_index = _cycle_edge_index(3)
    real_import = __import__

    def _selective_import(name: str, *args: object, **kwargs: object) -> object:
        if name in {"pynauty", "networkx"}:
            raise ImportError(f"blocked {name}")
        return real_import(name, *args, **kwargs)

    with (
        patch("builtins.__import__", side_effect=_selective_import),
        pytest.warns(UserWarning, match="identity"),
    ):
        partition = node_orbit_partition(edge_index, 3, method="auto")
    assert partition == identity_orbit_partition(3)


def test_orbit_tied_ring_reduces_params_vs_identity() -> None:
    """Auto ring orbits use one K_self; identity uses N factors."""
    num_nodes = 6
    latent_dim = 3
    edge_index = _cycle_edge_index(num_nodes)
    tied = GraphKoopmanOperator(
        latent_dim,
        init_mode="identity",
        auto_orbits=True,
    )
    identity = GraphKoopmanOperator(
        latent_dim,
        init_mode="identity",
        orbit_partition=identity_orbit_partition(num_nodes),
    )
    z = torch.randn(num_nodes, latent_dim)
    _ = tied.advance(z, edge_index=edge_index)
    assert tied.orbit_partition == ((0, 1, 2, 3, 4, 5),)
    assert len(identity.orbit_partition or ()) == num_nodes
    assert _trainable_params(tied) < _trainable_params(identity)


def test_fit_trains_auto_orbit_k_self() -> None:
    """``fit`` binds auto orbits before the optimizer so ``K_self`` is trained."""
    from torch_geometric.data import Data

    from koopman_graph.data import GraphSnapshotSequence

    torch.manual_seed(0)
    num_nodes = 6
    latent_dim = 2
    edge_index = _cycle_edge_index(num_nodes)
    sequence = GraphSnapshotSequence(
        [Data(x=torch.randn(num_nodes, 1), edge_index=edge_index) for _ in range(4)]
    )
    model = GraphKoopmanModel(
        encoder=GNNEncoder(1, 4, latent_dim, num_layers=1),
        decoder=GNNDecoder(latent_dim, 4, 1, num_layers=1),
        latent_dim=latent_dim,
        time_step=0.1,
        koopman="graph",
        koopman_auto_orbits=True,
    )
    assert model.koopman.orbit_partition is None
    history = model.fit(sequence, epochs=3, lr=1e-2)
    assert history is not None
    assert model.koopman.orbit_partition is not None
    assert model.koopman.uses_orbit_selves
    before = model.koopman._orbit_selves[0].K.detach().clone()
    _ = model.fit(sequence, epochs=2, lr=1e-2)
    after = model.koopman._orbit_selves[0].K.detach()
    assert not torch.allclose(before, after)


def test_orbit_tied_matches_shared_spectrum_on_ring() -> None:
    """One-orbit ring ties reproduce the shared-K_self effective spectrum."""
    torch.manual_seed(0)
    num_nodes = 6
    latent_dim = 2
    edge_index = _cycle_edge_index(num_nodes)
    tied = GraphKoopmanOperator(latent_dim, init_mode="identity", auto_orbits=True)
    shared = GraphKoopmanOperator(latent_dim, init_mode="identity")
    k_self = torch.tensor([[0.6, 0.1], [0.0, 0.5]])
    k_nbr = torch.tensor([[0.05, 0.0], [0.0, 0.04]])
    _ = tied.advance(torch.zeros(num_nodes, latent_dim), edge_index=edge_index)
    tied.set_dense_matrices(k_self, k_nbr)
    shared.set_dense_matrices(k_self, k_nbr)
    tied_eigs = tied.spectrum(edge_index, num_nodes).eigenvalues
    shared_eigs = shared.spectrum(edge_index, num_nodes).eigenvalues
    assert torch.allclose(tied_eigs.real, shared_eigs.real, atol=1e-5)
    assert torch.allclose(tied_eigs.imag, shared_eigs.imag, atol=1e-5)


def test_explicit_partition_overrides_auto_orbits() -> None:
    """Explicit partitions bind immediately and are not replaced by auto."""
    num_nodes = 4
    edge_index = _cycle_edge_index(num_nodes)
    explicit = ((0, 1), (2, 3))
    op = GraphKoopmanOperator(
        2,
        init_mode="identity",
        orbit_partition=explicit,
        auto_orbits=True,
    )
    assert op.orbit_partition == validate_orbit_partition(explicit, num_nodes)
    _ = op.advance(torch.randn(num_nodes, 2), edge_index=edge_index)
    assert op.orbit_partition == validate_orbit_partition(explicit, num_nodes)


def test_symmetry_config_round_trip(tmp_path: Path) -> None:
    """Format-1 checkpoints persist and restore the symmetry block."""
    encoder = GNNEncoder(in_channels=2, hidden_channels=4, latent_dim=3)
    decoder = GNNDecoder(latent_dim=3, hidden_channels=4, out_channels=2)
    model = GraphKoopmanModel(
        encoder=encoder,
        decoder=decoder,
        latent_dim=3,
        time_step=0.1,
        koopman="graph",
        koopman_auto_orbits=True,
        koopman_orbit_method="auto",
    )
    edge_index = _cycle_edge_index(4)
    _ = model.koopman.advance(
        torch.zeros(4, 3),
        edge_index=edge_index,
    )
    config = build_model_config(model)
    assert config["symmetry"] is not None
    assert config["symmetry"]["auto_orbits"] is True
    assert config["symmetry"]["orbit_partition"] == [[0, 1, 2, 3]]
    assert config["symmetry"]["method"] == "auto"

    path = tmp_path / "sym.pt"
    save_checkpoint(model, path)
    loaded = load_checkpoint(path)
    assert loaded.koopman.orbit_partition == ((0, 1, 2, 3),)
    assert loaded.koopman.auto_orbits is True
    rebuilt = reconstruct_model(config)
    assert rebuilt.koopman.orbit_partition == ((0, 1, 2, 3),)


def test_operators_import_without_requiring_networkx_at_module_level() -> None:
    """Core operator imports must not hard-depend on networkx."""
    import koopman_graph.graph_utils.symmetry as symmetry_mod
    import koopman_graph.operators as operators

    assert hasattr(operators, "GraphKoopmanOperator")
    # networkx is imported only inside orbit helpers, not at module import.
    assert "networkx" not in symmetry_mod.__dict__


def test_validate_orbit_partition_edge_cases() -> None:
    """Validation rejects empty / out-of-range / non-positive node counts."""
    with pytest.raises(ValueError, match="positive"):
        validate_orbit_partition(((0,),), num_nodes=0)
    with pytest.raises(ValueError, match="at least one orbit"):
        validate_orbit_partition((), num_nodes=2)
    with pytest.raises(ValueError, match="non-empty"):
        validate_orbit_partition(((), (0, 1)), num_nodes=2)
    with pytest.raises(ValueError, match="outside"):
        validate_orbit_partition(((0,), (3,)), num_nodes=2)


def test_identity_orbit_partition_rejects_non_positive() -> None:
    """Identity partition requires a positive node count."""
    with pytest.raises(ValueError, match="positive"):
        identity_orbit_partition(0)


def test_hyperedge_two_section_builds_clique_edges() -> None:
    """2-section connects co-members of each hyperedge both ways."""
    from koopman_graph.graph_utils import hyperedge_two_section

    hyperedge_index = torch.tensor(
        [[0, 1, 2, 1, 3], [0, 0, 0, 1, 1]],
        dtype=torch.long,
    )
    edges = hyperedge_two_section(hyperedge_index, num_nodes=4)
    pairs = {tuple(sorted(pair)) for pair in edges.t().tolist()}
    assert (0, 1) in pairs
    assert (0, 2) in pairs
    assert (1, 2) in pairs
    assert (1, 3) in pairs
    with pytest.raises(ValueError, match="\\(2, nnz\\)"):
        hyperedge_two_section(torch.zeros(3, 2), num_nodes=2)
    with pytest.raises(ValueError, match="positive"):
        hyperedge_two_section(hyperedge_index, num_nodes=0)
    empty = hyperedge_two_section(
        torch.tensor([[0], [0]], dtype=torch.long),
        num_nodes=2,
    )
    assert empty.shape == (2, 0)


def test_apply_and_assemble_orbit_self_helpers() -> None:
    """Orbit self maps and block assembly agree on tied nodes."""
    from koopman_graph.graph_utils import apply_orbit_self, assemble_orbit_self_blocks

    k0 = torch.diag(torch.tensor([0.5, 0.2]))
    k1 = torch.diag(torch.tensor([0.9, 0.1]))
    node_orbit = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    z = torch.randn(4, 2)
    z_next = apply_orbit_self(z, (k0, k1), node_orbit)
    assert torch.allclose(z_next[:2], z[:2] @ k0.T)
    assert torch.allclose(z_next[2:], z[2:] @ k1.T)
    blocks = assemble_orbit_self_blocks((k0, k1), node_orbit, num_nodes=4)
    assert blocks.shape == (4, 2, 2)
    assert torch.allclose(blocks[0], k0)
    assert torch.allclose(blocks[3], k1)
    with pytest.raises(ValueError, match="\\(num_nodes, latent_dim\\)"):
        apply_orbit_self(torch.randn(2, 2, 2), (k0,), torch.zeros(2, dtype=torch.long))
    with pytest.raises(ValueError, match="node_orbit must have shape"):
        apply_orbit_self(z, (k0, k1), torch.zeros(3, dtype=torch.long))
    with pytest.raises(ValueError, match="non-empty"):
        assemble_orbit_self_blocks((), node_orbit, num_nodes=4)
    # Unused orbit matrices are skipped (mask.any() is False).
    k2 = torch.eye(2)
    skipped = apply_orbit_self(z, (k0, k1, k2), node_orbit)
    assert torch.allclose(skipped, z_next)


def test_node_orbit_partition_input_validation() -> None:
    """Orbit helpers reject bad method / shapes / node counts."""
    edge_index = _cycle_edge_index(3)
    with pytest.raises(ValueError, match="method must be"):
        node_orbit_partition(edge_index, 3, method="wl")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive"):
        node_orbit_partition(edge_index, 0)
    with pytest.raises(ValueError, match="\\(2, E\\)"):
        node_orbit_partition(torch.zeros(3, 2), 3)


def test_node_orbit_partition_exact_requires_pynauty() -> None:
    """method='exact' raises ImportError when pynauty is unavailable."""
    edge_index = _cycle_edge_index(3)
    real_import = __import__

    def _block_pynauty(name: str, *args: object, **kwargs: object) -> object:
        if name == "pynauty":
            raise ImportError("blocked pynauty")
        return real_import(name, *args, **kwargs)

    with (
        patch("builtins.__import__", side_effect=_block_pynauty),
        pytest.raises(ImportError, match="pynauty"),
    ):
        node_orbit_partition(edge_index, 3, method="exact")


def test_node_orbit_partition_ignores_self_loops_and_empty_edges() -> None:
    """Empty graphs and self-loops still yield a valid partition."""
    empty = torch.empty((2, 0), dtype=torch.long)
    partition = node_orbit_partition(empty, 3, method="auto")
    assert validate_orbit_partition(partition, 3) == partition
    loops = torch.tensor([[0, 0], [0, 0]], dtype=torch.long)
    loop_partition = node_orbit_partition(loops, 2, method="auto")
    assert validate_orbit_partition(loop_partition, 2) == loop_partition
