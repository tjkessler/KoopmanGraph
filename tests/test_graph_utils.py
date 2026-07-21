"""Unit tests for shared graph and propagation utilities."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data

from koopman_graph.graph_utils import (
    advance_and_decode,
    autoregressive_latent_rollout,
    degree_support_mask,
    dense_symmetric_normalized_adjacency,
    dense_symmetric_normalized_laplacian,
    hold_last_topology_at,
    inverse_propagate_latent,
    propagate_latent,
    resolve_delta_t,
    resolve_edge_index,
    resolve_edge_weight,
    resolve_graph_inputs,
    snapshot_edge_weight,
    snapshot_to_device,
    snapshot_topology_at,
    symmetric_normalized_adjacency_edge_weights,
    symmetric_normalized_adjacency_matvec,
    symmetric_normalized_laplacian_matvec,
)
from koopman_graph.operators import ContinuousKoopmanOperator, KoopmanOperator


def _two_node_data(*, with_weights: bool = False) -> Data:
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    data = Data(x=torch.randn(2, 3), edge_index=edge_index)
    if with_weights:
        data.edge_weight = torch.tensor([0.5, 1.5], dtype=torch.float32)
    return data


def test_symmetric_normalized_adjacency_matches_dense_and_sparse() -> None:
    """Verify shared L_sym weights agree for dense assembly and sparse matvec."""
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    edge_weight = torch.tensor([1.0, 1.0, 2.0, 2.0])
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    num_nodes = 3

    weights = symmetric_normalized_adjacency_edge_weights(
        edge_index,
        num_nodes=num_nodes,
        edge_weight=edge_weight,
        dtype=torch.float32,
    )
    assert weights.shape == (4,)

    adj = dense_symmetric_normalized_adjacency(
        edge_index,
        num_nodes,
        edge_weight=edge_weight,
        dtype=torch.float32,
    )
    sparse = symmetric_normalized_adjacency_matvec(
        edge_index,
        x,
        edge_weight=edge_weight,
        num_nodes=num_nodes,
    )
    assert torch.allclose(sparse, adj @ x)

    l_sym_dense = torch.eye(num_nodes) - adj
    assert torch.allclose(x - sparse, l_sym_dense @ x)


def test_dense_adjacency_accumulates_duplicate_edges() -> None:
    """Dense assembly must sum duplicate edges like the sparse matvec."""
    edge_index = torch.tensor([[0, 0, 1], [1, 1, 0]], dtype=torch.long)
    x = torch.randn(2, 3)
    adj = dense_symmetric_normalized_adjacency(
        edge_index,
        num_nodes=2,
        dtype=torch.float32,
    )
    sparse = symmetric_normalized_adjacency_matvec(edge_index, x, num_nodes=2)
    assert torch.allclose(adj @ x, sparse)


def test_isolated_node_normalized_adjacency_has_zero_row() -> None:
    """Isolated nodes contribute zero degree and a zero row/column of Â."""
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    num_nodes = 3
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, -1.0]])
    adj = dense_symmetric_normalized_adjacency(
        edge_index,
        num_nodes,
        dtype=torch.float32,
    )
    assert torch.allclose(adj[2], torch.zeros(3))
    assert torch.allclose(adj[:, 2], torch.zeros(3))

    sparse = symmetric_normalized_adjacency_matvec(edge_index, x, num_nodes=num_nodes)
    assert torch.allclose(sparse[2], torch.zeros(2))

    support = degree_support_mask(
        edge_index,
        num_nodes=num_nodes,
        dtype=torch.float32,
    )
    assert torch.allclose(support, torch.tensor([1.0, 1.0, 0.0]))

    # L_sym = P - Â maps isolated-node features to zeros (not identity passthrough).
    laplacian_sparse = symmetric_normalized_laplacian_matvec(
        edge_index,
        x,
        num_nodes=num_nodes,
    )
    laplacian_dense = dense_symmetric_normalized_laplacian(
        edge_index,
        num_nodes,
        dtype=torch.float32,
    )
    assert torch.allclose(laplacian_sparse[2], torch.zeros(2))
    assert torch.allclose(laplacian_dense @ x, laplacian_sparse, atol=1e-6)
    assert torch.allclose(laplacian_dense[2], torch.zeros(3))
    assert torch.allclose(laplacian_dense[:, 2], torch.zeros(3))


def test_no_isolate_laplacian_reduces_to_i_minus_adj() -> None:
    """On fully positive-degree graphs, L_sym equals I - Â."""
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    num_nodes = 3
    x = torch.randn(3, 2)
    adj = dense_symmetric_normalized_adjacency(
        edge_index,
        num_nodes,
        dtype=torch.float32,
    )
    laplacian = dense_symmetric_normalized_laplacian(
        edge_index,
        num_nodes,
        dtype=torch.float32,
    )
    expected = torch.eye(num_nodes) - adj
    assert torch.allclose(laplacian, expected, atol=1e-6)
    sparse = symmetric_normalized_laplacian_matvec(edge_index, x, num_nodes=num_nodes)
    assert torch.allclose(sparse, expected @ x, atol=1e-6)


def test_snapshot_edge_weight_absent_and_present() -> None:
    """Verify optional edge weights are read from snapshots."""
    plain = _two_node_data()
    weighted = _two_node_data(with_weights=True)
    assert snapshot_edge_weight(plain) is None
    assert torch.equal(snapshot_edge_weight(weighted), weighted.edge_weight)


def test_resolve_graph_inputs_from_data_and_tensors() -> None:
    """Verify Data and tensor paths resolve consistently."""
    data = _two_node_data(with_weights=True)
    x, edge_index, edge_weight = resolve_graph_inputs(data, None)
    assert torch.equal(x, data.x)
    assert torch.equal(edge_index, data.edge_index)
    assert torch.equal(edge_weight, data.edge_weight)

    x2, edge_index2, edge_weight2 = resolve_graph_inputs(
        data.x,
        data.edge_index,
        data.edge_weight,
    )
    assert torch.equal(x2, data.x)
    assert torch.equal(edge_index2, data.edge_index)
    assert torch.equal(edge_weight2, data.edge_weight)


def test_resolve_edge_index_requires_edges_for_tensors() -> None:
    """Verify tensor input without edge_index raises."""
    with pytest.raises(ValueError, match="edge_index is required"):
        resolve_edge_index(torch.randn(3, 2), None)


def test_resolve_edge_weight_prefers_data_attribute() -> None:
    """Verify Data edge weights override the explicit argument."""
    data = _two_node_data(with_weights=True)
    ignored = torch.tensor([9.0, 9.0])
    assert torch.equal(resolve_edge_weight(data, ignored), data.edge_weight)
    assert resolve_edge_weight(data.x, ignored) is ignored


def test_snapshot_to_device_preserves_edge_weights() -> None:
    """Verify device transfer keeps topology and weights."""
    data = _two_node_data(with_weights=True)
    moved = snapshot_to_device(data, torch.device("cpu"))
    assert moved.x.device.type == "cpu"
    assert torch.equal(moved.edge_index, data.edge_index)
    assert torch.equal(moved.edge_weight, data.edge_weight)


def test_resolve_delta_t_explicit_and_default() -> None:
    """Verify resolve_delta_t prefers explicit values and soft-defaults to 1.0."""
    assert resolve_delta_t(None) == 1.0
    assert resolve_delta_t(None, default_delta_t=0.25) == 0.25
    assert resolve_delta_t(0.5, default_delta_t=0.1) == 0.5
    tensor_dt = torch.tensor(0.3)
    assert resolve_delta_t(tensor_dt, default_delta_t=1.0) is tensor_dt


def test_propagate_latent_discrete_matches_operator() -> None:
    """Verify discrete propagation uses the operator forward call."""
    torch.manual_seed(0)
    koopman = KoopmanOperator(latent_dim=4)
    z = torch.randn(3, 4)
    expected = koopman(z)
    assert torch.allclose(propagate_latent(koopman, z), expected)


def test_propagate_latent_continuous_default_delta_t() -> None:
    """Verify continuous mode uses default_delta_t when delta_t is omitted."""
    torch.manual_seed(1)
    koopman = ContinuousKoopmanOperator(latent_dim=4)
    z = torch.randn(2, 4)
    expected = koopman.advance(z, 0.25)
    got = propagate_latent(koopman, z, default_delta_t=0.25)
    assert torch.allclose(got, expected)


def test_propagate_latent_continuous_explicit_delta_t() -> None:
    """Verify explicit delta_t overrides default_delta_t."""
    torch.manual_seed(2)
    koopman = ContinuousKoopmanOperator(latent_dim=4)
    z = torch.randn(2, 4)
    expected = koopman.advance(z, 0.5)
    got = propagate_latent(koopman, z, delta_t=0.5, default_delta_t=0.1)
    assert torch.allclose(got, expected)


class _CustomContinuousPropagator(torch.nn.Module):
    """Minimal continuous-style contract module (not ContinuousKoopmanOperator)."""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.control_dim = 0
        self.parameterization = "dense"
        # Stable generator so matrix_exp is well-conditioned.
        self._generator = torch.nn.Parameter(-torch.eye(latent_dim))

    @property
    def matrix(self) -> torch.Tensor:
        return self._generator

    def advance(
        self,
        z: torch.Tensor,
        delta_t: float | torch.Tensor | None = None,
        *,
        control: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if delta_t is None:
            msg = "delta_t is required for custom continuous advance"
            raise ValueError(msg)
        del control
        delta = torch.as_tensor(delta_t, dtype=z.dtype, device=z.device)
        transition = torch.linalg.matrix_exp(self._generator * delta)
        return z @ transition.T

    def inverse_advance(
        self,
        z: torch.Tensor,
        delta_t: float | torch.Tensor | None = None,
        *,
        control: torch.Tensor | None = None,
        inverse_matrix: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del control, inverse_matrix
        if delta_t is None:
            msg = "delta_t is required for custom continuous inverse_advance"
            raise ValueError(msg)
        delta = torch.as_tensor(delta_t, dtype=z.dtype, device=z.device)
        transition = torch.linalg.matrix_exp(self._generator * delta)
        return z @ torch.linalg.inv(transition).T

    def bound_metric(self) -> torch.Tensor:
        return torch.linalg.eigvals(self._generator).real.max()


def test_propagate_latent_custom_continuous_uses_delta_t() -> None:
    """Custom continuous modules must receive resolved delta_t (no isinstance)."""
    torch.manual_seed(4)
    koopman = _CustomContinuousPropagator(3)
    z = torch.randn(2, 3)
    expected = koopman.advance(z, 0.4)
    got = propagate_latent(koopman, z, delta_t=0.4, default_delta_t=0.1)
    assert torch.allclose(got, expected)
    recovered = inverse_propagate_latent(koopman, got, delta_t=0.4)
    assert torch.allclose(recovered, z, atol=1e-5)


def test_inverse_propagate_latent_roundtrip_discrete() -> None:
    """Verify discrete inverse recovers the original latent approximately."""
    torch.manual_seed(3)
    koopman = KoopmanOperator(latent_dim=3, init_mode="identity")
    z = torch.randn(2, 3)
    advanced = propagate_latent(koopman, z)
    recovered = inverse_propagate_latent(koopman, advanced)
    assert torch.allclose(recovered, z, atol=1e-5)


def test_advance_and_decode_matches_manual_steps() -> None:
    """Verify one-step helper matches propagate + decode."""
    torch.manual_seed(4)
    koopman = KoopmanOperator(latent_dim=4, init_mode="identity")
    decoder = torch.nn.Linear(4, 3)
    z = torch.randn(2, 4)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    def decode(latent: torch.Tensor, _ei: torch.Tensor, _ew: torch.Tensor | None):
        return decoder(latent)

    z_next, prediction = advance_and_decode(
        koopman,
        decode,
        z,
        edge_index,
    )
    expected_z = propagate_latent(koopman, z)
    assert torch.allclose(z_next, expected_z)
    assert torch.allclose(prediction, decoder(expected_z))


def test_hold_last_topology_at_updates_and_retains() -> None:
    """Verify hold-last topology schedule updates then retains."""
    initial = _two_node_data()
    alt_edges = torch.tensor([[0, 0], [1, 1]], dtype=torch.long)
    future = [Data(x=torch.zeros(2, 3), edge_index=alt_edges)]
    topology_at = hold_last_topology_at(
        initial.edge_index,
        None,
        future,
    )
    first_ei, first_ew = topology_at(0)
    assert torch.equal(first_ei, alt_edges)
    assert first_ew is None
    second_ei, _ = topology_at(1)
    assert torch.equal(second_ei, alt_edges)


def test_snapshot_topology_at_uses_per_step_edges() -> None:
    """Verify per-step topology schedule indexes snapshots directly."""
    first = _two_node_data()
    second_edges = torch.tensor([[0, 0], [1, 1]], dtype=torch.long)
    second = Data(x=torch.zeros(2, 3), edge_index=second_edges)
    topology_at = snapshot_topology_at([first, second])
    assert torch.equal(topology_at(0)[0], first.edge_index)
    assert torch.equal(topology_at(1)[0], second_edges)


def test_autoregressive_latent_rollout_rejects_invalid_steps() -> None:
    """Verify shared rollout requires steps >= 1."""
    koopman = KoopmanOperator(latent_dim=2, init_mode="identity")
    z = torch.randn(2, 2)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    def decode(latent: torch.Tensor, _ei: torch.Tensor, _ew: torch.Tensor | None):
        return latent

    with pytest.raises(ValueError, match="steps"):
        autoregressive_latent_rollout(
            koopman,
            decode,
            z,
            steps=0,
            topology_at=hold_last_topology_at(edge_index),
        )


_DOCUMENTED_GRAPH_UTILS_EXPORTS = (
    "ControlAtFn",
    "DecoderFn",
    "DeltaTAtFn",
    "KoopmanPropagator",
    "TopologyAtFn",
    "advance_and_decode",
    "autoregressive_latent_rollout",
    "degree_support_mask",
    "dense_symmetric_normalized_adjacency",
    "dense_symmetric_normalized_laplacian",
    "hold_last_topology_at",
    "inverse_propagate_latent",
    "node_degrees",
    "pack_rollout_snapshots",
    "propagate_latent",
    "resolve_delta_t",
    "resolve_edge_index",
    "resolve_edge_weight",
    "resolve_graph_inputs",
    "snapshot_edge_weight",
    "snapshot_to_device",
    "snapshot_topology_at",
    "symmetric_normalized_adjacency_edge_weights",
    "symmetric_normalized_adjacency_matvec",
    "symmetric_normalized_laplacian_matvec",
)


def test_graph_utils_package_import_contract() -> None:
    """Verify package re-exports every documented graph_utils symbol."""
    import koopman_graph.graph_utils as graph_utils
    from koopman_graph.graph_utils import topology as topology_mod
    from koopman_graph.graph_utils.propagation import _topology_kwargs_for

    assert graph_utils.__all__ == list(_DOCUMENTED_GRAPH_UTILS_EXPORTS)
    for name in _DOCUMENTED_GRAPH_UTILS_EXPORTS:
        assert hasattr(graph_utils, name), name
        assert name in vars(graph_utils)

    # Topology peers stay importable; private helpers stay same-module only.
    assert callable(topology_mod.dense_symmetric_normalized_laplacian)
    assert callable(_topology_kwargs_for)
    assert not hasattr(graph_utils, "_topology_kwargs_for")
    assert not hasattr(topology_mod, "_topology_kwargs_for")
