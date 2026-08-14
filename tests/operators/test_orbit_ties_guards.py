"""Coverage and error-path tests for :mod:`koopman_graph.operators`."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch_geometric.data import Data, HeteroData

from koopman_graph.data import HeteroGraphSnapshotSequence
from koopman_graph.training.loop import bind_pending_orbit_ties

_TYPES = ("a", "b")

_EDGE_TYPES = (("a", "r", "b"),)


def _empty_multiplex_snapshot() -> HeteroData:
    snapshot = HeteroData()
    snapshot["node"].x = torch.zeros(2, 2)
    snapshot["node", "r", "node"].edge_index = torch.empty(2, 0, dtype=torch.long)
    return snapshot


class _BadHeteroSequence(HeteroGraphSnapshotSequence):
    def __init__(self) -> None:
        """Bypass container validation to exercise the defensive bind guard."""

    def __getitem__(self, _index: int) -> Data:
        return Data(x=torch.zeros(2, 2))


def test_orbit_binding_errors_and_empty_union() -> None:
    """Hetero orbit binding rejects bad snapshots and accepts empty bank unions."""
    calls: list[torch.Tensor] = []
    koopman = SimpleNamespace(
        auto_orbits=True,
        orbit_partition=None,
        is_typed=False,
        num_relations=1,
        ensure_orbit_binding=lambda _n, *, edge_index: calls.append(edge_index),
    )
    model = SimpleNamespace(koopman=koopman)
    with pytest.raises(ValueError, match="requires HeteroData"):
        bind_pending_orbit_ties(model, [_BadHeteroSequence()])  # type: ignore[list-item]

    typed = SimpleNamespace(
        auto_orbits=True,
        orbit_partition=None,
        is_typed=True,
        ensure_orbit_binding=lambda *_a, **_k: None,
    )
    with pytest.raises(ValueError, match="ensure_typed_orbit_binding"):
        bind_pending_orbit_ties(
            SimpleNamespace(koopman=typed),
            [HeteroGraphSnapshotSequence([_empty_multiplex_snapshot()])],
        )

    bind_pending_orbit_ties(
        model,
        [HeteroGraphSnapshotSequence([_empty_multiplex_snapshot()])],
    )
    assert calls[0].shape == (2, 0)


def test_bind_typed_and_nonempty_multiplex_auto_orbits() -> None:
    """Typed auto_orbits binding and nonempty multiplex union succeed."""
    typed_calls: list[object] = []
    typed = SimpleNamespace(
        auto_orbits=True,
        orbit_partition=None,
        is_typed=True,
        node_types=_TYPES,
        edge_types=_EDGE_TYPES,
        ensure_orbit_binding=lambda *_a, **_k: None,
        ensure_typed_orbit_binding=lambda banks, counts: typed_calls.append(
            (banks, counts)
        ),
    )
    snap = HeteroData()
    snap["a"].x = torch.randn(2, 2)
    snap["b"].x = torch.randn(2, 2)
    snap["a", "r", "b"].edge_index = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    bind_pending_orbit_ties(
        SimpleNamespace(koopman=typed),
        [HeteroGraphSnapshotSequence([snap, snap])],
    )
    assert typed_calls

    calls: list[torch.Tensor] = []
    multiplex = SimpleNamespace(
        auto_orbits=True,
        orbit_partition=None,
        is_typed=False,
        num_relations=1,
        ensure_orbit_binding=lambda _n, *, edge_index: calls.append(edge_index),
    )
    multi = HeteroData()
    multi["node"].x = torch.randn(3, 2)
    multi["node", "r0", "node"].edge_index = torch.tensor(
        [[0, 1], [1, 2]],
        dtype=torch.long,
    )
    bind_pending_orbit_ties(
        SimpleNamespace(koopman=multiplex),
        [HeteroGraphSnapshotSequence([multi, multi])],
    )
    assert calls and calls[0].numel() > 0


def test_orbit_ties_isotypic_config_and_binding_guards() -> None:
    """Isotypic flag conflicts and unbound / missing-topology bind errors."""
    from koopman_graph.operators import GraphKoopmanOperator

    with pytest.raises(ValueError, match="mutually exclusive with an explicit"):
        GraphKoopmanOperator(
            2,
            init_mode="identity",
            isotypic_symmetry=True,
            orbit_partition=((0, 1),),
        )
    with pytest.raises(ValueError, match="mutually exclusive with auto_orbits"):
        GraphKoopmanOperator(
            2,
            init_mode="identity",
            isotypic_symmetry=True,
            auto_orbits=True,
        )
    with pytest.raises(ValueError, match="orbit_method must be"):
        GraphKoopmanOperator(
            2,
            init_mode="identity",
            auto_orbits=True,
            orbit_method="wl",  # type: ignore[arg-type]
        )

    pending = GraphKoopmanOperator(2, init_mode="identity", auto_orbits=True)
    with pytest.raises(ValueError, match="edge_index or hyperedge_index"):
        pending.bind_auto_orbits(num_nodes=2)
    with pytest.raises(RuntimeError, match="orbit self bank is not allocated"):
        pending.orbit_self_matrices()

    bound = GraphKoopmanOperator(
        2,
        init_mode="identity",
        orbit_partition=((0,), (1,)),
    )
    with pytest.raises(ValueError, match="orbit partition was bound for 2 nodes"):
        bound.ensure_orbit_binding(3)


def test_orbit_ties_isotypic_bind_mismatch_and_symmetry_config() -> None:
    """Isotypic bind / topology-mismatch / symmetry_config paths."""
    from koopman_graph.operators import GraphKoopmanOperator

    edge = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    mock_report = MagicMock()
    mock_report.dimensions = (1, 1)
    mock_report.group_order = 2

    op = GraphKoopmanOperator(2, init_mode="identity", isotypic_symmetry=True)
    with (
        patch(
            "koopman_graph.operators.orbit_ties.compute_isotypic_decomposition",
            return_value=mock_report,
        ),
        patch(
            "koopman_graph.operators.orbit_ties.node_orbit_partition",
            return_value=((0,), (1,)),
        ),
    ):
        op.bind_auto_orbits(num_nodes=2, edge_index=edge)
    assert op.isotypic_decomposition is mock_report
    cfg = op.symmetry_config()
    assert cfg is not None
    assert cfg["symmetry"] == "isotypic"
    assert cfg["isotypic_dimensions"] == [1, 1]
    assert cfg["group_order"] == 2

    # Matching topology: both ensure_orbit_binding isotypic checks run.
    with patch(
        "koopman_graph.operators.orbit_ties.node_orbit_partition",
        return_value=((0,), (1,)),
    ):
        op.ensure_orbit_binding(2, edge_index=edge)

    with (
        patch(
            "koopman_graph.operators.orbit_ties.node_orbit_partition",
            return_value=((0, 1),),
        ),
        pytest.raises(ValueError, match="does not match the current topology"),
    ):
        op.ensure_orbit_binding(2, edge_index=edge)

    # Hyperedge 2-section path when only hyperedge_index is supplied.
    pending = GraphKoopmanOperator(2, init_mode="identity", auto_orbits=True)
    hyper = torch.tensor([[0, 1], [0, 0]], dtype=torch.long)
    with patch(
        "koopman_graph.operators.orbit_ties.node_orbit_partition",
        return_value=((0, 1),),
    ):
        pending.bind_auto_orbits(num_nodes=2, hyperedge_index=hyper)
    assert pending.orbit_partition == ((0, 1),)
    tied = pending.apply_tied_self(torch.randn(2, 2))
    assert tied.shape == (2, 2)
    blocks = pending.tied_self_blocks(2)
    assert blocks is not None and blocks.shape[0] == 2
    pending.reset_orbit_selves()
