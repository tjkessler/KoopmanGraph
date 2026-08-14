"""Tests for classical DMD peer baselines (TASK-1845–1847)."""

from __future__ import annotations

import pytest
import torch
from tests.helpers import REPO_ROOT
from torch_geometric.data import Data

import koopman_graph
import koopman_graph.baselines as baselines_pkg
from koopman_graph import GraphSnapshotSequence
from koopman_graph.baselines import (
    DMDBaseline,
    FBDMDBaseline,
    MRDMDBaseline,
    OptDMDBaseline,
    StreamingDMDBaseline,
    TLSDMDBaseline,
)
from koopman_graph.baselines.base import ClassicalBaseline

_REPO_SRC = REPO_ROOT / "src" / "koopman_graph" / "baselines"

_PEER_CLASSES = (
    FBDMDBaseline,
    TLSDMDBaseline,
    OptDMDBaseline,
    StreamingDMDBaseline,
    MRDMDBaseline,
)


def _linear_sequence(
    operator: torch.Tensor,
    initial_state: torch.Tensor,
) -> list[torch.Tensor]:
    """Generate flattened states following ``x_next = x @ K.T``."""
    states = [initial_state]
    for _ in range(5):
        states.append(states[-1] @ operator.T)
    return states


def _sequence_from_states(
    states: list[torch.Tensor],
    edge_index: torch.Tensor,
    *,
    num_nodes: int,
    in_channels: int,
    edge_weight: torch.Tensor | None = None,
) -> GraphSnapshotSequence:
    """Build a graph snapshot sequence from flattened states."""
    snapshots = []
    for state in states:
        fields: dict[str, torch.Tensor] = {
            "x": state.reshape(num_nodes, in_channels),
            "edge_index": edge_index,
        }
        if edge_weight is not None:
            fields["edge_weight"] = edge_weight
        snapshots.append(Data(**fields))
    return GraphSnapshotSequence(snapshots)


def _known_operator_sequence(
    synthetic_edge_index: torch.Tensor,
    *,
    edge_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, GraphSnapshotSequence, list[torch.Tensor]]:
    """Return a known linear operator and matching snapshot sequence."""
    operator = torch.tensor(
        [[0.8, 0.1], [-0.2, 1.05]],
        dtype=torch.float64,
    )
    states = _linear_sequence(
        operator,
        torch.tensor([1.0, -0.5], dtype=torch.float64),
    )
    sequence = _sequence_from_states(
        states,
        synthetic_edge_index,
        num_nodes=2,
        in_channels=1,
        edge_weight=edge_weight,
    )
    return operator, sequence, states


def test_classical_peers_exported_and_not_on_root_all() -> None:
    """Classical DMD peers export from baselines, not root ``__all__``."""
    for name in (
        "FBDMDBaseline",
        "TLSDMDBaseline",
        "OptDMDBaseline",
        "StreamingDMDBaseline",
        "MRDMDBaseline",
    ):
        assert name in baselines_pkg.__all__
        assert name not in set(koopman_graph.__all__)


@pytest.mark.parametrize("baseline_cls", _PEER_CLASSES)
def test_classical_peers_recover_linear_dynamics(
    baseline_cls: type[ClassicalBaseline],
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Full-rank peers recover a known flattened linear system."""
    operator, sequence, states = _known_operator_sequence(synthetic_edge_index)
    baseline = baseline_cls(time_step=0.25).fit(sequence)
    assert baseline.K is not None
    assert torch.allclose(baseline.K, operator, atol=1e-6)
    predictions = baseline.predict(sequence[0], steps=3)
    for prediction, expected in zip(predictions, states[1:4], strict=True):
        assert torch.allclose(prediction.x.reshape(-1), expected, atol=1e-6)


@pytest.mark.parametrize("baseline_cls", _PEER_CLASSES)
def test_classical_peers_preserve_prediction_topology(
    baseline_cls: type[ClassicalBaseline],
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Predictions copy initial topology and keep feature shape."""
    edge_weight = torch.arange(synthetic_edge_index.shape[1], dtype=torch.float64)
    _, sequence, _ = _known_operator_sequence(
        synthetic_edge_index,
        edge_weight=edge_weight,
    )
    prediction = baseline_cls().fit(sequence).predict(sequence[0], steps=1)[0]
    assert prediction.x.shape == (2, 1)
    assert torch.equal(prediction.edge_index, synthetic_edge_index)
    assert torch.equal(prediction.edge_weight, edge_weight)


@pytest.mark.parametrize("baseline_cls", _PEER_CLASSES)
def test_classical_peers_spectrum_matches_operator(
    baseline_cls: type[ClassicalBaseline],
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Spectrum eigenvalues match the fitted ``K``."""
    operator, sequence, _ = _known_operator_sequence(synthetic_edge_index)
    baseline = baseline_cls(time_step=0.5).fit(sequence)
    assert baseline.K is not None
    spectrum = baseline.spectrum()
    assert spectrum.time_step == 0.5
    expected = torch.linalg.eigvals(operator)
    got = spectrum.eigenvalues
    assert torch.allclose(
        torch.sort(got.real).values,
        torch.sort(expected.real).values,
        atol=1e-6,
    )


@pytest.mark.parametrize("baseline_cls", _PEER_CLASSES)
def test_classical_peers_reject_short_sequence(
    baseline_cls: type[ClassicalBaseline],
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Peers reject sequences shorter than their minimum length."""
    sequence = GraphSnapshotSequence(
        [Data(x=torch.ones(2, 1), edge_index=synthetic_edge_index)]
    )
    with pytest.raises(ValueError, match="at least"):
        baseline_cls().fit(sequence)


@pytest.mark.parametrize("baseline_cls", _PEER_CLASSES)
def test_classical_peers_reject_predict_before_fit(
    baseline_cls: type[ClassicalBaseline],
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Predict before fit raises."""
    initial = Data(x=torch.ones(2, 1), edge_index=synthetic_edge_index)
    with pytest.raises(RuntimeError):
        baseline_cls().predict(initial, steps=1)


@pytest.mark.parametrize("baseline_cls", _PEER_CLASSES)
def test_classical_peers_reject_invalid_steps(
    baseline_cls: type[ClassicalBaseline],
    synthetic_edge_index: torch.Tensor,
) -> None:
    """``steps`` must be positive."""
    _, sequence, _ = _known_operator_sequence(synthetic_edge_index)
    baseline = baseline_cls().fit(sequence)
    with pytest.raises(ValueError, match="steps"):
        baseline.predict(sequence[0], steps=0)


def test_streaming_matches_batch_dmd(
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Streaming Gram fit matches batch DMD on the same pairs."""
    operator, sequence, _ = _known_operator_sequence(synthetic_edge_index)
    batch = DMDBaseline().fit(sequence)
    streaming = StreamingDMDBaseline().fit(sequence)
    assert batch.K is not None and streaming.K is not None
    assert torch.allclose(streaming.K, batch.K, atol=1e-10)
    assert torch.allclose(streaming.K, operator, atol=1e-8)


def test_streaming_update_matches_full_fit(
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Prefix fit plus ``update`` matches a full-batch streaming fit."""
    _, sequence, _ = _known_operator_sequence(synthetic_edge_index)
    full = StreamingDMDBaseline().fit(sequence)
    prefix = GraphSnapshotSequence(list(sequence)[:4])
    incremental = StreamingDMDBaseline().fit(prefix)
    for snapshot in list(sequence)[4:]:
        incremental.update(snapshot)
    assert full.K is not None and incremental.K is not None
    assert torch.allclose(incremental.K, full.K, atol=1e-10)


def test_mrdmd_depth_two_tree_shape(
    synthetic_edge_index: torch.Tensor,
) -> None:
    """Depth-2 mrDMD exposes a root with two disjoint half-window children."""
    _, sequence, _ = _known_operator_sequence(synthetic_edge_index)
    baseline = MRDMDBaseline().fit(sequence)
    assert baseline.root is not None
    assert baseline.root.level == 0
    assert baseline.root.start == 0
    assert baseline.root.stop == sequence.num_timesteps
    assert len(baseline.root.children) == 2
    left_child, right_child = baseline.root.children
    assert left_child.level == 1
    assert right_child.level == 1
    assert left_child.stop == right_child.start
    assert left_child.start == 0
    assert right_child.stop == sequence.num_timesteps


def test_mrdmd_rejects_fewer_than_four_snapshots(
    synthetic_edge_index: torch.Tensor,
) -> None:
    """mrDMD requires four snapshots for a meaningful depth-2 tree."""
    sequence = GraphSnapshotSequence(
        [
            Data(x=torch.ones(2, 1) * float(t), edge_index=synthetic_edge_index)
            for t in range(3)
        ]
    )
    with pytest.raises(ValueError, match="at least four snapshots"):
        MRDMDBaseline().fit(sequence)


def test_classical_peers_honesty_docs_no_fabricated_doi() -> None:
    """Module docs name the methods and omit fabricated DOI citations."""
    for name, needle in (
        ("fbdmd.py", "forward"),
        ("tlsdmd.py", "total-least-squares"),
        ("optdmd.py", "optimized"),
        ("streaming_dmd.py", "streaming"),
        ("mrdmd.py", "multi-resolution"),
    ):
        source = (_REPO_SRC / name).read_text(encoding="utf-8").lower()
        assert needle in source
        assert "doi.org" not in source
        if name == "mrdmd.py":
            assert "depth-2" in source or "depth 2" in source
    assert isinstance(OptDMDBaseline(), ClassicalBaseline)
    assert isinstance(StreamingDMDBaseline(), ClassicalBaseline)
    assert isinstance(MRDMDBaseline(), ClassicalBaseline)
