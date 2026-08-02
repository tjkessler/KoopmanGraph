"""Teaching ``ForecasterProtocol`` contract (TASK-1923 / TASK-1928)."""

from __future__ import annotations

import pytest

import koopman_graph
from koopman_graph.baselines.gnn import (
    TEACHING_BASELINES,
    AGCRNBaseline,
    DCRNNBaseline,
    EmptyProtocolDeviationsError,
    ForecasterProtocol,
    GraphCastBaseline,
    GraphWaveNetBaseline,
    MTGNNBaseline,
    STGCNBaseline,
    STGODEBaseline,
)

# Minimal constructor kwargs keyed by class (matches TEACHING_BASELINES).
_TEACHING_KWARGS: dict[type, dict[str, object]] = {
    STGCNBaseline: {"history_len": 3, "num_st_blocks": 1, "kernel_size": 2},
    DCRNNBaseline: {"history_len": 2, "diffusion_steps": 1},
    GraphWaveNetBaseline: {
        "history_len": 4,
        "num_layers": 2,
        "adaptive_adj": False,
    },
    AGCRNBaseline: {"history_len": 2, "embed_dim": 4},
    MTGNNBaseline: {"history_len": 2, "num_layers": 2, "embed_dim": 4},
    STGODEBaseline: {"history_len": 2, "num_layers": 1},
    GraphCastBaseline: {"history_len": 2, "num_processor_layers": 1},
}

_EXPECTED_NAMES: dict[type, str] = {
    STGCNBaseline: "stgcn",
    DCRNNBaseline: "dcrnn",
    GraphWaveNetBaseline: "graph_wavenet",
    AGCRNBaseline: "agcrn",
    MTGNNBaseline: "mtgnn",
    STGODEBaseline: "stgode",
    GraphCastBaseline: "graphcast",
}


def _valid_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "name": "demo",
        "history_len": 3,
        "horizon": 12,
        "train_ratio": 0.7,
        "val_ratio": 0.1,
        "test_ratio": 0.2,
        "metric": "mae",
        "deviations": ("teaching-scale channels",),
    }
    base.update(overrides)
    return base


def test_forecaster_protocol_accepts_valid_fields() -> None:
    """Valid ratios, metric, and non-empty deviations construct cleanly."""
    protocol = ForecasterProtocol(**_valid_kwargs())  # type: ignore[arg-type]
    assert protocol.name == "demo"
    assert protocol.history_len == 3
    assert protocol.horizon == 12
    assert protocol.metric == "mae"
    assert protocol.deviations == ("teaching-scale channels",)


def test_empty_deviations_raises_named_error() -> None:
    """Empty deviations raise EmptyProtocolDeviationsError, not a bare ValueError."""
    with pytest.raises(EmptyProtocolDeviationsError, match="non-empty"):
        ForecasterProtocol(**_valid_kwargs(deviations=()))  # type: ignore[arg-type]
    with pytest.raises(EmptyProtocolDeviationsError):
        ForecasterProtocol(**_valid_kwargs(deviations=[]))  # type: ignore[arg-type]


def test_blank_deviation_entry_rejected() -> None:
    """Whitespace-only deviation strings are invalid."""
    with pytest.raises(ValueError, match="non-empty strings"):
        ForecasterProtocol(**_valid_kwargs(deviations=("  ",)))  # type: ignore[arg-type]


def test_ratios_must_sum_to_one() -> None:
    """Split ratios must sum to 1 within atol."""
    with pytest.raises(ValueError, match="sum"):
        ForecasterProtocol(
            **_valid_kwargs(train_ratio=0.5, val_ratio=0.1, test_ratio=0.2)  # type: ignore[arg-type]
        )


def test_ratio_bounds_and_metric_whitelist() -> None:
    """Reject non-positive ratios and unknown metrics."""
    with pytest.raises(ValueError, match="train_ratio"):
        ForecasterProtocol(**_valid_kwargs(train_ratio=0.0))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="metric"):
        ForecasterProtocol(**_valid_kwargs(metric="accuracy"))  # type: ignore[arg-type]


def test_history_and_horizon_must_be_positive() -> None:
    """history_len and horizon are positive integers."""
    with pytest.raises(ValueError, match="history_len"):
        ForecasterProtocol(**_valid_kwargs(history_len=0))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="horizon"):
        ForecasterProtocol(**_valid_kwargs(horizon=0))  # type: ignore[arg-type]


def test_teaching_baselines_registry_is_complete() -> None:
    """TEACHING_BASELINES lists every exported teaching forecaster class."""
    assert set(TEACHING_BASELINES) == set(_TEACHING_KWARGS)
    assert "TEACHING_BASELINES" in koopman_graph.baselines.gnn.__all__
    assert "TEACHING_BASELINES" not in koopman_graph.__all__


@pytest.mark.parametrize("cls", TEACHING_BASELINES)
def test_teaching_baseline_deviations_cover_required_topics(cls: type) -> None:
    """Every registered teaching baseline has itemized multi-topic deviations."""
    kwargs = _TEACHING_KWARGS[cls]
    model = cls(1, 4, 1, **kwargs)  # type: ignore[call-arg]
    protocol = model.protocol()
    assert isinstance(protocol, ForecasterProtocol)
    assert protocol.name == _EXPECTED_NAMES[cls]
    assert protocol.history_len == kwargs["history_len"]
    assert protocol.horizon == 12
    assert protocol.train_ratio == pytest.approx(0.7)
    assert protocol.val_ratio == pytest.approx(0.1)
    assert protocol.test_ratio == pytest.approx(0.2)
    assert protocol.metric == "mae"
    assert len(protocol.deviations) >= 5
    assert all(entry.strip() for entry in protocol.deviations)

    blob = " ".join(protocol.deviations).lower()
    # Required shared contract topics.
    assert "preprocess" in blob or "libcity" in blob
    assert "0.7" in blob and "split" in blob
    assert "adam" in blob and ("lr" in blob or "learning rate" in blob)
    assert "mse" in blob and "loss" in blob
    assert "predict pads" in blob or "lookback" in blob
    # At least one architecture / solver / problem-class specific line.
    assert any(
        key in blob
        for key in (
            "architecture",
            "solver",
            "problem class",
            "dependency",
            "data:",
        )
    )
    # Generic single-line disclaimer is insufficient.
    assert not (
        len(protocol.deviations) == 1 and "simplif" in protocol.deviations[0].lower()
    )


def test_protocol_symbols_exported_from_baselines_gnn_not_root() -> None:
    """ForecasterProtocol lives on baselines.gnn, not root __all__."""
    assert "ForecasterProtocol" in koopman_graph.baselines.gnn.__all__
    assert "EmptyProtocolDeviationsError" in koopman_graph.baselines.gnn.__all__
    assert "ForecasterProtocol" not in koopman_graph.__all__
    assert "EmptyProtocolDeviationsError" not in koopman_graph.__all__
